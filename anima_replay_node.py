import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


# ====================== Shared helpers ======================

def _safe_float_timestep(timestep) -> float:
    if torch.is_tensor(timestep):
        return float(timestep.flatten()[0].item())
    return float(timestep)


def _get_schedule_from_c(c: dict):
    try:
        t_opts = c.get("transformer_options", {})
        if not isinstance(t_opts, dict):
            return None

        sigmas = t_opts.get("sample_sigmas", None)
        if sigmas is None:
            sigmas = t_opts.get("sigmas", None)

        if torch.is_tensor(sigmas) and sigmas.numel() > 1:
            return sigmas.detach().flatten().float()
    except Exception:
        return None
    return None


def _progress_from_schedule(timestep, c: dict) -> Optional[float]:
    sigmas = _get_schedule_from_c(c)
    if sigmas is None or sigmas.numel() <= 1:
        return None

    try:
        t_val = _safe_float_timestep(timestep)
        target = torch.tensor([t_val], device=sigmas.device, dtype=sigmas.dtype)
        idx = int((sigmas - target).abs().argmin().item())
        denom = max(1, int(sigmas.numel()) - 1)
        return float(idx) / float(denom)
    except Exception:
        return None


def _looks_anima_or_cosmos(diffusion_model) -> bool:
    cls_name = diffusion_model.__class__.__name__.lower()
    mod_name = diffusion_model.__class__.__module__.lower()
    text = f"{mod_name} {cls_name}"
    return ("anima" in text) or ("cosmos" in text) or ("predict2" in text)


def _modulelist_score(name: str, module_list: nn.ModuleList) -> int:
    if len(module_list) < 4:
        return -10_000

    name_l = name.lower()
    score = 0

    for token, bonus in (
        ("blocks", 80),
        ("layers", 70),
        ("transformer", 40),
        ("dit", 35),
        ("double", 10),
        ("single", 10),
    ):
        if token in name_l:
            score += bonus

    score += min(len(module_list), 128)

    type_names = [m.__class__.__name__.lower() for m in module_list]
    unique_types = len(set(type_names))
    score += max(0, 30 - unique_types * 5)

    if any("block" in t for t in type_names):
        score += 30
    if any("layer" in t for t in type_names):
        score += 15

    if any(tok in name_l for tok in ("norm", "embed", "rope", "pos", "adapter", "proj")):
        score -= 40

    return score


def _find_best_block_container(diffusion_model) -> Tuple[str, nn.ModuleList, List[Tuple[str, int]]]:
    candidates = []

    for name, module in diffusion_model.named_modules():
        if isinstance(module, nn.ModuleList):
            score = _modulelist_score(name, module)
            if score > -1000:
                candidates.append((score, name, module))

    if not candidates:
        raise RuntimeError(
            "Could not find any repeated nn.ModuleList block container on this Anima/Cosmos model."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_name, best_module = candidates[0]
    summary = [(name, len(mod)) for _, name, mod in candidates[:10]]
    return best_name, best_module, summary


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _restore_all_previous_replay_patches(diffusion_model) -> int:
    """
    Safety cleanup for stale monkey-patches from older buggy versions.
    """
    restored = 0
    for module in diffusion_model.modules():
        if getattr(module, "_anima_replay_patched", False):
            orig = getattr(module, "_anima_replay_original_forward", None)
            if orig is not None:
                module.forward = orig
            if hasattr(module, "_anima_replay_patched"):
                delattr(module, "_anima_replay_patched")
            if hasattr(module, "_anima_replay_original_forward"):
                delattr(module, "_anima_replay_original_forward")
            restored += 1
    return restored


def _parse_block_indices(block_indices_text: str, max_block_index: int) -> List[int]:
    """
    Parses strings like:
      "3,4,5"
      "3-5,8"
      "3, 5, 8-10"
    into a sorted unique list of valid block indices.
    """
    if block_indices_text is None:
        raise RuntimeError("block_indices cannot be empty.")

    text = str(block_indices_text).strip()
    if not text:
        raise RuntimeError("block_indices cannot be empty.")

    result = set()

    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            pieces = [p.strip() for p in part.split("-") if p.strip()]
            if len(pieces) != 2:
                raise RuntimeError(
                    f"Invalid block range '{part}'. Use formats like '3-5' or '3,4,5,8'."
                )
            try:
                start = int(pieces[0])
                end = int(pieces[1])
            except ValueError:
                raise RuntimeError(
                    f"Invalid block range '{part}'. Use integers like '3-5'."
                )

            if end < start:
                start, end = end, start

            for idx in range(start, end + 1):
                if 0 <= idx <= max_block_index:
                    result.add(idx)
        else:
            try:
                idx = int(part)
            except ValueError:
                raise RuntimeError(
                    f"Invalid block index '{part}'. Use integers like '3' or ranges like '3-5'."
                )

            if 0 <= idx <= max_block_index:
                result.add(idx)

    parsed = sorted(result)
    if not parsed:
        raise RuntimeError(
            f"No valid block indices found in '{block_indices_text}'. "
            f"Valid range is 0 to {max_block_index}."
        )

    return parsed


# ====================== Spectrum forecaster ======================

class FastChebyshevForecaster:
    def __init__(self, m: int = 8, lam: float = 0.5):
        self.M = m
        self.K = max(m + 2, 8)
        self.lam = lam
        self.H_buf = []
        self.T_buf = []
        self.shape = None
        self.dtype = None
        self.t_max = 50.0

    def _taus(self, t: float) -> float:
        return (t / max(float(self.t_max), 1.0)) * 2.0 - 1.0

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        taus = taus.reshape(-1, 1)
        T = [torch.ones((taus.shape[0], 1), device=taus.device, dtype=torch.float32)]
        if self.M > 0:
            T.append(taus)
            for _ in range(2, self.M + 1):
                T.append(2 * taus * T[-1] - T[-2])
        return torch.cat(T[: self.M + 1], dim=1)

    def update(self, cnt: int, h: torch.Tensor):
        if self.shape is not None and h.shape != self.shape:
            self.reset_buffers()

        self.shape = h.shape
        self.dtype = h.dtype

        self.H_buf.append(h.reshape(-1))
        self.T_buf.append(self._taus(cnt))
        if len(self.H_buf) > self.K:
            self.H_buf.pop(0)
            self.T_buf.pop(0)

    def predict(self, cnt: int, w: float) -> torch.Tensor:
        device = self.H_buf[-1].device

        H = torch.stack(self.H_buf, dim=0).to(torch.float32)
        T = torch.tensor(self.T_buf, dtype=torch.float32, device=device)

        X = self._build_design(T)
        lamI = self.lam * torch.eye(self.M + 1, device=device, dtype=torch.float32)
        XtX = X.T @ X + lamI

        try:
            L = torch.linalg.cholesky(XtX)
        except RuntimeError:
            jitter = 1e-5 * XtX.diag().mean()
            L = torch.linalg.cholesky(XtX + jitter * torch.eye(self.M + 1, device=device, dtype=torch.float32))

        XtH = X.T @ H
        coef = torch.cholesky_solve(XtH, L)

        tau_star = torch.tensor([self._taus(cnt)], device=device, dtype=torch.float32)
        x_star = self._build_design(tau_star)
        pred_cheb = (x_star @ coef).squeeze(0)

        if len(self.H_buf) >= 2:
            h_i = self.H_buf[-1].to(torch.float32)
            h_im1 = self.H_buf[-2].to(torch.float32)
            h_taylor = h_i + 0.5 * (h_i - h_im1)
        else:
            h_taylor = self.H_buf[-1].to(torch.float32)

        res = (1.0 - w) * h_taylor + w * pred_cheb
        return torch.clamp(res, -10.0, 10.0).to(self.dtype).view(self.shape)

    def reset_buffers(self):
        self.H_buf.clear()
        self.T_buf.clear()
        self.shape = None
        self.dtype = None


# ====================== Combined node ======================

class AnimaLayerReplayPatcher:
    """
    Experimental Anima/Cosmos replay patcher with optional built-in Spectrum booster.

    Replay:
      - temporary, leak-free block replay
      - active only in the selected denoise window

    Spectrum:
      - optional
      - integrated into the same UNet wrapper so it can coexist with replay
      - exposes w, m, lam, warmup_steps
      - fixed internally:
          stop_caching_step = -1
    """

    SPECTRUM_STOP_CACHING_STEP = -1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_replay": ("BOOLEAN", {"default": True}),
                "block_indices": (
                    "STRING",
                    {
                        "default": "3,4,5",
                        "multiline": False,
                    },
                ),
                "denoise_start_pct": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_end_pct": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_spectrum": ("BOOLEAN", {"default": False}),
                "spectrum_w": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "spectrum_m": ("INT", {"default": 16, "min": 1, "max": 32, "step": 1}),
                "spectrum_lam": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 100.0, "step": 0.01}),
                "spectrum_warmup_steps": ("INT", {"default": 6, "min": 0, "max": 50, "step": 1}),
                "spectrum_window_size": ("INT", {"default": 2, "min": 1, "step": 1}),
                "spectrum_flex_window": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patches"

    def patch(
        self,
        model,
        block_indices: str,
        enable_replay: bool,
        denoise_start_pct: float,
        denoise_end_pct: float,
        enable_spectrum: bool,
        spectrum_w: float,
        spectrum_m: int,
        spectrum_lam: float,
        spectrum_warmup_steps: int,
        spectrum_window_size: int,
        spectrum_flex_window: float
    ):
        patched_model = model.clone()
        diffusion_model = patched_model.get_model_object("diffusion_model")

        if not _looks_anima_or_cosmos(diffusion_model):
            raise RuntimeError(
                "AnimaLayerReplayPatcher only supports Anima / Cosmos / Predict2-like diffusion models in ComfyUI."
            )

        _restore_all_previous_replay_patches(diffusion_model)

        _, block_list, _ = _find_best_block_container(diffusion_model)
        max_block_index = len(block_list) - 1
        selected_indices = _parse_block_indices(block_indices, max_block_index)

        denoise_start = _clamp01(denoise_start_pct)
        denoise_end = _clamp01(denoise_end_pct)
        if denoise_end < denoise_start:
            denoise_start, denoise_end = denoise_end, denoise_start

        spectrum_w = max(0.0, min(1.0, float(spectrum_w)))
        spectrum_m = max(1, int(spectrum_m))
        spectrum_lam = max(0.0, float(spectrum_lam))
        spectrum_warmup_steps = max(0, int(spectrum_warmup_steps))

        previous_wrapper = patched_model.model_options.get("model_function_wrapper", None)

        warned_non_tensor = {"shown": False}
        warned_replay_error = {"shown": False}

        spectrum_state = {
            "forecaster": None,
            "cnt": 0,
            "num_cached": 0,
            "curr_ws": float(spectrum_window_size),
            "last_t": float("inf"),
            "estimated_total_steps": 50,
        }

        def call_underlying(model_function, kwargs):
            if previous_wrapper is not None:
                return previous_wrapper(model_function, kwargs)
            return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])

        def make_replay_forward(orig_forward, block_index: int):
            def replay_forward(self, *args, **kwargs):
                out = orig_forward(*args, **kwargs)

                if not torch.is_tensor(out):
                    if not warned_non_tensor["shown"]:
                        print(
                            f"[AnimaReplay] Block {block_index} returned non-tensor output "
                            f"({type(out)}). Falling back to original output for this block."
                        )
                        warned_non_tensor["shown"] = True
                    return out

                try:
                    return orig_forward(out, *args[1:], **kwargs)
                except Exception as e:
                    if not warned_replay_error["shown"]:
                        print(
                            f"[AnimaReplay] Replay failed on block {block_index} "
                            f"with {type(e).__name__}: {e}. Falling back to original output."
                        )
                        warned_replay_error["shown"] = True
                    return out

            return replay_forward

        def run_with_optional_replay(model_function, kwargs, replay_active: bool):
            if not replay_active:
                return call_underlying(model_function, kwargs)

            patched_blocks = []
            original_forwards = []

            try:
                for idx in selected_indices:
                    block = block_list[idx]
                    original_forwards.append(block.forward)
                    patched_blocks.append(block)

                    replay_forward = make_replay_forward(block.forward, idx)
                    block.forward = replay_forward.__get__(block, block.__class__)

                return call_underlying(model_function, kwargs)
            finally:
                for block, orig_forward in zip(patched_blocks, original_forwards):
                    block.forward = orig_forward

        def reset_spectrum():
            spectrum_state["forecaster"] = None
            spectrum_state["cnt"] = 0
            spectrum_state["num_cached"] = 0
            spectrum_state["curr_ws"] = float(spectrum_window_size)
            spectrum_state["last_t"] = float("inf")
            spectrum_state["estimated_total_steps"] = 50

        def anima_unet_wrapper(model_function, kwargs):
            timestep = kwargs["timestep"]
            c = kwargs.get("c", {})

            replay_active = False
            if enable_replay:
                progress = _progress_from_schedule(timestep, c)
                replay_active = True if progress is None else (denoise_start <= progress <= denoise_end)

            if not enable_spectrum:
                return run_with_optional_replay(model_function, kwargs, replay_active)

            t_scalar = _safe_float_timestep(timestep)
            eps = 1e-7

            if t_scalar > spectrum_state["last_t"] + eps:
                reset_spectrum()

            spectrum_state["last_t"] = t_scalar

            schedule = _get_schedule_from_c(c)
            if schedule is not None and schedule.numel() > 1:
                spectrum_state["estimated_total_steps"] = int(schedule.numel())
            else:
                spectrum_state["estimated_total_steps"] = max(spectrum_state["estimated_total_steps"], spectrum_state["cnt"] + 1)

            if spectrum_state["forecaster"] is not None:
                spectrum_state["forecaster"].t_max = float(max(1, spectrum_state["estimated_total_steps"]))

            is_micro_final = False
            if self.SPECTRUM_STOP_CACHING_STEP == -1:
                auto_stop = int(spectrum_state["estimated_total_steps"] * 0.8)
                if spectrum_state["cnt"] >= auto_stop:
                    is_micro_final = True
            elif self.SPECTRUM_STOP_CACHING_STEP > 0 and spectrum_state["cnt"] >= self.SPECTRUM_STOP_CACHING_STEP:
                is_micro_final = True

            do_actual = True
            if spectrum_state["cnt"] >= spectrum_warmup_steps and not is_micro_final:
                current_ws = max(1, int(math.floor(spectrum_state["curr_ws"])))
                do_actual = ((spectrum_state["num_cached"] + 1) % current_ws) == 0

            if do_actual:
                out = run_with_optional_replay(model_function, kwargs, replay_active)

                if not torch.is_tensor(out):
                    spectrum_state["cnt"] += 1
                    spectrum_state["num_cached"] = 0
                    return out

                if spectrum_state["forecaster"] is None:
                    spectrum_state["forecaster"] = FastChebyshevForecaster(
                        m=spectrum_m,
                        lam=spectrum_lam,
                    )
                    spectrum_state["forecaster"].t_max = float(max(1, spectrum_state["estimated_total_steps"]))

                spectrum_state["forecaster"].update(spectrum_state["cnt"], out)

                if spectrum_state["cnt"] >= spectrum_warmup_steps:
                    spectrum_state["curr_ws"] += spectrum_flex_window

                spectrum_state["num_cached"] = 0
            else:
                if spectrum_state["forecaster"] is None or len(spectrum_state["forecaster"].H_buf) == 0:
                    out = run_with_optional_replay(model_function, kwargs, replay_active)
                    spectrum_state["num_cached"] = 0
                else:
                    out = spectrum_state["forecaster"].predict(
                        spectrum_state["cnt"],
                        w=spectrum_w,
                    ).to(kwargs["input"].dtype)
                    spectrum_state["num_cached"] += 1

            spectrum_state["cnt"] += 1
            return out

        patched_model.set_model_unet_function_wrapper(anima_unet_wrapper)
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "AnimaLayerReplayPatcher": AnimaLayerReplayPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLayerReplayPatcher": "Anima Layer Replay Patcher",
}
