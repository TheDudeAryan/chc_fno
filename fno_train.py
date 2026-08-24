"""FNO training for the Cahn-Hilliard equation."""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import time
from collections import deque
from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ModuleT = TypeVar("_ModuleT", bound=nn.Module)

DATASET = "test.npz"
RESUME = True

HARD_CONSTRAINTS = True             # master switch for the exact physics constraints
                                    # (mass projection + DC-free spectral weights);
                                    # False falls back to the soft MASS_WEIGHT penalty
ABLATION = "" if HARD_CONSTRAINTS else "_nohard"
MODEL_OUT = f"fno_ch{ABLATION}.pt"
CHECKPOINT = f"fno_ch{ABLATION}_last.pt"

OFF_CRITICAL = False                # test.npz ALREADY spans -0.4/-0.2/0/+0.2/+0.4,
OFF_CRITICAL_OFFS = (0.2, 0.4)      # 4 trajectories each.  Generating more only
OFF_CRITICAL_TRAJ = 3               # skews that balance and dilutes the critical
OFF_CRITICAL_SEED0 = 700000         # composition, which is the graded one.

DT = 0.01
N_AHEAD = 100
GAP = 10
STRIDE = N_AHEAD // GAP
DT_STEP = N_AHEAD * DT

CHANNELS = 64
MODES = 24
LAYERS = 5
LOCAL_KERNEL = 3
MODE_SCALING = "physical"

ZERO_DC = True
HARD_MASS_CONSERVATION = True
RESIDUAL_PREDICTION = True
MASS_WEIGHT = 0.1

if not HARD_CONSTRAINTS:
    ZERO_DC = False
    HARD_MASS_CONSERVATION = False

EPOCHS = 200
BATCH_SIZE = 32
STEPS_PER_EPOCH = 1500
LR = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
LR_FLOOR = 1e-2

MAX_UNROLL = 8
BPTT = 2
WARM_FRAC = 0.15
RAMP_FRAC = 0.55

LR_STAGE_DECAY = 0.7
NOISE_STD = 0.01
NOISE_PROB = 0.5

H1_WEIGHT = 0.5

CONTRACTIVITY_WEIGHT = 1.0
AMP_TARGET = 0.25
AMP_EPS = 1e-3
SMOOTH_WEIGHT = 1e-3

VAL_SEEDS = (31337, 24601, 8675309, 4815162, 27182818)
VAL_LATTICE = 128
VAL_T_END = 2000.0
VAL_T_START = 100.0
VAL_EVERY = 5
VAL_BATCHES = 32

PATIENCE = 8
PATIENCE_AFTER_RAMP = True          # early stopping only once the curriculum has
                                    # reached MAX_UNROLL -- the objective is still
                                    # getting harder while it ramps
MIN_DELTA = 2e-3
R2_THRESHOLD = 0.80
GT_CACHE = "gt_cache"

PRECISION = "fp32"                  # "fp32" | "fp64" | "tf32"
AMP = "off"                         # "off" | "bf16"   (fp64 requires "off")
COMPILE = False
GPU_DATA = "auto"
MAX_HOURS = 20.0
SEED = 0

PSI_MAX = 5.0
LOSS_SPIKE_FACTOR = 8.0             # skip a batch whose loss exceeds this x the running median
LOSS_SPIKE_WARMUP = 100             # batches to observe before the spike guard arms
SPIKE_ABORT_FRAC = 0.10             # skipping more of an epoch than this means the
                                    # model is broken, not the batch -- roll back
MAX_DELTA_REL = 0.5                 # |psi(t+1) - psi(t)| / |psi| is ~0.007 for the
                                    # true dt=1 map; past this it is not that map

DIVERGE_FACTOR = 3.0
MAX_RECOVERIES = 8
LR_RECOVERY_DECAY = 0.8

_TORCH_DTYPE = {"fp64": torch.float64, "fp32": torch.float32,
                "tf32": torch.float32}
_NUMPY_DTYPE = {"fp64": np.float64, "fp32": np.float32, "tf32": np.float32}


def torch_dtype() -> torch.dtype:
    return _TORCH_DTYPE[PRECISION]


def numpy_dtype():
    return _NUMPY_DTYPE[PRECISION]


def to_dtype(module: _ModuleT, dtype: torch.dtype) -> _ModuleT:
    """Cast a module's parameters and buffers, complex tensors included."""
    complex_dtype = (torch.complex128 if dtype == torch.float64
                     else torch.complex64)
    for mod in module.modules():
        for name, param in list(mod.named_parameters(recurse=False)):
            target = complex_dtype if param.is_complex() else dtype
            if param.dtype != target:
                setattr(mod, name, nn.Parameter(param.data.to(target),
                                                requires_grad=param.requires_grad))
        for name, buf in list(mod.named_buffers(recurse=False)):
            if buf is None:
                continue
            if buf.is_complex():
                mod._buffers[name] = buf.to(complex_dtype)
            elif buf.is_floating_point():
                mod._buffers[name] = buf.to(dtype)
    return module


def device_and_precision():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if PRECISION not in _TORCH_DTYPE:
        raise ValueError(f"PRECISION must be one of {sorted(_TORCH_DTYPE)}")
    allow_tf32 = PRECISION == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    if allow_tf32 and dev.type == "cuda":
        print("[setup] NOTE: TF32 measured amplification 0.402 vs 0.209 and "
              "R2@400 0.048 vs 0.917 against fp32 when used for *training*; "
              "it is harmless at inference", flush=True)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if PRECISION == "fp64" and AMP != "off":
        raise ValueError("AMP must be 'off' when PRECISION is 'fp64' -- "
                         "autocasting to bf16 would undo the double precision")

    def amp_ctx():
        if AMP == "bf16" and dev.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()
    return dev, amp_ctx

DX = 1.0


def initial_condition(l, seed, off=0.0):
    """ch.py's set_ic, same RNG stream."""
    np.random.seed(seed)
    ic = np.random.uniform(-0.1, 0.1, size=(l, l))
    return ic - ic.mean() + off


def _lap_np(psi):
    return (np.roll(psi, 1, 0) + np.roll(psi, -1, 0)
            + np.roll(psi, 1, 1) + np.roll(psi, -1, 1) - 4.0 * psi) / DX ** 2


def free_energy(psi):
    """Ginzburg-Landau density.  Cahn-Hilliard is a gradient flow, so this must"""
    gx = np.roll(psi, -1, 0) - psi
    gy = np.roll(psi, -1, 1) - psi
    return float(np.mean(0.25 * (psi ** 2 - 1.0) ** 2
                         + 0.5 * (gx ** 2 + gy ** 2)))


def _integrate(psi, n_total, every, n_frames, device="cpu", verbose=False,
               t_end=None):
    """Explicit-Euler Cahn-Hilliard, keeping every ``every``-th step."""
    l = psi.shape[-1]
    if device != "cpu":
        p = torch.as_tensor(psi, device=device, dtype=torch.float64)

        def lap(q):
            return (torch.roll(q, 1, -2) + torch.roll(q, -1, -2)
                    + torch.roll(q, 1, -1) + torch.roll(q, -1, -1)
                    - 4.0 * q) / DX ** 2
        buf = torch.empty((n_frames, l, l), dtype=torch.float64, device=device)
        buf[0] = p
        j = 1
        for n in range(1, n_total + 1):
            p = p + lap(p ** 3 - p - lap(p)) * DT
            if n % every == 0 and j < n_frames:
                buf[j] = p
                j += 1
        return buf.cpu().numpy()

    frames = np.empty((n_frames, l, l), dtype=np.float64)
    frames[0] = psi
    j = 1
    for n in range(1, n_total + 1):
        psi = psi + _lap_np(psi ** 3 - psi - _lap_np(psi)) * DT
        if n % every == 0 and j < n_frames:
            frames[j] = psi
            j += 1
            if verbose and j % 500 == 0:
                print(f"    t = {n * DT:8.1f} / {t_end:.1f}", flush=True)
    return frames


def ch_ground_truth(l, seed, off=0.0, t_end=2000.0, dt_step=DT_STEP,
                    cache_dir=GT_CACHE, device="cpu", verbose=True):
    """Reference trajectory sampled every ``dt_step``, cached on disk."""
    os.makedirs(cache_dir, exist_ok=True)

    path = os.path.join(cache_dir, f"gt_l{l}_seed{seed}_off{off:+.2f}"
                                   f"_t{t_end:g}_f64.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return z["times"], z["frames"]

    every = int(round(dt_step / DT))
    n_total = int(round(t_end / DT))
    n_frames = n_total // every + 1
    if verbose:
        print(f"[gt] solving l={l} seed={seed} off={off:+.2f} to t={t_end:g} "
              f"({n_total:,} steps)", flush=True)

    times = np.arange(n_frames) * dt_step
    frames = _integrate(initial_condition(l, seed, off), n_total, every,
                        n_frames, device, verbose, t_end)
    np.savez(path, times=times, frames=frames)
    return times, frames


def load_trajectories(path=DATASET, verbose=True):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run the advisor's ch.py first; it writes "
            f"test.npz into the working directory.")

    if verbose:
        print(f"[data] reading {path} "
              f"({os.path.getsize(path) / 1e9:.2f} GB on disk)", flush=True)
    with np.load(path) as npz:
        keys = set(npz.files)
        missing = {"train_X", "val_X", "val_y"} - keys
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}; expected the "
                           f"arrays ch.py writes, found {sorted(keys)}")
        val = npz["val_X"].astype(numpy_dtype())
        val_y = npz["val_y"]
        stride = _detect_stride(val, val_y)
        del val_y
        train = npz["train_X"].astype(numpy_dtype())

    if stride != STRIDE:
        raise ValueError(
            f"the dataset's label offset is {stride} frames, not STRIDE="
            f"{STRIDE}. ch.py's n_ahead/gap must have changed; set "
            f"N_AHEAD and GAP in the CONFIG block to match.")

    if verbose:
        print(f"[data] train {train.shape}  val {val.shape}  "
              f"{np.dtype(numpy_dtype()).name}, "
              f"{(train.nbytes + val.nbytes) / 1e9:.2f} GB", flush=True)
        print(f"[data] label identity verified: y[s] == X[s+{stride}], so one "
              f"model step = {stride} frames = {DT_STEP:g} of real time",
              flush=True)
        for name, arr in (("train", train), ("val", val)):
            comp = np.round(arr[:, 0].mean(axis=(1, 2)).astype(np.float64), 2)
            counts = {float(c): int((comp == c).sum())
                      for c in np.unique(comp)}
            print(f"[data] {name} compositions (mean psi -> trajectories): "
                  f"{counts}", flush=True)
    return train, val


def _detect_stride(x, y, max_stride=64):
    """Find s with y[:, i] == x[:, i + s]; that is n_ahead // gap."""
    probe = min(3, x.shape[1] // 2)
    ref = np.asarray(y[0, :probe], dtype=np.float64)
    for s in range(1, max_stride + 1):
        if s + probe > x.shape[1]:
            break
        cand = np.asarray(x[0, s:s + probe], dtype=np.float64)
        if np.abs(cand - ref).max() < 1e-6:
            return s
    raise ValueError(
        "could not match the label array to a shift of the feature array. "
        "The (X, y) pairs do not come from a single trajectory sampled on one "
        "grid, so multi-step targets cannot be reconstructed.")


def off_critical_trajectories(lattice, n_frames, device="cpu", verbose=True):
    """Training trajectories at off-critical composition, cached on disk.

    The advisor's dataset is a critical (50:50) quench only.  Cahn-Hilliard is
    odd under psi -> -psi, and WindowSampler already applies a random sign
    flip, so solving +off covers -off exactly; only positive offsets are run.
    Frames are sampled on ch.py's grid (every GAP solver steps) so the result
    concatenates directly onto train_X.
    """
    os.makedirs(GT_CACHE, exist_ok=True)
    tag = "_".join(f"{o:+.2f}" for o in OFF_CRITICAL_OFFS)
    path = os.path.join(GT_CACHE, f"offcrit_l{lattice}_n{n_frames}"
                                  f"_x{OFF_CRITICAL_TRAJ}_{tag}.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            frames = z["frames"]
        if verbose:
            print(f"[off] reusing {path} {frames.shape}", flush=True)
        return frames.astype(numpy_dtype())

    every = GAP
    n_total = (n_frames - 1) * every
    n_traj = len(OFF_CRITICAL_OFFS) * OFF_CRITICAL_TRAJ
    if verbose:
        print(f"[off] solving {n_traj} off-critical trajectories at l={lattice}"
              f", off={list(OFF_CRITICAL_OFFS)}, {n_total:,} steps each",
              flush=True)

    out = np.empty((n_traj, n_frames, lattice, lattice), dtype=np.float32)
    i = 0
    for off in OFF_CRITICAL_OFFS:
        for _ in range(OFF_CRITICAL_TRAJ):
            seed = OFF_CRITICAL_SEED0 + i
            psi = _integrate(initial_condition(lattice, seed, off), n_total,
                             every, n_frames, device)
            out[i] = psi.astype(np.float32)
            if verbose:
                print(f"[off]   {i + 1}/{n_traj} seed={seed} off={off:+.2f} "
                      f"mean={out[i, -1].mean():+.4f} "
                      f"range=[{out[i, -1].min():+.2f}, {out[i, -1].max():+.2f}]",
                      flush=True)
            i += 1
    np.savez(path, frames=out)
    if verbose:
        print(f"[off] cached {path} "
              f"({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)
    return out.astype(numpy_dtype())


class WindowSampler:
    def __init__(self, frames, device, data_device, n_steps=1, augment=True,
                 seed=0):
        self.data = torch.from_numpy(frames).to(data_device)
        self.device = device
        self.augment = augment
        self.gen = torch.Generator(device=data_device).manual_seed(seed)
        self.n_traj, self.n_frames = self.data.shape[0], self.data.shape[1]
        self.set_n_steps(n_steps)

    def set_n_steps(self, n_steps):
        self.n_steps = int(n_steps)
        span = STRIDE * self.n_steps
        self.last_start = self.n_frames - span - 1
        if self.last_start < 0:
            raise ValueError(
                f"MAX_UNROLL={self.n_steps} needs {span + 1} frames but the "
                f"trajectories only have {self.n_frames}")
        self.offsets = torch.arange(0, span + 1, STRIDE,
                                    device=self.data.device)

    def _gather(self, traj, start):
        return self.data[traj[:, None], start[:, None] + self.offsets[None, :]]

    def sample(self, batch_size):
        dev = self.data.device
        traj = torch.randint(0, self.n_traj, (batch_size,), device=dev,
                             generator=self.gen)
        start = torch.randint(0, self.last_start + 1, (batch_size,),
                              device=dev, generator=self.gen)
        win = self._gather(traj, start)
        if self.augment:
            win = self._augment(win)
        return win.to(self.device, non_blocking=True)

    def fixed_batches(self, batch_size, n_batches):
        """Deterministic, un-augmented -- for the cheap one-step val loss."""
        rng = np.random.default_rng(0)
        span = self.last_start + 1
        picks = rng.choice(self.n_traj * span,
                           size=min(self.n_traj * span, batch_size * n_batches),
                           replace=False)
        for i in range(0, len(picks), batch_size):
            chunk = picks[i:i + batch_size]
            traj = torch.as_tensor(chunk // span, device=self.data.device)
            start = torch.as_tensor(chunk % span, device=self.data.device)
            yield self._gather(traj, start).to(self.device, non_blocking=True)

    def _augment(self, win):
        b = win.shape[0]
        dev = win.device
        which = torch.randint(0, 8, (b,), device=dev, generator=self.gen)
        out = torch.empty_like(win)
        for elem in range(8):
            mask = which == elem
            if not bool(mask.any()):
                continue
            y = win[mask]
            k, flip = divmod(elem, 2)
            if flip:
                y = torch.flip(y, dims=(-1,))
            if k:
                y = torch.rot90(y, k, dims=(-2, -1))
            out[mask] = y
        sign = torch.randint(0, 2, (b, 1, 1, 1), device=dev,
                             generator=self.gen) * 2 - 1
        return out * sign.to(out.dtype)


class SpectralConv2d(nn.Module):
    """Learned Fourier multiplier on a grid of *physical* wavenumbers."""

    dc_mask: torch.Tensor

    def __init__(self, in_ch, out_ch, modes, n_ref, mode_scaling="physical",
                 zero_dc=False):
        super().__init__()
        self.in_channels, self.out_channels = in_ch, out_ch
        self.modes, self.n_ref = modes, n_ref
        self.mode_scaling, self.zero_dc = mode_scaling, zero_dc
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, 2 * modes + 1, modes + 1, 2))

        mask = torch.ones(1, 1, 2 * modes + 1, modes + 1, 1)
        if zero_dc:
            mask[0, 0, modes, 0, 0] = 0.0
        self.register_buffer("dc_mask", mask, persistent=False)
        self._cache = {}

    @property
    def effective_weight(self):
        return (self.weight * self.dc_mask) if self.zero_dc else self.weight

    def modes_for(self, n):
        cap = max(1, (n - 1) // 2)
        if self.mode_scaling == "index":
            return min(self.modes, cap)
        return max(1, min(int(round(self.modes * n / self.n_ref)), cap))

    def _resampled_weight(self, n):
        m_n = self.modes_for(n)
        if m_n == self.modes:
            return self.effective_weight
        key = (n, self.weight.device, self.weight.dtype)
        if not self.training and key in self._cache:
            return self._cache[key]
        w = self.effective_weight.permute(0, 1, 4, 2, 3).reshape(
            1, self.in_channels * self.out_channels * 2,
            2 * self.modes + 1, self.modes + 1)

        w = F.interpolate(w, size=(2 * m_n + 1, m_n + 1), mode="bilinear",
                          align_corners=True)
        w = w.reshape(self.in_channels, self.out_channels, 2,
                      2 * m_n + 1, m_n + 1).permute(0, 1, 3, 4, 2)
        if not self.training:
            self._cache[key] = w
        return w

    def train(self, mode=True):
        self._cache.clear()
        return super().train(mode)

    def forward(self, x):
        b, _, n, n2 = x.shape
        if n != n2:
            raise ValueError("expected a square lattice")
        m_n = self.modes_for(n)

        dev = x.device.type
        ctx = (torch.autocast(device_type=dev, enabled=False)
               if dev in ("cuda", "cpu") else contextlib.nullcontext())
        with ctx:
            real_dtype = self.weight.dtype
            x = x.to(real_dtype)
            w = torch.view_as_complex(
                self._resampled_weight(n).to(real_dtype).contiguous())
            x_ft = torch.fft.rfft2(x, norm="backward")
            out_ft = torch.zeros(b, self.out_channels, n, n // 2 + 1,
                                 dtype=x_ft.dtype, device=x.device)
            out_ft[:, :, :m_n + 1, :m_n + 1] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :m_n + 1, :m_n + 1],
                w[:, :, m_n:, :])
            out_ft[:, :, -m_n:, :m_n + 1] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, -m_n:, :m_n + 1],
                w[:, :, :m_n, :])
            return torch.fft.irfft2(out_ft, s=(n, n), norm="backward")

    def smoothness_penalty(self):
        """Resampling to a denser k-grid is only meaningful if R varies smoothly"""
        w = self.effective_weight
        return ((w[:, :, 1:] - w[:, :, :-1]).pow(2).mean()
                + (w[:, :, :, 1:] - w[:, :, :, :-1]).pow(2).mean())


class FNOBlock(nn.Module):
    """Spectral (global) path + local stencil path, wired residually."""

    def __init__(self, ch, modes, n_ref, mode_scaling, local_kernel, zero_dc):
        super().__init__()
        self.spectral = SpectralConv2d(ch, ch, modes, n_ref, mode_scaling,
                                       zero_dc)
        self.local = nn.Conv2d(ch, ch, local_kernel, padding=local_kernel // 2,
                               padding_mode="circular")
        self.mix = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return x + self.mix(F.gelu(self.spectral(x) + self.local(x)))


class PhysicalKBackbone(nn.Module):
    """Lift -> resolution-consistent FNO blocks -> project."""

    def __init__(self, in_ch, channels, modes, n_layers, n_ref, mode_scaling,
                 local_kernel, zero_dc):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Conv2d(in_ch, channels, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 1))
        self.blocks = nn.ModuleList([
            FNOBlock(channels, modes, n_ref, mode_scaling, local_kernel,
                     zero_dc) for _ in range(n_layers)])

        hidden_conv = nn.Conv2d(channels, 2 * channels, 1)
        out_conv = nn.Conv2d(2 * channels, 1, 1)
        with torch.no_grad():
            out_conv.weight.zero_()
            if out_conv.bias is not None:
                out_conv.bias.zero_()
        self.project = nn.Sequential(hidden_conv, nn.GELU(), out_conv)

    def forward(self, x):
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        return self.project(h)


class CahnHilliardFNO(nn.Module):
    """psi(t) -> psi(t + DT_STEP)."""

    def __init__(self, channels=CHANNELS, modes=MODES, n_layers=LAYERS,
                 n_ref=64, mode_scaling=MODE_SCALING,
                 local_kernel=LOCAL_KERNEL, zero_dc=ZERO_DC,
                 use_mean_channel=True,
                 hard_mass_conservation=HARD_MASS_CONSERVATION,
                 residual_prediction=RESIDUAL_PREDICTION):
        super().__init__()
        self.config = dict(channels=channels, modes=modes, n_layers=n_layers,
                           n_ref=n_ref, mode_scaling=mode_scaling,
                           local_kernel=local_kernel, zero_dc=zero_dc,
                           use_mean_channel=use_mean_channel,
                           hard_mass_conservation=hard_mass_conservation,
                           residual_prediction=residual_prediction)
        self.use_mean_channel = use_mean_channel
        self.hard_mass_conservation = hard_mass_conservation
        self.residual_prediction = residual_prediction
        in_ch = 2 if use_mean_channel else 1

        self.backbone = PhysicalKBackbone(in_ch, channels, modes, n_layers,
                                          n_ref, mode_scaling, local_kernel,
                                          zero_dc)

    def forward(self, psi):
        if psi.dim() == 3:
            psi = psi.unsqueeze(1)
        x = psi
        if self.use_mean_channel:
            x = torch.cat([psi, psi.mean(dim=(-2, -1), keepdim=True)
                           .expand_as(psi)], dim=1)
        out = self.backbone(x)

        if self.residual_prediction:
            if self.hard_mass_conservation:

                out = out - out.mean(dim=(-2, -1), keepdim=True)
            return psi + out
        if self.hard_mass_conservation:

            return out - out.mean(dim=(-2, -1), keepdim=True) \
                + psi.mean(dim=(-2, -1), keepdim=True)
        return out

    def smoothness_penalty(self):
        """Zero unless the physical_k backbone is in use -- the penalty"""
        terms = [m.smoothness_penalty() for m in self.modules()
                 if isinstance(m, SpectralConv2d)]
        if not terms:
            ref = next(self.parameters())
            return torch.zeros((), device=ref.device, dtype=torch.float64
                               if ref.dtype == torch.float64 else torch.float32)
        return torch.stack(terms).mean()

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_model(n_ref: int = 64, device: torch.device | str = "cpu",
                dtype: torch.dtype | None = None) -> CahnHilliardFNO:
    """Construct from the CONFIG globals and cast to the working precision."""
    model = CahnHilliardFNO(
        channels=CHANNELS, modes=MODES, n_layers=LAYERS, n_ref=n_ref,
        mode_scaling=MODE_SCALING, local_kernel=LOCAL_KERNEL, zero_dc=ZERO_DC,
        hard_mass_conservation=HARD_MASS_CONSERVATION,
        residual_prediction=RESIDUAL_PREDICTION)
    return to_dtype(model, dtype or torch_dtype()).to(device)


def _tensors_only(state_dict):
    """Keep only tensor entries, so a stray non-tensor cannot break a load."""
    return {k: v for k, v in state_dict.items() if torch.is_tensor(v)}


def save_model(model, path, **extra):
    torch.save({"config": model.config,
                "state_dict": _tensors_only(model.state_dict()),
                "precision": PRECISION, **extra}, path)


def load_checkpoint(path, device: torch.device | str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no 'config' entry -- not a checkpoint "
                         f"from fno_train.py")
    model = CahnHilliardFNO(**ckpt["config"])
    state = _tensors_only(ckpt["state_dict"])

    saved = next((v.dtype for v in state.values() if v.is_floating_point()),
                 torch_dtype())
    model = to_dtype(model, saved).to(device)
    model.load_state_dict(state)

    want = torch_dtype()
    if want != saved:
        model = to_dtype(model, want)
        print(f"[model] cast {saved} -> {want} (PRECISION={PRECISION}); the "
              f"checkpoint was trained in {saved}", flush=True)
    model.eval()
    return model, ckpt


def _rel_l2(pred, target, eps=1e-8):
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    return (torch.linalg.vector_norm(p - t, dim=1)
            / (torch.linalg.vector_norm(t, dim=1) + eps))


def loss_fn(pred, target):
    """Relative L2, plus the relative H1 seminorm, plus an optional soft mass"""
    loss = _rel_l2(pred, target).mean()
    if H1_WEIGHT > 0:
        gp = torch.cat([torch.roll(pred, -1, -2) - pred,
                        torch.roll(pred, -1, -1) - pred], dim=1)
        gt = torch.cat([torch.roll(target, -1, -2) - target,
                        torch.roll(target, -1, -1) - target], dim=1)
        loss = loss + H1_WEIGHT * _rel_l2(gp, gt).mean()
    if not HARD_MASS_CONSERVATION and MASS_WEIGHT > 0:
        drift = (pred.mean(dim=(-2, -1)) - target.mean(dim=(-2, -1))) ** 2
        loss = loss + MASS_WEIGHT * drift.mean()
    return loss


def amplification_penalty(model, x_in, f_x):
    """Charge the model for damping perturbations less than the true map does."""
    e = torch.randn_like(x_in)
    e = e - e.mean(dim=(-2, -1), keepdim=True)
    norm = e.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
    e = e * (AMP_EPS / norm)
    amp = ((model(x_in + e) - f_x).flatten(1).norm(dim=1)
           / e.flatten(1).norm(dim=1).clamp_min(1e-12))
    return torch.relu(amp - AMP_TARGET).pow(2).mean()


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 0.0 else -np.inf
    return 1.0 - ss_res / ss_tot


@torch.no_grad()
def rollout_frames(model, gt_times, gt_frames, t_start, t_end, device,
                   amp_ctx=None):
    """Yield (t, true, pred) for each autoregressive step from ``t_start``."""
    dt_gt = float(gt_times[1] - gt_times[0])
    if abs(dt_gt - DT_STEP) > 1e-9:
        raise ValueError(f"ground truth sampled every {dt_gt}, model step is "
                         f"{DT_STEP}")
    i0 = int(np.argmin(np.abs(gt_times - t_start)))
    i1 = int(np.argmin(np.abs(gt_times - t_end)))
    model.eval()
    ctx = amp_ctx if amp_ctx is not None else contextlib.nullcontext

    dtype = next(model.parameters()).dtype
    psi = torch.as_tensor(gt_frames[i0], dtype=dtype, device=device)[None, None]
    for k in range(i1 - i0):
        with ctx():
            psi = model(psi)
        j = i0 + k + 1
        yield (float(gt_times[j]),
               gt_frames[j].astype(np.float64),
               psi[0, 0].detach().to(torch.float64).cpu().numpy())


def rollout_score(model, gt_times, gt_frames, device, amp_ctx=None,
                  t_start=VAL_T_START, t_end=VAL_T_END,
                  threshold=R2_THRESHOLD):
    """(score, survival) over the fixed window.  score = mean of max(R2, 0)."""
    scores, survival = [], t_end
    for t, true, pred in rollout_frames(model, gt_times, gt_frames, t_start,
                                        t_end, device, amp_ctx):
        r = r2_score(true, pred)
        if r < threshold and survival == t_end:
            survival = t

        scores.append(r if np.isfinite(r) and r > 0.0 else 0.0)
    return (float(np.mean(scores)) if scores else 0.0), survival


def unroll_at(epoch, epochs, max_unroll, warm_frac=WARM_FRAC,
              ramp_frac=RAMP_FRAC):
    """One-step warmup, ramp to max_unroll by ``ramp_frac``, then hold."""
    if max_unroll <= 1:
        return 1
    warm = int(warm_frac * epochs)
    if epoch < warm:
        return 1
    n_stages = max(1, int(math.log2(max_unroll)))
    ramp_end = max(warm + n_stages, int(ramp_frac * epochs))
    span = max(1.0, (ramp_end - warm) / n_stages)
    return int(2 ** min(n_stages, int((epoch - warm) // span) + 1))


def try_compile(model, lattices, batch_size, device, amp_ctx):
    """torch.compile with a fallback that actually fires."""
    try:
        compiled = torch.compile(model, dynamic=False)
        dtype = next(model.parameters()).dtype
        for n in dict.fromkeys(lattices):
            with amp_ctx():
                compiled(torch.zeros(batch_size, 1, n, n, device=device,
                                     dtype=dtype)).sum().backward()
        model.zero_grad(set_to_none=True)
        print(f"[setup] torch.compile active for {sorted(set(lattices))}",
              flush=True)
        return compiled
    except Exception as exc:
        model.zero_grad(set_to_none=True)
        print(f"[setup] torch.compile failed ({type(exc).__name__}: "
              f"{str(exc)[:140]}); running eager", flush=True)
        return model


def print_config(model=None, extra=None):
    """Dump every CONFIG constant, for the record in the job log."""
    skip = {"TypeVar"}
    rows = [(k, v) for k, v in sorted(globals().items())
            if k.isupper() and k not in skip
            and isinstance(v, (int, float, str, bool, tuple, list, type(None)))]
    width = max(len(k) for k, _ in rows)
    print("\n" + "=" * 72, flush=True)
    print("CONFIGURATION", flush=True)
    print("=" * 72, flush=True)
    for k, v in rows:
        print(f"  {k:<{width}}  {v}", flush=True)
    if model is not None:
        print("-" * 72, flush=True)
        print("  MODEL", flush=True)
        for k, v in sorted(model.config.items()):
            print(f"    {k:<{width - 2}}  {v}", flush=True)
        n = model.n_parameters()
        dtype = next(model.parameters()).dtype
        bytes_per = torch.finfo(dtype).bits // 8
        print(f"    {'parameters':<{width - 2}}  {n:,}", flush=True)
        print(f"    {'dtype':<{width - 2}}  {dtype}", flush=True)
        print(f"    {'weight memory':<{width - 2}}  "
              f"{n * bytes_per / 1e6:.1f} MB", flush=True)
    if extra:
        print("-" * 72, flush=True)
        for k, v in extra.items():
            print(f"  {k:<{width}}  {v}", flush=True)
    print("=" * 72 + "\n", flush=True)


def main():
    t_launch = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device, amp_ctx = device_and_precision()

    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print(f"[setup] {prop.name}, {prop.total_memory / 1e9:.1f} GB, "
              f"torch {torch.__version__}", flush=True)
    print(f"[setup] device={device} precision={PRECISION} amp={AMP} "
          f"compile={COMPILE}", flush=True)

    train_arr, val_arr = load_trajectories(DATASET)
    lattice = train_arr.shape[-1]

    if OFF_CRITICAL and OFF_CRITICAL_TRAJ > 0 and OFF_CRITICAL_OFFS:
        extra = off_critical_trajectories(lattice, train_arr.shape[1],
                                          str(device))
        train_arr = np.concatenate([train_arr, extra], axis=0)
        share = 100.0 * len(extra) / train_arr.shape[0]
        del extra
        print(f"[off] train set is now {train_arr.shape} "
              f"({share:.0f}% off-critical)", flush=True)

    data_device = device
    if GPU_DATA == "off" or device.type != "cuda":
        data_device = torch.device("cpu")
    elif GPU_DATA == "auto":
        free, _ = torch.cuda.mem_get_info(device)
        if train_arr.nbytes + val_arr.nbytes > 0.4 * free:
            data_device = torch.device("cpu")
    print(f"[data] sampling from {data_device}", flush=True)

    train_data = WindowSampler(train_arr, device, data_device, 1, True, SEED)
    val_data = WindowSampler(val_arr, device, data_device, 1, False, SEED + 1)
    del train_arr, val_arr

    model = build_model(n_ref=lattice, device=device)
    print(f"[model] mode_scaling={MODE_SCALING}, zero_dc={ZERO_DC}, "
          f"{model.n_parameters():,} parameters, dtype={torch_dtype()}",
          flush=True)
    print(f"[model] HARD_CONSTRAINTS={HARD_CONSTRAINTS} -> "
          f"hard_mass_conservation={HARD_MASS_CONSERVATION} "
          f"zero_dc={ZERO_DC} residual_prediction={RESIDUAL_PREDICTION}",
          flush=True)
    if not HARD_CONSTRAINTS:
        print(f"[model] ablation: mass is only encouraged by the soft penalty "
              f"(MASS_WEIGHT={MASS_WEIGHT}); writing to {MODEL_OUT} so the "
              f"constrained model is not overwritten", flush=True)
    step_model = (try_compile(model, [lattice, VAL_LATTICE], BATCH_SIZE,
                              device, amp_ctx) if COMPILE else model)

    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)

    def cosine(epoch):
        p = min(epoch / max(EPOCHS, 1), 1.0)
        base = LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * p))
        stages = int(math.log2(unroll_at(epoch, EPOCHS, MAX_UNROLL)))
        return base * (LR_STAGE_DECAY ** stages)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine)

    start_epoch, history = 0, []
    best = {"score": -1.0, "survival": -1.0}
    if RESUME and os.path.exists(CHECKPOINT):
        ck = torch.load(CHECKPOINT, map_location=device, weights_only=False)

        saved_cfg = ck.get("config", {})
        drift = {k: (saved_cfg.get(k, "<absent>"), v)
                 for k, v in model.config.items() if saved_cfg.get(k) != v}
        if drift:
            print(f"[resume] {CHECKPOINT} was written by a different model; "
                  f"ignoring it and starting fresh. Differences "
                  f"(checkpoint -> current):", flush=True)
            for k, (was, now) in drift.items():
                print(f"           {k}: {was} -> {now}", flush=True)
            print("[resume] delete it, or set RESUME = False, to silence this.",
                  flush=True)
            ck = None
    else:
        ck = None

    if ck is not None:
        model.load_state_dict(_tensors_only(ck["state_dict"]))
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, best = ck["epoch"], ck.get("best", best)
        history = ck.get("history", [])
        best.setdefault("score", -1.0)
        print(f"[resume] {CHECKPOINT} at epoch {start_epoch}", flush=True)

    print(f"[val] l={VAL_LATTICE} ground truth for seeds {list(VAL_SEEDS)}",
          flush=True)
    gts = [ch_ground_truth(VAL_LATTICE, s, 0.0, VAL_T_END, DT_STEP,
                           device=str(device)) for s in VAL_SEEDS]

    recent_losses = deque(maxlen=10)
    batch_losses = deque(maxlen=400)
    healthy = None
    recoveries = 0
    stale = 0
    stopped_early_training = False

    for epoch in range(start_epoch, EPOCHS):
        k_max = unroll_at(epoch, EPOCHS, MAX_UNROLL)
        if train_data.n_steps != k_max:
            train_data.set_n_steps(k_max)
            batch_losses.clear()

        model.train()
        t0, running, skipped, done = time.time(), 0.0, 0, 0
        spiked = 0
        for _ in range(STEPS_PER_EPOCH):
            win = train_data.sample(BATCH_SIZE)

            k = int(rng.integers(1, k_max + 1))
            x = win[:, 0:1]
            if NOISE_STD > 0 and rng.random() < NOISE_PROB:
                noise = torch.randn_like(x) * NOISE_STD
                x = x + noise - noise.mean(dim=(-2, -1), keepdim=True)

            n_detached = max(0, k - BPTT)
            if n_detached:
                with torch.no_grad(), amp_ctx():
                    for _ in range(n_detached):
                        x = step_model(x)

                if not torch.isfinite(x).all() or float(x.abs().max()) > PSI_MAX:
                    skipped += 1
                    continue
                x = x.detach()

            opt.zero_grad(set_to_none=True)
            with amp_ctx():

                step_losses, x_in = [], x
                for j in range(n_detached, k):
                    x_in = x
                    x = step_model(x)
                    step_losses.append(loss_fn(x, win[:, j + 1:j + 2]))
                loss = torch.stack(step_losses).mean()
                if CONTRACTIVITY_WEIGHT > 0:
                    loss = loss + CONTRACTIVITY_WEIGHT * amplification_penalty(
                        step_model, x_in, x)
            if SMOOTH_WEIGHT > 0:
                loss = loss + SMOOTH_WEIGHT * model.smoothness_penalty()
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
            loss_val, psi_max = torch.stack(
                [loss.detach().reshape(()), x.detach().abs().max()]).tolist()

            spike = (len(batch_losses) >= LOSS_SPIKE_WARMUP
                     and LOSS_SPIKE_FACTOR > 0
                     and loss_val > LOSS_SPIKE_FACTOR
                     * float(np.median(batch_losses)))
            if spike or psi_max > PSI_MAX or not np.isfinite(psi_max):
                opt.zero_grad(set_to_none=True)
                skipped += 1
                spiked += 1

                if batch_losses:
                    cap = LOSS_SPIKE_FACTOR * float(np.median(batch_losses))
                    batch_losses.append(cap if not np.isfinite(loss_val)
                                        else min(loss_val, cap))
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            batch_losses.append(loss_val)
            running += loss_val
            done += 1
        sched.step()
        train_loss = running / done if done else float("inf")
        secs = time.time() - t0

        model.eval()
        with torch.no_grad():
            vals, deltas = [], []
            for win in val_data.fixed_batches(BATCH_SIZE, VAL_BATCHES):
                psi = win[:, 0:1]
                with amp_ctx():
                    pred = step_model(psi)
                    vals.append(float(loss_fn(pred, win[:, 1:2])))

                deltas.append(float((pred - psi).abs().mean()
                                    / psi.abs().mean().clamp_min(1e-8)))
        val_loss = float(np.mean(vals)) if vals else float("nan")
        delta_rel = float(np.mean(deltas)) if deltas else float("nan")

        median = float(np.median(recent_losses)) if recent_losses else None
        stalled = skipped > SPIKE_ABORT_FRAC * STEPS_PER_EPOCH
        unphysical = (not np.isfinite(delta_rel)) or delta_rel > MAX_DELTA_REL
        diverged = (median is not None and np.isfinite(median)
                    and (not np.isfinite(train_loss)
                         or train_loss > DIVERGE_FACTOR * median
                         or stalled or unphysical))
        if diverged and healthy is not None and recoveries < MAX_RECOVERIES:
            recoveries += 1
            model.load_state_dict(healthy["model"], strict=False)
            opt.load_state_dict(healthy["optimizer"])
            sched.base_lrs = [b * LR_RECOVERY_DECAY for b in sched.base_lrs]
            batch_losses.clear()
            why = (f"d/psi {delta_rel:.4f} > {MAX_DELTA_REL}" if unphysical
                   else f"{skipped}/{STEPS_PER_EPOCH} batches skipped"
                   if stalled else
                   f"train {train_loss:.5f} vs median {median:.5f}")
            print(f"epoch {epoch + 1:3d}/{EPOCHS} | DIVERGED ({why}); rolled "
                  f"back to epoch {healthy['epoch']}, lr x{LR_RECOVERY_DECAY} "
                  f"[{recoveries}/{MAX_RECOVERIES}]", flush=True)
            history.append({"epoch": epoch + 1, "diverged": True,
                            "train": train_loss, "skipped": skipped,
                            "stalled": bool(stalled),
                            "unphysical": bool(unphysical),
                            "delta_rel": delta_rel,
                            "rolled_back_to": healthy["epoch"]})
            continue
        if diverged:
            print(f"epoch {epoch + 1:3d}/{EPOCHS} | DIVERGED and out of "
                  f"recoveries; stopping. Best model is in {MODEL_OUT}.",
                  flush=True)
            break
        recent_losses.append(train_loss)
        healthy = {"epoch": epoch + 1,
                   "model": {k: v.detach().cpu().clone() for k, v in
                             _tensors_only(model.state_dict()).items()},
                   "optimizer": copy.deepcopy(opt.state_dict())}

        line = (f"epoch {epoch + 1:3d}/{EPOCHS} | unroll {k_max:2d} | "
                f"train {train_loss:.5f} | val1 {val_loss:.5f} | "
                f"d/psi {delta_rel:.4f} | lr {sched.get_last_lr()[0]:.2e} | "
                f"{secs:5.1f}s ({secs / STEPS_PER_EPOCH * 1e3:.0f} ms/step)"
                + (f" | {skipped} skipped"
                   + (f" ({spiked} spikes)" if spiked else "")
                   if skipped else ""))
        rec = {"epoch": epoch + 1, "unroll": k_max, "train": train_loss,
               "val1": val_loss, "delta_rel": delta_rel, "seconds": secs,
               "skipped": skipped, "spiked": spiked}

        if (epoch + 1) % VAL_EVERY == 0 or epoch == EPOCHS - 1:
            per_seed = [rollout_score(model, t, f, str(device), amp_ctx)
                        for t, f in gts]
            score = float(np.mean([p[0] for p in per_seed]))
            surv = float(np.mean([p[1] for p in per_seed]))
            rec.update(score=score, survival=surv,
                       per_seed=[p[1] for p in per_seed])
            line += (f" | rollout {score:.4f} | survives t={surv:6.0f} "
                     f"{[int(p[1]) for p in per_seed]}")
            if score > best["score"] + MIN_DELTA:
                best = {"score": score, "survival": surv, "epoch": epoch + 1,
                        "per_seed": [p[1] for p in per_seed]}
                save_model(model, MODEL_OUT, epoch=epoch + 1, best=best)
                line += "  <- best"
                stale = 0
            elif PATIENCE_AFTER_RAMP and k_max < MAX_UNROLL:
                line += f"  (no gain; patience held, unroll {k_max}<{MAX_UNROLL})"
            else:
                stale += 1
                line += f"  (no gain {stale}/{PATIENCE})" if PATIENCE else ""
        print(line, flush=True)
        history.append(rec)

        torch.save({"epoch": epoch + 1,
                    "state_dict": _tensors_only(model.state_dict()),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "best": best,
                    "config": model.config, "history": history}, CHECKPOINT)
        with open(os.path.splitext(MODEL_OUT)[0] + "_history.json", "w") as fh:
            json.dump(history, fh, indent=1)

        if PATIENCE and stale >= PATIENCE:
            print(f"[stop] rollout score has not improved by {MIN_DELTA} for "
                  f"{stale} validations ({stale * VAL_EVERY} epochs); best was "
                  f"{best['score']:.4f} at epoch {best.get('epoch')}",
                  flush=True)
            stopped_early_training = True
            break

        if MAX_HOURS and (time.time() - t_launch) > MAX_HOURS * 3600:
            print(f"[budget] {MAX_HOURS} h reached after epoch {epoch + 1}; "
                  f"resubmit to continue from {CHECKPOINT}", flush=True)
            break

    if device.type == "cuda":
        print(f"[mem] peak allocated "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    print(f"[done] best rollout score {best['score']:.4f} (survives to "
          f"t = {best['survival']:.0f}, epoch {best.get('epoch')}); "
          f"model saved to {MODEL_OUT}"
          + ("  [stopped early: no further improvement]"
             if stopped_early_training else ""), flush=True)

    hours = (time.time() - t_launch) / 3600.0
    print_config(model, extra={
        "epochs run": f"{len(history)} of {EPOCHS}",
        "best epoch": best.get("epoch"),
        "best rollout score": f"{best['score']:.4f}",
        "best survival": f"t = {best['survival']:.0f}",
        "per-seed survival": best.get("per_seed"),
        "divergence recoveries": f"{recoveries} of {MAX_RECOVERIES}",
        "batches skipped by the spike guard":
            sum(r.get("spiked", 0) for r in history),
        "wall clock": f"{hours:.2f} h",
        "peak GPU memory": (f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
                            if device.type == "cuda" else "n/a")})

if __name__ == "__main__":
    main()