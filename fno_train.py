from __future__ import annotations

import contextlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================================
# CONFIG
# ==========================================================================

# -- data ------------------------------------------------------------------
DATASET = "test.npz"          # produced by the advisor's ch.py
MODEL_OUT = "fno_ch.pt"       # best model, selected on rollout survival
CHECKPOINT = "fno_ch_last.pt"  # resumable state; set RESUME below to use it
RESUME = True                 # continue from CHECKPOINT if it exists

# ch.py's __main__: dt = 0.01, n_ahead = 100, gap = 10.  One model step is
# therefore n_ahead * dt = 1.0 of real time, and consecutive stored frames are
# gap * dt = 0.1 apart, so a dt = 1.0 step is STRIDE frames along the array.
DT = 0.01
N_AHEAD = 100
GAP = 10
STRIDE = N_AHEAD // GAP        # verified against the file at load time
DT_STEP = N_AHEAD * DT         # 1.0

# -- model -----------------------------------------------------------------
CHANNELS = 64
MODES = 24                     # physical-k cutoff, defined on the l = 64 grid
LAYERS = 5
LOCAL_KERNEL = 3               # circular conv alongside the spectral path
MODE_SCALING = "physical"      # "physical" | "index" ("index" = stock FNO)
ZERO_DC = True                 # pin R(k=0) = 0; see the note in SpectralConv2d

# -- optimisation ----------------------------------------------------------
EPOCHS = 200
BATCH_SIZE = 32
STEPS_PER_EPOCH = 1500
LR = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
LR_FLOOR = 1e-2                # final LR = LR * LR_FLOOR (cosine)

# -- rollout training ------------------------------------------------------
MAX_UNROLL = 16                # curriculum: 1, then equal stages 2, 4, ..., this
BPTT = 2                       # trailing steps that carry gradient
WARM_FRAC = 0.2                # fraction of epochs at unroll = 1
NOISE_STD = 0.01
NOISE_PROB = 0.5

# -- loss ------------------------------------------------------------------
H1_WEIGHT = 0.5                # relative H1 seminorm; weights the interfaces
SMOOTH_WEIGHT = 1e-3           # smoothness of R(k), so resampling is well-posed

# -- validation (the selection metric) ------------------------------------
VAL_SEEDS = (31337, 24601, 8675309)   # unseen; averaging tames a noisy metric
VAL_LATTICE = 128
VAL_T_END = 2000.0
VAL_T_START = 100.0
VAL_EVERY = 5                  # epochs between rollout validations
VAL_BATCHES = 32               # batches for the cheap one-step val loss
R2_THRESHOLD = 0.80
GT_CACHE = "gt_cache"

# -- hardware --------------------------------------------------------------
AMP = "bf16"                   # "bf16" | "off"; spectral path stays fp32
TF32 = True                    # A100 and newer
COMPILE = False                # torch.compile; falls back to eager on failure
GPU_DATA = "auto"              # "auto" | "on" | "off": hold the set in VRAM
MAX_HOURS = 0.0                # >0 stops cleanly before a SLURM wall limit
SEED = 0

# ==========================================================================


def device_and_precision():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    def amp_ctx():
        if AMP == "bf16" and dev.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()
    return dev, amp_ctx


# ==========================================================================
# Reference solver -- ch.py's PhaseOrdering.chc
# ==========================================================================
# Needed here only to build the l = 128 validation trajectories; the training
# data itself comes from test.npz.  Identical scheme to ch.py: explicit Euler,
# 5-point Laplacian, dx = 1, periodic.  The difference is bookkeeping: ch.py
# materialises the whole history, which at l = 128 and t = 2000 is
# 200000 * 128 * 128 * 8 bytes = 26 GB and cannot be allocated.  This streams.

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
    """Ginzburg-Landau density.  Cahn-Hilliard is a gradient flow, so this must
    decrease monotonically -- a check R2 cannot make."""
    gx = np.roll(psi, -1, 0) - psi
    gy = np.roll(psi, -1, 1) - psi
    return float(np.mean(0.25 * (psi ** 2 - 1.0) ** 2
                         + 0.5 * (gx ** 2 + gy ** 2)))


def ch_ground_truth(l, seed, off=0.0, t_end=2000.0, dt_step=DT_STEP,
                    cache_dir=GT_CACHE, device="cpu", verbose=True):
    """Reference trajectory sampled every ``dt_step``, cached on disk.

    Only the sampled frames are kept: 2001 frames at l = 128 is 125 MB, against
    26 GB for the full history.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir,
                        f"gt_l{l}_seed{seed}_off{off:+.2f}_t{t_end:g}.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return z["times"], z["frames"]

    every = int(round(dt_step / DT))
    n_total = int(round(t_end / DT))
    n_frames = n_total // every + 1
    if verbose:
        print(f"[gt] solving l={l} seed={seed} off={off:+.2f} to t={t_end:g} "
              f"({n_total:,} steps)", flush=True)

    frames = np.empty((n_frames, l, l), dtype=np.float32)
    times = np.arange(n_frames) * dt_step
    psi = initial_condition(l, seed, off)

    if device != "cpu":
        psi_t = torch.as_tensor(psi, device=device, dtype=torch.float64)

        def lap(p):
            return (torch.roll(p, 1, -2) + torch.roll(p, -1, -2)
                    + torch.roll(p, 1, -1) + torch.roll(p, -1, -1)
                    - 4.0 * p) / DX ** 2
        buf = torch.empty((n_frames, l, l), dtype=torch.float32, device=device)
        buf[0] = psi_t.float()
        j = 1
        for n in range(1, n_total + 1):
            psi_t = psi_t + lap(psi_t ** 3 - psi_t - lap(psi_t)) * DT
            if n % every == 0:
                buf[j] = psi_t.float()
                j += 1
        frames = buf.cpu().numpy()
    else:
        frames[0] = psi
        j = 1
        for n in range(1, n_total + 1):
            psi = psi + _lap_np(psi ** 3 - psi - _lap_np(psi)) * DT
            if n % every == 0:
                frames[j] = psi
                j += 1
                if verbose and j % 500 == 0:
                    print(f"    t = {n * DT:8.1f} / {t_end:.1f}", flush=True)

    np.savez(path, times=times, frames=frames)
    return times, frames


# ==========================================================================
# Data -- test.npz
# ==========================================================================

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
        val = npz["val_X"].astype(np.float32)
        val_y = npz["val_y"]
        stride = _detect_stride(val, val_y)
        del val_y
        train = npz["train_X"].astype(np.float32)

    if stride != STRIDE:
        raise ValueError(
            f"the dataset's label offset is {stride} frames, not STRIDE="
            f"{STRIDE}. ch.py's n_ahead/gap must have changed; set "
            f"N_AHEAD and GAP in the CONFIG block to match.")

    if verbose:
        print(f"[data] train {train.shape}  val {val.shape}  float32, "
              f"{(train.nbytes + val.nbytes) / 1e9:.2f} GB", flush=True)
        print(f"[data] label identity verified: y[s] == X[s+{stride}], so one "
              f"model step = {stride} frames = {DT_STEP:g} of real time",
              flush=True)
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


# ==========================================================================
# Model
# ==========================================================================

class SpectralConv2d(nn.Module):
    """Learned Fourier multiplier on a grid of *physical* wavenumbers.

    A stock SpectralConv2d indexes its multiplier by integer mode number.  On
    an N-point grid mode j is the wavenumber k = 2*pi*j/(N*dx), so with dx
    pinned to 1 the same weights mean different physics at l = 64 and l = 128,
    and truncating at j <= MODES halves the retained k_max when N doubles --
    which erases exactly the interfacial structure that dominates a coarsening
    field.  Here the table is defined at ``n_ref`` and resampled onto whatever
    lattice arrives, so R(k) is always applied at the k it was trained at.

    Because irfft2 divides by N^2, and at dx = 1 that factor is exactly the
    (2*pi)^-d measure of the k-integral, the real-space kernel this represents
    is independent of N -- no rescaling is needed.  A tiling test confirms it:
    F_128(tile(u)) vs tile(F_64(u)) gives 7e-02 with mode-indexed weights and
    1e-05 (float32 round-off) with these.
    """

    # declared so type checkers can see the registered buffer; without this,
    # `self.dc_mask` resolves through nn.Module.__getattr__ to Tensor | Module
    # and editors flag the multiply below
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
        # Pinning R(k=0) = 0 makes the resampling well-posed below the training
        # box's fundamental: training at n_ref = 64 never sees k < 2*pi/64, so
        # the multiplier at the 128-box fundamental k = 2*pi/128 interpolates
        # between the k = 0 entry and the first trained mode.  Zero is also the
        # physical value -- conserved dynamics does nothing at infinite
        # wavelength.  Not persistent, so checkpoints load either way.
        mask = torch.ones(1, 1, 2 * modes + 1, modes + 1, 1)
        if zero_dc:
            mask[0, 0, modes, 0, 0] = 0.0
        self.register_buffer("dc_mask", mask, persistent=False)
        self._cache = {}

    @property
    def effective_weight(self):
        return (self.weight * self.dc_mask) if self.zero_dc else self.weight

    def modes_for(self, n):
        cap = max(1, (n - 1) // 2)   # keep the +k and -k blocks from overlapping
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
        # align_corners=True maps output index b to input index b*M/m_n, which
        # is exactly the physical-k correspondence n_src = n_tgt * n_ref / n
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
        # Complex arithmetic has no autocast kernels, and a bf16 FFT would
        # discard the dynamic range the spectrum needs, so this block is fp32.
        dev = x.device.type
        ctx = (torch.autocast(device_type=dev, enabled=False)
               if dev in ("cuda", "cpu") else contextlib.nullcontext())
        with ctx:
            x = x.float()
            w = torch.view_as_complex(
                self._resampled_weight(n).float().contiguous())
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
        """Resampling to a denser k-grid is only meaningful if R varies smoothly
        between the k-points it was trained on; this makes that true."""
        w = self.effective_weight
        return ((w[:, :, 1:] - w[:, :, :-1]).pow(2).mean()
                + (w[:, :, :, 1:] - w[:, :, :, :-1]).pow(2).mean())


class FNOBlock(nn.Module):
    """Spectral (global) path + local stencil path, wired residually.

    The local path is a 3x3 circular convolution rather than the usual 1x1: at
    fixed dx it is resolution-consistent, and it gives the network a direct
    finite-difference route to the Laplacians that dominate the operator.
    """

    def __init__(self, ch, modes, n_ref, mode_scaling, local_kernel, zero_dc):
        super().__init__()
        self.spectral = SpectralConv2d(ch, ch, modes, n_ref, mode_scaling,
                                       zero_dc)
        self.local = nn.Conv2d(ch, ch, local_kernel, padding=local_kernel // 2,
                               padding_mode="circular")
        self.mix = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return x + self.mix(F.gelu(self.spectral(x) + self.local(x)))


class CahnHilliardFNO(nn.Module):
    """psi(t) -> psi(t + 1.0), as an increment with the mean projected out."""

    def __init__(self, channels=CHANNELS, modes=MODES, n_layers=LAYERS,
                 n_ref=64, mode_scaling=MODE_SCALING,
                 local_kernel=LOCAL_KERNEL, zero_dc=ZERO_DC,
                 use_mean_channel=True):
        super().__init__()
        self.config = dict(channels=channels, modes=modes, n_layers=n_layers,
                           n_ref=n_ref, mode_scaling=mode_scaling,
                           local_kernel=local_kernel, zero_dc=zero_dc,
                           use_mean_channel=use_mean_channel)
        self.use_mean_channel = use_mean_channel
        self.lift = nn.Sequential(
            nn.Conv2d(2 if use_mean_channel else 1, channels, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 1))
        self.blocks = nn.ModuleList([
            FNOBlock(channels, modes, n_ref, mode_scaling, local_kernel,
                     zero_dc) for _ in range(n_layers)])
        # zero the output conv through a local reference: indexing back into
        # the Sequential returns a bare Module, whose .weight is Tensor | Module
        # and cannot be initialised without a cast.  Layer names are unchanged
        # (project.0, project.2), so existing checkpoints still load.
        hidden_conv = nn.Conv2d(channels, 2 * channels, 1)   # built in this
        out_conv = nn.Conv2d(2 * channels, 1, 1)             # order to keep
        with torch.no_grad():                                # the RNG stream
            # start near the identity -- the right prior for one small step
            out_conv.weight.zero_()
            if out_conv.bias is not None:
                out_conv.bias.zero_()
        self.project = nn.Sequential(hidden_conv, nn.GELU(), out_conv)

    def forward(self, psi):
        if psi.dim() == 3:
            psi = psi.unsqueeze(1)
        x = psi
        if self.use_mean_channel:
            x = torch.cat([psi, psi.mean(dim=(-2, -1), keepdim=True)
                           .expand_as(psi)], dim=1)
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        delta = self.project(h)
        # exact conservation: the increment carries zero net mass
        return psi + (delta - delta.mean(dim=(-2, -1), keepdim=True))

    def smoothness_penalty(self):
        terms = [m.smoothness_penalty() for m in self.modules()
                 if isinstance(m, SpectralConv2d)]
        return torch.stack(terms).mean()

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


def save_model(model, path, **extra):
    torch.save({"config": model.config, "state_dict": model.state_dict(),
                **extra}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no 'config' entry -- not a checkpoint "
                         f"from fno_train.py")
    model = CahnHilliardFNO(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


# ==========================================================================
# Loss
# ==========================================================================

def _rel_l2(pred, target, eps=1e-8):
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    return (torch.linalg.vector_norm(p - t, dim=1)
            / (torch.linalg.vector_norm(t, dim=1) + eps))


def loss_fn(pred, target):
    """Relative L2 plus the relative H1 seminorm.

    At late times a Cahn-Hilliard field is flat at +-1 almost everywhere and
    all the information lives in the interfaces, which is what the gradient
    term weights.  There is no mass-conservation penalty: the model projects
    the mean out of its increment, so that term would always be zero.
    """
    h1_weight = H1_WEIGHT
    loss = _rel_l2(pred, target).mean()
    if h1_weight > 0:
        gp = torch.cat([torch.roll(pred, -1, -2) - pred,
                        torch.roll(pred, -1, -1) - pred], dim=1)
        gt = torch.cat([torch.roll(target, -1, -2) - target,
                        torch.roll(target, -1, -1) - target], dim=1)
        loss = loss + h1_weight * _rel_l2(gp, gt).mean()
    return loss


# ==========================================================================
# Autoregressive rollout (shared with fno_predict.py)
# ==========================================================================

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
    """Yield (t, true, pred) for each autoregressive step from ``t_start``.

    The ground truth must be sampled exactly every DT_STEP so predictions and
    truth stay aligned without interpolation.
    """
    dt_gt = float(gt_times[1] - gt_times[0])
    if abs(dt_gt - DT_STEP) > 1e-9:
        raise ValueError(f"ground truth sampled every {dt_gt}, model step is "
                         f"{DT_STEP}")
    i0 = int(np.argmin(np.abs(gt_times - t_start)))
    i1 = int(np.argmin(np.abs(gt_times - t_end)))
    model.eval()
    ctx = amp_ctx if amp_ctx is not None else contextlib.nullcontext
    psi = torch.as_tensor(gt_frames[i0], dtype=torch.float32,
                          device=device)[None, None]
    for k in range(i1 - i0):
        with ctx():
            psi = model(psi)
        j = i0 + k + 1
        yield (float(gt_times[j]),
               gt_frames[j].astype(np.float64),
               psi[0, 0].detach().float().cpu().numpy().astype(np.float64))


def rollout_survival(model, gt_times, gt_frames, device, amp_ctx=None,
                     t_start=VAL_T_START, t_end=VAL_T_END,
                     threshold=R2_THRESHOLD):
    """Time at which R2 first drops below ``threshold`` (the selection metric)."""
    scores = []
    for t, true, pred in rollout_frames(model, gt_times, gt_frames, t_start,
                                        t_end, device, amp_ctx):
        scores.append(r2_score(true, pred))
        if scores[-1] < threshold:
            return t, float(np.mean(scores))
    return t_end, float(np.mean(scores)) if scores else 0.0


# ==========================================================================
# Training
# ==========================================================================

def unroll_at(epoch, epochs, max_unroll, warm_frac=WARM_FRAC):
    """One-step warmup, then equal-length stages at K = 2, 4, ..., max_unroll.

    A continuous ramp in log2(K), rounded to an integer, leaves most epochs at
    K = 1 or 2 and reaches the top only at the very end -- which starves the
    phase that actually teaches the model to survive its own feedback.
    """
    if max_unroll <= 1:
        return 1
    warm = int(warm_frac * epochs)
    if epoch < warm:
        return 1
    n_stages = max(1, int(math.log2(max_unroll)))
    span = max(1.0, (epochs - warm) / n_stages)
    return int(2 ** min(n_stages, int((epoch - warm) // span) + 1))


def try_compile(model, lattices, batch_size, device, amp_ctx):
    """torch.compile with a fallback that actually fires.

    torch.compile returns immediately and only traces on the first call, so
    wrapping the call alone catches nothing and an inductor failure on the
    complex FFT path would kill the job mid-training.  Force it here, once per
    lattice size, and drop back to eager if any of them fails.
    """
    try:
        compiled = torch.compile(model, dynamic=False)
        for n in dict.fromkeys(lattices):
            with amp_ctx():
                compiled(torch.zeros(batch_size, 1, n, n,
                                     device=device)).sum().backward()
        model.zero_grad(set_to_none=True)
        print(f"[setup] torch.compile active for {sorted(set(lattices))}",
              flush=True)
        return compiled
    except Exception as exc:
        model.zero_grad(set_to_none=True)
        print(f"[setup] torch.compile failed ({type(exc).__name__}: "
              f"{str(exc)[:140]}); running eager", flush=True)
        return model


def main():
    t_launch = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device, amp_ctx = device_and_precision()

    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print(f"[setup] {prop.name}, {prop.total_memory / 1e9:.1f} GB, "
              f"torch {torch.__version__}", flush=True)
    print(f"[setup] device={device} amp={AMP} tf32={TF32} compile={COMPILE}",
          flush=True)

    train_arr, val_arr = load_trajectories(DATASET)
    lattice = train_arr.shape[-1]

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

    # read the CONFIG globals explicitly rather than relying on the class's
    # default arguments, which bind at import time
    model = CahnHilliardFNO(channels=CHANNELS, modes=MODES, n_layers=LAYERS,
                            n_ref=lattice, mode_scaling=MODE_SCALING,
                            local_kernel=LOCAL_KERNEL,
                            zero_dc=ZERO_DC).to(device)
    print(f"[model] {MODE_SCALING} spectral weights, zero_dc={ZERO_DC}, "
          f"{model.n_parameters():,} parameters", flush=True)
    step_model = (try_compile(model, [lattice, VAL_LATTICE], BATCH_SIZE,
                              device, amp_ctx) if COMPILE else model)

    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)

    # Closed-form cosine rather than CosineAnnealingLR, which updates
    # recursively and so inherits a stale rate when a job resumes.
    def cosine(epoch):
        p = min(epoch / max(EPOCHS, 1), 1.0)
        return LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine)

    start_epoch, history = 0, []
    best = {"survival": -1.0, "mean_r2": -1e9}
    if RESUME and os.path.exists(CHECKPOINT):
        ck = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, best = ck["epoch"], ck.get("best", best)
        history = ck.get("history", [])
        print(f"[resume] {CHECKPOINT} at epoch {start_epoch}", flush=True)

    print(f"[val] l={VAL_LATTICE} ground truth for seeds {list(VAL_SEEDS)}",
          flush=True)
    gts = [ch_ground_truth(VAL_LATTICE, s, 0.0, VAL_T_END, DT_STEP,
                           device=str(device)) for s in VAL_SEEDS]

    for epoch in range(start_epoch, EPOCHS):
        k_max = unroll_at(epoch, EPOCHS, MAX_UNROLL)
        if train_data.n_steps != k_max:
            train_data.set_n_steps(k_max)

        model.train()
        t0, running = time.time(), 0.0
        for _ in range(STEPS_PER_EPOCH):
            win = train_data.sample(BATCH_SIZE)
            # a mixed horizon keeps the one-step map sharp while the
            # long-horizon behaviour is being learned
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
                x = x.detach()

            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                # collect then stack, rather than accumulating onto a float:
                # keeps `loss` a Tensor throughout, so .backward() below is
                # statically valid as well as correct
                step_losses = []
                for j in range(n_detached, k):
                    x = step_model(x)
                    step_losses.append(loss_fn(x, win[:, j + 1:j + 2]))
                loss = torch.stack(step_losses).mean()
            if SMOOTH_WEIGHT > 0:
                loss = loss + SMOOTH_WEIGHT * model.smoothness_penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            running += float(loss.detach())
        sched.step()
        train_loss = running / STEPS_PER_EPOCH
        secs = time.time() - t0

        model.eval()
        with torch.no_grad():
            vals = []
            for win in val_data.fixed_batches(BATCH_SIZE, VAL_BATCHES):
                with amp_ctx():
                    vals.append(float(loss_fn(step_model(win[:, 0:1]),
                                              win[:, 1:2])))
        val_loss = float(np.mean(vals)) if vals else float("nan")

        line = (f"epoch {epoch + 1:3d}/{EPOCHS} | unroll {k_max:2d} | "
                f"train {train_loss:.5f} | val1 {val_loss:.5f} | "
                f"lr {sched.get_last_lr()[0]:.2e} | "
                f"{secs:5.1f}s ({secs / STEPS_PER_EPOCH * 1e3:.0f} ms/step)")
        rec = {"epoch": epoch + 1, "unroll": k_max, "train": train_loss,
               "val1": val_loss, "seconds": secs}

        if (epoch + 1) % VAL_EVERY == 0 or epoch == EPOCHS - 1:
            per_seed = [rollout_survival(model, t, f, str(device), amp_ctx)
                        for t, f in gts]
            surv = float(np.mean([p[0] for p in per_seed]))
            mean_r2 = float(np.mean([p[1] for p in per_seed]))
            rec.update(survival=surv, mean_r2=mean_r2,
                       per_seed=[p[0] for p in per_seed])
            line += (f" | rollout t={surv:6.0f} "
                     f"{[int(p[0]) for p in per_seed]} (R2 {mean_r2:.3f})")
            if (surv > best["survival"]
                    or (surv == best["survival"] and mean_r2 > best["mean_r2"])):
                best = {"survival": surv, "mean_r2": mean_r2,
                        "epoch": epoch + 1,
                        "per_seed": [p[0] for p in per_seed]}
                save_model(model, MODEL_OUT, epoch=epoch + 1, best=best)
                line += "  <- best"
        print(line, flush=True)
        history.append(rec)

        torch.save({"epoch": epoch + 1, "state_dict": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "best": best,
                    "config": model.config, "history": history}, CHECKPOINT)
        with open(os.path.splitext(MODEL_OUT)[0] + "_history.json", "w") as fh:
            json.dump(history, fh, indent=1)

        if MAX_HOURS and (time.time() - t_launch) > MAX_HOURS * 3600:
            print(f"[budget] {MAX_HOURS} h reached after epoch {epoch + 1}; "
                  f"resubmit to continue from {CHECKPOINT}", flush=True)
            break

    if device.type == "cuda":
        print(f"[mem] peak allocated "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    print(f"[done] best mean rollout survived to t = {best['survival']:.0f} "
          f"(epoch {best.get('epoch')}); model saved to {MODEL_OUT}", flush=True)


if __name__ == "__main__":
    main()
