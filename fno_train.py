"""FNO training for the Cahn-Hilliard equation."""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import errno
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from collections import deque
from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fno_gt
import fno_paths
import fno_runlog

_ModuleT = TypeVar("_ModuleT", bound=nn.Module)

WRITE_LOG = True

DATASET = os.path.join(fno_paths.data_dir(), "test.npz")
T_TRAIN = None
TAG = ""
RESUME = True

HARD_CONSTRAINTS = True
# None means Work/models/<train label>/fno_ch.pt and fno_ch_last.pt, worked
# out once the training horizon is known (resolve_outputs).
MODEL_OUT: str | None = None
CHECKPOINT: str | None = None
# None means the folders under Work/.  --smoke makes it Work/smoke, and the
# model files are then named under it (resolve_outputs).
ROOT: str | None = None

OFF_CRITICAL = False
OFF_CRITICAL_OFFS = (0.2, 0.4)
OFF_CRITICAL_TRAJ = 3
OFF_CRITICAL_SEED0 = 700000

DT = 0.01
N_AHEAD = 100
GAP = 10
STRIDE = N_AHEAD // GAP
DT_STEP = N_AHEAD * DT

CHANNELS = 38
MODES = 24
LAYERS = 5
LOCAL_KERNEL = 3
MODE_SCALING = "physical"

ZERO_DC = True
HARD_MASS_CONSERVATION = True
RESIDUAL_PREDICTION = True
MASS_WEIGHT = 0.1
# The head predicts M and the step is lap5(M), the solver's own stencil.
# --variant base turns it off; the no-hard-constraints ablation does so by
# itself (apply_overrides).
LAPLACIAN_OUTPUT = True
# On a lattice other than n_ref the spectral weights are resampled
# (SpectralConv2d._resampled_weight).  With zero_dc the masked DC entry would
# pull the new modes next to it towards 0; LOWK_FILL puts the mean of its
# trained neighbours there before the resampling.  On since 2026-10-22 (job
# 561907: R2(10000) 0.54 -> 0.86 at the critical quench); a model keeps its
# own setting in its config (lowk_fill; absent in older checkpoints, which
# are off), and set_lowk_fill changes it on a model that is already built.
LOWK_FILL = True
# The model gets the exact chemical potential mu = psi^3 - psi - lap(psi) of
# its input as a channel (CahnHilliardFNO.mu_channel).  The step is already
# lap5 of the head's output, and where mu is flat the true dynamics barely
# move, so the model can see the bulk equilibrium directly.  The training
# default from 2026-10-23, an experiment against job 562065; a model keeps its
# own setting in its config (mu_channel; absent in older checkpoints, which
# are off).  --fixed-step puts it back off (FIXED_STEP).
MU_CHANNEL = True
# The channel is cubic, so a one-pixel spike in the model's own state (where a
# small droplet dissolves, |psi| 2 to 3 after one step) is fed back as a much
# larger input and the rollout blows up past |psi| 5.  With MU_CLIP the channel
# is computed from psi clamped to [-MU_CLIP, MU_CLIP] (psi itself, the residual
# and the head are not clamped), and the spike heals.  2026-10-24, the seed-1
# divergence diagnosis: true |psi| stays below about 1.05 (training inputs with
# noise about 1.06), so the clip does not change training.  A model keeps its
# own setting in its config (mu_clip; absent in older checkpoints, which have
# none), set_mu_clip changes it on a model that is built, and fno_predict's
# MU_CLIP can override it for a prediction.  None: no clip.
MU_CLIP = 1.2

if not HARD_CONSTRAINTS:
    ZERO_DC = False
    HARD_MASS_CONSERVATION = False

# Variable time step is the default kind of training (DT_CONDITIONED); the
# constants below that differ between the two kinds hold the variable-step
# value, and --fixed-step (use_fixed_step) gives back FIXED_STEP's.  The
# step sizes and the loss of the variable kind are the VDT_* / W_* block
# further down.
DT_CONDITIONED = True
EPOCHS = 80
BATCH_SIZE = 128
STEPS_PER_EPOCH = 200
LR = 3e-3
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

TRAIN_ON_VAL = True

EMA = True
EMA_DECAY = 0.999

# The group protocol's ten validation seeds (they are also the tuning seeds of a prediction)
VAL_SEEDS = fno_gt.VAL_SEEDS
VAL_LATTICE = 128

MONITOR_SEEDS = (1618033, 1414213, 1732050, 2236067, 2645751)
MONITOR_OFFS = (-0.4, -0.2, 0.0, 0.2, 0.4)
MONITOR_LATTICE = 64
MONITOR_T_END = 2000.0
VAL_T_END = 10000.0
VAL_T_START = 100.0
VAL_EVERY = 2
VAL_BATCHES = 32

PATIENCE = 0
PATIENCE_AFTER_RAMP = True
MIN_DELTA = 2e-3
R2_THRESHOLD = 0.80
SELECT = "survival"
SURVIVAL_MIN_DELTA = 1.0
LOWK_T0 = 1000.0
GT_CACHE = fno_paths.gt_dir()
# the 4.0-spaced references of the variable-step validation (fno_gt)
REF_CACHE = fno_paths.ref_cache_dir()
LEAN_FORMAT = "ch_lean_v1"

PRECISION = "fp32"
AMP = "off"
COMPILE = False
GPU_DATA = "auto"
MAX_HOURS = "auto"
BUDGET_RESERVE_MIN = 10.0
# the exit code of a run that stopped on the wall-clock budget (EX_TEMPFAIL):
# non-zero, so SLURM marks the job failed and no afterok prediction starts on
# a model that is not finished
BUDGET_EXIT_CODE = 75
SEED = 0

PSI_MAX = 5.0
LOSS_SPIKE_FACTOR = 8.0
LOSS_SPIKE_WARMUP = 100
SPIKE_ABORT_FRAC = 0.10
MAX_DELTA_REL = 0.5

DIVERGE_FACTOR = 3.0
MAX_RECOVERIES = 8
LR_RECOVERY_DECAY = 0.8

REPLACE_RETRIES = 3
REPLACE_WAIT = 1.0

# The variable-step kind of training (the group protocol's method).  A chain of n steps
# starts at a random frame from T_MIN on, every step of it as long as one
# stride drawn log-uniformly between DT_MIN and DT_MAX (a multiple of the
# frame spacing FRAME_DT).
COND_DIM = 64
DT_MIN = 1.0
DT_MAX = 50.0
T_MIN = 35.0
FRAME_DT = GAP * DT
# (fraction of the epochs at which it starts, chain length)
VDT_UNROLL_STAGES = ((0.0, 1), (0.08, 2), (0.19, 4))
VDT_GRAD_STEPS = 4
VDT_LR_MIN = 1e-5
VDT_STAGE_LR_MULT = 0.65
VDT_BETAS = (0.9, 0.95)
VDT_NOISE = 0.015
W_INC = 0.25
W_GRAD = 0.25
# validation: a schedule rollout from VAL_T_START to VAL_T_END on the shared
# references (every VAL_QUANTUM), a seed diverges at |psi| > VAL_DIVERGE_ABS
VAL_ALPHA = 0.2
VAL_DT_MIN = 1.0
VAL_DT_MAX = 40.0
VAL_QUANTUM = fno_gt.GT_EVERY
VAL_DIVERGE_ABS = 5.0
# the one-step check (val1): strides on the reference grid, windows per stride
VAL1_DTS = (4.0, 12.0, 40.0)
VAL1_WINDOWS = 24
LOWK_DT = 48.0

# What --fixed-step puts back: the training of the model that takes one
# DT_STEP per call, as it was before the variable step.
FIXED_STEP = dict(
    DT_CONDITIONED=False, BATCH_SIZE=32, STEPS_PER_EPOCH=1500, LR=2e-3,
    EMA=False, VAL_SEEDS=(31337, 24601, 8675309, 4815162, 27182818),
    VAL_T_END=2000.0, VAL_EVERY=5, PATIENCE=8, SELECT="score",
    SURVIVAL_MIN_DELTA=10.0, MU_CHANNEL=False)


def use_fixed_step():
    """Train the fixed-step model: its constants back, the variable ones
    off.  Returns the lines saying what changed."""
    moved = []
    for name, value in FIXED_STEP.items():
        if globals()[name] != value:
            moved.append(f"{name} {globals()[name]!r} -> {value!r}")
            globals()[name] = value
    return moved


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


def set_precision_flags():
    """Set PyTorch's TF32 switches and matmul precision from PRECISION, and
    return whether TF32 is allowed.  PyTorch allows TF32 in convolutions
    (cudnn.allow_tf32) by default, so a process that never calls this runs
    them in TF32 whatever PRECISION says.  Training calls it through
    device_and_precision, and so does fno_predict."""
    if PRECISION not in _TORCH_DTYPE:
        raise ValueError(f"PRECISION must be one of {sorted(_TORCH_DTYPE)}")
    allow_tf32 = PRECISION == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    return allow_tf32


def device_and_precision():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allow_tf32 = set_precision_flags()
    if allow_tf32 and dev.type == "cuda":
        print("[setup] NOTE: TF32 measured amplification 0.402 vs 0.209 and "
              "R2@400 0.048 vs 0.917 against fp32 when used for *training*.  "
              "In the prediction (2026-10-22) it changed R2 by at most 0.001 "
              "but raised the GPU tile-check difference from 2.0e-4 to "
              "8.6e-4, so the prediction now runs in the training "
              "precision", flush=True)
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
    return (np.roll(psi, 1, -2) + np.roll(psi, -1, -2)
            + np.roll(psi, 1, -1) + np.roll(psi, -1, -1) - 4.0 * psi) / DX ** 2


def free_energy(psi):
    """Ginzburg-Landau density.  Cahn-Hilliard is a gradient flow, so this must"""
    gx = np.roll(psi, -1, 0) - psi
    gy = np.roll(psi, -1, 1) - psi
    return float(np.mean(0.25 * (psi ** 2 - 1.0) ** 2
                         + 0.5 * (gx ** 2 + gy ** 2)))


def _lap_t(q):
    """The 5-point periodic Laplacian on the last two axes (torch)."""
    return (torch.roll(q, 1, -2) + torch.roll(q, -1, -2)
            + torch.roll(q, 1, -1) + torch.roll(q, -1, -1)
            - 4.0 * q) / DX ** 2


def ch_step(p):
    """One explicit-Euler Cahn-Hilliard step, on the last two axes.

    The one copy of the update the torch paths use: _integrate for a single
    reference, and fno_predict's lock-step engine for a batch of seeds.  It
    is elementwise arithmetic and roll, with no reduction, so a batch of
    seeds advances bitwise as each would alone.
    """
    return p + _lap_t(p ** 3 - p - _lap_t(p)) * DT


def _integrate(psi, n_total, every, n_frames, device="cpu", verbose=False,
               t_end=None):
    """Explicit-Euler Cahn-Hilliard, keeping every ``every``-th step.

    ``psi`` is (l, l), giving frames (n_frames, l, l); or (B, l, l), a batch
    of seeds advanced together, giving (B, n_frames, l, l).  Both updates act
    on the last two axes and are elementwise plus roll, so each seed of a
    batch advances bitwise as it would alone.
    """
    l = psi.shape[-1]
    lead = tuple(psi.shape[:-2])
    if device != "cpu":
        # The step runs on the device; the frame buffer stays on the host and
        # each frame is copied once, when it is saved, so GPU memory does not
        # grow with the number of frames.
        p = torch.as_tensor(psi, device=device, dtype=torch.float64)
        buf = np.empty((*lead, n_frames, l, l), dtype=np.float64)
        buf[..., 0, :, :] = p.cpu().numpy()
        j = 1
        for n in range(1, n_total + 1):
            p = ch_step(p)
            if n % every == 0 and j < n_frames:
                buf[..., j, :, :] = p.cpu().numpy()
                j += 1
        return buf

    frames = np.empty((*lead, n_frames, l, l), dtype=np.float64)
    frames[..., 0, :, :] = psi
    j = 1
    for n in range(1, n_total + 1):
        psi = psi + _lap_np(psi ** 3 - psi - _lap_np(psi)) * DT
        if n % every == 0 and j < n_frames:
            frames[..., j, :, :] = psi
            j += 1
            if verbose and j % 500 == 0:
                print(f"    t = {n * DT:8.1f} / {t_end:.1f}", flush=True)
    return frames


def gt_cache_name(l, seed, off, t_end):
    """File name of one cached reference; fno_predict mirrors it."""
    return f"gt_l{l}_seed{seed}_off{off:+.2f}_t{t_end:g}_f64.npz"


def _gt_grid(t_end, dt_step):
    """(steps per frame, total steps, frames, frame times) of a reference."""
    every = int(round(dt_step / DT))
    n_total = int(round(t_end / DT))
    n_frames = n_total // every + 1
    return every, n_total, n_frames, np.arange(n_frames) * dt_step


def _device_label(device):
    dev = str(device)
    if not dev.startswith("cuda"):
        return dev
    return f"{dev} {torch.cuda.get_device_name(torch.device(dev))}"


def _provenance(device):
    """What a cache file records about the run that made it."""
    return dict(device=_device_label(device), solver_dtype="float64",
                solver_dt=DT, torch_version=torch.__version__,
                created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def _savez_atomic(path, **arrays):
    """np.savez to a partial beside ``path``, then rename it into place.

    A job killed mid-write leaves only the partial, never a truncated file
    under the final name that the next run would take for a cache.  The
    partial's name carries the host and pid, so two jobs solving the same
    file write separate partials and the last rename installs identical
    arrays.
    """
    host = re.sub(r"[^\w.-]", "_", socket.gethostname())
    tmp = f"{path[:-len('.npz')]}.{host}-{os.getpid()}.partial.npz"
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    try:
        os.replace(tmp, path)
    except OSError:
        print(f"  -> could not rename {tmp} to {path}; the finished file is "
              f"left as {tmp}", flush=True)
        raise


def _load_npz(path, names, tag):
    """The arrays ``names`` of a cache file, or None if it will not load.

    A damaged file (truncated, not a zip, a member missing) is moved aside as
    <path>.corrupt, with .<time_ns> added if that is taken, and the caller
    solves it again; nothing is deleted.  A MemoryError or a real OSError is
    raised instead: those say something about the machine, not the file.
    """
    try:
        # opened here: on a damaged zip np.load raises without closing its
        # own handle, and on Windows that blocks moving the file aside
        with open(path, "rb") as fh, np.load(fh) as z:
            return tuple(z[n] for n in names)
    except Exception as exc:
        if isinstance(exc, MemoryError) or (
                isinstance(exc, OSError)
                and exc.errno not in (None, errno.EINVAL)):
            print(f"[{tag}] could not read {path} "
                  f"({type(exc).__name__}: {exc}); left in place", flush=True)
            raise
        bad = path + ".corrupt"
        if os.path.exists(bad):
            bad += f".{time.time_ns()}"
        try:
            os.replace(path, bad)
            where = f"moved aside as {os.path.basename(bad)}"
        except OSError:
            where = "it will be overwritten"
        print(f"[{tag}] {os.path.basename(path)} is damaged "
              f"({type(exc).__name__}); {where}, solving it again",
              flush=True)
        return None


def ch_ground_truth(l, seed, off=0.0, t_end=2000.0, dt_step=DT_STEP,
                    cache_dir: str | None = GT_CACHE, device="cpu",
                    verbose=True):
    """Reference trajectory sampled every ``dt_step``, cached on disk.

    A falsy ``cache_dir`` solves and returns without touching the disk.  One
    l = 256 reference is 1049 MB while the rollout it feeds is 25 MB, so when a
    handful of seeds are being topped up, keeping the reference costs forty
    times what it saves -- and on a quota-limited filesystem it is what stops
    the job.  Re-solving is cheap by comparison: about 47 s per seed on an
    A100, because the integrator is kernel-launch bound rather than compute
    bound at these lattice sizes.
    """
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, gt_cache_name(l, seed, off, t_end))
        if os.path.exists(path):
            got = _load_npz(path, ("times", "frames"), "gt")
            if got is not None:
                return got

    every, n_total, n_frames, times = _gt_grid(t_end, dt_step)
    if verbose:
        print(f"[gt] solving l={l} seed={seed} off={off:+.2f} to t={t_end:g} "
              f"({n_total:,} steps)", flush=True)

    frames = _integrate(initial_condition(l, seed, off), n_total, every,
                        n_frames, device, verbose, t_end)
    if path is not None:
        _savez_atomic(path, times=times, frames=frames,
                      **_provenance(device))
    elif verbose:
        print(f"[gt] not cached (GT_CACHE off); "
              f"{frames.nbytes / 1e6:.0f} MB kept in memory only", flush=True)
    return times, frames


def ch_ground_truth_batch(l, seeds, off=0.0, t_end=2000.0, dt_step=DT_STEP,
                          cache_dir: str | None = GT_CACHE, device="cpu",
                          verbose=True):
    """References for several seeds, the missing ones solved together.

    Returns [(times, frames), ...] in the order of ``seeds``, each bit for
    bit what ch_ground_truth gives that seed: the solver is elementwise plus
    roll, so a seed advances alone or in a batch alike.  A seed whose file is
    there is read; the others are integrated along a leading batch axis and
    each written as a file of its own.  The whole batch is held in memory at
    once -- 1049 MB a seed at l = 256, t = 2000, on the device and again on
    the host -- so the caller sizes the chunks.  A falsy ``cache_dir`` solves
    every seed and writes nothing.
    """
    seeds = [int(s) for s in seeds]
    paths = {s: (os.path.join(cache_dir, gt_cache_name(l, s, off, t_end))
                 if cache_dir else None) for s in seeds}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    got: dict = {}
    todo = []
    for s, path in paths.items():
        if path is not None and os.path.exists(path):
            loaded = _load_npz(path, ("times", "frames"), "gt")
            if loaded is not None:
                got[s] = loaded
                continue
        todo.append(s)

    if todo:
        every, n_total, n_frames, times = _gt_grid(t_end, dt_step)
        if verbose:
            print(f"[gt] solving {len(todo)} of {len(paths)} seeds together: "
                  f"l={l} off={off:+.2f} to t={t_end:g} ({n_total:,} steps) "
                  f"seeds {todo}", flush=True)
        ic = np.stack([initial_condition(l, s, off) for s in todo])
        frames = _integrate(ic, n_total, every, n_frames, device, verbose,
                            t_end)
        info = _provenance(device)
        for b, s in enumerate(todo):
            got[s] = (times.copy(), frames[b])
            path = paths[s]
            if path is not None:
                _savez_atomic(path, times=times, frames=frames[b], **info)
        if verbose and not cache_dir:
            print(f"[gt] not cached (GT_CACHE off); "
                  f"{frames.nbytes / 1e6:.0f} MB kept in memory only",
                  flush=True)
    return [got[s] for s in seeds]


def _is_npy(path):
    """Whether a file is a plain .npy (by suffix or by its magic bytes)."""
    if str(path).lower().endswith(".npy"):
        return True
    try:
        with open(path, "rb") as fh:
            return fh.read(6) == b"\x93NUMPY"
    except OSError:
        return False


def lean_meta(path):
    """The sidecar of a lean dataset written by ch_gpu.py --lean, or None.

    None only for a file that is not an .npy at all.  An .npy without its
    .json sidecar, or with one of another format, is an error said here --
    before anything is read -- rather than an 8 GB load that then fails for
    an unrelated reason.
    """
    if not _is_npy(path):
        return None
    side = os.path.splitext(path)[0] + ".json"
    if not os.path.exists(side):
        raise ValueError(f"{path} is an .npy but its sidecar {side} is "
                         f"missing; copy the .json that ch_gpu.py --lean "
                         f"wrote next to it")
    with open(side) as fh:
        meta = json.load(fh)
    if meta.get("format") != LEAN_FORMAT:
        raise ValueError(f"{side} is not a {LEAN_FORMAT} sidecar "
                         f"(format {meta.get('format')!r})")
    return meta


def frames_for(t_train, meta):
    """How many X frames ch.py would write for horizon ``t_train``.

    Frames are every GAP solver steps and ch.py's X drops the last
    N_AHEAD // GAP of them (they exist only as labels), so a horizon of T
    gives 10 T - 10 frames at dt = 0.01, GAP = 10 -- 2990 at T = 300.
    """
    per_unit = 1.0 / (float(meta["dt"]) * int(meta["gap"]))
    return int(round(float(t_train) * per_unit)) - int(meta["shift"])


def load_lean(path, meta, t_train=None, merge=False, verbose=True):
    """Train and val arrays from a lean dataset, sliced to ``t_train``.

    The file is memory-mapped and only the requested frames are copied, once,
    into a contiguous float32 array: 4.1 GB at T = 1000, 8.2 GB at T = 2000,
    against 10.6 and 21.3 GB for reading ch.py's float64 npz.  With ``merge``
    (TRAIN_ON_VAL) the 25 rows come back as one train array and an empty val,
    so main() does not concatenate a second copy.
    """
    t_end = float(meta["t_end"])
    t = t_end if t_train is None else float(t_train)
    if t > t_end + 1e-9:
        raise ValueError(f"--t-train {t:g} is beyond {path}'s horizon "
                         f"{t_end:g}; regenerate it with ch_gpu.py --lean "
                         f"--end {t:g}")
    if int(meta["shift"]) != STRIDE:
        raise ValueError(f"{path} was written with a label offset of "
                         f"{meta['shift']} frames, not STRIDE={STRIDE}")
    if numpy_dtype() != np.float32:
        raise ValueError(f"{path} is float32; PRECISION={PRECISION} needs "
                         f"ch.py's float64 test.npz")
    per_unit = 1.0 / (float(meta["dt"]) * int(meta["gap"]))
    if abs(t * per_unit - round(t * per_unit)) > 1e-6:
        raise ValueError(f"--t-train {t:g} must be a multiple of the frame "
                         f"spacing {1.0 / per_unit:g}")
    n = frames_for(t, meta)
    need = STRIDE * MAX_UNROLL + 1
    if n < need:
        raise ValueError(f"--t-train {t:g} leaves {n} frames per trajectory; "
                         f"training windows need at least {need}")
    arr = np.load(path, mmap_mode="r")
    n_train = int(meta["n_train"])
    if merge:
        train = np.ascontiguousarray(arr[:, :n])
        # not train[:0]: a view would keep train alive after main() replaces it
        val = np.empty((0,) + train.shape[1:], train.dtype)
    else:
        train = np.ascontiguousarray(arr[:n_train, :n])
        val = np.ascontiguousarray(arr[n_train:, :n])
    del arr
    if verbose:
        print(f"[data] lean dataset {path}: t = 0..{t:g} of {t_end:g}, "
              f"{n} frames per trajectory, "
              f"{(train.nbytes + val.nbytes) / 1e9:.2f} GB float32"
              + (f"; all {train.shape[0]} rows ({n_train} train + "
                 f"{train.shape[0] - n_train} val) as one train set "
                 f"(TRAIN_ON_VAL)" if merge else ""), flush=True)
        print(f"[data] train {train.shape}  val {val.shape}", flush=True)
    return train, val


def load_trajectories(path=DATASET, verbose=True):
    meta = lean_meta(path)
    if meta is not None:
        return load_lean(path, meta, T_TRAIN, merge=TRAIN_ON_VAL,
                         verbose=verbose)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run ch_gpu.py first, which writes "
            f"{fno_paths.data_dir()}/test.npz, or the advisor's ch.py and "
            f"move its test.npz there.")

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
    if T_TRAIN is not None:
        want = int(round(float(T_TRAIN) / (DT * GAP))) - stride
        if train.shape[1] != want:
            raise ValueError(
                f"--t-train {T_TRAIN:g} needs {want} frames per trajectory "
                f"but {path} has {train.shape[1]}; an npz cannot be sliced "
                f"-- use a lean dataset (ch_gpu.py --lean)")

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
        got = _load_npz(path, ("frames",), "off")
        if got is not None:
            frames = got[0]
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
    _savez_atomic(path, frames=out, **_provenance(device))
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
                 zero_dc=False, lowk_fill=False):
        super().__init__()
        self.in_channels, self.out_channels = in_ch, out_ch
        self.modes, self.n_ref = modes, n_ref
        self.mode_scaling, self.zero_dc = mode_scaling, zero_dc
        self.lowk_fill = lowk_fill
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

    def _filled_weight(self):
        """The masked weight with its DC entry [modes, 0] replaced by the
        mean of the trained entries next to it, kx = +-1 at ky = 0 and ky = 1
        at kx = 0.  Only the interpolation looks at it: the new modes between
        DC and its neighbours then start from a trained value instead of
        from 0."""
        w, m = self.effective_weight, self.modes
        fill = (w[:, :, m - 1, 0] + w[:, :, m + 1, 0] + w[:, :, m, 1]) / 3.0
        return w + (1.0 - self.dc_mask) * fill[:, :, None, None, :]

    def _resampled_weight(self, n):
        m_n = self.modes_for(n)
        if m_n == self.modes:
            return self.effective_weight
        fill = bool(self.lowk_fill and self.zero_dc)
        key = (n, self.weight.device, self.weight.dtype, fill)
        if not self.training and key in self._cache:
            return self._cache[key]
        w = self._filled_weight() if fill else self.effective_weight
        w = w.permute(0, 1, 4, 2, 3).reshape(
            1, self.in_channels * self.out_channels * 2,
            2 * self.modes + 1, self.modes + 1)

        w = F.interpolate(w, size=(2 * m_n + 1, m_n + 1), mode="bilinear",
                          align_corners=True)
        w = w.reshape(self.in_channels, self.out_channels, 2,
                      2 * m_n + 1, m_n + 1).permute(0, 1, 3, 4, 2)
        if fill:
            # the DC entry is still masked: the filled value was only there
            # for its neighbours
            keep = torch.ones(1, 1, 2 * m_n + 1, m_n + 1, 1,
                              dtype=w.dtype, device=w.device)
            keep[0, 0, m_n, 0, 0] = 0.0
            w = w * keep
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
    """Spectral (global) path + local stencil path, wired residually.

    With ``cond_dim`` (a dt-conditioned model) the sum of the two paths is
    modulated per channel by the step embedding, x * (1 + g) + b, with a
    zero-initialised Linear(cond_dim, 2 ch): at the start the block does not
    depend on dt."""

    film: nn.Linear | None

    def __init__(self, ch, modes, n_ref, mode_scaling, local_kernel, zero_dc,
                 cond_dim=0, lowk_fill=False):
        super().__init__()
        self.spectral = SpectralConv2d(ch, ch, modes, n_ref, mode_scaling,
                                       zero_dc, lowk_fill)
        self.local = nn.Conv2d(ch, ch, local_kernel, padding=local_kernel // 2,
                               padding_mode="circular")
        self.mix = nn.Conv2d(ch, ch, 1)
        self.film = None
        if cond_dim:
            self.film = nn.Linear(cond_dim, 2 * ch)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x, cond=None):
        h = self.spectral(x) + self.local(x)
        if self.film is not None:
            if cond is None:
                raise ValueError("this block is dt-conditioned: pass cond")
            g, b = self.film(cond).chunk(2, dim=-1)
            h = h * (1.0 + g[:, :, None, None]) + b[:, :, None, None]
        return x + self.mix(F.gelu(h))


class PhysicalKBackbone(nn.Module):
    """Lift -> resolution-consistent FNO blocks -> project."""

    def __init__(self, in_ch, channels, modes, n_layers, n_ref, mode_scaling,
                 local_kernel, zero_dc, cond_dim=0, lowk_fill=False):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Conv2d(in_ch, channels, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 1))
        self.blocks = nn.ModuleList([
            FNOBlock(channels, modes, n_ref, mode_scaling, local_kernel,
                     zero_dc, cond_dim, lowk_fill) for _ in range(n_layers)])

        hidden_conv = nn.Conv2d(channels, 2 * channels, 1)
        out_conv = nn.Conv2d(2 * channels, 1, 1)
        with torch.no_grad():
            out_conv.weight.zero_()
            if out_conv.bias is not None:
                out_conv.bias.zero_()
        self.project = nn.Sequential(hidden_conv, nn.GELU(), out_conv)

    def forward(self, x, cond=None):
        h = self.lift(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.project(h)


def dt_embedding(dt, dim):
    """(B, dim + 1) features of the step size (as in the group protocol): sin and cos of
    log dt at dim / 2 log-spaced frequencies from 1 to 64, then log dt."""
    half = dim // 2
    logdt = torch.log(dt.clamp_min(1e-6))
    freqs = torch.exp(torch.linspace(0.0, math.log(64.0), half,
                                     device=dt.device, dtype=dt.dtype))
    ang = logdt[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang), logdt[:, None]], dim=-1)


class CahnHilliardFNO(nn.Module):
    """psi(t) -> psi(t + DT_STEP); with ``dt_conditioned`` psi(t) -> psi(t +
    dt) for a dt given per call (in time units).

    The conditioned model embeds log dt (dt_embedding), runs it through a
    small MLP to ``cond_dim`` numbers that every block turns into a per-channel
    scale and shift (FiLM, zero-initialised), and multiplies the projection by
    a learned gain exp(log_gain + gain_exp log dt) before the Laplacian head.
    The gain starts at dt (log_gain 0, gain_exp 1): the head's increment is
    that of one unit step, and the step over dt is dt times it.  Mass
    conservation, the zero DC mode and the residual are as for the fixed-step
    model, which this class stays when dt_conditioned is False.

    With ``mu_channel`` the input has one more channel after the mean: the
    chemical potential psi^3 - psi - lap(psi) of psi, periodic, dx = DX.  With
    ``mu_clip`` too it is that of psi clamped to [-mu_clip, mu_clip]: only the
    channel sees the clamp, not psi, the residual or the head.
    """

    cond: nn.Sequential | None
    log_gain: nn.Parameter | None
    gain_exp: nn.Parameter | None

    def __init__(self, channels=CHANNELS, modes=MODES, n_layers=LAYERS,
                 n_ref=64, mode_scaling=MODE_SCALING,
                 local_kernel=LOCAL_KERNEL, zero_dc=ZERO_DC,
                 use_mean_channel=True,
                 hard_mass_conservation=HARD_MASS_CONSERVATION,
                 residual_prediction=RESIDUAL_PREDICTION,
                 laplacian_output=False, dt_conditioned=False,
                 cond_dim=COND_DIM, lowk_fill=False, mu_channel=False,
                 mu_clip: float | None = None):
        super().__init__()
        if laplacian_output and not residual_prediction:
            raise ValueError("laplacian_output needs residual_prediction: "
                             "the head's Laplacian is an increment")
        self.config = dict(channels=channels, modes=modes, n_layers=n_layers,
                           n_ref=n_ref, mode_scaling=mode_scaling,
                           local_kernel=local_kernel, zero_dc=zero_dc,
                           use_mean_channel=use_mean_channel,
                           hard_mass_conservation=hard_mass_conservation,
                           residual_prediction=residual_prediction,
                           laplacian_output=laplacian_output,
                           dt_conditioned=dt_conditioned, cond_dim=cond_dim,
                           lowk_fill=lowk_fill, mu_channel=mu_channel,
                           mu_clip=None if mu_clip is None else float(mu_clip))
        self.dt_conditioned = bool(dt_conditioned)
        self.use_mean_channel = use_mean_channel
        self.mu_channel = bool(mu_channel)
        self.mu_clip = None if mu_clip is None else float(mu_clip)
        self.hard_mass_conservation = hard_mass_conservation
        self.residual_prediction = residual_prediction
        self.laplacian_output = laplacian_output
        in_ch = (2 if use_mean_channel else 1) + int(self.mu_channel)

        self.cond = self.log_gain = self.gain_exp = None
        if self.dt_conditioned:
            emb = 2 * (cond_dim // 2) + 1
            self.cond_in = emb
            self.cond = nn.Sequential(
                nn.Linear(emb, cond_dim), nn.GELU(),
                nn.Linear(cond_dim, cond_dim), nn.GELU())
            self.log_gain = nn.Parameter(torch.tensor(0.0))
            self.gain_exp = nn.Parameter(torch.tensor(1.0))

        self.backbone = PhysicalKBackbone(
            in_ch, channels, modes, n_layers, n_ref, mode_scaling,
            local_kernel, zero_dc, cond_dim if self.dt_conditioned else 0,
            lowk_fill)

    def forward(self, psi, dt=None):
        """``dt``: the step in time units, a tensor (B,) or a number; a
        fixed-step model ignores it, a conditioned one needs it."""
        if psi.dim() == 3:
            psi = psi.unsqueeze(1)
        x = psi
        if self.use_mean_channel:
            x = torch.cat([psi, psi.mean(dim=(-2, -1), keepdim=True)
                           .expand_as(psi)], dim=1)
        if self.mu_channel:
            # the chemical potential of psi, from the solver's own stencil:
            # flat where a bulk phase is at equilibrium
            m = (psi if self.mu_clip is None
                 else psi.clamp(-self.mu_clip, self.mu_clip))
            x = torch.cat([x, m ** 3 - m - _lap_t(m)], dim=1)
        if self.dt_conditioned:
            assert (self.cond is not None and self.log_gain is not None
                    and self.gain_exp is not None)
            if dt is None:
                raise ValueError("this model is dt-conditioned: call it with "
                                 "dt (the step in time units)")
            if not torch.is_tensor(dt):
                dt = torch.tensor(float(dt), device=psi.device)
            dt = dt.to(device=psi.device, dtype=psi.dtype)
            if dt.ndim == 0:
                dt = dt.expand(psi.shape[0])
            cond = self.cond(dt_embedding(dt, self.cond_in - 1))
            out = self.backbone(x, cond)
            gain = torch.exp(self.log_gain + self.gain_exp
                             * torch.log(dt.clamp_min(1e-6)))
            out = out * gain[:, None, None, None]
        else:
            out = self.backbone(x)
        if self.laplacian_output:
            out = _lap_t(out)

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


def step(model, x, dt=None):
    """One model step of size ``dt`` (time units; a number or a tensor (B,)).

    The one way to call a model: a fixed-step model takes its own DT_STEP and
    ignores ``dt``, a dt-conditioned one is given it (and refuses a call
    without).  Works through torch.compile and the like, which pass attribute
    reads on to the module.
    """
    if getattr(model, "dt_conditioned", False):
        if dt is None:
            raise ValueError("a dt-conditioned model needs dt in step()")
        return model(x, dt)
    return model(x)


def build_model(n_ref: int = 64, device: torch.device | str = "cpu",
                dtype: torch.dtype | None = None) -> CahnHilliardFNO:
    """Construct from the CONFIG globals and cast to the working precision."""
    model = CahnHilliardFNO(
        channels=CHANNELS, modes=MODES, n_layers=LAYERS, n_ref=n_ref,
        mode_scaling=MODE_SCALING, local_kernel=LOCAL_KERNEL, zero_dc=ZERO_DC,
        hard_mass_conservation=HARD_MASS_CONSERVATION,
        residual_prediction=RESIDUAL_PREDICTION,
        laplacian_output=LAPLACIAN_OUTPUT, dt_conditioned=DT_CONDITIONED,
        cond_dim=COND_DIM, lowk_fill=LOWK_FILL, mu_channel=MU_CHANNEL,
        mu_clip=MU_CLIP if MU_CHANNEL else None)
    return to_dtype(model, dtype or torch_dtype()).to(device)


def set_lowk_fill(model, on):
    """Switch the low-k weight fill on or off on a model that is built, and
    in its config (a checkpoint saved from it keeps the setting).  The
    resampled weights it cached were made with the other setting, so they
    go."""
    on = bool(on)
    for m in model.modules():
        if isinstance(m, SpectralConv2d):
            m.lowk_fill = on
            m._cache.clear()
    if isinstance(getattr(model, "config", None), dict):
        model.config["lowk_fill"] = on


def lowk_fill_of(model):
    """Whether any spectral layer of the model fills the low-k weights."""
    return any(m.lowk_fill for m in model.modules()
               if isinstance(m, SpectralConv2d))


def set_mu_clip(model, value):
    """Set the clamp of the mu channel's input on a model that is built (None:
    none), and in its config (a checkpoint saved from it keeps the setting).
    A model without a mu channel keeps the number but does not use it."""
    value = None if value is None else float(value)
    model.mu_clip = value
    if isinstance(getattr(model, "config", None), dict):
        model.config["mu_clip"] = value


def mu_clip_of(model):
    """The clamp the model's mu channel puts on psi, or None when it has none
    or has no mu channel."""
    return (getattr(model, "mu_clip", None)
            if getattr(model, "mu_channel", False) else None)


def _tensors_only(state_dict):
    """Keep only tensor entries, so a stray non-tensor cannot break a load."""
    return {k: v for k, v in state_dict.items() if torch.is_tensor(v)}


def run_meta():
    """What a checkpoint must agree on to be resumed by this run.  The
    variable-step kind says so (a fixed-step run's is as it always was)."""
    meta = {"dataset": os.path.basename(str(DATASET)), "t_train": T_TRAIN,
            "val_t_start": VAL_T_START, "val_t_end": VAL_T_END,
            "monitor_t_end": MONITOR_T_END, "tag": TAG}
    if DT_CONDITIONED:
        meta["dt_conditioned"] = True
    return meta


# Only a run with these may resume a checkpoint from before run_meta was
# recorded: such a checkpoint can only be the default fixed-step run.
DEFAULT_RUN_META = {"dataset": os.path.basename(str(DATASET)),
                    "t_train": None, "val_t_start": VAL_T_START,
                    "val_t_end": FIXED_STEP["VAL_T_END"],
                    "monitor_t_end": MONITOR_T_END, "tag": ""}


def _make_parent(path):
    """Make the folder ``path`` is about to be written into."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _replace(src, dst):
    """os.replace, tried again where the target is briefly in use.

    Windows refuses to replace a file that a scanner or a viewer has open,
    usually for a moment.  After REPLACE_RETRIES more tries, REPLACE_WAIT
    seconds apart, the PermissionError is raised and ``src`` is left alone.
    """
    for attempt in range(REPLACE_RETRIES + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_RETRIES:
                raise
            print(f"[save] {dst} is in use; trying again "
                  f"({attempt + 1}/{REPLACE_RETRIES})", flush=True)
            time.sleep(REPLACE_WAIT)


def _atomic(path, write):
    """Make ``path`` by ``write(<path>.partial)``, then rename it into place.

    A job killed or a disk filled in the middle of the write leaves the old
    file as it was.  The new file is flushed to disk before the rename, so
    a power cut cannot leave the final name on a file that was never
    written.  If the rename itself keeps failing, the finished new file
    stays as <path>.partial and the error is raised.
    """
    _make_parent(path)
    part = path + ".partial"
    try:
        write(part)
        with open(part, "r+b") as fh:
            os.fsync(fh.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(part)
        raise
    try:
        _replace(part, path)
    except OSError:
        print(f"[save] could not replace {path}; the new file is left as "
              f"{part}", flush=True)
        raise


def _write_json(path, obj):
    def write(to):
        with open(to, "w") as fh:
            json.dump(obj, fh, indent=1)
    _atomic(path, write)


def save_model(model, path, **extra):
    ckpt = {"config": model.config,
            "state_dict": _tensors_only(model.state_dict()),
            "precision": PRECISION, **extra}
    _atomic(path, lambda to: torch.save(ckpt, to))


def free_name(path):
    """``path``, or ``path`` with _<YYYYmmdd-HHMMSS> before its extension if
    that name is taken (and -2, -3, ... after it if the second is taken
    too), so a rotated file never replaces an earlier one."""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cand, k = f"{root}_{stamp}{ext}", 1
    while os.path.exists(cand):
        k += 1
        cand = f"{root}_{stamp}-{k}{ext}"
    return cand


def _suffixed(path, suffix):
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def _history_path(model_path):
    return os.path.splitext(model_path)[0] + "_history.json"


def _config_path(model_path):
    return os.path.splitext(model_path)[0] + "_config.json"


def _size_text(path):
    n = os.path.getsize(path)
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.1f} KB"


def kept_names(model_out, checkpoint):
    """The names a fresh start gives the best model and the state it moves
    aside, for the messages."""
    return (f"{os.path.basename(_suffixed(model_out, '_prev'))} and "
            f"{os.path.basename(_suffixed(checkpoint, '_stale'))}")


def rotate_old_files(model_out, checkpoint, why):
    """Move aside what an earlier start of this run left, so that this start
    cannot overwrite it.  Nothing is deleted.

    The best model becomes fno_ch_prev.pt, with its history and config as
    fno_ch_prev_history.json and fno_ch_prev_config.json, and a resume state
    that will not be resumed (``why`` says why) becomes fno_ch_last_stale.pt.
    A name that is taken gets a timestamp (free_name).  Returns the pairs
    (old name, new name).
    """
    prev = free_name(_suffixed(model_out, "_prev"))
    moves = [(model_out, prev, "best model"),
             (_history_path(model_out), free_name(_history_path(prev)),
              "its history"),
             (_config_path(model_out), free_name(_config_path(prev)),
              "its config"),
             (checkpoint, free_name(_suffixed(checkpoint, "_stale")),
              f"resume state, which will not be resumed ({why})")]
    done = []
    for old, new, what in moves:
        if not os.path.exists(old):
            continue
        size = _size_text(old)
        _replace(old, new)
        print(f"[keep] {what}: {old} kept as {new} ({size})", flush=True)
        done.append((old, new))
    if done:
        print("[keep] nothing was deleted; this run writes new files under "
              "the old names", flush=True)
    return done


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


# What a model setting was before a checkpoint could record it.
CONFIG_DEFAULTS = {"laplacian_output": False, "dt_conditioned": False,
                   "cond_dim": COND_DIM, "lowk_fill": False,
                   "mu_channel": False, "mu_clip": None}


def model_drift(saved, current):
    """{key: (saved, current)} for the model settings that differ.  A key the
    saved config lacks has the value it had before the key existed."""
    out = {}
    for k, v in current.items():
        if k == "mu_clip":
            # the clamp does not change training (true |psi| is below it), so
            # a run goes on with a checkpoint made before it, or with another
            continue
        was = saved.get(k, CONFIG_DEFAULTS.get(k, "<absent>"))
        if was != v:
            out[k] = (was, v)
    return out


# The flag that gives a model setting back the value a checkpoint was made
# with, for the refusal a resume prints (drift_refusal).
DRIFT_FLAGS = {
    "channels": lambda v: f"--channels {v}",
    "modes": lambda v: f"--modes {v}",
    "laplacian_output": lambda v: f"--variant {'lap' if v else 'base'}",
    "dt_conditioned": lambda v: "" if v else "--fixed-step"}


def drift_refusal(path, drift, kept):
    """What a resume says when the saved model is not the one this run
    builds.  ``drift`` is model_drift's answer, ``kept`` the names
    --no-resume would give the old files.

    A saved size is continued with its own flags plus --keep-folder, which
    leaves the size out of the folder's name: a size token in the name
    (c64m24) is for a new model, and puts the run in a new folder.
    """
    rows = "\n".join(f"  {k}: {was} -> {now}"
                     for k, (was, now) in drift.items())
    fix = [f for f in (DRIFT_FLAGS[k](was) for k, (was, _) in drift.items()
                       if k in DRIFT_FLAGS) if f]
    other = [k for k in drift if k not in DRIFT_FLAGS]
    how = ""
    if "dt_conditioned" in drift:
        how = (f"A fixed-step model and a dt-conditioned one cannot share a "
               f"folder: move the old folder to Archive/ (or use --tag NAME "
               f"for a folder of its own); --fixed-step trains the fixed-step "
               f"kind.  ")
    if fix:
        how += "To go on training it, give the saved model: " + " ".join(fix)
        if "channels" in drift or "modes" in drift:
            how += (" --keep-folder.  --keep-folder leaves the size out of "
                    "the folder's name, so this run finds the state here; "
                    "without it the size goes into the name and the run "
                    "starts in a folder of its own.  ")
        else:
            how += ".  "
    if other:
        how += (f"{', '.join(other)} can only be set back at the top of "
                f"fno_train.py.  ")
    return (f"[resume] {path} was written by a different model "
            f"(checkpoint -> current):\n{rows}\n{how}"
            f"To train the current model here instead, use --no-resume, "
            f"which keeps the old files as {kept}; or --tag NAME for a "
            f"folder of its own.")


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
    amp = ((step(model, x_in + e, DT_STEP) - f_x).flatten(1).norm(dim=1)
           / e.flatten(1).norm(dim=1).clamp_min(1e-12))
    return torch.relu(amp - AMP_TARGET).pow(2).mean()


def ema_decay(n):
    """The decay of the update after ``n`` earlier ones.  It starts low, so
    the average is not mostly the random start for the first thousand steps."""
    return min(EMA_DECAY, (1.0 + n) / (10.0 + n))


def _averaged(v):
    return v.is_floating_point() or v.is_complex()


class EmaWeights:
    """A running average of the model's weights.  It is what gets validated
    and saved as the best model; the model itself keeps training on."""

    def __init__(self, model, n=0):
        self.n = n
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items() if _averaged(v)}

    @torch.no_grad()
    def update(self, model):
        d = ema_decay(self.n)
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
        self.n += 1

    @torch.no_grad()
    def reset(self, model):
        """Start the average again from the model's weights."""
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].copy_(v)
        self.n = 0

    @torch.no_grad()
    def load(self, shadow, n):
        for k, v in self.shadow.items():
            v.copy_(shadow[k])
        self.n = int(n)

    @contextlib.contextmanager
    def swapped(self, model):
        """The model holds the averaged weights inside the block, its own
        again after it."""
        def clear_caches():
            for m in model.modules():
                if isinstance(m, SpectralConv2d):
                    m._cache.clear()
        sd = model.state_dict()
        with torch.no_grad():
            raw = {k: sd[k].detach().clone() for k in self.shadow}
            for k, v in self.shadow.items():
                sd[k].copy_(v)
        clear_caches()
        try:
            yield
        finally:
            with torch.no_grad():
                for k, v in raw.items():
                    sd[k].copy_(v)
            clear_caches()


def averaged(ema, model):
    return (ema.swapped(model) if ema is not None
            else contextlib.nullcontext())


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
            psi = step(model, psi, DT_STEP)
        j = i0 + k + 1
        yield (float(gt_times[j]),
               gt_frames[j].astype(np.float64),
               psi[0, 0].detach().to(torch.float64).cpu().numpy())


def rollout_curve(model, gt_times, gt_frames, device, amp_ctx=None,
                  t_start=VAL_T_START, t_end=VAL_T_END):
    """(times, R2) of the rollout, from ``t_start`` where R2 is 1 by
    construction, one entry per model step to ``t_end``."""
    if float(gt_times[-1]) < t_end - 1e-9:
        raise ValueError(f"the reference ends at t = {gt_times[-1]:g}, before "
                         f"the scored window's end {t_end:g}")
    i0 = int(np.argmin(np.abs(gt_times - t_start)))
    times, r2 = [float(gt_times[i0])], [1.0]
    for t, true, pred in rollout_frames(model, gt_times, gt_frames, t_start,
                                        t_end, device, amp_ctx):
        times.append(t)
        r2.append(r2_score(true, pred))
    return np.array(times), np.array(r2)


def score_from_curve(times, r2, t_end=VAL_T_END, threshold=R2_THRESHOLD):
    """(score, survival) of a rollout curve.  score = mean of max(R2, 0) over
    the steps; survival = the time of the first step below the threshold."""
    scores, survival = [], t_end
    for t, r in zip(times[1:], r2[1:]):
        if r < threshold and survival == t_end:
            survival = float(t)

        scores.append(float(r) if np.isfinite(r) and r > 0.0 else 0.0)
    return (float(np.mean(scores)) if scores else 0.0), survival


def rollout_score(model, gt_times, gt_frames, device, amp_ctx=None,
                  t_start=VAL_T_START, t_end=VAL_T_END,
                  threshold=R2_THRESHOLD):
    """(score, survival) over the fixed window.  score = mean of max(R2, 0)."""
    times, r2 = rollout_curve(model, gt_times, gt_frames, device, amp_ctx,
                              t_start, t_end)
    return score_from_curve(times, r2, t_end, threshold)


def interp_survival(times, r2, threshold, t_end):
    """The time R2 first falls below the threshold, interpolated linearly
    between the step before and the step after.  A curve that never does is
    cut off by the window, and returns ``t_end``; an R2 that is not a number
    fails at its own step."""
    r2 = np.asarray(r2, dtype=np.float64)
    bad = np.nonzero(~(r2 >= threshold))[0]
    if not bad.size:
        return float(t_end)
    i = int(bad[0])
    if i == 0 or not np.isfinite(r2[i]):
        return float(times[i])
    frac = (r2[i - 1] - threshold) / (r2[i - 1] - r2[i])
    return float(times[i - 1] + (times[i] - times[i - 1]) * frac)


def in_range_accuracy(times, r2, t_limit):
    """Mean of R2 clipped to [0, 1] over the steps up to ``t_limit``.  A step
    whose R2 is not a number counts as 0."""
    r2 = np.asarray(r2, dtype=np.float64)[1:]
    keep = np.asarray(times, dtype=np.float64)[1:] <= t_limit + 1e-9
    if not keep.any():
        return 0.0
    clipped = np.clip(np.where(np.isfinite(r2), r2, 0.0), 0.0, 1.0)
    return float(clipped[keep].mean())


def better_epoch(survival, accuracy, best_survival, best_accuracy, delta,
                 peak=None):
    """True when a validation takes over as the best under SELECT = survival:
    it survives longer by more than ``delta``, or within ``delta`` of the
    best and is more accurate where the model is meant to be good.  The
    band is measured from ``peak``, the longest survival any best has had,
    so a run of slightly worse but more accurate epochs cannot walk the
    best down ``delta`` at a time."""
    if survival > best_survival + delta:
        return True
    top = best_survival if peak is None else max(peak, best_survival)
    return survival >= top - delta and accuracy > best_accuracy


def train_horizon(n_frames):
    """T_TRAIN, or else what the training frames cover."""
    return T_TRAIN if T_TRAIN is not None else (n_frames + STRIDE) * DT * GAP


def lowk_masks(l, ratio):
    """Boolean l x l masks of four shells of the wavenumber index |n| = k l /
    2 pi.  ``ratio`` is this lattice over the training one, so the training
    box's fundamental is at |n| = ratio.  below: the modes longer than the
    training box, which it never saw; first: up to half as far again as its
    fundamental; mid: to four times it; high: the rest.  n = 0 is in none."""
    n = np.fft.fftfreq(l) * l
    r = np.hypot(n[:, None], n[None, :])
    return {"below": (r > 0.0) & (r < ratio),
            "first": (r >= ratio) & (r < 1.5 * ratio),
            "mid": (r >= 1.5 * ratio) & (r <= 4.0 * ratio),
            "high": r > 4.0 * ratio}


LOWK_SHELLS = ("below", "first", "mid", "high")


def lowk_times(times, dt=DT_STEP):
    """The start times of the gain: LOWK_T0 and the last frame that has one
    ``dt`` later, leaving out those the references do not hold."""
    t = np.asarray(times, dtype=np.float64)
    last = float(t[-1]) - dt
    out = []
    for want in dict.fromkeys((LOWK_T0, last)):
        i = int(np.argmin(np.abs(t - want)))
        if abs(t[i] - want) < 1e-6 and want <= last + 1e-6:
            out.append(float(t[i]))
    return out


@torch.no_grad()
def lowk_gains(model, gts, t0s, ratio, device, amp_ctx=None, dt=DT_STEP):
    """{t0: {shell: g}} with g = sum Re(conj(dP) dT) / sum |dT|^2 over the
    modes of a shell and over the seeds, one step from the true state.

    dP is what the model's step of ``dt`` adds to psi(t0), dT what the
    solver's does, both in float64.  g = 1: the increment has the true size in that shell;
    0: the model leaves it alone.  A shell with no modes (the lattice is no
    longer than the training one) or none of the true increment gives None.
    """
    model.eval()
    ctx = amp_ctx if amp_ctx is not None else contextlib.nullcontext
    dtype = next(model.parameters()).dtype
    out = {}
    for t0 in t0s:
        a, b = [], []
        for times, frames in gts:
            i0 = int(np.argmin(np.abs(times - t0)))
            i1 = int(np.argmin(np.abs(times - (t0 + dt))))
            a.append(np.asarray(frames[i0], dtype=np.float64))
            b.append(np.asarray(frames[i1], dtype=np.float64))
        x = torch.as_tensor(np.stack(a)).to(device=device, dtype=dtype)
        x = x[:, None]
        with ctx():
            pred = step(model, x, dt)
        base = x[:, 0].to("cpu", torch.float64)
        dp = pred[:, 0].to("cpu", torch.float64) - base
        d_true = torch.as_tensor(np.stack(b)) - base
        fp, ft = torch.fft.fft2(dp), torch.fft.fft2(d_true)
        num = (fp.conj() * ft).real.sum(dim=0).numpy()
        den = (ft.abs() ** 2).sum(dim=0).numpy()
        gain = {}
        for name, mask in lowk_masks(base.shape[-1], ratio).items():
            d = float(den[mask].sum())
            gain[name] = float(num[mask].sum()) / d if d > 0.0 else None
        out[float(t0)] = gain
    return out


def lowk_text(gains):
    """'0.80/1.00/1.01/0.99@1000': below/first/mid/high at each start."""
    return " ".join("/".join("-" if g[k] is None else f"{g[k]:.2f}"
                             for k in LOWK_SHELLS) + f"@{t0:g}"
                    for t0, g in gains.items())


def holdout_monitor(gts, device, dtype, n_windows, batch_size,
                    t_start=VAL_T_START):
    """Fixed one-step (psi, target) batches from the held-out VAL_SEEDS runs.

    When TRAIN_ON_VAL folds val_X into the training set, the dataset no longer
    contains anything the model has not seen, and a one-step loss measured on it
    would be a training number wearing a validation label.  These trajectories
    are integrated fresh from VAL_SEEDS at VAL_LATTICE and are in no part of
    test.npz, so the metric stays honest.  Frames before ``t_start`` are skipped
    so the monitor covers the same regime the rollout is graded on.

    Consecutive frames here are one model step apart, unlike test.npz where a
    step spans STRIDE frames, so pairs are taken adjacently.
    """
    rng = np.random.default_rng(0)
    index = [(si, i)
             for si, (times, frames) in enumerate(gts)
             for i in range(int(np.argmin(np.abs(times - t_start))),
                            len(frames) - 1)]
    if not index:
        raise ValueError(f"no frames past t_start={t_start} to monitor with")
    pick = rng.choice(len(index), size=min(n_windows, len(index)),
                      replace=False)
    x = np.stack([gts[index[j][0]][1][index[j][1]] for j in pick])
    y = np.stack([gts[index[j][0]][1][index[j][1] + 1] for j in pick])
    xt = torch.as_tensor(x, dtype=dtype).unsqueeze(1)
    yt = torch.as_tensor(y, dtype=dtype).unsqueeze(1)
    return [(xt[i:i + batch_size].to(device), yt[i:i + batch_size].to(device))
            for i in range(0, len(xt), batch_size)]


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


class ChainSampler:
    """Chains (psi(t0), psi(t0 + dt), ..., psi(t0 + n dt)) from the training
    trajectories, as in the group protocol.

    Every sample has a stride of its own, a whole number df of frames drawn
    log-uniformly between DT_MIN and DT_MAX (dt = df * FRAME_DT, 1.0 .. 50.0
    in steps of 0.1), capped so that a chain of n steps fits the trajectory;
    all its steps use that stride.  The chain starts at a frame uniform from
    T_MIN on.
    """

    def __init__(self, frames, device, data_device, dtype, seed=0):
        self.data = torch.from_numpy(frames).to(data_device)
        self.device, self.dtype = device, dtype
        self.rng = np.random.default_rng(seed)
        self.n_traj, self.n_frames = self.data.shape[0], self.data.shape[1]
        self.f_min = max(0, int(round((T_MIN - DT) / FRAME_DT)))
        self.df_min = max(1, int(round(DT_MIN / FRAME_DT)))
        self.df_max = int(round(DT_MAX / FRAME_DT))
        longest = max(n for _, n in VDT_UNROLL_STAGES)
        if self.n_frames - 1 - self.f_min < self.df_min * longest:
            raise ValueError(
                f"the trajectories have {self.n_frames} frames and chains "
                f"start at frame {self.f_min} (T_MIN = {T_MIN:g}): no room "
                f"for {longest} steps of {DT_MIN:g}; train on a longer "
                f"horizon or lower T_MIN")

    def df_hi(self, n_steps):
        return max(self.df_min, min(self.df_max, (self.n_frames - 1
                                                  - self.f_min) // n_steps))

    def max_dt(self, n_steps):
        """The largest dt a chain of n_steps fits into a trajectory."""
        return self.df_hi(n_steps) * FRAME_DT

    def sample(self, batch, n_steps=1, dt=None):
        """(chain, dt): a list of n_steps + 1 tensors (batch, 1, l, l) and the
        stride of each sample, a tensor (batch,) in time units.  ``dt`` fixes
        the stride (rounded to the frame spacing and capped) for a check."""
        df_hi = self.df_hi(n_steps)
        if dt is None:
            lo, hi = math.log(self.df_min), math.log(df_hi)
            df = np.clip(np.rint(np.exp(self.rng.uniform(lo, hi, batch))),
                         self.df_min, df_hi).astype(np.int64)
        else:
            df = np.minimum(np.full(batch, max(1, int(round(dt / FRAME_DT))),
                                    dtype=np.int64), df_hi)
        last = self.n_frames - 1 - df * n_steps
        traj = self.rng.integers(0, self.n_traj, batch)
        f0 = self.rng.integers(self.f_min, np.maximum(last, self.f_min + 1))
        src = self.data.device
        traj_i = torch.as_tensor(traj, device=src)
        chain = []
        for k in range(n_steps + 1):
            idx = torch.as_tensor(f0 + k * df, device=src)
            x = self.data[traj_i, idx]
            chain.append(x.to(self.device, dtype=self.dtype).unsqueeze(1))
        dt_t = torch.as_tensor(df * FRAME_DT, dtype=self.dtype,
                               device=self.device)
        return chain, dt_t

    def augment(self, chain, gen):
        """The lattice symmetries (a rotation, a flip), and psi -> -psi: the
        quenches of the data are symmetric about zero."""
        k = int(torch.randint(0, 4, (1,), generator=gen, device=self.device))
        if k:
            chain = [torch.rot90(c, k, dims=(-2, -1)) for c in chain]
        if torch.rand((), generator=gen, device=self.device) < 0.5:
            chain = [torch.flip(c, dims=(-1,)) for c in chain]
        if torch.rand((), generator=gen, device=self.device) < 0.5:
            chain = [-c for c in chain]
        return chain


def _sq(x):
    return (x ** 2).sum(dim=(1, 2, 3))


def r2_loss(pred, true):
    """||pred - true||^2 / ||true - mean(true)||^2, the 1 - R2 of a sample."""
    den = _sq(true - true.mean(dim=(1, 2, 3), keepdim=True))
    return (_sq(pred - true) / den.clamp_min(1e-12)).mean()


def increment_loss(pred, true, inp):
    """The error against the size of the true change over the step (a model
    that returns its input scores 1, however small the step is)."""
    return (_sq(pred - true) / _sq(true - inp).clamp_min(1e-12)).mean()


def _fwd_diff(z):
    return torch.roll(z, -1, -2) - z, torch.roll(z, -1, -1) - z


def gradient_loss(pred, true):
    px, py = _fwd_diff(pred)
    tx, ty = _fwd_diff(true)
    return ((_sq(px - tx) + _sq(py - ty))
            / (_sq(tx) + _sq(ty)).clamp_min(1e-12)).mean()


def add_noise(psi, sigma):
    """Zero-mean noise of standard deviation ``sigma`` on the chain input."""
    if sigma <= 0:
        return psi
    n = torch.randn_like(psi) * sigma
    return psi + (n - n.mean(dim=(-2, -1), keepdim=True))


def vdt_unroll(epoch, epochs=None):
    """(chain length, stage) of the 0-based ``epoch``: a stage starts at
    epoch >= 1 + int(fraction * epochs) counting from 1."""
    total = EPOCHS if epochs is None else epochs
    stage, unroll = 0, VDT_UNROLL_STAGES[0][1]
    for i, (frac, n) in enumerate(VDT_UNROLL_STAGES):
        if epoch + 1 >= 1 + int(frac * total):
            stage, unroll = i, n
    return unroll, stage


def vdt_lr(epoch, epochs=None):
    """The learning rate of the 0-based ``epoch``: a cosine from LR to
    VDT_LR_MIN over the run, times VDT_STAGE_LR_MULT per stage."""
    total = EPOCHS if epochs is None else epochs
    _, stage = vdt_unroll(epoch, total)
    frac = min(epoch / max(1, total - 1), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * frac))
    return (VDT_LR_MIN + (LR - VDT_LR_MIN) * cos) * VDT_STAGE_LR_MULT ** stage


def vdt_epoch(model, step_model, sampler, opt, ema, gen, unroll, device,
              amp_ctx):
    """One epoch of the group protocol's training: STEPS_PER_EPOCH batches of chains, the
    gradient through every one of the (at most VDT_GRAD_STEPS last) steps,
    each scored by the R2 loss + W_INC * increment loss + W_GRAD * gradient
    loss; the mean over the steps is minimised.  A step whose loss or
    gradient is not finite is skipped.

    Returns (summed loss, steps taken, steps skipped, summed d/psi).
    """
    bs = max(2, BATCH_SIZE // max(1, min(unroll, VDT_GRAD_STEPS)))
    run_loss = run_dpsi = 0.0
    n_seen = n_bad = 0
    for _ in range(STEPS_PER_EPOCH):
        chain, dt_t = sampler.sample(bs, unroll)
        chain = sampler.augment(chain, gen)
        sigma = VDT_NOISE * float(torch.rand((), generator=gen,
                                             device=device))
        psi = add_noise(chain[0], sigma)
        n_nograd = max(0, unroll - VDT_GRAD_STEPS)
        if n_nograd:
            with torch.no_grad(), amp_ctx():
                for _ in range(n_nograd):
                    psi = step(model, psi, dt_t)
            psi = psi.detach()
        terms, dpsi = [], 0.0
        with amp_ctx():
            for k in range(n_nograd, unroll):
                psi_in = psi
                pred, true = step(step_model, psi, dt_t), chain[k + 1]
                term = (r2_loss(pred, true)
                        + W_INC * increment_loss(pred, true, psi.detach())
                        + W_GRAD * gradient_loss(pred, true))
                if not HARD_MASS_CONSERVATION and MASS_WEIGHT > 0:
                    drift = (pred.mean(dim=(-2, -1))
                             - true.mean(dim=(-2, -1))) ** 2
                    term = term + MASS_WEIGHT * drift.mean()
                terms.append(term)
                if k == n_nograd:
                    with torch.no_grad():
                        ref = psi_in - psi_in.mean(dim=(1, 2, 3),
                                                   keepdim=True)
                        dpsi = float((_sq(pred - psi_in)
                                      / _sq(ref).clamp_min(1e-12)).sqrt()
                                     .mean())
                psi = pred
            loss = torch.stack(terms).mean()
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            n_bad += 1
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not torch.isfinite(gn):
            n_bad += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        if ema is not None:
            ema.update(model)
        run_loss += float(loss.detach())
        run_dpsi += dpsi
        n_seen += 1
    return run_loss, n_seen, n_bad, run_dpsi


class ValRefs:
    """The validation seeds' references: the memory-mapped files, the
    schedule of the rollout, and the reference frames at its times.

    The schedule is make_schedule(VAL_T_START, VAL_T_END, VAL_ALPHA,
    VAL_DT_MIN, VAL_DT_MAX, quantum = the reference spacing), so every one of
    its times is a reference frame.  ``truth`` is (times, seeds, l, l).
    """

    def __init__(self, paths):
        self.paths = list(paths)
        self.gts = [fno_gt.load_gt(p, VAL_T_END) for p in self.paths]
        grid = np.asarray(self.gts[0][0])
        self.spacing = float(grid[1] - grid[0])
        self.times = fno_gt.make_schedule(VAL_T_START, VAL_T_END, VAL_ALPHA,
                                          VAL_DT_MIN, VAL_DT_MAX,
                                          quantum=self.spacing)
        idx = [int(np.argmin(np.abs(grid - x))) for x in self.times]
        bad = [x for i, x in zip(idx, self.times)
               if abs(grid[i] - x) > 1e-6]
        if bad:
            raise ValueError(f"schedule times {bad[:3]} are not frames of "
                             f"{self.paths[0]}")
        self.truth = np.stack([np.asarray(f[idx], dtype=np.float32)
                               for _, f in self.gts], axis=1)


def open_val_refs(device, say=print):
    """ValRefs of VAL_SEEDS at VAL_LATTICE, critical quench, from REF_CACHE;
    the missing files are solved first, together, on ``device``."""
    args = (VAL_LATTICE, 0.0)
    span = (VAL_T_START, VAL_T_END, fno_gt.GT_EVERY)
    fno_gt.ground_truth_batch(REF_CACHE, *args, VAL_SEEDS, *span, DT, device)
    paths = [fno_gt.find_gt(REF_CACHE, *args, int(s), *span)
             for s in VAL_SEEDS]
    missing = [s for s, p in zip(VAL_SEEDS, paths) if p is None]
    if missing:
        raise FileNotFoundError(f"no reference for seeds {missing} in "
                                f"{REF_CACHE}")
    say(f"[val] l={VAL_LATTICE} references to t={VAL_T_END:g} for seeds "
        f"{list(VAL_SEEDS)} in {REF_CACHE}")
    refs = ValRefs([p for p in paths if p is not None])
    say(f"[val] schedule: alpha {VAL_ALPHA:g}, dt {VAL_DT_MIN:g}..{VAL_DT_MAX:g}"
        f" in steps of {refs.spacing:g}: {len(refs.times) - 1} steps from "
        f"t = {refs.times[0]:g} to {refs.times[-1]:g}")
    return refs


def alive_mask(psi, limit):
    """Per-sample bool (b,): finite everywhere and max |psi| <= limit."""
    flat = psi.detach().flatten(1)
    return torch.isfinite(flat).all(dim=1) & (flat.abs().amax(dim=1) <= limit)


@torch.no_grad()
def vdt_rollout(model, refs, device, amp_ctx=None, limit=None, horizon=None):
    """The schedule rollout of the validation seeds, all in one batch.

    Each seed starts from its reference at VAL_T_START and takes the
    schedule's steps with their own dt.  A seed diverges when its field goes
    non-finite or max |psi| exceeds ``limit`` (VAL_DIVERGE_ABS): it is frozen
    at its last good state (so it cannot feed NaN to the model), is scored
    R2 = -1 from that step on and fails there; the others go on.

    Returns a dict: times, r2 (times, seeds) with the -1 filled in, survival
    (per seed, interpolated first R2 < R2_THRESHOLD, censored at VAL_T_END),
    score (mean R2 over everything), accuracy (mean of R2 clipped to [0, 1]
    over the times up to min(2 horizon, VAL_T_END); ``horizon`` None means
    the whole window).
    """
    model.eval()
    limit = VAL_DIVERGE_ABS if limit is None else limit
    ctx = amp_ctx if amp_ctx is not None else contextlib.nullcontext
    dtype = next(model.parameters()).dtype
    times, truth = refs.times, refs.truth
    psi = torch.as_tensor(truth[0], dtype=dtype, device=device).unsqueeze(1)
    b = psi.shape[0]
    r2 = np.ones((len(times), b))
    alive = torch.ones(b, dtype=torch.bool, device=device)
    div_k = np.full(b, len(times))
    for i in range(1, len(times)):
        dt = torch.full((b,), float(times[i] - times[i - 1]), device=device,
                        dtype=dtype)
        with ctx():
            new = step(model, psi, dt)
        newly = alive & ~alive_mask(new, limit)
        if bool(newly.any()):
            div_k[newly.cpu().numpy()] = i
            alive = alive & ~newly
            if not bool(alive.any()):
                break
        psi = torch.where(alive.view(-1, 1, 1, 1), new, psi)
        tr = torch.as_tensor(truth[i], dtype=dtype, device=device).unsqueeze(1)
        num = _sq(psi - tr)
        den = _sq(tr - tr.mean(dim=(1, 2, 3), keepdim=True))
        r2[i] = (1 - num / den).double().cpu().numpy()

    failing = r2.copy()
    for k in range(b):
        r2[div_k[k]:, k] = -1.0
        failing[div_k[k]:, k] = np.nan
    survival = [interp_survival(times, failing[:, k], R2_THRESHOLD, VAL_T_END)
                for k in range(b)]
    t_in = VAL_T_END if horizon is None else min(2.0 * horizon, VAL_T_END)
    in_range = np.asarray(times) <= t_in + 1e-9
    kept = np.clip(r2[in_range], 0.0, 1.0)
    # no time in range (a horizon of 20 against a window that starts at 100,
    # as --smoke has): 0, as in_range_accuracy says
    return {"times": times, "r2": r2, "survival": survival,
            "score": float(r2.mean()),
            "accuracy": float(kept.mean()) if kept.size else 0.0}


def val1_batches(refs, device, dtype, n=None):
    """Fixed one-step (dt, psi, target) checks from the held-out references:
    VAL1_WINDOWS random frames of random seeds per stride of VAL1_DTS (those
    on the reference grid and inside the window)."""
    n = VAL1_WINDOWS if n is None else n
    rng = np.random.default_rng(0)
    n_frames = len(refs.gts[0][0])
    out = []
    for dt in VAL1_DTS:
        k = int(round(dt / refs.spacing))
        if k < 1 or abs(k * refs.spacing - dt) > 1e-6 or k > n_frames - 1:
            continue
        seed = rng.integers(0, len(refs.gts), n)
        f0 = rng.integers(0, n_frames - k, n)
        x = np.stack([np.asarray(refs.gts[s][1][f], dtype=np.float32)
                      for s, f in zip(seed, f0)])
        y = np.stack([np.asarray(refs.gts[s][1][f + k], dtype=np.float32)
                      for s, f in zip(seed, f0)])
        out.append((float(dt),
                    torch.as_tensor(x, dtype=dtype, device=device)[:, None],
                    torch.as_tensor(y, dtype=dtype, device=device)[:, None]))
    return out


@torch.no_grad()
def val1_loss(model, batches, amp_ctx=None):
    """Mean 1 - R2 of one step, over the strides of val1_batches."""
    model.eval()
    ctx = amp_ctx if amp_ctx is not None else contextlib.nullcontext
    vals = []
    for dt, x, y in batches:
        with ctx():
            pred = step(model, x, torch.full((x.shape[0],), dt,
                                             device=x.device, dtype=x.dtype))
        vals.append(float(r2_loss(pred, y)))
    return float(np.mean(vals)) if vals else float("nan")


def try_compile(model, lattices, batch_size, device, amp_ctx):
    """torch.compile with a fallback that actually fires."""
    try:
        compiled = torch.compile(model, dynamic=False)
        dtype = next(model.parameters()).dtype
        for n in dict.fromkeys(lattices):
            with amp_ctx():
                step(compiled, torch.zeros(batch_size, 1, n, n, device=device,
                                           dtype=dtype),
                     torch.ones(batch_size, device=device,
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


def config_constants():
    """(name, value) of every plain upper-case constant of this file."""
    skip = {"TypeVar"}
    plain = (int, float, str, bool, tuple, list, type(None))
    return [(k, v) for k, v in sorted(globals().items())
            if k.isupper() and k not in skip and isinstance(v, plain)]


def schedule_stamp():
    """The settings that shape the run's schedule, which a resume must keep.

    The stamp is stored in the resume state.  Left out on purpose, as they
    can change between starts without harm: PATIENCE, MIN_DELTA, the time
    budget, logging and VAL_EVERY.
    """
    names = ["EPOCHS", "STEPS_PER_EPOCH", "LR", "LR_FLOOR", "LR_STAGE_DECAY",
             "BATCH_SIZE", "MAX_UNROLL", "BPTT", "WARM_FRAC", "RAMP_FRAC",
             "WEIGHT_DECAY", "GRAD_CLIP", "H1_WEIGHT", "MASS_WEIGHT",
             "CONTRACTIVITY_WEIGHT", "AMP_TARGET", "AMP_EPS", "SMOOTH_WEIGHT",
             "NOISE_STD", "NOISE_PROB", "TRAIN_ON_VAL", "OFF_CRITICAL",
             "SEED", "PRECISION", "EMA", "SELECT", "AMP", "VAL_SEEDS",
             "VAL_LATTICE", "R2_THRESHOLD", "SURVIVAL_MIN_DELTA", "PSI_MAX",
             "LOSS_SPIKE_FACTOR", "DIVERGE_FACTOR", "MAX_RECOVERIES"]
    if DT_CONDITIONED:
        # the fixed-step loop's own settings do not apply
        names = [n for n in names if n not in (
            "LR_FLOOR", "LR_STAGE_DECAY", "MAX_UNROLL", "BPTT", "WARM_FRAC",
            "RAMP_FRAC", "H1_WEIGHT", "CONTRACTIVITY_WEIGHT", "AMP_TARGET",
            "AMP_EPS", "SMOOTH_WEIGHT", "NOISE_STD", "NOISE_PROB",
            "LOSS_SPIKE_FACTOR")]
        names += ["DT_CONDITIONED", "COND_DIM", "DT_MIN", "DT_MAX", "T_MIN",
                  "VDT_UNROLL_STAGES", "VDT_GRAD_STEPS", "VDT_LR_MIN",
                  "VDT_STAGE_LR_MULT", "VDT_BETAS", "VDT_NOISE", "W_INC",
                  "W_GRAD", "VAL_T_START", "VAL_T_END", "VAL_ALPHA",
                  "VAL_DT_MIN", "VAL_DT_MAX", "VAL_QUANTUM",
                  "VAL_DIVERGE_ABS"]
    if OFF_CRITICAL:
        names += ["OFF_CRITICAL_OFFS", "OFF_CRITICAL_TRAJ",
                  "OFF_CRITICAL_SEED0"]
    if EMA:
        names += ["EMA_DECAY"]
    return {n: globals()[n] for n in names}


def schedule_drift(saved, current):
    """{name: (saved, current)} for the settings of ``current`` that the
    saved stamp has and gives another value.  A setting the saved stamp does
    not have is not compared: it is from before the stamp knew it."""
    def plain(v):
        return list(v) if isinstance(v, tuple) else v
    return {k: (saved[k], v) for k, v in current.items()
            if k in saved and plain(saved[k]) != plain(v)}


def read_launches(path):
    """The launches an earlier start wrote to the config file ``path``: []
    where there is no file.

    A file that is not JSON, or has no list "launches" (such as one from
    before the list), is moved aside as fno_ch_config_old.json, with a
    timestamp if that name is taken, and [] is returned.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            old = json.load(fh)
    except ValueError:
        old = None
    launches = old.get("launches") if isinstance(old, dict) else None
    if isinstance(launches, list) and all(isinstance(x, dict)
                                          for x in launches):
        return launches
    kept = free_name(_suffixed(path, "_old"))
    _replace(path, kept)
    print(f"[keep] {path} holds no list of launches that can be read; kept "
          f"as {kept}, and a new list is started", flush=True)
    return []


def train_config(label, argv, resumed_from=None, launches=()):
    """How this run was set up: written beside the model and put in both
    checkpoints, so a model says how it was trained.

    ``launches`` are the earlier starts of this run, as read_launches gives
    them; this start is added to the end.  ``resumed_from`` is the epoch it
    took up from, and None for a fresh start.
    """
    when = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    launch = {"time": when, "host": socket.gethostname(),
              "argv": list(argv), "resumed_from": resumed_from}
    return {"label": label, "seed": SEED,
            "size": {"channels": CHANNELS, "modes": MODES},
            "run_meta": run_meta(), "schedule": schedule_stamp(),
            "constants": dict(config_constants()),
            "launches": [*launches, launch]}


def require_gpu(device, allow_cpu=False):
    """Refuse to run on the CPU inside a SLURM job.

    A job that asked for a GPU and did not get one would otherwise spend its
    whole time limit crawling.  --allow-cpu, or FNO_ALLOW_CPU=1 in the
    environment, says the CPU is what is wanted.
    """
    job = os.environ.get("SLURM_JOB_ID")
    if (not job or allow_cpu or torch.device(device).type == "cuda"
            or os.environ.get("FNO_ALLOW_CPU") == "1"):
        return
    raise SystemExit(f"SLURM job {job} has no CUDA device, so this would run "
                     f"on the CPU, far slower than the job expects.  Check "
                     f"the --gres line and the CUDA module; if the CPU is "
                     f"what you want, pass --allow-cpu or set "
                     f"FNO_ALLOW_CPU=1.")


def print_config(model=None, extra=None):
    """Dump every CONFIG constant, for the record in the job log."""
    rows = config_constants()
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


def diverge_reason(train_loss, delta_rel, skipped, median, blown, stalled,
                   unphysical):
    """Why the health check tripped, in the order the criteria are trusted.

    `median` is None on the first epoch of a run or a resume, so the relative
    criterion cannot be the reason there and must not be formatted.
    """
    if blown:
        return f"train loss {train_loss} is not finite"
    if unphysical:
        return f"d/psi {delta_rel:.4f} > {MAX_DELTA_REL}"
    if stalled:
        return f"{skipped}/{STEPS_PER_EPOCH} batches skipped"
    return f"train {train_loss:.5f} vs median {median:.5f}"


def slurm_deadline():
    """When the scheduler will kill this job, as a wall-clock epoch time.

    SLURM_JOB_END_TIME is set by recent versions and costs nothing to read;
    where it is missing, scontrol knows.  Returns None when neither answers,
    which is the case off the cluster and means there is no deadline at all.
    """
    raw = os.environ.get("SLURM_JOB_END_TIME")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        out = subprocess.run(["scontrol", "-o", "show", "job", job],
                             capture_output=True, text=True,
                             timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"EndTime=(\S+)", out or "")
    if not m or m.group(1) in ("Unknown", "None", "N/A"):
        return None
    try:
        return time.mktime(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


def resolve_budget(t_launch):
    """(deadline in epoch seconds or None, one line saying how it was decided).

    The old constant had to be kept in step with --time by hand, and drifted:
    ftp.sh asks for 8 h while MAX_HOURS said 18.5, so that job could only ever
    end in a SIGKILL.  Asking the scheduler removes the chance to get it wrong,
    and off the scheduler there is nothing to respect, so the run stops when
    EPOCHS or PATIENCE says to.
    """
    if isinstance(MAX_HOURS, str):
        if MAX_HOURS != "auto":
            raise ValueError('MAX_HOURS must be a number, None, or "auto", '
                             f"not {MAX_HOURS!r}")
        end = slurm_deadline()
        if end is None:
            return None, ("not under a scheduler; running until EPOCHS or "
                          "PATIENCE stops it")
        job = os.environ.get("SLURM_JOB_ID", "?")
        return end, (f"SLURM job {job} ends in "
                     f"{(end - time.time()) / 3600.0:.2f} h")
    if not MAX_HOURS:
        return None, "no limit set"
    return (t_launch + MAX_HOURS * 3600.0,
            f"MAX_HOURS = {MAX_HOURS} h from launch")


def _count(text):
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError(f"{text} is not a count above 0")
    return n


def _weight(text):
    w = float(text)
    if not w >= 0:
        raise argparse.ArgumentTypeError(f"{text} is not a weight of 0 or "
                                         f"more")
    return w


def _seed(text):
    n = int(text)
    if n < 0:
        raise argparse.ArgumentTypeError(f"{text} is below 0")
    return n


def max_modes(lattice):
    """The most Fourier modes a box of ``lattice`` points can hold."""
    return max(1, (int(lattice) - 1) // 2)


def check_modes(lattice):
    """Refuse --modes that the training boxes cannot hold."""
    if MODES > max_modes(lattice):
        raise SystemExit(f"--modes {MODES}: the training boxes are "
                         f"{lattice} wide and hold at most "
                         f"{max_modes(lattice)} modes")


def parse_args(argv=None):
    """One-off overrides, so a run needs no edit to this file.

    Everything defaults to None and leaves the module config alone, so
    running with no flags trains exactly as before.
    """
    ap = argparse.ArgumentParser(description="Train the Cahn-Hilliard FNO.")
    ap.add_argument("--dataset", metavar="PATH",
                    help=f"test.npz from ch.py, or a lean .npy from ch_gpu.py "
                         f"--lean (file: {DATASET})")
    ap.add_argument("--t-train", type=float, metavar="T",
                    help="train on t = 0..T; a lean dataset is sliced to it")
    ap.add_argument("--tag", metavar="NAME",
                    help="name the model's folder train_t<T>_NAME under "
                         "Work/models (letters, digits, '.', '+' and '-' "
                         "only; not c<C>m<M> or s<N>, which --channels, "
                         "--modes and --seed add before it); --t-train "
                         "alone records the tag T<value>, e.g. T1000, but "
                         "adds nothing to the name")
    ap.add_argument("--val-t-start", type=float, metavar="T",
                    help=f"validation rollouts start here (file: "
                         f"{VAL_T_START:g})")
    ap.add_argument("--val-t-end", type=float, metavar="T",
                    help=f"and are scored to here (file: {VAL_T_END:g})")
    ap.add_argument("--monitor-t-end", type=float, metavar="T",
                    help=f"horizon of the one-step monitor trajectories "
                         f"(file: {MONITOR_T_END:g})")
    ap.add_argument("--epochs", type=int, metavar="N",
                    help=f"(file: {EPOCHS})")
    ap.add_argument("--no-resume", action="store_true",
                    help="start fresh even if this run has a resume state; "
                         "the old files are kept as fno_ch_prev.pt and "
                         "fno_ch_last_stale.pt, never deleted")
    ap.add_argument("--accept-schedule", action="store_true",
                    help="resume a state made with another training "
                         "schedule anyway: the differences are printed and "
                         "noted in the history, and the new schedule goes "
                         "into the state (refused otherwise)")
    ap.add_argument("--variant", choices=("lap", "base"),
                    help="lap: the head predicts M and the step is its "
                         "Laplacian, which conserves mass exactly (the "
                         "folder gets _lap); base: the plain head, which "
                         "the no-hard-constraints ablation takes by itself "
                         "unless lap is asked for, and then refuses (file: "
                         f"{'lap' if LAPLACIAN_OUTPUT else 'base'})")
    ap.add_argument("--seed", type=_seed, metavar="N",
                    help=f"training seed (file: {SEED}); a seed other than "
                         f"0 puts s<N> in the folder's name, after any "
                         f"size token and before --tag")
    ap.add_argument("--channels", type=_count, metavar="C",
                    help=f"model width (file: {CHANNELS}); another width "
                         f"than the file's puts c<C>m<M> first in the "
                         f"folder's name")
    ap.add_argument("--modes", type=_count, metavar="M",
                    help=f"Fourier modes (file: {MODES}); at most "
                         f"(lattice - 1) // 2 of the training boxes; "
                         f"another count than the file's puts c<C>m<M> "
                         f"first in the folder's name")
    ap.add_argument("--keep-folder", action="store_true",
                    help="leave the c<C>m<M> token out of the folder's "
                         "name, to go on training a model of another size "
                         "in the folder it already has (with its own "
                         "flags: --channels 64 --keep-folder)")
    ap.add_argument("--ema", action="store_true",
                    help="validate and save a running average of the weights "
                         f"(decay {EMA_DECAY:g}, shorter at first) instead "
                         "of the weights themselves")
    ap.add_argument("--select", choices=("score", "survival"),
                    help="what picks the best model: score, the mean rollout "
                         "score; survival, the time R2 stays above "
                         f"{R2_THRESHOLD:g}, with the accuracy up to twice "
                         f"the training horizon as tie-break (file: "
                         f"{SELECT})")
    ap.add_argument("--no-hard-constraints", action="store_true",
                    help="the ablation without hard constraints: sets "
                         "HARD_CONSTRAINTS = False as the import-time block "
                         "does (no zero DC, no final mean removal), takes "
                         "the plain head, writes train_t<END>_nohard[_tag] "
                         "(--variant lap is refused with it)")
    ap.add_argument("--mass-weight", type=_weight, metavar="W",
                    help="the weight of the mass-drift penalty (>= 0; file: "
                         f"{MASS_WEIGHT:g}); needs --no-hard-constraints, "
                         "since the penalty acts only without hard mass "
                         "conservation")
    ap.add_argument("--no-mu-channel", action="store_true",
                    help="the ablation without the chemical-potential input "
                         "channel: sets MU_CHANNEL = False (the mu clamp "
                         "then has nothing to act on); needs --tag NAME, "
                         "since the folder's name does not say it and a "
                         "model trained with the channel must not be "
                         "overwritten")
    ap.add_argument("--fixed-step", action="store_true",
                    help="train the old kind of model, one DT_STEP per call "
                         "and the loop and loss it had (use_fixed_step); the "
                         "default is the variable-step model of the group protocol's "
                         "method, which takes dt per call")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run on the CPU even inside a SLURM job that has no "
                         "GPU (which is refused otherwise; FNO_ALLOW_CPU=1 "
                         "does the same)")
    ap.add_argument("--smoke", action="store_true",
                    help="a tiny end-to-end run (2 epochs of 4 steps, small "
                         "validation, a lean dataset cut to t = 20) that "
                         "writes only under Work/smoke; the other flags "
                         "still apply on top")
    ap.add_argument("--print-label", action="store_true",
                    help="print the train label these flags give (the "
                         "folder under Work/models) and exit; needs "
                         "--t-train, reads no data, writes nothing")
    return ap.parse_args(argv)


def apply_smoke(args):
    """The settings of --smoke: a tiny run of this very script, whose files
    all go under Work/smoke.

    Called right after parse_args, before the log is opened (so the log goes
    there too) and before apply_overrides, so a flag beside --smoke still
    wins.  A lean dataset is sliced to t = 20 unless --t-train says
    otherwise.  Returns the lines to print once the log is open.
    """
    global ROOT, GT_CACHE, MODEL_OUT, CHECKPOINT, EPOCHS, STEPS_PER_EPOCH
    global BATCH_SIZE, VAL_EVERY, PATIENCE, MAX_HOURS, RESUME, OFF_CRITICAL
    global VAL_SEEDS, VAL_LATTICE, VAL_T_START, VAL_T_END, VAL_BATCHES
    global MONITOR_SEEDS, MONITOR_OFFS, MONITOR_LATTICE, MONITOR_T_END
    global MAX_UNROLL, REF_CACHE, T_MIN, VAL1_WINDOWS
    path = args.dataset or DATASET
    try:
        lean = lean_meta(path) is not None
    except ValueError as exc:
        raise SystemExit(str(exc))
    ROOT = fno_paths.smoke_root()
    GT_CACHE = fno_paths.gt_dir(ROOT)
    REF_CACHE = fno_paths.ref_cache_dir(ROOT)
    MODEL_OUT = CHECKPOINT = None
    fno_runlog.LOG_DIR = fno_paths.log_dir(ROOT)
    EPOCHS, STEPS_PER_EPOCH, BATCH_SIZE, VAL_EVERY = 2, 4, 2, 1
    PATIENCE, MAX_HOURS, RESUME, OFF_CRITICAL = 0, None, False, False
    VAL_SEEDS, VAL_LATTICE = VAL_SEEDS[:1], 32
    VAL_T_START, VAL_T_END, VAL_BATCHES = 100.0, 110.0, 2
    if DT_CONDITIONED:
        # the end of the window must be a reference frame (every 4.0)
        VAL_T_END, T_MIN, VAL1_WINDOWS = 112.0, 0.0, 4
    MONITOR_SEEDS, MONITOR_OFFS = MONITOR_SEEDS[:1], MONITOR_OFFS[:1]
    MONITOR_LATTICE, MONITOR_T_END = 32, 110.0
    MAX_UNROLL = 2
    lines = [f"writing only under {ROOT}: EPOCHS {EPOCHS}, STEPS_PER_EPOCH "
             f"{STEPS_PER_EPOCH}, BATCH_SIZE {BATCH_SIZE}, VAL_EVERY "
             f"{VAL_EVERY}, PATIENCE {PATIENCE}, MAX_HOURS {MAX_HOURS}, "
             f"RESUME {RESUME}, OFF_CRITICAL {OFF_CRITICAL}, MAX_UNROLL "
             f"{MAX_UNROLL}; validation on seed {VAL_SEEDS[0]} at l = "
             f"{VAL_LATTICE}, t = {VAL_T_START:g}..{VAL_T_END:g}, "
             f"{VAL_BATCHES} batches; monitor on seed {MONITOR_SEEDS[0]} at "
             f"l = {MONITOR_LATTICE} to t = {MONITOR_T_END:g}"]
    if not lean:
        lines.append(f"{path} is not a lean .npy, which is the only kind that "
                     f"can be sliced; it is read whole")
    elif args.t_train is None:
        args.t_train = 20.0
        lines[0] += "; T_TRAIN 20, so a lean dataset is sliced to t = 20"
    return lines


def apply_overrides(args):
    """Rebind whatever the CLI named, and say so in the log."""
    global DATASET, T_TRAIN, TAG, MODEL_OUT, CHECKPOINT, RESUME, EPOCHS
    global LAPLACIAN_OUTPUT, EMA, SELECT, CHANNELS, MODES, SEED
    global HARD_CONSTRAINTS, ZERO_DC, HARD_MASS_CONSERVATION, MASS_WEIGHT
    global MU_CHANNEL
    moved = []
    if args.no_mu_channel and not args.tag:
        raise SystemExit("--no-mu-channel needs --tag NAME: the folder's "
                         "name does not say whether the model has the mu "
                         "channel, so without a tag this run would use the "
                         "default's folder (e.g. --tag nomu gives "
                         "train_t0500_lap_nomu)")
    if args.no_mu_channel:
        moved.append(f"MU_CHANNEL {MU_CHANNEL} -> False (no chemical-"
                     f"potential input channel; the mu clamp is not used)")
        MU_CHANNEL = False
    if args.mass_weight is not None and not args.no_hard_constraints:
        raise SystemExit("--mass-weight needs --no-hard-constraints: the "
                         "mass penalty acts only without hard mass "
                         "conservation")
    if args.no_hard_constraints:
        moved.append(f"HARD_CONSTRAINTS {HARD_CONSTRAINTS} -> False; "
                     f"ZERO_DC {ZERO_DC} -> False; HARD_MASS_CONSERVATION "
                     f"{HARD_MASS_CONSERVATION} -> False (the _nohard "
                     f"ablation)")
        HARD_CONSTRAINTS = False
        ZERO_DC = False
        HARD_MASS_CONSERVATION = False
    if args.mass_weight is not None:
        moved.append(f"MASS_WEIGHT {MASS_WEIGHT:g} -> {args.mass_weight:g}")
        MASS_WEIGHT = args.mass_weight
    size = (CHANNELS if args.channels is None else args.channels,
            MODES if args.modes is None else args.modes)
    other_size = size != (CHANNELS, MODES)
    try:
        label_tag = fno_paths.compose_tag(
            size if other_size and not args.keep_folder else None,
            args.seed or 0, args.tag or "")
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.dataset:
        moved.append(f"DATASET {DATASET!r} -> {args.dataset!r}")
        DATASET = args.dataset
    if args.t_train is not None:
        moved.append(f"T_TRAIN {T_TRAIN} -> {args.t_train:g}")
        T_TRAIN = float(args.t_train)
    tag = label_tag or (f"T{args.t_train:g}" if args.t_train is not None
                        else None)
    defining = (args.dataset or args.val_t_start is not None
                or args.val_t_end is not None
                or args.monitor_t_end is not None)
    if defining and not tag and ROOT is None:
        # Another dataset or validation window is another run: it must not
        # write to, or resume, the default fno_ch.pt / fno_ch_last.pt.
        # (--smoke writes under Work/smoke, so it may go without a name.)
        meta = lean_meta(DATASET) if args.dataset else None
        if meta is None:
            raise SystemExit("--dataset / --val-* / --monitor-t-end define a "
                             "different run; name it with --t-train T or "
                             "--tag NAME so it does not overwrite fno_ch.pt")
        tag = f"T{float(meta['t_end']):g}"
    if tag:
        TAG = tag
        MODEL_OUT = CHECKPOINT = None
        moved.append(f"TAG -> {tag!r}; the outputs are named by the training "
                     f"horizon and this tag")
    for name, val in (("VAL_T_START", args.val_t_start),
                      ("VAL_T_END", args.val_t_end),
                      ("MONITOR_T_END", args.monitor_t_end)):
        if val is not None:
            moved.append(f"{name} {globals()[name]:g} -> {val:g}")
            globals()[name] = float(val)
    if args.epochs is not None:
        moved.append(f"EPOCHS {EPOCHS} -> {args.epochs}")
        EPOCHS = args.epochs
    if args.variant is not None:
        moved.append(f"LAPLACIAN_OUTPUT {LAPLACIAN_OUTPUT} -> "
                     f"{args.variant == 'lap'}")
        LAPLACIAN_OUTPUT = args.variant == "lap"
    if LAPLACIAN_OUTPUT and not HARD_CONSTRAINTS and args.variant != "lap":
        moved.append("LAPLACIAN_OUTPUT True -> False: the no-hard-"
                     "constraints ablation takes the base head, as the "
                     "Laplacian one conserves mass by itself (--variant lap "
                     "is refused with it)")
        LAPLACIAN_OUTPUT = False
    if args.channels is not None:
        moved.append(f"CHANNELS {CHANNELS} -> {args.channels}")
        CHANNELS = args.channels
    if args.modes is not None:
        moved.append(f"MODES {MODES} -> {args.modes}")
        MODES = args.modes
    if args.keep_folder and other_size:
        moved.append(f"--keep-folder: the size c{size[0]}m{size[1]} stays "
                     f"out of the folder's name")
    if args.seed is not None:
        moved.append(f"SEED {SEED} -> {args.seed}")
        SEED = args.seed
    if args.ema:
        moved.append(f"EMA {EMA} -> True")
        EMA = True
    if args.select is not None:
        moved.append(f"SELECT {SELECT!r} -> {args.select!r}")
        SELECT = args.select
    if args.no_resume:
        moved.append("RESUME -> False")
        RESUME = False
    for m in moved:
        print(f"[cli] {m}", flush=True)
    return moved


def use_old_inputs():
    """Read the dataset and the references from where they were before the
    folder layout, if they are not at the new place yet and are still the
    file's own defaults (a flag that named another is left alone).  Only
    inputs: what a run writes always goes to the new places."""
    global DATASET, GT_CACHE
    if DATASET == os.path.join(fno_paths.data_dir(), "test.npz"):
        DATASET = fno_paths.legacy(DATASET, "test.npz", "the dataset")
    if GT_CACHE == fno_paths.gt_dir():
        GT_CACHE = fno_paths.legacy(GT_CACHE, "gt_cache", "the references")


def label_for(t):
    """The train label of a run to horizon ``t`` with the current settings.
    The tag joins it unless it is the automatic T<horizon>."""
    tag = "" if TAG == f"T{t:g}" else TAG
    return fno_paths.train_label(t, lap=LAPLACIAN_OUTPUT,
                                 nohard=not HARD_CONSTRAINTS, tag=tag)


def resolve_outputs(n_frames):
    """Name the model's files from the horizon of the data just loaded.

    Whichever of MODEL_OUT and CHECKPOINT is still None becomes
    Work/models/<train label>/fno_ch.pt or fno_ch_last.pt (under ROOT, which
    --smoke sets); one that was set (by a test, or an edit to this file) is
    left alone.  The horizon is T_TRAIN, or else what the frames cover.  The
    tag joins the label unless it is the automatic T<horizon>, which says
    nothing the label does not.

    A resume state that is still where the old code kept it, with none yet
    at the new place, is refused rather than silently started over from
    epoch 0: python migrate_layout.py moves it.  Returns the train label.
    """
    global MODEL_OUT, CHECKPOINT
    label = label_for(train_horizon(n_frames))
    if MODEL_OUT is None:
        MODEL_OUT = fno_paths.model_path(label, ROOT)
    derived = CHECKPOINT is None
    if CHECKPOINT is None:
        CHECKPOINT = os.path.join(fno_paths.model_dir(label, ROOT),
                                  fno_paths.STATE_FILE)
    print(f"[outputs] best model -> {MODEL_OUT}, resume state -> "
          f"{CHECKPOINT}", flush=True)
    ablation = "" if HARD_CONSTRAINTS else "_nohard"
    old = (f"fno_ch_{TAG}{ablation}_last.pt" if TAG
           else f"fno_ch{ablation}_last.pt")
    if (derived and RESUME and ROOT is None and os.path.exists(old)
            and not os.path.exists(CHECKPOINT)):
        raise SystemExit(f"found {old}, the resume state an earlier run of "
                         f"this name left where the old code kept it, but "
                         f"nothing at {CHECKPOINT}, so this run would start "
                         f"over from epoch 0.  python migrate_layout.py (a "
                         f"dry run; then --apply) moves it to {CHECKPOINT}; "
                         f"--no-resume starts a fresh run instead.")
    return label


def best_variant(ck):
    """The EMA, SELECT and LAPLACIAN_OUTPUT a saved best model was trained
    with, from the constants in its train_config.  A model with none (from
    before they were recorded) is EMA off, SELECT score and the base head;
    one from before MU_CHANNEL has no mu channel."""
    consts = (ck.get("train_config") or {}).get("constants") or {}
    return {"EMA": consts.get("EMA", False),
            "SELECT": consts.get("SELECT", "score"),
            "LAPLACIAN_OUTPUT": consts.get("LAPLACIAN_OUTPUT", False),
            "DT_CONDITIONED": consts.get("DT_CONDITIONED", False),
            "MU_CHANNEL": consts.get("MU_CHANNEL", False)}


def check_best_model(path):
    """Refuse to start fresh over another run's best model.

    The first validation of a fresh run saves its model to ``path``, so a
    file that is there and was written by another run would be lost.  What
    run wrote it is in the file (run_meta); a file with none is the default
    run's, from before settings were recorded -- such as a migrated legacy
    fno_ch.pt, which a lean --t-train 300 run now shares a folder with.  The
    same run may overwrite its own.  --smoke's folder is for throwaways.

    --ema and --select do not change the folder's name, nor does MU_CHANNEL,
    so a best model trained with another EMA, SELECT, LAPLACIAN_OUTPUT or
    MU_CHANNEL is refused too.
    """
    if ROOT is not None or not os.path.exists(path):
        return
    ck = torch.load(path, map_location="cpu", weights_only=False)
    saved = ck.get("run_meta")
    was_dt = bool((saved or {}).get("dt_conditioned", False))
    if was_dt != DT_CONDITIONED:
        kind = {True: "dt-conditioned (variable step)", False: "fixed-step"}
        raise SystemExit(
            f"{path} is the best model of a {kind[was_dt]} run, this run is "
            f"{kind[DT_CONDITIONED]} (dt_conditioned {was_dt} -> "
            f"{DT_CONDITIONED}), and its first validation would overwrite "
            f"it; move the old folder to Archive/, or use another --tag "
            f"NAME, which gives this run a folder of its own"
            + ("" if DT_CONDITIONED else " (the old kind is --fixed-step)")
            + ".")
    if (saved if saved is not None else DEFAULT_RUN_META) != run_meta():
        raise SystemExit(
            f"{path} is the best model of another run "
            f"({saved or 'no run settings recorded'}), not this one "
            f"({run_meta()}), and this run's first validation would "
            f"overwrite it; use another --tag NAME, which gives this run a "
            f"folder of its own.")
    was = best_variant(ck)
    now = {"EMA": EMA, "SELECT": SELECT, "LAPLACIAN_OUTPUT": LAPLACIAN_OUTPUT,
           "DT_CONDITIONED": DT_CONDITIONED, "MU_CHANNEL": MU_CHANNEL}
    rows = "\n".join(f"  {k}: {was[k]} -> {now[k]}" for k in now
                     if was[k] != now[k])
    if rows:
        raise SystemExit(
            f"{path} is the best model of a run trained another way "
            f"(saved -> this run):\n{rows}\n"
            f"--ema, --select and MU_CHANNEL do not change the folder, and "
            f"this run's first validation would overwrite it; use another "
            f"--tag NAME, which gives this run a folder of its own.")


def check_variant():
    """Refuse a model and a selection rule that do not go together."""
    if LAPLACIAN_OUTPUT and not HARD_CONSTRAINTS:
        raise SystemExit("--variant lap with HARD_CONSTRAINTS = False is "
                         "refused: the Laplacian head conserves mass exactly "
                         "by itself, so the ablation that leaves mass to a "
                         "soft penalty would measure nothing.  Use "
                         "--variant base for the ablation.")
    if SELECT not in ("score", "survival"):
        raise SystemExit(f"SELECT must be 'score' or 'survival', not "
                         f"{SELECT!r}")


def check_windows():
    """Fail at start-up, not at the first validation epoch."""
    for name in ("VAL_T_START", "VAL_T_END", "MONITOR_T_END"):
        v = globals()[name]
        if abs(v / DT_STEP - round(v / DT_STEP)) > 1e-9:
            raise SystemExit(f"{name} = {v:g} is not a multiple of the model "
                             f"step {DT_STEP:g}")
    if VAL_T_START >= VAL_T_END:
        raise SystemExit(f"the validation window [{VAL_T_START:g}, "
                         f"{VAL_T_END:g}] is empty")
    if DT_CONDITIONED:
        k = (VAL_T_END - VAL_T_START) / VAL_QUANTUM
        if abs(k - round(k)) > 1e-9:
            raise SystemExit(f"the validation window [{VAL_T_START:g}, "
                             f"{VAL_T_END:g}] is not a whole number of "
                             f"reference spacings ({VAL_QUANTUM:g}), so its "
                             f"end is not a reference frame")
    if (not DT_CONDITIONED and TRAIN_ON_VAL
            and VAL_T_START >= MONITOR_T_END):
        raise SystemExit(f"the monitor trajectories end at {MONITOR_T_END:g}, "
                         f"not after the window start {VAL_T_START:g}; raise "
                         f"--monitor-t-end")


def _healthy_state(epoch, model, opt):
    """What the divergence guard rolls back to: weights and optimizer."""
    return {"epoch": epoch,
            "model": {k: v.detach().cpu().clone() for k, v in
                      _tensors_only(model.state_dict()).items()},
            "optimizer": copy.deepcopy(opt.state_dict())}


def main(argv=None):
    args = parse_args(argv)
    fixed_notes = use_fixed_step() if args.fixed_step else []
    if args.print_label:
        # fp.sh and ftp.sh name the model this way, so it is the label the
        # training writes; the override notes go to stderr, the label alone
        # to stdout
        if args.t_train is None:
            raise SystemExit("--print-label needs --t-train")
        with contextlib.redirect_stdout(sys.stderr):
            apply_overrides(args)
            check_variant()
        print(label_for(float(args.t_train)))
        return
    smoke = apply_smoke(args) if args.smoke else []
    if WRITE_LOG:
        fno_runlog.start_log("fno_train")
    for line in smoke:
        print(f"[smoke] {line}", flush=True)
    if args.fixed_step:
        print("[cli] --fixed-step: the fixed-step model and loop"
              + "".join(f"; {m}" for m in fixed_notes), flush=True)
    apply_overrides(args)
    check_variant()
    use_old_inputs()
    check_windows()
    t_launch = time.time()
    deadline, budget_why = resolve_budget(t_launch)
    print(f"[budget] {budget_why}", flush=True)
    last_check, worst_cycle = t_launch, 0.0
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device, amp_ctx = device_and_precision()
    require_gpu(device, args.allow_cpu)

    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print(f"[setup] {prop.name}, {prop.total_memory / 1e9:.1f} GB, "
              f"torch {torch.__version__}", flush=True)
    print(f"[setup] device={device} precision={PRECISION} amp={AMP} "
          f"compile={COMPILE}", flush=True)

    train_arr, val_arr = load_trajectories(DATASET)
    if args.modes is not None:
        check_modes(train_arr.shape[-1])
    label = resolve_outputs(train_arr.shape[1])
    assert MODEL_OUT is not None and CHECKPOINT is not None
    lattice = train_arr.shape[-1]
    horizon = train_horizon(train_arr.shape[1])

    if TRAIN_ON_VAL and val_arr.shape[0]:
        n_train = train_arr.shape[0]
        train_arr = np.concatenate([train_arr, val_arr], axis=0)
        val_arr = val_arr[:0]
        comp = np.round(train_arr[:, 0].mean(axis=(1, 2)).astype(np.float64), 2)
        counts = {float(c): int((comp == c).sum()) for c in np.unique(comp)}
        print(f"[data] TRAIN_ON_VAL: training on all {train_arr.shape[0]} "
              f"trajectories ({n_train} train + "
              f"{train_arr.shape[0] - n_train} val); compositions {counts}",
              flush=True)
    elif TRAIN_ON_VAL:
        comp = np.round(train_arr[:, 0].mean(axis=(1, 2)).astype(np.float64), 2)
        counts = {float(c): int((comp == c).sum()) for c in np.unique(comp)}
        print(f"[data] TRAIN_ON_VAL: training on all {train_arr.shape[0]} "
              f"trajectories; compositions {counts}", flush=True)

    if OFF_CRITICAL and OFF_CRITICAL_TRAJ > 0 and OFF_CRITICAL_OFFS:
        if train_arr.nbytes > 4e9:
            print(f"[off] note: appending off-critical trajectories copies "
                  f"the {train_arr.nbytes / 1e9:.1f} GB training set; expect "
                  f"a peak of about {2.4 * train_arr.nbytes / 1e9:.0f} GB",
                  flush=True)
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

    chains = None
    if DT_CONDITIONED:
        chains = ChainSampler(train_arr, device, data_device, torch_dtype(),
                              SEED)
        train_data = val_data = None
        print(f"[data] chains of {[n for _, n in VDT_UNROLL_STAGES]} steps, "
              f"dt {DT_MIN:g}..{DT_MAX:g} in steps of {FRAME_DT:g}, from "
              f"t >= {T_MIN:g}; max trainable dt by unroll: "
              + ", ".join(f"{n} -> {chains.max_dt(n):.1f}"
                          for _, n in VDT_UNROLL_STAGES), flush=True)
    else:
        train_data = WindowSampler(train_arr, device, data_device, 1, True,
                                   SEED)
        val_data = (None if TRAIN_ON_VAL else
                    WindowSampler(val_arr, device, data_device, 1, False,
                                  SEED + 1))
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
    if LAPLACIAN_OUTPUT:
        print("[model] LAPLACIAN_OUTPUT: the head predicts M and the step is "
              "lap5(M), so mass is conserved by construction", flush=True)
    if model.mu_channel:
        print("[model] MU_CHANNEL: the input also holds mu = psi^3 - psi - "
              "lap(psi), the chemical potential"
              + (f" of psi clamped to +-{model.mu_clip:g}"
                 if model.mu_clip is not None else ""), flush=True)
    if EMA:
        print(f"[model] EMA: the rollout, the monitor and the best model use "
              f"a running average of the weights (decay {EMA_DECAY:g}, "
              f"shorter for the first steps)", flush=True)
    step_model = (try_compile(model, [lattice, VAL_LATTICE], BATCH_SIZE,
                              device, amp_ctx) if COMPILE else model)

    if DT_CONDITIONED:
        opt = torch.optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY, betas=VDT_BETAS)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY)

    def cosine(epoch):
        if DT_CONDITIONED:
            return vdt_lr(epoch) / LR
        p = min(epoch / max(EPOCHS, 1), 1.0)
        base = LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * p))
        stages = int(math.log2(unroll_at(epoch, EPOCHS, MAX_UNROLL)))
        return base * (LR_STAGE_DECAY ** stages)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine)

    start_epoch, history, stale = 0, [], 0
    best = {"score": -1.0, "survival": -1.0}
    schedule_change: dict[str, list] | None = None
    ck, fresh_why = None, ("resume is off (--no-resume or RESUME = False)"
                           if not RESUME else "there is no resume state")
    if RESUME and os.path.exists(CHECKPOINT):
        ck = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        saved_meta = ck.get("run_meta")
        if (saved_meta != run_meta() if saved_meta is not None
                else run_meta() != DEFAULT_RUN_META):
            raise SystemExit(
                f"[resume] {CHECKPOINT} belongs to another run "
                f"({saved_meta or 'no run settings recorded'}), not this one "
                f"({run_meta()}). Resuming it would mix the two; use another "
                f"--tag NAME, which gives this run a folder of its own.  "
                f"(--no-resume keeps the old files as "
                f"{kept_names(MODEL_OUT, CHECKPOINT)} and starts over, but "
                f"only where {MODEL_OUT} is this run's own.)")
        drift = model_drift(ck.get("config", {}), model.config)
        if drift:
            raise SystemExit(drift_refusal(
                CHECKPOINT, drift, kept_names(MODEL_OUT, CHECKPOINT)))
        if ck.get("schedule") is None:
            print(f"[resume] {CHECKPOINT} has no schedule stamp (it is from "
                  f"before they were recorded); resuming it without checking "
                  f"the schedule", flush=True)
        else:
            changed = schedule_drift(ck["schedule"], schedule_stamp())
            if changed:
                rows = "\n".join(f"  {k}: {was} -> {now}"
                                 for k, (was, now) in changed.items())
                if not args.accept_schedule:
                    raise SystemExit(
                        f"[resume] {CHECKPOINT} was made with another "
                        f"training schedule (checkpoint -> current):\n"
                        f"{rows}\n"
                        f"Resuming would reshape the run in the middle.  "
                        f"Put these back; or use --accept-schedule to go on "
                        f"with the new schedule; or use --no-resume, which "
                        f"starts fresh and keeps the old files as "
                        f"{kept_names(MODEL_OUT, CHECKPOINT)}.")
                print(f"[resume] {CHECKPOINT} was made with another training "
                      f"schedule; --accept-schedule goes on with the new one "
                      f"(checkpoint -> current):\n{rows}", flush=True)
                schedule_change = {k: [was, now]
                                   for k, (was, now) in changed.items()}
    if ck is None:
        check_best_model(MODEL_OUT)
        if ROOT is None:
            rotate_old_files(MODEL_OUT, CHECKPOINT, fresh_why)

    if ck is not None:
        model.load_state_dict(_tensors_only(ck["state_dict"]))
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, best = ck["epoch"], ck.get("best", best)
        history = ck.get("history", [])
        best.setdefault("score", -1.0)
        # the patience count goes on across starts; a state from before it
        # was saved starts the count again, as every start used to
        stale = ck.get("stale", 0)
        print(f"[resume] {CHECKPOINT} at epoch {start_epoch}", flush=True)

    ema = EmaWeights(model) if EMA else None
    if ema is not None and ck is not None:
        if ck.get("ema") is None:
            print(f"[resume] {CHECKPOINT} holds no averaged weights (it is "
                  f"from before EMA); the average starts from its weights",
                  flush=True)
        else:
            ema.load(ck["ema"], ck.get("ema_n", 0))

    config_file = _config_path(MODEL_OUT)
    cfg = train_config(label, argv if argv is not None else sys.argv[1:],
                       None if ck is None else start_epoch,
                       [] if ck is None else read_launches(config_file))
    _write_json(config_file, cfg)
    print(f"[outputs] how this run is set up -> {config_file}", flush=True)

    refs = None
    val1: list = []
    monitor: list = []
    if DT_CONDITIONED:
        refs = open_val_refs(device, lambda m: print(m, flush=True))
        gts = refs.gts
        val1 = val1_batches(refs, device, torch_dtype())
        print(f"[val] one-step check (val1): strides "
              f"{[b[0] for b in val1]} of the held-out references, "
              f"{VAL1_WINDOWS} windows each", flush=True)
    else:
        print(f"[val] l={VAL_LATTICE} ground truth for seeds "
              f"{list(VAL_SEEDS)}", flush=True)
        gts = [ch_ground_truth(VAL_LATTICE, s, 0.0, VAL_T_END, DT_STEP,
                               cache_dir=GT_CACHE, device=str(device))
               for s in VAL_SEEDS]

    if DT_CONDITIONED:
        pass
    elif val_data is None:
        pairs = list(zip(MONITOR_SEEDS, MONITOR_OFFS))
        print(f"[val] monitor ground truth: l={MONITOR_LATTICE}, "
              f"{len(pairs)} trajectories to t={MONITOR_T_END:.0f} "
              f"{[(s, round(o, 1)) for s, o in pairs]}", flush=True)
        mon_gts = [ch_ground_truth(MONITOR_LATTICE, seed, off, MONITOR_T_END,
                                   DT_STEP, cache_dir=GT_CACHE,
                                   device=str(device))
                   for seed, off in pairs]
        monitor = holdout_monitor(mon_gts, device, torch_dtype(),
                                  BATCH_SIZE * VAL_BATCHES, BATCH_SIZE,
                                  t_start=VAL_T_START)
        del mon_gts
        n_win = sum(x.shape[0] for x, _ in monitor)
        print(f"[val] one-step monitor: {n_win} windows from "
              f"{len(pairs)} held-out l={MONITOR_LATTICE} trajectories "
              f"over off={list(MONITOR_OFFS)}, t >= {VAL_T_START:.0f}; "
              f"independent of the {len(VAL_SEEDS)} rollout seeds", flush=True)
    else:
        monitor = [(w[:, 0:1], w[:, 1:2])
                   for w in val_data.fixed_batches(BATCH_SIZE, VAL_BATCHES)]
        print(f"[val] one-step monitor: test.npz val_X "
              f"({sum(x.shape[0] for x, _ in monitor)} windows)", flush=True)

    # The budget measures epochs; the ground-truth solves above are one-off.
    last_check = time.time()
    recent_losses = deque(maxlen=10)
    batch_losses = deque(maxlen=400)
    # A resumed run starts from a state that was already good on disk, so the
    # guard can roll back to it even if the first resumed epoch trips it.
    healthy = None if ck is None else _healthy_state(start_epoch, model, opt)
    recoveries = 0
    stopped_early_training = False
    stopped_on_budget = False

    # A run that already stopped on its patience, or ran its EPOCHS, trains
    # nothing more.  Nothing below the loop writes a file, so fno_ch.pt (whose
    # fingerprint the predictions and their caches key on) stays as it is.
    todo = range(start_epoch, EPOCHS)
    if PATIENCE and stale >= PATIENCE:
        print(f"[stop] {CHECKPOINT} already stopped on patience ({stale} "
              f"validations without a gain; best was {best['score']:.4f} at "
              f"epoch {best.get('epoch')}); nothing to train (raise PATIENCE "
              f"to continue; PATIENCE is not part of the schedule stamp)",
              flush=True)
        stopped_early_training = True
        todo = range(0)
    elif start_epoch >= EPOCHS:
        print(f"[stop] {CHECKPOINT} is at epoch {start_epoch} of {EPOCHS}; "
              f"nothing to train", flush=True)

    gen = torch.Generator(device=device).manual_seed(SEED)
    prev_unroll = None
    for epoch in todo:
        if DT_CONDITIONED:
            k_max = vdt_unroll(epoch)[0]
            max_unroll = max(n for _, n in VDT_UNROLL_STAGES)
            if prev_unroll is not None and k_max != prev_unroll:
                recent_losses.clear()
                print(f"[stage] unroll -> {k_max}; max trainable dt "
                      f"{chains.max_dt(k_max) if chains else 0:.1f}",
                      flush=True)
            prev_unroll = k_max
        else:
            k_max, max_unroll = unroll_at(epoch, EPOCHS, MAX_UNROLL), MAX_UNROLL
            assert train_data is not None
            if train_data.n_steps != k_max:
                train_data.set_n_steps(k_max)
                batch_losses.clear()

        model.train()
        t0, running, skipped, done = time.time(), 0.0, 0, 0
        spiked = bad_grad = 0
        dpsi_train = 0.0
        for _ in range(0 if DT_CONDITIONED else STEPS_PER_EPOCH):
            assert train_data is not None
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
                        x = step(step_model, x, DT_STEP)

                if not torch.isfinite(x).all() or float(x.abs().max()) > PSI_MAX:
                    skipped += 1
                    continue
                x = x.detach()

            opt.zero_grad(set_to_none=True)
            with amp_ctx():

                step_losses, x_in = [], x
                for j in range(n_detached, k):
                    x_in = x
                    x = step(step_model, x, DT_STEP)
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
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   GRAD_CLIP)
            if not torch.isfinite(gnorm):
                opt.zero_grad(set_to_none=True)
                skipped += 1
                bad_grad += 1
                continue
            opt.step()
            if ema is not None:
                ema.update(model)
            batch_losses.append(loss_val)
            running += loss_val
            done += 1
        if DT_CONDITIONED:
            assert chains is not None
            run_loss, done, skipped, dpsi_train = vdt_epoch(
                model, step_model, chains, opt, ema, gen, k_max, device,
                amp_ctx)
            running = run_loss
        sched.step()
        train_loss = running / done if done else float("inf")
        secs = time.time() - t0

        model.eval()
        if DT_CONDITIONED:
            with torch.no_grad(), averaged(ema, model):
                val_loss = val1_loss(step_model, val1, amp_ctx)
            delta_rel = dpsi_train / done if done else float("nan")
        else:
            with torch.no_grad(), averaged(ema, model):
                vals, deltas = [], []
                for psi, target in monitor:
                    with amp_ctx():
                        pred = step(step_model, psi, DT_STEP)
                        vals.append(float(loss_fn(pred, target)))

                    deltas.append(float((pred - psi).abs().mean()
                                        / psi.abs().mean().clamp_min(1e-8)))
            val_loss = float(np.mean(vals)) if vals else float("nan")
            delta_rel = float(np.mean(deltas)) if deltas else float("nan")

        median = float(np.median(recent_losses)) if recent_losses else None
        stalled = skipped > SPIKE_ABORT_FRAC * STEPS_PER_EPOCH
        unphysical = (not np.isfinite(delta_rel)) or delta_rel > MAX_DELTA_REL
        blown = not np.isfinite(train_loss)
        jumped = (median is not None and np.isfinite(median)
                  and train_loss > DIVERGE_FACTOR * median)
        diverged = blown or stalled or unphysical or jumped
        if diverged and healthy is not None and recoveries < MAX_RECOVERIES:
            recoveries += 1
            model.load_state_dict(healthy["model"], strict=False)
            opt.load_state_dict(healthy["optimizer"])
            if ema is not None:
                ema.reset(model)
            sched.base_lrs = [b * LR_RECOVERY_DECAY for b in sched.base_lrs]
            batch_losses.clear()
            why = diverge_reason(train_loss, delta_rel, skipped, median,
                                 blown, stalled, unphysical)
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
            why = diverge_reason(train_loss, delta_rel, skipped, median,
                                 blown, stalled, unphysical)
            where = ("no healthy epoch to roll back to yet"
                     if healthy is None else
                     f"out of recoveries [{recoveries}/{MAX_RECOVERIES}]")
            print(f"epoch {epoch + 1:3d}/{EPOCHS} | DIVERGED ({why}); "
                  f"{where}; stopping.", flush=True)
            if healthy is None:
                print("  The first epoch of this run was already unhealthy, "
                      "so there is nothing to recover to and no point "
                      "spending the wall clock. Check the settings that "
                      "changed since the last good run -- LOCAL_KERNEL and LR "
                      "are the two that have done this before.", flush=True)
            else:
                print(f"  Best model is in {MODEL_OUT}.", flush=True)
            history.append({"epoch": epoch + 1, "diverged": True,
                            "train": train_loss, "skipped": skipped,
                            "stalled": bool(stalled),
                            "unphysical": bool(unphysical),
                            "delta_rel": delta_rel,
                            "rolled_back_to": None})
            break
        recent_losses.append(train_loss)
        healthy = _healthy_state(epoch + 1, model, opt)

        line = (f"epoch {epoch + 1:3d}/{EPOCHS} | unroll {k_max:2d} | "
                f"train {train_loss:.5f} | val1 {val_loss:.5f} | "
                f"d/psi {delta_rel:.4f} | lr {sched.get_last_lr()[0]:.2e} | "
                f"{secs:5.1f}s ({secs / STEPS_PER_EPOCH * 1e3:.0f} ms/step)"
                + (f" | {skipped} skipped"
                   + (f" ({spiked} spikes)" if spiked else "")
                   + (f" ({bad_grad} non-finite gradients)"
                      if bad_grad else "")
                   if skipped else ""))
        rec = {"epoch": epoch + 1, "unroll": k_max, "train": train_loss,
               "val1": val_loss, "delta_rel": delta_rel, "seconds": secs,
               "skipped": skipped, "spiked": spiked,
               "nonfinite_grad": bad_grad}
        if schedule_change:
            rec["schedule_change"] = schedule_change
            schedule_change = None

        if (epoch + 1) % VAL_EVERY == 0 or epoch == EPOCHS - 1:
            gains: dict = {}
            improved = False
            with averaged(ema, model):
                if DT_CONDITIONED:
                    assert refs is not None
                    ro = vdt_rollout(model, refs, device, amp_ctx,
                                     horizon=horizon)
                    score, surv = ro["score"], float(np.mean(ro["survival"]))
                    acc = ro["accuracy"]
                    in_t = min(2.0 * horizon, VAL_T_END)
                    rec.update(score=score, survival=surv,
                               per_seed=ro["survival"],
                               survival_interp=surv,
                               per_seed_interp=ro["survival"], in_range=acc)
                    line += (f" | rollout {score:.4f} | in-range R2 "
                             f"{acc:.4f} (t <= {in_t:g}) | survives "
                             f"t={surv:7.1f} "
                             f"{[int(round(x)) for x in ro['survival']]}")
                    new_best = {"score": score, "survival": surv,
                                "epoch": epoch + 1,
                                "per_seed": ro["survival"], "accuracy": acc,
                                "peak_survival": max(
                                    surv, best.get("peak_survival", -1.0))}
                    improved = better_epoch(
                        surv, acc, best.get("survival", -1.0),
                        best.get("accuracy", -1.0), SURVIVAL_MIN_DELTA)
                    curves = []
                    lk_dt = LOWK_DT
                    lk_t = lowk_times(gts[0][0], lk_dt)
                    gains = lowk_gains(model, gts, lk_t, VAL_LATTICE / lattice,
                                       str(device), amp_ctx, dt=lk_dt)
                else:
                    curves = [rollout_curve(model, t, f, str(device), amp_ctx,
                                            t_start=VAL_T_START,
                                            t_end=VAL_T_END)
                              for t, f in gts]
                    per_seed = [score_from_curve(t, r, VAL_T_END)
                                for t, r in curves]
                    score = float(np.mean([p[0] for p in per_seed]))
                    surv = float(np.mean([p[1] for p in per_seed]))
                    rec.update(score=score, survival=surv,
                               per_seed=[p[1] for p in per_seed])
                    line += (f" | rollout {score:.4f} | survives "
                             f"t={surv:6.0f} {[int(p[1]) for p in per_seed]}")
                    new_best = {"score": score, "survival": surv,
                                "epoch": epoch + 1,
                                "per_seed": [p[1] for p in per_seed]}
                if DT_CONDITIONED:
                    pass
                elif SELECT == "survival":
                    in_t = min(2.0 * horizon, VAL_T_END)
                    interp = [interp_survival(t, r, R2_THRESHOLD, VAL_T_END)
                              for t, r in curves]
                    acc = float(np.mean([in_range_accuracy(t, r, in_t)
                                         for t, r in curves]))
                    surv_i = float(np.mean(interp))
                    rec.update(survival_interp=surv_i, per_seed_interp=interp,
                               in_range=acc)
                    line += (f" | interpolated survival {surv_i:7.1f} | "
                             f"in-range R2 {acc:.4f} (t <= {in_t:g})")
                    peak = best.get("peak_survival")
                    new_best.update(survival=surv_i, per_seed=interp,
                                    accuracy=acc,
                                    peak_survival=max(
                                        surv_i, best.get("survival", -1.0),
                                        -1.0 if peak is None else peak))
                    improved = better_epoch(
                        surv_i, acc, best.get("survival", -1.0),
                        best.get("accuracy", -1.0), SURVIVAL_MIN_DELTA, peak)
                else:
                    improved = score > best["score"] + MIN_DELTA
                if not DT_CONDITIONED:
                    gains = lowk_gains(model, gts, lowk_times(gts[0][0]),
                                       VAL_LATTICE / lattice, str(device),
                                       amp_ctx)
                if gains:
                    rec["lowk"] = {f"{t0:g}": g for t0, g in gains.items()}
                    line += f" | lowk g {lowk_text(gains)}"
                if improved:
                    best = new_best
                    save_model(model, MODEL_OUT, epoch=epoch + 1, best=best,
                               run_meta=run_meta(), train_config=cfg)
            if improved:
                line += "  <- best"
                stale = 0
            elif PATIENCE_AFTER_RAMP and k_max < max_unroll:
                line += (f"  (no gain; patience held, unroll "
                         f"{k_max}<{max_unroll})")
            else:
                stale += 1
                line += f"  (no gain {stale}/{PATIENCE})" if PATIENCE else ""
        print(line, flush=True)
        history.append(rec)

        state = {"epoch": epoch + 1,
                 "state_dict": _tensors_only(model.state_dict()),
                 "optimizer": opt.state_dict(),
                 "scheduler": sched.state_dict(), "best": best,
                 "stale": stale,
                 "config": model.config, "history": history,
                 "run_meta": run_meta(), "schedule": schedule_stamp(),
                 "train_config": cfg}
        if ema is not None:
            state.update(ema=ema.shadow, ema_n=ema.n)
        try:
            _atomic(CHECKPOINT, lambda to: torch.save(state, to))
        except OSError as exc:
            print(f"[save] WARNING: could not save the resume state after "
                  f"epoch {epoch + 1} ({exc}); {CHECKPOINT} keeps the state "
                  f"of an earlier epoch, and the next epoch tries again",
                  flush=True)
        _write_json(_history_path(MODEL_OUT), history)

        if PATIENCE and stale >= PATIENCE:
            gain = (f"rollout score has not improved by {MIN_DELTA}"
                    if SELECT == "score" else
                    f"survival has not improved by {SURVIVAL_MIN_DELTA:g}")
            print(f"[stop] {gain} for {stale} validations "
                  f"({stale * VAL_EVERY} epochs); best was "
                  f"{best['score']:.4f} at epoch {best.get('epoch')}",
                  flush=True)
            stopped_early_training = True
            break

        now = time.time()
        worst_cycle = max(worst_cycle, now - last_check)
        last_check = now
        if deadline is not None:
            reserve = max(BUDGET_RESERVE_MIN * 60.0, 1.5 * worst_cycle)
            if now + reserve > deadline:
                print(f"[budget] {(deadline - now) / 60.0:.1f} min left and an "
                      f"epoch has taken up to {worst_cycle / 60.0:.1f} min; "
                      f"stopping after epoch {epoch + 1} with time to save. "
                      f"Resubmit to continue from {CHECKPOINT}.", flush=True)
                stopped_on_budget = True
                break

    if device.type == "cuda":
        print(f"[mem] peak allocated "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    print(f"[done] best rollout score {best['score']:.4f} (survives to "
          f"t = {best['survival']:.0f}, epoch {best.get('epoch')}"
          + (f", in-range R2 {best['accuracy']:.4f}"
             if SELECT == "survival" and "accuracy" in best else "")
          + f"); model saved to {MODEL_OUT}"
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

    if stopped_on_budget:
        # after everything is saved and logged: the best model so far is
        # valid, but the run is not finished, so the job must not count as
        # done (patience, EPOCHS and divergence stops keep exit 0)
        print(f"[budget] exiting with code {BUDGET_EXIT_CODE} so SLURM marks "
              f"this job failed and no afterok prediction starts on a model "
              f"that is not finished; resubmit to continue.", flush=True)
        raise SystemExit(BUDGET_EXIT_CODE)

if __name__ == "__main__":
    main()
