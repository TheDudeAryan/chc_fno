"""Autoregressive prediction and evaluation for the Cahn-Hilliard FNO."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import contextlib
import copy
import errno
import functools
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
import warnings
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.gridspec import GridSpec

import fno_gt
import fno_paths
import fno_train
import fno_runlog
from fno_train import (DT_STEP, ch_ground_truth, ch_ground_truth_batch,
                       free_energy, load_checkpoint, r2_score, rollout_frames)

WRITE_LOG = True
# None means the folders under Work/ and Results/.  --smoke makes it
# Work/smoke, and the results and caches a horizon or --model names go there.
ROOT: str | None = None

PRECISION = "fp32"
fno_train.PRECISION = PRECISION
# The low-k weight fill of the spectral layers (fno_train.LOWK_FILL).  None:
# as the checkpoint's config says (off for a checkpoint made before the key
# existed).  True / False: set on the loaded model for this prediction,
# whatever its config says.
LOWK_FILL: bool | None = None
# The clamp on psi before the mu channel of a model that has one
# (fno_train.MU_CLIP): a cubic channel feeds a one-pixel spike back as a much
# larger input, and the rollout can blow up (2026-10-24, the seed-1 checkpoint
# at the critical quench).  None: as the checkpoint's config says (none for a
# checkpoint made before the key existed).  A number: set on the loaded model
# for this prediction, whatever its config says; 1.2 for now, so the two
# checkpoints made before the key get it.  A model without a mu channel
# ignores it.
MU_CLIP: float | None = 1.2

# The model the old fno_ch.pt was: trained to t = 500.
DEFAULT_MODEL = "train_t0500"
MODEL_PATH = fno_paths.model_path(DEFAULT_MODEL)
WRITE_SUBMISSION = True
LABEL = "fno_ch"
AUTO_LABEL = True

EVAL_SEEDS_BASE = (99999, 54321, 77777, 13579, 24680,
                   31415, 65537, 80085, 90210, 12321)
N_EVAL_SEEDS = 50
EVAL_SEED_RNG = 20260909


def make_eval_seeds(n, base=EVAL_SEEDS_BASE, rng_seed=EVAL_SEED_RNG):
    """n seeds the model has never seen, the original ten first.

    ch.py draws its trajectories from [0, 10000) for train and [10000, 20000)
    for val, and every training-side seed set in fno_train is either below
    100000 or above 1000000, so drawing the extras from [100000, 1000000)
    cannot collide with anything the model was fitted on.

    The first ten are the set every earlier result in this project used.
    Keeping them, in order, means their ground truths and rollouts stay cached
    and the old numbers remain directly comparable to the new ones.
    """
    seeds = list(base[:n])
    seen = set(seeds)
    rng = np.random.default_rng(rng_seed)
    while len(seeds) < n:
        cand = int(rng.integers(100000, 1000000))
        if cand not in seen:
            seen.add(cand)
            seeds.append(cand)
    return tuple(seeds)


EVAL_SEEDS = make_eval_seeds(N_EVAL_SEEDS)

# The group protocol's 50 test seeds, so the two architectures can be scored
# on the same trajectories (--seed-set group).  Copied from the group's
# prediction script (its test seeds; its paper seeds, used for the box sizes,
# are the same list).
GROUP_SEEDS = (99999, 54321, 77777, 31415, 27182, 61803, 41421, 57721,
             66260, 86420, 20264, 22348, 22911, 27614, 27924, 28096,
             29935, 31458, 32393, 33052, 33672, 34617, 35064, 35973,
             36643, 36963, 39748, 42192, 43125, 48417, 51170, 52066,
             54464, 58713, 60079, 63866, 67517, 68620, 70389, 76145,
             76214, 77086, 84331, 85507, 88080, 88971, 92280, 94245,
             98029, 98895)
SEED_SETS = ("fno", "group")
# Which set EVAL_SEEDS is, for the manifest.
SEED_SET = "fno"

EVAL_MODE = "all"
CRITICAL_OFF = 0.0
DATASET_OFFS = (-0.4, -0.2, 0.0, 0.2, 0.4)
OFF_CRITICAL_OFFS = (-0.4, 0.4)

SWEEP_SEEDS = EVAL_SEEDS
SWEEP_CONSISTENT_SEEDS = True
# Every composition of eval_offsets() is run at every one of these sizes.
SIZE_SWEEP = (128, 256, 512)
# The box sizes of the protocol before 2026-11-09, removed on 2026-11-11.  Any
# record, summary or curve of them that is left in a cache is ignored.
DROPPED_SIZES = (160, 192)


def is_dropped(lattice):
    """True when ``lattice`` (an int, or None for the base box) is a box size
    in DROPPED_SIZES."""
    try:
        return lattice is not None and int(lattice) in DROPPED_SIZES
    except (TypeError, ValueError):
        return False


# When set, planned_cases() returns exactly this list of (off, lattice, seeds)
# and nothing else: a redraw plans the cases its meta.json holds, whatever the
# present settings would plan.
PLAN_ONLY: list | None = None

LATTICE = 128

START_TIME = 100.0
END_TIME = 2000.0
# One folder per run holds everything: the run-wide figures, manifest.json and
# README.md at the top, and a folder per case with every figure of that case,
# its averages over all seeds, and the best seed's own folder.  It is named
# for the model and the window, Results/train_t0500_pred_t02000.
_DEFAULT_RUN = fno_paths.run_label(fno_paths.model_label(MODEL_PATH),
                                   END_TIME, START_TIME)
RESULTS_DIR = fno_paths.results_dir(_DEFAULT_RUN)
RESULTS_TEMPLATE = None
R2_THRESHOLD = 0.80
EARLY_STOP = False

SNAPSHOT_TIMES = (100.0, 500.0, 1000.0, 2000.0)
DIAG_EVERY = 10
DIVERGE_PSI = 5.0
R2_FLOOR = -1.0
SWEEP_MARKERS = 14
SWEEP_YMIN = None

GT_CACHE = fno_paths.gt_dir()
# Rollouts are cached only while a run goes, so a job stopped by a time limit
# resumes where it was; each case's files are deleted as soon as its outputs
# are written (KEEP_CACHE, --keep-cache, keeps them).
CACHE_RESULTS = True
RESULT_CACHE = fno_paths.rollout_dir(_DEFAULT_RUN)
# FNO's own outputs beyond the flat run folder (per-case summaries, CSVs,
# best-seed folders, film, diagnostics, bookkeeping) go to
# <RESULT_CACHE>/extras/, or to EXTRAS_ROOT when a redraw / regeneration sets
# it.  Cache scans and record loaders never look into it.
EXTRAS_NAME = "extras"
EXTRAS_ROOT: str | None = None
RUN_SOURCE = "fno_predict"
RUN_CACHE: str | None = None
# A film this run did not make is removed from extras/ (its rollouts are new,
# the old film is of other ones) -- unless the outputs are redrawn from
# records or a cache, which make none: make_submission sets this then.
KEEP_FILM = False
RECORDED_CHECKPOINT: str | None = None
# The device a run's rollouts took their steps on.  A redraw from records runs
# nothing, so make_submission gives it from the run's timing file.
RECORDED_DEVICE: str | None = None
_RUN_MODEL: dict = {}
KEEP_CACHE = False
# Unlike the temporary cache, a permanent compact record of every seed is
# kept: records/<case folder>/seed<N>.npz, beside meta.json in the cache's
# folder, which nothing here deletes.  A record holds a rollout as
# _save_result does, less the film frames.  Its size: the series come to about
# 0.08 MB, and the rest is the snapshots, 2 x l^2 x 8 bytes for each snapshot
# time.  With the usual four times that is 1.05 MB of frames at l = 128, so a
# record is about 1.1 MB before compression; at l = 256 it is 4.2 MB of frames
# and about 4.3 MB.  Smooth fields compress to less.  The film a record leaves
# out is about 190 frames x 2 x l^2 x 4 bytes: 25 MB at l = 128, 100 MB at
# l = 256.  --no-records turns records and meta.json off.
RECORDS = True
RECORDS_DIR = "records"
META_FILE = "meta.json"
# A case an earlier job finished -- same cache key, same seeds, the outputs
# this run wants -- is not rolled out again; REDO (--redo) does it anyway.
REDO = False
# A results folder whose manifest names another checkpoint is moved to
# Archive/superseded before the run writes anything; NO_SUPERSEDE
# (--no-supersede) stops the run instead.
NO_SUPERSEDE = False

# Two checks of the loaded model, made once per run and written to
# diagnostics.json in each results folder (--no-diagnostics skips them):
# a step on a field against a step on its 2 x 2 tiling, and the low-k
# increment gain.  They only read the model, and take no part in a rollout.
# The field is the first validation seed's critical quench, DIAG_TILE across,
# at t = DIAG_TILE_T; the gain is over the first DIAG_LOWK_SEEDS validation
# seeds at fno_train's VAL_LATTICE, up to the run's END_TIME (a fixed-step
# model: FIXED_STEP's VAL_T_END), unless DIAG_LOWK_L and DIAG_LOWK_T_END name
# others (--smoke does).  A dt-conditioned model is asked for a step of
# DIAG_TILE_DT in the tile test and of fno_train.LOWK_DT in the gain, as in the group protocol.
DIAGNOSTICS = True
DIAG_FILE = "diagnostics.json"
DIAG_TILE = 64
DIAG_TILE_T = 500.0
DIAG_TILE_DT = 10.0
DIAG_TILE_TOL = 1e-5
DIAG_LOWK_SEEDS = 3
DIAG_LOWK_L: int | None = None
DIAG_LOWK_T_END: float | None = None

INTERFACE_WIDTH = 2.0 ** 0.5
POROD_KM_RANGE = (2.0, 5.0)
POROD_MIN_DECADE = 1.2
LENGTH_KEY = "L_zero"
# The finite-size regime: L(t) past l / FINITE_SIZE, fewer than that many
# domains across the box, where coarsening stops following a power law.
FINITE_SIZE = 8.0

MIN_PSI_SPAN = 0.004

# Only the best seed keeps data of its own: a folder with its R2 curve, its
# numbers and its figures.  Every other seed enters through the averages.
PER_SEED_OUTPUT = True
PER_SEED_FIGURES = True
BEST_SEED_DIR = "best_seed{seed}"
MEAN_R2_FILE = "mean_r2.csv"
FIG_DPI = 170
PER_SEED_DPI = 130

WRITE_VIDEO = True
VIDEO_EVERY = 10

# How rollouts are computed.  "seed" is one seed at a time against a stored
# reference, as every run to date.  "lockstep" advances a batch of seeds
# together and streams the reference instead of storing it -- what makes
# END_TIME = 10000 or 20000 affordable (fno_predict --end-time chooses it).
# "schedule" is the engine of a dt-conditioned model (CONDITIONED, read from
# the checkpoint; chosen for it, and never for a fixed-step one): the model
# takes strides that grow with t (SCHED_*) and is scored against the 4.0-spaced
# references of fno_gt at the schedule's times, so a run to 10000 is about 250
# steps instead of 9900.
ENGINE = "seed"
CONDITIONED = False
# The strides: alpha * t clipped to [SCHED_DT_MIN, SCHED_DT_MAX] and rounded to
# the reference spacing (fno_gt.make_schedule).  Per quench, (alpha, dt_max)
# are tuned on TUNE_SEEDS, which are fno_train's validation seeds, less any the
# run also tests on (tune_seeds; --seed-set fno has one), over TUNE_ALPHAS x
# TUNE_DT_MAXS by the mean R2 at T/4, T/2, 3T/4 and T; the pick goes into
# meta.json and a resumed job keeps it.  Without TUNE (--no-tune) the SCHED_
# values are used.  The box sizes take the critical quench's pick.
SCHED_ALPHA = 0.20
SCHED_DT_MIN = 1.0
SCHED_DT_MAX = 40.0
TUNE = True
TUNE_ALPHAS = (0.10, 0.20, 0.35, 0.50, 0.70)
TUNE_DT_MAXS = (30.0, 40.0, 45.0, 50.0)
TUNE_SEEDS = fno_gt.VAL_SEEDS
# Free space kept back when the reference cache is filled, in bytes.
CACHE_MARGIN = 2e9
# In a Slurm job with FNO_GT_CACHE unset the cache is in the project folder on
# /iitjhome, which has a 200 GB hard per-user quota that disk_usage does not
# show.  A need above this limit (half the quota, the rest is for models and
# results) is refused there.
HOME_CACHE_LIMIT = 100e9
# Times the schedule must visit: the snapshots and every view.  The snapshots
# follow the group protocol: its main set (100, 500, 1000, 2000), every multiple
# of SCHEDULE_LONG_EVERY up to the end (2000, 4000, ..., 10000 for a run to
# 10000, on to 18000 and 20000 for one to 20000), and 5000, where R2 is
# reported.
SCHEDULE_SNAPSHOTS = (100.0, 500.0, 1000.0, 2000.0, 5000.0)
SCHEDULE_LONG_EVERY = 2000.0
# what a fixed-step model was validated on (the diagnostics read it)
FIXED_STEP: dict[str, Any] = fno_train.FIXED_STEP
# seeds per chunk of a rollout: ROLLOUT_CELLS // l^2 (10 at l = 128)
ROLLOUT_CELLS = 10 * 128 * 128
# None: fno_paths.ref_cache_dir (FNO_GT_CACHE, else Work/gt_cache)
REF_CACHE: str | None = None
EVAL_BATCH = 50
# Seed engine: missing references are solved this many at a time (--gt-batch);
# a batch is held in memory whole, 1 GB a seed at l = 256, so it is also
# kept within GT_BATCH_GB.
GT_BATCH = 16
GT_BATCH_GB = 4.0
FNO_SUB_BATCH = None
HOST_WORKERS = max(1, min(8, os.cpu_count() or 1))
# Outputs for several horizons from one rollout to the longest, e.g.
# (2000, 10000, 20000); each goes to RESULTS_TEMPLATE with {H} filled in.
VIEWS = None
# "auto": log time axes when the window spans more than 20x (not at 100-2000).
LOG_TIME = "auto"
CACHE_END_TIME = None
# With views, every view's film cadence: ((horizon, every), ...).  Frames are
# recorded whenever any view still running needs one, so the t = 2000 view of
# a run to 20000 gets the same 191-frame film a run to 2000 would.
VIDEO_SCHEDULE = None
VIDEO_DTYPE = np.float32
SNAPSHOTS_EXPLICIT = False
VIDEO_FPS = 20
VIDEO_DPI = 120


def psi_to_a(psi):
    """Order parameter to the A fraction, in percent.

    psi = c_A - c_B with c_A + c_B = 1, the standard binary-mixture convention,
    so c_A = (1 + psi) / 2.  ch.py never names the species; this is the reading
    that makes psi_bar interpretable as a composition rather than an abstract
    number.  psi = 0 is 50/50, psi = +1 is pure A, psi = -1 is pure B.
    """
    return 50.0 * (1.0 + np.asarray(psi, dtype=np.float64))


def a_to_psi(a_percent):
    """Inverse of psi_to_a, for the secondary axis."""
    return np.asarray(a_percent, dtype=np.float64) / 50.0 - 1.0


WRITE_TIMING = True
TIMING_JSON = "{label}_timing.json"
TIMING_TEXT = "{label}_timing.txt"

MODEL_FLAGS = {"hard_constraints": True, "hard_mass_conservation": True,
               "zero_dc": True, "label": LABEL, "note": ""}

PHASE_CMAP, ERR_CMAP = "RdBu_r", "magma"
C_TRUE, C_PRED = "#1B7837", "#762A83"

D = 2
_trapz = getattr(np, "trapz", None) or np.trapezoid


@functools.lru_cache(maxsize=8)
def _k_bins(n):
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0)
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    dk = 2.0 * np.pi / n
    idx = np.clip(np.rint(np.sqrt(kx ** 2 + ky ** 2) / dk).astype(np.int64),
                  0, n // 2)
    counts = np.bincount(idx.ravel(), minlength=n // 2 + 1).astype(np.float64)
    return idx.ravel(), counts, np.arange(n // 2 + 1) * dk


@functools.lru_cache(maxsize=8)
def _r_bins(n):
    a = np.minimum(np.arange(n), n - np.arange(n))
    rx, ry = np.meshgrid(a, a, indexing="ij")
    idx = np.clip(np.rint(np.sqrt(rx ** 2 + ry ** 2)).astype(np.int64),
                  0, n // 2)
    counts = np.bincount(idx.ravel(), minlength=n // 2 + 1).astype(np.float64)
    return idx.ravel(), counts, np.arange(n // 2 + 1, dtype=np.float64)


def structure_factor(field):
    """Angle-averaged S(k,t) = |FFT(psi - <psi>)|^2 / N^2."""
    field = np.asarray(field, dtype=np.float64)
    n = field.shape[-1]
    psd = np.abs(np.fft.fft2(field - field.mean())) ** 2 / (n * n)
    idx, counts, centres = _k_bins(n)
    sums = np.bincount(idx, weights=psd.ravel(), minlength=len(centres))
    return centres, np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)


def correlation_function(field):
    """Angle-averaged C(r,t), normalised to C(0) = 1 as in Eq. 1.88."""
    field = np.asarray(field, dtype=np.float64)
    n = field.shape[-1]
    fft = np.fft.fft2(field - field.mean())
    corr = np.real(np.fft.ifft2(np.abs(fft) ** 2)) / (n * n)
    idx, counts, centres = _r_bins(n)
    sums = np.bincount(idx, weights=corr.ravel(), minlength=len(centres))
    c_r = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
    return centres, (c_r / c_r[0] if c_r[0] > 0 else c_r)


def length_from_zero_crossing(r, c_r):
    """L = first zero of C(r,t)."""
    sign = np.sign(c_r)
    cross = np.nonzero((sign[:-1] > 0) & (sign[1:] <= 0))[0]
    if len(cross) == 0:
        return float("nan")
    i = int(cross[0])
    c0, c1 = c_r[i], c_r[i + 1]
    return float(r[i] if c0 == c1
                 else r[i] + (r[i + 1] - r[i]) * c0 / (c0 - c1))


def length_from_half(r, c_r):
    """L = the first r where C(r,t) falls to half of C(0), interpolated
    linearly between the shells either side of it."""
    c_r = np.asarray(c_r, dtype=np.float64)
    if len(c_r) < 2 or not c_r[0] > 0:
        return float("nan")
    half = 0.5 * c_r[0]
    below = np.nonzero(c_r[1:] <= half)[0]
    if len(below) == 0:
        return float("nan")
    j = int(below[0]) + 1
    c0, c1 = c_r[j - 1], c_r[j]
    return float(r[j - 1] + (r[j] - r[j - 1]) * (c0 - half) / (c0 - c1))


def length_from_peak(k, s_k):
    """L = 2*pi/k_m from the peak of S(k,t), parabolically refined."""
    if len(s_k) < 3:
        return float("nan")
    i = int(np.argmax(s_k[1:]) + 1)
    if 0 < i < len(s_k) - 1:
        y0, y1, y2 = s_k[i - 1], s_k[i], s_k[i + 1]
        den = y0 - 2.0 * y1 + y2
        shift = 0.0 if den == 0 else np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0)
        km = k[i] + shift * (k[1] - k[0])
    else:
        km = k[i]
    return float(2.0 * np.pi / km) if km > 0 else float("nan")


def length_from_moment(k, s_k, k_cut=1.0):
    """L = 2*pi/<k> with the k^(d-1) radial measure and a cutoff."""
    m = (k > 0) & (k <= k_cut)
    if not np.any(m):
        return float("nan")
    kk, w = k[m], s_k[m] * k[m] ** (D - 1)
    den = _trapz(w, kk)
    if den <= 0:
        return float("nan")
    k_avg = _trapz(w * kk, kk) / den
    return float(2.0 * np.pi / k_avg) if k_avg > 0 else float("nan")


def length_from_gradient(psi):
    """L ~ 1/<|grad psi|^2>: inverse interfacial density."""
    psi = np.asarray(psi, dtype=np.float64)
    gx = np.roll(psi, -1, 0) - psi
    gy = np.roll(psi, -1, 1) - psi
    dens = float(np.mean(gx ** 2 + gy ** 2))
    return 1.0 / dens if dens > 0 else float("nan")


def all_lengths(psi):
    k, s_k = structure_factor(psi)
    r, c_r = correlation_function(psi)
    return {"L_zero": length_from_zero_crossing(r, c_r),
            "L_peak": length_from_peak(k, s_k),
            "L_moment": length_from_moment(k, s_k),
            "L_grad": length_from_gradient(psi),
            "L_half": length_from_half(r, c_r)}


def porod_window(k, s_k):
    """The k range where Porod's law can actually hold, or None if there is none.

    Porod gives S ~ k^-(d+1) = k^-3 in 2D for sharp interfaces, valid for
    kL >> 1 -- past the peak, not at the far right of the plot.  It is bounded
    above by the interface width: the equilibrium profile here is tanh(x/sqrt2),
    so w = sqrt(2) lattice units and the form factor of that diffuse interface
    takes over near k ~ 1/w = 0.71, beyond which S falls far faster than k^-3.

    Measured on this data, the compensated S(k)*k^3 is flattest between about
    2 and 5 times the peak wavenumber -- varying by only ~1.4x across an octave
    there, against slopes of -4.5 to -7 further out.  Because the peak moves as
    the domains coarsen (k_m = 0.245 -> 0.147 from t = 500 to 2000), a fixed
    band cannot stay in that regime, so the window is computed per snapshot.
    """
    km = k[1:][int(np.argmax(s_k[1:]))]
    lo = POROD_KM_RANGE[0] * km
    hi = min(POROD_KM_RANGE[1] * km, 1.0 / INTERFACE_WIDTH)
    return (lo, hi) if hi > POROD_MIN_DECADE * lo else None


def fit_growth_exponent(t, L, lo=None, hi=None):
    """Least-squares log L = n log t + c; Lifshitz-Slyozov gives n = 1/3.
    Given ``lo`` and ``hi``, only the times inside [lo, hi] are fitted."""
    t, L = np.asarray(t, dtype=np.float64), np.asarray(L, dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(L) & (t > 0) & (L > 0)
    if lo is not None:
        m &= t >= lo - 1e-6
    if hi is not None:
        m &= t <= hi + 1e-6
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(t[m]), np.log(L[m]), 1)[0])


# The time the model was trained to, which the case figures mark.  A redraw
# from records has no checkpoint to read it from: meta.json gives it.
RECORDED_HORIZON: float | None = None
_RUN_HORIZON: float | None = None


def _ckpt_horizon(ckpt):
    """t_train from a checkpoint's run_meta, or None."""
    meta = ckpt.get("run_meta") if isinstance(ckpt, dict) else None
    t = meta.get("t_train") if isinstance(meta, dict) else None
    return float(t) if isinstance(t, (int, float)) and t > 0 else None


def horizon_of(path):
    """The training horizon the train label of a checkpoint names, or None."""
    p = fno_paths.parse_label(fno_paths.model_label(path)) if path else None
    return None if p is None else p["t_train"]


def train_horizon():
    """The time the model was trained to: what the train label of MODEL_PATH
    says, else the checkpoint's run_meta, else what a redraw was given.  None
    when none of them knows, and then no figure marks it."""
    for t in (horizon_of(MODEL_PATH), _RUN_HORIZON, RECORDED_HORIZON):
        if t:
            return float(t)
    return None


# What the checkpoint says it was trained with, for the manifest.  A redraw
# from records has no checkpoint: meta.json gives it.
RECORDED_TRAIN: dict | None = None
_RUN_TRAIN: dict | None = None


def _ckpt_train(ckpt):
    """The training seed and model setup a checkpoint records, as {"seed",
    "channels", "modes", "laplacian_output", "mu_channel", "lowk_fill"}; None
    for what it does not say (a checkpoint from before the seed was recorded
    has no seed; one without the two options was made without them)."""
    cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    cfg = cfg if isinstance(cfg, dict) else {}
    made = ckpt.get("train_config") if isinstance(ckpt, dict) else None
    seed = made.get("seed") if isinstance(made, dict) else None
    if seed is None and isinstance(made, dict):
        seed = (made.get("constants") or {}).get("SEED")
    return {"seed": seed if isinstance(seed, int) else None,
            "channels": cfg.get("channels"), "modes": cfg.get("modes"),
            "laplacian_output": bool(cfg.get("laplacian_output", False)),
            "mu_channel": bool(cfg.get("mu_channel", False)),
            "lowk_fill": bool(cfg.get("lowk_fill", False))}


def train_setup():
    """What the model being rolled out was trained with (_ckpt_train), or
    what a redraw was given, or None."""
    return _RUN_TRAIN if _RUN_TRAIN is not None else RECORDED_TRAIN


def seed_series(results, get):
    """Every seed's series ``get(result)`` on the longest seed's diagnostic
    times, as (times, stack); a seed that ended early is padded with NaN."""
    longest = max(results, key=lambda r: len(r.get("diag_times", ())))
    t = np.asarray(longest.get("diag_times", ()), dtype=np.float64)
    stack = np.full((len(results), t.size), np.nan)
    for i, r in enumerate(results):
        v = np.asarray(get(r), dtype=np.float64)[:t.size]
        stack[i, :v.size] = v
    return t, stack


def seed_mean(stack):
    """The mean over the seeds still running at each time."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0)


def seed_sd(stack):
    """The sample standard deviation over the seeds still running at each
    time (ddof 1 as in the group protocol, zero where fewer than two are)."""
    stack = np.asarray(stack, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(stack, axis=0, ddof=1)
    return np.nan_to_num(sd)


def _length_of(r, which, key):
    return (r.get(which) or {}).get(key, ())


def length_stacks(results, key=None):
    """(times, truth, FNO): one estimator's L(t) of every seed, NaN-padded."""
    key = LENGTH_KEY if key is None else key
    t, st = seed_series(results, lambda r: _length_of(r, "L_true", key))
    _, sp = seed_series(results, lambda r: _length_of(r, "L_pred", key))
    return t, st, sp


def length_means(results, key=None):
    """(times, truth, FNO): one estimator's L(t) averaged over the seeds."""
    t, st, sp = length_stacks(results, key)
    return t, seed_mean(st), seed_mean(sp)


def finite_size_onset(t, L, lattice=None):
    """The first time L(t) is past the finite-size limit l / FINITE_SIZE, or
    None if it never gets there."""
    lat = LATTICE if lattice is None else lattice
    over = np.nonzero(np.nan_to_num(np.asarray(L, dtype=np.float64), nan=0.0)
                      > lat / FINITE_SIZE)[0]
    return float(np.asarray(t)[over[0]]) if over.size else None


def case_marks(results, lattice=None):
    """(training horizon, finite-size onset) to mark on the time axes of a
    case's figures; the onset is where the seeds' mean truth L(t) is there."""
    t, st, _ = length_stacks(results)
    return train_horizon(), finite_size_onset(t, seed_mean(st), lattice)


def mark_time(ax, t0, t1, horizon=None, onset=None, short=False, symbol="L"):
    """On a time axis that covers t0 to t1: a dashed vertical line at the
    training horizon, and the finite-size regime shaded from its onset on.
    Nothing is drawn for what is unknown or lies off the axis.  ``short``
    gives the legend entries without the numbers (the run folder figures,
    which keep the numbers in figure_notes.txt), and ``symbol`` names the
    length the shaded regime is measured with there."""
    if onset is not None and onset < t1:
        ax.axvspan(onset, t1, color="0.5", alpha=0.14, lw=0, zorder=0,
                   label=(f"${symbol} > l/{FINITE_SIZE:g}$" if short else
                          f"finite-size regime ($L > l/{FINITE_SIZE:g}$)"))
    if horizon is not None and t0 < horizon < t1:
        ax.axvline(horizon, color="0.2", ls="--", lw=1.4,
                   label=("training horizon" if short else
                          f"training horizon $t$ = {horizon:g}"))


def _exponents(t, mt, mp, lo, hi):
    return {"window": [float(lo), float(hi)],
            "n_truth": fit_growth_exponent(t, mt, lo, hi),
            "n_fno": fit_growth_exponent(t, mp, lo, hi)}


def growth_windows(t, mt, mp, horizon=None):
    """The growth exponent of a mean L(t) over all of it, and separately
    over [start, horizon] and [horizon, end].  The two are None unless the
    horizon is known and the curve runs past it (and has points to fit)."""
    t = np.asarray(t, dtype=np.float64)
    out: dict = {"exponent": None, "exponent_train": None,
                 "exponent_extrap": None}
    if not t.size:
        return out
    out["exponent"] = _exponents(t, mt, mp, t[0], t[-1])
    if horizon is not None and t[0] < horizon < t[-1]:
        for name, lo, hi in (("exponent_train", t[0], horizon),
                             ("exponent_extrap", horizon, t[-1])):
            got = _exponents(t, mt, mp, lo, hi)
            if np.isfinite(got["n_truth"]) or np.isfinite(got["n_fno"]):
                out[name] = got
    return out


def length_summary(results, lattice=None):
    """What a case's seed-mean length curves say, for its summary: the
    exponents over the whole rollout and over each side of the training
    horizon, where the finite-size regime begins, and the half-maximum length
    when the rollouts recorded it."""
    lat = LATTICE if lattice is None else lattice
    horizon = train_horizon()
    t, mt, mp = length_means(results)
    out = {"length": LENGTH_KEY, "train_horizon": horizon}
    out.update(growth_windows(t, mt, mp, horizon))
    out["finite_size"] = {"factor": FINITE_SIZE, "limit": lat / FINITE_SIZE,
                          "onset": finite_size_onset(t, mt, lat)}
    out["L_half"] = None
    if all("L_half" in (r.get("L_true") or {})
           and "L_half" in (r.get("L_pred") or {}) for r in results):
        th, ht, hp = length_means(results, "L_half")
        if th.size:
            out["L_half"] = dict(_exponents(th, ht, hp, th[0], th[-1]),
                                 L_end=float(ht[-1]), L_end_fno=float(hp[-1]))
    return out


def eval_offsets():
    """The composition offsets EVAL_MODE asks for.

    "none" evaluates no composition at the base lattice, leaving only the
    critical case at the sizes of SIZE_SWEEP (grid_cases).  That is what a
    top-up run wants: filling in missing seeds at one box size should not
    drag the base lattice along behind it, re-rendering figures that already
    exist from cached rollouts.
    """
    if EVAL_MODE == "none":
        return ()
    if EVAL_MODE == "all":
        return tuple(sorted(DATASET_OFFS))
    if EVAL_MODE == "critical":
        return (CRITICAL_OFF,)
    if EVAL_MODE == "off_critical":
        return tuple(sorted(OFF_CRITICAL_OFFS))
    if EVAL_MODE == "both":
        return tuple(sorted((CRITICAL_OFF,) + tuple(OFF_CRITICAL_OFFS)))
    raise ValueError('EVAL_MODE must be "all", "critical", "off_critical", '
                     f'"both" or "none", not {EVAL_MODE!r}')


def seeds_for(off):
    """The graded case gets every seed; the sweep compositions get a subset."""
    return EVAL_SEEDS if abs(off) < 1e-12 else SWEEP_SEEDS


def pick_result(results):
    """The best seed: the one realisation every single-seed panel, the film
    and the kept seed folder show.  Returns (index, rank, its mean R2).

    The seeds are independent realisations, so averaging their fields would
    give mush rather than a morphology; one has to be chosen, and it is the
    same one everywhere -- make_submission.best_index: the highest mean R2
    among the seeds that never diverged and have a prediction at the final
    time, then among those with a prediction there, then among all.  The
    rank is by mean R2 over every seed, so a best seed that is not the top
    mean (that one diverged) says so.
    """
    scores = np.array([float(np.nanmean(r["r2"])) for r in results])
    idx = int(_submission().best_index(results)) if len(results) > 1 else 0
    rank = int(np.sum(scores > scores[idx])) + 1
    return idx, rank, float(scores[idx])


def case_suffix(off, lattice=None):
    """Critical runs at the default lattice keep the plain filenames."""
    tag = "" if abs(off) < 1e-12 else f"_off{off:+.2f}"
    return tag if lattice is None else f"{tag}_l{lattice}"


def case_tag(off, lattice=None):
    """How one case is named in the log."""
    lat = LATTICE if lattice is None else lattice
    tag = "critical" if abs(off) < 1e-12 else f"off-critical {off:+.2f}"
    if lattice is not None:
        tag += f", l={lat}"
    if not MODEL_FLAGS["hard_constraints"]:
        tag += ", hard constraints OFF"
    return tag


def extras_for(results_dir, base, multi):
    """The extras folder of one results folder: ``base``, or with several
    views (one results folder each) ``base``/<results folder name>."""
    if not multi:
        return base
    return os.path.join(base, os.path.basename(os.path.normpath(results_dir)))


def extras_dir():
    """Where the folder being written keeps FNO's extras (see EXTRAS_NAME)."""
    base = (EXTRAS_ROOT if EXTRAS_ROOT is not None
            else os.path.join(RESULT_CACHE, EXTRAS_NAME))
    return extras_for(RESULTS_DIR, base, bool(VIEWS) and len(VIEWS) > 1)


def case_dir(off, lattice=None):
    """The case's folder in extras/: everything one case produces apart from
    the flat figures of the run folder.  Its name is the internal case name,
    which records and meta.json use (case_name)."""
    return os.path.join(extras_dir(), case_name(off, lattice))


def case_name(off, lattice=None):
    """The internal name of a case: psi0+0.20_l128, psi0+0.00_l256 ...  The
    records, meta.json and the extras folders are keyed by it; the flat layout's name for
    file names is make_submission.case_token.

    The critical quench at the base lattice is also the l = LATTICE point of
    the size sweep.  Run again on the sweep seeds (when SWEEP_SEEDS differs
    from the eval seeds) it is a different case, so it gets a folder of its
    own rather than overwriting the composition case's figures.  The fno_ch
    files keep the case suffix in their names, so they stay identifiable once
    copied out; the submission figures carry the case in their titles.
    """
    lat = LATTICE if lattice is None else lattice
    name = f"psi0{off:+.2f}_l{lat}"
    if (lattice is not None and lat == LATTICE
            and abs(off - CRITICAL_OFF) < 1e-12
            and tuple(SWEEP_SEEDS) != tuple(seeds_for(CRITICAL_OFF))):
        name += "_sweep_seeds"
    return name


def case_path(off, lattice, name):
    d = case_dir(off, lattice)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def seed_path(off, lattice, seed, name):
    d = os.path.join(case_dir(off, lattice), BEST_SEED_DIR.format(seed=seed))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def submission_dir():
    """Where the submission bookkeeping goes: extras_dir() (the figures are
    in RESULTS_DIR, flat).  It was RESULTS_DIR in the old layout.

    There is no separate Submission folder any more.  manifest.json,
    README.md and sweeps.png sit at the top of the results folder and each
    case's figures in its own folder.  An ablation (a model without the hard
    constraints) is kept apart by evaluate_all, which gives its whole folder
    a _nohard suffix -- the submission figures carry no label in their names.
    """
    return extras_dir()


def _submission():
    """make_submission, imported late and bound to this very module.

    make_submission does `import fno_predict`.  When this file is the script
    being run it is __main__, not fno_predict, so that import would load a
    second copy -- one that never saw the command-line overrides.  Registering
    the running module under its own name first makes them the same object.
    The import is late because the dependency is circular.
    """
    if __name__ == "__main__":
        sys.modules.setdefault("fno_predict", sys.modules[__name__])
    import make_submission
    return make_submission


def submission_name():
    """How the submission figures name the model, including an ablation."""
    name = _submission().DISPLAY_NAME
    return name if MODEL_FLAGS["hard_constraints"] else (
        f"{name} [hard constraints off]")


def submission_keys(off, lattice=None):
    """The bookkeeping entries a case writes, as (kind, key), including the
    l = LATTICE alias.

    A composition at the base lattice is ("compositions", "+0.20").  The
    critical quench at another size is ("sizes", "256"); any other
    composition at another size is ("grid", "+0.20_l256").  The critical
    quench at the base lattice doubles as the box size l = LATTICE, entry
    ("sizes", "128"), only when no run of its own on the sweep seeds
    (the twin, whose figures stay in extras/) takes that name: the same
    condition as before.  One function decides it, so the alias written and
    the entry recorded cannot disagree.
    """
    if lattice is not None:
        if abs(off - CRITICAL_OFF) < 1e-12:
            return [("sizes", str(lattice))]
        return [("grid", f"{off:+.2f}_l{lattice}")]
    keys = [("compositions", f"{off:+.2f}")]
    if (abs(off - CRITICAL_OFF) < 1e-12 and LATTICE in SIZE_SWEEP
            and tuple(seeds_for(CRITICAL_OFF)) == tuple(SWEEP_SEEDS)):
        keys.append(("sizes", str(LATTICE)))
    return keys


_RUN_STAMP = {}
_RUN_STAMPS = {}
_RUN_BEST = {}
_SUBMITTED = {}


# A redraw from records has no checkpoint to hash: the records name it.
RECORDED_FINGERPRINT: str | None = None


def model_fingerprint():
    """The fingerprint of MODEL_PATH; with no such file, the one a redraw
    from records was given, or None."""
    if not os.path.isfile(MODEL_PATH):
        return RECORDED_FINGERPRINT
    return fno_paths.fingerprint(MODEL_PATH)


# What a prediction's numbers depend on besides the checkpoint: the precision,
# whether TF32 is allowed, the low-k weight fill and the clamp of the mu
# channel (None: no clamp, or no channel).  The stamp goes into
# meta.json and into the results folder's manifest, and a run that finds
# another stamp puts those results aside as it does another checkpoint's.
# Every run before the stamp existed has none, and is read as OLD_NUMERICS:
# fno_predict set no precision flags then, so PyTorch's default ran cuDNN
# convolutions in TF32, and the weights were not filled; a stamp from before
# the mu clamp has none.
OLD_NUMERICS = {"precision": "fp32", "tf32": True, "lowk_fill": False,
                "mu_clip": None}
# A redraw from records or from a cache runs no model: the stamp they were
# made with.  None is a run that has a model.
RECORDED_NUMERICS: dict | None = None
_RUN_NUMERICS: dict = {}


def numerics_of(model=None):
    """The stamp of a prediction with ``model``.  LOWK_FILL, when it is set,
    decides the fill; else the model's layers do (a stand-in without any:
    its config).  The same for MU_CLIP and the model's mu clamp, which only
    counts for a model that has a mu channel (with no model there is none to
    clamp: None)."""
    cfg = getattr(model, "config", None) or {}
    if LOWK_FILL is not None:
        fill = LOWK_FILL
    elif isinstance(model, torch.nn.Module):
        fill = fno_train.lowk_fill_of(model)
    else:
        fill = cfg.get("lowk_fill", False)
    has_mu = (getattr(model, "mu_channel", False)
              if isinstance(model, torch.nn.Module)
              else cfg.get("mu_channel", False))
    if not has_mu:
        clip = None
    elif MU_CLIP is not None:
        clip = MU_CLIP
    elif isinstance(model, torch.nn.Module):
        clip = fno_train.mu_clip_of(model)
    else:
        clip = cfg.get("mu_clip")
    return {"precision": PRECISION, "tf32": PRECISION == "tf32",
            "lowk_fill": bool(fill),
            "mu_clip": None if clip is None else float(clip)}


def full_numerics(stamp):
    """A recorded stamp with every key; one that is missing (None), or that
    lacks a key, has OLD_NUMERICS' value there."""
    if not isinstance(stamp, dict):
        return dict(OLD_NUMERICS)
    return {k: stamp.get(k, v) for k, v in OLD_NUMERICS.items()}


def numerics_now():
    """The stamp of the run in hand: evaluate_all fixes it from its model (or
    from RECORDED_NUMERICS); before that, from the settings alone."""
    return dict(_RUN_NUMERICS) if _RUN_NUMERICS else numerics_of()


def numerics_text(stamp):
    clip = stamp.get("mu_clip")
    return (f"{stamp['precision']}, TF32 {'on' if stamp['tf32'] else 'off'}, "
            f"low-k fill {'on' if stamp['lowk_fill'] else 'off'}"
            + ("" if clip is None else f", mu clip {clip:g}"))


def numerics_tag(stamp):
    """The stamp as a short word for a folder name."""
    return (stamp["precision"]
            + ("_tf32" if stamp["tf32"] and stamp["precision"] != "tf32"
               else "")
            + ("_lowk" if stamp["lowk_fill"] else "")
            + ("" if stamp.get("mu_clip") is None
               else f"_clip{stamp['mu_clip']:g}"))


def _mtime_text(st):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))


def run_stamp(ckpt=None):
    """Which checkpoint, and which run, wrote a submission entry.

    main() fixes it once per run from the checkpoint it loaded, so every
    entry one run writes carries the same time.  Called without a checkpoint
    before that -- write_case_outputs driven directly, as a test does -- it
    is made on first use with the checkpoint details left empty.
    """
    here = os.path.abspath(RESULTS_DIR)
    if ckpt is not None:
        _RUN_STAMPS.clear()
        _RUN_BEST.clear()
        if isinstance(ckpt, dict):
            _RUN_BEST.update(ckpt.get("best", {}))
    if here not in _RUN_STAMPS:
        st = os.stat(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        best = _RUN_BEST
        _RUN_STAMP.clear()
        _RUN_STAMP.update(
            written=time.strftime("%Y-%m-%dT%H:%M:%S"),
            model_path=MODEL_PATH,
            model_bytes=st.st_size if st else None,
            model_mtime=_mtime_text(st) if st else None,
            fingerprint=model_fingerprint(), numerics=numerics_now(),
            seed_set=SEED_SET,
            epoch=best.get("epoch"), rollout_score=best.get("score"),
            results_dir=os.path.abspath(RESULTS_DIR),
            window={"start": START_TIME, "end": END_TIME,
                    "snapshots": sorted(float(t) for t in SNAPSHOT_TIMES
                                        if t <= END_TIME)})
        _RUN_STAMPS[here] = dict(_RUN_STAMP)
    return dict(_RUN_STAMPS[here])


def write_submission_case(results, seeds, off, lattice=None, key=None):
    """This case's submission figures, in its folder, and its manifest entries.

    Runs while the rollouts are in memory, like the rest of
    write_case_outputs, so the run never holds more than one case.  The
    critical quench at the base lattice is also the l = LATTICE box size when
    submission_keys says so; both manifest entries then name the same folder.

    ``key`` is the rollouts' cache key.  run_case passes the one it worked out
    outside any view (inside one it differs); without it the key is worked
    out here, as for a plain run.

    The entries and the case's sweep curve are recorded as soon as the files
    are in place, so a run that dies later still leaves a manifest that
    matches the figures.  Once the files are published the case has
    succeeded: a failure to record it (a locked sweep_curves.npz, say) is
    reported but not raised, and write_submission records it again at the
    end of the run from what this returns.

    Returns (entries, curve record).
    """
    S = _submission()
    root = submission_dir()
    lat = LATTICE if lattice is None else lattice
    keys = submission_keys(off, lattice)
    if key is None and os.path.exists(MODEL_PATH):
        key = _cache_key(off, lat)
    folder = case_dir(off, lattice)
    # the twin of the critical case on other seeds has no flat-layout counterpart:
    # its figures stay in extras/
    twin = lattice is not None and lat == LATTICE
    token = S.case_token(off, lattice) + ("_sweep_seeds" if twin else "")
    stem = S.comp_slug(off) + ("_sweep_seeds" if twin else "")
    print(f"\n[submission] {case_tag(off, lattice)}: {len(results)} seeds "
          f"-> {RESULTS_DIR}/l{lat} ({token}), extras {folder}", flush=True)
    S.recover_run(RESULTS_DIR, root)
    entry, curve = S.write_case(results, seeds, off, lat,
                                folder if twin else RESULTS_DIR, key=key,
                                name=submission_name(), videos=WRITE_VIDEO,
                                root=RESULTS_DIR, extras=folder, token=token,
                                keep_film=KEEP_FILM, stem=stem,
                                sizedir="" if twin else None)
    stamp = run_stamp()
    entries = {k: {**entry, "run": stamp} for k in keys}
    record = S.curve_record(curve)
    for k, e in entries.items():
        _SUBMITTED.setdefault(os.path.abspath(root), {})[k] = (e, record)
    try:
        S.record_cases(root, entries, curve)
    except Exception as exc:
        print(f"[submission] WARNING: the figures for {case_tag(off, lattice)}"
              f" are published, but recording them failed ({type(exc).__name__}"
              f": {exc}); the end of the run records them again", flush=True)
        traceback.print_exc()
    return entries, record


def write_submission(cases, failed, failed_keys=()):
    """The run-level half of the submission folder: bookkeeping and sweep.

    Each case recorded its own manifest entry and sweep curve the moment its
    files were published (write_submission_case).  This records them all
    again -- a backstop, so one that failed to record still ends up
    documented.  What was published is taken from _SUBMITTED, filled the
    moment each case's files are in place, so a case that failed after its
    publish (its summary on a full disk, say) is not lost.  Then it
    does what needs every case at once.

    A case whose figures failed this run keeps its previous files and entry
    (write_case replaces them only when all succeed) and the entry is marked
    last_attempt_failed; so is one whose rollouts failed (failed_keys).  An
    entry this run did write is never marked.  The sweep is redrawn from
    every curve the folder holds, since one drawn from this run alone would
    lose panels the manifest still lists after a top-up -- but only when
    every documented case has a curve.  A case recorded before the curve
    store existed has none, and a sweep without it would silently drop a
    panel, so then the old sweeps.png is kept and the manifest's "sweep"
    says what is missing.  A curve with no entry (after a manifest had to be
    restarted) is drawn, its files being on disk, and listed as
    undocumented.  The outcome, a failed redraw included, is written into
    the manifest before any error goes further.

    Folder-level fields (snapshot times, start, final time) are worked out
    from every entry, not this run's, and a checkpoints summary says which
    model wrote which entries.  One folder takes one job at a time.
    """
    S = _submission()
    out = submission_dir()
    os.makedirs(out, exist_ok=True)
    S.recover_run(RESULTS_DIR, out)
    stamp = run_stamp()

    written = dict(_SUBMITTED.pop(os.path.abspath(out), {}))
    marked = set(failed_keys)
    for c in {id(c): c for c in cases if c is not None}.values():
        for k, entry in c.get("submission", {}).items():
            written[k] = (entry, c.get("submission_curve"))
        marked.update(c.get("submission_failed_keys", ()))
    marked -= set(written)

    stored = S.load_curves(out)
    fresh = {k: rec for k, (_, rec) in written.items() if rec is not None}
    curves = {**stored, **fresh}
    stale = []
    if fresh:
        try:
            S.save_curves(out, curves)
        except Exception as exc:
            stale = sorted(k for k, rec in fresh.items()
                           if k not in stored
                           or S.curve_digest(stored[k]) != S.curve_digest(rec))
            if stale:
                print(f"[submission] WARNING: could not store this run's "
                      f"sweep curves ({type(exc).__name__}: {exc}); this sweep "
                      f"is drawn from memory, and a later run will not redraw "
                      f"it until those cases are stored", flush=True)

    manifest = S.load_manifest(out)
    for (kind, k), (entry, _) in written.items():
        manifest[kind][k] = entry
    documented = {(kind, k) for kind in S.KINDS
                  for k in manifest[kind]}
    # A stored curve counts only if it is the one the entry was drawn with:
    # after a failed save the store can hold an older run's curve.
    def has_curve(kind, k):
        c, e = curves.get((kind, k)), manifest[kind][k]
        digest = e.get("curve_digest") if isinstance(e, dict) else None
        if digest is None and isinstance(e, dict) and e.get("curve_not_stored"):
            return False
        return c is not None and (digest is None
                                  or S.curve_digest(c) == digest)

    missing = sorted(f"{kind} {k}" for kind, k in documented
                     if not has_curve(kind, k))
    sweep: dict = {"cases": sorted(f"{kind} {k}" for kind, k in curves),
                   "missing_curves": missing,
                   "undocumented_curves": sorted(
                       f"{kind} {k}" for kind, k in set(curves) - documented)}
    if stale:
        sweep["not_stored"] = sorted(f"{kind} {k}" for kind, k in stale)
    error = None
    if missing:
        sweep["status"] = (f"not redrawn at {stamp['written']}: no stored "
                           f"curve for {len(missing)} documented case(s), so "
                           f"the images of r2_sweep_per_size/ and "
                           f"r2_sweep_per_case/ and the r2_time.png of "
                           f"their sizes keep their old files")
        print(f"[submission] the images of r2_sweep_per_size/ and "
              f"r2_sweep_per_case/ and the r2_time.png of the sizes of "
              f"{', '.join(missing)} NOT redrawn: no stored curve, and a "
              f"figure without them would drop lines.  Rerun the prediction "
              f"over those cases (cached "
              f"rollouts make it quick) to fill them in.",
              flush=True)
    elif curves:
        sweep["status"] = (f"r2_sweep drawn from the stored curves at "
                           f"{stamp['written']}")
    else:
        sweep["status"] = "nothing to draw"

    restarted = manifest.get("restarted")
    manifest = S.load_manifest(out)
    if restarted and "restarted" not in manifest:
        manifest["restarted"] = restarted
    for (kind, k), (entry, _) in written.items():
        manifest[kind][k] = entry
    for kind, k in sorted(marked):
        e = manifest[kind].get(k)
        if (isinstance(e, dict)
                and (e.get("run") or {}).get("written") != stamp["written"]):
            e["last_attempt_failed"] = stamp["written"]
    build_keys = {k: manifest.pop(k) for k in
                  ("cache", "failed", "rebuild_in_progress", "records")
                  if k in manifest}
    if build_keys:
        manifest["last_build"] = build_keys
    manifest.pop("missing", None)

    entries = [e for kind in S.KINDS
               for e in manifest[kind].values() if isinstance(e, dict)]
    ft = S.folder_times(entries)
    if ft["snapshot_times"] is None:
        ft["snapshot_times"] = sorted(float(t) for t in SNAPSHOT_TIMES
                                      if t <= END_TIME)
        ft["pdf_time"] = (ft["snapshot_times"][-1] if ft["snapshot_times"]
                          else None)
        ft["rollout_start"] = START_TIME
    if not ft["times_agree"]:
        print(f"[submission] WARNING: the cases in {out} were rolled out over "
              f"different times (start {ft['rollout_start']}, final "
              f"{ft['pdf_time']}); manifest.json lists them", flush=True)

    by_model, by_print, unknown = {}, {}, []
    for kind in S.KINDS:
        for k, e in manifest[kind].items():
            run = (e.get("run") or {}) if isinstance(e, dict) else {}
            if run.get("fingerprint"):
                by_print.setdefault(run["fingerprint"], []).append(
                    f"{kind} {k}")
            if run.get("model_mtime"):
                by_model.setdefault(run["model_mtime"], []).append(
                    f"{kind} {k}")
            else:
                unknown.append(f"{kind} {k}")
    manifest.update(source="fno_predict", lattice=LATTICE, **ft,
                    last_run=stamp, failed_last_run=list(failed),
                    sweep=sweep, seed_set=SEED_SET,
                    train_horizon=train_horizon(),
                    train_setup=train_setup(),
                    checkpoints={"by_model_mtime": by_model,
                                 "by_fingerprint": by_print,
                                 "unknown": unknown,
                                 "mixed": (len(by_model) + bool(unknown) > 1
                                           or len(by_print) > 1)})
    path = os.path.join(out, S.BOOKKEEPING)
    S.write_json(path, manifest)
    print(f"[saved] {path}  ({len(written)} case entries from this run; "
          f"{len(manifest['compositions'])} compositions at the base lattice, "
          f"{len(manifest['sizes'])} critical cases at other sizes and "
          f"{len(manifest['grid'])} other compositions at other sizes in "
          f"total)", flush=True)
    if manifest["checkpoints"]["mixed"]:
        print(f"[submission] WARNING: {out} holds cases from "
              f"{len(by_model)} checkpoint(s)"
              + (f" and {len(unknown)} of unknown origin (rebuilt from a "
                 f"cache by make_submission)" if unknown else "")
              + "; manifest.json 'checkpoints' says which", flush=True)
    # the top of the run folder, from every case the folder holds: the flat
    # layout's composition.csv, composition_all.png, the two r2_sweep folders,
    # manifest.json and report.txt; a failure is raised at the end
    try:
        S.write_run_outputs(RESULTS_DIR, out, run_facts())
    except Exception as exc:
        error = error or exc
        print(f"[submission] WARNING: the run folder's top level was not "
              f"completely written ({type(exc).__name__}: {exc})", flush=True)
        # the failure goes into the bookkeeping as well, as every part that
        # failed does, before it goes further
        manifest["failed_last_run"] = list(failed) + [str(exc)]
        if "r2_sweep" in str(exc):
            sweep["status"] = f"FAILED at {stamp['written']}: {exc}"
        try:
            S.write_json(path, manifest)
        except Exception as again:
            print(f"[submission] WARNING: and {path} could not record it "
                  f"({type(again).__name__}: {again})", flush=True)
    if error is not None:
        raise error
    return manifest


class _TimedModel(torch.nn.Module):
    """Wraps the model so rollout_frames' per-step calls can be timed.

    Timing the whole evaluate_seed loop would charge the model for R2, the
    structure factor, the length estimators and the free energy, none of which
    a deployed model would compute.  This counts only the forward passes.
    CUDA is asynchronous, so each call is bracketed by a synchronise; that adds
    overhead of its own, which makes the number reported here a slight
    over-estimate of the model's cost rather than a flattering one.
    """

    def __init__(self, model, device):
        super().__init__()
        self.inner = model
        self.sync = torch.device(device).type == "cuda"
        self.seconds = 0.0
        self.calls = 0

    def forward(self, x):
        if self.sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.inner(x)
        if self.sync:
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return out


def _gt_cache_path(l, seed, off, t_end):
    """Mirrors fno_train.ch_ground_truth's cache filename.

    Only used to tell a solve from a cache hit before the call, so the timing
    can say which it measured.  Kept in step with fno_train by hand.
    """
    return os.path.join(GT_CACHE,
                        f"gt_l{l}_seed{seed}_off{off:+.2f}_t{t_end:g}_f64.npz")


_RUN_SNAPSHOTS = ()


def default_snapshots(start, end):
    """Snapshot times for a horizon: today's for t <= 2000, 1-2-5 beyond.

    Always the input frame and the final time -- the submission figures judge
    a case at its last snapshot, so a horizon without one would be judged
    early.  2000 gives (100, 500, 1000, 2000) as always; 10000 gives
    (100, 1000, 2000, 5000, 10000) and 20000 adds 20000.  The schedule engine
    takes the group protocol's set instead (SCHEDULE_SNAPSHOTS and every multiple of
    SCHEDULE_LONG_EVERY, inside the horizon): (100, 500, 1000, 2000, 4000,
    5000, 6000, 8000, 10000) for 10000, and 12000 to 18000 more for 20000.
    """
    if ENGINE == "schedule":
        n_long = int(end / SCHEDULE_LONG_EVERY + 1e-9)
        base = [t for t in (*SCHEDULE_SNAPSHOTS,
                            *(k * SCHEDULE_LONG_EVERY
                              for k in range(1, n_long + 1)))
                if start < t < end]
        return tuple(sorted({float(start), float(end), *base}))
    if end <= 2000.0 + 1e-9:
        base = [t for t in (100.0, 500.0, 1000.0, 2000.0) if start <= t <= end]
    else:
        base = [float(c * 10 ** k) for k in range(3, 9) for c in (1, 2, 5)
                if start < c * 10 ** k < end]
    return tuple(sorted({float(start), float(end), *base}))


def default_video_every(start, end):
    """A film of about 190 frames whatever the horizon: 10 at 2000, 100 at
    20000 -- so a long run's frames cost no more memory than today's."""
    return max(10, int(round((end - start) / 1900.0)) * 10)


def _video_frame(k, t):
    """Whether model step k (at time t) records a film frame."""
    if k == 0:
        return True
    if VIDEO_SCHEDULE:
        return any((k + 1) % e == 0 for h, e in VIDEO_SCHEDULE
                   if t <= h + 0.5 * DT_STEP)
    return (k + 1) % VIDEO_EVERY == 0


def log_time():
    """Whether time axes are logarithmic: long windows, or LOG_TIME on."""
    if LOG_TIME == "on":
        return True
    if LOG_TIME == "off":
        return False
    return END_TIME / max(START_TIME, 1e-9) > 20.0 + 1e-9


def _time_axis(ax):
    """The rollout window on a time axis, log-scaled for long horizons."""
    ax.set_xlim(START_TIME, END_TIME)
    if log_time():
        ax.set_xscale("log")


def _cache_key(off, lattice):
    """Everything that changes what a cached rollout contains."""
    st = os.stat(MODEL_PATH)
    end = END_TIME if CACHE_END_TIME is None else CACHE_END_TIME
    snaps = SNAPSHOT_TIMES if CACHE_END_TIME is None else _RUN_SNAPSHOTS
    key = (st.st_size, int(st.st_mtime), lattice, off, START_TIME, end,
           EARLY_STOP, R2_THRESHOLD, DIAG_EVERY, DIVERGE_PSI, R2_FLOOR,
           tuple(sorted(snaps)))
    if ENGINE == "schedule":
        # what shapes the schedule, not the pick itself (it is in meta.json
        # and checked against each cached rollout)
        must = {float(t) for t in snaps} | {float(h) for h in VIEWS or ()}
        key += ("engine", ENGINE, SCHED_DT_MIN, fno_gt.GT_EVERY,
                tuple(sorted(must | {float(end)})), tune_signature(end))
    elif ENGINE != "seed" or VIDEO_EVERY != 10:
        # Only for runs that differ from every run so far, so that no
        # existing cache stops matching: a lock-step rollout is not bitwise
        # the per-seed one (batched convolutions), and the film cadence
        # decides which frames were kept.
        key += ("engine", ENGINE, "video_every", VIDEO_EVERY,
                "write_video", WRITE_VIDEO)
        if VIDEO_SCHEDULE:
            key += ("video_schedule", tuple(VIDEO_SCHEDULE),
                    "video_dtype", np.dtype(VIDEO_DTYPE).name)
    num = numerics_now()
    if num != OLD_NUMERICS:
        # a rollout made in other numerics is another rollout, though the
        # checkpoint is the same; the old numerics keep the keys they had
        # (a stamp with no mu clamp keeps the key it had before there was one)
        key += ("numerics", tuple(sorted(
            (k, v) for k, v in num.items()
            if not (k == "mu_clip" and v is None))))
    return hashlib.md5(repr(key).encode()).hexdigest()[:12]


def _result_path(off, seed, lattice):
    label = MODEL_FLAGS["label"]
    return os.path.join(RESULT_CACHE, f"{label}{case_suffix(off)}_l{lattice}"
                                      f"_seed{seed}_{_cache_key(off, lattice)}"
                                      f".npz")


def _result_arrays(res, film=True):
    """A rollout as the flat arrays of one npz, each as the result holds it.

    The cache keeps the film frames; a record (film=False) does not.
    """
    order = sorted(res["snapshots"])
    flat = {"times": res["times"], "r2": res["r2"],
            "mass_drift": res["mass_drift"],
            "mean_true": res["mean_true"], "mean_pred": res["mean_pred"],
            "diag_times": res["diag_times"],
            "energy_true": res["energy_true"],
            "energy_pred": res["energy_pred"],
            "t_stop": np.asarray(res["t_stop"]),
            "gt_seconds": np.asarray(res["gt_seconds"]),
            "gt_cached": np.asarray(res["gt_cached"]),
            "gt_steps": np.asarray(res["gt_steps"]),
            "fno_seconds": np.asarray(res["fno_seconds"]),
            "fno_steps": np.asarray(res["fno_steps"]),
            "stopped_early": np.asarray(res["stopped_early"]),
            "diverged_at": np.asarray(np.nan if res.get("diverged_at") is None
                                      else res["diverged_at"]),
            "snap_t": np.asarray(order, dtype=np.float64)}
    if res.get("t_start") is not None:
        # the time the rollout was seeded at, for a schedule that is not a
        # fixed step (a result without one starts one spacing before times[0])
        flat["t_start"] = np.asarray(float(res["t_start"]))
    for k, v in (res.get("schedule") or {}).items():
        flat[f"sched_{k}"] = np.asarray(v)
    if film and res.get("video_true") is not None and len(res["video_t"]):
        flat["vid_t"] = np.asarray(res["video_t"], dtype=np.float64)
        flat["vid_true"] = res["video_true"]
        flat["vid_pred"] = res["video_pred"]
    if order:
        flat["snap_true"] = np.stack([res["snapshots"][t][0] for t in order])
        flat["snap_pred"] = np.stack([res["snapshots"][t][1] for t in order])
    for name, arr in res["L_true"].items():
        flat[f"Ltrue_{name}"] = arr
    for name, arr in res["L_pred"].items():
        flat[f"Lpred_{name}"] = arr
    return flat


def _write_npz(path, flat, compress=False):
    """The arrays to ``path`` under a temporary name, then renamed over it, so
    a job killed mid-write never leaves a truncated file under the real one."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as fh:
            if compress:
                np.savez_compressed(fh, **flat)
            else:
                np.savez(fh, **flat)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _save_result(path, res):
    _write_npz(path, _result_arrays(res))


def has_start_sample(res):
    """Whether a schedule result's series begin at its t_start: the state it
    was seeded with is the reference, so its first sample is R2 = 1, as the
    group's curves have.  A result made before that (its first sample is the end of
    the first stride), a fixed-step one and a hand-made one have none."""
    t0, t = res.get("t_start"), res.get("times")
    return bool(res.get("schedule") and t0 is not None and t is not None
                and len(t) and float(t[0]) <= float(t0) + 1e-6)


def with_start_sample(res: dict[str, Any]) -> dict[str, Any]:
    """``res``, with the start sample put in front of the series of a
    schedule result made before it was recorded: t_start, R2 = 1, no drift,
    and the mean of the truth for both means (the state is the reference; the
    truth's mean does not move).  A result that has it, or is not a schedule
    result, comes back as it is, so nothing is counted twice."""
    if (not res.get("schedule") or res.get("t_start") is None
            or not len(res["times"]) or has_start_sample(res)):
        return res

    def front(a, v):
        a = np.asarray(a)
        return np.concatenate([np.asarray([v], dtype=a.dtype), a])

    out = dict(res)
    out["times"] = front(res["times"], float(res["t_start"]))
    out["r2"] = front(res["r2"], 1.0)
    out["mass_drift"] = front(res["mass_drift"], 0.0)
    if res.get("mean_true") is not None and res.get("mean_pred") is not None:
        mean0 = float(np.asarray(res["mean_true"])[0])
        out["mean_true"] = front(res["mean_true"], mean0)
        out["mean_pred"] = front(res["mean_pred"], mean0)
    return out


def n_steps_above(res):
    """How many steps of a rollout have R2 at or above the threshold (the
    start sample of a schedule result is no step)."""
    r2 = np.asarray(res["r2"])[int(has_start_sample(res)):]
    return int(np.sum(r2 >= R2_THRESHOLD))


def _result_from(z):
    order = z["snap_t"]
    snaps = {float(t): (z["snap_true"][i], z["snap_pred"][i])
             for i, t in enumerate(order)} if len(order) else {}
    res: dict[str, Any] = {"times": z["times"], "r2": z["r2"],
            "mass_drift": z["mass_drift"],
            "mean_true": (z["mean_true"] if "mean_true" in z.files
                          else None),
            "mean_pred": (z["mean_pred"] if "mean_pred" in z.files
                          else None),
            "diag_times": z["diag_times"],
            "energy_true": z["energy_true"],
            "energy_pred": z["energy_pred"],
            "t_stop": float(z["t_stop"]),
            "gt_seconds": (float(z["gt_seconds"])
                           if "gt_seconds" in z.files else None),
            "gt_cached": (bool(z["gt_cached"])
                          if "gt_cached" in z.files else None),
            "gt_steps": (int(z["gt_steps"])
                         if "gt_steps" in z.files else None),
            "fno_seconds": (float(z["fno_seconds"])
                            if "fno_seconds" in z.files else None),
            "fno_steps": (int(z["fno_steps"])
                          if "fno_steps" in z.files else None),
            "diverged_at": (None if "diverged_at" not in z.files
                            or not np.isfinite(z["diverged_at"])
                            else float(z["diverged_at"])),
            "stopped_early": bool(z["stopped_early"]),
            "t_start": (float(z["t_start"]) if "t_start" in z.files
                        else None),
            "schedule": ({k[6:]: z[k].item() for k in z.files
                          if k.startswith("sched_")} or None),
            "L_true": {k[6:]: z[k] for k in z.files
                       if k.startswith("Ltrue_")},
            "L_pred": {k[6:]: z[k] for k in z.files
                       if k.startswith("Lpred_")},
            "video_t": (z["vid_t"] if "vid_t" in z.files else None),
            "video_true": (z["vid_true"] if "vid_true" in z.files
                           else None),
            "video_pred": (z["vid_pred"] if "vid_pred" in z.files
                           else None),
            "snapshots": snaps}
    return with_start_sample(res)


def _load_result(path):
    # Opened here, not by np.load, which on a damaged zip raises without
    # closing its handle -- on Windows that blocks moving the file aside.
    with open(path, "rb") as fh, np.load(fh) as z:
        return _result_from(z)


def records_dir(cache=None):
    """Where the permanent records are: in the rollout cache's own folder."""
    return os.path.join(RESULT_CACHE if cache is None else cache, RECORDS_DIR)


def record_file(cache, name, seed):
    return os.path.join(records_dir(cache), name, f"seed{int(seed)}.npz")


def record_path(off, lattice, seed):
    """The record of one seed of one case, named for the case's folder."""
    return record_file(RESULT_CACHE, os.path.basename(case_dir(off, lattice)),
                       seed)


def meta_file(cache=None):
    return os.path.join(RESULT_CACHE if cache is None else cache, META_FILE)


def save_record(off, lattice, seed, res, key):
    """One seed's permanent record: the rollout as the cache holds it, but
    for the film frames, and who made it -- seed, case, checkpoint
    fingerprint, cache key and the rollout's window -- so that a redraw needs
    nothing else.  Arrays keep the dtype the result holds them in."""
    lat = LATTICE if lattice is None else lattice
    end = END_TIME if CACHE_END_TIME is None else CACHE_END_TIME
    flat = _result_arrays(res, film=False)
    flat.update(seed=np.asarray(int(seed)), off=np.asarray(float(off)),
                lattice=np.asarray(lat),
                fingerprint=np.asarray(model_fingerprint() or ""),
                cache_key=np.asarray(key or ""),
                t_start=np.asarray(float(START_TIME)),
                t_end=np.asarray(float(end)))
    _write_npz(record_path(off, lattice, seed), flat, compress=True)


def load_record(path):
    """(rollout, who made it) from a record."""
    with open(path, "rb") as fh, np.load(fh) as z:
        who = {"seed": int(z["seed"]), "off": float(z["off"]),
               "lattice": int(z["lattice"]),
               "fingerprint": str(z["fingerprint"]) or None,
               "cache_key": str(z["cache_key"]) or None,
               "t_start": float(z["t_start"]), "t_end": float(z["t_end"])}
        return _result_from(z), who


def keep_record(off, lattice, seed, res, key):
    """The seed's record, unless RECORDS is off.  It is a by-product, so a
    failure to write it (a full quota) is reported and the case goes on; the
    case is then not complete in meta.json."""
    if not RECORDS:
        return
    try:
        save_record(off, lattice, seed, res, key)
    except Exception as exc:
        print(f"  -> could not keep the record of seed {seed} "
              f"({type(exc).__name__}: {exc}); the case is drawn anyway, "
              f"but a redraw from records will not have this seed",
              flush=True)


def _plain(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


def read_meta(cache):
    """meta.json of a records folder, or None if missing or unreadable."""
    try:
        with open(meta_file(cache), encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None
    ok = isinstance(meta, dict) and isinstance(meta.get("cases"), dict)
    return meta if ok else None


def _write_meta(meta):
    meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(RESULT_CACHE, exist_ok=True)
    _write_json_atomic(meta_file(), meta)


def grid_cases():
    """The cases the present settings call for, as (off, lattice, seeds):
    every composition of eval_offsets() at the base lattice (lattice None,
    seeds_for(off)), then at every other size of SIZE_SWEEP (the sweep
    seeds), size by size.  When the sweep seeds differ from the eval seeds
    the critical case is also run at the base lattice on the sweep seeds
    (the twin), as before; no other composition has one.  EVAL_MODE "none"
    (a top-up) plans the critical case at the sizes alone."""
    offs = eval_offsets()
    sweep = tuple(SWEEP_SEEDS)
    if EVAL_MODE == "none":
        # a top-up run: the critical case at the sizes, nothing at the base
        return [(CRITICAL_OFF, lat, sweep) for lat in SIZE_SWEEP]
    plan: list[tuple[float, int | None, tuple]] = [
        (off, None, tuple(seeds_for(off))) for off in offs]
    for lat in SIZE_SWEEP:
        if lat != LATTICE:
            plan += [(off, lat, sweep) for off in offs]
    twin = CRITICAL_OFF in offs and tuple(seeds_for(CRITICAL_OFF)) != sweep
    if twin and (LATTICE in SIZE_SWEEP or SWEEP_CONSISTENT_SEEDS):
        plan.append((CRITICAL_OFF, LATTICE, sweep))
    return plan


def kept_cases(meta, grid):
    """The cases of an earlier run of this records folder that the grid does
    not plan but meta.json marks complete, with every record on disk, as
    (off, lattice, seeds).  A re-prediction keeps them: they are drawn from
    their records, not rolled out, and need no reference."""
    if not (RECORDS and isinstance(meta, dict)):
        return []
    have = {_case_name(off, lattice) for off, lattice, _ in grid}
    out = []
    for name, info in sorted((meta.get("cases") or {}).items()):
        if name in have or not isinstance(info, dict):
            continue
        try:
            off, lattice = float(info["off"]), info.get("lattice")
            lattice = None if lattice is None else int(lattice)
            seeds = tuple(int(s) for s in info["seeds"])
        except (KeyError, TypeError, ValueError):
            continue
        if is_dropped(lattice):
            continue
        if (info.get("complete") and seeds and _case_name(off, lattice) == name
                and all(os.path.isfile(record_file(RESULT_CACHE, name, sd))
                        for sd in seeds)):
            out.append((off, lattice, seeds))
    return out


def planned_cases(meta=None):
    """The cases a run of the present settings makes, as (off, lattice,
    seeds): the grid (grid_cases), then the cases the run keeps from an
    earlier job (kept_cases, read from ``meta`` or this run's meta.json).
    The decisions of _evaluate_cases, made ahead."""
    if PLAN_ONLY is not None:
        return list(PLAN_ONLY)
    grid = grid_cases()
    if meta is None:
        meta = _META or read_meta(RESULT_CACHE)
    return grid + kept_cases(meta, grid)


def _case_name(off, lattice):
    return os.path.basename(case_dir(off, lattice))


def records_are_other(old, now):
    """Why the records ``old`` (a meta.json) are not this run's, as (name for
    the archive folder, what they are, what this run uses); None if they are.
    Another checkpoint, or the same one in other numerics."""
    if now and old.get("fingerprint") not in (None, now):
        fp = old.get("fingerprint") or "unknown"
        return fp, f"records of checkpoint {fp}", now
    was = full_numerics(old.get("numerics"))
    if now and old.get("fingerprint") == now and was != numerics_now():
        return (f"{now}-{numerics_tag(was)}",
                f"records of this checkpoint made in other numerics "
                f"({numerics_text(was)})", numerics_text(numerics_now()))
    return None


def _put_records_aside(why):
    """Records of another run, under this label, go to Archive/superseded --
    never deleted, never mixed with this run's.  ``why`` is what
    records_are_other says."""
    label = os.path.basename(os.path.normpath(RESULT_CACHE))
    name, held, uses = why
    if NO_SUPERSEDE:
        raise SystemExit(
            f"--no-supersede: {RESULT_CACHE} holds {held}, and this run uses "
            f"{uses}.  Nothing was moved or written.  Leave out "
            f"--no-supersede to move them to Archive/superseded, or use "
            f"another --result-cache.")
    when = time.strftime("%Y%m%d-%H%M%S")
    dest = _unused(os.path.join(fno_paths.archive_dir(ROOT), "superseded",
                                f"{label}_{name}_{when}",
                                "rollout_cache", label))
    for src in (records_dir(), meta_file()):
        if os.path.exists(src):
            _move(src, os.path.join(dest, os.path.basename(src)))
    print(f"[records] {RESULT_CACHE} held {held}, not this run's; moved to "
          f"{dest} (nothing is deleted)", flush=True)


_META: dict = {}
# the schedule of each quench ("+0.0" -> its pick), meta.json's "schedule"
_PICKS: dict = {}


def start_records(model, ckpt):
    """Open meta.json for this run: who and what it is, and every case it
    plans, not complete yet.  Records of another checkpoint under the same
    label are put aside first.  A case an earlier job of this checkpoint
    finished stays complete while its records are all there."""
    now = model_fingerprint()
    old = read_meta(RESULT_CACHE)
    why = records_are_other(old, now) if old is not None else None
    if why is not None:
        _put_records_aside(why)
        old = None
    cfg = getattr(model, "config", None) or {}
    dtype = None
    if hasattr(model, "parameters"):
        dtype = str(next(iter(model.parameters())).dtype)
    n_par = model.n_parameters() if hasattr(model, "n_parameters") else None
    best = ckpt.get("best") if isinstance(ckpt, dict) else None
    cases = dict(old["cases"]) if old else {}
    names = []
    grid = {_case_name(off, lattice) for off, lattice, _ in grid_cases()}
    for off, lattice, seeds in planned_cases(old or {}):
        name = _case_name(off, lattice)
        names.append(name)
        if name not in grid and PLAN_ONLY is None:
            # a case kept from an earlier job: its entry stays as it is
            continue
        key = (_cache_key(off, LATTICE if lattice is None else lattice)
               if os.path.exists(MODEL_PATH) else None)
        was = cases.get(name) or {}
        kept = bool(was.get("complete") and was.get("seeds") == list(seeds)
                    and was.get("cache_key") == key
                    and all(os.path.isfile(record_path(off, lattice, sd))
                            for sd in seeds))
        cases[name] = {"off": off, "lattice": lattice, "seeds": list(seeds),
                       "cache_key": key, "complete": kept}
    when = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta = {
        "format": 1,
        "label": os.path.basename(os.path.normpath(RESULT_CACHE)),
        "output_label": MODEL_FLAGS["label"],
        "fingerprint": now, "numerics": numerics_now(),
        "checkpoint": MODEL_PATH,
        "model": json.loads(json.dumps(
            {"config": cfg, "n_parameters": n_par, "dtype": dtype,
             "best": best}, default=_plain)),
        "train_horizon": train_horizon(),
        "train_setup": train_setup(),
        "start": START_TIME, "end": END_TIME,
        "views": [float(h) for h in VIEWS] if VIEWS else [],
        "snapshots": sorted(float(t) for t in SNAPSHOT_TIMES),
        "snapshots_explicit": bool(SNAPSHOTS_EXPLICIT),
        "log_time": LOG_TIME, "lattice": LATTICE, "eval_mode": EVAL_MODE,
        "size_sweep": [int(x) for x in SIZE_SWEEP], "seed_set": SEED_SET,
        "eval_seeds": [int(x) for x in EVAL_SEEDS],
        "sweep_seeds": [int(x) for x in SWEEP_SEEDS],
        "planned_cases": names, "cases": cases,
        "created": (old or {}).get("created", when)}
    if ENGINE == "schedule":
        # the (alpha, dt_max) of each quench, kept so a resumed job (and a
        # redraw) uses the same schedules
        meta["engine"] = ENGINE
        meta["schedule"] = dict((old or {}).get("schedule") or {})
        _PICKS.clear()
        _PICKS.update(meta["schedule"])
    _META.clear()
    _META.update(meta)
    _write_meta(_META)
    print(f"[records] keeping every seed's record in {records_dir()} and "
          f"progress in {meta_file()}", flush=True)


def case_ok(res):
    """Whether a case's outputs are all written: it came back, and none of its
    views' submission figures failed."""
    parts = [] if res is None else (list(res.values()) if VIEWS else [res])
    return res is not None and not any(c.get("submission_error")
                                       for c in parts)


def note_case(off, lattice, seeds, res):
    """meta.json after a case: complete once its outputs are written and each
    of its seeds has a record."""
    if not _META:
        return
    name = _case_name(off, lattice)
    info = _META["cases"].setdefault(name, {
        "off": off, "lattice": lattice, "seeds": list(seeds),
        "cache_key": None})
    info["complete"] = bool(case_ok(res) and all(
        os.path.isfile(record_path(off, lattice, sd)) for sd in seeds))
    _write_meta(_META)


def records_status(cache, meta):
    """{case folder name: complete} for the cases meta.json plans, true only
    where it says so and every seed's record is really there."""
    out = {}
    for name in meta.get("planned_cases") or list(meta["cases"]):
        info = meta["cases"].get(name) or {}
        if is_dropped(info.get("lattice")):
            continue
        out[name] = bool(info.get("complete") and info.get("seeds") and all(
            os.path.isfile(record_file(cache, name, sd))
            for sd in info["seeds"]))
    return out


def remove_empty_cache():
    """The rollout cache folder, once the cases have taken their files out of
    it.  Records and meta.json are not the cache: while they are there it
    stays, and nothing of theirs is touched."""
    if os.path.isdir(records_dir()):
        for here, _, _ in os.walk(records_dir(), topdown=False):
            try:
                os.rmdir(here)
            except OSError:
                pass
    try:
        os.rmdir(RESULT_CACHE)
    except OSError:
        if os.path.exists(meta_file()) or os.path.isdir(records_dir()):
            print(f"[cache] {RESULT_CACHE} stays: it holds the records of "
                  f"this run", flush=True)
        return
    print(f"[cache] {RESULT_CACHE} is empty, and removed", flush=True)


def evaluate_seed(model, seed, device, off=0.0, lattice=None):
    lattice = LATTICE if lattice is None else lattice
    gt_cached = os.path.exists(_gt_cache_path(lattice, seed, off, END_TIME))
    _t0 = time.perf_counter()
    gt_times, gt_frames = ch_ground_truth(lattice, seed, off, END_TIME,
                                          DT_STEP, cache_dir=GT_CACHE,
                                          device=device)
    gt_seconds = time.perf_counter() - _t0
    model = _TimedModel(model, device)
    times, r2s, drift = [], [], []
    vid_t, vid_true, vid_pred = [], [], []
    mean_t, mean_p = [], []
    diag_t, L_true, L_pred = [], {}, {}
    energy_t, energy_p, snaps = [], [], {}
    mass0 = float(gt_frames[int(round(START_TIME / DT_STEP))].mean())

    want = sorted(SNAPSHOT_TIMES)
    if want and abs(want[0] - START_TIME) < 0.5 * DT_STEP:
        f0 = gt_frames[int(round(START_TIME / DT_STEP))].astype(np.float64)
        snaps[want.pop(0)] = (f0, f0.copy())

    t_stop, stopped, diverged_at = END_TIME, False, None
    for k, (t, true, pred) in enumerate(
            rollout_frames(model, gt_times, gt_frames, START_TIME, END_TIME,
                           device)):

        finite = bool(np.all(np.isfinite(pred)))
        pmax = float(np.abs(pred).max()) if finite else float("inf")
        if not finite or pmax > DIVERGE_PSI:
            diverged_at = t
            print(f"    [diverged] |psi| = {pmax:.3e} at t = {t:.0f}; stopping "
                  f"and scoring the remainder at R2 = {R2_FLOOR}", flush=True)
            break

        r2 = max(float(r2_score(true, pred)), R2_FLOOR)
        times.append(t)
        r2s.append(r2)
        drift.append(abs(float(pred.mean()) - mass0))
        mean_t.append(float(true.mean()))
        mean_p.append(float(pred.mean()))

        if WRITE_VIDEO and _video_frame(k, t):
            vid_t.append(t)
            vid_true.append(true.astype(np.float32))
            vid_pred.append(pred.astype(np.float32))

        if (k + 1) % DIAG_EVERY == 0 or k == 0:
            diag_t.append(t)
            for name, val in all_lengths(true).items():
                L_true.setdefault(name, []).append(val)
            for name, val in all_lengths(pred).items():
                L_pred.setdefault(name, []).append(val)
            energy_t.append(free_energy(true))
            energy_p.append(free_energy(pred))

        while want and t >= want[0] - 0.5 * DT_STEP:
            snaps[want.pop(0)] = (true, pred)

        if (k + 1) % 200 == 0:
            print(f"    t = {t:7.1f} | R2 = {r2:7.4f}", flush=True)

        if r2 < R2_THRESHOLD and not stopped:
            t_stop, stopped = t, True
            if EARLY_STOP:
                break

    if diverged_at is not None:
        if not stopped:
            t_stop, stopped = diverged_at, True
        i_last = (int(np.argmin(np.abs(gt_times - times[-1]))) if times
                  else int(round(START_TIME / DT_STEP)))
        i_end = int(np.argmin(np.abs(gt_times - END_TIME)))
        for j in range(i_last + 1, i_end + 1):
            times.append(float(gt_times[j]))
            r2s.append(R2_FLOOR)
            drift.append(drift[-1] if drift else 0.0)
            mean_t.append(float(gt_frames[j].mean()))
            mean_p.append(float("nan"))
        for t in [w for w in want if w <= END_TIME + 0.5 * DT_STEP]:
            frame = gt_frames[int(np.argmin(np.abs(gt_times - t)))]
            snaps[t] = (frame.astype(np.float64),
                        np.full_like(frame, np.nan, dtype=np.float64))
            want.remove(t)

    return {"times": np.asarray(times), "r2": np.asarray(r2s),
            "gt_seconds": gt_seconds, "gt_cached": gt_cached,
            "gt_steps": int(round(END_TIME / fno_train.DT)),
            "fno_seconds": model.seconds, "fno_steps": model.calls,
            "mass_drift": np.asarray(drift),
            "mean_true": np.asarray(mean_t), "mean_pred": np.asarray(mean_p),
            "t_stop": t_stop,
            "diverged_at": diverged_at,
            "stopped_early": stopped, "diag_times": np.asarray(diag_t),
            "L_true": {k: np.asarray(v) for k, v in L_true.items()},
            "L_pred": {k: np.asarray(v) for k, v in L_pred.items()},
            "energy_true": np.asarray(energy_t),
            "energy_pred": np.asarray(energy_p), "snapshots": snaps,
            "video_t": np.asarray(vid_t),
            "video_true": (np.stack(vid_true) if vid_true else None),
            "video_pred": (np.stack(vid_pred) if vid_pred else None)}


def figure_r2_and_snapshots(results, seeds, path, off=0.0, lattice=None,
                            dpi=None):
    lattice = LATTICE if lattice is None else lattice
    solo = len(results) == 1
    label, note = MODEL_FLAGS["label"], MODEL_FLAGS["note"]
    shown, rank, shown_r2 = pick_result(results)
    first = results[shown]
    snaps = [t for t in SNAPSHOT_TIMES if t in first["snapshots"]]
    ncol = max(len(snaps), 1)

    fig = plt.figure(figsize=(3.4 * ncol + 1.2, 11.5), facecolor="white")
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.25, 3], hspace=0.20)
    gs = outer[1].subgridspec(3, ncol, hspace=0.06, wspace=0.06)

    ax = fig.add_subplot(outer[0])

    n_max = max(len(r["r2"]) for r in results)
    stack = np.full((len(results), n_max), np.nan)
    for i, r in enumerate(results):
        stack[i, :len(r["r2"])] = r["r2"]
    times = results[int(np.argmax([len(r["r2"]) for r in results]))]["times"]
    mean_curve = np.nanmean(stack, axis=0)
    overall = float(np.nanmean(stack))

    ax.plot(times, mean_curve, color="k", lw=2.8,
            label=(f"seed {seeds[0]}" if solo else
                   f"mean over {len(results)} unseen seeds"))

    final = float(mean_curve[-1])
    ax.plot([times[-1]], [final], marker="o", ms=7, color="k", zorder=5)
    ax.annotate(f"$R^2$ = {final:.4f}  at  $t$ = {times[-1]:.0f}",
                xy=(times[-1], final), xytext=(-12, 14),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=11, color="k", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="0.6", lw=0.8, alpha=0.92))

    lo = float(np.nanmin(mean_curve))
    bottom = max(R2_FLOOR - 0.05, lo - 0.05)
    _time_axis(ax)
    ax.set_ylim(bottom, 1.02)
    ax.set_xlabel("real time $t$")
    ax.set_ylabel("$R^2$ vs. ground truth")
    ax.grid(alpha=0.3, ls=":")
    mark_time(ax, times[0], times[-1], *case_marks(results, lattice))
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ranks = None if solo else _submission().seed_ranks(results, seeds)
    caption = (f"single realisation: seed {seeds[0]}, mean $R^2$ = "
               f"{shown_r2:.4f}" if ranks is None else
               f"snapshots below: seed {seeds[shown]} "
               f"({'best' if rank == 1 else 'best healthy'} of "
               f"{len(results)}, rank {rank}, mean $R^2$ = {shown_r2:.4f}); "
               f"median seed {ranks['median']['seed']} "
               f"({ranks['median']['mean_r2']:.4f}), worst seed "
               f"{ranks['worst']['seed']} ({ranks['worst']['mean_r2']:.4f})"
               f"   |   the curve above is the {len(results)}-seed mean")

    held = sum(1 for r in results if not r["stopped_early"])
    blew = sum(1 for r in results if r.get("diverged_at") is not None)
    verdict = (f"mean $R^2$ = {overall:.4f}"
               + (f"; {'held' if held else 'did not hold'} throughout"
                  if solo else
                  f"; {held}/{len(results)} seeds held throughout")
               + (f"; {blew}/{len(results)} DIVERGED" if blew else ""))
    comp = ("critical quench" if abs(off) < 1e-12
            else f"off-critical quench $\\bar\\psi={off:+.2f}$")
    ax.set_title(f"{label} — autoregressive rollout at "
                 f"$l={lattice}$, $t={START_TIME:.0f}$ to "
                 f"${END_TIME:.0f}$, {comp}{note}  ({verdict})\n{caption}",
                 fontweight="bold", fontsize=13)

    rows = [("ground truth", 0, PHASE_CMAP, (-1, 1)),
            ("FNO prediction", 1, PHASE_CMAP, (-1, 1)),
            ("|error|", 2, ERR_CMAP, (0, 1))]
    for ri, (name, which, cmap, vlim) in enumerate(rows):
        for ci, t in enumerate(snaps):
            true, pred = first["snapshots"][t]
            img = [true, pred, np.abs(true - pred)][which]
            axi = fig.add_subplot(gs[ri, ci])
            im = axi.imshow(img, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                            interpolation="nearest")
            axi.set_xticks([]); axi.set_yticks([])
            if ri == 0:
                axi.set_title(f"$t$ = {t:.0f}   "
                              f"($R^2$ = {r2_score(true, pred):.3f})", fontsize=11)
            if ci == 0:
                axi.set_ylabel(name, fontsize=11)
            if ci == len(snaps) - 1:
                fig.colorbar(im, ax=axi, fraction=0.046, pad=0.03)

    fno_paths.save_figure(fig, path, FIG_DPI if dpi is None else dpi,
                          bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def snapshot_means(results, tt):
    """The spectra at snapshot time ``tt`` averaged over the seeds, for the
    truth and the FNO: {"true" or "pred": (k, S(k), r, C(r), L, n seeds)}.
    A seed without that snapshot, or whose FNO field is not finite, is left
    out of that side; L is the mean of the seeds' own first zeros of C."""
    out = {}
    for side, which in (("true", 0), ("pred", 1)):
        got, size = [], []
        for res in results:
            snap = res["snapshots"].get(tt)
            if snap is None or not np.all(np.isfinite(snap[which])):
                continue
            k, s_k = structure_factor(snap[which])
            r, c_r = correlation_function(snap[which])
            got.append((k, s_k, r, c_r))
            size.append(length_from_zero_crossing(r, c_r))
        if got:
            ok = [x for x in size if np.isfinite(x)]
            out[side] = (got[0][0], np.mean([g[1] for g in got], axis=0),
                         got[0][2], np.mean([g[3] for g in got], axis=0),
                         float(np.mean(ok)) if ok else float("nan"),
                         len(got))
    return out


def free_energy_monotone(results):
    """One flag per seed: the FNO's free energy never rises (a rise of at
    most 1e-9 is rounding)."""
    return [bool(np.all(np.diff(r["energy_pred"]) <= 1e-9)) for r in results]


def figure_kinetics(results, seeds, path, off=0.0, lattice=None, dpi=None,
                    notes=None):
    """Puri's phase-ordering kinetics for a case: L(t), S(k,t), the scaling
    functions and the free energy, averaged over the seeds.  Given one
    result it is that seed's own figure.  The panels carry letters (a) to
    (f) and no titles; what they show and the numbers that used to be in
    the titles go into ``notes`` ({file name: lines}, figure_notes.txt)."""
    lattice = LATTICE if lattice is None else lattice
    if not any(len(r["diag_times"]) for r in results):
        return
    n_seeds = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.0), facecolor="white")
    (a, b, c), (d, e, f) = axes
    t, Lt, Lp = length_means(results)
    horizon, onset = case_marks(results, lattice)
    note = [f"phase-ordering kinetics of one case, lattice {lattice}, "
            + (f"seed {seeds[0]}" if n_seeds == 1
               else f"mean over {n_seeds} seeds")
            + (f", psi0 = {off:+.2f}" if abs(off) >= 1e-12 else "")
            + "; solid: Solver, dashed: FNO in (b) to (d)"]

    if n_seeds > 1:
        # mean +- 1 sd over the seeds (the group's panel (a)); a seed that ended
        # early leaves the sd to the seeds still running
        _, st_stack, sp_stack = length_stacks(results)
        for mean, stack, col in ((Lt, st_stack, C_TRUE),
                                 (Lp, sp_stack, C_PRED)):
            sd = seed_sd(stack)
            a.fill_between(t, np.maximum(mean - sd, 1e-3), mean + sd,
                           color=col, alpha=0.15, lw=0)
    a.loglog(t, Lt, color=C_TRUE, lw=2.4, label="Solver")
    a.loglog(t, Lp, color=C_PRED, lw=2.0, ls="--", label="FNO")
    ok = np.isfinite(Lt) & (Lt > 0)
    if ok.any():
        t0 = np.exp(np.mean(np.log(t[ok])))
        l0 = np.exp(np.mean(np.log(Lt[ok])))
        tr = np.array([t[0], t[-1]])
        a.loglog(tr, l0 * (tr / t0) ** (1 / 3), "k:", lw=2,
                 label="1/3 slope")
    mark_time(a, t[0], t[-1], horizon, onset, short=True)
    a.set_xlabel("$t$"); a.set_ylabel(f"$L(t)$  [{LENGTH_KEY}]")
    a.set_title("(a)", loc="left", fontweight="bold")
    a.grid(alpha=0.3, which="both", ls=":"); a.legend(fontsize=9)
    note.append(
        f"(a) domain size L(t) ({LENGTH_KEY}, the first zero of C(r,t)) on "
        "log-log axes"
        + (", mean +- 1 sd over the seeds" if n_seeds > 1 else "")
        + f"; growth exponent over the whole rollout: "
        f"Solver n = {fit_growth_exponent(t, Lt):.3f}, FNO n = "
        f"{fit_growth_exponent(t, Lp):.3f}; dotted line: t^(1/3) "
        "(Lifshitz-Slyozov), anchored at the geometric-mean time"
        + (f"; training horizon t = {horizon:g}" if horizon is not None
           else "")
        + (f"; grey region: L > l/{FINITE_SIZE:g} from t = {onset:g}"
           if onset is not None else ""))

    avail = [x for x in SNAPSHOT_TIMES
             if any(x in r["snapshots"] for r in results)]
    means = {tt: snapshot_means(results, tt) for tt in avail}
    cmap = plt.get_cmap("viridis")

    for i, tt in enumerate(avail):
        col = cmap(i / max(len(avail) - 1, 1))
        if "true" in means[tt]:
            k, s_t = means[tt]["true"][:2]
            b.loglog(k[k > 0], s_t[k > 0], color=col, lw=2.0,
                     label=f"$t$ = {tt:.0f}")
        if "pred" in means[tt]:
            k, s_p = means[tt]["pred"][:2]
            b.loglog(k[k > 0], s_p[k > 0], color=col, lw=1.4, ls="--")
    porod = ""
    if avail and "true" in means[avail[-1]]:
        k_ref, s_ref = means[avail[-1]]["true"][:2]
        win = porod_window(k_ref, s_ref)
        if win is not None:
            lo, hi = win
            mid = (lo * hi) ** 0.5
            anchor = float(np.interp(mid, k_ref[1:], s_ref[1:]))
            kk = np.array([lo, hi])
            b.loglog(kk, anchor * (kk / mid) ** -3, "k:", lw=2,
                     label=r"Porod $k^{-3}$")
            porod = f"; dotted line: Porod k^-3 over k = {lo:.2f}-{hi:.2f}"
    b.set_xlabel("$k$"); b.set_ylabel("$S(k,t)$")
    b.set_title("(b)", loc="left", fontweight="bold")
    b.grid(alpha=0.3, which="both", ls=":"); b.legend(fontsize=8)
    note.append("(b) structure factor S(k,t) at t = "
                + ", ".join(f"{tt:.0f}" for tt in avail) + porod)

    for i, tt in enumerate(avail):
        col = cmap(i / max(len(avail) - 1, 1))
        for side, ls, lw in (("true", "-", 2.0), ("pred", "--", 1.4)):
            if side not in means[tt]:
                continue
            k, s_k, r, c_r, L, _ = means[tt][side]
            if np.isfinite(L) and L > 0:
                c.loglog(k[k > 0] * L, s_k[k > 0] / L ** D, color=col, ls=ls,
                         lw=lw)
                d.plot(r / L, c_r, color=col, ls=ls, lw=lw)
    c.set_xlabel("$kL$"); c.set_ylabel("$L^{-d}S(k,t)$")
    c.set_title("(c)", loc="left", fontweight="bold")
    c.grid(alpha=0.3, which="both", ls=":")
    d.axhline(0, color="k", lw=0.8)
    d.set_xlim(0, 5); d.set_xlabel("$r/L$"); d.set_ylabel("$C(r,t)$")
    d.set_title("(d)", loc="left", fontweight="bold")
    d.grid(alpha=0.3, ls=":")
    note.append("(c) dynamical scaling S = L^d f(kL), Puri Eq. 1.90, same "
                "times as (b)")
    note.append("(d) scaling function C = g(r/L), Puri Eq. 1.88, same "
                "times as (b)")

    _, e_true = seed_series(results, lambda r: r["energy_true"])
    _, e_pred = seed_series(results, lambda r: r["energy_pred"])
    e.plot(t, seed_mean(e_true), color=C_TRUE, lw=2.4, label="Solver")
    e.plot(t, seed_mean(e_pred), color=C_PRED, lw=2.0, ls="--", label="FNO")
    falling = free_energy_monotone(results)
    mono = ("yes" if all(falling) else "no" if n_seeds == 1
            else f"{sum(falling)} of {n_seeds} seeds")
    mark_time(e, t[0], t[-1], horizon, onset, short=True)
    e.set_xlabel("$t$"); e.set_ylabel(r"$\langle f \rangle$")
    e.set_title("(e)", loc="left", fontweight="bold")
    e.grid(alpha=0.3, ls=":"); e.legend(fontsize=9)
    note.append("(e) free energy per site <f>, seed mean; FNO monotone "
                f"(never rises, np.diff <= 1e-9): {mono}")

    # (f) <psi>(t) of the truth and the FNO with the material A content c_A
    # on the right, and the composition at the end (the group's panel (f))
    if all(r.get("mean_true") is not None and r.get("mean_pred") is not None
           for r in results):
        t_m, m_true = _r2_stack(results, "mean_true")
        _, m_pred = _r2_stack(results, "mean_pred")
        mt, mp = seed_mean(m_true), seed_mean(m_pred)
        psi0 = float(mt[np.isfinite(mt)][0]) if np.isfinite(mt).any() else off
        f.plot(t_m, mt, color=C_TRUE, lw=2.2, label="Solver")
        f.plot(t_m, mp, color=C_PRED, lw=2.0, ls="--", label="FNO")
        f.axhline(psi0, color="0.35", lw=1.0, ls=":", label="initial mean")
        both = np.concatenate([mt, mp])
        both = both[np.isfinite(both)]
        half = max(1.3 * (float(np.abs(both - psi0).max()) if both.size
                          else 0.0), 1e-3)
        f.set_ylim(psi0 - half, psi0 + half)
        mark_time(f, t_m[0], t_m[-1], horizon, onset, short=True)
        sec = f.secondary_yaxis("right", functions=(psi_to_a, a_to_psi))
        sec.set_ylabel("material A content  $c_A$  (%)")
        alive = np.nonzero(np.isfinite(mp))[0]
        a_t = float(psi_to_a(mt[-1])) if np.isfinite(mt[-1]) else float("nan")
        a_p = float(psi_to_a(mp[alive[-1]])) if alive.size else float("nan")
        f.legend(fontsize=8)
        note.append(
            "(f) mean order parameter <psi>(t)"
            + ("" if n_seeds == 1 else ", seed mean")
            + f"; initial mean = {psi0:.4f}; composition at t = "
            f"{t_m[-1]:g}: Solver A {a_t:.2f} / B {100 - a_t:.2f} %, FNO "
            f"A {a_p:.2f} / B {100 - a_p:.2f} %")
    else:
        f.text(0.5, 0.5, "cached results predate the mean order parameter;\n"
               "roll the case out again to record it", ha="center",
               va="center", transform=f.transAxes)
        note.append("(f) mean order parameter: not recorded in these "
                    "cached results")
    f.set_title("(f)", loc="left", fontweight="bold")
    f.set_xlabel("$t$"); f.set_ylabel(r"$\langle\psi\rangle$")
    f.grid(alpha=0.3, ls=":")

    fig.tight_layout()
    fno_paths.save_figure(fig, path, FIG_DPI if dpi is None else dpi,
                          bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")
    if notes is not None:
        notes[os.path.basename(path)] = note


def figure_conservation(results, seeds, path, off=0.0, lattice=None,
                        dpi=None):
    """psi-bar against time, and how far the rollout drifts from the truth.

    Cahn-Hilliard is Model B: psi-bar is exactly conserved, so the top panel is
    a statement that nothing happens -- both curves sit on the composition they
    started at.  The bottom panel is where the information is, because the
    deviation is far too small to see on a linear axis.

    For one seed (the best seed's own figure) its own curves are drawn.  For a
    case, only averages over the seeds: the mean psi-bar of truth and
    prediction, and the mean and the worst deviation at each step -- no seed
    but the best keeps a curve of its own.  The headline numbers are maxima
    over every seed either way.
    """
    if any(r.get("mean_true") is None or r.get("mean_pred") is None
           for r in results):
        print(f"[skip] {os.path.basename(path)}: cached results predate the "
              f"conservation figure; roll the case out again without its "
              f"cached rollouts to record them.", flush=True)
        return
    lattice = LATTICE if lattice is None else lattice
    label, note = MODEL_FLAGS["label"], MODEL_FLAGS["note"]

    fig, (a, b) = plt.subplots(2, 1, figsize=(9.5, 7.4), sharex=True,
                               facecolor="white",
                               gridspec_kw={"height_ratios": [1.0, 1.15]})

    worst, worst_true, drift = 0.0, 0.0, 0.0
    end_pred = off
    n_seeds = len(results)
    if n_seeds == 1:
        t = results[0]["times"]
        mt = np.asarray(results[0]["mean_true"])
        mp = np.asarray(results[0]["mean_pred"])
        a.plot(t, mt, color=C_TRUE, lw=2.2, alpha=0.85, label="ground truth")
        a.plot(t, mp, color=C_PRED, lw=1.3, ls="--", label="FNO prediction")
        d = np.abs(mp - mt)
        b.semilogy(t, np.where(d > 0, d, np.nan), lw=1.2, alpha=0.8,
                   color=SWEEP_COLORS[0], label=f"seed {seeds[0]}")
    else:
        t, st = _r2_stack(results, "mean_true")
        _, sp = _r2_stack(results, "mean_pred")
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.abs(sp - st)
            dm, dx = np.nanmean(d, axis=0), np.nanmax(d, axis=0)
            a.plot(t, np.nanmean(st, axis=0), color=C_TRUE, lw=2.2,
                   alpha=0.85, label=f"ground truth, mean of {n_seeds} seeds")
            a.plot(t, np.nanmean(sp, axis=0), color=C_PRED, lw=1.3, ls="--",
                   label=f"FNO prediction, mean of {n_seeds} seeds")
        b.semilogy(t, np.where(dm > 0, dm, np.nan), lw=1.4,
                   color=SWEEP_COLORS[0], label=f"mean over {n_seeds} seeds")
        b.semilogy(t, np.where(dx > 0, dx, np.nan), lw=1.0, ls="--",
                   color=SWEEP_COLORS[2], label="worst seed at each step")
    for res in results:
        mt, mp = np.asarray(res["mean_true"]), np.asarray(res["mean_pred"])
        d = np.abs(mp - mt)
        with np.errstate(invalid="ignore"):
            worst = max(worst, float(np.nanmax(d)) if np.isfinite(d).any()
                        else 0.0)
            worst_true = max(worst_true, float(np.nanmax(np.abs(mt - mt[0]))))
            if np.isfinite(mp).any():
                drift = max(drift, float(np.nanmax(np.abs(mp - mp[0]))))
                if abs(float(mp[np.isfinite(mp)][-1]) - off) > abs(end_pred
                                                                  - off):
                    end_pred = float(mp[np.isfinite(mp)][-1])

    a.axhline(off, color="k", ls=":", lw=1.2,
              label=rf"nominal $\bar\psi_0 = {off:+.2f}$")
    marks = case_marks(results, lattice)
    mark_time(a, t[0], t[-1], *marks)
    span = max(MIN_PSI_SPAN, 1.5 * max(worst, drift))
    a.set_ylim(off - span, off + span)
    a.set_ylabel(r"$\bar\psi(t)$")
    a.grid(alpha=0.3, ls=":")
    a.legend(loc="upper right", fontsize=9, ncol=3)

    comp = a.secondary_yaxis("right", functions=(psi_to_a, a_to_psi))
    comp.set_ylabel(r"$c_A$  (%)     [$c_B = 100 - c_A$]")

    who = (f"seed {seeds[0]}" if n_seeds == 1
           else f"mean and worst of {n_seeds} seeds")
    a.set_title(f"Order-parameter conservation over the rollout ({who})",
                fontweight="bold", fontsize=12)
    a.annotate(f"start  {off:+.4f}  ->  {psi_to_a(off):.3f}% A / "
               f"{100 - psi_to_a(off):.3f}% B          "
               f"FNO end  {end_pred:+.4f}  ->  {psi_to_a(end_pred):.3f}% A / "
               f"{100 - psi_to_a(end_pred):.3f}% B",
               xy=(0.5, 0.10), xycoords="axes fraction", ha="center",
               fontsize=9.5, color="#333333")
    a.annotate(f"worst composition shift over the rollout: "
               f"{50.0 * drift:.2e} percentage points",
               xy=(0.5, 0.03), xycoords="axes fraction", ha="center",
               fontsize=9, color="#555555", style="italic")

    b.axhline(np.finfo(np.float64).eps, color="#888888", ls="--", lw=1.2,
              label=f"float64 epsilon = {np.finfo(np.float64).eps:.1e}")
    b.set_xlabel("real time $t$")
    b.set_ylabel(r"$|\bar\psi_{\rm FNO} - \bar\psi_{\rm truth}|$")
    _time_axis(b)
    b.grid(alpha=0.3, which="both", ls=":")
    mark_time(b, t[0], t[-1], *marks)
    b.legend(loc="upper left", fontsize=8, ncol=2)
    hard = MODEL_FLAGS["hard_constraints"]
    b.set_title("Deviation from the conserved value\n"
                f"worst {worst:.2e}   |   the solver's own drift "
                f"{worst_true:.2e}   |   hard mass conservation "
                f"{'ON' if hard else 'OFF'}",
                fontweight="bold", fontsize=11)

    comp = ("critical quench" if abs(off) < 1e-12
            else f"off-critical quench $\\bar\\psi = {off:+.2f}$")
    fig.suptitle(f"{label}{note} — $\\bar\\psi$ conservation at "
                 f"$l = {lattice}$, {comp}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fno_paths.save_figure(fig, path, FIG_DPI if dpi is None else dpi,
                          bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def video_writer(fps):
    """MP4 where ffmpeg exists, animated GIF where it does not.

    ffmpeg is a system binary rather than a Python package, so it is present on
    a laptop and a coin flip on a cluster login node.  Pillow ships with
    matplotlib and always works, so the figure is never lost to a missing
    codec -- only its container changes, and the caller is told which it got.
    """
    if animation.FFMpegWriter.isAvailable():
        return "mp4", animation.FFMpegWriter(
            fps=fps, bitrate=3200, metadata={"artist": MODEL_FLAGS["label"]})
    return "gif", animation.PillowWriter(fps=fps)


def figure_video(res, seed, path, off=0.0, lattice=None):
    """The rollout as a film: truth, prediction and error side by side.

    Snapshots at four times cannot show *when* a domain merges wrongly, which
    is the failure this model actually has.  Playing every VIDEO_EVERY-th step
    does, and it costs nothing extra to solve -- the frames are recorded during
    the rollout that was happening anyway.
    """
    if res.get("video_true") is None or not len(res.get("video_t", [])):
        print(f"[skip] {os.path.basename(path)}: this rollout was cached "
              f"before WRITE_VIDEO, so it has no frames; roll the case out "
              f"again without its cached rollouts to record them.",
              flush=True)
        return
    lattice = LATTICE if lattice is None else lattice
    t = np.asarray(res["video_t"])
    # A long run with views keeps its frames as float16 (about 450 of them);
    # the film is drawn in float32 either way.
    true = np.asarray(res["video_true"], dtype=np.float32)
    pred = np.asarray(res["video_pred"], dtype=np.float32)
    r2 = np.interp(t, res["times"], res["r2"])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), facecolor="white")
    ims = []
    for ax, (name, cmap, vlim) in zip(
            axes, (("ground truth", PHASE_CMAP, (-1, 1)),
                   ("FNO prediction", PHASE_CMAP, (-1, 1)),
                   ("|error|", ERR_CMAP, (0, 1)))):
        im = ax.imshow(np.zeros((lattice, lattice)), cmap=cmap,
                       vmin=vlim[0], vmax=vlim[1], interpolation="nearest",
                       animated=True)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ims.append(im)

    comp = ("critical quench" if abs(off) < 1e-12
            else f"off-critical quench $\\bar\\psi = {off:+.2f}$")
    head = fig.suptitle("", fontsize=13, fontweight="bold")

    def draw(i):
        ims[0].set_data(true[i])
        ims[1].set_data(pred[i])
        ims[2].set_data(np.abs(true[i].astype(np.float64)
                               - pred[i].astype(np.float64)))
        head.set_text(f"{MODEL_FLAGS['label']}{MODEL_FLAGS['note']} — "
                      f"$l = {lattice}$, {comp}, seed {seed}\n"
                      f"$t$ = {t[i]:.0f}      $R^2$ = {r2[i]:.4f}")
        return ims + [head]

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    anim = animation.FuncAnimation(fig, draw, frames=len(t), blit=False)
    ext, writer = video_writer(VIDEO_FPS)
    out = os.path.splitext(path)[0] + "." + ext
    anim.save(out, writer=writer, dpi=VIDEO_DPI)
    plt.close(fig)
    size = os.path.getsize(out) / 1e6
    print(f"[saved] {out}  ({len(t)} frames, {size:.1f} MB"
          + ("" if ext == "mp4" else "; ffmpeg missing, wrote a GIF instead")
          + ")")


SWEEP_COLORS = ("#08519C", "#4292C6", "#C51B8A", "#41AB5D", "#005A32")
SWEEP_SHAPES = ("o", "s", "^", "D", "v", "<", ">", "p")


def _sweep_panel(ax, curves, labels, ylabel=True):
    """One panel of the sweep figure: R2(t), one line per case."""
    lo = 1.0
    for i, ((t, r), lab) in enumerate(zip(curves, labels)):
        r = np.clip(np.nan_to_num(np.asarray(r, dtype=np.float64),
                                  nan=R2_FLOOR, posinf=1.0, neginf=R2_FLOOR),
                    R2_FLOOR, 1.0)
        every = (0.07 if log_time() else
                 max(1, len(t) // max(SWEEP_MARKERS, 1)))
        ax.plot(t, r, color=SWEEP_COLORS[i % len(SWEEP_COLORS)],
                marker=SWEEP_SHAPES[i % len(SWEEP_SHAPES)], markevery=every,
                markersize=5, lw=1.6, label=lab)
        lo = min(lo, float(r.min()))
    _time_axis(ax)
    if SWEEP_YMIN is not None:
        bottom = float(SWEEP_YMIN)
    else:
        bottom = float(max(lo - 0.02, R2_FLOOR - 0.05))
    ax.set_ylim(bottom, 1.005)
    ax.set_ylabel("$R^2$" if ylabel else "$R^2$")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=9, loc="lower left", frameon=False)
    ax.tick_params(direction="in", top=True, right=True)


def figure_sweeps(by_off, by_size, path):
    """R2(t) per composition, and per system size -- the reference figure."""
    n_seeds = max((c["stats"]["n_seeds"]
                   for c in list(by_off.values()) + list(by_size.values())
                   if "stats" in c), default=len(SWEEP_SEEDS))
    panels = []
    if len(by_off) > 1:
        panels.append((
            [(by_off[o]["times"], by_off[o]["mean_curve"])
             for o in sorted(by_off)],
            [rf"$\psi_0 = {o:.1f}$" for o in sorted(by_off)],
            "different mixture compositions"))
    if len(by_size) > 1:
        panels.append((
            [(by_size[l]["times"], by_size[l]["mean_curve"])
             for l in sorted(by_size)],
            [rf"$L = {l}$" for l in sorted(by_size)],
            "different system sizes"))
    if not panels:
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(5.6, 3.4 * len(panels)),
                             sharex=True, facecolor="white")
    axes = np.atleast_1d(axes)
    for k, (ax, (curves, labels, _)) in enumerate(zip(axes, panels)):
        _sweep_panel(ax, curves, labels)
        ax.annotate(f"({chr(97 + k)})", xy=(0.94, 0.90),
                    xycoords="axes fraction", fontsize=13)
    axes[-1].set_xlabel("$t$")
    what = " and ".join(p[2] for p in panels)
    fig.suptitle(f"{MODEL_FLAGS['label']}{MODEL_FLAGS['note']} — $R^2$ for "
                 f"{what}\n(mean over {n_seeds} seeds, autoregressive from "
                 f"$t={START_TIME:.0f}$)", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fno_paths.save_figure(fig, path, 200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def timing_totals(rows):
    """One case's timing as sums and counts over its seeds.

    A case keeps no per-seed rows, so the timing report is built from these:
    sums rather than means, so that a report over several cases -- or over
    cases a resumed job skipped -- comes out exactly as one over their seeds
    would.  A seed whose reference came from disk is not counted as solved.
    """
    t = {"n": 0, "n_solved": 0, "fno_seconds": 0.0, "ms_per_step": 0.0,
         "n_ms": 0, "gt_seconds_solved": 0.0, "fno_seconds_solved": 0.0,
         "gt_over_fno_solved": 0.0, "n_ratio": 0}
    for r in rows:
        gt_s, fno_s = r.get("gt_seconds"), r.get("fno_seconds")
        if gt_s is None or fno_s is None:
            continue
        t["n"] += 1
        t["fno_seconds"] += fno_s
        if r.get("fno_steps"):
            t["ms_per_step"] += 1e3 * fno_s / r["fno_steps"]
            t["n_ms"] += 1
        if not r.get("gt_cached"):
            t["n_solved"] += 1
            t["gt_seconds_solved"] += gt_s
            t["fno_seconds_solved"] += fno_s
            if fno_s > 0:
                t["gt_over_fno_solved"] += gt_s / fno_s
                t["n_ratio"] += 1
    return t


def _add_totals(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def timing_rows(case):
    """The case's line of the timing report, from its timing_totals."""
    t = (case.get("stats") or {}).get("timing") or {}
    if not t.get("n"):
        return []
    return [{"off": case["off"], "lattice": case["lattice"], "totals": t}]


def write_timing_report(cases, device, model, path_json, path_txt):
    """Cost of every rollout and every reference solve, as its own artefact.

    One line per case, from the sums its summary keeps (timing_totals): no
    seed has a row of its own.  ``model`` and ``device`` are None when the
    rollouts are not this run's (make_submission regenerating a folder from
    a cache); the report then gives the device as unknown.
    """
    by_case = {}
    for c in cases:
        if c is None:
            continue
        for r in timing_rows(c):
            k = (r["off"], r["lattice"])
            by_case[k] = _add_totals(by_case.get(k, {}), r["totals"])
    if not by_case:
        print("[timing] nothing to report (no case carried timings)",
              flush=True)
        return
    window = (END_TIME - START_TIME) / END_TIME
    n_all = sum(t["n"] for t in by_case.values())
    n_solved = sum(t["n_solved"] for t in by_case.values())
    dev = None if device is None else torch.device(device)
    meta = {
        "device": ((RECORDED_DEVICE or "unknown") if device is None
                   else str(device)),
        "gpu": (torch.cuda.get_device_name(0)
                if dev is not None and dev.type == "cuda" else None),
        "precision": None if device is None else PRECISION,
        "model_dtype": (None if model is None
                        else str(next(model.parameters()).dtype)),
        "model_parameters": None if model is None else model.n_parameters(),
        "fingerprint": model_fingerprint(),
        "dt_step": None if ENGINE == "schedule" else DT_STEP,
        "solver_dt": fno_train.DT,
        "euler_steps_per_model_step": (None if ENGINE == "schedule" else
                                       int(round(DT_STEP / fno_train.DT))),
        "t_start": START_TIME, "t_end": END_TIME,
        "n_rollouts": n_all,
        "n_ground_truths_solved": n_solved,
        "n_ground_truths_from_cache": n_all - n_solved}

    def mean(total, n):
        return total / n if n else None

    rows = []
    for (off, lat), t in sorted(by_case.items()):
        rows.append({
            "off": off, "lattice": lat, "n_rollouts": t["n"],
            "n_solved": t["n_solved"],
            "gt_euler_steps_same_window": int(round(
                (END_TIME - START_TIME) / fno_train.DT)),
            "gt_seconds_same_window_mean": mean(
                window * t["gt_seconds_solved"], t["n_solved"]),
            "fno_seconds_mean": mean(t["fno_seconds"], t["n"]),
            "fno_ms_per_step_mean": mean(t["ms_per_step"], t["n_ms"]),
            "speedup_mean": mean(window * t["gt_over_fno_solved"],
                                 t["n_ratio"])})
    with open(path_json, "w") as fh:
        json.dump({"meta": meta, "cases": rows}, fh, indent=1)

    bar = "=" * 78
    out = [bar, "  TIMING", bar,
           f"  device                    {meta['device']}"
           + (f" ({meta['gpu']})" if meta["gpu"] else ""),
           f"  precision                 {meta['precision'] or 'unknown'}",
           "  model parameters          "
           + ("unknown" if meta["model_parameters"] is None
              else f"{meta['model_parameters']:,}"),
           f"  checkpoint fingerprint    {meta['fingerprint'] or 'unknown'}",
           ("  one model step            a stride of the schedule "
            f"(dt-conditioned), the solver steps at dt={meta['solver_dt']}"
            if meta["dt_step"] is None else
            f"  one model step            {meta['dt_step']:g} of real time "
            f"= {meta['euler_steps_per_model_step']} Euler steps at dt="
            f"{meta['solver_dt']}"),
           f"  rollout window            t = {START_TIME:.0f} .. {END_TIME:.0f}",
           f"  ground truths solved      {meta['n_ground_truths_solved']} of "
           f"{meta['n_rollouts']} (the rest were already in "
           f"{ref_cache() if ENGINE == 'schedule' else GT_CACHE})",
           "",
           "  Solver time is scaled to the rollout window before comparing.",
           "  A cached reference is a disk read, not a solve, so those rows",
           "  are excluded from the speedup column.",
           "",
           f"  {'case':>14} {'seeds':>6} {'solver s':>10} {'FNO s':>9} "
           f"{'ms/step':>9} {'speedup':>9}",
           "  " + "-" * 74]
    nan = float("nan")
    for r in rows:
        name = f"psi0={r['off']:+.1f} l={r['lattice']}"
        fno = r["fno_seconds_mean"]
        ms = nan if r["fno_ms_per_step_mean"] is None else r["fno_ms_per_step_mean"]
        if r["n_solved"]:
            gt = r["gt_seconds_same_window_mean"]
            sp = nan if r["speedup_mean"] is None else r["speedup_mean"]
            out.append(f"  {name:>14} {r['n_rollouts']:>6} {gt:>10.1f} "
                       f"{fno:>9.1f} {ms:>9.2f} {sp:>8.1f}x")
        else:
            out.append(f"  {name:>14} {r['n_rollouts']:>6} {'cached':>10} "
                       f"{fno:>9.1f} {ms:>9.2f} {'--':>9}")
    out.append("  " + "-" * 74)
    if n_solved:
        gt = window * sum(t["gt_seconds_solved"]
                          for t in by_case.values()) / n_solved
        fno = sum(t["fno_seconds_solved"] for t in by_case.values()) / n_solved
        sp = gt / fno
        out.append(f"  over the {n_solved} rollouts whose reference was "
                   f"actually solved: solver {gt:.1f} s, FNO {fno:.1f} s, "
                   f"speedup {sp:.1f}x")
        if sp < 1.5 and meta["precision"] and meta["dt_step"]:
            out.append("")
            out.append(f"  NOTE: the model is evaluated in {PRECISION}, and one")
            out.append(f"  call replaces only "
                       f"{meta['euler_steps_per_model_step']} Euler steps of a")
            out.append("  scheme that is itself cheap per step, so a speedup "
                       "near or below 1 is")
            out.append("  a statement about this hardware and this precision, "
                       "not about the")
            out.append("  method.  fp64 runs at 1/32 rate on consumer cards "
                       "and 1/2 on an A100.")
            out.append("  Re-run with PRECISION = \"fp32\" to see the "
                       "deployment-relevant number.")
    else:
        out.append("  every reference came from cache, so no speedup is "
                   "quoted; delete gt_cache to measure one")
    out.append(bar)
    text = chr(10).join(out)
    with open(path_txt, "w") as fh:
        fh.write(text + chr(10))
    print(chr(10) + text, flush=True)
    print(f"[saved] {path_json}", flush=True)
    print(f"[saved] {path_txt}", flush=True)


def finite_max(values):
    """The largest finite number among ``values``, or None when there is
    none.  A seed that diverged at its first step has only NaN to offer, and
    max() over a list that holds a NaN answers by where the NaN sits."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else None


def seed_summary(res, seed):
    """The one-seed row that goes into both the case summary and its own file."""
    mean_err = (None if res.get("mean_pred") is None else finite_max(
        np.abs(np.asarray(res["mean_pred"], dtype=np.float64)
               - np.asarray(res["mean_true"], dtype=np.float64)).ravel()))
    return {
        "seed": seed, "t_stop": res["t_stop"],
        "stopped_early": res["stopped_early"],
        "steps_above_threshold": n_steps_above(res),
        "diverged_at": res.get("diverged_at"),
        "mean_r2": float(res["r2"].mean()),
        "final_r2": float(res["r2"][-1]),
        "r2_at": {str(t): float(r2_score(*res["snapshots"][t]))
                  for t in SNAPSHOT_TIMES if t in res["snapshots"]},
        "growth_exponent_truth": fit_growth_exponent(
            res["diag_times"], res["L_true"].get(LENGTH_KEY, ())),
        "growth_exponent_fno": fit_growth_exponent(
            res["diag_times"], res["L_pred"].get(LENGTH_KEY, ())),
        "gt_seconds": res.get("gt_seconds"),
        "gt_cached": res.get("gt_cached"),
        "fno_seconds": res.get("fno_seconds"),
        "fno_steps": res.get("fno_steps"),
        "max_mass_drift": float(res["mass_drift"].max()),
        "max_mean_error": mean_err}


def write_seed_outputs(res, seed, off, lattice, suffix, label):
    """The best seed's own folder: its R2 curve, its numbers, its figures.

    No other seed gets one.
    """
    lat = LATTICE if lattice is None else lattice
    stem = f"{label}{suffix}_seed{seed}"

    np.savetxt(seed_path(off, lattice, seed, f"{stem}_r2.csv"),
               np.column_stack([res["times"], res["r2"]]),
               delimiter=",", header="t,r2", comments="")
    with open(seed_path(off, lattice, seed, f"{stem}_summary.json"),
              "w") as fh:
        json.dump(seed_summary(res, seed), fh, indent=1)
    if not PER_SEED_FIGURES:
        return

    figure_r2_and_snapshots(
        [res], [seed],
        seed_path(off, lattice, seed, f"{stem}_r2_and_snapshots.png"),
        off, lat, dpi=PER_SEED_DPI)
    figure_kinetics(
        [res], [seed], seed_path(off, lattice, seed, f"{stem}_kinetics.png"),
        off, lat, dpi=PER_SEED_DPI)
    figure_conservation(
        [res], [seed],
        seed_path(off, lattice, seed, f"{stem}_conservation.png"),
        off, lat, dpi=PER_SEED_DPI)


def _prune_seed_folders(folder, keep):
    """Remove best-seed folders an earlier run left for another seed.

    A case folder holds one seed folder, the current best seed's; one left by
    an earlier run of the same case (whose best seed was another) would read
    as a second.  Folders of the old one-per-seed layout (seed<N>/) are not
    this layout's to remove, so they are only reported.
    """
    if not os.path.isdir(folder):
        return
    own = re.compile(re.escape(BEST_SEED_DIR).replace(r"\{seed\}", r"\d+") + "$")
    mine = BEST_SEED_DIR.format(seed=keep) if keep is not None else None
    legacy = 0
    for d in sorted(os.listdir(folder)):
        if not os.path.isdir(os.path.join(folder, d)):
            continue
        if own.match(d) and d != mine:
            try:
                shutil.rmtree(os.path.join(folder, d))
                print(f"[best seed] removed {d}/ (an earlier run's best seed)",
                      flush=True)
            except OSError as exc:
                print(f"[best seed] WARNING: could not remove {d}/ ({exc})",
                      flush=True)
        elif re.fullmatch(r"seed\d+", d):
            legacy += 1
    if legacy:
        print(f"[best seed] note: {folder} still holds {legacy} seed<N>/ "
              f"folder(s) of the old per-seed layout; delete them by hand",
              flush=True)


def old_layout_files(root):
    """Files and folders the old one-folder-per-seed layout left in a results
    folder: every seed's seed<N>/ folder and r2 csv, the all-seed R2 figure,
    the seed matrix.  This layout never writes them, and never deletes them
    either -- they are an earlier run's outputs."""
    found = []
    if not os.path.isdir(root):
        return found
    for d in sorted(os.listdir(root)):
        path = os.path.join(root, d)
        if d.endswith("_seed_matrix.png"):
            found.append(d)
        elif os.path.isdir(path) and d.startswith("psi0"):
            found.append(f"{d}/ (one folder per case, the layout before the "
                         f"flat one)")
            for f in sorted(os.listdir(path)):
                if (re.fullmatch(r"seed\d+", f)
                        or f.endswith("_r2_all_seeds.png")
                        or re.search(r"_r2_seed\d+\.csv$", f)):
                    found.append(f"{d}/{f}")
    return found


def _r2_stack(results, key="r2"):
    """Every seed's series on the longest seed's times, NaN-padded."""
    n_max = max(len(r["r2"]) for r in results)
    stack = np.full((len(results), n_max), np.nan)
    for i, r in enumerate(results):
        v = np.asarray(r.get(key) if r.get(key) is not None else (),
                       dtype=np.float64)[:n_max]
        stack[i, :v.size] = v
    longest = results[int(np.argmax([len(r["r2"]) for r in results]))]
    return np.asarray(longest["times"], dtype=np.float64), stack


def _write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def _seed_digest(seeds):
    return hashlib.md5(",".join(str(int(s)) for s in seeds)
                       .encode()).hexdigest()[:12]


def write_case_outputs(results, seeds, off, lattice=None, key=None):
    """Every file one case produces, given its rollouts, into its folder.

    Split out of run_case so the same code can run on cached rollouts without
    the model (make_submission regenerates a folder that way).  The folder
    gets:

      * the case figures -- R2 with the best seed's snapshots, kinetics and
        conservation over the seeds -- and, unless WRITE_SUBMISSION is off,
        the submission figures, the film and their averaged data
        (make_submission.write_case);
      * the averages over every seed: MEAN_R2_FILE (R2 at each step with its
        spread, how many seeds it averages, and the mean mass drift) and the
        case summary;
      * the best seed's own folder (PER_SEED_OUTPUT), and no other seed's
        data.

    The summary is written last, atomically, with a record of what produced
    it: the cache key, the seeds and the outputs.  A later job reads that
    record to skip the case rather than roll it out again (finished_case),
    so it must only exist once everything else is on disk.  ``key`` is the
    rollouts' cache key; without one (a regeneration, a test) the case is
    never taken for finished.
    """
    label = MODEL_FLAGS["label"]
    suffix = case_suffix(off, lattice)
    lat = LATTICE if lattice is None else lattice
    seeds = tuple(seeds)
    tag = case_tag(off, lattice)
    bar = "=" * 72
    folder = case_dir(off, lattice)
    summary_path = case_path(off, lattice, f"{label}{suffix}_summary.json")
    # An earlier job's summary would mark the case finished while it is
    # being rewritten: it goes now and the new one is written last.
    try:
        os.remove(summary_path)
    except FileNotFoundError:
        pass

    summary = [seed_summary(r, sd) for r, sd in zip(results, seeds)]
    best, rank, _ = pick_result(results)

    if PER_SEED_OUTPUT:
        print(f"\n[best seed] {seeds[best]} (rank {rank} of {len(results)} "
              f"by mean R2) -> {os.path.join(folder, BEST_SEED_DIR.format(seed=seeds[best]))}",
              flush=True)
        write_seed_outputs(results[best], seeds[best], off, lattice, suffix,
                           label)
    _prune_seed_folders(folder, seeds[best] if PER_SEED_OUTPUT else None)

    figure_r2_and_snapshots(
        results, seeds,
        case_path(off, lattice, f"{label}{suffix}_r2_and_snapshots.png"),
        off, lat)
    figure_conservation(
        results, seeds,
        case_path(off, lattice, f"{label}{suffix}_conservation.png"), off, lat)

    times, stack = _r2_stack(results)
    _, drift = _r2_stack(results, "mass_drift")
    with np.errstate(invalid="ignore"):
        mean_curve = np.nanmean(stack, axis=0)
        spread = np.nanstd(stack, axis=0)
        mean_drift = np.nanmean(drift, axis=0)
    counts = np.sum(np.isfinite(stack), axis=0)
    mean_path = case_path(off, lattice, MEAN_R2_FILE)
    np.savetxt(mean_path + ".tmp",
               np.column_stack([times, counts, mean_curve, spread,
                                mean_drift]),
               delimiter=",", header="t,n_seeds,r2_mean,r2_std,"
                                     "mass_drift_mean",
               comments="", fmt=("%.18e", "%d", "%.18e", "%.18e", "%.18e"))
    os.replace(mean_path + ".tmp", mean_path)

    submission, submission_curve, submission_error = {}, None, None
    if WRITE_SUBMISSION:
        try:
            submission, submission_curve = write_submission_case(
                results, seeds, off, lattice, key=key)
        except Exception as exc:
            submission_error = f"{type(exc).__name__}: {exc}"
            print(f"\n[error] submission figures for {tag} FAILED; the "
                  f"other outputs of the case are unaffected:", flush=True)
            traceback.print_exc()

    per_seed = np.array([r["r2"].mean() for r in results])
    steps = np.array([row["steps_above_threshold"] for row in summary],
                     dtype=float)
    n = len(per_seed)
    sd = float(per_seed.std(ddof=1)) if n > 1 else 0.0
    def across(key, only=None):
        v = np.array([float(row[key]) for row in summary
                      if row[key] is not None and (only is None or only(row))],
                     dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "n": int(v.size)}
    stats = {
        "off": off,
        "lattice": lat,
        "directory": folder,
        "hard_constraints": MODEL_FLAGS["hard_constraints"],
        "hard_mass_conservation": MODEL_FLAGS["hard_mass_conservation"],
        "zero_dc": MODEL_FLAGS["zero_dc"],
        "n_seeds": n,
        "mean_r2_all_seeds": float(np.mean([v for r in results
                                            for v in r["r2"]])),
        "mean_r2_per_seed_mean": float(per_seed.mean()),
        "mean_r2_per_seed_std": sd,
        "mean_r2_standard_error": sd / np.sqrt(n) if n > 1 else 0.0,
        "mean_r2_worst_seed": float(per_seed.min()),
        "mean_r2_best_seed": float(per_seed[best]),
        "worst_seed": int(seeds[int(np.argmin(per_seed))]),
        "best_seed": int(seeds[best]),
        "best_seed_rank": rank,
        "median_seed": _submission().seed_ranks(results, seeds)[
            "median"]["seed"],
        "steps_above_threshold_total": int(steps.sum()),
        "steps_above_threshold_mean": float(steps.mean()),
        "seeds_holding_throughout": int(sum(1 for r in results
                                            if not r["stopped_early"])),
        "seeds_diverged": int(sum(1 for r in results
                                  if r.get("diverged_at") is not None)),
        "max_mass_drift": float(max(row["max_mass_drift"]
                                    for row in summary)),
        "max_mean_error": finite_max(row["max_mean_error"]
                                     for row in summary),
        # over the seeds: their growth exponents, and when R2 first fell
        # below the threshold for those that did
        "growth_exponent_truth": across("growth_exponent_truth"),
        "growth_exponent_fno": across("growth_exponent_fno"),
        # the seed-mean L(t): exponents on each side of the training
        # horizon, where the finite-size regime begins, and L_half
        "mean_length": length_summary(results, lat),
        "final_r2": across("final_r2"),
        "t_stop_of_stopped": across("t_stop",
                                    only=lambda row: row["stopped_early"]),
        "timing": timing_totals(summary),
        "record": {"key": key, "seeds": n, "seed_digest": _seed_digest(seeds),
                   "fingerprint": model_fingerprint(),
                   "numerics": numerics_now(),
                   "window": [START_TIME, END_TIME],
                   "submission": bool(WRITE_SUBMISSION
                                      and submission_error is None),
                   # a film the case really has: none when its cached
                   # rollouts carry no frames, or without the submission
                   "video": any(bool(e.get("film"))
                                for e in submission.values()),
                   # a resumed case records its entry again, under the keys
                   # the resuming run gives it (the curve from MEAN_R2_FILE)
                   "entry": (next(iter(submission.values()))
                             if submission else None),
                   "curve": ({k: submission_curve[k]
                              for k in ("n", "n_end", "n_div", "lattice")}
                             if submission_curve else None),
                   "figures": {"log_time": bool(log_time()),
                               "per_seed_output": bool(PER_SEED_OUTPUT),
                               "per_seed_figures": bool(PER_SEED_FIGURES)},
                   "complete": submission_error is None,
                   "written": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    _write_json_atomic(summary_path, stats)

    print("\n" + bar)
    for row in summary:
        print(f"  seed {row['seed']:>6}: R2 >= {R2_THRESHOLD} until t = "
              f"{row['t_stop']:>6.0f}  ({row['steps_above_threshold']:>4} "
              f"steps), mean R2 = {row['mean_r2']:.4f}, growth exponent "
              f"{row['growth_exponent_fno']:.3f} "
              f"(truth {row['growth_exponent_truth']:.3f})")
    print("-" * 72)
    print(f"  [{tag}] mean of all R2 = {stats['mean_r2_all_seeds']:.4f}  |  "
          f"per-seed mean {stats['mean_r2_per_seed_mean']:.4f} "
          f"+/- {stats['mean_r2_standard_error']:.4f} (s.e., n={n})")
    print(f"  worst seed {stats['worst_seed']} at "
          f"{stats['mean_r2_worst_seed']:.4f}, best seed "
          f"{stats['best_seed']} at {stats['mean_r2_best_seed']:.4f}, "
          f"{stats['seeds_holding_throughout']}/{n} held throughout, "
          f"{stats['steps_above_threshold_total']} steps total")
    print(bar, flush=True)

    case = {"off": off, "lattice": lat, "times": times,
            "mean_curve": mean_curve, "stats": stats,
            "submission": submission, "submission_curve": submission_curve}
    if submission_error:
        case["submission_error"] = submission_error
        case["submission_failed_keys"] = submission_keys(off, lattice)
    return case


def finished_case(off, lattice, seeds, key):
    """The case as an earlier job left it, if that job finished it; or None.

    Finished means the case summary is there (write_case_outputs writes it
    last) and its record matches this run: the same cache key -- model file,
    window, settings -- the same seeds, and at least the outputs this run
    wants.  A record that names a checkpoint fingerprint must name this
    model's; one from before fingerprints names none and still counts.  The
    case is rebuilt from its summary and MEAN_R2_FILE for the
    run-wide outputs; its figures and manifest entries stay as they are.
    With VIEWS every view must be finished, and the case comes back per view.
    """
    if key is None:
        return None
    if VIEWS:
        got = {}
        for h in VIEWS:
            with view_context(h):
                c = _finished_one(off, lattice, seeds, key)
            if c is None:
                return None
            got[h] = c
        return got
    return _finished_one(off, lattice, seeds, key)


def _finished_one(off, lattice, seeds, key):
    label = MODEL_FLAGS["label"]
    folder = case_dir(off, lattice)
    path = os.path.join(folder,
                        f"{label}{case_suffix(off, lattice)}_summary.json")
    try:
        with open(path) as fh:
            stats = json.load(fh)
        rec = stats.get("record") or {}
        figs = rec.get("figures") or {}
        # a record from before fingerprints names none, and still counts; so
        # does one from before the numerics stamp (supersede_old_runs has
        # already put aside the results of a run in other numerics)
        made_by = rec.get("fingerprint")
        if not (rec.get("complete") and rec.get("key") == key
                and (made_by is None or made_by == model_fingerprint())
                and ("numerics" not in rec
                     or full_numerics(rec["numerics"]) == numerics_now())
                and rec.get("seed_digest") == _seed_digest(seeds)
                and (rec.get("submission") or not WRITE_SUBMISSION)
                and (rec.get("video")
                     or not (WRITE_VIDEO and WRITE_SUBMISSION))
                and figs.get("log_time") == bool(log_time())
                and (figs.get("per_seed_output") or not PER_SEED_OUTPUT)
                and (figs.get("per_seed_figures") or not PER_SEED_FIGURES)):
            return None
        # the summary is in extras/, which belongs to the cache; the figures
        # it recorded must be in this run's folder (a run sent to another
        # --results-dir has none of them, and draws the case again)
        made = (rec.get("entry") or {}).get("files") or ()
        if WRITE_SUBMISSION and not all(
                os.path.isfile(os.path.join(RESULTS_DIR, f))
                or os.path.isfile(os.path.join(folder, f)) for f in made):
            return None
        data = np.loadtxt(os.path.join(folder, MEAN_R2_FILE), delimiter=",",
                          skiprows=1, ndmin=2)
    except (OSError, ValueError):
        return None
    lat = LATTICE if lattice is None else lattice
    submission, curve = {}, None
    if WRITE_SUBMISSION and isinstance(rec.get("entry"), dict):
        # recorded again, so an entry an interrupted job never got into
        # manifest.json (or a manifest that had to be restarted) comes back,
        # under the keys this run gives the case
        S = _submission()
        parts = rec.get("curve") or {}
        try:
            curve = S.curve_record({
                "times": data[:, 0], "mean_curve": data[:, 2],
                "sd_curve": data[:, 3],
                "lattice": parts.get("lattice", lat),
                "stats": {"n_seeds": parts["n"], "n_end": parts["n_end"],
                          "n_div": parts["n_div"]}})
        except (KeyError, TypeError, ValueError):
            curve = None
        if (curve is not None and S.curve_digest(curve)
                != rec["entry"].get("curve_digest")):
            curve = None
        submission = {k: dict(rec["entry"])
                      for k in submission_keys(off, lattice)}
    return {"off": off, "lattice": lat,
            "times": data[:, 0], "mean_curve": data[:, 2], "stats": stats,
            "submission": submission, "submission_curve": curve,
            "resumed": True}


def case_cache_files(off, lattice, seeds):
    """The files a case's rollouts are cached in while the run goes."""
    lat = LATTICE if lattice is None else lattice
    return [_result_path(off, s, lat) for s in seeds]


def drop_case_cache(off, lattice, seeds, keep=()):
    """Delete a finished case's cached rollouts, and any set-aside copies.

    Only this run's files (its cache key) go: rollouts of another model or
    window in the same folder are left alone, and so are the paths in
    ``keep`` (files another, unfinished case shares).
    """
    keep = {os.path.normcase(os.path.abspath(p)) for p in keep}
    n = 0
    for path in case_cache_files(off, lattice, seeds):
        if os.path.normcase(os.path.abspath(path)) in keep:
            continue
        if fno_paths.under(path, records_dir()):
            continue
        for f in [path] + glob.glob(glob.escape(path) + ".corrupt*"):
            try:
                os.remove(f)
                n += 1
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"[cache] WARNING: could not delete {f} ({exc})",
                      flush=True)
    if n:
        print(f"[cache] {case_tag(off, lattice)}: its outputs are written, so "
              f"its {n} cached rollout(s) are deleted", flush=True)
    return n


class _Track:
    """One seed's record in the lock-step engine -- evaluate_seed's, step for
    step, so a batch produces exactly the dict a per-seed run would.  The
    schedule engine uses it too: ``every`` = 1 measures at every schedule
    time, and ``schedule`` (how the rollout stepped) goes into the result
    with its start time."""

    def __init__(self, seed, mass0, want, every=None, schedule=None):
        self.seed, self.mass0, self.want = seed, mass0, list(want)
        self.every = DIAG_EVERY if every is None else every
        self.schedule = schedule
        self.times, self.r2s, self.drift = [], [], []
        self.mean_t, self.mean_p = [], []
        self.vid_t, self.vid_true, self.vid_pred = [], [], []
        self.diag_t, self.L_true, self.L_pred = [], {}, {}
        self.energy_t, self.energy_p, self.snaps = [], [], {}
        self.t_stop, self.stopped, self.diverged_at = END_TIME, False, None
        self.done = False
        self.steps = 0
        self.just_diverged = False

    def start_sample(self, t, mean):
        """The first sample of a schedule rollout, at its start: the state
        is the reference, so R2 = 1 exactly, with no drift and the same mean
        in truth and prediction (the group's curves start there too)."""
        self.times.append(float(t))
        self.r2s.append(1.0)
        self.drift.append(0.0)
        self.mean_t.append(float(mean))
        self.mean_p.append(float(mean))

    @property
    def running(self):
        """Still being rolled out: neither diverged nor stopped early."""
        return self.diverged_at is None and not self.done

    def _pad(self, t, true):
        """After a divergence: what evaluate_seed pads with, from the
        streamed truth rather than a stored reference."""
        self.times.append(t)
        self.r2s.append(R2_FLOOR)
        self.drift.append(self.drift[-1] if self.drift else 0.0)
        self.mean_t.append(float(true.mean()))
        self.mean_p.append(float("nan"))
        while self.want and t >= self.want[0] - 0.5 * DT_STEP:
            w = self.want.pop(0)
            if w <= END_TIME + 0.5 * DT_STEP:
                self.snaps[w] = (np.array(true, dtype=np.float64),
                                 np.full_like(true, np.nan, dtype=np.float64))

    def observe(self, k, t, true, pred):
        if self.done:
            return
        if self.diverged_at is not None:
            self._pad(t, true)
            return
        self.steps += 1
        finite = bool(np.all(np.isfinite(pred)))
        pmax = float(np.abs(pred).max()) if finite else float("inf")
        if not finite or pmax > DIVERGE_PSI:
            self.diverged_at, self.just_diverged = t, True
            print(f"    [diverged] seed {self.seed}: |psi| = {pmax:.3e} at "
                  f"t = {t:.0f}; scoring the remainder at R2 = {R2_FLOOR}",
                  flush=True)
            if not self.stopped:
                self.t_stop, self.stopped = t, True
            self._pad(t, true)
            return
        r2 = max(float(r2_score(true, pred)), R2_FLOOR)
        self.times.append(t)
        self.r2s.append(r2)
        self.drift.append(abs(float(pred.mean()) - self.mass0))
        self.mean_t.append(float(true.mean()))
        self.mean_p.append(float(pred.mean()))
        if WRITE_VIDEO and _video_frame(k, t):
            self.vid_t.append(t)
            self.vid_true.append(true.astype(VIDEO_DTYPE))
            self.vid_pred.append(pred.astype(VIDEO_DTYPE))
        if (k + 1) % self.every == 0 or k == 0:
            self.diag_t.append(t)
            for name, val in all_lengths(true).items():
                self.L_true.setdefault(name, []).append(val)
            for name, val in all_lengths(pred).items():
                self.L_pred.setdefault(name, []).append(val)
            self.energy_t.append(free_energy(true))
            self.energy_p.append(free_energy(pred))
        while self.want and t >= self.want[0] - 0.5 * DT_STEP:
            self.snaps[self.want.pop(0)] = (np.array(true, dtype=np.float64),
                                            np.array(pred, dtype=np.float64))
        if r2 < R2_THRESHOLD and not self.stopped:
            self.t_stop, self.stopped = t, True
            if EARLY_STOP:
                self.done = True

    def result(self, gt_seconds, fno_seconds):
        res = self._result(gt_seconds, fno_seconds)
        if self.schedule is not None:
            res.update(t_start=float(START_TIME), schedule=dict(self.schedule),
                       gt_cached=True)
        return res

    def _result(self, gt_seconds, fno_seconds):
        return {"times": np.asarray(self.times), "r2": np.asarray(self.r2s),
                "gt_seconds": gt_seconds, "gt_cached": False,
                "gt_steps": int(round(END_TIME / fno_train.DT)),
                "fno_seconds": fno_seconds, "fno_steps": self.steps,
                "mass_drift": np.asarray(self.drift),
                "mean_true": np.asarray(self.mean_t),
                "mean_pred": np.asarray(self.mean_p),
                "t_stop": self.t_stop, "diverged_at": self.diverged_at,
                "stopped_early": self.stopped,
                "diag_times": np.asarray(self.diag_t),
                "L_true": {k: np.asarray(v) for k, v in self.L_true.items()},
                "L_pred": {k: np.asarray(v) for k, v in self.L_pred.items()},
                "energy_true": np.asarray(self.energy_t),
                "energy_pred": np.asarray(self.energy_p),
                "snapshots": self.snaps,
                "video_t": np.asarray(self.vid_t),
                "video_true": (np.stack(self.vid_true) if self.vid_true
                               else None),
                "video_pred": (np.stack(self.vid_pred) if self.vid_pred
                               else None)}


def _forward(model, psi, sub):
    """model(psi) in chunks of ``sub`` seeds, halving the chunk on OOM."""
    while True:
        try:
            if sub >= psi.shape[0]:
                return model(psi), sub
            return torch.cat([model(psi[i:i + sub])
                              for i in range(0, psi.shape[0], sub)]), sub
        except torch.cuda.OutOfMemoryError:
            if sub <= 1:
                raise
            torch.cuda.empty_cache()
            sub = max(1, sub // 2)
            print(f"    [lockstep] out of GPU memory; forward passes now "
                  f"{sub} seeds at a time", flush=True)


@torch.no_grad()
def evaluate_seeds(model, seeds, device, off=0.0, lattice=None):
    """The lock-step engine: a batch of seeds rolled out together.

    evaluate_seed solves each reference to END_TIME, stores it, then rolls
    the model against it: at END_TIME = 20000 that is 2.6 GB (l = 128) to
    10.5 GB (l = 256) per seed, and 50 seeds one after another take about 70 h
    per model on an A100.  Here the truth for every seed is advanced one model
    step at a time with fno_train.ch_step -- the update _integrate uses, so
    the frames are bitwise those of the stored references -- and the model is
    applied to the whole batch.  Nothing but the current frame is kept.

    Every per-step quantity is computed on the host by the same numpy
    functions evaluate_seed calls, in HOST_WORKERS threads, and each seed's
    result has exactly evaluate_seed's keys, so the caches, figures and
    submission code need no change.  The model's forward is batched, which
    is not bitwise the per-seed forward (convolution and FFT kernels depend
    on the batch shape); the cache key records the engine, so the two kinds
    of rollout never mix.

    gt_seconds is the batch's solver time divided by its seeds; fno_seconds
    is the batch's model time per seed-step times the steps this seed took,
    as the per-seed engine counts it.  Once no seed is still running the
    model is not called, and once every seed has stopped early or diverged
    the loop ends: the series are then cut at that time, as the seeds that
    diverged are scored R2 = -1 only while the batch is still being rolled.
    """
    lattice = LATTICE if lattice is None else lattice
    dev = torch.device(device)
    n_seed = len(seeds)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    every = int(round(DT_STEP / fno_train.DT))
    n_start = int(round(START_TIME / fno_train.DT))
    j0 = int(round(START_TIME / DT_STEP))
    n_model = int(round((END_TIME - START_TIME) / DT_STEP))
    dtype = next(model.parameters()).dtype
    model.eval()

    t0 = time.perf_counter()
    p = torch.as_tensor(np.stack([fno_train.initial_condition(lattice, s, off)
                                  for s in seeds]),
                        dtype=torch.float64, device=dev)
    for _ in range(n_start):
        p = fno_train.ch_step(p)
    sync()
    gt_time = time.perf_counter() - t0
    first = p.cpu().numpy()
    want = sorted(SNAPSHOT_TIMES)
    tracks = []
    for i, seed in enumerate(seeds):
        w = list(want)
        tr = _Track(seed, float(first[i].mean()), [])
        if w and abs(w[0] - START_TIME) < 0.5 * DT_STEP:
            tr.snaps[w.pop(0)] = (np.array(first[i]), np.array(first[i]))
        tr.want = w
        tracks.append(tr)
    psi = p.to(dtype)[:, None]
    del first

    sub = FNO_SUB_BATCH or n_seed
    fno_time = 0.0
    pool = (concurrent.futures.ThreadPoolExecutor(HOST_WORKERS)
            if HOST_WORKERS > 1 and n_seed > 1 else None)
    print(f"    [lockstep] {n_seed} seeds together, l={lattice}, "
          f"t = {START_TIME:.0f}..{END_TIME:.0f} ({n_model:,} model steps, "
          f"{n_model * every:,} solver steps each)", flush=True)
    try:
        model_steps = 0
        for k in range(n_model):
            if all(tr.done or tr.diverged_at is not None for tr in tracks):
                break
            ta = time.perf_counter()
            for _ in range(every):
                p = fno_train.ch_step(p)
            sync()
            tb = time.perf_counter()
            if any(tr.running for tr in tracks):
                psi, sub = _forward(model, psi, sub)
                model_steps += 1
            sync()
            tc = time.perf_counter()
            gt_time += tb - ta
            fno_time += tc - tb
            true_h = p.cpu().numpy()
            pred_h = psi[:, 0].to(torch.float64).cpu().numpy()
            t = float((j0 + k + 1) * DT_STEP)
            jobs = [functools.partial(tr.observe, k, t, true_h[i], pred_h[i])
                    for i, tr in enumerate(tracks)]
            if pool is not None:
                for f in [pool.submit(j) for j in jobs]:
                    f.result()
            else:
                for j in jobs:
                    j()
            for i, tr in enumerate(tracks):
                if tr.just_diverged:
                    psi[i].zero_()
                    tr.just_diverged = False
            if (k + 1) % 200 == 0 or k + 1 == n_model:
                live = [tr.r2s[-1] for tr in tracks
                        if tr.diverged_at is None and not tr.done and tr.r2s]
                print(f"    t = {t:8.1f} | mean R2 = "
                      f"{(np.mean(live) if live else float('nan')):7.4f} over "
                      f"{len(live)} running | {(tc - ta) * 1e3:6.1f} ms/step",
                      flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    per_step = fno_time / (n_seed * max(model_steps, 1))
    out = []
    for tr in tracks:
        out.append(tr.result(gt_time / n_seed, per_step * tr.steps))
        tr.vid_true = tr.vid_pred = []      # one copy of the frames at a time
    return out


def ref_cache():
    """Where the 4.0-spaced references are kept (fno_gt: the group format)."""
    return REF_CACHE or fno_paths.ref_cache_dir(ROOT)


def schedule_must_hit():
    """The times every schedule visits: the snapshots, the views, the end."""
    got = {float(t) for t in SNAPSHOT_TIMES} | {float(h) for h in VIEWS or ()}
    got.add(float(END_TIME))
    return sorted(t for t in got if START_TIME < t <= END_TIME + 1e-9)


def tune_seeds():
    """TUNE_SEEDS without any seed the run also tests on (EVAL_SEEDS or
    SWEEP_SEEDS): a schedule picked on a trajectory it is then scored on
    would be flattered.  The group protocol refuses such a run; here the seed
    is left out (check_tune_seeds says so), as --seed-set fno shares 90210."""
    test = {int(s) for s in (*EVAL_SEEDS, *SWEEP_SEEDS)}
    return [int(s) for s in TUNE_SEEDS if int(s) not in test]


def check_tune_seeds():
    """Warn of the tune seeds left out because the run tests on them, and
    stop when none is left to tune on."""
    keep = tune_seeds()
    left = [int(s) for s in TUNE_SEEDS if int(s) not in keep]
    if left:
        print(f"[warn] tune seeds {left} are also test seeds; the schedule "
              f"is tuned on the other {len(keep)} of {len(TUNE_SEEDS)}, "
              f"never on a seed it is scored on", flush=True)
    if not keep:
        raise SystemExit("every tune seed is also a test seed, so there is "
                         "nothing to tune the schedule on; use --no-tune or "
                         "other seeds")


def tune_signature(end=None):
    """What a tuned pick depends on besides the model."""
    if not TUNE:
        return ("fixed", SCHED_ALPHA, SCHED_DT_MAX)
    return ("tuned", START_TIME, END_TIME if end is None else end,
            tuple(TUNE_ALPHAS), tuple(TUNE_DT_MAXS), tuple(tune_seeds()))


def schedule_times(alpha, dt_max, dt_min=None, must_hit=None):
    """The times of a rollout: START_TIME to END_TIME, strides alpha * t
    between dt_min and dt_max on the reference spacing, must-hit times in."""
    return fno_gt.make_schedule(
        START_TIME, END_TIME, alpha, SCHED_DT_MIN if dt_min is None else dt_min,
        dt_max, quantum=fno_gt.GT_EVERY,
        must_hit=schedule_must_hit() if must_hit is None else must_hit)


def tune_report():
    """Where the tuning is scored: T/4, T/2, 3T/4 and T, on the reference
    grid (the group protocol's report times)."""
    t0, t1, ev = START_TIME, END_TIME, fno_gt.GT_EVERY
    rep = sorted({round(t0 + round((x - t0) / ev) * ev, 6)
                  for x in (t1 / 4, t1 / 2, 0.75 * t1, t1)
                  if t0 + 1e-9 < x <= t1 + 1e-9})
    return rep or [float(t1)]


def frame_index(gt_t, times):
    """Index of the reference frame at each schedule time (all on the grid)."""
    gt_t = np.asarray(gt_t, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    idx = np.clip(np.searchsorted(gt_t, times - 1e-6), 0, len(gt_t) - 1)
    bad = np.abs(gt_t[idx] - times) >= 1e-6
    if bad.any():
        raise ValueError(f"no reference frame at t = {times[bad][:5]}; the "
                         f"window and the snapshots must lie on the "
                         f"{fno_gt.GT_EVERY:g}-spaced grid from {START_TIME:g}")
    return idx


# Solver seconds of each reference this job solved, {(l, off, seed): s}: the
# batch's time over its trajectories.  The schedule engine reads its references
# from files, so this is what lets a result say that its reference was solved
# (by the prewarm) and what that cost, instead of coming from the cache.
_REF_SECONDS: dict[tuple[int, float, int], float] = {}


def solve_references(l, off, seeds, device):
    """Solve the references of ``seeds`` the cache lacks (fno_gt) and note
    what each cost in _REF_SECONDS.  Returns how many were solved."""
    cache = ref_cache()
    missing = [int(s) for s in dict.fromkeys(seeds)
               if fno_gt.find_gt(cache, l, off, int(s), START_TIME, END_TIME,
                                 fno_gt.GT_EVERY) is None]
    t0 = time.perf_counter()
    n = fno_gt.ground_truth_batch(cache, l, off, seeds, START_TIME, END_TIME,
                                  fno_gt.GT_EVERY, fno_gt.SOLVER_DT, device)
    if n:
        each = (time.perf_counter() - t0) / n
        _REF_SECONDS.update({(l, off, s): each for s in missing})
    return n


def load_references(l, off, seeds, device):
    """[(times, fields)] of the references up to END_TIME, the frames memory
    mapped; any that are missing are solved first (fno_gt)."""
    cache = ref_cache()
    solve_references(l, off, seeds, device)
    out = []
    for s in seeds:
        path = fno_gt.find_gt(cache, l, off, s, START_TIME, END_TIME,
                              fno_gt.GT_EVERY)
        if path is None:
            raise FileNotFoundError(f"no reference for l={l} off={off:+.1f} "
                                    f"seed {s} in {cache}")
        out.append(fno_gt.load_gt(path, END_TIME))
    return out


def check_reference_space(need):
    """Before anything is solved: print what the missing references need and
    stop when the disk cannot hold them.  ``need`` maps (l, off) to seeds.
    Returns (bytes needed, files needed)."""
    cache = ref_cache()
    total = files = 0
    for (lat, off), seeds in need.items():
        n = sum(1 for s in dict.fromkeys(int(s) for s in seeds)
                if fno_gt.find_gt(cache, lat, off, s, START_TIME, END_TIME,
                                  fno_gt.GT_EVERY) is None)
        files += n
        total += n * fno_gt.gt_file_bytes(lat, START_TIME, END_TIME,
                                          fno_gt.GT_EVERY)
    free = fno_paths.free_bytes(cache)
    print(f"[gt] need {total / 1e9:.1f} GB for {files} files; "
          f"{free / 1e9:.1f} GB free in {cache}", flush=True)
    if not files:
        return total, files
    if total > free - CACHE_MARGIN:
        raise SystemExit(
            f"[gt] the missing references need {total / 1e9:.1f} GB but "
            f"{cache} has {free / 1e9:.1f} GB free (keeping "
            f"{CACHE_MARGIN / 1e9:.0f} GB back).  Nothing was solved.  Point "
            f"FNO_GT_CACHE at a disk with room (on the cluster "
            f"/scratch/data/<user>/fno_gt_cache).")
    in_slurm = bool(os.environ.get("SLURM_JOB_ID"))
    if in_slurm and not os.environ.get("FNO_GT_CACHE") and not REF_CACHE:
        print("[gt] WARNING: FNO_GT_CACHE is not set, so the references go to\n"
              f"[gt]   {cache}\n"
              f"[gt]   and need {total / 1e9:.1f} GB there.  /iitjhome has a "
              "200 GB hard quota per user that df and\n"
              "[gt]   disk_usage do not show.  Pass FNO_GT_CACHE=/scratch/data/"
              "<user>/fno_gt_cache in the sbatch --export line.",
              flush=True)
        if total > HOME_CACHE_LIMIT:
            raise SystemExit(
                f"[gt] {total / 1e9:.1f} GB of references would go to {cache} "
                f"on /iitjhome (limit {HOME_CACHE_LIMIT / 1e9:.0f} GB, the "
                f"quota is 200 GB).  Nothing was solved.  Set FNO_GT_CACHE to "
                f"a folder on /scratch/data and submit again.")
    return total, files


def prewarm_references(device, skip_done=True):
    """Solve, or find in the cache, the references the run needs, before the
    first rollout: the seeds of each case of the grid, and the tune seeds of
    each quench still to be tuned.  A case an earlier job finished needs none
    (it is skipped with ``skip_done``), and neither does a case kept from an
    earlier job: it is drawn from its records.  A job stopped by its time limit has kept
    what it solved."""
    t0 = time.perf_counter()
    need: dict[tuple[int, float], list[int]] = {}
    done = {name for name, info in (_META.get("cases") or {}).items()
            if info.get("complete")} if skip_done else set()
    for off, lattice, seeds in grid_cases():
        if _case_name(off, lattice) in done:
            continue
        lat = LATTICE if lattice is None else lattice
        want = need.setdefault((lat, off), [])
        want += [int(s) for s in seeds if int(s) not in want]
        if TUNE and stored_pick(off) is None:
            base = need.setdefault((LATTICE, off), [])
            base += [s for s in tune_seeds() if s not in base]
    print(f"\n[gt] prewarm to t = {END_TIME:g} in {ref_cache()}", flush=True)
    check_reference_space(need)
    total = solved = 0
    for (lat, off), seeds in need.items():
        n = solve_references(lat, off, seeds, device)
        total += len(seeds)
        solved += n
        print(f"[gt]   l={lat} off={off:+.1f}: {len(seeds)} trajectories, "
              f"{n} solved, {len(seeds) - n} cached", flush=True)
    print(f"[gt] {total} trajectories ready ({solved} solved now) in "
          f"{time.perf_counter() - t0:.0f} s", flush=True)


@torch.no_grad()
def roll_schedule(model, psi0, times, device, on_step):
    """The model's strides over ``times`` from psi0 (b, l, l).

    After each stride on_step(i, psi) sees the state at times[i] -- (b, 1, l,
    l), on the device -- and may edit it (a diverged seed is set to 0); it
    returns whether any seed is still alive, and when none is the model is
    not called again.  Returns (index of the last time reached, model calls,
    model seconds)."""
    dev = torch.device(device)
    dtype = next(model.parameters()).dtype
    model.eval()
    psi = torch.as_tensor(psi0, dtype=dtype, device=dev).unsqueeze(1)
    b = psi.shape[0]
    calls, seconds, last = 0, 0.0, 0
    for i in range(1, len(times)):
        dt = torch.full((b,), float(times[i] - times[i - 1]), device=dev,
                        dtype=dtype)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ta = time.perf_counter()
        psi = fno_train.step(model, psi, dt)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        seconds += time.perf_counter() - ta
        calls += 1
        last = i
        if not on_step(i, psi):
            break
    return last, calls, seconds


def score_schedule(model, truth, times, device):
    """R2 of every seed at every time of a rollout (len(times), b), for the
    tuning: a seed that diverges (|psi| > DIVERGE_PSI, or not finite) is
    scored R2_FLOOR from there on.  truth is (len(times), b, l, l)."""
    n, b = truth.shape[:2]
    r2 = np.full((n, b), R2_FLOOR)
    r2[0] = 1.0
    alive = torch.ones(b, dtype=torch.bool, device=device)
    n_div = [0]

    def on_step(i, psi):
        nonlocal alive
        newly = alive & ~fno_train.alive_mask(psi, DIVERGE_PSI)
        if bool(newly.any()):
            alive = alive & ~newly
            psi[newly] = 0.0
            n_div[0] += int(newly.sum())
        tr = torch.as_tensor(truth[i], dtype=psi.dtype,
                             device=device).unsqueeze(1)
        sse = ((psi - tr) ** 2).sum(dim=(1, 2, 3)).double()
        sst = ((tr - tr.mean(dim=(1, 2, 3), keepdim=True)) ** 2).sum(
            dim=(1, 2, 3)).double()
        ok = alive.cpu().numpy()
        r2[i] = np.where(ok, np.maximum((1.0 - sse / sst).cpu().numpy(),
                                        R2_FLOOR), R2_FLOOR)
        return bool(ok.any())

    roll_schedule(model, truth[0], times, device, on_step)
    return r2, n_div[0]


def tune_edges(pick, table, alphas=None, dt_maxs=None):
    """Where a pick sits on its grid and how many candidates diverged.

    Returns {"alpha_smallest", "alpha_largest", "dt_max_smallest",
    "dt_max_largest" (bools), "k", "n"}: k candidates of the n in ``table``
    had at least one diverged tuning seed.  The grid is ``alphas`` and
    ``dt_maxs`` when given, else the one in the pick's signature (read with
    ast.literal_eval, so an old run is judged on its own grid), else the
    current TUNE_ALPHAS and TUNE_DT_MAXS."""
    if alphas is None or dt_maxs is None:
        grid = None
        try:
            sig = ast.literal_eval(str(pick.get("signature")))
            if sig[0] == "tuned":
                grid = (tuple(sig[3]), tuple(sig[4]))
        except (ValueError, SyntaxError, IndexError, TypeError):
            grid = None
        a_grid, d_grid = grid or (tuple(TUNE_ALPHAS), tuple(TUNE_DT_MAXS))
        alphas = a_grid if alphas is None else alphas
        dt_maxs = d_grid if dt_maxs is None else dt_maxs
    a, d = float(pick["alpha"]), float(pick["dt_max"])
    rows = list(table or [])
    return {"alpha_smallest": a <= min(alphas) + 1e-9,
            "alpha_largest": a >= max(alphas) - 1e-9,
            "dt_max_smallest": d <= min(dt_maxs) + 1e-9,
            "dt_max_largest": d >= max(dt_maxs) - 1e-9,
            "k": sum(1 for r in rows if (r.get("n_diverged") or 0) > 0),
            "n": len(rows)}


def tune_lines(edges):
    """The [tune] lines for a pick: the count, then one per edge."""
    out = [f"  [tune] {edges['k']} of {edges['n']} candidates had at least "
           f"one diverged tuning seed"]
    if edges["dt_max_smallest"]:
        out.append("  [tune] warning: dt_max is the smallest of the grid; a "
                   "smaller dt_max was not tried")
    if edges["dt_max_largest"]:
        out.append("  [tune] warning: dt_max is the largest of the grid; a "
                   "larger dt_max was not tried")
    if edges["alpha_smallest"]:
        out.append("  [tune] warning: alpha is the smallest of the grid; a "
                   "smaller alpha was not tried")
    if edges["alpha_largest"]:
        out.append("  [tune] warning: alpha is the largest of the grid; a "
                   "larger alpha was not tried")
    return out


def tune_schedule(model, device, off):
    """(alpha, dt_max, table, score): the schedule of a quench, picked on the
    tune seeds -- never a test seed -- by the mean R2 at tune_report()."""
    seeds = tune_seeds()
    report = tune_report()
    hits = sorted(set(report) | set(schedule_must_hit()))
    refs = load_references(LATTICE, off, seeds, device)
    gt_t = refs[0][0]
    print(f"[tuning] schedule for psi0 = {off:+.1f} on {len(seeds)} "
          f"validation seeds {seeds}", flush=True)
    print(f"[tuning] score = mean R2 at t = "
          f"{', '.join(f'{t:g}' for t in report)}", flush=True)
    print(f"  {'alpha':>7} {'dt_max':>7} {'steps':>6} {'score':>9}"
          + "".join(f"{'R2(' + format(t, 'g') + ')':>11}" for t in report)
          + f"{'div':>5}", flush=True)
    table = []
    for a in TUNE_ALPHAS:
        for dm in TUNE_DT_MAXS:
            times = schedule_times(a, dm, must_hit=hits)
            idx = frame_index(gt_t, times)
            truth = np.stack([np.asarray(f[idx]) for _, f in refs], axis=1)
            r2, n_div = score_schedule(model, truth, times, device)
            mean = r2.mean(axis=1)
            at = [float(mean[int(np.argmin(np.abs(times - t)))])
                  for t in report]
            score = float(np.mean(at))
            table.append({"alpha": float(a), "dt_max": float(dm),
                          "steps": len(times) - 1, "score": score,
                          "n_diverged": n_div})
            print(f"  {a:7.3f} {dm:7.1f} {len(times) - 1:6d} {score:9.4f}"
                  + "".join(f"{v:11.4f}" for v in at) + f"{n_div:5d}",
                  flush=True)
            del truth
    best = max(table, key=lambda r: r["score"])
    print(f"  best: alpha={best['alpha']:g}, dt_max={best['dt_max']:g} "
          f"({best['steps']} steps) -> score {best['score']:.4f}", flush=True)
    for line in tune_lines(tune_edges(best, table, TUNE_ALPHAS, TUNE_DT_MAXS)):
        print(line, flush=True)
    return best["alpha"], best["dt_max"], table, best["score"]


def pick_key(off):
    return f"{off:+.1f}"


def stored_pick(off):
    """The quench's pick from an earlier job (meta.json), when it was made
    for these settings; else None."""
    got = _PICKS.get(pick_key(off))
    if got and got.get("signature") == repr(tune_signature()):
        return got
    return None


def schedule_pick(model, device, off):
    """(alpha, dt_max) of a quench: the stored pick, else tuned on the tune
    seeds, else the SCHED_ values.  A new pick is stored in meta.json."""
    got = stored_pick(off)
    if got is not None:
        print(f"[schedule] psi0 = {off:+.1f}: alpha {got['alpha']:g}, dt_max "
              f"{got['dt_max']:g} (from an earlier job: {got['source']})",
              flush=True)
        return got
    table, score = [], None
    if TUNE:
        alpha, dt_max, table, score = tune_schedule(model, device, off)
        source = f"tuned on {len(tune_seeds())} validation seeds"
    else:
        alpha, dt_max = SCHED_ALPHA, SCHED_DT_MAX
        source = "SCHED_ALPHA / SCHED_DT_MAX"
    pick = {"alpha": float(alpha), "dt_max": float(dt_max),
            "dt_min": float(SCHED_DT_MIN), "source": source,
            "signature": repr(tune_signature()), "score": score,
            "t_start": float(START_TIME), "t_end": float(END_TIME),
            "table": table}
    _PICKS[pick_key(off)] = pick
    if _META:
        _META.setdefault("schedule", {})[pick_key(off)] = pick
        _write_meta(_META)
    print(f"[schedule] psi0 = {off:+.1f}: alpha {alpha:g}, dt_max {dt_max:g} "
          f"({source})", flush=True)
    return pick


def same_schedule(res, pick, times):
    """Whether a cached rollout was made with this pick and these times."""
    s = res.get("schedule")
    return bool(s and abs(float(s["alpha"]) - pick["alpha"]) < 1e-9
                and abs(float(s["dt_max"]) - pick["dt_max"]) < 1e-9
                and int(s["steps"]) == len(times) - 1)


@torch.no_grad()
def evaluate_schedule(model, seeds, device, refs, times, pick):
    """The schedule engine: a chunk of seeds rolled out together.

    Each starts from its reference at START_TIME and takes the schedule's
    strides, the model told each one's length.  At every time of the schedule
    the seed is measured against the reference frame there by the lock-step
    engine's _Track (R2, mass, both lengths, free energy, the snapshots);
    a seed that diverges is scored R2 = -1 from there to the end of the
    schedule, stays in the means, and once no seed is alive the model is not
    called again (the references are still read, for the truth means).
    Returns ([result per seed], model calls).  Every result has t_start, its
    schedule and the absolute times, which begin at t_start with R2 = 1 (the
    state is the reference; has_start_sample) and go on with each stride's
    end."""
    n = len(times)
    truth = np.stack([np.asarray(f[frame_index(t, times)]) for t, f in refs],
                     axis=1)
    sched = {"kind": "schedule", "alpha": float(pick["alpha"]),
             "dt_min": float(pick["dt_min"]), "dt_max": float(pick["dt_max"]),
             "steps": n - 1, "quantum": float(fno_gt.GT_EVERY)}
    want = sorted(SNAPSHOT_TIMES)
    tracks = []
    for b, seed in enumerate(seeds):
        w = list(want)
        first = np.array(truth[0, b], dtype=np.float64)
        tr = _Track(seed, float(first.mean()), [], every=1, schedule=sched)
        tr.start_sample(float(times[0]), tr.mass0)
        if w and abs(w[0] - START_TIME) < 0.5 * DT_STEP:
            tr.snaps[w.pop(0)] = (first, first.copy())
        tr.want = w
        tracks.append(tr)
    pool = (concurrent.futures.ThreadPoolExecutor(HOST_WORKERS)
            if HOST_WORKERS > 1 and len(seeds) > 1 else None)
    print(f"    [schedule] {len(seeds)} seeds together, l="
          f"{truth.shape[-1]}, t = {times[0]:g}..{times[-1]:g} "
          f"({n - 1} steps, alpha {sched['alpha']:g}, dt_max "
          f"{sched['dt_max']:g})", flush=True)

    def on_step(i, psi):
        true_h = truth[i].astype(np.float64)
        pred_h = psi[:, 0].to(torch.float64).cpu().numpy()
        jobs = [functools.partial(tr.observe, i - 1, float(times[i]),
                                  true_h[b], pred_h[b])
                for b, tr in enumerate(tracks)]
        if pool is not None:
            for f in [pool.submit(j) for j in jobs]:
                f.result()
        else:
            for j in jobs:
                j()
        for b, tr in enumerate(tracks):
            if tr.diverged_at is not None:
                psi[b].zero_()
            tr.just_diverged = False
        if i % 50 == 0:
            live = [tr.r2s[-1] for tr in tracks if tr.running and tr.r2s]
            print(f"    t = {times[i]:8.1f} | mean R2 = "
                  f"{(np.mean(live) if live else float('nan')):7.4f} over "
                  f"{len(live)} running", flush=True)
        return any(tr.running for tr in tracks)

    try:
        last, calls, seconds = roll_schedule(model, truth[0], times, device,
                                             on_step)
        if last < n - 1:
            print(f"    [schedule] no seed alive at t = {times[last]:g}; the "
                  f"model is not called for the remaining "
                  f"{n - 1 - last} step(s)", flush=True)
        for i in range(last + 1, n):
            true_h = truth[i].astype(np.float64)
            for b, tr in enumerate(tracks):
                tr.observe(i - 1, float(times[i]), true_h[b], None)
    finally:
        if pool is not None:
            pool.shutdown()
    per_step = seconds / (len(seeds) * max(calls, 1))
    return [tr.result(0.0, per_step * tr.steps) for tr in tracks], calls


def schedule_case(model, device, off, lattice, seeds, key):
    """The rollouts of one case by the schedule engine: those an earlier job
    cached (with this pick) are taken, the others rolled out in chunks of
    ROLLOUT_CELLS // l^2 seeds."""
    lat = LATTICE if lattice is None else lattice
    pick = schedule_pick(model, device, off)
    times = schedule_times(pick["alpha"], pick["dt_max"], pick["dt_min"])
    results: list = [None] * len(seeds)
    todo = []
    for i, seed in enumerate(seeds):
        cache = _result_path(off, seed, lat)
        res = _load_cached(cache)
        if res is not None and not same_schedule(res, pick, times):
            print(f"  -> {cache} has another schedule; rolling out again",
                  flush=True)
            res = None
        if res is None:
            todo.append(i)
        else:
            results[i] = res
            keep_record(off, lattice, seed, res, key)
    per = max(1, int(ROLLOUT_CELLS // (lat * lat)))
    for c in range(0, len(todo), per):
        chunk = todo[c:c + per]
        names = [seeds[i] for i in chunk]
        print(f"\n[seeds] {', '.join(str(s) for s in names)}", flush=True)
        refs = load_references(lat, off, names, device)
        got, _ = evaluate_schedule(model, names, device, refs, times, pick)
        for i, res in zip(chunk, got):
            solve = _REF_SECONDS.get((lat, off, seeds[i]))
            if solve is not None:
                # this job solved the reference (in the prewarm): it did not
                # come from the cache, and the solve is what it cost
                res.update(gt_cached=False, gt_seconds=solve)
            results[i] = res
            if CACHE_RESULTS:
                _cache_rollout(_result_path(off, seeds[i], lat), res)
            keep_record(off, lattice, seeds[i], res, key)
    return results


def truncate_results(res, horizon, every=None, snaps=None):
    """A rollout seen only up to ``horizon``: what a run with END_TIME =
    horizon would have recorded.

    Rollouts are deterministic, so a run to 20000 contains the run to 10000
    as its prefix; this cuts every series at the horizon and re-reads the
    stopping state from what happened before it.  With ``every`` the film
    keeps only the frames a run to that horizon would have made (the run
    recorded frames for every view's cadence, VIDEO_SCHEDULE); with
    ``snaps`` only that run's snapshot times are kept (the run recorded every
    view's, so the 10000 view would otherwise carry the 2000 view's t = 500).
    """
    h = float(horizon) + 0.5 * DT_STEP
    out = dict(res)
    n = int(np.searchsorted(np.asarray(res["times"], dtype=float), h))
    for k in ("times", "r2", "mass_drift", "mean_true", "mean_pred"):
        if res.get(k) is not None:
            out[k] = np.asarray(res[k])[:n]
    nd = int(np.searchsorted(np.asarray(res["diag_times"], dtype=float), h))
    out["diag_times"] = np.asarray(res["diag_times"])[:nd]
    for k in ("L_true", "L_pred"):
        out[k] = {name: np.asarray(v)[:nd] for name, v in res[k].items()}
    for k in ("energy_true", "energy_pred"):
        out[k] = np.asarray(res[k])[:nd]
    out["snapshots"] = {t: v for t, v in res["snapshots"].items()
                        if t <= h and (snaps is None or any(
                            abs(t - s) < 0.5 * DT_STEP for s in snaps))}
    vt = res.get("video_t")
    if vt is not None and len(vt):
        vt = np.asarray(vt, dtype=float)
        keep = vt <= h
        if every:
            k = np.rint((vt - START_TIME) / DT_STEP).astype(int) - 1
            keep &= (k == 0) | ((k + 1) % every == 0)
        idx = np.nonzero(keep)[0]
        if idx.size and idx[-1] == idx.size - 1:
            pick = slice(0, idx.size)          # a prefix: no copy
        else:
            pick = idx
        out["video_t"] = np.asarray(res["video_t"])[pick]
        for key in ("video_true", "video_pred"):
            if res.get(key) is not None:
                out[key] = res[key][pick]
    div = res.get("diverged_at")
    out["diverged_at"] = div if div is not None and div <= h else None
    if res.get("stopped_early") and res["t_stop"] <= h:
        out["t_stop"], out["stopped_early"] = res["t_stop"], True
    else:
        out["t_stop"], out["stopped_early"] = float(horizon), False
    full = res.get("fno_steps") or 0
    if res.get("schedule"):
        # a stride per time of the schedule, which has no fixed step (and the
        # start sample is no stride)
        strides = n - int(has_start_sample(res))
        steps = min(full, strides)
        out["schedule"] = dict(res["schedule"], steps=strides)
    else:
        steps = min(full, int(round((float(horizon) - START_TIME) / DT_STEP)))
    out["fno_steps"] = steps
    if res.get("fno_seconds") is not None and full:
        out["fno_seconds"] = res["fno_seconds"] * steps / full
    if res.get("gt_steps"):
        new = int(round(float(horizon) / fno_train.DT))
        if res.get("gt_seconds") is not None:
            out["gt_seconds"] = res["gt_seconds"] * new / res["gt_steps"]
        out["gt_steps"] = new
    return out


def view_dir(horizon):
    """RESULTS_DIR for one view: RESULTS_TEMPLATE with {H} filled in."""
    return (RESULTS_TEMPLATE or RESULTS_DIR).format(H=f"{horizon:g}")


@contextlib.contextmanager
def view_context(horizon):
    """Write one view's outputs as if the run had ended at ``horizon``."""
    global END_TIME, SNAPSHOT_TIMES, RESULTS_DIR, CACHE_END_TIME
    global _RUN_SNAPSHOTS
    global VIDEO_EVERY
    saved = (END_TIME, SNAPSHOT_TIMES, RESULTS_DIR, CACHE_END_TIME,
             _RUN_SNAPSHOTS, VIDEO_EVERY)
    if CACHE_END_TIME is None:
        CACHE_END_TIME, _RUN_SNAPSHOTS = END_TIME, tuple(SNAPSHOT_TIMES)
    END_TIME = float(horizon)
    if SNAPSHOTS_EXPLICIT:
        SNAPSHOT_TIMES = tuple(t for t in saved[1] if t <= END_TIME + 1e-9)
    else:
        # the snapshots a run to this horizon would have had -- (100, 500,
        # 1000, 2000) for the 2000 view, not the 20000 run's set cut short
        SNAPSHOT_TIMES = tuple(sorted(set(default_snapshots(START_TIME,
                                                            END_TIME))))
    if VIDEO_SCHEDULE:
        VIDEO_EVERY = dict(VIDEO_SCHEDULE).get(END_TIME, VIDEO_EVERY)
    RESULTS_DIR = view_dir(horizon)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        yield
    finally:
        (END_TIME, SNAPSHOT_TIMES, RESULTS_DIR, CACHE_END_TIME,
         _RUN_SNAPSHOTS, VIDEO_EVERY) = saved


def _load_cached(cache):
    """A cached rollout, or None if absent or damaged (then moved aside)."""
    if not (CACHE_RESULTS and os.path.exists(cache)):
        return None
    try:
        res = _load_result(cache)
    except Exception as exc:
        if isinstance(exc, MemoryError) or (
                isinstance(exc, OSError)
                and exc.errno not in (None, errno.EINVAL)):
            print(f"  -> could not read {cache} "
                  f"({type(exc).__name__}: {exc}); left in place",
                  flush=True)
            raise
        bad = cache + ".corrupt"
        if os.path.exists(bad):
            bad += f".{time.time_ns()}"
        try:
            os.replace(cache, bad)
            where = f"moved aside as {os.path.basename(bad)}"
        except OSError:
            where = "it will be overwritten"
        print(f"  -> cached rollout unreadable ({type(exc).__name__});"
              f" {where}, rolling out again", flush=True)
        return None
    print(f"  -> reusing {cache}", flush=True)
    return res


def _cache_rollout(cache, res):
    """Cache one rollout for a resumed job.  The cache is only a resume aid,
    so a failure to write it (a full quota) is reported and the case goes
    on: its outputs need a fraction of that space."""
    try:
        _save_result(cache, res)
    except Exception as exc:
        print(f"  -> could not cache {cache} ({type(exc).__name__}: {exc}); "
              f"going on without it", flush=True)
        return
    print(f"  -> cached {cache}", flush=True)


def gt_batch_size(lat):
    """GT_BATCH, or fewer when that many float64 references to END_TIME
    would not fit in GT_BATCH_GB (3 at l = 256 to t = 2000)."""
    per_seed = (int(round(END_TIME / DT_STEP)) + 1) * lat * lat * 8
    return max(1, min(GT_BATCH, int(GT_BATCH_GB * 1e9 // per_seed)))


def _solve_missing_references(device, off, lat, seeds):
    """Solve, GT_BATCH at a time, the references the seed engine is about to
    need and does not have.

    A seed whose rollout is already cached needs none, and with GT_CACHE off
    nothing is kept, so each reference is solved when its seed comes up.
    Returns {seed: solver seconds per seed} for the seeds it solved, which
    run_case puts on their rollouts as evaluate_seed would have.
    """
    if ENGINE != "seed" or not GT_CACHE:
        return {}
    need = []
    for seed in dict.fromkeys(seeds):
        if CACHE_RESULTS and os.path.exists(_result_path(off, seed, lat)):
            continue
        if not os.path.exists(_gt_cache_path(lat, seed, off, END_TIME)):
            need.append(seed)
    solved = {}
    step = gt_batch_size(lat)
    for c in range(0, len(need), step):
        chunk = need[c:c + step]
        t0 = time.perf_counter()
        ch_ground_truth_batch(lat, chunk, off, END_TIME, DT_STEP,
                              cache_dir=GT_CACHE, device=device)
        solved.update(dict.fromkeys(
            chunk, (time.perf_counter() - t0) / len(chunk)))
    return solved


def run_case(model, device, off, lattice=None, seeds=None):
    """One case: roll every seed out, then write everything it produces.

    With ENGINE = "lockstep" the seeds not already cached are rolled out
    EVAL_BATCH at a time by evaluate_seeds; with "schedule" (a dt-conditioned
    model) by schedule_case; otherwise one by one, as always, after the
    references they lack are solved GT_BATCH at a time.
    With VIEWS the case is written once per horizon and a dict
    {horizon: case} is returned instead of the case.
    """
    lat = LATTICE if lattice is None else lattice
    seeds = seeds_for(off) if seeds is None else tuple(seeds)
    tag = case_tag(off, lattice)
    bar = "=" * 72
    print(f"\n{bar}\n[case] {tag}: l={lat}, {len(seeds)} seeds, "
          f"t={START_TIME:.0f}..{END_TIME:.0f}"
          + (f", {ENGINE} engine" if ENGINE != "seed" else "")
          + f"\n[case] outputs -> {case_dir(off, lattice)}\n{bar}",
          flush=True)

    # worked out here, outside any view: inside one the key differs
    key = _cache_key(off, lat) if os.path.exists(MODEL_PATH) else None
    results: list = [None] * len(seeds)
    solved = _solve_missing_references(device, off, lat, seeds)
    if ENGINE == "schedule":
        results = schedule_case(model, device, off, lattice, seeds, key)
    elif ENGINE == "lockstep":
        todo = []
        for i, seed in enumerate(seeds):
            cache = _result_path(off, seed, lat)
            res = _load_cached(cache)
            if res is None:
                todo.append(i)
            else:
                results[i] = res
                keep_record(off, lattice, seed, res, key)
        for c in range(0, len(todo), max(1, EVAL_BATCH)):
            chunk = todo[c:c + max(1, EVAL_BATCH)]
            print(f"\n[seeds] {', '.join(str(seeds[i]) for i in chunk)}",
                  flush=True)
            got = evaluate_seeds(model, [seeds[i] for i in chunk], device,
                                 off, lat)
            for i, res in zip(chunk, got):
                results[i] = res
                if CACHE_RESULTS:
                    _cache_rollout(_result_path(off, seeds[i], lat), res)
                keep_record(off, lattice, seeds[i], res, key)
    else:
        for i, seed in enumerate(seeds):
            print(f"\n[seed {seed}] l={lat}, off={off:+.2f}", flush=True)
            cache = _result_path(off, seed, lat)
            res = _load_cached(cache)
            if res is None:
                res = evaluate_seed(model, seed, device, off, lat)
                if seed in solved:
                    res.update(gt_cached=False, gt_seconds=solved[seed])
                if CACHE_RESULTS:
                    _cache_rollout(cache, res)
            keep_record(off, lattice, seed, res, key)
            results[i] = res

    done = [r for r in results if r is not None]
    for seed, res in zip(seeds, done):
        n_above = n_steps_above(res)
        verdict = (f"R2 first fell below {R2_THRESHOLD} at t = "
                   f"{res['t_stop']:.0f}" if res["stopped_early"] else
                   f"held R2 >= {R2_THRESHOLD} through t = {END_TIME:.0f}")
        if res.get("diverged_at") is not None:
            verdict += f"; ROLLOUT DIVERGED at t = {res['diverged_at']:.0f}"
        print(f"  seed {seed}: {verdict}; {n_above} steps above threshold, "
              f"mean R2 = {res['r2'].mean():.4f}", flush=True)

    return emit_case(done, seeds, off, lattice, key)


def emit_case(done, seeds, off, lattice, key):
    """The outputs of a case from its rollouts: once, or once per view, each
    cut at its horizon.  A redraw from records writes them through this too."""
    if VIEWS:
        cases = {}
        for h in VIEWS:
            every = dict(VIDEO_SCHEDULE).get(h) if VIDEO_SCHEDULE else None
            with view_context(h):
                cases[h] = write_case_outputs(
                    [truncate_results(r, h, every, SNAPSHOT_TIMES)
                     for r in done], seeds, off, lattice, key=key)
        return cases
    return write_case_outputs(done, seeds, off, lattice, key=key)


def case_from_records(off, lattice, seeds):
    """A case kept from an earlier job, drawn from its records: nothing is
    rolled out and no reference is asked for.  The records must be this
    checkpoint's and be the case's seeds in order."""
    name = _case_name(off, lattice)
    got = [load_record(record_path(off, lattice, s)) for s in seeds]
    keys = {w["cache_key"] for _, w in got}
    prints = {w["fingerprint"] for _, w in got}
    now = model_fingerprint()
    if (len(keys) > 1 or len(prints) > 1
            or [w["seed"] for _, w in got] != [int(s) for s in seeds]
            or (now and prints != {now})):
        raise ValueError(
            f"the records of {name} do not belong together or to this "
            f"checkpoint: cache keys {sorted(map(str, keys))}, checkpoints "
            f"{sorted(map(str, prints))}, this run {now}")
    print(f"\n[case] {case_tag(off, lattice)}: kept from an earlier job; "
          f"{len(got)} seeds drawn from their records, nothing rolled out",
          flush=True)
    return emit_case([r for r, _ in got], seeds, off, lattice,
                     key=next(iter(keys)))


def parse_args(argv=None):
    """One-off overrides, so a single run needs no edit to this file.

    Everything here defaults to None and leaves the module config alone, so
    running with no flags behaves exactly as before.
    """
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--eval-mode",
                    choices=("all", "critical", "off_critical", "both",
                             "none"),
                    help=f"which compositions to evaluate; 'none' runs only "
                         f"the lattice sweep (file: {EVAL_MODE})")
    ap.add_argument("--seeds", type=int, metavar="N",
                    help=f"how many unseen seeds (file: {len(EVAL_SEEDS)}); "
                         f"with --seed-set group, the first N of its "
                         f"{len(GROUP_SEEDS)}")
    ap.add_argument("--seed-set", choices=SEED_SETS,
                    help=f"fno: this project's seeds, make_eval_seeds (the "
                         f"default for a fixed-step model); group: the "
                         f"{len(GROUP_SEEDS)} test seeds of the group protocol, to "
                         f"score both on the same trajectories (the default "
                         f"for a dt-conditioned model; file: {SEED_SET})")
    ap.add_argument("--sizes", metavar="L,L,...",
                    help="box sizes; every composition of --eval-mode is run "
                         "at each of them, the base lattice included (file: "
                         + ",".join(str(x) for x in SIZE_SWEEP) + ")")
    ap.add_argument("--no-video", action="store_true",
                    help="skip recording and rendering rollout videos")
    ap.add_argument("--no-submission", action="store_true",
                    help="skip the submission figures, film, manifest and "
                         "README with their averaged P(psi) and L(t) (the "
                         "fno_ch figures, mean_r2.csv and summary stay)")
    ap.add_argument("--no-diagnostics", action="store_true",
                    help=f"skip the two checks of the loaded model (tile "
                         f"invariance, low-k gain) that are otherwise made "
                         f"once and written to {DIAG_FILE} in each results "
                         f"folder")
    ap.add_argument("--no-gt-cache", action="store_true",
                    help="solve each reference without writing it to disk "
                         "(one l=256 reference is ~1 GB; use this for a "
                         "small top-up on a quota-limited filesystem)")
    ap.add_argument("--model", metavar="PATH",
                    help=f"checkpoint to evaluate (file: {MODEL_PATH}); the "
                         f"results are named for its folder, so "
                         f"Work/models/train_t1000/fno_ch.pt gives "
                         f"Results/train_t1000_pred_t02000")
    ap.add_argument("--results-dir", metavar="DIR",
                    help="where results go; with --views it may contain {H} "
                         "or {H:0>5}, the view's horizon (default with a "
                         "horizon flag or --model: Results/<train label>_"
                         "pred_t<end>, e.g. Results/train_t1000_pred_t20000)")
    ap.add_argument("--result-cache", metavar="DIR",
                    help=f"where rollouts are cached while the run goes "
                         f"(file: {RESULT_CACHE}; default with a horizon "
                         f"flag or --model: Work/rollout_cache/<run label>, "
                         f"named as the results folder is)")
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep the cached rollouts once a case's outputs are "
                         "written, instead of deleting them")
    ap.add_argument("--no-records", action="store_true",
                    help="keep no permanent record of each seed and no "
                         "meta.json (by default both sit beside the "
                         "temporary cache and survive it; "
                         "python make_submission.py --records redraws from "
                         "them)")
    ap.add_argument("--redo", action="store_true",
                    help="roll out again cases an earlier job already "
                         "finished into these folders (drop it when "
                         "resubmitting a stopped --redo job, or the cases it "
                         "already redid are redone once more)")
    ap.add_argument("--no-supersede", action="store_true",
                    help="stop, instead of moving to Archive/superseded, "
                         "when a results folder this run would write holds "
                         "results of another checkpoint")
    ap.add_argument("--gt-cache-dir", metavar="DIR",
                    help=f"stored references: the seed engine's (file: "
                         f"{GT_CACHE}), or the schedule engine's 4.0-spaced "
                         f"ones (FNO_GT_CACHE, else Work/gt_cache)")
    ap.add_argument("--start-time", type=float, metavar="T",
                    help=f"rollouts start here (file: {START_TIME:g})")
    ap.add_argument("--end-time", type=float, metavar="T",
                    help=f"and end here (file: {END_TIME:g}); beyond 2000 "
                         f"this chooses the lock-step engine, scaled "
                         f"snapshots and film cadence")
    ap.add_argument("--views", metavar="H,H,...",
                    help="write outputs for each horizon from one rollout "
                         "to the longest, e.g. 2000,10000,20000")
    ap.add_argument("--snapshots", metavar="t,t,...",
                    help="snapshot times (default: scaled to the horizon)")
    ap.add_argument("--video-every", type=int, metavar="N",
                    help="film every N model steps (default: scaled to the "
                         "horizon, 10 at 2000)")
    ap.add_argument("--diag-every", type=int, metavar="N",
                    help=f"length/energy diagnostics every N steps (file: "
                         f"{DIAG_EVERY})")
    ap.add_argument("--engine", choices=("seed", "lockstep", "schedule"),
                    help="one seed at a time with stored references, or a "
                         "batch of seeds with the reference streamed (a "
                         "fixed-step model); a dt-conditioned model always "
                         "takes the schedule engine, which is not a choice")
    ap.add_argument("--no-tune", action="store_true",
                    help=f"schedule engine: use alpha {SCHED_ALPHA:g}, dt_max "
                         f"{SCHED_DT_MAX:g} instead of tuning each quench on "
                         f"the validation seeds")
    ap.add_argument("--video", action="store_true",
                    help="lock-step engine: record and render the film (off "
                         "there by default; the seed engine always does, "
                         "the schedule engine has none)")
    ap.add_argument("--batch", type=int, metavar="B",
                    help=f"lock-step seeds per batch (file: {EVAL_BATCH})")
    ap.add_argument("--gt-batch", type=int, metavar="N",
                    help=f"seed engine: missing references solved N at a time "
                         f"(file: {GT_BATCH}); a batch is held in memory "
                         f"whole, about 1 GB a seed at l=256")
    ap.add_argument("--log-time", choices=("auto", "on", "off"),
                    help="logarithmic time axes (auto: windows over 20x)")
    ap.add_argument("--smoke", action="store_true",
                    help="a tiny end-to-end run of the newest model that "
                         "python fno_train.py --smoke made (2 seeds, the "
                         "critical case at l = 32 and 48, t = 100..120); it "
                         "writes only under Work/smoke and loads no other "
                         "checkpoint; the other flags still apply on top")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run on the CPU even inside a SLURM job that has no "
                         "GPU (which is refused otherwise; FNO_ALLOW_CPU=1 "
                         "does the same)")
    return ap.parse_args(argv)


def _times(flag, text):
    """A comma-separated list of times from the command line."""
    try:
        return tuple(sorted({float(x) for x in text.split(",") if x.strip()}))
    except ValueError:
        raise SystemExit(f"{flag} {text!r}: expected numbers separated by "
                         f"commas, e.g. 2000,10000,20000")


def apply_gt_batch(args, moved):
    """--gt-batch: how many missing references the seed engine solves
    together."""
    global GT_BATCH
    if args.gt_batch is None:
        return
    if args.gt_batch < 1:
        raise SystemExit(f"--gt-batch must be at least 1 "
                         f"(got {args.gt_batch})")
    moved.append(f"GT_BATCH {GT_BATCH} -> {args.gt_batch}")
    GT_BATCH = args.gt_batch


def apply_horizon(args, moved):
    """Horizon flags, and the settings that follow from the horizon.

    A run given a horizon flag or --model gets a results folder and a rollout
    cache of its own, named for the model and the window (so it never lands
    among the t = 2000 results), snapshot times that include the horizon (the
    submission judges a case at its last snapshot), a film of about 190
    frames, and -- beyond 2000 -- the lock-step engine.  With views,
    each view gets the snapshots and film a run to that horizon would have
    had.  Each is only a default: the matching flag overrides it.  Every
    value is checked here, before any reference is solved.
    """
    global START_TIME, END_TIME, SNAPSHOT_TIMES, VIDEO_EVERY, DIAG_EVERY
    global ENGINE, EVAL_BATCH, VIEWS, RESULTS_DIR, RESULTS_TEMPLATE, LOG_TIME
    global VIDEO_SCHEDULE, VIDEO_DTYPE, SNAPSHOTS_EXPLICIT, RESULT_CACHE
    global WRITE_VIDEO
    if CONDITIONED:
        if args.engine in ("seed", "lockstep"):
            raise SystemExit(f"--engine {args.engine}: a dt-conditioned model "
                             f"takes a step of any length and is run by the "
                             f"schedule engine only")
        ENGINE = "schedule"
    elif args.engine == "schedule":
        raise SystemExit("--engine schedule needs a dt-conditioned model; "
                         "this one takes a fixed step")
    views: tuple[float, ...] = (_times("--views", args.views) if args.views
                                else ())
    end = args.end_time
    if views and end is None:
        end = max(views)
    horizon = bool(args.start_time is not None or end is not None or views)
    if args.start_time is not None:
        moved.append(f"START_TIME {START_TIME:g} -> {args.start_time:g}")
        START_TIME = float(args.start_time)
    if end is not None:
        moved.append(f"END_TIME {END_TIME:g} -> {end:g}")
        END_TIME = float(end)
    for name, v in (("--start-time", START_TIME), ("--end-time", END_TIME),
                    *((f"view {h:g}", h) for h in views)):
        if abs(v / DT_STEP - round(v / DT_STEP)) > 1e-9:
            raise SystemExit(f"{name} {v:g} is not a multiple of the model "
                             f"step {DT_STEP:g}")
        k = (v - START_TIME) / fno_gt.GT_EVERY
        if ENGINE == "schedule" and abs(k - round(k)) > 1e-9:
            raise SystemExit(f"{name} {v:g} is not on the reference grid "
                             f"({START_TIME:g} + a multiple of "
                             f"{fno_gt.GT_EVERY:g})")
    if END_TIME <= START_TIME:
        raise SystemExit(f"the rollout must end after it starts (start "
                         f"{START_TIME:g}, end {END_TIME:g})")
    if START_TIME < 0:
        raise SystemExit(f"--start-time {START_TIME:g} is before t = 0")
    if views:
        if max(views) > END_TIME + 1e-9 or min(views) <= START_TIME:
            raise SystemExit(f"--views {args.views}: every view must lie in "
                             f"({START_TIME:g}, {END_TIME:g}]")
        VIEWS = views
        moved.append(f"VIEWS -> {', '.join(f'{v:g}' for v in views)}")
    if args.snapshots:
        SNAPSHOT_TIMES = _times("--snapshots", args.snapshots)
        SNAPSHOTS_EXPLICIT = True
        moved.append(f"SNAPSHOT_TIMES -> {SNAPSHOT_TIMES}")
    elif horizon:
        want = set(default_snapshots(START_TIME, END_TIME)) | set(views)
        for h in views:
            want |= set(default_snapshots(START_TIME, h))
        SNAPSHOT_TIMES = tuple(sorted(want))
        moved.append(f"SNAPSHOT_TIMES -> {SNAPSHOT_TIMES} (scaled)")
    for h in (*views, END_TIME):
        if not any(abs(h - t) < 1e-9 for t in SNAPSHOT_TIMES):
            raise SystemExit(f"t = {h:g} has no snapshot, so it could not be "
                             f"judged there; add it to --snapshots")
    if ENGINE == "schedule":
        for t in SNAPSHOT_TIMES:
            k = (t - START_TIME) / fno_gt.GT_EVERY
            if abs(k - round(k)) > 1e-9:
                raise SystemExit(f"snapshot t = {t:g} is not on the reference "
                                 f"grid ({START_TIME:g} + a multiple of "
                                 f"{fno_gt.GT_EVERY:g}); the schedule must "
                                 f"hit it")
    if any(t < START_TIME - 1e-9 or t > END_TIME + 1e-9
           for t in SNAPSHOT_TIMES):
        raise SystemExit(f"snapshots {SNAPSHOT_TIMES} must lie in "
                         f"[{START_TIME:g}, {END_TIME:g}]")
    for flag, val in (("--video-every", args.video_every),
                      ("--diag-every", args.diag_every),
                      ("--batch", args.batch)):
        if val is not None and val < 1:
            raise SystemExit(f"{flag} must be at least 1 (got {val})")
    if args.video_every is not None:
        VIDEO_EVERY = args.video_every
        moved.append(f"VIDEO_EVERY -> {VIDEO_EVERY}")
    elif horizon:
        VIDEO_EVERY = default_video_every(START_TIME, END_TIME)
        if len(views) > 1:
            VIDEO_SCHEDULE = tuple((h, default_video_every(START_TIME, h))
                                   for h in views)
            VIDEO_DTYPE = np.float16
            moved.append("film per view: "
                         + ", ".join(f"every {e} to {h:g}"
                                     for h, e in VIDEO_SCHEDULE)
                         + " (float16 frames)")
        else:
            moved.append(f"VIDEO_EVERY -> {VIDEO_EVERY} (scaled)")
    if args.diag_every is not None:
        DIAG_EVERY = args.diag_every
        moved.append(f"DIAG_EVERY -> {DIAG_EVERY}")
    if ENGINE != "schedule":
        if args.engine:
            ENGINE = args.engine
        elif END_TIME > 2000.0 + 1e-9:
            ENGINE = "lockstep"
    if ENGINE != "seed":
        moved.append(f"ENGINE -> {ENGINE}")
    if WRITE_VIDEO and (ENGINE == "schedule"
                        or (ENGINE == "lockstep" and not args.video)):
        WRITE_VIDEO = False
        moved.append("WRITE_VIDEO True -> False (" + (
            "a schedule rollout has no film" if ENGINE == "schedule" else
            "the lock-step engine makes no film unless --video") + ")")
    if args.batch is not None:
        EVAL_BATCH = args.batch
        moved.append(f"EVAL_BATCH -> {EVAL_BATCH}")
    apply_gt_batch(args, moved)
    if args.log_time:
        LOG_TIME = args.log_time
    if args.results_dir:
        RESULTS_DIR = args.results_dir
    if horizon or args.model:
        # named for the model (its folder, or its file when it has none) and
        # the window; with views the folders are a template, the cache is for
        # the rollout to the longest
        train = fno_paths.model_label(MODEL_PATH)
        run = fno_paths.run_label(train, END_TIME, START_TIME)
        if not args.results_dir:
            RESULTS_DIR = fno_paths.results_dir(
                fno_paths.run_template(train, START_TIME) if views else run,
                ROOT)
        if not args.result_cache:
            RESULT_CACHE = fno_paths.rollout_dir(run, ROOT)
            moved.append(f"RESULT_CACHE -> {RESULT_CACHE!r}")
    try:
        RESULTS_DIR.format(H="0")
    except (KeyError, IndexError, ValueError):
        raise SystemExit(f"--results-dir {RESULTS_DIR!r} may contain {{H}} "
                         f"and no other braces")
    per_view = RESULTS_DIR.format(H="1") != RESULTS_DIR.format(H="2")
    if views:
        longest = max(views)
        if len(views) > 1 and not per_view:
            raise SystemExit("--views writes one results folder per horizon; "
                             "put {H} in --results-dir")
        RESULTS_TEMPLATE = RESULTS_DIR
        RESULTS_DIR = view_dir(longest)
    elif per_view:
        RESULTS_DIR = RESULTS_DIR.format(H=f"{END_TIME:g}")
    if args.results_dir or horizon or args.model:
        moved.append(f"RESULTS_DIR -> {RESULTS_TEMPLATE or RESULTS_DIR}")


def smoke_cases(conditioned=False):
    """The tiny set of cases and window of --smoke, and where it keeps them.

    make_submission --smoke takes the same, so that its rebuild asks the
    cache for just the cases this run made.  A dt-conditioned model's
    snapshots lie on the 4.0-spaced reference grid.  Returns what it set, in
    a line.
    """
    global ROOT, GT_CACHE, EVAL_SEEDS, SWEEP_SEEDS, EVAL_MODE, LATTICE
    global SIZE_SWEEP, START_TIME, END_TIME, SNAPSHOT_TIMES, DIAG_EVERY
    global VIDEO_EVERY, KEEP_CACHE, DIAG_TILE, DIAG_TILE_T, DIAG_LOWK_SEEDS
    global DIAG_LOWK_L, DIAG_LOWK_T_END
    ROOT = fno_paths.smoke_root()
    GT_CACHE = fno_paths.gt_dir(ROOT)
    EVAL_SEEDS = SWEEP_SEEDS = make_eval_seeds(2)
    EVAL_MODE, LATTICE, SIZE_SWEEP = "critical", 32, (32, 48)
    START_TIME, END_TIME = 100.0, 120.0
    SNAPSHOT_TIMES = ((100.0, 112.0, 120.0) if conditioned
                      else (100.0, 110.0, 120.0))
    DIAG_EVERY = VIDEO_EVERY = 2
    KEEP_CACHE = True
    DIAG_TILE, DIAG_TILE_T = LATTICE, START_TIME
    DIAG_LOWK_SEEDS, DIAG_LOWK_L, DIAG_LOWK_T_END = 1, LATTICE, START_TIME
    return (f"seeds {list(EVAL_SEEDS)} (the sweep too), EVAL_MODE "
            f"{EVAL_MODE!r}, LATTICE {LATTICE}, SIZE_SWEEP {SIZE_SWEEP}, "
            f"t = {START_TIME:g}..{END_TIME:g} with snapshots at "
            f"{SNAPSHOT_TIMES}, DIAG_EVERY {DIAG_EVERY}, VIDEO_EVERY "
            f"{VIDEO_EVERY}, KEEP_CACHE {KEEP_CACHE}, diagnostics at l = "
            f"{LATTICE} tiled to {2 * LATTICE} on 1 seed")


def apply_smoke(args):
    """The settings of --smoke: a tiny run of this very script on the newest
    model under Work/smoke/models, writing only under Work/smoke.

    Called right after parse_args, before the log is opened (so the log goes
    there too), before use_old_inputs (so the old fno_ch.pt can never stand
    in for the model) and before apply_overrides, so a flag beside --smoke
    still wins.  Returns the lines to print once the log is open.
    """
    global MODEL_PATH, RESULTS_DIR, RESULT_CACHE
    root = fno_paths.smoke_root()
    if args.model:
        if not fno_paths.under(args.model, root):
            raise SystemExit(f"--smoke never loads a production checkpoint: "
                             f"--model {args.model} is not under {root}")
        model = args.model
    else:
        model = fno_paths.latest_model(root)
        if model is None:
            raise SystemExit("no smoke model; run python fno_train.py "
                             "--smoke first")
    cases = smoke_cases(checkpoint_conditioned(model))
    MODEL_PATH = model
    run = fno_paths.run_label(fno_paths.model_label(model), END_TIME,
                              START_TIME)
    RESULTS_DIR = fno_paths.results_dir(run, ROOT)
    RESULT_CACHE = fno_paths.rollout_dir(run, ROOT)
    fno_runlog.LOG_DIR = fno_paths.log_dir(ROOT)
    return [f"writing only under {root}: model {model}, results "
            f"{RESULTS_DIR}, rollouts {RESULT_CACHE}; {cases}"]


def checkpoint_conditioned(path):
    """Whether the checkpoint at ``path`` is a dt-conditioned model (its
    config says so; one without the key, or no file, is fixed-step)."""
    if not path or not os.path.exists(path):
        return False
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return bool((ckpt.get("config") or {}).get("dt_conditioned", False))
    except Exception:
        return False


def apply_overrides(args):
    """Rebind whatever the CLI named, and say so in the log."""
    global EVAL_MODE, SIZE_SWEEP, EVAL_SEEDS, SWEEP_SEEDS, WRITE_VIDEO
    global GT_CACHE, WRITE_SUBMISSION, MODEL_PATH, RESULT_CACHE, REF_CACHE
    global KEEP_CACHE, REDO, SEED_SET, NO_SUPERSEDE, DIAGNOSTICS, RECORDS
    global CONDITIONED, TUNE
    moved = []
    if args.model:
        moved.append(f"MODEL_PATH {MODEL_PATH!r} -> {args.model!r}")
        MODEL_PATH = args.model
    if args.result_cache:
        moved.append(f"RESULT_CACHE -> {args.result_cache!r}")
        RESULT_CACHE = args.result_cache
    if args.gt_cache_dir:
        moved.append(f"GT_CACHE -> {args.gt_cache_dir!r}")
        GT_CACHE = args.gt_cache_dir
        REF_CACHE = args.gt_cache_dir
    # the kind of model decides the engine (and the seed set), so it is read
    # before the horizon is
    CONDITIONED = checkpoint_conditioned(MODEL_PATH)
    if CONDITIONED:
        moved.append("dt-conditioned model: schedule engine")
    if args.no_tune and TUNE:
        moved.append("TUNE True -> False (alpha, dt_max as configured)")
        TUNE = False
    apply_horizon(args, moved)
    if args.eval_mode:
        moved.append(f"EVAL_MODE {EVAL_MODE!r} -> {args.eval_mode!r}")
        EVAL_MODE = args.eval_mode
    # a dt-conditioned model is scored on the group protocol's test seeds unless told
    # otherwise (not in --smoke, which has its own two)
    seed_set = args.seed_set or ("group" if CONDITIONED and not args.smoke
                                 else None)
    if seed_set and seed_set != SEED_SET:
        moved.append(f"SEED_SET {SEED_SET!r} -> {seed_set!r}")
        SEED_SET = seed_set
    if args.seeds or seed_set == "group":
        follows = SWEEP_SEEDS == EVAL_SEEDS
        if SEED_SET == "group":
            n = args.seeds or len(GROUP_SEEDS)
            if not 1 <= n <= len(GROUP_SEEDS):
                raise SystemExit(f"--seeds {n}: the group seed set has "
                                 f"{len(GROUP_SEEDS)} seeds")
            seeds = GROUP_SEEDS[:n]
        else:
            seeds = make_eval_seeds(args.seeds)
        moved.append(f"seeds {len(EVAL_SEEDS)} -> {len(seeds)}"
                     + ("" if follows else "; SWEEP_SEEDS left alone"))
        EVAL_SEEDS = seeds
        if follows:
            SWEEP_SEEDS = EVAL_SEEDS
    if ENGINE == "schedule" and TUNE:
        check_tune_seeds()
    if args.sizes:
        sizes = tuple(int(x) for x in args.sizes.split(",") if x.strip())
        moved.append(f"SIZE_SWEEP {SIZE_SWEEP} -> {sizes}")
        SIZE_SWEEP = sizes
    if args.no_video and WRITE_VIDEO:
        moved.append("WRITE_VIDEO True -> False")
        WRITE_VIDEO = False
    if args.no_submission and WRITE_SUBMISSION:
        moved.append("WRITE_SUBMISSION True -> False")
        WRITE_SUBMISSION = False
    if args.keep_cache and not KEEP_CACHE:
        moved.append("KEEP_CACHE -> True (cached rollouts are kept)")
        KEEP_CACHE = True
    if args.no_records and RECORDS:
        moved.append("RECORDS True -> False (no per-seed records, no "
                     "meta.json)")
        RECORDS = False
    if args.redo and not REDO:
        moved.append("REDO -> True (finished cases are rolled out again)")
        REDO = True
    if args.no_supersede and not NO_SUPERSEDE:
        moved.append("NO_SUPERSEDE -> True (results of another checkpoint "
                     "stop the run)")
        NO_SUPERSEDE = True
    if args.no_diagnostics and DIAGNOSTICS:
        moved.append("DIAGNOSTICS True -> False (no tile or low-k checks)")
        DIAGNOSTICS = False
    if args.no_gt_cache and GT_CACHE:
        moved.append(f"GT_CACHE {GT_CACHE!r} -> off (references not kept)")
        GT_CACHE = ""
    for line in moved:
        print(f"[cli] {line}", flush=True)
    if not moved:
        print("[cli] no overrides; using the settings in this file",
              flush=True)
    return moved


def use_old_inputs():
    """Read the model and the references from where they were before the
    folder layout, if they are not at the new place yet and are still the
    file's own defaults (a flag that named others is left alone).  The
    results folder was named from the new default model, as train_t0500,
    before this.  Only inputs: what a run writes always goes to the new
    places."""
    global GT_CACHE, MODEL_PATH
    if GT_CACHE == fno_paths.gt_dir():
        GT_CACHE = fno_paths.legacy(GT_CACHE, "gt_cache", "the references")
    if MODEL_PATH == fno_paths.model_path(DEFAULT_MODEL):
        MODEL_PATH = fno_paths.legacy(MODEL_PATH, "fno_ch.pt", "the model")


def check_results_root():
    """Refuse to write into the old results/ under the name Results/.

    Where the filesystem ignores case they are one folder, and the first
    run would put its files among the old ones; python migrate_layout.py
    moves them out first.  Only a run that writes under the results root
    (Results/, or Work/smoke/Results with --smoke) is checked.
    """
    top = fno_paths.results_root(ROOT)
    if not fno_paths.under(RESULTS_TEMPLATE or RESULTS_DIR, top):
        return
    clash = fno_paths.case_clash(fno_paths.RESULTS, ROOT or ".")
    if clash:
        raise SystemExit(f"{clash}/ (the old results folder) is the same "
                         f"folder as {fno_paths.RESULTS}/ on this "
                         f"filesystem; run python migrate_layout.py first")


def setup_precision():
    """Run in the precision the model was trained in.  Without this PyTorch's
    default applies: cuDNN convolutions in TF32 whatever PRECISION says,
    which the training and its validation never had.  Returns whether TF32 is
    allowed."""
    fno_train.PRECISION = PRECISION
    tf32 = fno_train.set_precision_flags()
    flags = (f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}, "
             f"matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}")
    print(f"[setup] precision={PRECISION}, TF32 {'on' if tf32 else 'off'} "
          f"({flags}) as in training"
          + ("" if tf32 else "; PyTorch's default would run convolutions "
             "in TF32"), flush=True)
    return tf32


def apply_lowk_fill(model):
    """LOWK_FILL, when it is set, on the loaded model; and say what the run
    uses, from where."""
    if LOWK_FILL is not None and isinstance(model, torch.nn.Module):
        fno_train.set_lowk_fill(model, LOWK_FILL)
    fill = numerics_of(model)["lowk_fill"]
    print(f"[setup] low-k weight fill {'on' if fill else 'off'} "
          + ("(LOWK_FILL)" if LOWK_FILL is not None
             else "(the checkpoint's config)"), flush=True)


def apply_mu_clip(model):
    """MU_CLIP, when it is set, on the loaded model; and say what the run
    uses, from where."""
    if isinstance(model, torch.nn.Module):
        has_mu = bool(getattr(model, "mu_channel", False))
    else:
        has_mu = bool((getattr(model, "config", None) or {}).get("mu_channel"))
    if not has_mu:
        print("[setup] mu-channel clamp not used (the model has no mu "
              "channel)", flush=True)
        return
    if MU_CLIP is not None and isinstance(model, torch.nn.Module):
        fno_train.set_mu_clip(model, MU_CLIP)
    clip = numerics_of(model)["mu_clip"]
    print("[setup] mu-channel clamp "
          + ("off" if clip is None else f"|psi| <= {clip:g}") + " "
          + ("(MU_CLIP)" if MU_CLIP is not None
             else "(the checkpoint's config)"), flush=True)


# ---- timing mode (FNO_TIMING) ------------------------------------------
# One job times the solver and the networks on one device, with the code the
# real runs use, on a part of a run, and scales it up.  See README section 1.
TIMING_SIZES = (128, 256, 512)
TIMING_BATCH = 50
TIMING_SOLVER_T = 200.0          # solver time units timed (20,000 steps)
TIMING_CALLS = 20                # network calls timed, after 3 warm-up calls
TIMING_CALLS_PER_RUN = 320       # calls of a default run (318 to 321)
TIMING_FULL_CHECK = True         # also solve t = 0..10000 at the first size
TIMING_WINDOW_STEPS = 990000     # solver steps of t = 100..10000 at dt 0.01
TIMING_WARMUP_T = 10.0
TIMING_WARMUP_CALLS = 3
TIMING_FULL_T = 10000.0
# earlier real jobs, printed beside the result and never judged
TIMING_REFERENCE = {
    "solver_256_s_per_run": 24.2,
    "default_network_s_per_run": {"128": 1.0, "256": 9.1, "512": 52.3}}


def timing_knobs():
    """The timing constants, with the environment's overrides (for tests;
    real jobs use the defaults): FNO_TIMING_SIZES (":"-separated),
    FNO_TIMING_BATCH, FNO_TIMING_SOLVER_T, FNO_TIMING_CALLS and
    FNO_TIMING_FULL (0 or 1)."""
    env = os.environ.get
    sizes = env("FNO_TIMING_SIZES")
    full = env("FNO_TIMING_FULL")
    return {
        "sizes": (tuple(int(s) for s in sizes.split(":") if s)
                  if sizes else tuple(TIMING_SIZES)),
        "batch": int(env("FNO_TIMING_BATCH") or TIMING_BATCH),
        "solver_t": float(env("FNO_TIMING_SOLVER_T") or TIMING_SOLVER_T),
        "calls": int(env("FNO_TIMING_CALLS") or TIMING_CALLS),
        "full": TIMING_FULL_CHECK if not full else full.strip() == "1"}


def timing_models(spec, model_path=None):
    """[(name, checkpoint path)] the FNO_TIMING value asks for: "1" is the
    --model checkpoint (``model_path``, else MODEL_PATH), else ":"-separated labels or paths ending in .pt."""
    if spec.strip() == "1":
        path = model_path or MODEL_PATH
        if not os.path.isfile(path):
            raise SystemExit(f"[timing] no checkpoint {path}")
        return [(fno_paths.model_label(path), path)]
    out = []
    for item in [s.strip() for s in spec.split(":") if s.strip()]:
        if item.endswith(".pt"):
            path, name = item, fno_paths.model_label(item)
        else:
            path, name = fno_paths.model_path(item, ROOT), item
        if not os.path.isfile(path):
            raise SystemExit(f"[timing] FNO_TIMING: no checkpoint '{item}' "
                             f"(looked for {path})")
        out.append((name, path))
    if not out:
        raise SystemExit("[timing] FNO_TIMING names no model")
    return out


def fresh_folder(env_name, prefix):
    """The output folder of a measuring mode: the variable ``env_name``, else
    Work/analysis/<prefix>_<date>; never an existing one (_2, _3, ... is
    added)."""
    base = os.environ.get(env_name) or os.path.join(
        fno_paths._work("analysis", ROOT),
        prefix + "_" + time.strftime("%Y-%m-%d"))
    base = base.rstrip("/\\")
    path, k = base, 1
    while os.path.exists(path):
        k += 1
        path = f"{base}_{k}"
    return path


def timing_folder():
    """The output folder: FNO_TIMING_OUT, else Work/analysis/timing_<date>;
    never an existing one (_2, _3, ... is added)."""
    return fresh_folder("FNO_TIMING_OUT", "timing")


def _timing_sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def timing_solve(tmp, l, t_end, seeds, device):
    """Seconds fno_gt.ground_truth_batch takes to solve ``seeds`` from t = 0
    to ``t_end`` at composition 0 with the reference cache's dt and save
    spacing, writing the frames into a folder of its own under ``tmp`` (the
    same file writing a cache fill does), removed after."""
    folder = os.path.join(tmp, f"solve_l{l}_t{t_end:g}")
    try:
        _timing_sync(device)
        t0 = time.perf_counter()
        n = fno_gt.ground_truth_batch(folder, l, 0.0, list(seeds), 0.0, t_end,
                                      fno_gt.GT_EVERY, fno_gt.SOLVER_DT,
                                      device)
        _timing_sync(device)
        sec = time.perf_counter() - t0
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    if n != len(seeds):
        raise RuntimeError(f"[timing] solved {n} of {len(seeds)} seeds")
    return sec


@torch.no_grad()
def timing_network(model, l, nb, calls, device):
    """Seconds per call of fno_train.step on a batch of ``nb`` fields of l x l:
    3 warm-up calls, then ``calls`` timed.  A dt-conditioned model is given
    step lengths cycling through 1 .. SCHED_DT_MAX, as the schedule does."""
    model.eval()
    dtype = next(model.parameters()).dtype
    ic = np.stack([fno_gt.make_ic(l, int(s), 0.0)
                   for s in GROUP_SEEDS[:1] * nb])
    psi = torch.as_tensor(ic, dtype=dtype, device=device).unsqueeze(1)
    cond = bool(getattr(model, "dt_conditioned", False))
    cycle = (1.0, 2.0, 5.0, 10.0, 20.0, SCHED_DT_MAX)

    def call(i):
        dt = (torch.full((nb,), cycle[i % len(cycle)], device=device,
                         dtype=dtype) if cond else None)
        return fno_train.step(model, psi, dt)

    for i in range(TIMING_WARMUP_CALLS):
        call(i)
    _timing_sync(device)
    t0 = time.perf_counter()
    for i in range(calls):
        call(i)
    _timing_sync(device)
    return (time.perf_counter() - t0) / calls


def timing_table(rep):
    """The plain-text report of ``rep`` (the timing.json dict)."""
    m = rep["meta"]
    names = [x["name"] for x in rep["models"]]
    bar = "=" * 78
    out = [bar, "  TIMING (solver and networks, one device, one job)", bar,
           f"  device                    {m['device']}"
           + (f" ({m['gpu']})" if m["gpu"] else ""),
           f"  host                      {m['host']}",
           f"  torch / CUDA              {m['torch']} / {m['cuda']}",
           f"  precision                 {m['precision']}",
           f"  batch B                   {m['batch']} runs (solver); a "
           f"network call takes ROLLOUT_CELLS // l^2 of them, as a real run",
           f"  solver timed              t = 0 .. {m['solver_t']:g} = "
           f"{m['solver_steps']:,} steps; per run x "
           f"{m['solver_scale']:.4g} (990,000 steps = the rollout window "
           f"t = 100 .. 10000)",
           f"  network timed             {m['calls']} calls after "
           f"{m['warmup_calls']} warm-up; per run x {m['calls_per_run']} "
           f"calls (a default run makes 318 to 321)"]
    if m["cpu"]:
        out.append("  NOTE                      CPU timing: not comparable "
                   "with a GPU job")
    for x in rep["models"]:
        out.append(f"  model                     {x['name']}: "
                   f"{x['parameters']:,} parameters, "
                   f"{'dt-conditioned' if x['dt_conditioned'] else 'fixed step'}")
    out += ["", "  s per run = scaled to a whole run; ms per call = one "
            "network call on its batch.", ""]
    head = f"  {'l':>5} {'solver s':>9}"
    for n in names:
        head += f" | {n[:22]:>22} {'s/run':>8} {'ms/call':>8} {'speedup':>8}"
    out += [head, "  " + "-" * (len(head) - 2)]
    for r in rep["rows"]:
        line = f"  {r['l']:>5} {r['solver_s_per_run']:>9.2f}"
        for x in r["models"]:
            line += (f" | {x['name'][:22]:>22} {x['s_per_run']:>8.3f} "
                     f"{x['ms_per_call']:>8.2f} {x['speedup']:>7.1f}x")
        out.append(line)
    out.append("  (network batch per call: " + ", ".join(
        f"l={r['l']}: {r['network_batch']}" for r in rep["rows"]) + ")")
    v = rep.get("validation")
    out += ["", "  Validation: the whole solve against the short-span estimate"]
    if v is None:
        out.append("  not run (FNO_TIMING_FULL=0)")
    else:
        out += [f"  l = {v['l']}, B = {v['batch']}, t = 0 .. {v['t_end']:g}: "
                f"{v['full_seconds']:.1f} s in all",
                f"  per run, scaled to the window (x {v['scale']:.4g}): "
                f"{v['full_s_per_run']:.2f} s",
                f"  short-span estimate:                  "
                f"{v['short_s_per_run']:.2f} s",
                f"  difference (full - short) / short:    "
                f"{v['difference_percent']:+.1f} %"]
    ref = rep["reference"]
    out += ["", "  reference, other job (not compared, no pass or fail):",
            f"    solver at 256: {ref['solver_256_s_per_run']} s per run",
            "    default network, s per run: " + ", ".join(
                f"l={k}: {v}" for k, v in
                ref["default_network_s_per_run"].items()), bar]
    return "\n".join(out)


def run_timing(spec, args):
    """The timing mode (FNO_TIMING): writes timing.txt and timing.json to the
    output folder and nothing else.  Returns 0."""
    import socket
    knobs = timing_knobs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (device == "cpu" and not args.allow_cpu
            and os.environ.get("FNO_ALLOW_CPU") != "1"):
        raise SystemExit("[timing] no CUDA device; pass --allow-cpu for a CPU "
                         "timing")
    models = timing_models(spec, args.model)
    setup_precision()
    loaded = []
    for name, path in models:
        model, _ = load_checkpoint(path, device=device)
        apply_lowk_fill(model)
        apply_mu_clip(model)
        loaded.append((name, model))
    out = timing_folder()
    tmp = os.path.join(out, "_solver_tmp")
    os.makedirs(tmp)
    B, T = knobs["batch"], knobs["solver_t"]
    seeds = list(GROUP_SEEDS[:B])
    steps = int(round(T / fno_gt.SOLVER_DT))
    scale = TIMING_WINDOW_STEPS / steps
    if device == "cpu":
        print("[timing] CPU timing: not comparable with a GPU job", flush=True)
    rows = []
    try:
        for l in knobs["sizes"]:
            print(f"[timing] l = {l}: solver", flush=True)
            timing_solve(tmp, l, TIMING_WARMUP_T, seeds, device)
            sec = timing_solve(tmp, l, T, seeds, device)
            solver = sec / B * scale
            nb = min(B, max(1, int(ROLLOUT_CELLS // (l * l))))
            row = {"l": l, "solver_seconds": sec, "solver_s_per_run": solver,
                   "network_batch": nb, "models": []}
            for name, model in loaded:
                print(f"[timing] l = {l}: {name}", flush=True)
                per_call = timing_network(model, l, nb, knobs["calls"],
                                          device)
                share = per_call / nb * TIMING_CALLS_PER_RUN
                row["models"].append({
                    "name": name, "s_per_call": per_call,
                    "ms_per_call": 1e3 * per_call, "s_per_run": share,
                    "speedup": solver / share})
            rows.append(row)
        valid = None
        if knobs["full"] and rows:
            l = rows[0]["l"]
            print(f"[timing] validation: l = {l}, t = 0 .. "
                  f"{TIMING_FULL_T:g}", flush=True)
            sec = timing_solve(tmp, l, TIMING_FULL_T, seeds, device)
            vscale = TIMING_WINDOW_STEPS / int(round(TIMING_FULL_T
                                                     / fno_gt.SOLVER_DT))
            full = sec / B * vscale
            short = rows[0]["solver_s_per_run"]
            valid = {"l": l, "batch": B, "t_end": TIMING_FULL_T,
                     "full_seconds": sec, "scale": vscale,
                     "full_s_per_run": full, "short_s_per_run": short,
                     "difference_percent": 100.0 * (full - short) / short}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    dev = torch.device(device)
    rep = {"meta": {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        "host": socket.gethostname(), "torch": torch.__version__,
        "cuda": getattr(getattr(torch, "version", None), "cuda", None), "precision": PRECISION,
        "cpu": dev.type == "cpu", "batch": B, "solver_t": T,
        "solver_steps": steps, "solver_scale": scale,
        "calls": knobs["calls"], "warmup_calls": TIMING_WARMUP_CALLS,
        "calls_per_run": TIMING_CALLS_PER_RUN,
        "window_steps": TIMING_WINDOW_STEPS},
        "models": [{"name": n, "parameters": m.n_parameters(),
                    "dt_conditioned": bool(getattr(m, "dt_conditioned",
                                                   False))}
                   for n, m in loaded],
        "rows": rows, "validation": valid, "reference": TIMING_REFERENCE}
    text = timing_table(rep)
    with open(os.path.join(out, "timing.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(out, "timing.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(text, flush=True)
    print(f"[timing] written to {out}", flush=True)
    return 0


# ---- drift mode (FNO_DRIFT) --------------------------------------------
# How <psi> wanders during a rollout, with and without the final mean
# removal: one case, one common schedule, any number of models.  See README
# section 1.
DRIFT_OFF = 0.0
DRIFT_LATTICE = 128
DRIFT_T_END = 10000.0
DRIFT_ALPHA = 0.1                # used when the cache holds no pick
DRIFT_DT_MAX = 30.0
DRIFT_PICK_LABEL = "train_t0500_lap"
DRIFT_PICK_CACHE: str | None = None     # None: the label's rollout cache
DRIFT_NOMEAN = "+nomean"


def drift_entries(spec):
    """[{"spec", "label", "path", "nomean"}] the FNO_DRIFT value asks for:
    ":"-separated labels or paths ending in .pt, each with an optional
    +nomean.  A missing checkpoint stops the run here."""
    out = []
    for item in [s.strip() for s in spec.split(":") if s.strip()]:
        nomean = item.endswith(DRIFT_NOMEAN)
        base = item[:-len(DRIFT_NOMEAN)] if nomean else item
        if base.endswith(".pt"):
            path, label = base, fno_paths.model_label(base)
        else:
            path, label = fno_paths.model_path(base, ROOT), base
        if not os.path.isfile(path):
            raise SystemExit(f"[drift] FNO_DRIFT: no checkpoint '{base}' "
                             f"(looked for {path}); nothing was rolled out")
        out.append({"spec": item, "label": label, "path": path,
                    "nomean": nomean})
    if not out:
        raise SystemExit("[drift] FNO_DRIFT names no model")
    return out


def drift_pick(t_end):
    """(alpha, dt_max, where it came from): the psi0 = 0 pick in the meta.json
    of DRIFT_PICK_LABEL's rollout cache when it is there, else DRIFT_ALPHA and
    DRIFT_DT_MAX."""
    cache = DRIFT_PICK_CACHE or fno_paths.rollout_dir(
        fno_paths.run_label(DRIFT_PICK_LABEL, t_end, START_TIME), ROOT)
    meta = read_meta(cache)
    got = ((meta or {}).get("schedule") or {}).get(pick_key(DRIFT_OFF))
    if isinstance(got, dict) and "alpha" in got and "dt_max" in got:
        return (float(got["alpha"]), float(got["dt_max"]),
                f"the pick in {meta_file(cache)}")
    return (DRIFT_ALPHA, DRIFT_DT_MAX,
            f"DRIFT_ALPHA / DRIFT_DT_MAX (no pick in {meta_file(cache)})")


def drift_knobs():
    """(t_end, how many seeds, 0 for all): FNO_DRIFT_T_END and
    FNO_DRIFT_SEEDS, for tests and short checks only; a real job uses the
    defaults."""
    env = os.environ.get
    return (float(env("FNO_DRIFT_T_END") or DRIFT_T_END),
            int(env("FNO_DRIFT_SEEDS") or 0))


def _nulls(a):
    """A list (of lists) of floats with nan as None, for json."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        return None if np.isnan(a) else float(a)
    return [_nulls(x) for x in a]


@torch.no_grad()
def drift_rollout(model, psi0, times, device):
    """<psi_pred> of each seed at every time of the schedule: (len(times), b),
    nan from the step a seed diverges (|psi| > DIVERGE_PSI or not finite),
    which is also the divergence rule of the other schedule code."""
    b = psi0.shape[0]
    pred = np.full((len(times), b), np.nan)
    pred[0] = np.asarray(psi0, dtype=np.float64).mean(axis=(-2, -1))
    alive = torch.ones(b, dtype=torch.bool, device=device)

    def on_step(i, psi):
        nonlocal alive
        alive = alive & fno_train.alive_mask(psi, DIVERGE_PSI)
        psi[~alive] = 0.0
        mean = psi[:, 0].to(torch.float64).mean(dim=(-2, -1)).cpu().numpy()
        pred[i] = np.where(alive.cpu().numpy(), mean, np.nan)
        return bool(alive.any())

    roll_schedule(model, psi0, times, device, on_step)
    return pred


def drift_summary(pred, true):
    """The numbers of one entry from <psi_pred> (n, b) and <psi_true> (n, b)."""
    dev = pred - true
    seen = np.isfinite(dev)
    n_seen = seen.sum(axis=1)
    mean = np.where(n_seen > 0, np.where(seen, dev, 0.0).sum(axis=1)
                    / np.maximum(n_seen, 1), np.nan)
    peak = np.where(seen.any(axis=0),
                    np.abs(np.where(seen, dev, 0.0)).max(axis=0), np.nan)
    return {"deviation": dev, "mean_deviation": mean, "n_alive": n_seen,
            "max_abs_per_seed": peak,
            "max_abs_median": float(np.nanmedian(peak)),
            "max_abs_max": float(np.nanmax(peak)),
            "n_diverged": int((~seen[-1]).sum())}


def drift_text(rep):
    """The plain-text drift.txt of ``rep`` (the drift.json dict)."""
    m = rep["meta"]
    out = ["DRIFT of <psi_pred> - <psi_true>, psi0 = "
           f"{m['off']:+.1f}, l = {m['lattice']}, {m['n_seeds']} seeds "
           f"({m['seed_set']} set), "
           f"t = {m['t_start']:g} to {m['t_end']:g}",
           f"  schedule: alpha {m['alpha']:g}, dt_max {m['dt_max']:g}, "
           f"{len(rep['times']) - 1} steps ({m['schedule_source']})",
           f"  {m['host']}, {m['device']}, {m['date']}, job "
           f"{m['job_id'] or 'none'}", ""]
    for e in rep["entries"]:
        mean = e["mean_deviation"]
        last = next((x for x in reversed(mean) if x is not None), None)
        peak = max((abs(x) for x in mean if x is not None), default=None)
        w = e["train_mass_weight"]
        wtxt = "not recorded" if w is None else format(w, "g")
        if w is not None and e["trained_hard_mass_conservation"]:
            wtxt += " (unused: trained with hard mass conservation)"
        out.append(
            f"{e['spec']}: fp {e['fingerprint']}, lap "
            f"{e['laplacian_output']}, hard_mass {e['hard_mass_conservation']}"
            f", zero_dc {e['zero_dc']}, train mass weight {wtxt} | "
            f"seed-mean dev at end {'n/a' if last is None else format(last, '+.3e')}, "
            f"largest {'n/a' if peak is None else format(peak, '.3e')} | "
            f"max|dev| over seeds: median {e['max_abs_median']:.3e}, max "
            f"{e['max_abs_max']:.3e} | diverged {e['n_diverged']} of "
            f"{m['n_seeds']}")
    return "\n".join(out)


def run_drift(spec, args):
    """The drift mode (FNO_DRIFT): writes drift.json and drift.txt to the
    output folder and nothing else (the references the case lacks are solved
    into the reference cache, as in any run).  Returns 0."""
    import socket
    global END_TIME
    t_end, n_seeds = drift_knobs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (device == "cpu" and not args.allow_cpu
            and os.environ.get("FNO_ALLOW_CPU") != "1"):
        raise SystemExit("[drift] no CUDA device; pass --allow-cpu for a CPU "
                         "run")
    entries = drift_entries(spec)
    setup_precision()
    for e in entries:
        model, ckpt = load_checkpoint(e["path"], device=device)
        apply_lowk_fill(model)
        apply_mu_clip(model)
        e["trained_hard_mass"] = bool(model.hard_mass_conservation)
        if e["nomean"]:
            if not model.hard_mass_conservation:
                raise SystemExit(
                    f"[drift] {e['spec']}: the model has no final mean "
                    f"removal (hard_mass_conservation is False), so "
                    f"{DRIFT_NOMEAN} changes nothing; refused")
            model.hard_mass_conservation = False
            model.config["hard_mass_conservation"] = False
        e["model"] = model
        e["train_mass_weight"] = ((ckpt.get("train_config") or {})
                                  .get("constants") or {}).get("MASS_WEIGHT")
    alpha, dt_max, source = drift_pick(t_end)
    print(f"[drift] schedule: alpha {alpha:g}, dt_max {dt_max:g}, {source}",
          flush=True)
    must = [t for t in SCHEDULE_SNAPSHOTS if START_TIME < t < t_end] + [t_end]
    times = np.asarray(fno_gt.make_schedule(
        START_TIME, t_end, alpha, SCHED_DT_MIN, dt_max,
        quantum=fno_gt.GT_EVERY, must_hit=must), dtype=np.float64)
    # the test seeds of a dt-conditioned run: the group protocol's, as
    # apply_overrides chooses them (this mode skips it), unless --seed-set
    # fno asks for the other set
    seed_set = args.seed_set or "group"
    pool = GROUP_SEEDS if seed_set == "group" else EVAL_SEEDS
    seeds = [int(s) for s in (pool[:n_seeds] if n_seeds else pool)]
    lat, off = DRIFT_LATTICE, DRIFT_OFF
    print(f"[drift] {len(entries)} entries, psi0 = {off:+.1f}, l = {lat}, "
          f"{len(seeds)} seeds, t = {START_TIME:g}..{t_end:g} "
          f"({len(times) - 1} steps)", flush=True)
    pred = {e["spec"]: np.full((len(times), len(seeds)), np.nan)
            for e in entries}
    true = np.full((len(times), len(seeds)), np.nan)
    old_end, END_TIME = END_TIME, t_end
    try:
        check_reference_space({(lat, off): seeds})
        per = max(1, int(ROLLOUT_CELLS // (lat * lat)))
        for c in range(0, len(seeds), per):
            names = seeds[c:c + per]
            sl = slice(c, c + len(names))
            refs = load_references(lat, off, names, device)
            idx = [frame_index(t, times) for t, _ in refs]
            psi0 = np.stack([np.asarray(f[i[0]]) for (_, f), i
                             in zip(refs, idx)])
            for b, ((_, f), i) in enumerate(zip(refs, idx)):
                true[:, c + b] = np.asarray(f[i], dtype=np.float64).mean(
                    axis=(-2, -1))
            for e in entries:
                print(f"[drift] {e['spec']}: seeds {names[0]}..{names[-1]}",
                      flush=True)
                pred[e["spec"]][:, sl] = drift_rollout(
                    e["model"], psi0, times, device)
    finally:
        END_TIME = old_end
    rep_entries = []
    for e in entries:
        s = drift_summary(pred[e["spec"]], true)
        m = e["model"]
        rep_entries.append({
            "spec": e["spec"], "label": e["label"],
            "fingerprint": fno_paths.fingerprint(e["path"]),
            "laplacian_output": bool(m.laplacian_output),
            "hard_mass_conservation": bool(m.hard_mass_conservation),
            "zero_dc": bool(m.config.get("zero_dc", True)),
            "trained_hard_mass_conservation": e["trained_hard_mass"],
            "train_mass_weight": e["train_mass_weight"],
            "mean_pred": _nulls(pred[e["spec"]].T),
            "deviation": _nulls(s["deviation"].T),
            "mean_deviation": _nulls(s["mean_deviation"]),
            "n_alive": [int(x) for x in s["n_alive"]],
            "max_abs_per_seed": _nulls(s["max_abs_per_seed"]),
            "max_abs_median": s["max_abs_median"],
            "max_abs_max": s["max_abs_max"],
            "n_diverged": s["n_diverged"]})
    dev = torch.device(device)
    rep = {"meta": {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        "host": socket.gethostname(), "torch": torch.__version__,
        "precision": PRECISION, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "job_id": os.environ.get("SLURM_JOB_ID"), "off": off, "lattice": lat,
        "n_seeds": len(seeds), "seed_set": seed_set, "seeds": seeds,
        "t_start": float(START_TIME),
        "t_end": float(t_end), "alpha": alpha, "dt_max": dt_max,
        "schedule_source": source, "diverge_psi": DIVERGE_PSI},
        "times": [float(t) for t in times], "mean_true": _nulls(true.T),
        "entries": rep_entries}
    text = drift_text(rep)
    out = fresh_folder("FNO_DRIFT_OUT", "drift")
    os.makedirs(out)
    with open(os.path.join(out, "drift.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(out, "drift.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(text, flush=True)
    print(f"[drift] written to {out}", flush=True)
    return 0


# ---- ablation mode (FNO_ABLATE) ----------------------------------------
# What a model setting is worth: the grid of the real run (15 cases, the
# group protocol's 50 test seeds), on the schedules of the default model's
# real run, one R2 table per entry.  Nothing goes through the normal
# prediction output (no records, no cache, no results folder), so a run with
# other numerics can never move the default's results aside.  See README
# section 1.
ABLATE_PICK_LABEL = "train_t0500_lap"
ABLATE_PICK_END = 10000.0        # the horizon of that run
ABLATE_PICK_CACHE: str | None = None    # None: the label's rollout cache
ABLATE_T_END = 10000.0
ABLATE_TIMES = (2000.0, 5000.0, 10000.0)    # make_submission.R2_REPORT_TIMES
ABLATE_COMMON_BOX = 256          # the box whose critical quench is in the 6
ABLATE_NOFILL = "+nofill"
ABLATE_NOCLAMP = "+noclamp"


def ablate_entries(spec):
    """[{"spec", "label", "path", "nofill", "noclamp", "mu_channel",
    "lowk_fill"}] the FNO_ABLATE value asks for: ":"-separated labels or
    paths ending in .pt, each with optional +nofill and +noclamp.  Stops the
    run here, before anything is rolled out, for a missing checkpoint, a
    model that is not dt-conditioned, +noclamp on a model with no mu channel
    and +nofill on a model with no low-k fill."""
    out = []
    for item in [s.strip() for s in spec.split(":") if s.strip()]:
        base, nofill, noclamp = item, False, False
        while True:
            if base.endswith(ABLATE_NOFILL):
                base, nofill = base[:-len(ABLATE_NOFILL)], True
            elif base.endswith(ABLATE_NOCLAMP):
                base, noclamp = base[:-len(ABLATE_NOCLAMP)], True
            else:
                break
        if base.endswith(".pt"):
            path, label = base, fno_paths.model_label(base)
        else:
            path, label = fno_paths.model_path(base, ROOT), base
        if not os.path.isfile(path):
            raise SystemExit(f"[ablate] FNO_ABLATE: no checkpoint '{base}' "
                             f"(looked for {path}); nothing was rolled out")
        cfg = torch.load(path, map_location="cpu",
                         weights_only=False).get("config") or {}
        has_mu = bool(cfg.get("mu_channel", False))
        has_fill = bool(cfg.get("lowk_fill", False))
        if not cfg.get("dt_conditioned", False):
            raise SystemExit(f"[ablate] {item}: the model is not "
                             f"dt-conditioned; refused")
        if noclamp and not has_mu:
            raise SystemExit(f"[ablate] {item}: the model has no mu channel, "
                             f"so {ABLATE_NOCLAMP} changes nothing; refused")
        if nofill and not has_fill:
            raise SystemExit(f"[ablate] {item}: the model has no low-k weight "
                             f"fill, so {ABLATE_NOFILL} changes nothing; "
                             f"refused")
        out.append({"spec": item, "label": label, "path": path,
                    "nofill": nofill, "noclamp": noclamp})
    if not out:
        raise SystemExit("[ablate] FNO_ABLATE names no model")
    return out


def ablate_knobs():
    """(t_end, how many seeds (0: all), the case names to keep (empty: all)):
    FNO_ABLATE_T_END, FNO_ABLATE_SEEDS and FNO_ABLATE_CASES (":"-separated,
    like psi0+0.00_l256), for tests and short checks only; a real job uses
    the defaults."""
    env = os.environ.get
    names = [s.strip() for s in (env("FNO_ABLATE_CASES") or "").split(":")
             if s.strip()]
    return (float(env("FNO_ABLATE_T_END") or ABLATE_T_END),
            int(env("FNO_ABLATE_SEEDS") or 0), names)


def ablate_picks(offs):
    """({pick key: {"alpha", "dt_max", "dt_min"}}, the schedule's must-hit
    snapshot times, where they came from): the picks of the default model's
    real run, read from the meta.json of its rollout cache.  A real run gives
    every case the pick of its composition (schedule_case calls
    schedule_pick with the case's offset only), whatever the box size.
    Stops when the meta.json, a needed pick or the snapshot list is
    missing."""
    cache = ABLATE_PICK_CACHE or fno_paths.rollout_dir(
        fno_paths.run_label(ABLATE_PICK_LABEL, ABLATE_PICK_END, START_TIME),
        ROOT)
    meta = read_meta(cache)
    if meta is None:
        raise SystemExit(f"[ablate] no meta.json with the schedule picks of "
                         f"{ABLATE_PICK_LABEL} in {cache}; nothing was "
                         f"rolled out")
    sched = meta.get("schedule") or {}
    picks = {}
    for off in sorted(set(offs)):
        got = sched.get(pick_key(off))
        if not (isinstance(got, dict) and "alpha" in got
                and "dt_max" in got):
            raise SystemExit(f"[ablate] {meta_file(cache)} has no schedule "
                             f"pick for psi0 = {off:+.1f}; nothing was "
                             f"rolled out")
        picks[pick_key(off)] = {
            "alpha": float(got["alpha"]), "dt_max": float(got["dt_max"]),
            "dt_min": float(got.get("dt_min", SCHED_DT_MIN))}
    snaps = meta.get("snapshots")
    if not snaps:
        raise SystemExit(f"[ablate] {meta_file(cache)} lists no snapshot "
                         f"times; nothing was rolled out")
    return (picks, sorted(float(t) for t in snaps),
            f"the picks in {meta_file(cache)}")


def ablate_cases(names):
    """[(off, lattice)] of the real grid (grid_cases), lattice an int; only
    the named ones when ``names`` is not empty."""
    plan = [(off, LATTICE if lat is None else int(lat))
            for off, lat, _ in grid_cases()]
    if names:
        have = {_case_name(off, None if lat == LATTICE else lat): (off, lat)
                for off, lat in plan}
        bad = [n for n in names if n not in have]
        if bad:
            raise SystemExit(f"[ablate] FNO_ABLATE_CASES: no case {bad}; "
                             f"the grid has {sorted(have)}")
        plan = [have[n] for n in names]
    return plan


def ablate_is_common(off, lat):
    """The common 6 cases: the five compositions at l = 128 and the critical
    quench at l = 256 (make_submission.COMMON_CASES)."""
    return lat == LATTICE or (lat == ABLATE_COMMON_BOX
                              and abs(off - CRITICAL_OFF) < 1e-12)


@torch.no_grad()
def ablate_score(model, truth, times, device):
    """(R2 (len(times), b), diverged_at (b, nan: never)) of a rollout from
    truth[0], scored as _Track scores it in evaluate_schedule: r2_score of
    the float64 fields, floored at R2_FLOOR, 1 at the start; a seed whose
    field is not finite or has |psi| > DIVERGE_PSI is R2_FLOOR from that
    step on and is not stepped again."""
    n, b = truth.shape[:2]
    r2 = np.full((n, b), R2_FLOOR)
    r2[0] = 1.0
    div = np.full(b, np.nan)

    def on_step(i, psi):
        true_h = truth[i].astype(np.float64)
        pred_h = psi[:, 0].to(torch.float64).cpu().numpy()
        for k in range(b):
            if not np.isnan(div[k]):
                continue
            finite = bool(np.all(np.isfinite(pred_h[k])))
            pmax = float(np.abs(pred_h[k]).max()) if finite else float("inf")
            if not finite or pmax > DIVERGE_PSI:
                div[k] = float(times[i])
                psi[k].zero_()
                continue
            r2[i, k] = max(float(r2_score(true_h[k], pred_h[k])), R2_FLOOR)
        return bool(np.isnan(div).any())

    roll_schedule(model, truth[0], times, device, on_step)
    return r2, div


def ablate_case_numbers(times, r2, div):
    """One case of one entry from ``ablate_score``: make_submission's
    r2_at_times (the all-seeds mean, the survivors-only mean, n diverged, at
    ABLATE_TIMES), the seeds that diverged by the end, and every seed's R2
    at those times."""
    sub = _submission()
    rows = [{"times": times, "r2": r2[:, k],
             "diverged_at": None if np.isnan(div[k]) else float(div[k])}
            for k in range(r2.shape[1])]
    at = sub.r2_at_times(rows, ABLATE_TIMES)
    per_seed = {}
    for want in ABLATE_TIMES:
        i = sub.grid_index(times, want)
        if i is not None:
            per_seed[format(want, "g")] = [float(x) for x in r2[i]]
    return {"n_seeds": int(r2.shape[1]), "r2_at_times": at,
            "n_diverged": int((~np.isnan(div)).sum()),
            "diverged_at": [None if np.isnan(x) else float(x) for x in div],
            "per_seed_r2": per_seed}


def ablate_pool(cases):
    """The numbers of a group of cases (dicts from ablate_case_numbers) as
    r2_group_table pools them: {"cases", "seeds", "n_diverged", "at":
    {"2000": {"mean", "n", "mean_alive", "n_alive", "diverged"}}}, the means
    weighted by the seeds behind them."""
    at = {}
    for want in ABLATE_TIMES:
        k = format(want, "g")
        rows = [c["r2_at_times"][k] for c in cases if k in c["r2_at_times"]]
        n = sum(r["n"] for r in rows)
        if not n:
            continue
        alive = [r for r in rows if r.get("n_alive")
                 and r.get("mean_alive") is not None]
        na = sum(r["n_alive"] for r in alive)
        at[k] = {"mean": sum(r["mean"] * r["n"] for r in rows) / n, "n": n,
                 "mean_alive": (sum(r["mean_alive"] * r["n_alive"]
                                    for r in alive) / na if na else None),
                 "n_alive": na, "diverged": sum(r["diverged"] for r in rows)}
    return {"cases": len(cases), "seeds": sum(c["n_seeds"] for c in cases),
            "n_diverged": sum(c["n_diverged"] for c in cases), "at": at}


def ablate_groups(cases):
    """{"pooled", "common_6", "by_size": {"128": ...}} of one entry's
    {case name: numbers with "off" and "lattice"}.  "common_6" is None unless
    the run has all six of those cases."""
    items = list(cases.values())
    common = [c for c in items if ablate_is_common(c["off"], c["lattice"])]
    want = ({(off, LATTICE) for off in DATASET_OFFS}
            | {(CRITICAL_OFF, ABLATE_COMMON_BOX)})
    have = {(c["off"], c["lattice"]) for c in common}
    sizes = sorted({c["lattice"] for c in items})
    return {"pooled": ablate_pool(items),
            "common_6": ablate_pool(common) if want <= have else None,
            "by_size": {str(lat): ablate_pool(
                [c for c in items if c["lattice"] == lat]) for lat in sizes}}


def _ablate_cell(g, k):
    r = (g or {}).get("at", {}).get(k)
    if not r:
        return f"{'-':>22}"
    ma = "-" if r["mean_alive"] is None else format(r["mean_alive"], ".3f")
    return f"{format(r['mean'], '.3f') + ' (' + ma + ') ' + str(r['diverged']):>22}"


def ablate_text(rep):
    """The plain-text ablate.txt of ``rep`` (the ablate.json dict)."""
    m = rep["meta"]
    keys = [format(t, "g") for t in m["report_times"]]
    head = f"  {'':<18}{'cases':>5}" + "".join(f"{'t = ' + k:>22}" for k in keys)
    out = ["ABLATION: R2 of the real grid on the schedules of "
           f"{m['schedule_label']}",
           f"  {m['n_seeds']} seeds ({m['seed_set']} set) per case, "
           f"t = {m['t_start']:g} to {m['t_end']:g}; schedules from "
           f"{m['schedule_source']}",
           f"  {m['host']}, {m['device']}, {m['date']}, job "
           f"{m['job_id'] or 'none'}",
           "  cells: all-seeds mean (survivors-only mean) n diverged by then",
           ""]
    for e in rep["entries"]:
        state = "" if e["complete"] else "  [NOT COMPLETE: stopped early]"
        out += [f"{e['spec']}: fp {e['fingerprint']}, low-k fill "
                f"{'on' if e['lowk_fill'] else 'off'}, mu clamp "
                + ("off" if e["mu_clip"] is None else f"{e['mu_clip']:g}")
                + f", mu channel {'yes' if e['mu_channel'] else 'no'}, "
                f"{len(e['cases'])} of {m['n_cases']} cases{state}", head]
        g = e["groups"]
        rows = [("pooled", g["pooled"]), ("common 6 cases", g["common_6"])]
        rows += [(f"l = {lat}", v) for lat, v in g["by_size"].items()]
        for name, grp in rows:
            n = "-" if grp is None else grp["cases"]
            out.append(f"  {name:<18}{n:>5}"
                       + "".join(_ablate_cell(grp, k) for k in keys))
        out.append("  per case:")
        for name, c in e["cases"].items():
            out.append(f"    {name:<16}{c['n_seeds']:>3} seeds, "
                       f"{c['n_diverged']} diverged by the end, alpha "
                       f"{c['schedule']['alpha']:g}, dt_max "
                       f"{c['schedule']['dt_max']:g} |"
                       + "".join(_ablate_cell({"at": c["r2_at_times"]}, k)
                                 for k in keys))
        out.append("")
    return "\n".join(out)


def _ablate_write(out, rep):
    """ablate.json and ablate.txt, each replaced whole (a job stopped by its
    time limit leaves the last complete pair)."""
    rep["meta"]["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = os.path.join(out, "ablate.txt.tmp")
    with open(tmp, "w") as fh:
        fh.write(ablate_text(rep) + "\n")
    os.replace(tmp, os.path.join(out, "ablate.txt"))
    _write_json_atomic(os.path.join(out, "ablate.json"), rep)


def run_ablate(spec, args):
    """The ablation mode (FNO_ABLATE): ablate.json and ablate.txt in the
    output folder and nothing else (the references the grid lacks are solved
    into the reference cache, as in any run).  An entry is a model with its
    low-k fill and/or its mu clamp switched off; every other setting is a
    normal prediction's.  Returns 0."""
    import socket
    global END_TIME
    t_end, n_seeds, names = ablate_knobs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (device == "cpu" and not args.allow_cpu
            and os.environ.get("FNO_ALLOW_CPU") != "1"):
        raise SystemExit("[ablate] no CUDA device; pass --allow-cpu for a CPU "
                         "run")
    entries = ablate_entries(spec)
    plan = ablate_cases(names)
    picks, snaps, source = ablate_picks([off for off, _ in plan])
    seed_set = args.seed_set or "group"
    pool = GROUP_SEEDS if seed_set == "group" else EVAL_SEEDS
    seeds = [int(s) for s in (pool[:n_seeds] if n_seeds else pool)]
    must = sorted({t for t in (*snaps, *ABLATE_TIMES)
                   if START_TIME < t < t_end - 1e-9} | {float(t_end)})
    sched = {k: np.asarray(fno_gt.make_schedule(
        START_TIME, t_end, p["alpha"], p["dt_min"], p["dt_max"],
        quantum=fno_gt.GT_EVERY, must_hit=must), dtype=np.float64)
        for k, p in picks.items()}
    out = fresh_folder("FNO_ABLATE_OUT", "ablate")
    os.makedirs(out)
    setup_precision()
    print(f"[ablate] {len(entries)} entries, {len(plan)} cases, "
          f"{len(seeds)} seeds, t = {START_TIME:g}..{t_end:g}; schedules "
          f"from {source}", flush=True)
    dev = torch.device(device)
    rep = {"meta": {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        "host": socket.gethostname(), "torch": torch.__version__,
        "precision": PRECISION, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "job_id": os.environ.get("SLURM_JOB_ID"), "n_seeds": len(seeds),
        "seed_set": seed_set, "seeds": seeds, "n_cases": len(plan),
        "t_start": float(START_TIME), "t_end": float(t_end),
        "report_times": [float(t) for t in ABLATE_TIMES],
        "schedule_label": ABLATE_PICK_LABEL, "schedule_source": source,
        "schedules": picks, "diverge_psi": DIVERGE_PSI},
        "entries": []}
    old_end, END_TIME = END_TIME, t_end
    try:
        check_reference_space({(lat, off): seeds for off, lat in plan})
        for e in entries:
            model, _ = load_checkpoint(e["path"], device=device)
            apply_lowk_fill(model)
            apply_mu_clip(model)
            if e["nofill"]:
                fno_train.set_lowk_fill(model, False)
            if e["noclamp"]:
                fno_train.set_mu_clip(model, None)
            clip = fno_train.mu_clip_of(model)
            ent = {"spec": e["spec"], "label": e["label"],
                   "fingerprint": fno_paths.fingerprint(e["path"]),
                   "nofill": e["nofill"], "noclamp": e["noclamp"],
                   "lowk_fill": bool(fno_train.lowk_fill_of(model)),
                   "mu_clip": None if clip is None else float(clip),
                   "mu_channel": bool(getattr(model, "mu_channel", False)),
                   "complete": False, "cases": {}, "groups": None}
            rep["entries"].append(ent)
            print(f"[ablate] {e['spec']}: low-k fill "
                  f"{'on' if ent['lowk_fill'] else 'off'}, mu clamp "
                  f"{ent['mu_clip']}", flush=True)
            for off, lat in plan:
                times = sched[pick_key(off)]
                r2 = np.full((len(times), len(seeds)), R2_FLOOR)
                div = np.full(len(seeds), np.nan)
                per = max(1, int(ROLLOUT_CELLS // (lat * lat)))
                for c in range(0, len(seeds), per):
                    chunk = seeds[c:c + per]
                    print(f"[ablate] {e['spec']} psi0 = {off:+.1f} l = {lat}: "
                          f"seeds {chunk[0]}..{chunk[-1]}", flush=True)
                    refs = load_references(lat, off, chunk, device)
                    truth = np.stack(
                        [np.asarray(f[frame_index(t, times)])
                         for t, f in refs], axis=1)
                    sl = slice(c, c + len(chunk))
                    r2[:, sl], div[sl] = ablate_score(model, truth, times,
                                                      device)
                    del truth, refs
                num = ablate_case_numbers(times, r2, div)
                p = picks[pick_key(off)]
                num.update(off=float(off), lattice=int(lat),
                           schedule={"alpha": p["alpha"],
                                     "dt_max": p["dt_max"],
                                     "dt_min": p["dt_min"],
                                     "steps": len(times) - 1})
                ent["cases"][_case_name(off, None if lat == LATTICE else lat)
                             ] = num
                ent["groups"] = ablate_groups(ent["cases"])
                _ablate_write(out, rep)
            ent["complete"] = True
            _ablate_write(out, rep)
            del model
    finally:
        END_TIME = old_end
    print(ablate_text(rep), flush=True)
    print(f"[ablate] written to {out}", flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    timing = os.environ.get("FNO_TIMING", "").strip()
    if timing:
        return run_timing(timing, args)
    drift = os.environ.get("FNO_DRIFT", "").strip()
    if drift:
        return run_drift(drift, args)
    ablate = os.environ.get("FNO_ABLATE", "").strip()
    if ablate:
        return run_ablate(ablate, args)
    smoke = apply_smoke(args) if args.smoke else []
    if WRITE_LOG:
        fno_runlog.start_log("fno_predict")
    for line in smoke:
        print(f"[smoke] {line}", flush=True)
    apply_overrides(args)
    use_old_inputs()
    check_results_root()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fno_train.require_gpu(device, args.allow_cpu)
    if not os.path.exists(MODEL_PATH):
        pattern = os.path.join(fno_paths.models_dir(), "*",
                               fno_paths.MODEL_FILE)
        have = ", ".join(sorted(glob.glob(pattern))) or "none"
        raise FileNotFoundError(f"'{MODEL_PATH}' not found. Train one with "
                                f"fno_train.py, or pass --model PATH. "
                                f"Found in {pattern}: {have}.")
    setup_precision()
    model, ckpt = load_checkpoint(MODEL_PATH, device=device)
    apply_lowk_fill(model)
    apply_mu_clip(model)
    return evaluate_all(model, ckpt, device)


def names_other_checkpoint(run, now):
    """Whether a manifest's run record names a checkpoint other than ``now``
    (the dict checkpoint_now makes).

    By fingerprint when the record has one; a record from before
    fingerprints, by the size or time of the file it recorded.
    """
    if not isinstance(run, dict):
        return False
    if run.get("fingerprint"):
        return run["fingerprint"] != now["fingerprint"]
    return any(run.get(k) is not None and run[k] != now[k]
               for k in ("model_bytes", "model_mtime"))


def checkpoint_now():
    """What run_stamp records of MODEL_PATH, to tell another checkpoint by,
    and of the numerics it is run in, to tell another run of it by."""
    st = os.stat(MODEL_PATH)
    return {"fingerprint": model_fingerprint(), "model_bytes": st.st_size,
            "model_mtime": _mtime_text(st), "numerics": numerics_now()}


def _manifest_runs(folder):
    """The run records in the manifest of ``folder`` -- last_run, the
    checkpoint (and numerics) it names at the top, and each case's run --
    or [] when it has no readable manifest."""
    try:
        with open(os.path.join(folder, _submission().MANIFEST)) as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        return []
    if not isinstance(manifest, dict):
        return []
    runs = [manifest.get("last_run")]
    if manifest.get("checkpoint_fingerprint"):
        runs.append({"fingerprint": manifest["checkpoint_fingerprint"],
                     "numerics": manifest.get("checkpoint_numerics")})
    for kind in _submission().KINDS:
        entries = manifest.get(kind)
        if isinstance(entries, dict):
            runs += [e.get("run") for e in entries.values()
                     if isinstance(e, dict)]
    return runs


def other_checkpoint_of(folder, now):
    """The fingerprint of the other checkpoint the manifest in ``folder``
    names ('unknown' when its record has none), or None if it names only
    this one, or has no readable manifest."""
    for run in _manifest_runs(folder):
        if names_other_checkpoint(run, now):
            return (run or {}).get("fingerprint") or "unknown"
    return None


def other_numerics_of(folder, now):
    """The numerics stamp of the first run record in the manifest of
    ``folder`` that names this checkpoint but was made in other numerics, or
    None if there is none.  A record with no stamp is OLD_NUMERICS; one that
    names no checkpoint at all says nothing."""
    for run in _manifest_runs(folder):
        if (isinstance(run, dict) and not names_other_checkpoint(run, now)
                and any(run.get(k) is not None for k in (
                    "fingerprint", "model_bytes", "model_mtime"))):
            was = full_numerics(run.get("numerics"))
            if was != now["numerics"]:
                return was
    return None


def stale_reason(folder, now):
    """Why the results in ``folder`` are not this run's, as (name for the
    archive folder, what they are, what this run uses); None if they are.
    Another checkpoint's, or this one's in other numerics."""
    fp = other_checkpoint_of(folder, now)
    if fp is not None:
        return fp, f"results of checkpoint {fp}", now["fingerprint"]
    was = other_numerics_of(folder, now)
    if was is not None:
        return (f"{now['fingerprint']}-{numerics_tag(was)}",
                f"results of this checkpoint made in other numerics "
                f"({numerics_text(was)})", numerics_text(now["numerics"]))
    return None


def cache_is_old(cache):
    """Whether the rollouts in ``cache`` were made by another model: there
    are some, and none has a cache key this run's cases would use."""
    try:
        names = [n for n in os.listdir(cache) if n.endswith(".npz")]
    except OSError:
        return False
    cases = [(off, LATTICE if lat is None else lat)
             for off, lat, _ in grid_cases()]
    keys = {_cache_key(off, lat) for off, lat in cases}
    return bool(names) and not any(n.endswith(f"_{k}.npz")
                                   for n in names for k in keys)


def _move(src, dst):
    """Move a folder without deleting anything: os.replace, and shutil.move
    only where that is across devices.  A move that fails leaves no empty
    folders of the archive behind."""
    parent = os.path.dirname(dst)
    os.makedirs(parent, exist_ok=True)
    try:
        try:
            os.replace(src, dst)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.move(src, dst)
    except OSError:
        with contextlib.suppress(OSError):
            os.removedirs(parent)
        raise


def _unused(path):
    n, fresh = 1, path
    while os.path.exists(fresh):
        n += 1
        fresh = f"{path}-{n}"
    return fresh


def supersede_old_runs():
    """Before the run writes anything, put aside results of another model.

    Every results folder this run will write (each view's) is looked at.  One
    whose manifest names another checkpoint -- a different fingerprint, or
    for a manifest from before fingerprints a different size or time -- is
    moved to Archive/superseded/<label>_<old fingerprint>_<time>/Results/,
    and the rollout cache with it when the old model made it (extras/, FNO's
    own outputs of the run, always goes: to rollout_cache/<cache>/extras/ of
    the same archive folder), so the run starts fresh and the two models'
    results are never mixed in one folder.
    The same holds for a manifest that names this checkpoint but was made in
    other numerics (numerics_now: precision, TF32, low-k fill; a manifest
    with no stamp is OLD_NUMERICS); its folder is named
    <label>_<fingerprint>-<old numerics>_<time>.
    A folder that names only this checkpoint, in these numerics, is a top-up
    or a resumed job and stays.  NO_SUPERSEDE stops the run instead of
    moving.  Nothing is deleted; if a move fails the run stops here, with
    nothing written.  --smoke has its own Archive under its own root.
    """
    if not os.path.isfile(MODEL_PATH):
        return
    folders = [view_dir(h) for h in VIEWS] if VIEWS else [RESULTS_DIR]
    now = checkpoint_now()
    stale = []
    for folder in dict.fromkeys(folders):
        why = stale_reason(folder, now)
        if why is not None:
            stale.append((folder, *why))
    if not stale:
        return
    archive = os.path.join(fno_paths.archive_dir(ROOT), "superseded")
    if NO_SUPERSEDE:
        raise SystemExit(
            "--no-supersede: " + "; ".join(
                f"{folder} holds {held}, and this run uses {uses}"
                for folder, _, held, uses in stale)
            + f".  Nothing was moved or written.  Leave out --no-supersede "
            f"to move them to {archive}, or give --results-dir another "
            f"folder.")

    cache = RESULT_CACHE
    in_cache = (os.path.isdir(cache)
                and fno_paths.under(cache, fno_paths.rollout_root(ROOT))
                and os.path.abspath(cache)
                != os.path.abspath(fno_paths.rollout_root(ROOT)))
    take_cache = in_cache and cache_is_old(cache)
    if os.path.isdir(cache) and not take_cache:
        print(f"[supersede] the rollouts in {cache} stay: "
              + ("they are not under the standard rollout_cache folder"
                 if not in_cache else "this model made them"), flush=True)

    when = time.strftime("%Y%m%d-%H%M%S")
    bar = "!" * 72
    print(f"\n{bar}\n  SUPERSEDED: this run uses checkpoint "
          f"{now['fingerprint']} ({numerics_text(now['numerics'])}), and "
          f"{len(stale)} results folder(s) hold another run's.  Moving them "
          f"to {archive}/ (nothing is deleted)\n{bar}", flush=True)
    for folder, _, held, _ in stale:
        print(f"  {folder}: {held}", flush=True)
    names = [os.path.basename(os.path.normpath(f)) for f, *_ in stale]
    multi = bool(VIEWS) and len(VIEWS) > 1
    xbase = (EXTRAS_ROOT if EXTRAS_ROOT is not None
             else os.path.join(cache, EXTRAS_NAME))
    xdirs = [extras_for(f, xbase, multi) for f, *_ in stale]
    home = (names.index(os.path.basename(os.path.normpath(cache)))
            if os.path.basename(os.path.normpath(cache)) in names else 0)
    moved = []
    for i, ((folder, old, _, _), label) in enumerate(zip(stale, names)):
        dest = _unused(os.path.join(archive, f"{label}_{old}_{when}"))
        steps = [(folder, os.path.join(dest, "Results", label))]
        if take_cache and i == home:
            # the cache carries its extras/ (every view's) with it
            steps.append((cache, os.path.join(dest, "rollout_cache",
                                              os.path.basename(cache))))
        elif not take_cache and os.path.isdir(xdirs[i]):
            steps.append((xdirs[i], os.path.join(
                dest, "rollout_cache", os.path.basename(cache), EXTRAS_NAME,
                *([os.path.basename(xdirs[i])] if multi else []))))
        for src, dst in steps:
            try:
                _move(src, dst)
            except OSError as exc:
                raise SystemExit(
                    f"could not move {src} to {dst} ({type(exc).__name__}: "
                    f"{exc}).  Something probably has a file in it open (a "
                    f"viewer, an editor, a sync client): close it and run "
                    f"again.  The run stopped before writing anything."
                    + (f"  Already moved: {', '.join(moved)}." if moved
                       else "")) from exc
            moved.append(f"{src} -> {dst}")
            print(f"[supersede] moved {src} -> {dst}", flush=True)


def _finite(x):
    """A float, or None for what is not a finite number (JSON has no nan)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def tile_invariance(model, device) -> dict:
    """One step on a field and on its 2 x 2 tiling, compared.

    The step on the tiled field should be the tiling of the step on the
    field: nothing in the model knows where a pixel is, and with physical
    mode scaling its spectral weights at twice the lattice sit at the same
    wavenumbers as at the first.  More than rounding between them means the
    model does not carry from a small box to a larger one.  Returns the
    largest difference and the L2 difference over the L2 norm of the tiled
    step.

    It tests the model, not the GPU's float32 kernels: on the cluster GPU in
    float32 the difference was 2e-4 (over the limit), and the same checkpoint
    on the CPU gave 1.7e-6 in float32 and 3e-15 in float64.  So the steps are
    taken by a float64 copy of the model on the CPU.  The model itself is
    left as it is, and the field is solved on ``device`` as before.
    """
    l = int(DIAG_TILE)
    cond = bool(getattr(model, "dt_conditioned", False))
    if cond:
        # the field is solved here, not cached: no file in any reference
        # cache for a 64 x 64 box; the step asked for is DIAG_TILE_DT long
        seed = int(fno_gt.VAL_SEEDS[0])
        times, frames = fno_gt.integrate(fno_gt.make_ic(l, seed, 0.0)[None],
                                         DIAG_TILE_T, [DIAG_TILE_T],
                                         device=device)
        assert frames is not None
        frames = frames[:, 0]
    else:
        seed = int(FIXED_STEP["VAL_SEEDS"][0])
        times, frames = ch_ground_truth(l, seed, 0.0, DIAG_TILE_T, DT_STEP,
                                        cache_dir=GT_CACHE, device=device)
    i = int(np.argmin(np.abs(times - DIAG_TILE_T)))
    twin = copy.deepcopy(model).to("cpu", torch.float64).eval()
    psi = torch.as_tensor(frames[i], dtype=torch.float64,
                          device="cpu")[None, None]
    dt = DIAG_TILE_DT if cond else None
    with torch.no_grad():
        small = fno_train.step(twin, psi, dt)
        big = fno_train.step(twin, psi.repeat(1, 1, 2, 2), dt)
    want = small.repeat(1, 1, 2, 2)
    diff = big - want
    max_abs = float(diff.abs().max())
    return {"l": l, "tiled_l": 2 * l, "seed": seed, "t": float(times[i]),
            "dt": dt, "device": "cpu", "dtype": "float64",
            "max_abs": _finite(max_abs),
            "rel_l2": _finite(float(diff.norm() / want.norm())),
            "tolerance": DIAG_TILE_TOL,
            "invariant": bool(max_abs <= DIAG_TILE_TOL)}


def lowk_check(model, device) -> dict:
    """fno_train's low-k increment gain, one step from the true state, on the
    validation seeds' references.

    A fixed-step model takes its own step on the old 1.0-spaced references of
    FIXED_STEP's validation seeds up to FIXED_STEP's VAL_T_END.  A
    dt-conditioned one takes fno_train.LOWK_DT on the 4.0-spaced references
    (the tune seeds', already there after the prewarm) up to the run's
    END_TIME.  The starts are fno_train.lowk_times of the reference up to
    END_TIME, as in training: LOWK_T0 and the last frame that has a next one,
    where the reference holds them.
    """
    lat = int(fno_train.VAL_LATTICE if DIAG_LOWK_L is None else DIAG_LOWK_L)
    cond = bool(getattr(model, "dt_conditioned", False))
    if cond:
        t_end = float(END_TIME if DIAG_LOWK_T_END is None
                      else DIAG_LOWK_T_END)
        dt = float(fno_train.LOWK_DT)
        if t_end - START_TIME < dt + 1e-9:
            return {"skipped": f"the references end at {t_end:g}, less than "
                               f"{dt:g} after the start"}
        seeds = [int(s) for s in fno_gt.VAL_SEEDS[:DIAG_LOWK_SEEDS]]
        refs = []
        fno_gt.ground_truth_batch(ref_cache(), lat, 0.0, seeds, START_TIME,
                                  t_end, fno_gt.GT_EVERY, fno_gt.SOLVER_DT,
                                  device)
        for s in seeds:
            path = fno_gt.find_gt(ref_cache(), lat, 0.0, s, START_TIME, t_end,
                                  fno_gt.GT_EVERY)
            assert path is not None
            refs.append(fno_gt.load_gt(path, t_end))
        gts = refs
    else:
        t_end = float(FIXED_STEP["VAL_T_END"]
                      if DIAG_LOWK_T_END is None else DIAG_LOWK_T_END)
        dt = float(DT_STEP)
        seeds = [int(s) for s in
                 FIXED_STEP["VAL_SEEDS"][:DIAG_LOWK_SEEDS]]
        gts = ch_ground_truth_batch(lat, seeds, 0.0, t_end, DT_STEP,
                                    cache_dir=GT_CACHE, device=device)
    times = gts[0][0]
    t0s = fno_train.lowk_times(times[times <= END_TIME + 1e-9], dt)
    if not t0s:
        return {"skipped": "the references hold none of the start times"}
    n_ref = int((getattr(model, "config", None) or {}).get("n_ref", 64))
    gains = fno_train.lowk_gains(model, gts, t0s, lat / n_ref, str(device),
                                 dt=dt)
    return {"lattice": lat, "n_ref": n_ref, "seeds": seeds, "dt": dt,
            "shells": list(fno_train.LOWK_SHELLS),
            "gain": {f"{t0:g}": {k: _finite(v) for k, v in g.items()}
                     for t0, g in gains.items()},
            "text": fno_train.lowk_text(gains)}


def _diag_part(name, fn, *args) -> dict:
    """One check; a failure is reported and recorded, and the run goes on."""
    try:
        return fn(*args)
    except Exception as exc:
        print(f"[diagnostics] {name} failed ({type(exc).__name__}: {exc}); "
              f"the run goes on without it", flush=True)
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


def diagnostics_text(rec):
    """The few lines a diagnostics record is printed as."""
    def num(x):
        return "nan" if x is None else f"{x:.2e}"

    tile, lowk = rec["tile"], rec["lowk"]
    if "error" in tile:
        lines = ["[diagnostics] tile invariance: no result (it failed)"]
    else:
        lines = [f"[diagnostics] tile invariance, l={tile['l']} tiled to "
                 f"{tile['tiled_l']} at t={tile['t']:g}"
                 + (f", dt={tile['dt']:g}" if tile.get("dt") else "")
                 + (f", {tile['dtype']} on {tile.get('device')}"
                    if tile.get("dtype") else "")
                 + f": largest difference "
                 f"{num(tile['max_abs'])}, relative L2 {num(tile['rel_l2'])}, "
                 + ("within " if tile["invariant"] else "NOT INVARIANT, over ")
                 + f"{tile['tolerance']:g}"]
    if "error" in lowk:
        lines.append("[diagnostics] low-k gain: no result (it failed)")
    elif "skipped" in lowk:
        lines.append(f"[diagnostics] low-k gain skipped: {lowk['skipped']}")
    else:
        lines.append(f"[diagnostics] low-k gain at l={lowk['lattice']}, "
                     f"dt={lowk['dt']:g}, "
                     f"{len(lowk['seeds'])} seed(s), below/first/mid/high@"
                     f"start: {lowk['text']}")
    lines.append(f"[diagnostics] took {rec['seconds']:.1f} s")
    return "\n".join(lines)


def diagnostics(model, device) -> dict:
    """Two checks of the loaded model: tile invariance and the low-k gain.

    Returns the record that goes in diagnostics.json and prints a summary.
    Nothing here changes the model or a rollout; a check that fails is
    recorded as an error and the other still runs.
    """
    t0 = time.perf_counter()
    rec = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "model_path": MODEL_PATH, "fingerprint": model_fingerprint()}
    rec["tile"] = _diag_part("the tile check", tile_invariance, model, device)
    rec["lowk"] = _diag_part("the low-k gain", lowk_check, model, device)
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    print(diagnostics_text(rec), flush=True)
    return rec


def write_diagnostics(rec):
    """diagnostics.json in every results folder this run writes: each view's,
    or RESULTS_DIR."""
    base = (EXTRAS_ROOT if EXTRAS_ROOT is not None
            else os.path.join(RESULT_CACHE, EXTRAS_NAME))
    multi = bool(VIEWS) and len(VIEWS) > 1
    folders = ([extras_for(view_dir(h), base, multi) for h in VIEWS]
               if VIEWS else [extras_for(RESULTS_DIR, base, False)])
    for folder in dict.fromkeys(folders):
        path = os.path.join(folder, DIAG_FILE)
        try:
            os.makedirs(folder, exist_ok=True)
            _write_json_atomic(path, rec)
        except OSError as exc:
            print(f"[diagnostics] WARNING: could not write {path} "
                  f"({type(exc).__name__}: {exc})", flush=True)
        else:
            print(f"[saved] {path}", flush=True)


def evaluate_all(model, ckpt, device, case_fn=None, resume=None, clean=None):
    """A prediction run once the model is loaded: every case, then the
    run-wide outputs.  Returns the exit status (1 if anything failed).

    Before anything is written, a results folder of another checkpoint is
    put aside (supersede_old_runs).  A regeneration from cached rollouts
    (``case_fn``) leaves folders where they are, and makes no diagnostics.
    A run makes them once, here, before its cases (DIAGNOSTICS; a model that
    is not a torch module has nothing to check).

    ``case_fn`` produces one case, run_case by default; make_submission hands
    in one that reads cached rollouts instead, with no model, to regenerate a
    folder.  ``resume`` takes a case an earlier job finished from its record
    instead of rolling it out again (default: unless REDO); ``clean`` deletes
    a case's cached rollouts once its outputs are written (default: unless
    KEEP_CACHE).  A regeneration turns both off.

    A run (no ``case_fn``) ends by rebuilding the index of the results root,
    Results/README.md and comparison.png (refresh_index).
    """
    global RESULTS_DIR, RESULTS_TEMPLATE, EXTRAS_ROOT, _RUN_HORIZON, _RUN_TRAIN
    fno_paths.forget_fingerprints()
    resume = (not REDO) if resume is None else resume
    clean = (CACHE_RESULTS and not KEEP_CACHE) if clean is None else clean
    ckpt = ckpt if isinstance(ckpt, dict) else {}
    _RUN_HORIZON = _ckpt_horizon(ckpt)
    cfg = getattr(model, "config", None) or {}
    _RUN_TRAIN = _ckpt_train(ckpt) if ckpt.get("config") else None
    _RUN_MODEL.clear()
    if model is not None:
        dtype = next(model.parameters()).dtype
        _RUN_MODEL.update(
            config=json.loads(json.dumps(cfg, default=_plain)),
            n_parameters=model.n_parameters(), dtype=str(dtype),
            device=None if device is None else str(device))
        print(f"[model] {MODEL_PATH}  ({model.n_parameters():,} parameters, "
              f"backbone={cfg.get('backbone', 'physical_k')}, dtype={dtype})")
        print(f"[model] hard_mass_conservation="
              f"{cfg.get('hard_mass_conservation', True)}, residual_prediction="
              f"{cfg.get('residual_prediction', True)}"
              + (f", mode_scaling={cfg['mode_scaling']}, "
                 f"zero_dc={cfg.get('zero_dc')}"
                 if cfg.get("backbone", "physical_k") == "physical_k" else ""))
    if "best" in ckpt:
        print(f"[model] selected at epoch {ckpt['best'].get('epoch')} "
              f"(rollout score {ckpt['best'].get('score', float('nan')):.4f}, "
              f"survives to t={ckpt['best'].get('survival', float('nan')):.0f})")

    hard_mc = bool(cfg.get("hard_mass_conservation", True))
    zero_dc = bool(cfg.get("zero_dc", True))
    hard = hard_mc and zero_dc
    MODEL_FLAGS.update(
        hard_constraints=hard,
        hard_mass_conservation=hard_mc,
        zero_dc=zero_dc,
        label=LABEL if (hard or not AUTO_LABEL) else f"{LABEL}_nohard",
        note="" if hard else "  [HARD CONSTRAINTS OFF]")
    print(f"[model] hard constraints {'ON' if hard else 'OFF'} "
          f"(hard_mass_conservation={hard_mc}, zero_dc={zero_dc}); "
          f"outputs labelled {MODEL_FLAGS['label']}")
    folders = (RESULTS_DIR, RESULTS_TEMPLATE, EXTRAS_ROOT)
    if not hard and AUTO_LABEL:
        # the submission figures carry no label, so an ablation gets a
        # folder of its own rather than overwriting the real model's; its
        # extras (case summaries, the bookkeeping manifest, sweep curves)
        # are kept apart from the real model's in the same cache the same way
        RESULTS_DIR = fno_paths.nohard_name(RESULTS_DIR)
        if RESULTS_TEMPLATE:
            RESULTS_TEMPLATE = fno_paths.nohard_name(RESULTS_TEMPLATE)
        if EXTRAS_ROOT is None:
            EXTRAS_ROOT = os.path.join(RESULT_CACHE, EXTRAS_NAME + "_nohard")
        print(f"[model] so this run writes to "
              f"{RESULTS_TEMPLATE or RESULTS_DIR}", flush=True)
    # the numerics this run is stamped with: the model's, or for a redraw the
    # ones its records were made in
    _RUN_NUMERICS.clear()
    _RUN_NUMERICS.update(numerics_of(model) if RECORDED_NUMERICS is None
                         else full_numerics(RECORDED_NUMERICS))
    print(f"[model] numerics: {numerics_text(_RUN_NUMERICS)}", flush=True)
    try:
        if WRITE_SUBMISSION:
            # before anything is written: a folder in the old flat layout
            # is not written into
            _submission().refuse_flat(
                [view_dir(h) for h in VIEWS] if VIEWS else [RESULTS_DIR])
        if case_fn is None:
            supersede_old_runs()
            if DIAGNOSTICS and isinstance(model, torch.nn.Module):
                write_diagnostics(diagnostics(model, device))
            if RECORDS:
                start_records(model, ckpt)
            if ENGINE == "schedule":
                prewarm_references(device, skip_done=resume)
        status = _evaluate_cases(model, ckpt, device, case_fn, resume, clean)
        if case_fn is None:
            refresh_index()
        return status
    finally:
        RESULTS_DIR, RESULTS_TEMPLATE, EXTRAS_ROOT = folders
        _RUN_HORIZON = _RUN_TRAIN = None
        _RUN_NUMERICS.clear()
        _META.clear()
        _PICKS.clear()
        _REF_SECONDS.clear()


def run_facts():
    """What the top of a run folder (manifest.json, report.txt) says of the
    run beyond its cases: model, checkpoint, windows, seeds, flags."""
    m = _RUN_MODEL
    return {
        "label": os.path.basename(os.path.normpath(RESULTS_DIR)),
        "source": RUN_SOURCE, "cache": RUN_CACHE or RESULT_CACHE,
        "checkpoint": MODEL_PATH or RECORDED_CHECKPOINT,
        "fingerprint": model_fingerprint(), "numerics": numerics_now(),
        "n_parameters": m.get("n_parameters"), "dtype": m.get("dtype"),
        "device": m.get("device") or RECORDED_DEVICE,
        "arch": m.get("config") or {},
        "best": {k: v for k, v in _RUN_BEST.items()
                 if not isinstance(v, (list, dict))},
        "train_horizon": train_horizon(), "train_setup": train_setup(),
        "lattice": LATTICE, "size_sweep": [int(x) for x in SIZE_SWEEP],
        "eval_seeds": [int(x) for x in EVAL_SEEDS], "seed_set": SEED_SET,
        "start": START_TIME, "end": END_TIME,
        "snapshots": sorted(float(t) for t in SNAPSHOT_TIMES
                            if t <= END_TIME),
        "threshold": R2_THRESHOLD, "engine": ENGINE,
        "schedules": {k: {f: v[f] for f in ("alpha", "dt_min", "dt_max",
                                            "source")}
                      for k, v in _PICKS.items()},
        "hard_mass_conservation": MODEL_FLAGS["hard_mass_conservation"],
        "zero_dc": MODEL_FLAGS["zero_dc"]}


def refresh_index():
    """Rebuild Results/README.md, comparison.png and its per-case and
    per-size forms at the end of a prediction run, and draw the training
    curves of the runs that lack them (make_submission.write_results_index;
    the models' histories are looked for under ROOT, the smoke root under
    --smoke).

    Only when the run's folders are in the results root, so a run sent
    elsewhere (--results-dir) does not write beside folders that are not its
    own; --index does the root named.  Under --smoke the root is the smoke
    root's Results.  A failure is reported and does not fail the run.
    """
    root = fno_paths.results_root(ROOT)
    here = RESULTS_TEMPLATE or RESULTS_DIR
    if (not fno_paths.under(here, root)
            or os.path.abspath(here) == os.path.abspath(root)):
        print(f"[index] {here} is not a folder of {root}, so {root} is left "
              f"as it is; python make_submission.py --index rebuilds it",
              flush=True)
        return
    try:
        _submission().write_results_index(root, ROOT)
    except (Exception, SystemExit) as exc:
        print(f"[index] WARNING: the index of {root} was not rebuilt "
              f"({type(exc).__name__}: {exc})", flush=True)
        traceback.print_exc()


def _evaluate_cases(model, ckpt, device, case_fn, resume, clean):
    for root in ([view_dir(h) for h in VIEWS] if VIEWS else [RESULTS_DIR]):
        old = old_layout_files(root)
        if old:
            print(f"[results] WARNING: {root} still holds {len(old)} file(s) "
                  f"of the old per-seed layout (e.g. {old[0]}); they are not "
                  f"this run's and stay as they are -- delete them by hand, "
                  f"or write to a new folder", flush=True)
    if not VIEWS:
        os.makedirs(RESULTS_DIR, exist_ok=True)
    run_stamp(ckpt)
    if VIEWS:
        # each view's folder gets a stamp with that view's window, made on
        # first use inside view_context; the checkpoint details are kept
        _RUN_STAMPS.clear()

    failed, failed_keys = [], []
    _SUBMITTED.clear()
    deferred = []

    def attempt(name, fn, *args, **kw):
        """Never let one case take the whole deliverable down with it."""
        try:
            return fn(*args, **kw)
        except Exception:
            failed.append(name)
            print(f"\n[error] {name} FAILED -- continuing with the rest:",
                  flush=True)
            traceback.print_exc()
            return None

    def finish(res, off, lattice, seeds):
        """A case whose outputs are all written no longer needs its cache."""
        if not clean:
            return
        ok = case_ok(res)
        if res is not None and not ok:
            print(f"[cache] {case_tag(off, lattice)}: kept, as its submission "
                  f"figures failed; a rerun reuses it", flush=True)
        lat = LATTICE if lattice is None else lattice
        if (abs(off - CRITICAL_OFF) < 1e-12 and lat == LATTICE
                and tuple(seeds_for(CRITICAL_OFF)) != tuple(SWEEP_SEEDS)):
            # the composition case and its sweep-seed twin share cache
            # files: they go at the end, and only those no failed twin needs
            deferred.append((off, lattice, seeds, ok))
            return
        if ok:
            drop_case_cache(off, lattice, seeds)

    def one(name, off, lattice=None, seeds=None, kept=False):
        """A case: an earlier job's finished one, one kept from an earlier
        job's records, or rolled out now."""
        sd = seeds_for(off) if seeds is None else tuple(seeds)
        lat = LATTICE if lattice is None else lattice
        res = None
        if kept and case_fn is None:
            mark = len(failed)
            res = attempt(name, case_from_records, off, lattice, sd)
            if res is None and len(failed) > mark:
                failed_keys.extend(submission_keys(off, lattice))
            note_case(off, lattice, sd, res)
            return res
        if resume and os.path.exists(MODEL_PATH):
            res = finished_case(off, lattice, sd, _cache_key(off, lat))
            if res is not None:
                print(f"\n[case] {case_tag(off, lattice)}: already finished "
                      f"in {case_dir(off, lattice)} by an earlier job with "
                      f"this model, window and seeds; not rolled out again "
                      f"(--redo does)", flush=True)
        if res is None:
            kw = ({} if lattice is None and seeds is None
                  else {"lattice": lattice, "seeds": seeds})
            mark = len(failed)
            res = attempt(name, run_case if case_fn is None else case_fn,
                          model, device, off, **kw)
            if res is None and len(failed) > mark:
                # its kept entries are marked; a case the case function had
                # nothing for (not in a rebuild's cache) did not fail
                failed_keys.extend(submission_keys(off, lattice))
        note_case(off, lattice, sd, res)
        finish(res, off, lattice, sd)
        return res

    # by_off: the compositions at the base lattice; by_size: every other
    # case, by (composition, lattice)
    by_off: dict = {}
    by_size: dict = {}
    grid = {_case_name(off, lattice) for off, lattice, _ in grid_cases()}
    for off, lattice, seeds in planned_cases():
        kept = _case_name(off, lattice) not in grid
        if lattice is None:
            res = one(f"composition {off:+.2f}", off, kept=kept)
            if res is not None:
                by_off[off] = res
        else:
            name = (f"lattice {lattice}" if abs(off - CRITICAL_OFF) < 1e-12
                    else f"psi0 {off:+.2f}, lattice {lattice}")
            res = one(name, off, lattice, seeds, kept=kept)
            if res is not None:
                by_size[(off, lattice)] = res

    fig_off = dict(by_off)
    if (SWEEP_CONSISTENT_SEEDS and CRITICAL_OFF in fig_off
            and tuple(seeds_for(CRITICAL_OFF)) != tuple(SWEEP_SEEDS)):
        same = by_size.get((CRITICAL_OFF, LATTICE))
        if same is None:
            same = one("critical on the sweep seeds", CRITICAL_OFF, LATTICE,
                       SWEEP_SEEDS)
        if same is not None:
            fig_off[CRITICAL_OFF] = same
            print(f"\n[sweep] psi_0 = {CRITICAL_OFF:+.1f} redrawn from the "
                  f"{len(SWEEP_SEEDS)} sweep seeds so every curve in "
                  f"{MODEL_FLAGS['label']}_sweeps.png matches; the "
                  f"{len(EVAL_SEEDS)}-seed result stays in "
                  f"{MODEL_FLAGS['label']}_summary.json", flush=True)
    needed = [p for off, lattice, seeds, ok in deferred if not ok
              for p in case_cache_files(off, lattice, seeds)]
    for off, lattice, seeds, ok in deferred:
        if ok:
            drop_case_cache(off, lattice, seeds, keep=needed)

    run_failed = []

    def outputs(by_off, by_size, fig_off, view=None):
        """Everything written from all the cases at once, for one view."""
        label = MODEL_FLAGS["label"]
        mark = len(failed)
        where = f" ({view:g})" if view is not None else ""

        cases = [by_off[o] for o in sorted(by_off)]
        cases += [by_size[k] for k in sorted(by_size, key=lambda k: k[::-1])
                  if not any(by_size[k] is c for c in cases)]

        everything = cases + list(fig_off.values())
        for c in {id(c): c for c in everything}.values():
            if c.get("submission_error"):
                failed.append(f"submission for psi0={c['off']:+.2f} "
                              f"l={c['lattice']}{where}")
        if WRITE_SUBMISSION:
            attempt(f"submission manifest and sweep{where}", write_submission,
                    everything, run_failed + failed[mark:], failed_keys)

        if len(by_off) > 1:
            bar = "=" * 72
            print("\n" + bar)
            print(f"  composition sweep{where}")
            print(f"  {'psi_0':>7} {'seeds':>6} {'mean R2':>9} "
                  f"{'R2@' + format(END_TIME, '.0f'):>9} "
                  f"{'held':>7} {'steps/seed':>11}")
            for o in sorted(by_off):
                st = by_off[o]["stats"]
                print(f"  {o:>+7.1f} {st['n_seeds']:>6} "
                      f"{st['mean_r2_all_seeds']:>9.4f} "
                      f"{by_off[o]['mean_curve'][-1]:>9.4f} "
                      f"{st['seeds_holding_throughout']:>3}/{st['n_seeds']:<3} "
                      f"{st['steps_above_threshold_mean']:>11.0f}")
            print(bar, flush=True)

        if WRITE_TIMING:
            attempt(f"timing report{where}", write_timing_report, cases,
                    device, model,
                    os.path.join(extras_dir(),
                                 TIMING_JSON.format(label=label)),
                    os.path.join(extras_dir(),
                                 TIMING_TEXT.format(label=label)))

        print("\n" + "=" * 72)
        print(f"  outputs, one folder per case, under {RESULTS_DIR}/")
        for c in cases:
            st = c["stats"]
            print(f"  {os.path.basename(st['directory']):<18} "
                  f"{st['n_seeds']:>2} seeds  mean R2 "
                  f"{st['mean_r2_all_seeds']:.4f}  "
                  f"(worst seed {st['worst_seed']} at "
                  f"{st['mean_r2_worst_seed']:.4f}, best seed "
                  f"{st['best_seed']} at {st['mean_r2_best_seed']:.4f})"
                  + ("  [from an earlier job]" if c.get("resumed") else ""))
        print("=" * 72, flush=True)

    run_failed.extend(failed)
    if VIEWS:
        for h in VIEWS:
            with view_context(h):
                outputs({o: c[h] for o, c in by_off.items()},
                        {l: c[h] for l, c in by_size.items()},
                        {o: c[h] for o, c in fig_off.items()}, view=h)
    else:
        outputs(by_off, by_size, fig_off)
    if clean:
        remove_empty_cache()

    if failed:
        print(chr(10) + "!" * 72)
        print(f"  {len(failed)} case(s) FAILED and were skipped: "
              f"{', '.join(failed)}")
        print("  every other case above completed; tracebacks are earlier "
              "in this log")
        print("!" * 72, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
