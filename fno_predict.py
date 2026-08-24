"""Autoregressive prediction and evaluation for the Cahn-Hilliard FNO."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

import fno_train
from fno_train import (DT_STEP, ch_ground_truth, free_energy, load_checkpoint,
                       r2_score, rollout_frames)

PRECISION = "fp64"              # "fp32" | "fp64" | "tf32" -- evaluation only;
fno_train.PRECISION = PRECISION  # an fp32 checkpoint is cast to it on load

MODEL_PATH = "fno_ch.pt"
RESULTS_DIR = "results"
LABEL = "fno_ch"
AUTO_LABEL = True               # tag outputs when the checkpoint has no hard constraints

EVAL_SEEDS = (99999, 54321, 77777, 13579, 24680,
              31415, 65537, 80085, 90210, 12321)

EVAL_MODE = "all"               # "all" | "critical" | "off_critical" | "both"
CRITICAL_OFF = 0.0
DATASET_OFFS = (-0.4, -0.2, 0.0, 0.2, 0.4)      # the compositions ch.py provides
OFF_CRITICAL_OFFS = (-0.4, 0.4)

SWEEP_SEEDS = (99999, 54321, 77777)   # seeds for every case except the graded one;
                                      # set = EVAL_SEEDS for 10 everywhere (+22 GB gt_cache)
SWEEP_CONSISTENT_SEEDS = True         # draw every sweep curve from SWEEP_SEEDS, so the
                                      # panels are comparable to each other
SIZE_SWEEP = (128, 160, 192, 256)     # panel (b) of the reference figure;
                                      # () disables it (saves ~6.9 GB of gt_cache)

LATTICE = 128

START_TIME = 100.0
END_TIME = 2000.0
R2_THRESHOLD = 0.80
EARLY_STOP = False

SNAPSHOT_TIMES = (100.0, 500.0, 1000.0, 2000.0)
DIAG_EVERY = 10
DIVERGE_PSI = 5.0               # stop the rollout once |psi| leaves physical range
R2_FLOOR = -1.0                 # floor for reported R2; the model has failed by then
SHOW_SPREAD = True
SWEEP_MARKERS = 14              # markers per curve on the sweep figure
SWEEP_YMIN = None               # pin the sweep y-axis (e.g. 0.5); None autoscales

GT_CACHE = "gt_cache"
CACHE_RESULTS = True
RESULT_CACHE = "result_cache"

POROD_BAND = (0.3, 0.7)
LENGTH_KEY = "L_zero"

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
            "L_grad": length_from_gradient(psi)}


def fit_growth_exponent(t, L):
    """Least-squares log L = n log t + c; Lifshitz-Slyozov gives n = 1/3."""
    t, L = np.asarray(t, dtype=np.float64), np.asarray(L, dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(L) & (t > 0) & (L > 0)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(t[m]), np.log(L[m]), 1)[0])


def eval_offsets():
    """The composition offsets EVAL_MODE asks for."""
    if EVAL_MODE == "all":
        return tuple(sorted(DATASET_OFFS))
    if EVAL_MODE == "critical":
        return (CRITICAL_OFF,)
    if EVAL_MODE == "off_critical":
        return tuple(sorted(OFF_CRITICAL_OFFS))
    if EVAL_MODE == "both":
        return tuple(sorted((CRITICAL_OFF,) + tuple(OFF_CRITICAL_OFFS)))
    raise ValueError('EVAL_MODE must be "all", "critical", "off_critical" or '
                     f'"both", not {EVAL_MODE!r}')


def seeds_for(off):
    """The graded case gets every seed; the sweep compositions get a subset."""
    return EVAL_SEEDS if abs(off) < 1e-12 else SWEEP_SEEDS


def case_suffix(off, lattice=None):
    """Critical runs at the default lattice keep the plain filenames."""
    tag = "" if abs(off) < 1e-12 else f"_off{off:+.2f}"
    return tag if lattice is None else f"{tag}_l{lattice}"


def _cache_key(off, lattice):
    """Everything that changes what a cached rollout contains."""
    st = os.stat(MODEL_PATH)
    key = (st.st_size, int(st.st_mtime), lattice, off, START_TIME, END_TIME,
           EARLY_STOP, R2_THRESHOLD, DIAG_EVERY, DIVERGE_PSI, R2_FLOOR,
           tuple(sorted(SNAPSHOT_TIMES)))
    return hashlib.md5(repr(key).encode()).hexdigest()[:12]


def _result_path(off, seed, lattice):
    label = MODEL_FLAGS["label"]
    return os.path.join(RESULT_CACHE, f"{label}{case_suffix(off)}_l{lattice}"
                                      f"_seed{seed}_{_cache_key(off, lattice)}"
                                      f".npz")


def _save_result(path, res):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    order = sorted(res["snapshots"])
    flat = {"times": res["times"], "r2": res["r2"],
            "mass_drift": res["mass_drift"], "diag_times": res["diag_times"],
            "energy_true": res["energy_true"],
            "energy_pred": res["energy_pred"],
            "t_stop": np.asarray(res["t_stop"]),
            "stopped_early": np.asarray(res["stopped_early"]),
            "diverged_at": np.asarray(np.nan if res.get("diverged_at") is None
                                      else res["diverged_at"]),
            "snap_t": np.asarray(order, dtype=np.float64)}
    if order:
        flat["snap_true"] = np.stack([res["snapshots"][t][0] for t in order])
        flat["snap_pred"] = np.stack([res["snapshots"][t][1] for t in order])
    for name, arr in res["L_true"].items():
        flat[f"Ltrue_{name}"] = arr
    for name, arr in res["L_pred"].items():
        flat[f"Lpred_{name}"] = arr
    np.savez(path, **flat)


def _load_result(path):
    with np.load(path) as z:
        order = z["snap_t"]
        snaps = {float(t): (z["snap_true"][i], z["snap_pred"][i])
                 for i, t in enumerate(order)} if len(order) else {}
        return {"times": z["times"], "r2": z["r2"],
                "mass_drift": z["mass_drift"],
                "diag_times": z["diag_times"],
                "energy_true": z["energy_true"],
                "energy_pred": z["energy_pred"],
                "t_stop": float(z["t_stop"]),
                "diverged_at": (None if "diverged_at" not in z.files
                                or not np.isfinite(z["diverged_at"])
                                else float(z["diverged_at"])),
                "stopped_early": bool(z["stopped_early"]),
                "L_true": {k[6:]: z[k] for k in z.files
                           if k.startswith("Ltrue_")},
                "L_pred": {k[6:]: z[k] for k in z.files
                           if k.startswith("Lpred_")},
                "snapshots": snaps}


def evaluate_seed(model, seed, device, off=0.0, lattice=None):
    lattice = LATTICE if lattice is None else lattice
    gt_times, gt_frames = ch_ground_truth(lattice, seed, off, END_TIME,
                                          DT_STEP, cache_dir=GT_CACHE,
                                          device=device)
    times, r2s, drift = [], [], []
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
        for t in list(want):
            frame = gt_frames[int(np.argmin(np.abs(gt_times - t)))]
            snaps[t] = (frame.astype(np.float64),
                        np.full_like(frame, np.nan, dtype=np.float64))
            want.remove(t)

    return {"times": np.asarray(times), "r2": np.asarray(r2s),
            "mass_drift": np.asarray(drift), "t_stop": t_stop,
            "diverged_at": diverged_at,
            "stopped_early": stopped, "diag_times": np.asarray(diag_t),
            "L_true": {k: np.asarray(v) for k, v in L_true.items()},
            "L_pred": {k: np.asarray(v) for k, v in L_pred.items()},
            "energy_true": np.asarray(energy_t),
            "energy_pred": np.asarray(energy_p), "snapshots": snaps}


def figure_r2_and_snapshots(results, seeds, path, off=0.0, lattice=None):
    lattice = LATTICE if lattice is None else lattice
    label, note = MODEL_FLAGS["label"], MODEL_FLAGS["note"]
    first = results[0]
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

    lo_b = mean_curve
    if SHOW_SPREAD and len(results) > 2:
        lo_b = np.nanpercentile(stack, 10, axis=0)
        hi_b = np.nanpercentile(stack, 90, axis=0)
        ax.fill_between(times, lo_b, hi_b, color="k", alpha=0.13, lw=0,
                        label="10th-90th percentile over seeds")
    ax.plot(times, mean_curve, color="k", lw=2.8,
            label=f"mean over {len(results)} unseen seeds")
    ax.axhline(overall, color="#2B5B84", ls="-.", lw=2.0,
               label=f"mean of all $R^2$ = {overall:.4f}")

    ax.annotate(f"mean of all $R^2$ = {overall:.4f}",
                xy=(END_TIME, overall), xytext=(-6, 6),
                textcoords="offset points", ha="right", fontsize=10,
                color="#2B5B84", fontweight="bold")
    ax.axhline(R2_THRESHOLD, color="#D9534F", ls="--", lw=2.0,
               label=f"advisor threshold $R^2$ = {R2_THRESHOLD:.2f}")

    below = np.nonzero(mean_curve < R2_THRESHOLD)[0]
    t_cross = float(times[below[0]]) if len(below) else None
    if t_cross is not None:
        ax.axvline(t_cross, color="#D9534F", ls=":", lw=1.6, alpha=0.8)
        ax.annotate(f"mean crosses at $t$ = {t_cross:.0f}",
                    xy=(t_cross, R2_THRESHOLD), xytext=(8, 14),
                    textcoords="offset points", fontsize=10,
                    color="#D9534F", fontweight="bold")

    lo = float(np.nanmin(lo_b))
    ax.set_xlim(START_TIME, END_TIME)
    ax.set_ylim(max(-1.0, min(-0.05, lo - 0.05)), 1.02)
    ax.set_xlabel("real time $t$")
    ax.set_ylabel("$R^2$ vs. ground truth")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(loc="lower left", fontsize=9, ncol=2)

    held = sum(1 for r in results if not r["stopped_early"])
    blew = sum(1 for r in results if r.get("diverged_at") is not None)
    verdict = (f"mean $R^2$ = {overall:.4f}"
               + (f", crosses {R2_THRESHOLD:.2f} at $t$ = {t_cross:.0f}"
                  if t_cross is not None else
                  f", never drops below {R2_THRESHOLD:.2f}")
               + f"; {held}/{len(results)} seeds held throughout"
               + (f"; {blew}/{len(results)} DIVERGED" if blew else ""))
    comp = ("critical quench" if abs(off) < 1e-12
            else f"off-critical quench $\\bar\\psi={off:+.2f}$")
    ax.set_title(f"{label} — autoregressive rollout at "
                 f"$l={lattice}$, $t={START_TIME:.0f}$ to "
                 f"${END_TIME:.0f}$, {comp}{note}  ({verdict})",
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

    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def figure_kinetics(res, path, off=0.0, lattice=None):
    lattice = LATTICE if lattice is None else lattice
    if len(res["diag_times"]) == 0:
        return
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.0), facecolor="white")
    (a, b, c), (d, e, f) = axes
    t = res["diag_times"]
    Lt, Lp = res["L_true"][LENGTH_KEY], res["L_pred"][LENGTH_KEY]

    a.loglog(t, Lt, color=C_TRUE, lw=2.4, label="ground truth")
    a.loglog(t, Lp, color=C_PRED, lw=2.0, ls="--", label="FNO")
    ok = np.isfinite(Lt) & (Lt > 0)
    if ok.any():
        t0 = np.exp(np.mean(np.log(t[ok])))
        l0 = np.exp(np.mean(np.log(Lt[ok])))
        tr = np.array([t[0], t[-1]])
        a.loglog(tr, l0 * (tr / t0) ** (1 / 3), "k:", lw=2,
                 label=r"Lifshitz–Slyozov $t^{1/3}$")
    a.set_xlabel("$t$"); a.set_ylabel(f"$L(t)$  [{LENGTH_KEY}]")
    a.set_title(f"Domain growth:  truth $n$ = {fit_growth_exponent(t, Lt):.3f}, "
                f" FNO $n$ = {fit_growth_exponent(t, Lp):.3f}", fontweight="bold")
    a.grid(alpha=0.3, which="both", ls=":"); a.legend(fontsize=9)

    avail = [x for x in SNAPSHOT_TIMES if x in res["snapshots"]]
    cmap = plt.get_cmap("viridis")

    for i, tt in enumerate(avail):
        col = cmap(i / max(len(avail) - 1, 1))
        true, pred = res["snapshots"][tt]
        k, s_t = structure_factor(true)
        _, s_p = structure_factor(pred)
        m = k > 0
        b.loglog(k[m], s_t[m], color=col, lw=2.0, label=f"$t$ = {tt:.0f}")
        b.loglog(k[m], s_p[m], color=col, lw=1.4, ls="--")
    k_ref, s_ref = structure_factor(res["snapshots"][avail[-1]][0])
    anchor = float(np.interp(0.45, k_ref[1:], s_ref[1:]))
    kk = np.array(POROD_BAND)
    b.loglog(kk, anchor * (kk / 0.45) ** -3, "k:", lw=2, label=r"Porod $k^{-3}$")
    b.set_xlabel("$k$"); b.set_ylabel("$S(k,t)$")
    b.set_title("Structure factor (solid: truth, dashed: FNO)", fontweight="bold")
    b.grid(alpha=0.3, which="both", ls=":"); b.legend(fontsize=8)

    for i, tt in enumerate(avail):
        col = cmap(i / max(len(avail) - 1, 1))
        for field, ls, lw in ((res["snapshots"][tt][0], "-", 2.0),
                              (res["snapshots"][tt][1], "--", 1.4)):
            k, s_k = structure_factor(field)
            r, c_r = correlation_function(field)
            L = length_from_zero_crossing(r, c_r)
            if np.isfinite(L) and L > 0:
                m = k > 0
                c.loglog(k[m] * L, s_k[m] / L ** D, color=col, ls=ls, lw=lw)
    c.set_xlabel("$kL$"); c.set_ylabel("$L^{-d}S(k,t)$")
    c.set_title(r"Dynamical scaling  $S = L^{d}f(kL)$  (Eq. 1.90)",
                fontweight="bold")
    c.grid(alpha=0.3, which="both", ls=":")

    for i, tt in enumerate(avail):
        col = cmap(i / max(len(avail) - 1, 1))
        for field, ls, lw in ((res["snapshots"][tt][0], "-", 2.0),
                              (res["snapshots"][tt][1], "--", 1.4)):
            r, c_r = correlation_function(field)
            L = length_from_zero_crossing(r, c_r)
            if np.isfinite(L) and L > 0:
                d.plot(r / L, c_r, color=col, ls=ls, lw=lw)
    d.axhline(0, color="k", lw=0.8)
    d.set_xlim(0, 5); d.set_xlabel("$r/L$"); d.set_ylabel("$C(r,t)$")
    d.set_title(r"Scaling function  $C = g(r/L)$  (Eq. 1.88)", fontweight="bold")
    d.grid(alpha=0.3, ls=":")

    e.plot(t, res["energy_true"], color=C_TRUE, lw=2.4, label="ground truth")
    e.plot(t, res["energy_pred"], color=C_PRED, lw=2.0, ls="--", label="FNO")
    mono = bool(np.all(np.diff(res["energy_pred"]) <= 1e-9))
    e.set_xlabel("$t$"); e.set_ylabel(r"$\langle f \rangle$")
    e.set_title(f"Free-energy decay — FNO monotone: {'yes' if mono else 'no'}",
                fontweight="bold")
    e.grid(alpha=0.3, ls=":"); e.legend(fontsize=9)

    for key, col in zip(("L_zero", "L_peak", "L_moment", "L_grad"),
                        ("#1B7837", "#2B5B84", "#B35806", "#762A83")):
        f.loglog(t, res["L_true"][key], color=col, lw=2.0, label=f"{key} (truth)")
        f.loglog(t, res["L_pred"][key], color=col, lw=1.3, ls="--")
    f.set_xlabel("$t$"); f.set_ylabel("$L(t)$")
    f.set_title(f"Length estimators (dashed: FNO)\nmax mass drift over rollout "
                f"= {res['mass_drift'].max():.2e}", fontweight="bold")
    f.grid(alpha=0.3, which="both", ls=":"); f.legend(fontsize=8)

    comp = ("" if abs(off) < 1e-12
            else f", $\\bar\\psi={off:+.2f}$")
    fig.suptitle(f"{MODEL_FLAGS['label']}{comp}{MODEL_FLAGS['note']} — "
                 f"phase-ordering kinetics at $l={lattice}$ "
                 "(Puri, Kinetics of Phase Transitions, ch. 1)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


SWEEP_COLORS = ("#08519C", "#4292C6", "#C51B8A", "#41AB5D", "#005A32")
SWEEP_SHAPES = ("o", "s", "^", "D", "v", "<", ">", "p")


def _sweep_panel(ax, curves, labels, ylabel=True):
    """One panel of the sweep figure: R2(t), one line per case."""
    lo = 1.0
    for i, ((t, r), lab) in enumerate(zip(curves, labels)):
        r = np.clip(np.nan_to_num(np.asarray(r, dtype=np.float64),
                                  nan=R2_FLOOR, posinf=1.0, neginf=R2_FLOOR),
                    R2_FLOOR, 1.0)
        every = max(1, len(t) // max(SWEEP_MARKERS, 1))
        ax.plot(t, r, color=SWEEP_COLORS[i % len(SWEEP_COLORS)],
                marker=SWEEP_SHAPES[i % len(SWEEP_SHAPES)], markevery=every,
                markersize=5, lw=1.6, label=lab)
        lo = min(lo, float(r.min()))
    ax.axhline(R2_THRESHOLD, color="#888888", ls="--", lw=1.2, zorder=0)
    ax.set_xlim(START_TIME, END_TIME)
    if SWEEP_YMIN is not None:
        bottom = float(SWEEP_YMIN)
    else:
        bottom = min(lo - 0.02, R2_THRESHOLD - 0.02)
        bottom = float(np.clip(bottom, R2_FLOOR - 0.05, R2_THRESHOLD - 0.02))
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
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def run_case(model, device, off, lattice=None, seeds=None):
    """One case: roll every seed out, write CSVs, figures and summary."""
    label = MODEL_FLAGS["label"]
    suffix = case_suffix(off, lattice)
    lat = LATTICE if lattice is None else lattice
    seeds = seeds_for(off) if seeds is None else tuple(seeds)
    tag = "critical" if abs(off) < 1e-12 else f"off-critical {off:+.2f}"
    if lattice is not None:
        tag += f", l={lat}"
    if not MODEL_FLAGS["hard_constraints"]:
        tag += ", hard constraints OFF"
    bar = "=" * 72
    print(f"\n{bar}\n[case] {tag}: l={lat}, {len(seeds)} seeds, "
          f"t={START_TIME:.0f}..{END_TIME:.0f}\n{bar}", flush=True)

    results, summary = [], []
    for seed in seeds:
        print(f"\n[seed {seed}] l={lat}, off={off:+.2f}", flush=True)
        cache = _result_path(off, seed, lat)
        if CACHE_RESULTS and os.path.exists(cache):
            res = _load_result(cache)
            print(f"  -> reusing {cache}", flush=True)
        else:
            res = evaluate_seed(model, seed, device, off, lat)
            if CACHE_RESULTS:
                _save_result(cache, res)
                print(f"  -> cached {cache}", flush=True)

        n_above = int(np.sum(res["r2"] >= R2_THRESHOLD))
        verdict = (f"R2 first fell below {R2_THRESHOLD} at t = {res['t_stop']:.0f}"
                   if res["stopped_early"] else
                   f"held R2 >= {R2_THRESHOLD} through t = {END_TIME:.0f}")
        if res.get("diverged_at") is not None:
            verdict += (f"; ROLLOUT DIVERGED at t = "
                        f"{res['diverged_at']:.0f}")
        print(f"  -> {verdict}; {n_above} steps above threshold, "
              f"mean R2 = {res['r2'].mean():.4f}", flush=True)
        results.append(res)

        summary.append({
            "seed": seed, "t_stop": res["t_stop"],
            "stopped_early": res["stopped_early"],
            "steps_above_threshold": n_above,
            "diverged_at": res.get("diverged_at"),
            "mean_r2": float(res["r2"].mean()),
            "final_r2": float(res["r2"][-1]),
            "r2_at": {str(t): float(r2_score(*res["snapshots"][t]))
                      for t in SNAPSHOT_TIMES if t in res["snapshots"]},
            "growth_exponent_truth": fit_growth_exponent(
                res["diag_times"], res["L_true"][LENGTH_KEY]),
            "growth_exponent_fno": fit_growth_exponent(
                res["diag_times"], res["L_pred"][LENGTH_KEY]),
            "max_mass_drift": float(res["mass_drift"].max())})
        np.savetxt(os.path.join(RESULTS_DIR,
                                f"{label}{suffix}_r2_seed{seed}.csv"),
                   np.column_stack([res["times"], res["r2"]]),
                   delimiter=",", header="t,r2", comments="")

    figure_r2_and_snapshots(
        results, seeds,
        os.path.join(RESULTS_DIR, f"{label}{suffix}_r2_and_snapshots.png"),
        off, lat)
    figure_kinetics(
        results[0],
        os.path.join(RESULTS_DIR, f"{label}{suffix}_kinetics.png"), off, lat)

    per_seed = np.array([r["r2"].mean() for r in results])
    steps = np.array([row["steps_above_threshold"] for row in summary],
                     dtype=float)
    n = len(per_seed)
    sd = float(per_seed.std(ddof=1)) if n > 1 else 0.0
    stats = {
        "off": off,
        "lattice": lat,
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
        "mean_r2_best_seed": float(per_seed.max()),
        "steps_above_threshold_total": int(steps.sum()),
        "steps_above_threshold_mean": float(steps.mean()),
        "seeds_holding_throughout": int(sum(1 for r in results
                                            if not r["stopped_early"])),
        "seeds_diverged": int(sum(1 for r in results
                                  if r.get("diverged_at") is not None)),
        "seeds": summary}
    with open(os.path.join(RESULTS_DIR,
                           f"{label}{suffix}_summary.json"), "w") as fh:
        json.dump(stats, fh, indent=1)

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
    print(f"  worst seed {stats['mean_r2_worst_seed']:.4f}, best seed "
          f"{stats['mean_r2_best_seed']:.4f}, "
          f"{stats['seeds_holding_throughout']}/{n} held throughout, "
          f"{stats['steps_above_threshold_total']} steps total")
    print(bar, flush=True)

    n_max = max(len(r["r2"]) for r in results)
    stack = np.full((len(results), n_max), np.nan)
    for i, r in enumerate(results):
        stack[i, :len(r["r2"])] = r["r2"]
    longest = results[int(np.argmax([len(r["r2"]) for r in results]))]
    return {"off": off, "lattice": lat, "times": longest["times"],
            "mean_curve": np.nanmean(stack, axis=0), "stats": stats}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"'{MODEL_PATH}' not found. Run fno_train.py "
                                f"first.")
    model, ckpt = load_checkpoint(MODEL_PATH, device=device)
    cfg = model.config
    dtype = next(model.parameters()).dtype
    print(f"[model] {MODEL_PATH}  ({model.n_parameters():,} parameters, "
          f"backbone={cfg.get('backbone', 'physical_k')}, dtype={dtype})")
    print(f"[model] hard_mass_conservation="
          f"{cfg.get('hard_mass_conservation', True)}, residual_prediction="
          f"{cfg.get('residual_prediction', True)}"
          + (f", mode_scaling={cfg['mode_scaling']}, zero_dc={cfg.get('zero_dc')}"
             if cfg.get("backbone", "physical_k") == "physical_k" else ""))
    if "best" in ckpt:
        print(f"[model] selected at epoch {ckpt['best'].get('epoch')} "
              f"(rollout score {ckpt['best'].get('score', float('nan')):.4f}, "
              f"survives to t={ckpt['best']['survival']:.0f})")

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

    failed = []

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

    by_off = {}
    for off in eval_offsets():
        res = attempt(f"composition {off:+.2f}", run_case, model, device, off)
        if res is not None:
            by_off[off] = res

    by_size = {}
    for lat in SIZE_SWEEP:
        res = attempt(f"lattice {lat}", run_case, model, device, CRITICAL_OFF,
                      lattice=lat, seeds=SWEEP_SEEDS)
        if res is not None:
            by_size[lat] = res

    fig_off = dict(by_off)
    if (SWEEP_CONSISTENT_SEEDS and CRITICAL_OFF in fig_off
            and tuple(seeds_for(CRITICAL_OFF)) != tuple(SWEEP_SEEDS)):
        same = by_size.get(LATTICE)
        if same is None:
            same = attempt(f"critical on the sweep seeds", run_case, model,
                           device, CRITICAL_OFF, lattice=LATTICE,
                           seeds=SWEEP_SEEDS)
        if same is not None:
            fig_off[CRITICAL_OFF] = same
            print(f"\n[sweep] psi_0 = {CRITICAL_OFF:+.1f} redrawn from the "
                  f"{len(SWEEP_SEEDS)} sweep seeds so every curve in "
                  f"{MODEL_FLAGS['label']}_sweeps.png matches; the "
                  f"{len(EVAL_SEEDS)}-seed result stays in "
                  f"{MODEL_FLAGS['label']}_summary.json", flush=True)

    label = MODEL_FLAGS["label"]
    attempt("sweep figure", figure_sweeps, fig_off, by_size,
            os.path.join(RESULTS_DIR, f"{label}_sweeps.png"))

    if len(by_off) > 1:
        bar = "=" * 72
        print("\n" + bar)
        print("  composition sweep")
        print(f"  {'psi_0':>7} {'seeds':>6} {'mean R2':>9} {'R2@2000':>9} "
              f"{'held':>7} {'steps/seed':>11}")
        for o in sorted(by_off):
            st = by_off[o]["stats"]
            print(f"  {o:>+7.1f} {st['n_seeds']:>6} "
                  f"{st['mean_r2_all_seeds']:>9.4f} "
                  f"{by_off[o]['mean_curve'][-1]:>9.4f} "
                  f"{st['seeds_holding_throughout']:>3}/{st['n_seeds']:<3} "
                  f"{st['steps_above_threshold_mean']:>11.0f}")
        print(bar, flush=True)

    if failed:
        print(chr(10) + "!" * 72)
        print(f"  {len(failed)} case(s) FAILED and were skipped: "
              f"{', '.join(failed)}")
        print("  every other case above completed; tracebacks are earlier "
              "in this log")
        print("!" * 72, flush=True)

if __name__ == "__main__":
    main()
