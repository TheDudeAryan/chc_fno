"""Autoregressive prediction and evaluation for the Cahn-Hilliard FNO.

    python fno_predict.py

Everything is configured in the CONFIG block below -- no command-line flags.
Imports the solver and the model from fno_train.py, so keep the two files in
the same directory.

Follows the advisor's protocol exactly: start from the true state at t = 100 on
unseen trajectories at l = 128, step autoregressively to t = 2000, score every
snapshot with R2, and stop once R2 falls below 0.80.

It also checks the physics R2 cannot see, following Puri, *Kinetics of Phase
Transitions*, ch. 1:

  * dynamical scaling      C(r,t) = g(r/L),  S(k,t) = L^d f(kL)   (1.88, 1.90)
  * Porod tail             S ~ k^-(d+1) = k^-3 in d = 2           (1.100)
  * conservation sum rule  S(0,t) = 0, and f(p) ~ p^4 as p -> 0   (1.184)
  * Lifshitz-Slyozov       L(t) ~ t^(1/3) for conserved kinetics
  * free-energy decay, which must be monotone since CH is a gradient flow

Writes into RESULTS_DIR: a two-panel R2 + snapshot figure, a six-panel kinetics
figure, per-seed R2 csv, and a json summary.
"""

from __future__ import annotations

import functools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from fno_train import (DT_STEP, ch_ground_truth, free_energy, load_checkpoint,
                       r2_score, rollout_frames)

# ==========================================================================
# CONFIG
# ==========================================================================

MODEL_PATH = "fno_ch.pt"
RESULTS_DIR = "results"
LABEL = "fno_ch"

# Unseen trajectories.  ch.py draws training seeds from [0, 10000) and
# validation seeds from [10000, 20000); fno_train.py additionally spends
# 31337 / 24601 / 8675309 on model selection.  These are outside all of that.
EVAL_SEEDS = (99999, 54321, 77777)
OFF = 0.0                      # order-parameter mean; 0.0 = critical quench
LATTICE = 128                  # the advisor's evaluation resolution

START_TIME = 100.0
END_TIME = 2000.0
R2_THRESHOLD = 0.80
EARLY_STOP = True              # False keeps rolling, to see the full decay

SNAPSHOT_TIMES = (100.0, 500.0, 1000.0, 2000.0)   # Puri fig. 1.10 uses these
DIAG_EVERY = 10                # steps between kinetics diagnostics
GT_CACHE = "gt_cache"

# The Porod window is bounded below by the peak (k ~ 2*pi/L) and above by the
# interface width (xi_b = sqrt(2), so k ~ 0.7).  On a 128^2 lattice at
# t <= 2000 that leaves only k in [0.3, 0.7]; outside it a fit returns the peak
# on one side and the tanh interface profile on the other, not Porod.
POROD_BAND = (0.3, 0.7)
LENGTH_KEY = "L_zero"          # primary L(t) estimator; see all_lengths()

PHASE_CMAP, ERR_CMAP = "RdBu_r", "magma"
C_TRUE, C_PRED = "#1B7837", "#762A83"

# ==========================================================================

D = 2                                          # spatial dimension
_trapz = getattr(np, "trapz", None) or np.trapezoid


# ==========================================================================
# Phase-ordering diagnostics
# ==========================================================================

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
    a = np.minimum(np.arange(n), n - np.arange(n))      # periodic distance
    rx, ry = np.meshgrid(a, a, indexing="ij")
    idx = np.clip(np.rint(np.sqrt(rx ** 2 + ry ** 2)).astype(np.int64),
                  0, n // 2)
    counts = np.bincount(idx.ravel(), minlength=n // 2 + 1).astype(np.float64)
    return idx.ravel(), counts, np.arange(n // 2 + 1, dtype=np.float64)


def structure_factor(field):
    """Angle-averaged S(k,t) = |FFT(psi - <psi>)|^2 / N^2.

    Normalised so (1/N^2) sum_k S(k) = <(psi - <psi>)^2>, the discrete form of
    Puri's int d^dk/(2pi)^d S(k) = C(0) at dx = 1.
    """
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
    """L = first zero of C(r,t).

    The primary estimator: it is a property of the scaling function g(r/L)
    alone, so it is insensitive to the Porod tail and the lattice cutoff.  On
    exact solver output it recovers n = 0.318 against Lifshitz-Slyozov's 1/3.
    """
    sign = np.sign(c_r)
    cross = np.nonzero((sign[:-1] > 0) & (sign[1:] <= 0))[0]
    if len(cross) == 0:
        return float("nan")
    i = int(cross[0])
    c0, c1 = c_r[i], c_r[i + 1]
    return float(r[i] if c0 == c1
                 else r[i] + (r[i + 1] - r[i]) * c0 / (c0 - c1))


def length_from_peak(k, s_k):
    """L = 2*pi/k_m from the peak of S(k,t), parabolically refined.

    Quantised: bins are 2*pi/N apart, so by t = 2000 on a 128 lattice the peak
    sits around bin 3.  A cross-check, not the headline number.
    """
    if len(s_k) < 3:
        return float("nan")
    i = int(np.argmax(s_k[1:]) + 1)     # skip k = 0, where S vanishes (1.184)
    if 0 < i < len(s_k) - 1:
        y0, y1, y2 = s_k[i - 1], s_k[i], s_k[i + 1]
        den = y0 - 2.0 * y1 + y2
        shift = 0.0 if den == 0 else np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0)
        km = k[i] + shift * (k[1] - k[0])
    else:
        km = k[i]
    return float(2.0 * np.pi / km) if km > 0 else float("nan")


def length_from_moment(k, s_k, k_cut=1.0):
    """L = 2*pi/<k> with the k^(d-1) radial measure and a cutoff.

    The cutoff is mandatory: with the Porod tail S ~ k^-(d+1) the first-moment
    integrand goes as k^-1, so <k> diverges logarithmically with the lattice
    cutoff and the "domain size" would track resolution rather than physics.
    Even cut off it stays cutoff-sensitive (n ~ 0.24 against 1/3), which is why
    it is not the primary estimator.
    """
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


# ==========================================================================
# Rollout
# ==========================================================================

def evaluate_seed(model, seed, device):
    gt_times, gt_frames = ch_ground_truth(LATTICE, seed, OFF, END_TIME,
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

    t_stop, stopped = END_TIME, False
    for k, (t, true, pred) in enumerate(
            rollout_frames(model, gt_times, gt_frames, START_TIME, END_TIME,
                           device)):
        r2 = r2_score(true, pred)
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

    return {"times": np.asarray(times), "r2": np.asarray(r2s),
            "mass_drift": np.asarray(drift), "t_stop": t_stop,
            "stopped_early": stopped, "diag_times": np.asarray(diag_t),
            "L_true": {k: np.asarray(v) for k, v in L_true.items()},
            "L_pred": {k: np.asarray(v) for k, v in L_pred.items()},
            "energy_true": np.asarray(energy_t),
            "energy_pred": np.asarray(energy_p), "snapshots": snaps}


# ==========================================================================
# Figures
# ==========================================================================

def figure_r2_and_snapshots(results, seeds, path):
    first = results[0]
    snaps = [t for t in SNAPSHOT_TIMES if t in first["snapshots"]]
    ncol = max(len(snaps), 1)

    fig = plt.figure(figsize=(3.4 * ncol + 1.2, 11.5), facecolor="white")
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.25, 3], hspace=0.20)
    gs = outer[1].subgridspec(3, ncol, hspace=0.06, wspace=0.06)

    ax = fig.add_subplot(outer[0])
    for seed, res in zip(seeds, results):
        ax.plot(res["times"], res["r2"], lw=1.6, alpha=0.85, label=f"seed {seed}")
    if len(results) > 1:
        n = min(len(r["r2"]) for r in results)
        ax.plot(results[0]["times"][:n],
                np.mean([r["r2"][:n] for r in results], axis=0),
                color="k", lw=2.6, label="mean")
    ax.axhline(R2_THRESHOLD, color="#D9534F", ls="--", lw=1.8,
               label=f"advisor threshold R$^2$ = {R2_THRESHOLD:.2f}")
    for res in results:
        if res["stopped_early"]:
            ax.axvline(res["t_stop"], color="#D9534F", ls=":", lw=1.2, alpha=0.7)
    lo = float(np.min([r["r2"].min() for r in results]))
    ax.set_xlim(START_TIME, END_TIME)
    ax.set_ylim(max(-1.0, min(-0.05, lo - 0.05)), 1.02)
    ax.set_xlabel("real time $t$")
    ax.set_ylabel("$R^2$ vs. ground truth")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(loc="lower left", fontsize=9, ncol=2)

    stopped = [(s, r) for s, r in zip(seeds, results) if r["stopped_early"]]
    held = len(results) - len(stopped)
    parts = []
    if held:
        parts.append(f"{held}/{len(results)} held through $t$ = {END_TIME:.0f}")
    if stopped:
        parts.append("dropped below threshold at " + ", ".join(
            f"$t$ = {r['t_stop']:.0f} (seed {s})" for s, r in stopped))
    ax.set_title(f"{LABEL} — autoregressive rollout at $l={LATTICE}$, "
                 f"$t={START_TIME:.0f} \\to {END_TIME:.0f}$ ({'; '.join(parts)})",
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


def figure_kinetics(res, path):
    if len(res["diag_times"]) == 0:
        return
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.0), facecolor="white")
    (a, b, c), (d, e, f) = axes
    t = res["diag_times"]
    Lt, Lp = res["L_true"][LENGTH_KEY], res["L_pred"][LENGTH_KEY]

    # (a) growth law
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

    # (b) structure factor
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

    # (c) dynamical-scaling collapse
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

    # (d) correlation-function collapse
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

    # (e) free energy -- must be monotone for a gradient flow
    e.plot(t, res["energy_true"], color=C_TRUE, lw=2.4, label="ground truth")
    e.plot(t, res["energy_pred"], color=C_PRED, lw=2.0, ls="--", label="FNO")
    mono = bool(np.all(np.diff(res["energy_pred"]) <= 1e-9))
    e.set_xlabel("$t$"); e.set_ylabel(r"$\langle f \rangle$")
    e.set_title(f"Free-energy decay — FNO monotone: {'yes' if mono else 'no'}",
                fontweight="bold")
    e.grid(alpha=0.3, ls=":"); e.legend(fontsize=9)

    # (f) all four length estimators + conservation
    for key, col in zip(("L_zero", "L_peak", "L_moment", "L_grad"),
                        ("#1B7837", "#2B5B84", "#B35806", "#762A83")):
        f.loglog(t, res["L_true"][key], color=col, lw=2.0, label=f"{key} (truth)")
        f.loglog(t, res["L_pred"][key], color=col, lw=1.3, ls="--")
    f.set_xlabel("$t$"); f.set_ylabel("$L(t)$")
    f.set_title(f"Length estimators (dashed: FNO)\nmax mass drift over rollout "
                f"= {res['mass_drift'].max():.2e}", fontweight="bold")
    f.grid(alpha=0.3, which="both", ls=":"); f.legend(fontsize=8)

    fig.suptitle(f"{LABEL} — phase-ordering kinetics at $l={LATTICE}$ "
                 "(Puri, Kinetics of Phase Transitions, ch. 1)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ==========================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"'{MODEL_PATH}' not found. Run fno_train.py "
                                f"first.")
    model, ckpt = load_checkpoint(MODEL_PATH, device=device)
    print(f"[model] {MODEL_PATH}  ({model.n_parameters():,} parameters, "
          f"mode_scaling={model.config['mode_scaling']}, "
          f"zero_dc={model.config.get('zero_dc')})")
    if "best" in ckpt:
        print(f"[model] selected at epoch {ckpt['best'].get('epoch')} "
              f"(validation rollout to t={ckpt['best']['survival']:.0f})")

    results, summary = [], []
    for seed in EVAL_SEEDS:
        print(f"\n[seed {seed}] l={LATTICE}, off={OFF:+.1f}", flush=True)
        res = evaluate_seed(model, seed, device)
        n_above = int(np.sum(res["r2"] >= R2_THRESHOLD))
        verdict = (f"R2 first fell below {R2_THRESHOLD} at t = {res['t_stop']:.0f}"
                   if res["stopped_early"] else
                   f"held R2 >= {R2_THRESHOLD} through t = {END_TIME:.0f}")
        print(f"  -> {verdict}; {n_above} steps above threshold, "
              f"mean R2 = {res['r2'].mean():.4f}", flush=True)
        results.append(res)

        summary.append({
            "seed": seed, "t_stop": res["t_stop"],
            "stopped_early": res["stopped_early"],
            "steps_above_threshold": n_above,
            "mean_r2": float(res["r2"].mean()),
            "final_r2": float(res["r2"][-1]),
            "r2_at": {str(t): float(r2_score(*res["snapshots"][t]))
                      for t in SNAPSHOT_TIMES if t in res["snapshots"]},
            "growth_exponent_truth": fit_growth_exponent(
                res["diag_times"], res["L_true"][LENGTH_KEY]),
            "growth_exponent_fno": fit_growth_exponent(
                res["diag_times"], res["L_pred"][LENGTH_KEY]),
            "max_mass_drift": float(res["mass_drift"].max())})
        np.savetxt(os.path.join(RESULTS_DIR, f"{LABEL}_r2_seed{seed}.csv"),
                   np.column_stack([res["times"], res["r2"]]),
                   delimiter=",", header="t,r2", comments="")

    figure_r2_and_snapshots(
        results, EVAL_SEEDS,
        os.path.join(RESULTS_DIR, f"{LABEL}_r2_and_snapshots.png"))
    figure_kinetics(
        results[0], os.path.join(RESULTS_DIR, f"{LABEL}_kinetics.png"))
    with open(os.path.join(RESULTS_DIR, f"{LABEL}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    print("\n" + "=" * 72)
    for s in summary:
        print(f"  seed {s['seed']}: R2 >= {R2_THRESHOLD} until t = "
              f"{s['t_stop']:.0f}  ({s['steps_above_threshold']} steps), "
              f"mean R2 = {s['mean_r2']:.4f}, growth exponent "
              f"{s['growth_exponent_fno']:.3f} "
              f"(truth {s['growth_exponent_truth']:.3f})")
    print("=" * 72)


if __name__ == "__main__":
    main()
