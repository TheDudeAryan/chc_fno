"""The case figures on data whose answers are known.

The seed-averaged kinetics figure (its +-1 sd band in panel (a) and the mean
order parameter of panel (f)), the training-horizon marker, the growth
exponent on each side of that horizon, the finite-size shading and the
half-maximum length L_half.  Every rollout is synthetic, nothing is solved,
and everything is written under _scratch.
"""
import json
import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.figure as mfig
import numpy as np
from matplotlib.collections import PolyCollection

from _paths import SCRATCH

os.chdir(SCRATCH)
import fno_predict as P

S = P._submission()
ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


BASE = os.path.join(SCRATCH, "figures")
shutil.rmtree(BASE, ignore_errors=True)
os.makedirs(BASE)
# FNO's own outputs (the per-case folders of extras/) go under BASE too
P.RESULT_CACHE = os.path.join(BASE, "cache")
P.EXTRAS_ROOT = None

# What a saved figure held, by path: each axes' lines, shaded spans, legend.
SEEN = {}
_savefig = mfig.Figure.savefig


def span_of(patch):
    if hasattr(patch, "get_width"):
        return patch.get_x(), patch.get_x() + patch.get_width()
    x = patch.get_xy()[:, 0]
    return float(x.min()), float(x.max())


def _remember(self, fname, *args, **kwargs):
    if isinstance(fname, str):
        SEEN[os.path.normcase(os.path.abspath(fname))] = {
            "title": "\n".join([t.get_text() for t in self.texts]
                               + [ax.get_title() for ax in self.axes]),
            "axes": [{
                "lines": [(ln.get_label(), np.array(ln.get_xdata(), float),
                           np.array(ln.get_ydata(), float),
                           ln.get_linestyle()) for ln in ax.lines],
                "spans": [span_of(p) for p in ax.patches
                          if p.get_label().startswith(
                              ("finite-size", "$L > l/", "$L_{"))],
                "fills": [np.vstack([q.vertices for q in c.get_paths()])
                          for c in ax.collections
                          if isinstance(c, PolyCollection)],
                "secondary": len(ax.child_axes),
                "legend": ([t.get_text() for t in ax.get_legend().get_texts()]
                           if ax.get_legend() else [])}
                for ax in self.axes]}
    return _savefig(self, fname, *args, **kwargs)


setattr(mfig.Figure, "savefig", _remember)


def seen(path):
    key = os.path.normcase(os.path.abspath(path))
    if key in SEEN:
        return SEEN[key]
    # make_submission.write_case draws into <run folder>/.partial_<token>/
    # and publishes by rename, so a flat file was saved under that name
    folder, name = os.path.split(key)
    for k, v in SEEN.items():
        stage, base = os.path.split(k)
        if (base == name and os.path.dirname(stage) == folder
                and os.path.basename(stage).startswith(".partial")):
            return v
    raise KeyError(path)


def marked(path, h):
    """How many axes of a saved figure have the dashed training-horizon line
    at h, and how many shade a finite-size span (their extents)."""
    lines = spans = 0
    extents = []
    for ax in seen(path)["axes"]:
        lines += any(ls == "--" and len(x) == 2 and np.all(x == h)
                     for _, x, _, ls in ax["lines"])
        if ax["spans"]:
            spans += 1
            extents.append(ax["spans"][0])
    return lines, spans, extents


def line(path, ax, label):
    for name, x, y, _ in seen(path)["axes"][ax]["lines"]:
        if name == label:
            return x, y
    raise KeyError(label)


print("1. L_half: the first r where C(r) falls to half of C(0)")
r = np.arange(8.0)
check(abs(P.length_from_half(r, [1, .8, .4, .1, 0, -.1, -.2, -.1]) - 1.75)
      < 1e-12, "between shells: C = 0.8 at r = 1 and 0.4 at r = 2 -> 1.75")
check(P.length_from_half(r[:4], [1, .75, .5, .2]) == 2.0,
      "a shell exactly at half is the answer: 2.0")
check(abs(P.length_from_half(r, [2, 1.5, 1.2, .8, .5, .1, 0, 0]) - 2.5)
      < 1e-12, "the half is of C(0), not of 1: with C(0) = 2, 1.0 lies "
      "halfway between C = 1.2 at r = 2 and 0.8 at r = 3 -> 2.5")
sigma = 10.0
rr = np.arange(0.0, 64.0)
got = P.length_from_half(rr, np.exp(-rr ** 2 / (2 * sigma ** 2)))
check(abs(got - sigma * np.sqrt(2 * np.log(2))) < 0.02,
      f"a Gaussian C(r) of width 10 gives 10 sqrt(2 ln 2) = 11.77 ({got:.3f})")
check(np.isnan(P.length_from_half(r[:4], [1, .9, .8, .7]))
      and np.isnan(P.length_from_half(r[:4], [0, 0, 0, 0])),
      "a C(r) that never falls to half, or has C(0) = 0, has no L_half")
g = np.random.default_rng(3)
noise = np.fft.fft2(g.standard_normal((64, 64)))
k2 = np.fft.fftfreq(64)[:, None] ** 2 + np.fft.fftfreq(64)[None, :] ** 2
field = np.real(np.fft.ifft2(noise * np.exp(-k2 / (2 * 0.04 ** 2))))
lens = P.all_lengths(field)
check("L_half" in lens and np.isfinite(lens["L_half"])
      and 0 < lens["L_half"] < lens["L_zero"],
      f"all_lengths has L_half, below L_zero for a smooth field "
      f"({lens['L_half']:.2f} < {lens['L_zero']:.2f}); "
      f"LENGTH_KEY stays {P.LENGTH_KEY}")

print("\n2. the growth exponent on each side of the training horizon")
t = np.arange(110.0, 2001.0, 10.0)
h = 500.0
mt = np.where(t <= h, t ** (1 / 3), h ** (1 / 3) * (t / h) ** 0.25)
mp = np.where(t <= h, t ** 0.4, h ** 0.4 * (t / h) ** 0.2)
w = P.growth_windows(t, mt, mp, h)
tr, ex = w["exponent_train"], w["exponent_extrap"]
check(abs(tr["n_truth"] - 1 / 3) < 1e-9 and abs(ex["n_truth"] - 0.25) < 1e-9,
      f"truth: t^(1/3) up to the horizon, t^(1/4) after "
      f"({tr['n_truth']:.4f}, {ex['n_truth']:.4f})")
check(abs(tr["n_fno"] - 0.4) < 1e-9 and abs(ex["n_fno"] - 0.2) < 1e-9,
      "and the FNO's own, not the truth's")
check(tr["window"] == [110.0, 500.0] and ex["window"] == [500.0, 2000.0],
      f"the windows are [start, horizon] and [horizon, end]: "
      f"{tr['window']}, {ex['window']}")
whole = P.fit_growth_exponent(t, mt)
check(w["exponent"]["n_truth"] == whole and w["exponent"]["window"]
      == [110.0, 2000.0] and 0.25 < whole < 1 / 3,
      f"the single-window exponent is kept ({whole:.4f})")
check(all(P.growth_windows(t, mt, mp, x)[k] is None
          for x in (None, 100.0, 110.0, 2000.0, 5000.0)
          for k in ("exponent_train", "exponent_extrap")),
      "no horizon, or one at or outside the rollout: no two windows")
check(abs(P.fit_growth_exponent(t, mt, 500.0, 2000.0) - 0.25) < 1e-9
      and np.isnan(P.fit_growth_exponent(t, mt, 500.0, 510.0)),
      "a window with fewer than three points has no exponent")

print("\n3. the finite-size onset")
lat, ramp = 64, 1.0 + t / 200.0
check(P.finite_size_onset(t, ramp, lat) == 1410.0,
      "L = 1 + t/200 passes l/8 = 8 after t = 1400: first time 1410 (at "
      "1400 L equals the limit, which is not past it)")
check(P.finite_size_onset(t, ramp, 128) is None
      and P.finite_size_onset(t, np.full(t.size, np.nan), lat) is None,
      "never past the limit (or no data): no onset")
check(P.finite_size_onset(t, np.r_[np.full(5, np.nan), ramp[5:]], lat)
      == 1410.0, "a NaN stretch is not past the limit either")

print("\n4. which horizon the figures know")
KNOWN = os.path.join(BASE, "models", "train_t0500", "fno_ch.pt")
LOOSE = os.path.join(BASE, "models", "fno_ch.pt")
P.RECORDED_HORIZON = P._RUN_HORIZON = None
P.MODEL_PATH = KNOWN
check(P.train_horizon() == 500.0, "a train label names it: 500")
P.MODEL_PATH = LOOSE
check(P.train_horizon() is None, "a loose checkpoint file does not")
check(P._ckpt_horizon({"run_meta": {"t_train": 300}}) == 300.0
      and P._ckpt_horizon({"run_meta": {"t_train": None}}) is None
      and P._ckpt_horizon({}) is None and P._ckpt_horizon(None) is None,
      "the checkpoint's run_meta gives it, when it has one")
P._RUN_HORIZON = 300.0
check(P.train_horizon() == 300.0, "and is used when the label cannot")
P.MODEL_PATH = KNOWN
check(P.train_horizon() == 500.0, "the label comes first")
P._RUN_HORIZON, P.RECORDED_HORIZON = None, 700.0
P.MODEL_PATH = ""
check(P.train_horizon() == 700.0, "a redraw from records gives its own")
P.RECORDED_HORIZON = None

print("\n5. the figures, on three synthetic seeds")
P.WRITE_LOG = False
P.START_TIME, P.END_TIME = 100.0, 2000.0
P.SNAPSHOT_TIMES = (100.0, 500.0, 1000.0, 2000.0)
P.LATTICE = lat
P.LOG_TIME = "off"
P.MODEL_FLAGS.update(hard_constraints=True, hard_mass_conservation=True,
                     zero_dc=True, label="fno_ch", note="")
SEEDS = (11, 22, 33)
SCALE = (0.75, 0.875, 1.375)
TIMES = np.arange(101.0, 2001.0)


def fake_result(i, half=True):
    """Seed i: lengths that are SCALE[i] times the ramp, so the seed mean is
    the ramp itself and no seed is."""
    g = np.random.default_rng(100 + i)
    snaps = {}
    for tt in P.SNAPSHOT_TIMES:
        f = np.tanh(3 * g.standard_normal((24, 24)))
        snaps[tt] = (f, np.clip(f + 0.05 * g.standard_normal(f.shape),
                                -1.05, 1.05))
    s = SCALE[i]
    out = {
        "times": TIMES, "r2": np.linspace(0.99, 0.90 - 0.02 * i, TIMES.size),
        "mass_drift": np.full(TIMES.size, 1e-9),
        "mean_true": np.zeros(TIMES.size),
        "mean_pred": np.full(TIMES.size, 1e-9),
        "diag_times": t, "energy_true": np.linspace(1, 0.5, t.size) * s,
        "energy_pred": np.linspace(1, 0.55, t.size) * s ** 2,
        "t_stop": 2000.0, "stopped_early": False, "diverged_at": None,
        "L_true": {k: ramp * s for k in ("L_zero", "L_peak", "L_moment",
                                         "L_grad")},
        "L_pred": {k: ramp * s * 1.05 for k in ("L_zero", "L_peak",
                                                "L_moment", "L_grad")},
        "snapshots": snaps, "gt_seconds": 1.0, "gt_cached": False,
        "gt_steps": 1, "fno_seconds": 0.5, "fno_steps": 1900,
        "video_t": np.asarray([]), "video_true": None, "video_pred": None}
    if half:
        out["L_true"]["L_half"] = 0.5 * ramp * s
        out["L_pred"]["L_half"] = 0.6 * ramp * s
    return out


results = [fake_result(i) for i in range(3)]


NOTES: dict = {}


def draw(tag, res=results, seeds=SEEDS, lattice=lat):
    """The four case figures into BASE/tag/, the paths by name."""
    d = os.path.join(BASE, tag)
    os.makedirs(d, exist_ok=True)
    paths = {k: os.path.join(d, f"{k}.png")
             for k in ("r2", "kinetics", "conservation", "growth")}
    P.figure_r2_and_snapshots(res, seeds, paths["r2"], 0.0, lattice)
    P.figure_kinetics(res, seeds, paths["kinetics"], 0.0, lattice,
                      notes=NOTES.setdefault(tag, {}))
    P.figure_conservation(res, seeds, paths["conservation"], 0.0, lattice)
    S.figure_growth(res, 0.0, paths["growth"], lattice,
                    notes=NOTES.setdefault(tag, {}))
    return paths


AXES = {"r2": 1, "kinetics": 3, "conservation": 2, "growth": 1}
P.MODEL_PATH = KNOWN
known = draw("known")
for name, n in AXES.items():
    lines, spans, ext = marked(known[name], 500.0)
    check(lines == n, f"{name}: the dashed horizon line is on all {n} time "
                      f"axes ({lines})")
    check(spans == n and all(e == (1410.0, 2000.0) for e in ext),
          f"{name}: and the finite-size regime is shaded on them, from the "
          f"onset 1410 to the last time ({ext[:1]})")
legends = [ax["legend"] for ax in seen(known["growth"])["axes"]]
check("training horizon" in legends[0]
      and r"$L_{\rm zero} > l/8$" in legends[0]
      and legends[0][:2] == ["FNO", "Solver"],
      f"both are in the legend, which is short: {legends[0]}")
gn = NOTES["known"]["growth.png"]
check(any(x.startswith("training window, t = 110-500: n = ") for x in gn),
      f"the fit on the training window is in the notes: {gn[-4:]}")
check(any(x.startswith("late exponent, t = 500-2000") for x in gn)
      and "mean +- 1 sd over 3 seeds" in gn[0]
      and seen(known["growth"])["title"].strip() == "",
      "and so is the late-window exponent; the figure has no title")

P.MODEL_PATH = LOOSE
unknown = draw("unknown")
for name in AXES:
    lines, spans, _ = marked(unknown[name], 500.0)
    check(lines == 0 and spans == AXES[name],
          f"{name}: horizon unknown -> no horizon line, shading stays "
          f"({lines} lines)")
check(not any("training" in leg for leg in seen(unknown["growth"])["axes"][0]
              ["legend"]), "and the growth legend names no training window")
P.MODEL_PATH = KNOWN
short = draw("short", lattice=128)
check(all(marked(short[n], 500.0)[1] == 0 for n in AXES),
      "l = 128: L never gets past l/8 = 16, so nothing is shaded")
for label, name in (("late", "train_t5000"), ("early", "train_t0100")):
    P.MODEL_PATH = os.path.join(BASE, "models", name, "fno_ch.pt")
    off_axis = draw(label)
    check(all(marked(off_axis[n], P.train_horizon())[0] == 0 for n in AXES),
          f"a horizon of {P.train_horizon():g} is off the rollout's axis: "
          f"no line")
P.MODEL_PATH = KNOWN

print("\n6. the averaged kinetics figure is the mean of the seeds")
kin = known["kinetics"]
x, y = line(kin, 0, "Solver")
check(np.allclose(y, np.mean([r["L_true"]["L_zero"] for r in results], 0),
                  rtol=1e-12, atol=0)
      and np.allclose(y, ramp, rtol=1e-12, atol=0)
      and not any(np.allclose(y, r["L_true"]["L_zero"]) for r in results),
      "panel (a): truth L(t) is the seed mean, and no seed's own")
x, y = line(kin, 0, "FNO")
check(np.allclose(y, np.mean([r["L_pred"]["L_zero"] for r in results], 0),
                  rtol=1e-12, atol=0), "panel (a): and so is the FNO's")
x, y = line(kin, 4, "Solver")
check(np.allclose(y, np.mean([r["energy_true"] for r in results], 0),
                  rtol=1e-12, atol=0), "panel (e): free energy, the mean")
ax_a = seen(kin)["axes"][0]
sd_band = 1.0 * np.std(SCALE, ddof=1)
check(len(ax_a["fills"]) == 2
      and all(np.isclose(f[:, 1].max(), (ramp * (1 + sd_band)).max(),
                         rtol=1e-9)
              and np.isclose(f[:, 1].min(), (ramp * (1 - sd_band)).min(),
                             rtol=1e-9) for f in ax_a["fills"][:1]),
      "panel (a): a +-1 sd band over the seeds around the truth's L(t) "
      "(and one around the FNO's)")
x, y = line(kin, 5, "Solver")
check(np.allclose(y, 0.0, atol=1e-12)
      and seen(kin)["axes"][5]["secondary"] == 1,
      "panel (f): <psi>(t) of the truth, with the c_A axis on the right")
x, y = line(kin, 5, "FNO")
check(np.allclose(y, 1e-9, rtol=1e-12, atol=0),
      "panel (f): and of the FNO, as seed means")
check(not any(n == "L_half (truth)" for ax in seen(kin)["axes"]
              for n, *_ in ax["lines"])
      and "composition at t = 2000" in " ".join(NOTES["known"][
          "kinetics.png"]),
      "panel (f) is not the length estimators any more; its notes give "
      "the composition at the end")
sk = [P.structure_factor(r["snapshots"][500.0][0]) for r in results]
x, y = line(kin, 1, "$t$ = 500")
want = np.mean([s for _, s in sk], 0)[sk[0][0] > 0]
check(np.allclose(y, want, rtol=1e-12, atol=0)
      and not any(np.allclose(y, s[sk[0][0] > 0]) for _, s in sk),
      "panel (b): S(k) at t = 500 is the mean of the seeds' spectra")
title = seen(kin)["title"]
check("mean over 3 seeds" in NOTES["known"]["kinetics.png"][0]
      and "seed 11" not in NOTES["known"]["kinetics.png"][0],
      "and the notes say it is a mean over 3 seeds")
one = os.path.join(BASE, "one", "kin.png")
os.makedirs(os.path.dirname(one))
ONE_N: dict = {}
P.figure_kinetics(results[1:2], SEEDS[1:2], one, 0.0, lat, notes=ONE_N)
x, y = line(one, 0, "Solver")
check(np.array_equal(y, results[1]["L_true"]["L_zero"])
      and "seed 22" in ONE_N["kin.png"][0]
      and seen(one)["axes"][0]["fills"] == [],
      "one result gives that seed's own figure, titled with its seed, with "
      "no sd band")
sk1 = P.structure_factor(results[1]["snapshots"][500.0][0])
x, y = line(one, 1, "$t$ = 500")
check(np.array_equal(y, sk1[1][sk1[0] > 0]), "with its own S(k)")

print("\n7. a case: its kinetics averaged (flat), the best seed's in extras/")
OUT = os.path.join(BASE, "case")
P.RESULTS_DIR = OUT
P.WRITE_SUBMISSION = False
P.PER_SEED_OUTPUT = P.PER_SEED_FIGURES = True
P.WRITE_VIDEO = False
P.MODEL_PATH = KNOWN
case = P.write_case_outputs(results, SEEDS, 0.0, None)
folder = P.case_dir(0.0, None)
best = case["stats"]["best_seed"]
sub = os.path.join(folder, f"best_seed{best}")
own = os.path.join(sub, f"fno_ch_seed{best}_kinetics.png")
check(folder == os.path.join(P.RESULT_CACHE, "extras",
                             P.case_name(0.0, None))
      and best == 11 and os.path.isfile(own)
      and not os.path.exists(os.path.join(folder, "fno_ch_kinetics.png")),
      f"the best seed's kinetics is in extras/<case>/best_seed{best}/; the "
      f"case's own is drawn by make_submission, flat")
entry_m, _ = S.write_case(results, SEEDS, 0.0, lat, OUT, videos=False,
                          extras=folder, token="psi0p0.00")
mean_png = os.path.join(OUT, f"l{lat}",
                        "phase_ordering_kinetics_psi0p0.00.png")
x, y = line(mean_png, 0, "Solver")
check(np.allclose(y, ramp, rtol=1e-12, atol=0)
      and "mean over 3 seeds" in entry_m["notes"][
          "phase_ordering_kinetics_psi0p0.00.png"][0],
      "phase_ordering_kinetics_psi0p0.00.png is the mean over the seeds")
x, y = line(own, 0, "Solver")
check(np.allclose(y, ramp * SCALE[0], rtol=1e-12, atol=0)
      and seen(own)["axes"][0]["fills"] == [],
      f"the best seed's is seed {best}'s own, with no sd band")
check(sorted(os.listdir(sub)) == sorted(
    [f"fno_ch_seed{best}_{n}" for n in (
        "r2.csv", "summary.json", "r2_and_snapshots.png",
        "kinetics.png", "conservation.png")]),
      f"the best-seed folder: {sorted(os.listdir(sub))}")
with open(os.path.join(folder, "fno_ch_summary.json")) as fh:
    summ = json.load(fh)
ml = summ["mean_length"]
check(ml["length"] == "L_zero" and ml["train_horizon"] == 500.0
      and ml["finite_size"] == {"factor": 8.0, "limit": 8.0, "onset": 1410.0},
      f"summary: the training horizon and the finite-size onset "
      f"({ml['finite_size']})")
check(ml["exponent"]["window"] == [110.0, 2000.0]
      and ml["exponent_train"]["window"] == [110.0, 500.0]
      and ml["exponent_extrap"]["window"] == [500.0, 2000.0]
      and ml["exponent_train"]["n_truth"] > 0
      and ml["exponent_extrap"]["n_fno"] > 0,
      "and the exponent over the whole rollout and on each side of the "
      "horizon, with their windows")
check(np.isclose(ml["L_half"]["L_end"], 0.5 * ramp[-1], rtol=1e-12)
      and np.isclose(ml["L_half"]["L_end_fno"], 0.6 * ramp[-1], rtol=1e-12)
      and ml["L_half"]["window"] == [110.0, 2000.0],
      "and L_half's end values and exponents")

P.MODEL_PATH = LOOSE
P.RESULTS_DIR = os.path.join(BASE, "case_unknown")
P.RESULT_CACHE = os.path.join(BASE, "cache_unknown")
P.write_case_outputs(results, SEEDS, 0.0, None)
with open(os.path.join(P.case_dir(0.0, None), "fno_ch_summary.json")) as fh:
    ml = json.load(fh)["mean_length"]
check(ml["train_horizon"] is None and ml["exponent_train"] is None
      and ml["exponent_extrap"] is None
      and ml["exponent"]["window"] == [110.0, 2000.0],
      "with no horizon known, the summary has the single exponent only")
P.MODEL_PATH = KNOWN

print("\n8. domain_growth.csv and the manifest entry")
out = os.path.join(BASE, "sub")
xout = os.path.join(BASE, "sub_extras")
entry, _ = S.write_case(results, SEEDS, 0.0, lat, out, videos=False,
                        extras=xout, token="psi0p0.00")
check(os.path.isfile(os.path.join(xout, "domain_growth.csv"))
      and not os.path.exists(os.path.join(out, "domain_growth.csv")),
      "domain_growth.csv is in extras/, not in the run folder")
growth_csv = np.loadtxt(os.path.join(xout, "domain_growth.csv"),
                        delimiter=",", skiprows=1)
with open(os.path.join(xout, "domain_growth.csv")) as fh:
    head = fh.readline().strip().split(",")
col = {n: growth_csv[:, i] for i, n in enumerate(head)}
check(head[:5] == ["t", "L_true_mean", "L_pred_mean", "n_seeds_true",
                   "n_seeds_pred"]
      and head[5:] == ["L_half_true_mean", "L_half_pred_mean",
                       "train_horizon", "n_train_true", "n_train_pred",
                       "n_extrap_true", "n_extrap_pred"],
      "the old columns first, then L_half and the two windows' exponents")
check(np.allclose(col["L_half_true_mean"], 0.5 * ramp, rtol=1e-12, atol=0)
      and np.allclose(col["L_half_pred_mean"], 0.6 * ramp, rtol=1e-12,
                      atol=0)
      and np.allclose(col["L_true_mean"], ramp, rtol=1e-12, atol=0),
      "L_half is in the CSV beside L_zero")
w = P.growth_windows(t, ramp, ramp * 1.05, 500.0)
check(np.all(col["train_horizon"] == 500.0)
      and np.allclose(col["n_train_true"], w["exponent_train"]["n_truth"])
      and np.allclose(col["n_extrap_pred"], w["exponent_extrap"]["n_fno"])
      and col["n_train_true"][0] != col["n_extrap_true"][0],
      "the horizon and both windows' exponents, truth and FNO")
g = entry["growth"]
check(g["finite_size"]["onset"] == 1410.0 and g["train_horizon"] == 500.0
      and g["exponent_extrap"]["window"] == [500.0, 2000.0]
      and np.isclose(g["L_half"]["L_end"], 0.5 * ramp[-1], rtol=1e-12)
      and "n_truth" in g and g["fit_window"] == [110.0, 2000.0],
      "the manifest entry has them too, and keeps its old keys")

print("\n9. rollouts that carry no L_half (an older cache) still draw")
old = [fake_result(i, half=False) for i in range(3)]
P.MODEL_PATH = KNOWN
d_old = draw("old", old)
check(not any("1/2" in n for ax in seen(d_old["growth"])["axes"]
              for n, *_ in ax["lines"])
      and any(x.startswith("beyond training, t = 500-2000: n = ")
              for x in NOTES["old"]["growth.png"]),
      "no L_half line in the growth figure, which then keeps the zero "
      "crossing as its main curve and both fit windows in the legend")
out = os.path.join(BASE, "sub_old")
xout = os.path.join(BASE, "sub_old_extras")
entry, _ = S.write_case(old, SEEDS, 0.0, lat, out, videos=False,
                        extras=xout, token="psi0p0.00")
cs = np.loadtxt(os.path.join(xout, "domain_growth.csv"), delimiter=",",
                skiprows=1)
check(np.all(np.isnan(cs[:, 5:7])) and entry["growth"]["L_half"] is None,
      "the CSV's L_half columns are NaN and the manifest says None")
check(all(np.array_equal(a, b, equal_nan=True)
          for a, b in zip(cs[:, :5].T, growth_csv[:, :5].T)),
      "and the old columns are the same")

print("\nFIGURES PASS" if ok else "\nFIGURES FAIL")
sys.exit(0 if ok else 1)
