"""The report additions, on rollouts whose answers are known.

The best, median and worst seed; the composition tracking (the flat layout's
composition.csv, the case's dense one in extras/, composition_all.png and the
A / B table of report.txt); the long-horizon snapshots; the P(psi) range and
its lever-rule numbers; and the results overview with its comparison figure.
Every rollout is synthetic (evaluate_seed is stubbed), the checkpoints are
small files of this suite's own, and nothing is written outside _scratch.
"""
import contextlib
import csv
import io as _io
import json
import os
import re
import shutil
import time
import types
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.figure as mfig
import numpy as np

from _paths import ROOT, SCRATCH, project_state

os.chdir(ROOT)
PROJECT = project_state()
import fno_paths
import fno_predict as P
import fno_runlog

S = P._submission()
ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


BASE = os.path.join(SCRATCH, "report")
shutil.rmtree(BASE, ignore_errors=True)
os.makedirs(BASE)
CKPT500 = os.path.join(BASE, "models", "train_t0500", "fno_ch.pt")
CKPT300 = os.path.join(BASE, "models", "train_t0300", "fno_ch.pt")
for _path, _text in ((CKPT500, b"a checkpoint of this suite"),
                     (CKPT300, b"another checkpoint of this suite")):
    os.makedirs(os.path.dirname(_path))
    with open(_path, "wb") as fh:
        fh.write(_text)
FP500, FP300 = fno_paths.fingerprint(CKPT500), fno_paths.fingerprint(CKPT300)

# What a saved figure held, in the order saved: its texts, titles, the lines
# of each axes (label, x, y, line style, colour, marker), the legend of each
# axes, and how many images it drew.
SEEN = []
_savefig = mfig.Figure.savefig


def _remember(self, fname, *args, **kwargs):
    if isinstance(fname, str):
        SEEN.append((os.path.normcase(os.path.abspath(fname)), {
            "texts": [t.get_text() for t in self.texts]
            + [t.get_text() for ax in self.axes for t in ax.texts],
            "titles": [ax.get_title() or ax.get_title("left")
                       for ax in self.axes],
            "images": sum(len(ax.images) for ax in self.axes),
            "axes": [{"lines": [(ln.get_label(),
                                 np.array(ln.get_xdata(), float),
                                 np.array(ln.get_ydata(), float),
                                 ln.get_linestyle(), ln.get_color(),
                                 ln.get_marker())
                                for ln in ax.lines],
                      "xscale": ax.get_xscale(), "ylim": ax.get_ylim(),
                      "yscale": ax.get_yscale(),
                      "legend": [t.get_text() for t in
                                 (ax.get_legend().get_texts()
                                  if ax.get_legend() else [])]}
                     for ax in self.axes]}))
    return _savefig(self, fname, *args, **kwargs)


setattr(mfig.Figure, "savefig", _remember)


EMPTY: Any = {"missing": True, "texts": [], "titles": [], "images": 0,
               "axes": [{"lines": [], "xscale": "", "ylim": (0.0, 0.0),
                         "yscale": "", "legend": []}]}


def seen(name, where=""):
    """The last figure saved under this name (a temporary ".partial" name,
    with or without the host-pid of the index's own, counts as the real one)
    below the folder ``where``; EMPTY if none."""
    key = os.path.normcase(os.path.abspath(where)) if where else ""
    hits = [i for p, i in SEEN
            if re.sub(r"\.\w+-\d+\.partial", ".partial",
                      os.path.basename(p)).replace(".partial", "") == name
            and key in p]
    return hits[-1] if hits else EMPTY


def text_of(info):
    return "\n".join(info["texts"] + info["titles"])


KEYS = ("EVAL_MODE", "SIZE_SWEEP", "EVAL_SEEDS", "SWEEP_SEEDS", "WRITE_VIDEO",
        "RESULTS_DIR", "RESULTS_TEMPLATE", "VIEWS", "WRITE_TIMING",
        "PER_SEED_FIGURES", "PER_SEED_OUTPUT", "WRITE_SUBMISSION", "WRITE_LOG",
        "START_TIME", "END_TIME", "SNAPSHOT_TIMES", "MODEL_PATH", "REDO",
        "KEEP_CACHE", "load_checkpoint", "evaluate_seed", "RESULT_CACHE",
        "GT_CACHE", "CACHE_RESULTS", "RECORDS", "ROOT", "NO_SUPERSEDE",
        "SEED_SET", "LATTICE", "LABEL", "AUTO_LABEL", "LOG_TIME",
        "SNAPSHOTS_EXPLICIT", "RECORDED_FINGERPRINT", "CACHE_END_TIME",
        "_RUN_SNAPSHOTS", "DIAG_EVERY", "VIDEO_EVERY", "VIDEO_SCHEDULE",
        "evaluate_seeds", "EVAL_BATCH", "VIDEO_DTYPE", "ENGINE",
        "DIAGNOSTICS", "figure_kinetics", "figure_conservation")
SAVED = {k: getattr(P, k) for k in KEYS}
SAVED.update(WRITE_LOG=False, MODEL_PATH=CKPT500, GT_CACHE="",
             RESULTS_DIR=os.path.join(BASE, "results_default"),
             RESULT_CACHE=os.path.join(BASE, "cache_default"),
             ROOT=os.path.join(BASE, "root"))
LOG_DIR = fno_runlog.LOG_DIR
WORK = fno_paths.WORK
RESULTS = fno_paths.RESULTS
FLAGS = dict(hard_constraints=True, hard_mass_conservation=True,
             zero_dc=True, label="fno_ch", note="")


class FakeModel:
    config = {"backbone": "physical_k", "hard_mass_conservation": True,
              "residual_prediction": True, "mode_scaling": 1.0,
              "zero_dc": True}

    def n_parameters(self):
        return 1234

    def parameters(self):
        yield types.SimpleNamespace(dtype="float32")


LAT = 24
DRIFT = [0.0]


def rollout(seed, off=0.0, level=None, drift=0.0, die=None, means=True,
            peaked=False, blip=False):
    """A synthetic rollout over the present window.  ``level`` makes R2
    constant, ``drift`` makes the FNO's mean psi leave the starting one at
    that rate, ``die`` ends the FNO there (NaN after, R2 at the floor),
    ``means=False`` is a rollout cached before the means were kept, and
    ``peaked`` has 95 % of the sites at +0.99 and the rest at -0.99 (one
    site of the FNO at 0 with ``blip``)."""
    g = np.random.default_rng([int(seed), int(round(off * 10)) + 10, LAT])
    t = np.arange(P.START_TIME + 1, P.END_TIME + 1)
    diag = t[::max(1, t.size // 19)]
    r2 = (np.full(t.size, float(level)) if level is not None else
          np.clip(1.0 - (t - t[0]) / 9000.0 - 0.01 * g.random(), -1, 1))
    mp = off + drift * (t - t[0])
    if die is not None:
        gone = t > die
        r2 = np.where(gone, P.R2_FLOOR, r2)
        mp = np.where(gone, np.nan, mp)
        diag = diag[diag <= die]
    snaps = {}
    for tt in P.SNAPSHOT_TIMES:
        if peaked:
            f = np.where(g.random((LAT, LAT)) < 0.95, 0.99, -0.99)
            q = f.copy()
            if blip:
                q[0, 0] = 0.0
        else:
            f = np.tanh(3 * g.standard_normal((LAT, LAT))) + off
            q = np.clip(f + 0.05 * g.standard_normal(f.shape), -1.05, 1.05)
        if tt == P.START_TIME:
            q = f.copy()
        if die is not None and tt > die:
            q = np.full_like(f, np.nan)
        snaps[tt] = (f, q)
    L = np.linspace(5, 14, diag.size)
    return {
        "times": t, "r2": r2, "mass_drift": np.full(t.size, 1e-9),
        "mean_true": np.full(t.size, off) if means else None,
        "mean_pred": mp if means else None,
        "diag_times": diag,
        "energy_true": np.linspace(1, 0.5, diag.size),
        "energy_pred": np.linspace(1, 0.5, diag.size),
        "t_stop": float(t[-1]), "stopped_early": False,
        "diverged_at": None if die is None else float(die),
        "L_true": {k: L for k in ("L_zero", "L_peak", "L_moment",
                                  "L_grad", "L_half")},
        "L_pred": {k: L * 1.01 for k in ("L_zero", "L_peak", "L_moment",
                                         "L_grad", "L_half")},
        "snapshots": snaps,
        "gt_seconds": 1.0, "gt_cached": False, "gt_steps": 1,
        "fno_seconds": 0.5, "fno_steps": 1900,
        "video_t": np.asarray([]), "video_true": None, "video_pred": None}


def fake_evaluate_seed(model, seed, device, off=0.0, lattice=None):
    return rollout(seed, off, drift=DRIFT[0])


def fake_evaluate_seeds(model, seeds, device, off=0.0, lattice=None):
    return [rollout(s, off, drift=DRIFT[0]) for s in seeds]


def no_rollout(*a, **k):
    raise AssertionError("a rollout was asked for")


def nothing(*a, **k):
    return None


def setup(big=False):
    """The module's settings back, and the three big case figures (not under
    test here; t_records draws them) left out unless ``big``."""
    for k, v in SAVED.items():
        setattr(P, k, v)
    fno_runlog.LOG_DIR = LOG_DIR
    fno_paths.WORK = WORK
    fno_paths.RESULTS = RESULTS
    P.MODEL_FLAGS.update(FLAGS)
    P.load_checkpoint = lambda path, device=None: (FakeModel(), {
        "best": {"epoch": 7, "score": 0.91, "survival": 2000.0}})
    P.evaluate_seed = fake_evaluate_seed
    P.evaluate_seeds = fake_evaluate_seeds
    P.WRITE_TIMING = False
    P.PER_SEED_FIGURES = False
    P.LATTICE = LAT
    if not big:
        P.figure_kinetics = P.figure_conservation = nothing


def window(end, snaps):
    P.START_TIME, P.END_TIME = 100.0, float(end)
    P.SNAPSHOT_TIMES = tuple(float(s) for s in snaps)


def quiet(fn, *a, **k) -> tuple[Any, str]:
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            out = fn(*a, **k)
        except SystemExit as exc:
            out = exc.code
    return out, buf.getvalue()


def run_main(argv):
    return quiet(P.main, argv)


def flags(results, cache, model=CKPT500, eval_mode="all", seeds="3"):
    return ["--model", model, "--results-dir", results, "--result-cache",
            cache, "--eval-mode", eval_mode, "--sizes", str(LAT), "--seeds",
            seeds, "--no-gt-cache", "--no-diagnostics", "--no-video"]


def redraw(cache, out):
    real = S.fno_runlog.start_log
    S.fno_runlog.start_log = lambda *a, **k: None
    try:
        return quiet(S.main, ["--records", cache, "--out", out])
    finally:
        S.fno_runlog.start_log = real


def files_under(d):
    return sorted(os.path.relpath(os.path.join(b, f), d).replace(os.sep, "/")
                  for b, _, fs in os.walk(d) for f in fs)


def read(p):
    with open(p, "rb") as fh:
        return fh.read()


def text(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def table(path):
    """A CSV's columns by header name: numbers as arrays, the rest as lists
    of text."""
    with open(path, encoding="utf-8", newline="") as fh:
        head, *rows = list(csv.reader(fh))
    out = {}
    for i, name in enumerate(head):
        col = [row[i] for row in rows]
        try:
            out[name] = np.array(col, dtype=float)
        except ValueError:
            out[name] = col
    return out


def comp_rows(md):
    """The rows of the A / B table of a README (not the table of cases,
    whose rows begin alike)."""
    return [r for r in rows_of(md, "| psi_0 =") if " / " in r[1]]


def rows_of(md, start):
    """The cells of the table rows of a README that start with ``start``."""
    return [[c.strip() for c in ln.strip().strip("|").split("|")]
            for ln in md.splitlines() if ln.startswith(start)]


print("1. the best, median and worst seed")
setup()
window(2000, (100, 500, 1000, 2000))
SEEDS5 = (11, 22, 33, 44, 55)
LEVELS = {11: 0.5, 22: 0.9, 33: 0.7, 44: 0.3, 55: 0.99}
res5 = [rollout(s, 0.2, level=LEVELS[s], drift=1.2e-6,
                die=1990 if s == 55 else None) for s in SEEDS5]
mean5 = {s: float(np.mean(r["r2"])) for s, r in zip(SEEDS5, res5)}
check(max(mean5, key=lambda s: mean5[s]) == 55 and mean5[55] > 0.95,
      "the fixture: seed 55 has the highest mean R2 but diverged at 1990")
ranks = S.seed_ranks(res5, SEEDS5)
check([ranks[k]["seed"] for k in ("best", "median", "worst")]
      == [22, 33, 44],
      f"best is the best healthy seed (the diverged one with the top mean "
      f"cannot be), median the middle by mean R2, worst the lowest: "
      f"{[ranks[k]['seed'] for k in ('best', 'median', 'worst')]}")
check(abs(ranks["median"]["mean_r2"] - mean5[33]) < 1e-12
      and abs(ranks["best"]["mean_r2"] - mean5[22]) < 1e-12
      and ranks["best"]["r2_at_end"] is not None
      and abs(ranks["best"]["r2_at_end"]
              - float(P.r2_score(*res5[1]["snapshots"][2000.0]))) < 1e-12,
      "each comes with its mean R2 and its R2 at the final time")
res4 = [rollout(s, 0.0, level=v) for s, v in zip((1, 2, 3, 4),
                                                 (0.2, 0.9, 0.5, 0.7))]
r4 = S.seed_ranks(res4, (1, 2, 3, 4))
check(r4["median"]["seed"] == 3 and r4["best"]["seed"] == 2
      and r4["worst"]["seed"] == 1,
      "with four seeds the median is the lower of the middle two (0.5)")
res_tie = [rollout(s, 0.0, level=v) for s, v in zip((7, 8, 9),
                                                    (0.4, 0.4, 0.9))]
r3 = S.seed_ranks(res_tie, (7, 8, 9))
check(r3["worst"]["seed"] == 7 and r3["median"]["seed"] == 8,
      "ties keep the order of the seeds")
one = S.seed_ranks(res4[:1], (1,))
check({one[k]["seed"] for k in one} == {1},
      "a single seed is the best, median and worst")

CASE_ROOT = os.path.join(BASE, "case_root")       # a run folder, flat
XTRAS = os.path.join(BASE, "case_extras")         # the case's extras folder
TOK = "psi0p0.20"
(entry, curve), _ = quiet(S.write_case, res5, SEEDS5, 0.2, LAT, CASE_ROOT,
                          key="k", videos=False, root=CASE_ROOT,
                          extras=XTRAS, token=TOK)
snap = seen(f"snapshots_{TOK}.png", CASE_ROOT)
SNOTE = " ".join(entry["notes"][f"snapshots_{TOK}.png"])
check(not snap.get("missing") and "best seed 22 (0.900)" in SNOTE
      and "median seed 33 (0.700)" in SNOTE
      and "worst seed 44 (0.300)" in SNOTE
      and text_of(snap).count("seed") == 0,
      "the notes of snapshots_psi0p0.20.png name the three seeds with their "
      "mean R2; the figure carries no caption")
check(not snap.get("missing") and "median " in SNOTE
      and "the best of 5" in SNOTE,
      "and the notes keep what the caption said (R2 at the end, median and "
      "worst)")
ent = entry["seed_ranks"]
check([ent[k]["seed"] for k in ("best", "median", "worst")] == [22, 33, 44]
      and entry["snapshots"]["seed"] == 22,
      "the manifest entry holds the three, and the shown seed is the best")
quiet(P.figure_r2_and_snapshots, res5, SEEDS5,
      os.path.join(CASE_ROOT, "r2.png"), 0.2, LAT)
r2fig = seen("r2.png", CASE_ROOT)
check(not r2fig.get("missing") and "median seed 33 (0.7000)" in text_of(r2fig)
      and "worst seed 44 (0.3000)" in text_of(r2fig)
      and "seed 22 (best healthy of 5" in text_of(r2fig),
      "the R2 figure's caption names them too, after the shown seed")
quiet(P.figure_r2_and_snapshots, res5[1:2], (22,),
      os.path.join(CASE_ROOT, "r2solo.png"), 0.2, LAT)
check("median seed" not in text_of(seen("r2solo.png", CASE_ROOT)),
      "a one-seed figure names no median or worst")

print("\n2. composition tracking")
comp = table(os.path.join(XTRAS, "composition.csv"))
t = comp["t"]
check(list(comp) == ["t", "psi_mean_truth", "psi_mean_pred", "A_pct_truth",
                     "B_pct_truth", "A_pct_pred", "B_pct_pred", "drift_pred",
                     "n_seeds"],
      f"the case's dense composition.csv (in extras/) columns: {list(comp)}")
check(t[0] == 101.0 and t[-1] == 2000.0 and np.all(np.diff(t) > 0)
      and 100 < t.size <= S.COMPOSITION_POINTS + 6
      and all(x in t for x in (500.0, 1000.0, 2000.0)),
      f"thinned to {t.size} increasing times, from the first step to the "
      f"last, with every snapshot time")
check(np.allclose(comp["A_pct_truth"], 60.0, atol=1e-9)
      and np.allclose(comp["A_pct_truth"] + comp["B_pct_truth"], 100.0)
      and np.allclose(comp["A_pct_pred"] + comp["B_pct_pred"], 100.0),
      "psi0 = +0.2 is 60 % A and 40 % B, and A + B = 100 throughout")
want = 0.2 + 1.2e-6 * (t - 101.0)
check(np.allclose(comp["psi_mean_pred"], want, atol=1e-11)
      and np.allclose(comp["drift_pred"], want - 0.2, atol=1e-11)
      and np.allclose(comp["A_pct_pred"], 50.0 * (1 + want), atol=1e-8),
      "the FNO's seed-mean psi follows the known drift of 1.2e-6 per t, "
      "and its A % is c_A = (1 + psi) / 2")
n_seeds = dict(zip(t, comp["n_seeds"]))
check(n_seeds[500.0] == 5 and n_seeds[1000.0] == 5 and n_seeds[2000.0] == 4,
      "the FNO's mean is over the seeds still running: 5, then 4 once the "
      "diverged one is gone")
c = entry["composition"]
check(c["off"] == 0.2 and c["lattice"] == LAT and c["n_seeds"] == 5
      and abs(c["start"]["A_pct_truth"] - 60.0) < 1e-9
      and abs(c["start"]["B_pct_truth"] - 40.0) < 1e-9
      and c["start"]["t"] == 101.0 and c["end"]["t_truth"] == 2000.0
      and c["end"]["t_pred"] == 2000.0,
      "the manifest's start: 60 / 40 % at the first step")
check(abs(c["end"]["A_pct_pred"] - 50.0 * (1 + 0.2 + 1.2e-6 * 1899)) < 1e-8
      and abs(c["end"]["A_pct_truth"] - 60.0) < 1e-9,
      "and its end, FNO and truth")
check(abs(c["shift_pp"] - 50.0 * 1.2e-6 * 1899) < 1e-8
      and abs(c["drift_rate_psi_per_t"] - 1.2e-6) < 1e-12
      and abs(c["drift_rate_pp_per_1000t"] - 0.06) < 1e-9
      and abs(c["max_abs_drift_pred"] - 1.2e-6 * 1899) < 1e-11,
      f"shift {c['shift_pp']:.6f} pp = 50 x 1.2e-6 x 1899, drift rate 1.2e-6 "
      f"per t = 0.06 pp per 1000 t")
check(entry["files"] == [f"l{LAT}/{k}_{TOK}.png" for k in (
          "snapshots", "domain_growth", "psi_distribution", "composition",
          "free_energy")]
      and entry["extras"]["files"] == ["psi_distribution.csv",
                                       "domain_growth.csv",
                                       "composition.csv"]
      and entry["token"] == TOK,
      f"the case's file lists: the run folder's figures {entry['files']}, "
      f"and the data in extras/ {entry['extras']['files']}")
# The flat layout's composition series of the case: the start and every snapshot time
rows = entry["flat"]["composition_series"]
check([r["t"] for r in rows] == [100.0, 500.0, 1000.0, 2000.0]
      and [r["n_seeds"] for r in rows] == [5, 5, 5, 4]
      and rows[0]["A_pct_truth"] == 60.0 and rows[0]["B_pct_truth"] == 40.0
      and abs(rows[-1]["A_pct_pred"] - 50.0 * (1 + 0.2 + 1.2e-6 * 1899))
      < 1e-8 and abs(rows[-1]["drift_pred"] - 1.2e-6 * 1899) < 1e-11,
      "the flat layout's composition series: the start and the snapshot "
      "times, with the seeds still running at each")
flat_csv = os.path.join(BASE, "flat_composition.csv")
quiet(S.write_composition_csv, flat_csv, [entry["flat"]])
whole = table(flat_csv)
check(list(whole) == list(S.CSV_COLS) and whole["case"] == [TOK] * 4
      and list(whole["t"]) == [100.0, 500.0, 1000.0, 2000.0]
      and np.allclose(whole["psi0"], 0.2)
      and np.allclose(whole["A_pct_truth"], 60.0),
      "composition.csv at the top of a run is the flat layout's: its columns, a row per "
      "snapshot time and case")
old = [rollout(s, 0.2, means=False) for s in SEEDS5[:2]]
check(S.composition_series(old) is None,
      "rollouts cached before the means were kept have no composition")
OLD_ROOT, OLD_X = os.path.join(BASE, "old_root"), os.path.join(BASE, "old_x")
shutil.copytree(CASE_ROOT, OLD_ROOT)
shutil.copytree(XTRAS, OLD_X)
(old_entry, _), _ = quiet(S.write_case, old, SEEDS5[:2], 0.2, LAT, OLD_ROOT,
                          key="k", videos=False, root=OLD_ROOT,
                          extras=OLD_X, token=TOK)
check("composition" not in old_entry
      and not os.path.exists(os.path.join(OLD_ROOT, f"l{LAT}",
                                          f"composition_{TOK}.png"))
      and not os.path.exists(os.path.join(OLD_X, "composition.csv"))
      and f"l{LAT}/composition_{TOK}.png" not in old_entry["files"]
      and "composition.csv" not in old_entry["extras"]["files"],
      "such a case has no entry and no file, and one an earlier run left "
      "there is removed with the rest of the case's files")

print("\n3. the P(psi) range and the lever rule")
window(2000, (100, 500, 1000, 2000))
peak = [rollout(s, 0.9, peaked=True, blip=(s == 1)) for s in (1, 2, 3)]
PDF_PNG = os.path.join(BASE, "pdf.png")
data = {}
PDF_NOTES: dict = {}
got, log = quiet(S.figure_psi_pdf, peak, 0.9, PDF_PNG, LAT, data, PDF_NOTES)
pooled = [np.concatenate([r["snapshots"][2000.0][i].ravel() for r in peak])
          for i in (0, 1)]
bins = np.linspace(S.PDF_RANGE[0], S.PDF_RANGE[1], S.PDF_BINS + 1)
hs = [np.histogram(x, bins=bins, density=True)[0] for x in pooled]
low = 0.5 * min(h[h > 0].min() for h in hs)
high = 2.0 * max(h.max() for h in hs)
fig = seen("pdf.png")
check(max(hs[0].max(), hs[1].max()) > S.PDF_YLIM[1]
      and min(h[h > 0].min() for h in hs) < S.PDF_YLIM[0],
      "the fixture's peaks are above the old window and its thinnest bin "
      "below it")
ylim = fig["axes"][0]["ylim"]
check(not fig.get("missing") and abs(ylim[0] - low) < 1e-12 * low
      and abs(ylim[1] - high) < 1e-9 * high
      and got["y_range"] == [low, high],
      f"the y-range follows the data: {low:.3g} to {high:.3g}")
check(got["peak_density_true"] < high and got["peak_density_pred"] < high
      and got["peak_density_true"] > S.PDF_YLIM[1],
      "so both peaks are in view, the truth's one beyond the old window")
check(S.pdf_ylim(np.array([0.0, 0.2, 5.0]), np.array([0.01, 0.0, 3.0]))
      == (0.005, 10.0) and S.pdf_ylim(np.zeros(3)) == S.PDF_YLIM,
      "half the thinnest non-zero density to twice the highest; the old "
      "window when there is nothing to scale to")
lever = 0.5 * (1 + 0.9)
check(abs(got["lever_rule"] - lever) < 1e-12
      and abs(got["lever_A_pct"] - 95.0) < 1e-9
      and abs(got["lever_B_pct"] - 5.0) < 1e-9
      and abs(got["A_pct_true"] - 100 * got["frac_pos_true"]) < 1e-12
      and abs(got["A_pct_pred"] - 100 * got["frac_pos_pred"]) < 1e-12
      and abs(got["B_pct_pred"] - (100 - got["A_pct_pred"])) < 1e-12
      and abs(got["lever_shift_pred"] - (got["frac_pos_pred"] - lever))
      < 1e-12, "the manifest has the lever-rule fractions of the mean "
      "composition and the measured ones, and the FNO's shift")
check(not fig.get("missing") and "lever rule at psi0 = +0.9: 95.0 % "
      "A-rich / 5.0 % B-rich" in "\n".join(PDF_NOTES["pdf.png"])
      and f"Solver {got['A_pct_true']:.1f} / " in "\n".join(
          PDF_NOTES["pdf.png"])
      and "lever rule" not in text_of(fig),
      "and figure_notes.txt says them, not the figure")
pdf_csv = os.path.join(BASE, "pdf_case")
os.makedirs(pdf_csv)
S.write_case_data(pdf_csv, data)
cols = table(os.path.join(pdf_csv, "psi_distribution.csv"))
check(list(cols)[:3] == ["psi", "density_true", "density_pred"]
      and list(cols)[3:] == ["lever_A_pct", "A_rich_true_pct",
                             "A_rich_pred_pct"]
      and np.all(cols["lever_A_pct"] == cols["lever_A_pct"][0])
      and abs(cols["lever_A_pct"][0] - 95.0) < 1e-9
      and abs(cols["A_rich_true_pct"][0] - got["A_pct_true"]) < 1e-9
      and abs(cols["A_rich_pred_pct"][0] - got["A_pct_pred"]) < 1e-9
      and len(cols["psi"]) == S.PDF_BINS,
      "psi_distribution.csv keeps its first three columns and adds the "
      "lever rule, the truth's and the FNO's A-rich % (one value down the "
      "rows)")

print("\n4. snapshots in one figure")
window(5000, (100, 500, 1000, 2000, 3000, 5000))
long_res = [rollout(s, 0.0) for s in (1, 2, 3)]
LONG_ROOT = os.path.join(BASE, "long_root")
LONG_X = os.path.join(BASE, "long_extras")
LTOK = "psi0p0.00"
(entry_l, _), _ = quiet(S.write_case, long_res, (1, 2, 3), 0.0, LAT,
                        LONG_ROOT, key="k", videos=False, root=LONG_ROOT,
                        extras=LONG_X, token=LTOK)
main_fig = seen(f"snapshots_{LTOK}.png", LONG_ROOT)
check(S.snapshot_columns([100.0, 500.0, 3000.0, 5000.0]) == [100.0, 500.0,
                                                           5000.0]
      and S.snapshot_columns([100.0, 112.0, 120.0]) == [100.0, 112.0, 120.0]
      and S.snapshot_columns([100.0, 500.0, 1000.0, 1200.0])
      == [100.0, 500.0, 1000.0, 1200.0]
      and S.snapshot_columns([500.0, 1000.0]) == [500.0, 1000.0]
      and S.snapshot_columns([112.0, 120.0]) == [112.0, 120.0],
      "the columns are those of SNAPSHOT_COLUMNS the rollout has (a record "
      "may lack t = 100) and its last time, and all its times when it has "
      "fewer than two")
check(not os.path.exists(os.path.join(LONG_ROOT, f"l{LAT}",
                                      f"snapshots_long_{LTOK}.png"))
      and f"l{LAT}/snapshots_long_{LTOK}.png" not in entry_l["files"]
      and "snapshots_long" not in entry_l
      and f"l{LAT}/snapshots_{LTOK}.png" in entry_l["files"]
      and f"l{LAT}/free_energy_{LTOK}.png" in entry_l["files"],
      "the run folder has one snapshots figure, no long one, and the new "
      "free-energy figure")
check(not main_fig.get("missing") and main_fig["images"] == 3 * 5
      and entry_l["snapshots"]["times"] == [100.0, 500.0, 1000.0, 2000.0,
                                            5000.0],
      f"snapshots_<case>.png has a column for each of the times the rollout "
      f"has ({main_fig and main_fig['images']} panels)")
check("R2" not in text_of(main_fig) and "seed" not in text_of(main_fig)
      and "Solver" in main_fig["axes"][0]["legend"] + [
          t for t in main_fig["texts"]] + ["Solver"],
      "the figure has no caption and no R2")
NOTE_L = S.read_notes(os.path.join(LONG_ROOT, S.NOTES_FILE))
check(not os.path.isfile(os.path.join(LONG_ROOT, S.NOTES_FILE))
      and f"snapshots_{LTOK}.png" in entry_l["notes"]
      and any("seed shown" in x
              for x in entry_l["notes"][f"snapshots_{LTOK}.png"])
      and any("R2 of the shown seed at t = 100: " in x
              for x in entry_l["notes"][f"snapshots_{LTOK}.png"])
      and any("FNO monotone" in x
              for x in entry_l["notes"][f"free_energy_{LTOK}.png"]),
      "write_case keeps the text in the entry's notes (the run writes "
      "figure_notes.txt), and it has the R2 of each column")
window(2000, (100, 500, 1000, 2000))
short_res = [rollout(s, 0.0) for s in (1, 2, 3)]
(entry_s, _), _ = quiet(S.write_case, short_res, (1, 2, 3), 0.0, LAT,
                        LONG_ROOT, key="k", videos=False, root=LONG_ROOT,
                        extras=LONG_X, token=LTOK)
check(entry_s["snapshots"]["times"] == [100.0, 500.0, 1000.0, 2000.0],
      "writing the case again for a rollout that ends at 2000 has four "
      "columns")
BLK = {"a.png": ["one", "two " * 40], "b.png": ["x"]}
os.makedirs(os.path.join(BASE, "notes"))
for f in ("a.png", "b.png"):
    open(os.path.join(BASE, "notes", f), "wb").close()
S.write_figure_notes(os.path.join(BASE, "notes"), BLK)
S.write_figure_notes(os.path.join(BASE, "notes"), {"b.png": ["y"]})
os.remove(os.path.join(BASE, "notes", "a.png"))
S.write_figure_notes(os.path.join(BASE, "notes"), {})
check(S.read_notes(os.path.join(BASE, "notes", S.NOTES_FILE))
      == {"b.png": ["y"]},
      "figure_notes.txt keeps the blocks of the other figures, replaces the "
      "one written, and drops the block of a figure that is gone")

print("\n5. a run: composition, snapshots and the report, and its redraw")
setup()
window(3000, (100, 500, 1000, 2000, 3000))
DRIFT[0] = 2e-7
RES_ROOT = os.path.join(BASE, "root", "Results")
RUN1 = os.path.join(RES_ROOT, "train_t0500_pred_t03000")
CACHE1 = os.path.join(BASE, "root", "Work", "rollout_cache",
                      "train_t0500_pred_t03000")
OLDDIR = os.path.join(RES_ROOT, "results_runs_2026-09-30")
os.makedirs(os.path.join(OLDDIR, "A_run"))
with open(os.path.join(OLDDIR, "README.md"), "w", encoding="utf-8") as fh:
    fh.write("# Results - Cahn-Hilliard FNO, three runs\n\nWritten by hand.\n")
with open(os.path.join(OLDDIR, "A_run", "manifest.json"), "w") as fh:
    fh.write("{}")
BROKEN = os.path.join(RES_ROOT, "broken_run")
os.makedirs(BROKEN)
with open(os.path.join(BROKEN, "manifest.json"), "w") as fh:
    fh.write("{not json")
HANDS = {}
for d in (OLDDIR, BROKEN):
    for f in files_under(d):
        p = os.path.join(d, *f.split("/"))
        HANDS[p] = (read(p), os.stat(p).st_mtime_ns)
code, log = run_main(flags(RUN1, CACHE1))
check(code == 0, f"a run to 3000 over the five compositions exits 0: {code}")
cases5 = ["psi0m0.40", "psi0m0.20", "psi0p0.00", "psi0p0.20", "psi0p0.40"]
internal5 = [f"psi0{o}_l{LAT}" for o in
             ("-0.40", "-0.20", "+0.00", "+0.20", "+0.40")]
missing = [S.case_file(k, tok) for tok in cases5
           for k in ("snapshots", "composition", "free_energy")
           if not os.path.isfile(os.path.join(RUN1, S.case_file(k, tok)))]
check(not missing and not [f for f in os.listdir(os.path.join(RUN1, f"l{LAT}"))
                           if f.startswith("snapshots_long")],
      f"every case has its snapshots, composition and free-energy figure, "
      f"in the folder of its box size, and no long one: {missing}")
NOTES1 = S.read_notes(os.path.join(RUN1, S.NOTES_FILE))
NOTESL = S.read_notes(os.path.join(RUN1, f"l{LAT}", S.NOTES_FILE))
# (this suite stubs the kinetics figure, which writes its own notes)
WANT_N = {S.fig_name(k, tok.split("_")[0]) for tok in cases5
          for k in S.FIG_KINDS if k != "phase_ordering_kinetics"} | {
    "composition_all.png", "r2_time.png"}
NOTESW = S.read_notes(os.path.join(RUN1, "r2_sweep_per_size", S.NOTES_FILE))
WANT_TOP = {f"l{LAT}.png"}
check(WANT_N <= set(NOTESL) and WANT_TOP == set(NOTESW)
      and not any(n.startswith("r2_sweep") for n in NOTES1)
      and all(os.path.isfile(os.path.join(RUN1, n)) for n in NOTES1)
      and all(os.path.isfile(os.path.join(RUN1, "r2_sweep_per_size", n))
              for n in NOTESW)
      and all(os.path.isfile(os.path.join(RUN1, f"l{LAT}", n))
              for n in NOTESL),
      f"figure_notes.txt has a block for each figure of its folder, and "
      f"only for those: the run-wide figures at the top, the case figures, "
      f"composition_all.png and r2_time.png in the size's folder (missing "
      f"{sorted(WANT_N - set(NOTESL))}, {sorted(WANT_TOP - set(NOTESW))})")
m1 = jload(os.path.join(RUN1, "manifest.json"))
book1 = jload(os.path.join(CACHE1, "extras", S.BOOKKEEPING))
check(sorted(m1["cases"]) == sorted(cases5)
      and all("snapshots" in c and "composition_series" in c
              and "seeds" in c and "r2_curve" in c
              and "snapshots_long" not in c
              for c in m1["cases"].values())
      and m1["train_t_end"] == 500.0
      and "snapshot_times_long" not in m1
      and all(m1["snapshot_times_main"][t] == [100.0, 500.0, 1000.0, 2000.0,
                                               3000.0]
              for t in cases5),
      "manifest.json has the cases by token with their new entries, the "
      "training horizon, and the snapshot times of each")
check(book1["train_horizon"] == 500.0
      and all("snapshots" in e and "composition" in e
              and "seed_ranks" in e and "notes" in e for e in book1["compositions"].values()),
      "and the bookkeeping in extras/ keeps the entries of the old kind")
whole = table(os.path.join(RUN1, "composition.csv"))
head = text(os.path.join(RUN1, "composition.csv")).splitlines()[0]
per_case = [table(os.path.join(CACHE1, "extras", c, "composition.csv"))
            for c in internal5]
check(head == ",".join(S.CSV_COLS) and len(whole["t"]) == 5 * 5
      and sorted(set(whole["psi0"])) == [-0.4, -0.2, 0.0, 0.2, 0.4]
      and all(len(x["t"]) > 100 for x in per_case),
      "composition.csv at the top holds the flat layout's rows (five snapshot times of "
      "five cases); each case's dense table is in its extras folder")
cases_col = [ln.split(",")[0] for ln in
             text(os.path.join(RUN1, "composition.csv")).splitlines()[1:]]
check(sorted(set(cases_col)) == sorted(cases5)
      and cases_col == sorted(cases_col, key=cases5.index),
      "in the order of psi0, each case's rows together")
fig = seen("composition_all.png", os.path.join(RUN1, f"l{LAT}"))
lines = fig["axes"][0]["lines"]
check(not fig.get("missing") and fig["axes"][0]["xscale"] == "log"
      and sum(ln[3] == "-" and ln[0].startswith("$") for ln in lines) == 5
      and sum(ln[3] == "--" and ln[0].startswith("_") for ln in lines) == 5,
      f"composition_all.png: five compositions, truth solid and FNO dashed, "
      f"on a log t axis ({len(lines)} lines)")
flat = sorted(round(float(ln[2][0]), 3) for ln in lines
              if ln[3] == "-")
check(flat == [30.0, 40.0, 50.0, 60.0, 70.0],
      f"each truth line sits at its c_A: {flat}")
marks = [ln for ln in fig["axes"][0]["lines"] if len(ln[1]) == 2
         and np.all(ln[1] == 500.0)]
check(len(marks) == 1,
      "the training horizon (500) is marked, with the helper the case "
      "figures use")
rep5 = text(os.path.join(RUN1, "report.txt"))
row = [ln.split() for ln in rep5.splitlines()
       if ln.startswith("  psi0p0.20 ")][0]
shift = 50.0 * 2e-7 * 2899
check("MIXTURE COMPOSITION AND ITS CONSERVATION" in rep5
      and row[:5] == ["psi0p0.20", "60.000", "60.000", f"{60 + shift:.3f}",
                      f"{shift:.4f}"]
      and row[5] == f"{2e-7 * 2899:.2e}" and row[6] == "2.00e-07",
      f"report.txt has the A / B table: {row}")
check("psi0p0.20: c_A (%) over time" in rep5
      and not os.path.exists(os.path.join(RUN1, "README.md")),
      "and the A / B tables over time; the run folder has no README.md")
check(not [f for f in files_under(RUN1) if f.endswith((".tmp", ".partial.png"))
           or ".partial" in f],
      "no temporary file is left in the run folder")

print("\n6. the redraw from the records is the same")
setup()
window(3000, (100, 500, 1000, 2000, 3000))
P.evaluate_seed = P.evaluate_seeds = no_rollout
RE1 = os.path.join(BASE, "redraw1")
code, log = redraw(CACHE1, RE1)
check(code is None, f"the redraw exits cleanly: {code}")


def pngs_and_csvs(d):
    return {f: read(os.path.join(d, *f.split("/"))) for f in files_under(d)
            if f.endswith((".png", ".csv", ".pdf"))}


a, b = pngs_and_csvs(RUN1), pngs_and_csvs(RE1)
check(a == b and len(a) >= 25 and "composition.csv" in a
      and f"l{LAT}/composition_all.png" in a and f"r2_sweep_per_size/l{LAT}.png" in a
      and not [f for f in a if f.endswith(".pdf") and "sweep" in f]
      and f"l{LAT}/snapshots_psi0p0.20.png" in a,
      f"{len(a)} figures and CSVs, the new ones too, byte for byte")
m2 = jload(os.path.join(RE1, "manifest.json"))
check(all(m1["cases"][k][f] == m2["cases"][k][f]
          for k in m1["cases"]
          for f in ("composition_series", "seeds",
                    "psi_pdf", "snapshots", "r2_at_end"))
      and m1["train_t_end"] == m2["train_t_end"]
      and m2["records"]["partial"] is False,
      "and the manifest's new entries")


def section(txt, n):
    """Section n of a report.txt: from its heading to the next one."""
    head = [i for i, ln in enumerate(txt.splitlines())
            if ln.startswith(f"  {n}  ")][0]
    lines = txt.splitlines()[head:]
    stop = [i for i, ln in enumerate(lines[1:], 1)
            if ln[:3] == "  " + str(n + 1)[0] and ln[3:5] == "  "]
    return "\n".join(lines[:stop[0] if stop else None])


rep6 = text(os.path.join(RE1, "report.txt"))
check(all(section(rep6, n) == section(rep5, n) for n in (2, 3, 4, 7))
      and len(section(rep5, 2).splitlines()) > 30
      and "redrawn from records" in rep6,
      "and the report's composition, solve time, summary and conventions "
      "sections")
check(not os.path.exists(os.path.join(os.path.dirname(RE1), "README.md")),
      "a redraw writes no overview")

print("\n7. the results index")
setup()
window(3000, (100, 500, 1000, 2000, 3000))
INDEX = os.path.join(RES_ROOT, "README.md")
check(os.path.isfile(INDEX) and not os.path.exists(
    os.path.join(RES_ROOT, "comparison.png")),
      "the run's end rebuilt Results/README.md; one run is not enough for "
      "comparison.png")
idx = text(INDEX)
# R2 at the end is the snapshot's (the flat layout's), the curve's last value the series'
ends = np.array([float(P.r2_score(*rollout(s, 0.0)["snapshots"][3000.0]))
                 for s in P.EVAL_SEEDS[:3]])
expect = float(ends.mean())
curve_end = float(np.mean([rollout(s, 0.0)["r2"][-1]
                           for s in P.EVAL_SEEDS[:3]]))
cells = rows_of(idx, "| `train_t0500_pred_t03000/`")
check(len(cells) == 1 and cells[0][1] == "1,234"
      and cells[0][2] == "t = 500" and cells[0][3] == "t = 3000"
      and cells[0][4] == "3" and cells[0][5] == "0"
      and cells[0][6] == f"{expect:.4f} +- {float(ends.std(ddof=1)):.4f}",
      f"its row: label, parameters, horizons, seeds, diverged and the "
      f"critical mean R2 at 3000 = {expect:.4f}: "
      f"{cells[0] if cells else None}")
check(f"| psi0p0.20 |" in idx and "## train_t0500_pred_t03000" in idx
      and "[manifest.json](" not in idx,
      "with a table of its cases (by the flat layout's tokens) and no links into the "
      "run folder")
check("| `broken_run/` | - | - | - | - | - | manifest.json cannot be read |"
      in idx, "a manifest that cannot be read is listed as such")
check("## Other folders" in idx and "- `results_runs_2026-09-30/` - Results "
      "- Cahn-Hilliard FNO, three runs" in idx
      and "A_run" not in idx,
      "a folder without a manifest is listed with its README's first line, "
      "and what is nested in it is not a run")
check("`comparison.png` is drawn once two runs have a curve" in idx,
      "and the page says why there is no comparison yet")

DRIFT[0] = 0.0
window(2000, (100, 500, 1000, 2000))
RUN2 = os.path.join(RES_ROOT, "train_t0300_pred_t02000")
CACHE2 = os.path.join(BASE, "root", "Work", "rollout_cache",
                      "train_t0300_pred_t02000")
code, log = run_main(flags(RUN2, CACHE2, CKPT300, "critical", "2"))
check(code == 0 and "[index]" not in log, f"a second run, to 2000, from "
      f"another checkpoint, exits 0: {code}")
idx = text(INDEX)
order = [ln.split("`")[1] for ln in idx.splitlines()
         if ln.startswith("| `train_")]
check(order == ["train_t0300_pred_t02000/", "train_t0500_pred_t03000/"],
      f"the runs are in order of the horizon they were trained to: {order}")
cells = rows_of(idx, "| `train_t0300_pred_t02000/`")[0]
ends2 = np.array([float(P.r2_score(*rollout(s, 0.0)["snapshots"][2000.0]))
                  for s in P.EVAL_SEEDS[:2]])
check(cells[2] == "t = 300" and cells[3] == "t = 2000" and cells[4] == "2"
      and cells[6] == f"{float(ends2.mean()):.4f} +- "
                      f"{float(ends2.std(ddof=1)):.4f}",
      f"the second row: {cells}")
cmp_png = os.path.join(RES_ROOT, "comparison.png")
check(os.path.isfile(cmp_png) and "puts their critical curves" in idx,
      "two runs: comparison.png is drawn at the run's end")
fig = seen("comparison.png", RES_ROOT)
curves = [ln for ln in fig["axes"][0]["lines"] if ln[0].startswith("train_")]
check(len(curves) == 2 and fig["axes"][0]["xscale"] == "log"
      and sorted(float(ln[1][-1]) for ln in curves) == [2000.0, 3000.0]
      and all(ln[0].startswith("train_t0") and ":" not in ln[0]
              for ln in curves)
      and any(f"R2(3000) = {curve_end:.3f}" in x for x in S.read_notes(
          os.path.join(RES_ROOT, S.NOTES_FILE))["comparison.png"]),
      "it has one curve per run on a log t axis, each to its own end, "
      "labelled with the run; the R2 at the end is in figure_notes.txt")
vlines = sorted(float(ln[1][0]) for ln in fig["axes"][0]["lines"]
                if len(ln[1]) == 2 and ln[1][0] == ln[1][1]
                and ln[0].startswith("_"))
check(vlines == [300.0, 500.0],
      f"the training horizon of each run is marked: {vlines}")
check({p: (read(p), os.stat(p).st_mtime_ns) for p in HANDS} == HANDS
      and os.listdir(BROKEN) == ["manifest.json"]
      and sorted(os.listdir(OLDDIR)) == ["A_run", "README.md"],
      "the hand-written folder and the unreadable manifest are untouched, "
      "bytes and times, and nothing was moved aside")
check([f for f in files_under(RES_ROOT) if "/" not in f]
      == ["README.md", "comparison.png", "figure_notes.txt"],
      "the index added only README.md, comparison.png and figure_notes.txt "
      "at the top")
before = (read(INDEX), read(cmp_png))
quiet(S.write_results_index, RES_ROOT)
check((read(INDEX), read(cmp_png)) == before,
      "built again from the same folders, both come out byte for byte the "
      "same")

print("\n7b. writes are atomic, and the index keeps what is not its own")
real_replace = os.replace


def failing(match):
    def replace(src, dst, *a, **k):
        if str(dst).endswith(match):
            raise PermissionError("held open by a viewer")
        return real_replace(src, dst, *a, **k)
    return replace


for match in ("README.md", "comparison.png"):
    os.replace = failing(match)
    try:
        err, log = quiet(S.write_results_index, RES_ROOT)
    except PermissionError:
        err = "raised"
    finally:
        os.replace = real_replace
    check((read(INDEX), read(cmp_png)) == before,
          f"a failed write of {match} leaves the old file whole")
leftover = [f for f in os.listdir(RES_ROOT)
            if f.endswith(".tmp") or ".partial" in f]
check(not leftover, f"and no temporary file: {leftover}")

print("\n   the index is written through a temporary of this process's own")
renames = []


def spying(src, dst, *a, **k):
    renames.append((os.path.basename(str(src)),
                    os.path.basename(str(dst))))
    return real_replace(src, dst, *a, **k)


os.replace = spying
try:
    quiet(S.write_results_index, RES_ROOT)
finally:
    os.replace = real_replace
mine = [pair for pair in renames if pair[1] in ("README.md", "comparison.png")]
want_tmp = [(os.path.basename(S.shared_tmp(cmp_png)), "comparison.png"),
            (os.path.basename(S.shared_tmp(INDEX)), "README.md")]
check(mine == want_tmp and str(os.getpid()) in mine[0][0]
      and str(os.getpid()) in mine[1][0]
      and mine[0][0] != "comparison.partial.png"
      and mine[1][0] != "README.md.tmp",
      f"README.md and comparison.png are renamed into place from names with "
      f"the host and pid: {mine}")
check((read(INDEX), read(cmp_png)) == before,
      "and the files come out byte for byte the same")
real_pid = os.getpid
os.getpid = lambda: 424242
try:
    other = S.shared_tmp(INDEX), S.shared_tmp(cmp_png)
finally:
    os.getpid = real_pid
check(other[0] != S.shared_tmp(INDEX) and other[1] != S.shared_tmp(cmp_png)
      and "424242" in other[0] and "424242" in other[1],
      "another process gets other names, so two jobs never share one")
old_a = os.path.join(RES_ROOT, "README.md.nodeX-77.tmp")
old_b = os.path.join(RES_ROOT, "comparison.nodeX-77.partial.png")
live = os.path.join(RES_ROOT, "README.md.nodeY-88.tmp")
for name in (old_a, old_b, live):
    with open(name, "wb") as fh:
        fh.write(b"half")
two_days = time.time() - 172800
for name in (old_a, old_b):
    os.utime(name, (two_days, two_days))
quiet(S.write_results_index, RES_ROOT)
check(not os.path.exists(old_a) and not os.path.exists(old_b)
      and os.path.exists(live) and (read(INDEX), read(cmp_png)) == before,
      "temporaries a dead job left (a day old) are cleared when the index "
      "is written, and one a live job may be writing is not touched")
for name in (old_a, old_b):
    with open(name, "wb") as fh:
        fh.write(b"half")
    os.utime(name, (two_days, two_days))
S.recover_interrupted(RES_ROOT)
check(not os.path.exists(old_a) and not os.path.exists(old_b)
      and os.path.exists(live),
      "and recover_interrupted clears them the way it clears the fixed ones")
os.remove(live)
FOREIGN = os.path.join(BASE, "foreign", "Results")
os.makedirs(os.path.join(FOREIGN, "r"))
with open(os.path.join(FOREIGN, "r", "manifest.json"), "w") as fh:
    fh.write("{}")
with open(os.path.join(FOREIGN, "README.md"), "w") as fh:
    fh.write("my own notes\n")
res, log = quiet(S.write_results_index, FOREIGN)
check(text(os.path.join(FOREIGN, "README.md")) == "my own notes\n"
      and "was not written by this index" in log and res["written"] is False,
      "a README.md this index did not write is kept, and the log says so")
check(quiet(S.write_results_index, os.path.join(BASE, "nowhere"))[0]
      is not None, "a results folder that is not there is refused")

print("\n8. python make_submission.py --index, and --smoke")
setup()
CLI = os.path.join(BASE, "cli", "Results")
for run in ("train_t0100_pred_t00500", "train_t0200_pred_t00500"):
    shutil.copytree(RUN2, os.path.join(CLI, run))
fno_paths.RESULTS = CLI
real_start = S.fno_runlog.start_log
S.fno_runlog.start_log = lambda *a, **k: None
try:
    code, log = quiet(S.main, ["--index"])
    cli_before = read(os.path.join(CLI, "README.md"))
    smoke_work = os.path.join(BASE, "cli", "Work")
    fno_paths.WORK = smoke_work
    fno_paths.RESULTS = RESULTS
    smoke_results = os.path.join(smoke_work, "smoke", "Results")
    shutil.copytree(RUN2, os.path.join(smoke_results,
                                       "train_t0100_pred_t0200"))
    code2, log2 = quiet(S.main, ["--index", "--smoke"])
    bad, log3 = quiet(S.main, ["--index", "--all"])
finally:
    S.fno_runlog.start_log = real_start
    setup()
check(code is None and os.path.isfile(os.path.join(CLI, "README.md"))
      and os.path.isfile(os.path.join(CLI, "comparison.png"))
      and "train_t0100_pred_t00500" in text(os.path.join(CLI, "README.md")),
      "--index writes Results/README.md and comparison.png")
check(code2 is None and os.path.isfile(os.path.join(smoke_results,
                                                    "README.md"))
      and not os.path.exists(os.path.join(smoke_results, "comparison.png"))
      and read(os.path.join(CLI, "README.md")) == cli_before
      and "train_t0100_pred_t0200" in text(os.path.join(smoke_results,
                                                         "README.md")),
      "with --smoke the index goes in the smoke root's Results (one run: "
      "no comparison)")
check(bad == 2, "--index refuses to be combined with --all")

print("\n9. a run outside the results root leaves the root alone")
setup()
window(2000, (100, 500, 1000, 2000))
ELSE = os.path.join(BASE, "elsewhere", "train_t0500_pred_t02000")
ROOT_BEFORE = {f: read(os.path.join(RES_ROOT, *f.split("/")))
               for f in files_under(RES_ROOT) if "/" not in f}
code, log = run_main(flags(ELSE, os.path.join(BASE, "root", "Work", "c3"),
                           CKPT500, "critical", "2"))
check(code == 0 and "is not a folder of" in log
      and {f: read(os.path.join(RES_ROOT, *f.split("/")))
           for f in files_under(RES_ROOT) if "/" not in f} == ROOT_BEFORE,
      "a run sent to another folder (--results-dir) does not rewrite the "
      "index of Results/")

print("\n10. R2 at fixed times, what the model was trained with, and a seed "
      "that diverges at its first step")
setup()
window(3000, (100, 500, 1000, 2000, 3000))
R10 = os.path.join(BASE, "r10")
shutil.rmtree(R10, ignore_errors=True)
T_IDX = 2000 - 101


def r2_2000(seed, off):
    return float(rollout(seed, off)["r2"][T_IDX])


def case_vals(m, kind, key):
    """What the fake rollouts give each seed of a bookkeeping case at 2000."""
    off = float(key) if kind == "compositions" else 0.0
    return [r2_2000(s, off) for s in m[kind][key]["seed_list"]]


vals = {(k, key): case_vals(book1, k, key)
        for k in ("compositions", "sizes") for key in book1[k]}
check(all(e["r2_at_times"].keys() == {"2000"}
          and e["r2_at_times"]["2000"]["n"] == len(e["seed_list"])
          and e["r2_at_times"]["2000"]["diverged"] == 0
          and np.isclose(e["r2_at_times"]["2000"]["mean"],
                         np.mean(vals[(kind, key)]), rtol=0, atol=1e-12)
          for kind in ("compositions", "sizes")
          for key, e in book1[kind].items())
      and all(c["r2_at_times"].keys() == {"2000"}
              for c in m1["cases"].values()),
      "every case's entry, in the bookkeeping and in manifest.json, has its "
      "mean R2 at t = 2000 (the run reaches no later reporting time), over "
      "its seeds")
comp_all = [v for (k, _), vs in vals.items() if k == "compositions"
            for v in vs]
size_all = [v for (k, _), vs in vals.items() if k == "sizes" for v in vs]
g = S.group_r2(book1) or {}
check(g and g["pooled"]["cases"] == 5
      and g["compositions"]["cases"] == 5 and g["sizes"]["cases"] == 1
      and np.isclose(g["pooled"]["at"]["2000"]["mean"], np.mean(comp_all))
      and np.isclose(g["compositions"]["at"]["2000"]["mean"],
                     np.mean(comp_all))
      and np.isclose(g["sizes"]["at"]["2000"]["mean"], np.mean(size_all))
      and g["pooled"]["at"]["2000"]["n"] == len(comp_all),
      "pooled: the five cases once each (the critical quench is also the "
      "one box size); per group: that group's own seeds")
fixed = {ln.split()[0]: ln.split()
         for ln in rep5.splitlines()
         if ln.startswith(("  group ", "  pooled ", "  compositions ",
                           "  box sizes ")) and len(ln.split()) > 3}
mc, ms = f"{np.mean(comp_all):.3f}", f"{np.mean(size_all):.3f}"
check(fixed.get("group", [])[2:] == ["t", "=", "2000"]
      and fixed.get("pooled") == ["pooled", "5", mc, f"({mc})", "0"]
      and fixed.get("compositions") == ["compositions", "5", mc, f"({mc})",
                                        "0"]
      and fixed.get("box") == ["box", "sizes", "1", ms, f"({ms})", "0"],
      f"report.txt has the R2 at the reporting times the run reaches, "
      f"pooled and per group, both means and the diverged count: {fixed}")
flat5 = " ".join(rep5.split())      # the prose is wrapped to 78 columns
check("R2 = -1 from then on" in flat5 and "(the all-seeds mean)" in flat5
      and "the survivors-only mean, which leaves diverged seeds out" in flat5
      and "Each cell: the all-seeds mean over the group's cases (a diverged "
          "seed counts R2 = -1); in brackets the survivors-only mean "
          "(diverged seeds left out)" in flat5
      and fno_paths.OLD_STORED_NAME not in flat5.lower(),
      "and says that a diverged seed counts at R2 = -1 in the all-seeds "
      "mean, that the survivors-only mean is in brackets only, and names "
      "no other project")
idx = text(INDEX)
cells = rows_of(idx, "| `train_t0500_pred_t03000/`")[0]
check("critical quench" in idx and "stays in the mean" in idx
      and cells[6].split()[0] == f"{expect:.4f}",
      f"the overview shows the critical mean R2 at the end of each run: "
      f"{cells[6]}")
check("is the all-seeds mean of the critical quench" in idx
      and "The survivors-only mean (diverged seeds left out)" in idx
      and "Each box size has a folder" in idx
      and fno_paths.OLD_STORED_NAME not in idx.lower(),
      "and says which mean that column is and where the other one is, and "
      "names no other project")


def go(name, ckpt=None, die_first=None, drift=0.0, seeds="3"):
    """A critical-quench run into its own folder, with a checkpoint of
    these contents, and a seed that diverges at its first step: the first
    one (die_first True), the last (False), or none (None)."""
    setup()
    window(3000, (100, 500, 1000, 2000, 3000))
    DRIFT[0] = drift
    base = {"best": {"epoch": 7, "score": 0.91, "survival": 2000.0}}
    base.update(ckpt or {})
    P.load_checkpoint = lambda path, device=None: (FakeModel(), base)

    def evaluate(model, sds, device, off=0.0, lattice=None):
        n = len(sds)
        return [rollout(s, off, drift=drift,
                        die=100.0 if die_first is not None
                        and i == (0 if die_first else n - 1) else None)
                for i, s in enumerate(sds)]
    P.evaluate_seeds = evaluate
    P.evaluate_seed = lambda model, sd, device, off=0.0, lattice=None: \
        evaluate(model, [sd], device, off, lattice)[0]
    out = os.path.join(R10, name, "Results", "train_t0500_pred_t03000")
    cache = os.path.join(R10, name, "Work", "train_t0500_pred_t03000")
    code, log = run_main(flags(out, cache, CKPT500, "critical", seeds))
    DRIFT[0] = 0.0
    return code, out, cache, log


def summary(cache):
    folder = os.path.join(cache, "extras", f"psi0+0.00_l{LAT}")
    name = [f for f in os.listdir(folder) if f.endswith("_summary.json")]
    return jload(os.path.join(folder, name[0])) if len(name) == 1 else {}


code_f, out_f, cache_f, log_f = go("first", die_first=True, drift=2e-6)
code_l, out_l, cache_l, log_l = go("last", die_first=False, drift=2e-6)
code_n, out_n, cache_n, log_n = go("none", drift=2e-6)
want = 2e-6 * (3000 - 101)
sf, sl, sn = summary(cache_f), summary(cache_l), summary(cache_n)
check(code_f == 0 and code_l == 0 and code_n == 0
      and np.isclose(sn["max_mean_error"], want, rtol=1e-9),
      f"three runs of the critical quench exit 0, and a plain one has "
      f"max_mean_error {sn.get('max_mean_error')} = the drift {want:.3e}")
check(sf.get("max_mean_error") is not None
      and sl.get("max_mean_error") is not None
      and np.isclose(sf["max_mean_error"], want, rtol=1e-9)
      and np.isclose(sl["max_mean_error"], want, rtol=1e-9),
      f"a seed that diverges at its first step, first or last in the "
      f"order, leaves the finite seeds' maximum: "
      f"{sf.get('max_mean_error')}, {sl.get('max_mean_error')}")
check(sf["seeds_diverged"] == 1 and sl["seeds_diverged"] == 1,
      "and is counted as diverged")
mf = jload(os.path.join(out_f, "manifest.json"))
bf = jload(os.path.join(cache_f, "extras", S.BOOKKEEPING))
e = bf["compositions"]["+0.00"]
seeds_f = e["seed_list"]
mine = [-1.0] + [r2_2000(s, 0.0) for s in seeds_f[1:]]
at = e["r2_at_times"]["2000"]
check(at["n"] == 3 and at["diverged"] == 1
      and np.isclose(at["mean"], np.mean(mine))
      and abs(at["mean"] - np.mean(mine[1:])) > 0.05
      and mf["cases"]["psi0p0.00"]["r2_at_times"]["2000"] == at,
      f"the diverged seed counts at R2 = -1 in the mean at 2000 "
      f"({at['mean']:.4f}, not the survivors' {np.mean(mine[1:]):.4f})")
repf = text(os.path.join(out_f, "report.txt"))
check(at["n_alive"] == 2 and np.isclose(at["mean_alive"], np.mean(mine[1:])),
      f"and the survivors-only mean is kept beside it "
      f"({at['n_alive']} seeds, {at['mean_alive']:.4f})")
row = [ln for ln in repf.splitlines()       # not the prose's "pooled row."
       if ln.startswith("  pooled ") and len(ln.split()) > 3][0]
check(row.split()[2:] == [f"{np.mean(mine):.3f}",
                          f"({np.mean(mine[1:]):.3f})", "1"]
      and mf["n_diverged"] == 1,
      f"and so is the report's table, with both means and the diverged "
      f"count: {row.strip()}")

ckpt = {"config": {"channels": 24, "modes": 8, "laplacian_output": True,
                   "mu_channel": True, "lowk_fill": True},
        "train_config": {"seed": 3}}
code_t, out_t, cache_t, log_t = go("trained", ckpt=ckpt)
bt = jload(os.path.join(cache_t, "extras", S.BOOKKEEPING))
want_setup = {"seed": 3, "channels": 24, "modes": 8,
              "laplacian_output": True, "mu_channel": True, "lowk_fill": True}
rept = text(os.path.join(out_t, "report.txt"))
opts_on = "checkpoint options   mu_channel=True, lowk_fill=True"
check(code_t == 0 and bt["train_setup"] == want_setup
      and json.dumps(want_setup) in rept
      and "laplacian_output=True" in rept and opts_on in rept
      and jload(os.path.join(cache_t, "meta.json"))[
          "train_setup"] == want_setup,
      "a checkpoint's seed, size, head and mu_channel / lowk_fill are in "
      "the bookkeeping, the report (section 1) and the cache's meta.json")
check(bf["train_setup"] is None and "variant / size       null" in repf
      and "checkpoint options   mu_channel=False, lowk_fill=False" in repf,
      "a checkpoint that records none leaves them out, and its options "
      "read as off (absent = False)")
RE10 = os.path.join(R10, "redraw")
P.evaluate_seed = P.evaluate_seeds = no_rollout
code_r, log_r = redraw(cache_t, RE10)
br = jload(os.path.join(cache_t, "extras", S.BOOKKEEPING))
repr_ = text(os.path.join(RE10, "report.txt"))
check(code_r is None and br["train_setup"] == want_setup
      and json.dumps(want_setup) in repr_ and opts_on in repr_
      and [ln for ln in repr_.splitlines() if ln.startswith("    t = ")]
      == [ln for ln in rept.splitlines() if ln.startswith("    t = ")]
      and jload(os.path.join(RE10, "manifest.json"))["cases"]["psi0p0.00"][
          "r2_at_times"] == jload(os.path.join(out_t, "manifest.json"))[
          "cases"]["psi0p0.00"]["r2_at_times"],
      "a redraw from the records keeps both, report and all")

# a record from before the two options: the report falls back to the model's
# config, and says so when it has neither
old_facts = {"train_setup": {"seed": 3, "channels": 24, "modes": 8,
                             "laplacian_output": True},
             "arch": {"lowk_fill": True}}
check(S._checkpoint_options(old_facts) == "mu_channel=False, lowk_fill=True"
      and S._checkpoint_options({"train_setup": None, "arch": {}})
      == "not recorded"
      and S._checkpoint_options({"train_setup": want_setup})
      == "mu_channel=True, lowk_fill=True",
      "an older record's options come from the model config, or are "
      "reported as not recorded")

print("\n11. the training curves figure, the history it reads, the case "
      "comparisons and the size panels")
setup()
S.WRITE_TRAINING = True                # the suites keep it off (_paths.py)
BOX11 = os.path.join(BASE, "fig11")
os.makedirs(BOX11)


def hist_rows(n=24, diverged=None, validated=True):
    """A training history: an entry for each epoch with the unroll depth
    stepping 1, 2, 4 at epochs 1, 4 and 8, the rollout validation at every
    second epoch, and ``diverged`` the epoch of a rolled-back blow-up."""
    rows = []
    for e in range(1, n + 1):
        if e == diverged:
            rows.append({"epoch": e, "diverged": True, "train": 2.6e4,
                         "rolled_back_to": e - 1})
            continue
        row = {"epoch": e, "unroll": 1 if e < 4 else 2 if e < 8 else 4,
               "train": 0.08 / e ** 0.5, "val1": 3e-5 + 1e-6 * (e % 5)}
        if validated and e % 2 == 0:
            row.update(score=0.9 - 0.8 / e, survival=100.0 * e,
                       per_seed=[100.0 * e, 120.0 * e, 90.0 * e],
                       in_range=1.0 - 0.5 / e)
        rows.append(row)
    return rows


def curves_of(ax):
    """The data lines of an axes of a seen figure: more than two points, or a
    single marked point, not the reference lines."""
    return [ln for ln in ax["lines"] if len(ln[1]) > 2]


def verticals(ax, x):
    return [ln for ln in ax["lines"] if len(ln[1]) == 2
            and ln[1][0] == ln[1][1] == x]


rows11 = hist_rows(diverged=9)
pick = S.chosen_row(rows11, {"epoch": 20, "score": 0.9 - 0.8 / 20})
check(pick is rows11[19] and S.chosen_row(rows11, {"epoch": 20}) is pick
      and S.chosen_row(rows11, {"epoch": 20, "score": 0.5}) is None
      and S.chosen_row(rows11, {"epoch": 21, "score": 0.1}) is None
      and S.chosen_row(rows11, {"epoch": 9, "score": 0.1}) is None
      and S.chosen_row(rows11, None) is None
      and S.chosen_row(rows11, {"epoch": "x"}) is None,
      "chosen_row: the entry of the selected epoch; none when its score is "
      "another (a history of another training), when it was not validated "
      "or was the blown-up epoch, or when the selection is not an epoch")
TN: dict = {}
quiet(S.figure_training_curves, rows11, os.path.join(BOX11, "tc.png"),
      "Cahn-Hilliard FNO", "train_t0500", 1234, pick, 500.0, TN)
fig = seen("tc.png", BOX11)
ax_a, ax_b, ax_c = fig["axes"] if len(fig["axes"]) == 3 else [EMPTY["axes"][0]] * 3
check(not fig.get("missing") and len(fig["axes"]) == 3
      and fig["titles"] == ["(a)", "(b)", "(c)"]
      and [a["yscale"] for a in fig["axes"]] == ["log", "linear", "linear"],
      "three panels on one epoch axis: the loss on a log scale, the rollout "
      "validation, the survival time")
by = {ln[0].split(" (")[0]: ln for ln in curves_of(ax_a)}
tr, va = by.get("training loss"), by.get("validation loss")
want_x = [float(e) for e in range(1, 25) if e != 9]
check(tr is not None and va is not None and list(tr[1]) == want_x
      and np.allclose(tr[2], 0.08 / np.sqrt(tr[1]))
      and list(va[1]) == want_x
      and tr[4] == S.TRAIN_COLORS["train"]
      and va[4] == S.TRAIN_COLORS["val1"],
      "(a) the training and the one-step validation loss of every epoch but "
      "the blown-up one (9, whose loss of 2.6e4 is not drawn)")
cross = [ln for ln in ax_a["lines"] if ln[5] == "x"]
check(len(cross) == 1 and list(cross[0][1]) == [9.0]
      and 0.08 / np.sqrt(10) <= cross[0][2][0] <= 0.08 / np.sqrt(8),
      "a cross marks epoch 9, on the training curve where it was left out")
dots = [ln for a in (ax_a, ax_b, ax_c) for ln in a["lines"] if ln[5] == "o"]
check(all(len(verticals(a, 20.0)) == 1 for a in fig["axes"])
      and all(list(ln[1]) == [20.0] for ln in dots)
      and np.allclose([ln[2][0] for ln in dots], [
          0.08 / np.sqrt(20), 3e-5 + 1e-6 * (20 % 5), 0.9 - 0.8 / 20,
          1.0 - 0.5 / 20, 2000.0], rtol=1e-12, atol=0.0)
      and [ln[4] for ln in dots] == [S.TRAIN_COLORS[k] for k in (
          "train", "val1", "score", "in_range", "survival")],
      "the selected epoch is a vertical line in every panel, with a dot on "
      "each of the five curves, in its colour, at its value")
steps = [x for x in (3.5, 7.5) if all(
    any(ln[3] == ":" and len(ln[1]) == 2 and ln[1][0] == x for ln in a["lines"])
    for a in fig["axes"])]
check(steps == [3.5, 7.5]
      and not [t for t in fig["texts"] if "unroll" in t]
      and any("epoch 4 (unroll 2), epoch 8 (unroll 4)" in x
              for x in TN["tc.png"])
      and not any("unroll 1" in x for x in TN["tc.png"]),
      "the unroll depth stepping to 2 (epoch 4) and 4 (epoch 8) is a dotted "
      "line in each panel, named in the notes")
sc = {ln[0]: ln for ln in curves_of(ax_b)}
check(sorted(sc) == ["in-range $R^2$", "rollout score"]
      and all(list(ln[1]) == [float(e) for e in range(2, 25, 2)]
              for ln in sc.values())
      and sc["rollout score"][4] == S.TRAIN_COLORS["score"]
      and sc["in-range $R^2$"][4] == S.TRAIN_COLORS["in_range"]
      and not any(ln[3] == "--" and list(ln[2]) == [0.8, 0.8]
                  for ln in ax_b["lines"]),
      "(b) the score and the in-range R2 (up to twice the training horizon) "
      "at the validated epochs only, and no R2 = 0.80 line")
sv = curves_of(ax_c)
check(len(sv) == 1 and np.allclose(sv[0][2], 100.0 * sv[0][1])
      and sv[0][4] == S.TRAIN_COLORS["survival"]
      and ax_c["ylim"][0] == 0.0 and ax_c["ylim"][1] > 2400.0,
      "(c) the survival time of the validated epochs, from 0")
check(len(set(S.TRAIN_COLORS.values())) == 5
      and ax_a["legend"] == ["training loss", "validation loss",
                             "selected epoch", "diverged epoch"]
      and not any(x.startswith("threshold") for x in ax_b["legend"])
      and ax_c["legend"] == ["survival time", "seed range"]
      and any("diverged epoch 9, loss 2.6e+04, not drawn" in x
              for x in TN["tc.png"]),
      "five colours, one for each series, and a key above each panel")
check(not fig["texts"]
      and "training curves of train_t0500" in TN["tc.png"][0]
      and "1,234 parameters" in TN["tc.png"]
      and "selected epoch 20 of 24: rollout score 0.860, in-range R2 0.975, "
          "survives to t = 2000" in TN["tc.png"],
      "the figure has no title; the notes name the model, its parameters "
      "and the selected epoch with its numbers")
quiet(S.figure_training_curves, hist_rows(), os.path.join(BOX11, "t2.png"),
      "Cahn-Hilliard FNO", "train_t0500", None, None, None, TN)
f2 = seen("t2.png", BOX11)
check(not f2.get("missing")
      and not [ln for a in f2["axes"] for ln in a["lines"]
               if ln[5] in ("x", "o")]
      and not [a for a in f2["axes"] if len(verticals(a, 20.0))]
      and "24 epochs, the selected one is not marked" in TN["t2.png"]
      and not any("selected" in t or "diverged" in t
                  for a in f2["axes"] for t in a["legend"])
      and not any("parameters" in t for t in TN["t2.png"])
      and f2["axes"][1]["legend"][0] == "rollout score"
      and f2["axes"][1]["legend"][1] == "in-range $R^2$",
      "with no selected epoch, no blown-up epoch, no parameter count and no "
      "horizon: no marker, no line, no cross, and the figure still draws")
quiet(S.figure_training_curves, hist_rows(validated=False),
      os.path.join(BOX11, "t3.png"), "Cahn-Hilliard FNO", "train_t0500", 5,
      None, 500.0)
f3 = seen("t3.png", BOX11)
check(not f3.get("missing") and len(f3["axes"]) == 3
      and not curves_of(f3["axes"][1]) and not curves_of(f3["axes"][2])
      and len(curves_of(f3["axes"][0])) == 2,
      "a history that has no rollout validation (only losses) draws its "
      "loss panel and leaves the others empty, instead of failing")

print("\n   reading the history")
H11 = os.path.join(BOX11, "h")
os.makedirs(H11)
hist_file = os.path.join(H11, "fno_ch_history.json")
with open(hist_file, "w") as fh:
    json.dump([{"epoch": 3, "train": 3.0}, "junk", {"epoch": 1, "train": 1.0},
               {"no": "epoch"}, {"epoch": True}, {"epoch": 2, "train": 2.0}],
              fh)
got11 = S.read_history(hist_file)
check(got11 is not None and [r["epoch"] for r in got11] == [1, 2, 3]
      and S.read_history(os.path.join(H11, "absent.json")) is None,
      "read_history: the epochs in order, junk entries dropped; a file that "
      "is not there is None")
for bad_text in ('{"epoch": 1}', "[", "[]", '["x"]'):
    with open(hist_file, "w") as fh:
        fh.write(bad_text)
    check(S.read_history(hist_file) is None,
          f"and so is a file holding {bad_text}")
CK_BESIDE = os.path.join(H11, "fno_ch.pt")        # hist_file is beside it
CK_ALONE = os.path.join(BOX11, "alone", "fno_ch.pt")   # nothing beside it
W11 = os.path.join(BOX11, "W")
os.makedirs(os.path.join(W11, "models", "train_t0300"))
by_label = fno_paths.history_path("train_t0300", W11)
with open(by_label, "w") as fh:
    fh.write("[]")
LAB11 = "train_t0300_pred_t01000"
check(S.history_file(LAB11, CK_ALONE, W11) == by_label
      and S.history_file(LAB11, None, W11) == by_label
      and S.history_file(LAB11, "", W11) == by_label
      and S.history_file(LAB11, CK_ALONE, os.path.join(BOX11, "else")) is None
      and S.history_file("not_a_label", CK_ALONE, W11) is None
      and S.history_file("not_a_label", None, W11) is None,
      "history_file: with no history beside the checkpoint (or none "
      "recorded) the model's folder under the root given, by the label's "
      "train label; nothing for another root, or for a label that is not one")
check(S.history_file(LAB11, CK_BESIDE, W11) == hist_file
      and S.history_file("not_a_label", CK_BESIDE, None) == hist_file,
      "the history beside the checkpoint comes first, whatever the label")
REP11 = os.path.join(BOX11, "rep")
os.makedirs(REP11)
with open(os.path.join(REP11, "report.txt"), "w") as fh:
    fh.write("  1  TRAINED MODEL\n\n  checkpoint selection\n"
             "    accuracy                 0.98\n"
             "    epoch                    58\n"
             "    score                    0.97067502617836\n"
             "    survival                 9514.77\n\n"
             "  training configuration\n    SEED  3\n")
check(S.report_selection(REP11) == {"accuracy": "0.98", "epoch": "58",
                                    "score": "0.97067502617836",
                                    "survival": "9514.77"}
      and S.selected_checkpoint(REP11, {}) == {
          "epoch": 58, "score": 0.97067502617836, "survival": 9514.77}
      and S.selected_checkpoint(REP11, {"checkpoint_best": {
          "epoch": 12, "score": 0.5}}) == {"epoch": 12, "score": 0.5}
      and S.selected_checkpoint(os.path.join(BOX11, "none"), {}) is None
      and S.report_selection(os.path.join(BOX11, "none")) == {},
      "the selection of an older run is the block of report.txt (stopping "
      "at the next one), and a manifest that has it comes first")
with open(os.path.join(REP11, "report.txt"), "w") as fh:
    fh.write("  checkpoint selection\n    -\n\n  training configuration\n")
check(S.report_selection(REP11) == {}
      and S.selected_checkpoint(REP11, {"checkpoint_best": {
          "epoch": "soon"}}) is None,
      "a block with no entries, or an epoch that is not a number, says "
      "nothing")
plain_best = S.best_record({"epoch": np.int64(5)}) or {}
check(S.best_record({"epoch": np.int64(5), "score": np.float64(0.5),
                     "survival": float("nan"), "per_seed": [1.0],
                     "accuracy": True, "peak_survival": 7}) == {
          "epoch": 5, "score": 0.5, "peak_survival": 7}
      and type(plain_best.get("epoch")) is int
      and S.best_record({}) is None and S.best_record(None) is None,
      "best_record keeps the plain finite numbers of the checkpoint's best "
      "(numpy scalars made plain, so the manifest can be written), else None")


# --- runs made by hand: manifest.json with the curves the overview reads ----
INFO11 = {"psi0p0.00": ("composition", 0.0, 128, 40),
          "psi0m0.40": ("composition", -0.4, 128, 50),
          "l256": ("box", 0.0, 256, 10), "l384": ("box", 0.0, 384, 10)}
GRIDS11 = {"psi0p0.00": [100.0, 1000.0, 10000.0],
           "psi0m0.40": [100.0, 300.0, 3000.0, 10000.0],
           "l256": [100.0, 10000.0], "l384": [100.0, 10000.0]}


def put_run(root, label, slopes, trained, drop=(), cut=None, checkpoint=None,
            at_times=False):
    """A run folder with a manifest.json whose cases have mean R2 curves that
    are exactly linear in log t (1 at t = 100, down by slope * log10(t / 100)
    / 2), each on a time grid of its own; ``drop`` leaves cases out and
    ``cut`` ends the l256 curve there."""
    cases = {}
    for tok, (kind, psi0, lat, n) in INFO11.items():
        if tok in drop or tok not in slopes:
            continue
        grid = GRIDS11[tok] if not (cut and tok == "l256") else [100.0, cut]
        mean = [1.0 - slopes[tok] * np.log10(t / 100.0) / 2.0 for t in grid]
        cases[tok] = {"kind": kind, "token": tok, "lattice": lat,
                      "psi0": psi0, "n_seeds": n,
                      "r2_at_end": {"mean": mean[-1], "sd": 0.01},
                      "r2_curve": {"t": grid, "mean": mean,
                                   "sd": [0.0] * len(grid)}}
        if at_times:
            cases[tok]["r2_at_times"] = {"10000": {
                "mean": mean[-1], "n": n, "diverged": 0, "n_alive": n,
                "mean_alive": mean[-1]}}
    os.makedirs(os.path.join(root, label), exist_ok=True)
    with open(os.path.join(root, label, "manifest.json"), "w") as fh:
        json.dump({"label": label, "train_t_end": trained, "pdf_time": 10000.0,
                   "n_parameters": 1000, "n_diverged": 0,
                   "checkpoint": checkpoint, "cases": cases,
                   "aliases": {}, "files": ["manifest.json"]}, fh)
    return os.path.join(root, label)


def pooled_at(slopes, t, drop=()):
    """The seed-weighted mean of the cases' curves at t, by the formula."""
    w = {k: INFO11[k][3] for k in slopes if k not in drop}
    return 1.0 - sum(w[k] * slopes[k] for k in w) / sum(w.values()) \
        * np.log10(t / 100.0) / 2.0


SL = {"psi0p0.00": 0.2, "psi0m0.40": 0.1, "l256": 0.4}
RES11 = os.path.join(BOX11, "Results")
WK11 = os.path.join(BOX11, "Work")
cfg11 = fno_paths.config_path("train_t1000_lap", WK11)
os.makedirs(os.path.dirname(cfg11))
with open(cfg11, "w") as fh:
    json.dump({"label": "train_t1000_lap",
               "size": {"channels": 38, "modes": 24}}, fh)
SIZES = {   # label: (slopes, drop, cut)
    "train_t1000_lap_pred_t10000": (SL, (), None),
    "train_t1000_lap_s1_pred_t10000": (
        {k: v * 0.5 for k, v in SL.items()}, (), None),
    "train_t1000_lap_c16m8_pred_t10000": (SL, ("l256",), None),
    "train_t1000_lap_c24m8_pred_t10000": (
        {k: v * 0.8 for k, v in SL.items()}, (), None),
    "train_t1000_lap_c16m12_pred_t10000": (
        {k: v * 3.0 for k, v in SL.items()}, (), 3000.0),
    "train_t1000_lap_c64m12_pred_t10000": (
        {k: v * 0.4 for k, v in SL.items()}, (), None)}
for label, (sl, dr, cut) in SIZES.items():
    put_run(RES11, label, sl, 1000.0, dr, cut)
runs11 = [S.run_row(os.path.join(RES11, lab), WK11) for lab in SIZES]
by_lab = {r["label"]: r for r in runs11}
dflt = by_lab["train_t1000_lap_pred_t10000"]
pc = S.pooled_curve(dflt)
check(pc is not None and pc[2] == 3
      and list(pc[0]) == [100.0, 300.0, 1000.0, 3000.0, 10000.0]
      and np.allclose(pc[1], pooled_at(SL, pc[0])),
      "pooled_curve: the cases' curves, each on a grid of its own, read on "
      "the union of the grids and weighted by their seeds (40, 50 and 10): "
      "the formula's values, and the last time is the end of every case")
pc8 = S.pooled_curve(by_lab["train_t1000_lap_c16m8_pred_t10000"])
pc12 = S.pooled_curve(by_lab["train_t1000_lap_c16m12_pred_t10000"])
check(pc8 is not None and pc8[2] == 2
      and np.allclose(pc8[1], pooled_at(SL, pc8[0], ("l256",)))
      and pc12 is not None and pc12[0][-1] == 3000.0
      and list(pc12[0]) == [100.0, 300.0, 1000.0, 3000.0]
      and np.allclose(pc12[1], pooled_at(
          {k: v * 3.0 for k, v in SL.items()}, pc12[0])),
      "a run with fewer cases is pooled over those it has, and one whose "
      "case ends earlier is pooled over the window all its cases cover")
empty = dict(dflt, curves={})
check(S.pooled_curve(empty) is None
      and S.pooled_curve(dict(dflt, curves={"psi0p0.00": None},
                              case_info=dflt["case_info"])) is None,
      "and a run with no curve has no pooled one")
check([S.width_colour(w) for w in S.SIZE_WIDTHS] == list(S.SIZE_RAMP)
      and S.width_colour(8) == S.SIZE_RAMP[0]
      and S.width_colour(128) == S.SIZE_RAMP[-1]
      and S.width_colour(28) not in S.SIZE_RAMP
      and S.width_colour(28) == S.width_colour(28.0),
      "width_colour: the ramp's steps at the widths of the sweep, a width "
      "between two mixed from them, one outside the end step")
ends11 = S.size_ends(runs11)
group = S.size_runs(runs11, 1000.0)
check(ends11 == [1000.0] and len(group) == len(SIZES),
      "the runs of one training horizon with a size or a seed in their "
      "label make a size figure")
SN: dict = {}
quiet(S.figure_size_comparison, group, 1000.0,
      os.path.join(BOX11, "sizes.png"), "Cahn-Hilliard FNO", SN)
fs = seen("sizes.png", BOX11)
if fs.get("missing") or len(fs["axes"]) != 3:
    fs = dict(EMPTY, axes=[EMPTY["axes"][0]] * 3, titles=["", "", ""])
check(not fs.get("missing") and len(fs["axes"]) == 3
      and fs["titles"] == ["(a)", "(b)", "(c)"]
      and len({a["ylim"] for a in fs["axes"]}) == 1
      and fs["axes"][0]["ylim"][0] <= 0.7
      and fs["axes"][0]["ylim"][1] == 1.01
      and all(a["xscale"] == "log" for a in fs["axes"]),
      "one panel for each number of modes, on a shared R2 axis from 0.7 (or "
      "lower, to hold the data) to 1.01, with a log time axis")
p8, p12, p24 = ([ln for ln in curves_of(a)] for a in fs["axes"])
check([ln[4] for ln in p8] == [S.width_colour(16), S.width_colour(24)]
      and [ln[4] for ln in p12] == [S.width_colour(16), S.width_colour(64)]
      and [ln[4] for ln in p24] == [S.INK, S.INK]
      and [ln[3] for ln in p24] == ["-", "--"]
      and all(ln[3] == "-" for ln in p8 + p12),
      "a curve for each width, coloured by the width and the same in every "
      "panel; the default size is black, solid, and the same size trained "
      "with another seed is black dashed")
TAIL = ["training horizon"]
want = [f"8 modes, 16 ch: 1k parameters, pooled R2(10000) = "
        f"{pooled_at(SL, 10000.0, ('l256',)):.3f} (2 of 3 cases)",
        f"8 modes, 24 ch: 1k parameters, pooled R2(10000) = "
        f"{pooled_at({k: v * 0.8 for k, v in SL.items()}, 10000.0):.3f}"]
check(fs["axes"][0]["legend"] == ["16 ch", "24 ch"] + TAIL
      and fs["axes"][1]["legend"] == ["16 ch", "64 ch"] + TAIL
      and fs["axes"][2]["legend"] == ["38 ch (default)",
                                      "38 ch (default), seed 1"] + TAIL
      and SN["sizes.png"][3:5] == want
      and SN["sizes.png"][-2:] == [
          f"24 modes, 38 ch (default): 1k parameters, pooled R2(10000) = "
          f"{pooled_at(SL, 10000.0):.3f}",
          f"24 modes, 38 ch (default), seed 1: 1k parameters, pooled "
          f"R2(10000) = "
          f"{pooled_at({k: v * 0.5 for k, v in SL.items()}, 10000.0):.3f}"],
      "the legends are short (the widths, the horizon); the "
      "notes give each model's parameter count and the pooled R2 at the end "
      "of every curve (its own end, 3000 for the run whose case stops "
      "there), and say when a run has fewer cases")
check(not any(ln[3] == "--" and list(ln[2]) == [0.8, 0.8]
              for a in fs["axes"] for ln in a["lines"])
      and not any("threshold" in x for x in SN["sizes.png"])
      and all(any(ln[3] == ":" and list(ln[1]) == [1000.0, 1000.0]
                  for ln in a["lines"]) for a in fs["axes"])
      and not fs["texts"]
      and "trained to t = 1000" in SN["sizes.png"][0]
      and "pooled over the 3 cases (2 compositions, 1 box size)"
      in SN["sizes.png"][0]
      and "black: the default size (38 channels, 24 modes)"
      in SN["sizes.png"][1]
      and "dashed: another training seed" in SN["sizes.png"][1],
      "no 0.80 line is drawn, the training horizon is marked; the pooling "
      "and what the colours and the line style mean are in the notes")

print("\n   a comparison for each case")
CMP11 = os.path.join(BOX11, "cmp")
os.makedirs(CMP11)
PL = {"train_t0300_pred_t10000": ({"psi0p0.00": 0.5, "psi0m0.40": 0.6}, 300.0),
      "train_t0500_pred_t10000": ({"psi0p0.00": 0.3, "psi0m0.40": 0.2,
                                   "l256": 0.1, "l384": 0.1}, 500.0),
      "train_t1000_lap_pred_t10000": ({"psi0p0.00": 0.1, "psi0m0.40": 0.1,
                                       "l256": 0.05}, 1000.0)}
for label, (sl, tr_) in PL.items():
    put_run(CMP11, label, sl, tr_)
rows_c = [S.run_row(os.path.join(CMP11, lab), WK11) for lab in PL]
CN: dict = {}
made = S.draw_case_comparisons(CMP11, rows_c, "Cahn-Hilliard FNO")
CMPD = os.path.join(CMP11, "comparisons")
check(made == ["psi0m0.40", "l256"]
      and sorted(f for f in files_under(CMP11) if f.endswith(".png"))
      == ["comparisons/l128/comparison_psi0m0.40.png",
          "comparisons/l256/comparison_psi0p0.00.png"]
      and not os.path.exists(os.path.join(CMP11, "comparison_psi0p0.00.png"))
      and not os.path.exists(os.path.join(CMPD, "l128",
                                          "comparison_psi0p0.00.png")),
      f"comparisons/l<size>/comparison_<composition>.png for each case that "
      f"two runs have, in the order of the manifests (l384, which only one "
      f"run has, is left out), but none for the critical quench at l = 128, "
      f"which is comparison.png: {made}")
check(sorted(f for f in files_under(CMPD) if f.endswith(".txt"))
      == ["l128/figure_notes.txt", "l256/figure_notes.txt"],
      "each size's folder has a figure_notes.txt of its own")
CN["comparison_l256.png"] = S.read_notes(os.path.join(
    CMPD, "l256", S.NOTES_FILE))["comparison_psi0p0.00.png"]
CN["comparison_psi0m0.40.png"] = S.read_notes(os.path.join(
    CMPD, "l128", S.NOTES_FILE))["comparison_psi0m0.40.png"]
fc = seen("comparison_psi0p0.00.png", CMP11)
legend_l = fc["axes"][0]["legend"] if not fc.get("missing") else []
check(not fc.get("missing")
      and [ln[4] for ln in fc["axes"][0]["lines"]
           if ln[0].startswith("train_")] == [S.RUN_COLORS[1],
                                              S.RUN_COLORS[2]]
      and not fc["titles"][0] and not fc["texts"]
      and "l = 256" in CN["comparison_l256.png"][0]
      and "critical quench" in CN["comparison_l256.png"][0]
      and "off-critical" not in CN["comparison_l256.png"][0]
      and "10 seeds" in CN["comparison_l256.png"][2]
      and "10 seeds" in CN["comparison_l256.png"][3]
      and legend_l[0].startswith("train_t0") and "seeds" not in legend_l[0],
      "a horizon keeps its colour in every file: the 500 and 1000 runs are "
      "the second and third colours in comparison_l256.png, though the 300 "
      "run has no such case; the legend names the runs, the notes name the "
      "box and each run's own number of seeds")
fp_ = seen("comparison_psi0m0.40.png", CMP11)
check(not fp_.get("missing")
      and len([ln for ln in fp_["axes"][0]["lines"]
               if ln[0].startswith("train_")]) == 3
      and "off-critical quench" in CN["comparison_psi0m0.40.png"][0]
      and "psi0 = -0.4" in CN["comparison_psi0m0.40.png"][0]
      and "l = 128" in CN["comparison_psi0m0.40.png"][0]
      and any("50 seeds" in x for x in CN["comparison_psi0m0.40.png"][2:]),
      "an off-critical case names its composition and the base box, in the "
      "notes")
only = os.path.join(BOX11, "only_crit")
for label, (sl, tr_) in PL.items():
    put_run(only, label, {"psi0p0.00": sl["psi0p0.00"]}, tr_)
rows_o = [S.run_row(os.path.join(only, lab), WK11) for lab in PL]
check(S.draw_case_comparisons(only, rows_o, "x") == []
      and not [f for f in os.listdir(only) if f.endswith(".png")],
      "runs that hold only the critical quench get no per-case file: "
      "comparison.png says it all")
quiet(S.figure_run_comparison, rows_c, os.path.join(CMP11, "comparison.png"),
      "Cahn-Hilliard FNO", None, None, CN)
f0 = seen("comparison.png", CMP11)
check(not f0["titles"][0] and "critical quench" in CN["comparison.png"][0]
      and f0["axes"][0]["legend"][:3] == [
          "train_t0300_pred_t10000", "train_t0500_pred_t10000",
          "train_t1000_lap_pred_t10000"]
      and CN["comparison.png"][2:] == [
          "train_t0300_pred_t10000: trained to t = 300, 40 seeds, "
          "R2(10000) = 0.500",
          "train_t0500_pred_t10000: trained to t = 500, 40 seeds, "
          "R2(10000) = 0.700",
          "train_t1000_lap_pred_t10000: trained to t = 1000, 40 seeds, "
          "R2(10000) = 0.900"],
      "comparison.png: the critical curve of each run, named in the legend; "
      "the critical case's seeds and R2 at the end are in the notes")
# stale temporaries of the new figures, from a job that was killed
stale = ["comparison_psi0m0.40.nodeX-77.partial.png",
         "comparison_t1000.nodeX-77.partial.png",
         "training_curves.nodeX-77.partial.png"]
for name in stale:
    with open(os.path.join(CMP11, name), "wb") as fh:
        fh.write(b"half")
    old_t = time.time() - 172800
    os.utime(os.path.join(CMP11, name), (old_t, old_t))
keep = os.path.join(CMPD, "l128", "comparison_psi0m0.40.png")
S.clear_shared_tmp(CMP11)
check(not [f for f in stale if os.path.exists(os.path.join(CMP11, f))]
      and os.path.exists(keep) and os.path.exists(
          os.path.join(CMP11, "comparison.png")),
      "the temporaries of a per-case figure, a size figure and a training "
      "figure that a dead job left are cleared like the index's, and the "
      "figures themselves are not")
check(S.SHARED_TMP.fullmatch(os.path.basename(
          S.shared_tmp(os.path.join(CMP11, "comparison_l256.png"))))
      and S.SHARED_TMP.fullmatch(os.path.basename(
          S.shared_tmp(os.path.join(CMP11, "training_curves.png"))))
      and not S.SHARED_TMP.fullmatch("comparison_l256.png")
      and not S.SHARED_TMP.fullmatch("training_curves.png"),
      "and what shared_tmp names for them is what the pattern clears")

print("\n   runs with different cases: pooling over the common ones, and "
      "end500_vs_end1000.png")
S.WRITE_TRAINING = False
INFO11.update({"psi0p0.20_l256": ("box", 0.2, 256, 50),
               "l512": ("box", 0.0, 512, 50)})
GRIDS11.update({"psi0p0.20_l256": [100.0, 10000.0],
                "l512": [100.0, 10000.0]})
GRID_RES = os.path.join(BOX11, "grid_results")
OLD_SL = {"psi0p0.00": 0.2, "psi0m0.40": 0.1, "l256": 0.4}
NEW_SL = dict(OLD_SL, **{"psi0p0.20_l256": 0.3, "l512": 0.5})
put_run(GRID_RES, "train_t0300_lap_pred_t10000", OLD_SL, 300.0,
        at_times=True)
put_run(GRID_RES, "train_t0500_lap_pred_t10000", NEW_SL, 500.0,
        at_times=True)
put_run(GRID_RES, "train_t1000_lap_pred_t10000",
        {k: v * 0.5 for k, v in NEW_SL.items()}, 1000.0, at_times=True)
rows_g = [S.run_row(os.path.join(GRID_RES, lab), WK11) for lab in (
    "train_t0300_lap_pred_t10000", "train_t0500_lap_pred_t10000",
    "train_t1000_lap_pred_t10000")]
common = S.common_tokens(rows_g)
check(common == ["psi0p0.00", "psi0m0.40", "l256"]
      and S.common_tokens(rows_g[1:]) == list(NEW_SL)
      and S.common_tokens([]) == [],
      f"common_tokens: the cases every run has, in the first run's order: "
      f"{common}")
whole = S.pooled_curve(rows_g[1])
only_common = S.pooled_curve(S.restrict_row(rows_g[1], common))
check(whole is not None and only_common is not None
      and whole[2] == 5 and only_common[2] == 3
      and np.allclose(only_common[1], pooled_at(OLD_SL, only_common[0]))
      and not np.allclose(whole[1][-1], only_common[1][-1])
      and S.restrict_row(rows_g[1], common)["curves"].keys() == set(common)
      and len(rows_g[1]["cases"]) == 5,
      "restrict_row: the pooled curve of a run with more cases covers the "
      "common ones alone, by the formula; the run itself is not changed")
r_old = S.pooled_r2_end(rows_g[0])
r_new = S.pooled_r2_end(S.restrict_row(rows_g[1], common))
r_all = S.pooled_r2_end(rows_g[1])
check(r_old is not None and r_new is not None and r_all is not None
      and np.isclose(r_new, r_old, atol=1e-12)
      and not np.isclose(r_all, r_old),
      "and so does its pooled R2 at the end: the same as the run that has "
      "only those cases")
# a results root: the figures that pool over cases, and the new one
quiet(S.write_results_index, GRID_RES, WK11)
pairs = S.end_pair(rows_g)
check(pairs is not None and [r["parsed"]["t_train"] for r in pairs]
      == [500.0, 1000.0]
      and S.end_pair(rows_g[:2]) is None
      and S.end_pair([rows_g[0], rows_g[2]]) is None,
      "end_pair finds the plain Laplacian runs trained to 500 and to 1000, "
      "and None when either is missing")
endfig = os.path.join(GRID_RES, S.END_FIG)
fe = seen(S.END_FIG, GRID_RES)
check(os.path.isfile(endfig) and not fe.get("missing")
      and len(fe["axes"]) == 3 and all(a["xscale"] == "log"
                                       for a in fe["axes"])
      and not fe["titles"][0].startswith("end500")
      and not any("0.80" in x or "threshold" in x for x in fe["texts"]),
      "end500_vs_end1000.png: one panel for each composition both runs "
      "have (-0.4, 0 and +0.2), a log2 box axis, no threshold line")
NE = S.read_notes(os.path.join(GRID_RES, S.NOTES_FILE))
nl = NE.get(S.END_FIG, [])
check(nl and "psi0 = +0.0, trained to 500: l = 128: " in " ".join(nl)
      and "l = 512" in " ".join(nl) and "l = 256" in " ".join(nl)
      and any(x.startswith("psi0 = +0.2, trained to 1000: l = 256:")
              and "l = 128" not in x for x in nl)
      and "50 seeds" in " ".join(nl),
      "its notes give the seed mean, the standard error and the seeds at "
      "each size, and only the sizes both runs have")
idx_g = text(os.path.join(GRID_RES, "README.md"))
check(S.END_FIG in idx_g and "comparisons/l<size>/" in idx_g
      and "every run in the figure has" in idx_g,
      "the index lists the figure and says how the pooled figures and the "
      "per-case comparisons are laid out")
check(sorted(f for f in files_under(GRID_RES)
             if f.startswith("comparisons/") and f.endswith(".png"))
      == ["comparisons/l128/comparison_psi0m0.40.png",
          "comparisons/l256/comparison_psi0p0.00.png",
          "comparisons/l256/comparison_psi0p0.20.png",
          "comparisons/l512/comparison_psi0p0.00.png"]
      and not [f for f in os.listdir(GRID_RES)
               if f.startswith("comparison_") and f != "comparison_t1000.png"],
      "the per-case comparisons are in comparisons/l<size>/, named by the "
      "composition, none at the top")
for lab in ("train_t0500_lap_pred_t10000", "train_t1000_lap_pred_t10000"):
    mm = jload(os.path.join(GRID_RES, lab, "manifest.json"))
    mm["cases"] = {k: v for k, v in mm["cases"].items()
                   if k in ("psi0p0.00", "psi0m0.40")}
    with open(os.path.join(GRID_RES, lab, "manifest.json"), "w") as fh:
        json.dump(mm, fh)
os.remove(endfig)
quiet(S.write_results_index, GRID_RES, WK11)
fe2 = seen(S.END_FIG, GRID_RES)
check(os.path.isfile(endfig) and len(fe2["axes"]) == 2,
      "the same two runs with only the base size: one point for each model "
      "in the panels of the compositions they share")

# the pooled R2 at the end of a run, in the manifest and the report
def flat_stub(tok, lat, n, mean):
    return {"token": tok, "lattice": lat, "n_seeds": n, "psi0": 0.0,
            "r2_at_end": {"mean": mean}}


eight = {t: flat_stub(t, 128 if t.startswith("psi0") else int(t[1:]), 50,
                     0.9 + 0.01 * i) for i, t in enumerate(S.COMMON_CASES)}
pg = S.pooled_groups(eight)
check(pg["common_6_cases"] is not None and pg["common_6_cases"]["cases"] == 6
      and np.isclose(pg["common_6_cases"]["mean"],
                     np.mean([0.9 + 0.01 * i for i in range(6)]))
      and pg["all_cases"]["seeds"] == 300
      and sorted(pg["per_size"]) == ["l128", "l256"]
      and pg["per_size"]["l128"]["cases"] == 5,
      "pooled_groups: the common 6 cases, all the cases and each size")
more = dict(eight, **{"psi0p0.20_l256": flat_stub("psi0p0.20_l256", 256, 10,
                                                 0.5)})
pm = S.pooled_groups(more)
less = {t: c for t, c in eight.items() if t != "l256"}
assert pm["common_6_cases"] is not None and pg["common_6_cases"] is not None
assert pm["all_cases"] is not None
check(np.isclose(pm["common_6_cases"]["mean"], pg["common_6_cases"]["mean"])
      and pm["all_cases"]["cases"] == 7
      and pm["per_size"]["l256"]["seeds"] == 60
      and S.pooled_groups(less)["common_6_cases"] is None
      and S.pooled_groups({})["all_cases"] is None,
      "the headline stays the common 6 when a run has more cases, and is None "
      "when it lacks one")
pm1 = jload(os.path.join(RUN1, "manifest.json")).get("pooled_r2_at_end") or {}
check(pm1.get("common_6_cases") is None and pm1.get("all_cases")
      and pm1["all_cases"]["cases"] == 5 and f"l{LAT}" in pm1["per_size"],
      "a run's manifest.json has the pooled R2 at the end (common 6 cases "
      "only when it has all of them), over all cases, and per size")
rep_all = text(os.path.join(RUN1, "report.txt"))
check("not all six are in this run" in rep_all
      and "end-of-rollout R2 pooled" in rep_all,
      "and so does its report")

print("\n12. the speed-up line of a part-cached run")
tm_solved = {"n_seeds": 2, "gt_cached": 0, "gt_seconds": 100.0,
             "fno_seconds": 20.0, "fno_steps": 10}
tm_solved2 = {"n_seeds": 2, "gt_cached": 0, "gt_seconds": 60.0,
              "fno_seconds": 20.0, "fno_steps": 10}
tm_cached = {"n_seeds": 2, "gt_cached": 2, "gt_seconds": 0.0,
             "fno_seconds": 20.0, "fno_steps": 10}
tm_half = {"n_seeds": 2, "gt_cached": 1, "gt_seconds": 50.0,
           "fno_seconds": 20.0, "fno_steps": 10}
mixed = S._speedup_text([tm_solved, tm_cached, tm_solved2, tm_half])
check(mixed.startswith("4.0x over 2 of 4 cases"),
      f"a part-cached run: the ratio is over the solved cases only, "
      f"(100 + 60) / (20 + 20), not the all-case sum: {mixed!r}")
check(S._speedup_text([tm_solved]).startswith("5.0x over 1 of 1 cases")
      and S._speedup_text([tm_cached, tm_half]).startswith("not available")
      and S._speedup_text([]).startswith("not available")
      and S._speedup_text([{}, {"n_seeds": 2, "gt_cached": 0}]).startswith(
          "not available"),
      "all solved: over k of k; none solved or no timing: not available")
REP12 = os.path.join(BASE, "report12.txt")
quiet(S.write_report, REP12, {"label": "r", "written": "now", "cases": {},
                              "diverge_abs": 5.0, "threshold": 0.8,
                              "checkpoint": "m.pt",
                              "checkpoint_fingerprint": "abc"}, {})
rep_t = text(REP12)
check([ln for ln in rep_t.splitlines() if ln.strip().startswith("speedup")][
    0].split(None, 1)[1].startswith("not available"),
      "report.txt prints the line in its SOLVE TIME section")


# the step schedule section (5): an edge pick with diverged candidates, a
# pick without a table, and a run without picks
GRID_A, GRID_D = (0.1, 0.2, 0.35), (30.0, 40.0)
TAB = [{"alpha": a, "dt_max": d, "steps": 80, "score": 0.9,
        "n_diverged": 3 if (a, d) in ((0.1, 30.0), (0.2, 30.0)) else 0}
       for a in GRID_A for d in GRID_D]
PICK_E = {"alpha": 0.1, "dt_max": 30.0, "dt_min": 1.0,
          "source": "tuned on 10 validation seeds", "score": 0.9123,
          "signature": repr(("tuned", 100.0, 10000.0, GRID_A, GRID_D, (1, 2))),
          "table": TAB}
PICK_N = {"alpha": 0.2, "dt_max": 40.0, "dt_min": 1.0,
          "source": "SCHED_ALPHA / SCHED_DT_MAX", "score": None,
          "signature": repr(("fixed", 0.2, 40.0)), "table": []}


def report_picks(picks):
    path = os.path.join(BASE, "report_picks.txt")
    quiet(S.write_report, path, {"label": "r", "written": "now", "cases": {},
                                 "diverge_abs": 5.0, "threshold": 0.8,
                                 "checkpoint": "m.pt",
                                 "checkpoint_fingerprint": "abc"}, {}, None,
          picks)
    return text(path)


rep_e = report_picks({"-0.4": PICK_E, "+0.2": PICK_N})
heads_e = [ln.strip() for ln in rep_e.splitlines() if re.match(r"\s+\d  [A-Z]", ln)]
check(heads_e[4:8] == ["5  STEP SCHEDULE", "6  DIAGNOSTICS",
                       "7  LOW-k INCREMENT RESPONSE", "8  CONVENTIONS"],
      f"report.txt has the sections 5 to 8 in order: {heads_e}")
lines_e = rep_e.splitlines()
idx_e = [i for i, ln in enumerate(lines_e) if "psi0 -0.4" in ln]
line_e = [lines_e[i] for i in idx_e]
next_e = (lines_e[idx_e[0] + 1] if len(idx_e) == 1
          and idx_e[0] + 1 < len(lines_e) else "")
check(len(line_e) == 1 and "alpha 0.1, dt_max 30, 80 steps, score 0.9123" in
      line_e[0] and "tuned on 10 validation seeds" in line_e[0]
      and "2 of 6 candidates had diverged tuning seeds" in next_e
      and "dt_max smallest" in next_e and "alpha smallest" in next_e
      and next_e.startswith(" " * 13) and len(next_e) <= 110,
      f"an edge pick with diverged candidates, on two lines: {line_e} / "
      f"{next_e!r}")
check(any("psi0 +0.2" in ln and "fixed by hand" in ln
          for ln in rep_e.splitlines())
      and "a smaller step was not tried" in " ".join(rep_e.split())
      and 'key "schedule"' in " ".join(rep_e.split()),
      "a pick without a table says so, and the two-line note is there")
rep_0 = report_picks({})
check("no step schedule was picked: a fixed-step run" in rep_0
      and "8  CONVENTIONS" in rep_0,
      "a run without picks gets one line")
old_pick = {k: v for k, v in PICK_E.items() if k != "table"}
check("an old meta.json with no candidate table" in
      report_picks({"+0.0": old_pick}),
      "an old pick with no table gets its line")

S.WRITE_TRAINING = False
setup()
check(project_state() == PROJECT,
      "nothing was written into the project, outside _scratch")
print("\nALL PASSED" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
