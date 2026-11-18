"""Exercise main()'s wiring, and the folder a run leaves, with the model and
the rollouts stubbed out.

One run writes two places.  The run folder (Results/<label>/) is flat:
<kind>_<case token>.png for every case, composition.csv,
composition_all.png, r2_sweep.png/.pdf, manifest.json and report.txt.  FNO's
own outputs -- per case the case figures, the averages over every seed and the
best seed's own folder, and for the run the timing report, the sweep curves
and the bookkeeping manifest -- are in extras/ of the rollout cache
(Work/rollout_cache/<label>/extras/<case>/), and nothing per seed beside the
best seed.  Every rollout here is synthetic, so the best seed is known before
the code picks it.
"""
import contextlib
import io as _io
import json
import os
import shutil
import sys
import types

import matplotlib.figure as mfig
import numpy as np

from _paths import REAL_MODEL, ROOT, SCRATCH, project_state

if REAL_MODEL is None:
    sys.exit("no trained checkpoint to run against: put one at "
             "Work/models/train_t0500/fno_ch.pt (see REAL_MODEL in _paths.py)")

os.chdir(ROOT)
PROJECT = project_state()
import fno_paths
import fno_predict as P
# records and meta.json (t_records.py) stay in the cache's folder, which
# the checks below expect to be removed once it is empty
P.RECORDS = False
import make_submission as S

TMP = os.path.join(SCRATCH, "layout_results")
RERUN = os.path.join(SCRATCH, "layout_rerun")
CACHE = os.path.join(SCRATCH, "layout_cache")
for _d in (TMP, TMP + "_nohard", RERUN, CACHE):
    shutil.rmtree(_d, ignore_errors=True)
P.MODEL_PATH = REAL_MODEL
P.RESULTS_DIR = TMP
P.RESULT_CACHE = CACHE
P.GT_CACHE = os.path.join(SCRATCH, "layout_gt_cache")
P.WRITE_TIMING = False
P.WRITE_SUBMISSION = False
P.WRITE_LOG = False
FLAGS = dict(hard_constraints=True, hard_mass_conservation=True,
             zero_dc=True, label="fno_ch", note="")
P.MODEL_FLAGS.update(FLAGS)

fails = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        fails.append(msg)


# Every figure's titles, by the path it was saved to: how a check reads which
# seed a figure shows without looking at its pixels.
TITLES = {}
_savefig = mfig.Figure.savefig


def _remember(self, fname, *args, **kwargs):
    if isinstance(fname, str):
        texts = [t.get_text() for t in self.texts]
        texts += [ax.get_title() for ax in self.axes]
        TITLES[os.path.normcase(os.path.abspath(fname))] = "\n".join(texts)
    return _savefig(self, fname, *args, **kwargs)


setattr(mfig.Figure, "savefig", _remember)

# Which rollout each kinetics figure is drawn from, by the path it is saved
# to: the title names a seed, this says whose data is behind it.
KINETICS = {}
_kinetics = P.figure_kinetics


def _kinetics_spy(results, seeds, path, *args, **kwargs):
    key = os.path.normcase(os.path.abspath(path))
    KINETICS[key] = results
    got = _kinetics(results, seeds, path, *args, **kwargs)
    # the text of the figure is in its notes now (figure_notes.txt)
    for line in (kwargs.get("notes") or {}).get(os.path.basename(path), ()):
        TITLES[key] = TITLES.get(key, "") + "\n" + line
    return got


P.figure_kinetics = _kinetics_spy


def titles(path):
    return TITLES.get(os.path.normcase(os.path.abspath(path)), "")


fake_ck = {"best": {"epoch": 55, "score": 0.93, "survival": 2000.0}}


class FakeModel:
    config = {"backbone": "physical_k", "hard_mass_conservation": True,
              "residual_prediction": True, "mode_scaling": 1.0,
              "zero_dc": True}

    def n_parameters(self):
        return 0

    def parameters(self):
        yield types.SimpleNamespace(dtype="float32")


P.load_checkpoint = lambda path, device=None: (FakeModel(), fake_ck)
# main() asks checkpoint_conditioned() which engine the model needs, and that
# reads the file REAL_MODEL names -- a dt-conditioned model since 2026-10-26,
# which would send every run here to the schedule engine (no film, the other
# test seeds).  The model this suite stubs is a fixed-step one, so say so.
P.checkpoint_conditioned = lambda path: bool(
    (fake_ck.get("config") or {}).get("dt_conditioned", False))

rng = np.random.default_rng(0)


def fake_seed(lat, off, level, diverge_at=None, stop_at=None, drift=1e-9,
              frames=0):
    """One synthetic rollout, shaped as evaluate_seed returns it.

    A divergence scores the rest of the window at the floor and leaves NaN
    predictions, as evaluate_seed does; stop_at cuts the series short, as
    EARLY_STOP would.
    """
    t = np.arange(P.START_TIME + 1, P.END_TIME + 1)
    r2 = np.clip(level - (t - t[0]) / 9000.0, -1, 1)
    mean_pred = np.full(t.size, off) + drift
    snaps = {tt: (rng.standard_normal((lat, lat)),
                  rng.standard_normal((lat, lat)))
             for tt in P.SNAPSHOT_TIMES}
    if diverge_at is not None:
        r2 = np.where(t >= diverge_at, P.R2_FLOOR, r2)
        mean_pred = np.where(t >= diverge_at, np.nan, mean_pred)
        snaps = {tt: (a, np.full((lat, lat), np.nan) if tt >= diverge_at
                      else b) for tt, (a, b) in snaps.items()}
    if stop_at is not None:
        keep = t <= stop_at
        t, r2, mean_pred = t[keep], r2[keep], mean_pred[keep]
        snaps = {tt: v for tt, v in snaps.items() if tt <= stop_at}
    dt = t[::100]
    res = {
        "times": t, "r2": r2,
        "mass_drift": np.full(t.size, drift),
        "mean_true": np.full(t.size, off), "mean_pred": mean_pred,
        "diag_times": dt,
        "energy_true": np.linspace(1, 0.5, dt.size),
        "energy_pred": np.linspace(1, 0.5, dt.size),
        "t_stop": float(t[-1] if stop_at is None else stop_at),
        "stopped_early": bool(stop_at is not None or r2[-1] < 0.8),
        "diverged_at": diverge_at,
        "L_true": {k: np.linspace(4, 20, dt.size)
                   for k in ("L_zero", "L_peak", "L_moment", "L_grad")},
        "L_pred": {k: np.linspace(4, 20, dt.size)
                   for k in ("L_zero", "L_peak", "L_moment", "L_grad")},
        "snapshots": snaps,
        "gt_seconds": 1.0, "gt_cached": False, "gt_steps": 1,
        "fno_seconds": 0.5, "fno_steps": 1900,
        "video_t": np.asarray([]), "video_true": None, "video_pred": None}
    if frames:
        res["video_t"] = np.linspace(t[0], t[-1], frames)
        res["video_true"] = rng.standard_normal(
            (frames, lat, lat)).astype(np.float32)
        res["video_pred"] = rng.standard_normal(
            (frames, lat, lat)).astype(np.float32)
    return res


def expect(results, seeds):
    """What the case must show, worked out here: the best seed by
    make_submission's rule, the top mean, the rank and the scalar stats."""
    means = np.array([float(np.nanmean(r["r2"])) for r in results])
    best = int(S.best_index(results))
    n_max = max(len(r["r2"]) for r in results)
    stack = np.full((len(results), n_max), np.nan)
    for i, r in enumerate(results):
        stack[i, :len(r["r2"])] = r["r2"]
    def across(vals):
        v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
        return {"mean": float(v.mean()) if v.size else None,
                "std": (float(v.std(ddof=1)) if v.size > 1
                        else 0.0 if v.size else None),
                "n": int(v.size)}

    return {"seeds": tuple(seeds), "best": int(seeds[best]),
            "best_result": results[best], "results": results,
            "growth_truth": across([P.fit_growth_exponent(
                r["diag_times"], r["L_true"][P.LENGTH_KEY]) for r in results]),
            "growth_fno": across([P.fit_growth_exponent(
                r["diag_times"], r["L_pred"][P.LENGTH_KEY]) for r in results]),
            "final_r2": across([float(r["r2"][-1]) for r in results]),
            "r2_mean": np.nanmean(stack, axis=0),
            "r2_std": np.nanstd(stack, axis=0),
            "argmax": int(seeds[int(np.argmax(means))]),
            "rank": int(np.sum(means > means[best])) + 1,
            "worst": int(seeds[int(np.argmin(means))]),
            "means": means,
            "n_diverged": sum(r.get("diverged_at") is not None
                              for r in results),
            "n_held": sum(not r["stopped_early"] for r in results),
            "lengths": [len(r["r2"]) for r in results],
            "max_drift": max(float(r["mass_drift"].max()) for r in results),
            "max_error": max(float(np.nanmax(np.abs(r["mean_pred"]
                                                    - r["mean_true"])))
                             for r in results)}


calls = []
EXPECT = {}
CASES = {}
FRAMES = [0]


def fake_run_case(model, device, off, lattice=None, seeds=None):
    """One synthetic case, written through the real write_case_outputs.

    The seeds are ranked in a shuffled order, so the best is not simply the
    first; at psi0 = +0.40 the seed with the top mean R2 diverges near the
    end, so the best seed is the runner-up.
    """
    calls.append((off, lattice))
    lat = P.LATTICE if lattice is None else lattice
    seeds = P.seeds_for(off) if seeds is None else tuple(seeds)
    order = rng.permutation(len(seeds))
    results = [fake_seed(lat, off, 1.0 - 0.03 * order[i],
                         diverge_at=(1990.0 if off > 0.3 and order[i] == 0
                                     else None),
                         frames=FRAMES[0])
               for i in range(len(seeds))]
    name = os.path.basename(P.case_dir(off, lattice))
    EXPECT[name] = expect(results, seeds)
    case = P.write_case_outputs(results, seeds, off, lattice)
    CASES[name] = {"times": case["times"], "mean_curve": case["mean_curve"],
                   "stats": case["stats"]}
    return case


P.run_case = fake_run_case

SAVED = {k: getattr(P, k) for k in
         ("EVAL_MODE", "SIZE_SWEEP", "EVAL_SEEDS", "SWEEP_SEEDS",
          "WRITE_VIDEO", "WRITE_TIMING", "WRITE_SUBMISSION",
          "PER_SEED_FIGURES", "PER_SEED_OUTPUT", "RESULT_CACHE", "GT_CACHE",
          "KEEP_CACHE", "REDO", "RESULTS_TEMPLATE")}


def restore(results_dir):
    for k, v in SAVED.items():
        setattr(P, k, v)
    P.RESULTS_DIR = results_dir
    P.MODEL_FLAGS.update(FLAGS)
    FakeModel.config = dict(FakeModel.config, hard_mass_conservation=True)
    calls.clear()
    EXPECT.clear()
    CASES.clear()


def run_main(argv):
    """main(), its log captured; any line saying something failed is shown."""
    with contextlib.redirect_stdout(_io.StringIO()) as log:
        code = P.main(argv)
    text = log.getvalue()
    for ln in text.splitlines():
        if "[error]" in ln or "FAILED" in ln:
            print(f"   | {ln}")
    return code, text


def listing(folder):
    return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


def subdirs(folder):
    return [d for d in listing(folder)
            if os.path.isdir(os.path.join(folder, d))]


def files_in(folder):
    return [f for f in listing(folder)
            if os.path.isfile(os.path.join(folder, f))]


def files_rec(folder):
    """Every file below folder, as a path from it."""
    return sorted(os.path.relpath(os.path.join(b_, f), folder).replace(
        os.sep, "/") for b_, _, fs in os.walk(folder) for f in fs)


def case_files(suffix, label="fno_ch"):
    """The files write_case_outputs leaves in a case's extras folder,
    submission off (the case's kinetics figure is drawn flat, by
    make_submission)."""
    return sorted([f"{label}{suffix}_r2_and_snapshots.png",
                   f"{label}{suffix}_conservation.png",
                   f"{label}{suffix}_summary.json", P.MEAN_R2_FILE])


def read_mean_r2(folder):
    path = os.path.join(folder, P.MEAN_R2_FILE)
    with open(path) as fh:
        head = fh.readline().strip()
    return head, np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def read_summary(folder, suffix, label="fno_ch"):
    return read_json(os.path.join(folder, f"{label}{suffix}_summary.json"))


SCALARS = ("off", "lattice", "directory", "hard_constraints",
           "hard_mass_conservation", "zero_dc", "n_seeds",
           "mean_r2_all_seeds", "mean_r2_per_seed_mean",
           "mean_r2_per_seed_std", "mean_r2_standard_error",
           "mean_r2_worst_seed", "mean_r2_best_seed", "worst_seed",
           "best_seed", "best_seed_rank", "steps_above_threshold_total",
           "steps_above_threshold_mean", "seeds_holding_throughout",
           "seeds_diverged", "max_mass_drift", "max_mean_error", "timing",
           "record")


def kinetics_of(size, stem):
    """(titles, rollouts) of a case's phase_ordering_kinetics figure:
    make_submission draws it in a stage folder of the box size's folder and
    publishes it by rename, so it is looked up by the stage's path."""
    name = S.fig_name("phase_ordering_kinetics", stem)
    for path, res in KINETICS.items():
        stage = os.path.dirname(path)
        if (os.path.basename(path) == name
                and os.path.basename(stage).startswith(".partial")
                and os.path.basename(os.path.dirname(stage)) == size):
            return titles(path), res
    return "", None


def check_case(folder, suffix, exp, case, figures=True):
    """Everything one case folder must hold, and nothing it must not."""
    d = os.path.basename(folder)
    best = exp["best"]
    seed_dir = P.BEST_SEED_DIR.format(seed=best)
    dirs = subdirs(folder)
    check(dirs == [seed_dir],
          f"{d}: exactly one seed folder, {seed_dir}/ of the best seed by "
          f"make_submission.best_index: {dirs}")
    files = files_in(folder)
    check(files == case_files(suffix),
          f"{d}: the case figures, summary and {P.MEAN_R2_FILE} and nothing "
          f"else: {files}")
    check(not any("seed" in f for f in files),
          f"{d}: no per-seed files loose in the case folder")
    check(not any("all_seeds" in f or "seed_matrix" in f for f in files),
          f"{d}: no r2_all_seeds or seed_matrix figure")

    st = read_summary(folder, suffix)
    check("seeds" not in st, f"{d}: the summary has no per-seed 'seeds' list")
    miss = [k for k in SCALARS if k not in st]
    check(not miss, f"{d}: the summary keeps its scalar stats (missing "
                    f"{miss})")

    def close(a, b):
        return (a.get("n") == b["n"]
                and all((a.get(k) is None) == (b[k] is None)
                        and (b[k] is None or abs(a[k] - b[k]) < 1e-12)
                        for k in ("mean", "std")))
    check(all(isinstance(st.get(k), dict) and close(st[k], exp[e])
              for k, e in (("growth_exponent_truth", "growth_truth"),
                           ("growth_exponent_fno", "growth_fno"),
                           ("final_r2", "final_r2"))),
          f"{d}: the growth exponents and final R2 are averaged over the "
          f"seeds, as worked out here ({st.get('growth_exponent_truth')}, "
          f"{exp['growth_truth']})")
    n = len(exp["seeds"])
    means = exp["means"]
    check(st.get("n_seeds") == n and st.get("best_seed") == best
          and st.get("best_seed_rank") == exp["rank"]
          and st.get("worst_seed") == exp["worst"],
          f"{d}: n_seeds {st.get('n_seeds')}, best seed "
          f"{st.get('best_seed')} rank {st.get('best_seed_rank')}, worst "
          f"{st.get('worst_seed')} (expected {n}, {best} rank {exp['rank']}, "
          f"{exp['worst']})")
    idx = exp["seeds"].index(best)
    check(abs(st.get("mean_r2_best_seed", np.nan) - means[idx]) < 1e-12
          and abs(st.get("mean_r2_per_seed_mean", np.nan)
                  - float(means.mean())) < 1e-12
          and abs(st.get("mean_r2_worst_seed", np.nan)
                  - float(means.min())) < 1e-12,
          f"{d}: best, mean and worst per-seed mean R2 match the seeds")
    check(st.get("seeds_diverged") == exp["n_diverged"]
          and st.get("seeds_holding_throughout") == exp["n_held"],
          f"{d}: {st.get('seeds_diverged')} diverged, "
          f"{st.get('seeds_holding_throughout')} held (expected "
          f"{exp['n_diverged']}, {exp['n_held']})")
    check(st.get("max_mass_drift") == exp["max_drift"]
          and st.get("max_mean_error") == exp["max_error"],
          f"{d}: max_mass_drift {st.get('max_mass_drift')} and "
          f"max_mean_error {st.get('max_mean_error')} are the max over seeds "
          f"({exp['max_drift']}, {exp['max_error']})")
    rec = st.get("record") or {}
    check(rec.get("seeds") == n
          and rec.get("seed_digest") == P._seed_digest(exp["seeds"])
          and rec.get("complete") is True,
          f"{d}: the record names the {n} seeds and says complete")
    check(st == json.loads(json.dumps(case["stats"])),
          f"{d}: the summary on disk is the case's stats")

    head, data = read_mean_r2(folder)
    check(head == "t,n_seeds,r2_mean,r2_std,mass_drift_mean",
          f"{d}: {P.MEAN_R2_FILE} header {head!r}")
    mc = np.asarray(case["mean_curve"], dtype=np.float64)
    check(data.shape[0] == mc.size
          and data[:, 2].tobytes() == mc.tobytes(),
          f"{d}: {P.MEAN_R2_FILE} r2_mean is the case's mean_curve bitwise")
    check(data.shape[0] == exp["r2_mean"].size
          and np.array_equal(data[:, 2], exp["r2_mean"])
          and np.array_equal(data[:, 3], exp["r2_std"]),
          f"{d}: and both are the mean and spread over the seeds' own R2 "
          f"curves, worked out here")
    check(data.shape[0] == mc.size
          and data[:, 0].tobytes()
          == np.asarray(case["times"], dtype=np.float64).tobytes(),
          f"{d}: {P.MEAN_R2_FILE} t is the case's times")
    counts = np.array([sum(1 for ln in exp["lengths"] if ln > i)
                       for i in range(data.shape[0])])
    check(np.array_equal(data[:, 1], counts),
          f"{d}: n_seeds column counts the seeds at each step "
          f"({int(data[0, 1])} .. {int(data[-1, 1])}, expected "
          f"{counts[0]} .. {counts[-1]})")

    sub = os.path.join(folder, seed_dir)
    stem = f"fno_ch{suffix}_seed{best}"
    have = listing(sub)
    want = sorted([f"{stem}_r2.csv", f"{stem}_summary.json"]
                  + ([f"{stem}_r2_and_snapshots.png",
                      f"{stem}_kinetics.png",
                      f"{stem}_conservation.png"] if figures else []))
    check(have == want, f"{d}/{seed_dir}: its R2 curve, numbers"
                        + (" and figures, kinetics included" if figures
                           else "") + f", no film: {have}")
    own = read_json(os.path.join(sub, f"{stem}_summary.json"))
    check(own.get("seed") == best,
          f"{d}/{seed_dir}: its summary is seed {own.get('seed')}'s")
    if figures:
        own_path = os.path.join(sub, f"{stem}_kinetics.png")
        check(KINETICS.get(os.path.normcase(os.path.abspath(own_path)))
              == [exp["best_result"]],
              f"{d}/{seed_dir}: and the best seed's own kinetics is drawn "
              f"from that seed's rollout alone")
    r2t = titles(os.path.join(folder, f"fno_ch{suffix}_r2_and_snapshots.png"))
    check(f"seed {best} " in r2t and f"rank {exp['rank']}," in r2t,
          f"{d}: the R2 figure's snapshots are seed {best}, rank "
          f"{exp['rank']}")


print("1. main(): one extras folder per case, the best seed's folder in each")
restore(TMP)
P.SIZE_SWEEP = (128, 512)
P.EVAL_MODE = "both"
P.PER_SEED_FIGURES = True
code, _ = run_main([])
check(code == 0, f"main() succeeded (exit {code})")
check((P.CRITICAL_OFF, 128) not in calls,
      "l=128 at the critical composition is not run a second time")
check(len(calls) == 6,
      f"one call per distinct case, got {len(calls)}: {calls}")
check(P.RESULTS_DIR == TMP, "RESULTS_DIR is as it was after the run")

XR = os.path.join(CACHE, P.EXTRAS_NAME)    # FNO's own outputs of the run
dirs = subdirs(XR)
check(dirs == ["psi0+0.00_l128", "psi0+0.00_l512", "psi0+0.40_l128",
               "psi0+0.40_l512", "psi0-0.40_l128", "psi0-0.40_l512"],
      f"one extras folder per case: {dirs}")

check(listing(TMP) == [],
      f"with the submission off the Results folder gets nothing, and the "
      f"old fno_ch_sweeps.png is gone for good: {listing(TMP)}")
walked = [f for base in (TMP, XR) for _, _, fs in os.walk(base) for f in fs]
check(not any("seed_matrix" in f or "r2_all_seeds" in f for f in walked),
      "no seed-matrix or all-seeds figure anywhere in the run")
check(not any(s.startswith("seed") for d in dirs
              for s in subdirs(os.path.join(XR, d))),
      "no seed<N>/ folder for every seed")

hot = EXPECT.get("psi0+0.40_l128")
check(hot is not None and hot["best"] != hot["argmax"] and hot["rank"] == 2,
      "psi0 = +0.40: the top-mean seed diverged, so the best seed is the "
      "runner-up (rank 2)")
check(any(e["best"] != e["seeds"][0] for e in EXPECT.values()),
      "the best seed is not simply the first seed")

suffixes = {"psi0+0.00_l128": "", "psi0+0.00_l512": "_l512",
            "psi0+0.40_l128": "_off+0.40", "psi0-0.40_l128": "_off-0.40",
            "psi0+0.40_l512": "_off+0.40_l512",
            "psi0-0.40_l512": "_off-0.40_l512"}
for d in dirs:
    if d in EXPECT and d in CASES:
        check_case(os.path.join(XR, d), suffixes.get(d, "?"), EXPECT[d],
                   CASES[d])
    else:
        check(False, f"{d}: a case this run wrote")


print("\n2. a rerun of a case whose best seed changed")
restore(RERUN)
P.PER_SEED_FIGURES = False
seeds5 = P.EVAL_SEEDS[:5]
first = [fake_seed(128, 0.0, lv) for lv in (0.90, 0.92, 0.99, 0.91, 0.93)]
with contextlib.redirect_stdout(_io.StringIO()):
    P.write_case_outputs(first, seeds5, 0.0)
folder = P.case_dir(0.0)
old_best = P.BEST_SEED_DIR.format(seed=seeds5[2])
check(subdirs(folder) == [old_best],
      f"the first run keeps {old_best}/: {subdirs(folder)}")
legacy = os.path.join(folder, f"seed{seeds5[0]}")
os.makedirs(legacy)
with open(os.path.join(legacy, "old.csv"), "w") as fh:
    fh.write("t,r2\n")

# The second run: seed 0 now has the top mean but diverges late, seed 3 was
# stopped early (EARLY_STOP), seed 4 is the healthy best, seed 1 drifts most.
second = [fake_seed(128, 0.0, 1.00, diverge_at=1990.0),
          fake_seed(128, 0.0, 0.92, drift=3e-7),
          fake_seed(128, 0.0, 0.90),
          fake_seed(128, 0.0, 0.95, stop_at=1500.0),
          fake_seed(128, 0.0, 0.97)]
exp2 = expect(second, seeds5)
check(exp2["best"] == seeds5[4] and exp2["argmax"] == seeds5[0],
      "the rule prefers the healthy seed 4 over the diverged top mean")
with contextlib.redirect_stdout(_io.StringIO()) as log2:
    case2 = P.write_case_outputs(second, seeds5, 0.0)
text2 = log2.getvalue()
new_best = P.BEST_SEED_DIR.format(seed=seeds5[4])
check(not os.path.exists(os.path.join(folder, old_best)),
      f"the earlier run's {old_best}/ is removed")
check(f"removed {old_best}/" in text2, "and the log says so")
check(os.path.isfile(os.path.join(legacy, "old.csv")),
      f"the legacy seed{seeds5[0]}/ folder is left alone, with its file")
check("seed<N>/" in text2 and "old per-seed layout" in text2,
      "and reported as of the old layout")
check(subdirs(folder) == sorted([new_best, f"seed{seeds5[0]}"]),
      f"only the new best seed's folder beside it: {subdirs(folder)}")
# what an earlier layout left in the RESULTS folder: a folder per case with a
# folder per seed in it, and the seed matrix
old_case = os.path.join(P.RESULTS_DIR, "psi0+0.00_l128")
os.makedirs(os.path.join(old_case, f"seed{seeds5[0]}"))
with open(os.path.join(old_case, "fno_ch_snapshots.png"), "w") as fh:
    fh.write("x")
with open(os.path.join(P.RESULTS_DIR, "fno_ch_seed_matrix.png"), "w") as fh:
    fh.write("x")
open(os.path.join(P.RESULTS_DIR, "manifest.json"), "w").close()
old = P.old_layout_files(P.RESULTS_DIR)
check(old == ["fno_ch_seed_matrix.png",
              "psi0+0.00_l128/ (one folder per case, the layout before the "
              "flat one)", f"psi0+0.00_l128/seed{seeds5[0]}"],
      f"old_layout_files finds the old layout's leftovers, and only them "
      f"(not the extras' own legacy folder, nor the flat files): {old}")
was = (P.WRITE_SUBMISSION, P.WRITE_TIMING)
P.WRITE_SUBMISSION = P.WRITE_TIMING = False
try:
    with contextlib.redirect_stdout(_io.StringIO()) as log3:
        P.evaluate_all(None, {}, None, case_fn=lambda *a, **k: None,
                       resume=False, clean=False)
finally:
    P.WRITE_SUBMISSION, P.WRITE_TIMING = was
check(f"still holds {len(old)} file(s) of the old per-seed layout"
      in log3.getvalue()
      and os.path.isfile(os.path.join(legacy, "old.csv"))
      and os.path.isdir(old_case),
      "a run into that folder warns about them at the start, and leaves them")
shutil.rmtree(old_case)
os.remove(os.path.join(P.RESULTS_DIR, "fno_ch_seed_matrix.png"))
os.remove(os.path.join(P.RESULTS_DIR, "manifest.json"))
shutil.rmtree(legacy, ignore_errors=True)
check_case(folder, "", exp2, case2, figures=False)
bt = titles(os.path.join(folder, "fno_ch_r2_and_snapshots.png"))
check("best healthy" in bt,
      "the R2 figure calls the shown seed the best healthy one")


print("\n3. the whole run: the flat run folder, the extras beside the records")
restore(TMP)
shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(CACHE, ignore_errors=True)
before = set(listing(SCRATCH))
FRAMES[0] = 4
P.WRITE_TIMING = True
P.WRITE_SUBMISSION = True
P.PER_SEED_FIGURES = False
code, _ = run_main(["--eval-mode", "critical", "--sizes", "128,512",
                    "--seeds", "3"])
FRAMES[0] = 0
check(code == 0, f"main() succeeded (exit {code})")
check(P.submission_dir() == P.extras_dir() == XR,
      "submission_dir() is the extras folder (the figures are in the run "
      "folder itself)")
beside = sorted(set(listing(SCRATCH)) - before
                - {os.path.basename(TMP), os.path.basename(CACHE)})
check(not beside, f"no Submission folder, or any other, beside the results: "
                  f"{beside}")
check(not any(d.lower().startswith("submission") for d in listing(TMP)),
      "and none inside them")
FLAT = sorted(
    [S.case_file(kind, tok) for tok in ("psi0p0.00", "l512")
     for kind in S.FIG_KINDS]
    + [f"l{n}/{f}" for n in (128, 512) for f in S.SIZE_FILES]
    + list(S.RUN_FILES)
    + ["r2_sweep_per_case/psi0p0.00.png", "r2_sweep_per_case/figure_notes.txt"])
top = files_rec(TMP)
check(top == FLAT and subdirs(TMP) == ["l128", "l512", "r2_sweep_per_case"],
      f"the run folder: the run's files at the top and a folder for each "
      f"box size (l128, l512) with the critical case's figures and the "
      f"size's own files: {top} {subdirs(TMP)}")
check(not any(f.startswith(".partial") for f in listing(TMP)),
      "no stage or hold left behind in it")
check(subdirs(XR) == ["psi0+0.00_l128", "psi0+0.00_l512"]
      and sorted(files_in(XR)) == sorted(
          ["fno_ch_timing.json", "fno_ch_timing.txt", S.BOOKKEEPING,
           S.CURVES_FILE]),
      f"the extras: a folder per case, and the run's timing report, "
      f"bookkeeping manifest and sweep curves: {subdirs(XR)} {files_in(XR)}")
film = None
for d, suffix in (("psi0+0.00_l128", ""), ("psi0+0.00_l512", "_l512")):
    here = os.path.join(XR, d)
    files = files_in(here)
    films = [f for f in files if f.startswith("rollout.")]
    film = films[0] if films and film is None else film
    want = sorted(case_files(suffix) + list(S.CASE_DATA) + films)
    check(len(films) == 1 and files == want,
          f"{d}: the case figures, summary, averages, data and the film are "
          f"in its extras folder: {files}")
    check(not any(x.startswith(".partial") for x in listing(here)),
          f"{d}: no stage or hold left behind")
    rec = read_summary(here, suffix).get("record") or {}
    check(rec.get("submission") is True and rec.get("video") is True
          and rec.get("complete") is True,
          f"{d}: its summary records the submission files and the film "
          f"(submission {rec.get('submission')}, video {rec.get('video')})")
    best = EXPECT.get(d, {}).get("best")
    sd = subdirs(here)
    inside = listing(os.path.join(here, sd[0])) if sd else []
    check(sd == [P.BEST_SEED_DIR.format(seed=best)]
          and not any(f.endswith((".mp4", ".gif")) for f in inside),
          f"{d}: one seed folder, the best seed's, and no film in it: {sd}")
for tok, size, d in (("psi0p0.00", "l128", "psi0+0.00_l128"),
                     ("l512", "l512", "psi0+0.00_l512")):
    kin_titles, drawn = kinetics_of(size, "psi0p0.00")
    n = len(EXPECT[d]["seeds"])
    check(f"mean over {n} seeds" in kin_titles
          and f"seed {EXPECT[d]['best']} " not in kin_titles,
          f"{size}/phase_ordering_kinetics_psi0p0.00.png is the mean over "
          f"the {n} seeds")
    check(drawn is not None and len(drawn) == n
          and all(a is b for a, b in zip(drawn, EXPECT[d]["results"])),
          f"{tok}: drawn from every seed's rollout, in the case's order")

man = read_json(os.path.join(TMP, "manifest.json"))
cases3 = man.get("cases") or {}
check(sorted(cases3) == ["l512", "psi0p0.00"]
      and man.get("aliases") == {"l128": "psi0p0.00"}
      and man.get("label") == os.path.basename(TMP)
      and cases3["psi0p0.00"].get("kind") == "composition"
      and cases3["l512"].get("kind") == "box",
      f"manifest.json is the flat schema: the cases by token, and l128 an alias of "
      f"the critical case: {sorted(cases3)}, {man.get('aliases')}")
check(set(man.get("files", ())) == set(top)
      and all(os.path.isfile(os.path.join(TMP, f))
              for f in man.get("files", ())),
      "every file the manifest lists is in the run folder, and no other")
book = read_json(os.path.join(XR, S.BOOKKEEPING))
crit, s128, s192 = ((book.get("compositions") or {}).get("+0.00") or {},
                    (book.get("sizes") or {}).get("128") or {},
                    (book.get("sizes") or {}).get("512") or {})
check(crit.get("folder") == "psi0p0.00" and s128.get("folder") == "psi0p0.00"
      and s192.get("folder") == "l512",
      f"the bookkeeping: composition +0.00 and size 128 both name the token "
      f"psi0p0.00, size 512 names l512: {crit.get('folder')}, "
      f"{s128.get('folder')}, {s192.get('folder')}")
check(crit.get("files") == s128.get("files")
      and "l128/snapshots_psi0p0.00.png" in (crit.get("files") or [])
      and all(f.startswith("l512/") for f in s192.get("files", [])),
      f"and the same run-folder files: {crit.get('files')}")
named = [f for e in (crit, s128, s192) for f in e.get("files", [])]
check(bool(named) and all(os.path.isfile(os.path.join(TMP, f))
                          for f in named),
      "every file the bookkeeping names is there")
check(film is not None
      and film in ((crit.get("extras") or {}).get("files") or []),
      f"the case film ({film}) is among the extras the bookkeeping names")
timing = read_json(os.path.join(XR, "fno_ch_timing.json"))
check(sorted((c["off"], c["lattice"]) for c in timing.get("cases", []))
      == [(0.0, 128), (0.0, 512)]
      and all(c["n_rollouts"] == 3 for c in timing.get("cases", [])),
      f"the timing report has one line per case: {timing.get('cases')}")


print("\n4. sweep seeds that differ: the l=128 sweep case has its own extras folder")
restore(TMP)
shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(CACHE, ignore_errors=True)
P.EVAL_SEEDS = P.make_eval_seeds(3)
P.SWEEP_SEEDS = P.EVAL_SEEDS[:2]
P.EVAL_MODE = "critical"
P.SIZE_SWEEP = (128,)
P.WRITE_VIDEO = False
P.PER_SEED_FIGURES = False
P.WRITE_SUBMISSION = True
FOUND = []


def caching_run_case(model, device, off, lattice=None, seeds=None):
    """fake_run_case, caching its rollouts as run_case does, after noting how
    many of them an earlier case had already cached."""
    sd = P.seeds_for(off) if seeds is None else tuple(seeds)
    files = P.case_cache_files(off, lattice, sd)
    FOUND.append(sum(os.path.exists(f) for f in files))
    for f in files:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f, "wb") as fh:
            fh.write(b"a cached rollout")
    return fake_run_case(model, device, off, lattice, seeds)


P.run_case = caching_run_case
code, _ = run_main([])
P.run_case = fake_run_case
check(code == 0, f"main() succeeded (exit {code})")
check(calls == [(0.0, None), (0.0, 128)],
      f"the critical case, then l=128 on the sweep seeds: {calls}")
check(FOUND == [0, 2],
      f"the sweep case still finds its 2 rollouts the critical case cached: "
      f"the two share them, so neither deletes them early ({FOUND})")
check(not any(os.path.exists(f)
              for f in P.case_cache_files(0.0, None, P.EVAL_SEEDS))
      and listing(CACHE) == [P.EXTRAS_NAME],
      "once both are written their rollouts go; only extras/ is left in the "
      "cache folder")
dirs4 = subdirs(XR)
check(dirs4 == ["psi0+0.00_l128", "psi0+0.00_l128_sweep_seeds"],
      f"two extras folders: {dirs4}")
for d, suffix, n in (("psi0+0.00_l128", "", 3),
                     ("psi0+0.00_l128_sweep_seeds", "_l128", 2)):
    here = os.path.join(XR, d)
    st = read_summary(here, suffix)
    best = EXPECT.get(d, {}).get("best")
    check(st.get("n_seeds") == n and st.get("best_seed") == best
          and subdirs(here) == [P.BEST_SEED_DIR.format(seed=best)],
          f"{d}: its own {n}-seed summary and best seed folder "
          f"{subdirs(here)}")
twin = os.path.join(XR, "psi0+0.00_l128_sweep_seeds")
check(S.fig_name("snapshots", "psi0p0.00_sweep_seeds") in files_in(twin)
      and not any("sweep_seeds" in f for f in files_rec(TMP)),
      "the twin's figures stay in its extras folder: the flat layout has no such case, "
      "so the run folder does not get them")
man4 = read_json(os.path.join(TMP, "manifest.json"))
book4 = read_json(os.path.join(XR, S.BOOKKEEPING))
check(((book4.get("compositions") or {}).get("+0.00") or {}).get("folder")
      == "psi0p0.00"
      and ((book4.get("sizes") or {}).get("128") or {}).get("folder")
      == "l128_sweep_seeds",
      "the bookkeeping's size 128 entry names the sweep-seed case")
check(sorted(man4.get("cases", {})) == ["psi0p0.00"]
      and man4.get("skipped_cases") == ["sizes 128: drawn apart in extras/"],
      "while manifest.json lists the critical case only, and says the twin "
      "was drawn apart")


print("\n5. a model without the hard constraints writes to _nohard")
restore(TMP)
shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(CACHE, ignore_errors=True)
FakeModel.config = dict(FakeModel.config, hard_mass_conservation=False)
P.PER_SEED_FIGURES = False
P.WRITE_SUBMISSION = True
code, _ = run_main(["--eval-mode", "critical", "--sizes", "128",
                    "--seeds", "2", "--no-video"])
check(code == 0, f"main() succeeded (exit {code})")
check(not os.path.exists(TMP), f"nothing written to {os.path.basename(TMP)}")
abl = TMP + "_nohard"
check(subdirs(abl) == ["l128"]
      and os.path.isfile(os.path.join(abl, "manifest.json"))
      and os.path.isfile(os.path.join(abl, "l128", "snapshots_psi0p0.00.png")),
      f"{os.path.basename(abl)}/ holds the run folder of the ablation")
XN = os.path.join(CACHE, P.EXTRAS_NAME + "_nohard")
check(os.path.isfile(os.path.join(XN, "psi0+0.00_l128",
                                  "fno_ch_nohard_summary.json"))
      and os.path.isfile(os.path.join(XN, S.BOOKKEEPING)),
      "its case files carry the _nohard label, and it keeps its extras in "
      "extras_nohard/ so they do not mix with the real model's")
check(not os.path.exists(os.path.join(CACHE, P.EXTRAS_NAME)),
      "(the real model's extras/ is not touched)")
check(P.EXTRAS_ROOT is None, "EXTRAS_ROOT is restored after the run")
check(P.RESULTS_DIR == TMP, "RESULTS_DIR is restored after the run")

# a run with views writes through RESULTS_TEMPLATE: the ablation moves that too
restore(TMP)
shutil.rmtree(abl, ignore_errors=True)
FakeModel.config = dict(FakeModel.config, hard_mass_conservation=False)
TEMPLATE = TMP + "_P{H}"
P.RESULTS_TEMPLATE = TEMPLATE
P.EVAL_MODE, P.SIZE_SWEEP = "critical", ()
P.WRITE_SUBMISSION = P.WRITE_TIMING = False
SEEN = []


def peek(model, device, off, lattice=None, seeds=None):
    """A case that only notes where the run is writing."""
    SEEN.append((P.RESULTS_DIR, P.RESULTS_TEMPLATE))
    return None


with contextlib.redirect_stdout(_io.StringIO()):
    P.evaluate_all(FakeModel(), fake_ck, "cpu", case_fn=peek, resume=False,
                   clean=False)
check(SEEN == [(abl, TEMPLATE + "_nohard")],
      f"during the run both RESULTS_DIR and RESULTS_TEMPLATE carry _nohard: "
      f"{SEEN}")
check(P.RESULTS_DIR == TMP and P.RESULTS_TEMPLATE == TEMPLATE,
      "and both are restored after it")

# folders named by the layout take the token into the train label, where
# fno_paths.nohard_name puts it, rather than at the end
LAB = os.path.join(SCRATCH, "layout_labels")
named = os.path.join(LAB, "train_t0500_pred_t20000")
named_tpl = os.path.join(LAB, "train_t0500_pred_t{H:0>5}")
P.RESULTS_DIR, P.RESULTS_TEMPLATE = named, named_tpl
SEEN.clear()
with contextlib.redirect_stdout(_io.StringIO()):
    P.evaluate_all(FakeModel(), fake_ck, "cpu", case_fn=peek, resume=False,
                   clean=False)
want_dir = os.path.join(LAB, "train_t0500_nohard_pred_t20000")
want_tpl = os.path.join(LAB, "train_t0500_nohard_pred_t{H:0>5}")
check(SEEN == [(want_dir, want_tpl)],
      f"an ablation of a run folder is train_t0500_nohard_pred_t20000, and "
      f"its template {{H:0>5}} likewise: {SEEN}")
check(want_tpl.format(H="2000")
      == os.path.join(LAB, fno_paths.run_label("train_t0500_nohard", 2000))
      and want_tpl.format(H="20000") == want_dir,
      "so each view's folder is the ablation's own run label")
check(P.RESULTS_DIR == named and P.RESULTS_TEMPLATE == named_tpl,
      "and both are restored after it")
SEEN.clear()
P.RESULTS_DIR = os.path.join(LAB, "train_t0500_nohard_pred_t20000")
P.RESULTS_TEMPLATE = None
with contextlib.redirect_stdout(_io.StringIO()):
    P.evaluate_all(FakeModel(), fake_ck, "cpu", case_fn=peek, resume=False,
                   clean=False)
check(SEEN == [(P.RESULTS_DIR, None)],
      f"a folder that already says nohard is left as it is: {SEEN}")

restore(TMP)
new = sorted(project_state() - PROJECT)
check(not new, f"no run wrote into the project folder, Work/ or Results/ "
               f"{new[:3]}")
print(f"\n{'all checks passed' if not fails else str(len(fails)) + ' FAILED'}")
for _d in (TMP, TMP + "_nohard", RERUN, CACHE, LAB):
    shutil.rmtree(_d, ignore_errors=True)
sys.exit(1 if fails else 0)
