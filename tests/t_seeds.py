"""50 evaluation seeds: still unseen, still comparable, still legible."""
import contextlib
import io as _io
import json
import os
import shutil
import sys

import matplotlib.figure as mfig
import numpy as np

from _paths import ROOT, SCRATCH

os.chdir(ROOT)
import ch_gpu as C
import fno_predict as P
import fno_train as T
T.use_fixed_step()
import make_submission as S

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


print("1. the seed set")
ev = P.EVAL_SEEDS
check(len(ev) == 50, f"50 seeds, got {len(ev)}")
check(len(set(ev)) == 50, "no duplicates")
check(ev[:10] == P.EVAL_SEEDS_BASE,
      "the original ten come first, so their caches still hit")
check(P.make_eval_seeds(50) == ev, "deterministic across calls")
check(P.make_eval_seeds(10) == P.EVAL_SEEDS_BASE,
      "asking for 10 reproduces exactly the old set")
check(P.make_eval_seeds(50)[:20] == P.make_eval_seeds(20),
      "growing the count only appends; it never reshuffles")

train = {s for s, _ in C.plan()[0]}
val = {s for s, _ in C.plan()[1]}
reserved = set(T.VAL_SEEDS) | set(T.MONITOR_SEEDS)
print(f"   | dataset seeds live in {min(train)}..{max(train)} and "
      f"{min(val)}..{max(val)}")
print(f"   | extras drawn from 100000..999999, e.g. {ev[10:14]}")
check(not (set(ev) & train), "disjoint from the 20 training trajectories")
check(not (set(ev) & val), "disjoint from the 5 val trajectories")
check(not (set(ev) & reserved),
      "disjoint from VAL_SEEDS and MONITOR_SEEDS")
check(all(s >= 100000 for s in ev[10:]),
      "every added seed is clear of the dataset's [0, 20000) range")

print("\n2. the case figures stay readable at 50, and show one seed")
P.MODEL_FLAGS.update(hard_constraints=True, hard_mass_conservation=True,
                     zero_dc=True, label="fno_ch", note="")
P.WRITE_LOG = False
P.WRITE_SUBMISSION = False
P.PER_SEED_FIGURES = False
# nothing here rolls out or solves, but the defaults are project-relative
P.RESULT_CACHE = os.path.join(SCRATCH, "seed_cache")
P.GT_CACHE = os.path.join(SCRATCH, "seed_gt_cache")
OUT = os.path.join(SCRATCH, "seed_figures")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(3)
t = np.arange(P.START_TIME + 1, P.END_TIME + 1)
T_END = max(P.SNAPSHOT_TIMES)
LAT = 128
LENGTHS = ("L_zero", "L_peak", "L_moment", "L_grad")

# Every saved figure's titles and legend entries, by the path it went to:
# how a check reads which seed a figure shows, and how many lines it names.
FIGS = {}
_savefig = mfig.Figure.savefig


def _remember(self, fname, *args, **kwargs):
    if isinstance(fname, str):
        legends = []
        for ax in self.axes:
            leg = ax.get_legend()
            if leg is not None:
                legends.append([x.get_text() for x in leg.get_texts()])
        FIGS[os.path.normcase(os.path.abspath(fname))] = {
            "titles": "\n".join([x.get_text() for x in self.texts]
                                + [ax.get_title() for ax in self.axes]
                                + [ax.get_title("left") for ax in self.axes]),
            "legends": legends}
    return _savefig(self, fname, *args, **kwargs)


setattr(mfig.Figure, "savefig", _remember)


def seen(path):
    return FIGS.get(os.path.normcase(os.path.abspath(path)),
                    {"titles": "", "legends": []})


def fake(n):
    """n synthetic rollouts, their mean R2 0.03 apart in a shuffled order.

    Returns (results, by): by[k] is the index of the k-th best mean, so the
    ranking is known before any code picks a seed.  The gap is wider than
    what a divergence at t = 1990 costs a mean, so blow() never reorders.
    """
    order = rng.permutation(n)
    out = []
    for i in range(n):
        r2 = np.clip(1.0 - 0.03 * order[i] - (t - t[0]) / 9000.0, -1, 1)
        below = r2 < P.R2_THRESHOLD
        dt = t[::100]
        out.append({
            "times": t, "r2": r2, "mass_drift": np.full(t.size, 1e-9),
            "mean_true": np.zeros(t.size), "mean_pred": np.full(t.size, 1e-9),
            "diag_times": dt,
            "energy_true": np.linspace(1, 0.5, dt.size),
            "energy_pred": np.linspace(1, 0.5, dt.size),
            "t_stop": float(t[np.argmax(below)] if below.any() else t[-1]),
            "stopped_early": bool(below.any()), "diverged_at": None,
            "L_true": {k: np.linspace(4, 20, dt.size) for k in LENGTHS},
            "L_pred": {k: np.linspace(4, 20, dt.size) for k in LENGTHS},
            "snapshots": {s: (rng.standard_normal((LAT, LAT)),
                              rng.standard_normal((LAT, LAT)))
                          for s in P.SNAPSHOT_TIMES},
            "gt_seconds": 1.0, "gt_cached": False, "gt_steps": 1,
            "fno_seconds": 0.5, "fno_steps": 1900})
    return out, [int(np.flatnonzero(order == k)[0]) for k in range(n)]


def blow(res, finite_end=False):
    """Diverge a rollout at t = 1990 as evaluate_seed records it: R2 at the
    floor and no prediction from then on.  finite_end keeps a prediction in
    the final snapshot -- a seed flagged diverged that still has one there,
    which the rule ranks below every seed that never diverged."""
    late = res["times"] >= 1990.0
    res["diverged_at"] = 1990.0
    res["stopped_early"] = True
    res["r2"] = np.where(late, P.R2_FLOOR, res["r2"])
    res["mean_pred"] = np.where(late, np.nan, res["mean_pred"])
    if not finite_end:
        truth, _ = res["snapshots"][T_END]
        res["snapshots"][T_END] = (truth, np.full_like(truth, np.nan))


for n in (10, 50):
    res, by = fake(n)
    # the top mean diverges with nothing at the final time, the runner-up is
    # flagged diverged with a prediction there: the best is the third
    blow(res[by[0]])
    blow(res[by[1]], finite_end=True)
    seeds = P.EVAL_SEEDS[:n]
    want = int(seeds[by[2]])
    P.RESULTS_DIR = os.path.join(OUT, f"results_{n}")
    P.PER_SEED_FIGURES = True
    with contextlib.redirect_stdout(_io.StringIO()):
        P.write_case_outputs(res, seeds, 0.0)
    P.PER_SEED_FIGURES = False
    folder = P.case_dir(0.0)
    figs = {k: os.path.join(folder, f"fno_ch_{k}.png")
            for k in ("r2_and_snapshots", "kinetics", "conservation")}
    # the case's own kinetics figure is drawn flat by make_submission now
    # (phase_ordering_kinetics_<token>.png); the figure itself is P's
    knotes: dict = {}
    with contextlib.redirect_stdout(_io.StringIO()):
        P.figure_kinetics(res, seeds, figs["kinetics"], 0.0, LAT,
                          notes=knotes)
    check(all(os.path.exists(f) for f in figs.values())
          and folder == os.path.join(P.RESULT_CACHE, "extras",
                                     P.case_name(0.0, None)),
          f"the three case figures rendered for {n} seeds, in extras/<case>/")
    legends = seen(figs["r2_and_snapshots"])["legends"]
    marks = ("finite-size regime", "training horizon")
    check(len(legends) == 1
          and legends[0][0] == f"mean over {n} unseen seeds"
          and all(e.startswith(marks) for e in legends[0][1:]),
          f"{n} seeds: the R2 panel's legend is the mean (and the time "
          f"marks), no threshold entry, not a line per seed: {legends}")
    check(f"seed {want} (best healthy of {n}, rank 3,"
          in seen(figs["r2_and_snapshots"])["titles"],
          f"{n} seeds: its snapshots are seed {want}, the best healthy one, "
          f"rank 3")
    own = os.path.join(folder, P.BEST_SEED_DIR.format(seed=want),
                       f"fno_ch_seed{want}_kinetics.png")
    check(f"mean over {n} seeds" in knotes["fno_ch_kinetics.png"][0]
          and f"seed {want}" not in knotes["fno_ch_kinetics.png"][0]
          and seen(figs["kinetics"])["titles"].count("(a)") == 1
          and os.path.exists(own),
          f"{n} seeds: the kinetics figure is their mean, and the best "
          f"seed {want}'s own is in its folder")
    cons = seen(figs["conservation"])
    entries = [e for leg in cons["legends"] for e in leg]
    check(f"(mean and worst of {n} seeds)" in cons["titles"]
          and f"mean over {n} seeds" in entries
          and "worst seed at each step" in entries
          and not any(e.startswith("seed ") for e in entries),
          f"{n} seeds: the conservation figure draws the mean over the seeds "
          f"and the worst at each step, no seed's own curve: "
          f"{cons['legends']}")
    kept = sorted(d for d in os.listdir(folder)
                  if os.path.isdir(os.path.join(folder, d)))
    check(kept == [P.BEST_SEED_DIR.format(seed=want)],
          f"{n} seeds: one seed folder, the best seed's, not {n}: {kept}")
    with open(os.path.join(folder, "fno_ch_summary.json")) as fh:
        st = json.load(fh)
    check(st.get("n_seeds") == n and st.get("best_seed") == want
          and st.get("best_seed_rank") == 3 and st.get("seeds_diverged") == 2
          and "seeds" not in st,
          f"{n} seeds: the summary counts them, names seed "
          f"{st.get('best_seed')} rank {st.get('best_seed_rank')} and lists "
          f"none")

res5, _ = fake(5)
path5 = os.path.join(OUT, "conservation_5.png")
with contextlib.redirect_stdout(_io.StringIO()):
    P.figure_conservation(res5, P.EVAL_SEEDS[:5], path5, 0.0, LAT)
named = [e for leg in seen(path5)["legends"] for e in leg
         if e.startswith("seed ")]
check(named == [] and "mean over 5 seeds" in [
          e for leg in seen(path5)["legends"] for e in leg],
      f"at 5 seeds, too, the conservation figure names no seed: {named}")
path1 = os.path.join(OUT, "conservation_1.png")
with contextlib.redirect_stdout(_io.StringIO()):
    P.figure_conservation(res5[:1], P.EVAL_SEEDS[:1], path1, 0.0, LAT)
check([e for leg in seen(path1)["legends"] for e in leg
       if e.startswith("seed ")] == [f"seed {P.EVAL_SEEDS[0]}"],
      "while one seed's own figure (the best seed's) names that seed")

print("\n3. the best-seed rule at 50 seeds")
res, by = fake(50)
idx, rank, score = P.pick_result(res)
check(idx == by[0] and rank == 1
      and score == float(np.nanmean(res[by[0]]["r2"])),
      f"nothing diverged: the top mean, rank 1 (got index {idx}, rank "
      f"{rank})")
check(idx == S.best_index(res), "pick_result is make_submission.best_index")
blow(res[by[0]])
idx, rank, _ = P.pick_result(res)
check(idx == by[1] and rank == 2 and idx == S.best_index(res),
      f"the top mean diverged: the runner-up, rank 2 (got rank {rank})")
perm = rng.permutation(50)
idx_p, rank_p, _ = P.pick_result([res[int(i)] for i in perm])
check(int(perm[idx_p]) == idx and rank_p == rank,
      "the pick follows the seed, not its place in the list")
for k in range(50):
    blow(res[by[k]], finite_end=k >= 3)
idx, rank, _ = P.pick_result(res)
check(idx == by[3] and rank == 4 and idx == S.best_index(res),
      f"every seed diverged: the top mean of those with a prediction at the "
      f"final time (got rank {rank})")
for k in range(50):
    blow(res[by[k]])
idx, rank, _ = P.pick_result(res)
check(idx == by[0] and rank == 1 and idx == S.best_index(res),
      f"no seed has a prediction at the final time: the top mean of all "
      f"(got rank {rank})")
check(P.pick_result(res[:1])[:2] == (0, 1), "a single seed is its own best")

print(f"\n   | figures written to {OUT}")
print("\n" + ("SEEDS PASS" if ok else "SEEDS FAIL"))
sys.exit(0 if ok else 1)
