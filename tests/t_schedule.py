"""The schedule engine of fno_predict.py: a dt-conditioned model's prediction.

  1. the engine is chosen from the checkpoint: schedule for a dt-conditioned
     one (the group seeds by default, no film), refused for the others and back
  2. the schedule: must-hit times, the group protocol's step count, tuning report times
  3. the engine on synthetic references: t_start and absolute times, the
     result's schedule, must-hit snapshots, divergence (R2 = -1 kept, the
     seed still in the means), no model call once no seed is alive, chunking
  4. a tiny random-init dt-conditioned model end to end at a small lattice
     with the references solved into a temporary FNO_GT_CACHE: a job killed
     after tuning, the pick kept in meta.json and reused on resume, the run
     folder flat, the records carrying t_start and the schedule
  5. the diagnostics take a dt (tile 10, low-k 48), a fixed-step model keeps
     today's, and the low-k check reads FIXED_STEP's window, not fno_train's
  6. the lock-step engine of a fixed-step model: no film unless --video,
     and the loop ends once every seed has diverged

Also: the snapshot set follows the group protocol's rule to a run of 20000, a tune seed that
is also a test seed is left out, and a run made the way fp.sh makes it
(--views, no --snapshots, the film off, the best seed's figures on) is redrawn
from its records in a fresh process with the same snapshot times.

Everything is written under tests/_scratch.
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

from typing import Any

import numpy as np
import torch
from torch import nn

from _paths import ROOT, SCRATCH, project_state

BOX = os.path.join(SCRATCH, "schedule_box")
shutil.rmtree(BOX, ignore_errors=True)
os.makedirs(BOX)
os.chdir(BOX)
PROJECT = project_state()
REFS = os.path.join(BOX, "refs")
os.environ["FNO_GT_CACHE"] = REFS

import fno_paths
fno_paths.WORK = os.path.join(BOX, "Work")
import fno_gt
import fno_predict as P
import fno_train as T
import make_submission as S

ok = True
FS: dict[str, Any] = T.FIXED_STEP


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def captured(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


KEYS = ("START_TIME", "END_TIME", "SNAPSHOT_TIMES", "VIDEO_EVERY", "DIAG_EVERY",
        "ENGINE", "CONDITIONED", "EVAL_BATCH", "VIEWS", "RESULTS_DIR",
        "RESULTS_TEMPLATE", "LOG_TIME", "CACHE_END_TIME", "WRITE_VIDEO",
        "MODEL_PATH", "RESULT_CACHE", "GT_CACHE", "REF_CACHE", "LATTICE",
        "SIZE_SWEEP", "EVAL_SEEDS", "SWEEP_SEEDS", "EVAL_MODE", "WRITE_LOG",
        "WRITE_TIMING", "PER_SEED_FIGURES", "KEEP_CACHE", "REDO",
        "load_checkpoint", "CACHE_RESULTS", "WRITE_SUBMISSION", "SEED_SET",
        "VIDEO_SCHEDULE", "VIDEO_DTYPE", "SNAPSHOTS_EXPLICIT", "_RUN_SNAPSHOTS",
        "TUNE", "TUNE_ALPHAS", "TUNE_DT_MAXS", "TUNE_SEEDS", "ROLLOUT_CELLS",
        "HOST_WORKERS", "DIAGNOSTICS", "DIAG_TILE", "DIAG_TILE_T",
        "DIAG_LOWK_L", "DIAG_LOWK_T_END", "DIAG_LOWK_SEEDS", "RECORDS",
        "ch_ground_truth_batch", "ch_ground_truth", "evaluate_schedule",
        "tune_schedule", "PER_SEED_OUTPUT", "NO_SUPERSEDE",
        "load_references", "FIXED_STEP", "solve_references", "PLAN_ONLY")
SAVED = {k: getattr(P, k) for k in KEYS}
SAVED_LOWK = T.lowk_gains


def restore():
    for k, v in SAVED.items():
        setattr(P, k, v)
    T.lowk_gains = SAVED_LOWK
    P._PICKS.clear()
    P._META.clear()
    P._REF_SECONDS.clear()


L = 16


def tiny_model(conditioned):
    torch.manual_seed(0)
    return T.CahnHilliardFNO(channels=8, modes=4, n_ref=16,
                             laplacian_output=True,
                             dt_conditioned=conditioned).double()


COND = os.path.join(BOX, "cond", "fno_ch.pt")
FIXED = os.path.join(BOX, "fixed", "fno_ch.pt")
for path, cond in ((COND, True), (FIXED, False)):
    os.makedirs(os.path.dirname(path))
    T.save_model(tiny_model(cond), path)

# ---------------------------------------------------------------------------
print("1. the engine follows the checkpoint")
check(P.checkpoint_conditioned(COND) and not P.checkpoint_conditioned(FIXED)
      and not P.checkpoint_conditioned(os.path.join(BOX, "none.pt")),
      "checkpoint_conditioned reads the config (missing file: fixed-step)")


def overrides(argv):
    restore()
    P.WRITE_LOG = False
    args = P.parse_args(argv)
    err, buf = None, io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            P.apply_overrides(args)
    except SystemExit as exc:
        err = str(exc)
    state = {k: getattr(P, k) for k in ("ENGINE", "CONDITIONED", "SEED_SET",
                                         "WRITE_VIDEO", "TUNE")}
    state["seeds"] = tuple(P.EVAL_SEEDS)
    state["snaps"] = tuple(P.SNAPSHOT_TIMES)
    state["err"] = err
    state["log"] = buf.getvalue()
    state["tune"] = P.tune_seeds()
    state["signature"] = repr(P.tune_signature())
    restore()
    return state


s = overrides(["--model", COND, "--views", "10000"])
check(s["ENGINE"] == "schedule" and s["CONDITIONED"] and s["err"] is None,
      "a dt-conditioned model: ENGINE = schedule without any flag")
check(s["SEED_SET"] == "group" and s["seeds"] == P.GROUP_SEEDS,
      "its default test seeds are the group's 50")
check(not s["WRITE_VIDEO"], "no film in the schedule engine")
check(s["snaps"] == (100.0, 500.0, 1000.0, 2000.0, 4000.0, 5000.0, 6000.0,
                     8000.0, 10000.0),
      f"snapshots 100 500 1000 2000 4000 5000 6000 8000 10000 ({s['snaps']})")
GROUP_10000 = (100.0, 500.0, 1000.0, 2000.0, 4000.0, 5000.0, 6000.0, 8000.0,
             10000.0)
GROUP_20000 = GROUP_10000 + (12000.0, 14000.0, 16000.0, 18000.0, 20000.0)
s = overrides(["--model", COND, "--views", "20000"])
check(s["snaps"] == GROUP_20000 and s["err"] is None,
      f"T_END = 20000 (fp.sh): the snapshots go on to 18000 and 20000 "
      f"({s['snaps']})")
s = overrides(["--model", COND, "--seed-set", "fno", "--seeds", "7"])
check(s["SEED_SET"] == "fno" and s["seeds"] == P.make_eval_seeds(7),
      "--seed-set fno stays available")
s = overrides(["--model", COND, "--seed-set", "fno"])
check(90210 in s["seeds"] and 90210 in fno_gt.VAL_SEEDS,
      "FNO's 50 seeds share one with the tune seeds (90210)")
check(90210 not in s["tune"] and len(s["tune"]) == len(fno_gt.VAL_SEEDS) - 1
      and set(s["tune"]) <= set(fno_gt.VAL_SEEDS)
      and "[warn] tune seeds [90210] are also test seeds" in s["log"]
      and "90210" not in s["signature"],
      f"so the schedule engine leaves it out of the tuning, with a warning, "
      f"and the tuning signature says so ({s['tune']})")
s = overrides(["--model", COND, "--seed-set", "fno", "--no-tune"])
check("[warn]" not in s["log"] and s["err"] is None,
      "no tuning, nothing to warn of")
s = overrides(["--model", COND])
check(s["tune"] == list(fno_gt.VAL_SEEDS) and "[warn]" not in s["log"]
      and "90210" in s["signature"],
      "the group seeds share none with them: all ten, no warning")
restore()
P.ENGINE, P.EVAL_SEEDS = "schedule", P.GROUP_SEEDS[:3]
P.SWEEP_SEEDS, P.TUNE_SEEDS = P.EVAL_SEEDS, P.GROUP_SEEDS[:2]
try:
    quiet(P.check_tune_seeds)
    err = None
except SystemExit as exc:
    err = str(exc)
check(err and "every tune seed is also a test seed" in err,
      "a run whose tune seeds are all test seeds stops")
restore()
s = overrides(["--model", COND, "--engine", "lockstep"])
check(s["err"] and "schedule engine only" in s["err"],
      f"--engine lockstep with a dt-conditioned model is refused ({s['err']})")
s = overrides(["--model", FIXED, "--engine", "schedule"])
check(s["err"] and "dt-conditioned" in s["err"],
      "--engine schedule with a fixed-step model is refused")
s = overrides(["--model", FIXED, "--end-time", "10000"])
check(s["ENGINE"] == "lockstep" and not s["CONDITIONED"]
      and s["SEED_SET"] == "fno" and not s["WRITE_VIDEO"],
      "a fixed-step model: the lock-step engine beyond 2000, FNO's seeds, "
      "and (D6) no film")
s = overrides(["--model", FIXED, "--end-time", "10000", "--video"])
check(s["ENGINE"] == "lockstep" and s["WRITE_VIDEO"],
      "--video gives the lock-step engine its film back")
s = overrides(["--model", FIXED, "--end-time", "1000"])
check(s["ENGINE"] == "seed" and s["WRITE_VIDEO"],
      "the seed engine keeps its film")
s = overrides(["--model", COND, "--end-time", "1002", "--start-time", "100"])
check(s["err"] and "reference grid" in s["err"],
      f"an end time off the 4.0 grid is refused ({s['err']})")
s = overrides(["--model", COND, "--end-time", "400", "--snapshots", "100,402,"
               "400"])
check(s["err"] and "reference grid" in s["err"], "so is a snapshot off it")
s = overrides(["--model", COND, "--no-tune"])
check(not s["TUNE"], "--no-tune")

# ---------------------------------------------------------------------------
print("2. the schedule")
restore()
P.START_TIME, P.END_TIME, P.VIEWS = 100.0, 10000.0, (3000.0, 10000.0)
P.ENGINE = "schedule"
P.SNAPSHOT_TIMES = tuple(sorted(set(P.default_snapshots(100.0, 10000.0))
                                | {3000.0}))
hits = P.schedule_must_hit()
check(hits == [500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 8000.0,
               10000.0], f"must-hit: the snapshots and the views ({hits})")
times = P.schedule_times(0.2, 40.0)
check(all(any(abs(t - h) < 1e-9 for t in times) for h in hits)
      and times[0] == 100.0 and times[-1] == 10000.0,
      "every must-hit time is in the schedule, which spans 100..10000")
group = fno_gt.make_schedule(100.0, 10000.0, 0.2, 1.0, 40.0, quantum=4.0,
                           must_hit=(500.0, 1000.0, 2000.0, 4000.0, 6000.0,
                                     8000.0, 10000.0))
check(len(group) - 1 == 255 and len(times) - 1 <= 262,
      f"steps: the group protocol's schedule has {len(group) - 1}, this one {len(times) - 1} "
      f"(the same plus 5000 and 3000), not the 9,900 of a fixed step")
check(np.all(np.diff(times) >= 4.0 - 1e-9)
      and np.allclose(np.round((times - 100.0) / 4.0), (times - 100.0) / 4.0),
      "every time is a reference frame (100 + a multiple of 4)")
check(P.tune_report() == [2500.0, 5000.0, 7500.0, 10000.0],
      f"the tuning scores T/4, T/2, 3T/4, T ({P.tune_report()})")


def group_rule(t0, t1):
    """The group protocol's snapshot set (its main 100 500 1000 2000, every
    multiple of 2000, the end) plus FNO's 5000, written out again."""
    main = {x for x in (100.0, 500.0, 1000.0, 2000.0, 5000.0)
            if t0 <= x <= t1}
    long = {2000.0 * k for k in range(1, int(t1 // 2000.0) + 1)
            if t0 < 2000.0 * k <= t1}
    return tuple(sorted(main | long | {t1}))


check(P.default_snapshots(100.0, 10000.0) == GROUP_10000,
      f"the snapshots of a run to 10000 are exactly the group protocol's 8 and 5000 "
      f"({P.default_snapshots(100.0, 10000.0)})")
check(all(P.default_snapshots(100.0, e) == group_rule(100.0, e)
          for e in (400.0, 1500.0, 2000.0, 2500.0, 10000.0, 13000.0,
                    20000.0, 30000.0)),
      "and follow the group protocol's rule at other ends, 20000 and 30000 among them")
check(P.default_snapshots(100.0, 20000.0) == GROUP_20000,
      f"a run to 20000 adds 12000 to 18000 ({P.default_snapshots(100.0, 20000.0)})")
P.END_TIME, P.VIEWS = 20000.0, (20000.0,)
P.SNAPSHOT_TIMES = P.default_snapshots(100.0, 20000.0)
hits = P.schedule_must_hit()
times = P.schedule_times(0.2, 40.0)
check(all(t in hits for t in (12000.0, 14000.0, 16000.0, 18000.0, 20000.0))
      and all(any(abs(t - h) < 1e-9 for t in times) for h in hits)
      and times[0] == 100.0 and times[-1] == 20000.0,
      "end = 20000: 12000..18000 are must-hit times and in the schedule")
P.END_TIME, P.VIEWS = 220.0, None
check(P.tune_report() == [108.0, 164.0, 220.0],
      f"and on the grid for a short window (T/4 is before the start) ({P.tune_report()})")

# ---------------------------------------------------------------------------
print("3. the engine on synthetic references")
restore()
P.START_TIME, P.END_TIME, P.VIEWS = 100.0, 220.0, None
P.ENGINE = "schedule"
P.SNAPSHOT_TIMES = (100.0, 140.0, 220.0)
P.WRITE_VIDEO = False
P.HOST_WORKERS = 2
gt_t = np.arange(100.0, 220.0 + 2.0, 4.0)


def synthetic(n, l=L, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n):
        base = rng.uniform(-0.3, 0.3, (l, l))
        base -= base.mean()
        k = np.arange(len(gt_t))[:, None, None]
        out.append((gt_t, (np.tanh(base[None] * (1 + 0.3 * k))).astype(
            np.float32)))
    return out


class Stand(nn.Module):
    """A dt-conditioned stand-in: seed b grows at the rate rates[b] per unit
    of time, so the seeds diverge at different times (rate 0: never)."""
    dt_conditioned = True

    def __init__(self, rates):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.rates = torch.tensor(rates, dtype=torch.float64)
        self.calls = 0
        self.dts = []
        self.config = {"dt_conditioned": True, "n_ref": L}

    def __deepcopy__(self, memo):
        # the tile check steps a copy of the model: its dts land in this list
        twin = Stand(self.rates.tolist())
        twin.dts = self.dts
        return twin

    def forward(self, x, dt):
        self.calls += 1
        dt = torch.as_tensor(dt, dtype=x.dtype).reshape(-1).expand(x.shape[0])
        self.dts.append(dt.detach().clone())
        r = self.rates[:x.shape[0]].view(-1, 1, 1, 1)
        return x * (1.0 + dt.view(-1, 1, 1, 1) * r) * self.w


alpha, dt_max = 0.5, 30.0
pick = {"alpha": alpha, "dt_max": dt_max, "dt_min": 1.0}
times = P.schedule_times(alpha, dt_max)
n = len(times) - 1
refs = synthetic(4)
model = Stand([0.0, 0.05, 0.1, 0.2])
got, calls = quiet(P.evaluate_schedule, model, [11, 12, 13, 14], "cpu", refs,
                   times, pick)
r0, r1, r2_, r3 = got
check(all(r["t_start"] == 100.0 for r in got), "every result has t_start = 100")
check(np.allclose(r0["times"], times) and r0["times"][0] == 100.0
      and len(r0["times"]) == n + 1,
      "its times are the schedule's absolute times, from t_start on")
check(all(P.has_start_sample(r) for r in got)
      and all(r["r2"][0] == 1.0 and r["mass_drift"][0] == 0.0
              and r["mean_pred"][0] == r["mean_true"][0] for r in got),
      "the first sample is the start: R2 = 1 exactly, no drift, the truth's "
      "mean in both")
check(r0["schedule"] == {"kind": "schedule", "alpha": alpha, "dt_min": 1.0,
                         "dt_max": dt_max, "steps": n, "quantum": 4.0},
      f"and its schedule ({r0['schedule']})")
check(140.0 in r0["snapshots"] and 220.0 in r0["snapshots"]
      and 100.0 in r0["snapshots"] and 140.0 in times,
      "the snapshot times are hit, and the input frame is kept")
check(r0["diverged_at"] is None and len(r0["r2"]) == n + 1
      and np.all(np.isfinite(r0["r2"])),
      "rate 0: no divergence, the start and one R2 per stride")
div = [r["diverged_at"] for r in (r1, r2_, r3)]
check(all(d is not None for d in div) and div[0] >= div[1] >= div[2]
      and div[0] > div[2],
      f"faster growth diverges earlier, at different times ({div})")
for r in (r1, r2_, r3):
    after = r["times"] >= r["diverged_at"] - 1e-9
    check(len(r["r2"]) == n + 1 and np.all(r["r2"][after] == P.R2_FLOOR)
          and np.all(r["r2"][~after] >= P.R2_FLOOR)
          and np.all(np.isnan(r["mean_pred"][after]))
          and np.all(np.isfinite(r["mean_true"])),
          f"a seed diverged at t = {r['diverged_at']:g} is R2 = -1 from "
          f"there to the end and keeps its full-length series")
check(r3["stopped_early"] and r3["t_stop"] <= r3["diverged_at"],
      "a diverged seed counts as failed at its divergence")
check(np.all(np.isnan(r1["snapshots"][220.0][1]))
      and np.all(np.isfinite(r1["snapshots"][220.0][0])),
      "a snapshot after the divergence has the truth and no prediction")
_, stack = P._r2_stack(got)
check(np.all(np.isfinite(stack)) and stack.shape == (4, n + 1)
      and np.all(stack[:, 0] == 1.0)
      and abs(stack[:, -1].mean() - (r0["r2"][-1] - 3.0) / 4.0) < 1e-12,
      "the mean over the seeds starts at 1 and keeps the diverged ones at -1")
check(len(r0["diag_times"]) == n and len(r0["L_true"]["L_zero"]) == n,
      "the lengths and energies are measured at every schedule time")
check(len({tuple(dt.tolist()) for dt in model.dts}) > 1
      and all(dt.shape == (4,) for dt in model.dts),
      "the model is told a dt per seed, and the strides differ")
check(calls == n == len(model.dts)
      and abs(sum(float(dt[0]) for dt in model.dts)
              - (times[-1] - times[0])) < 1e-9
      and all(abs(float(dt[0]) - (times[i + 1] - times[i])) < 1e-9
              for i, dt in enumerate(model.dts)),
      "the seed that never diverges takes every stride, and its dts add up "
      "to the time covered")

# no seed alive: the model is not called again
bad = Stand([0.3, 0.3, 0.3])
got, calls = quiet(P.evaluate_schedule, bad, [1, 2, 3], "cpu", synthetic(3),
                   times, pick)
check(0 < calls < n and bad.calls == calls,
      f"once every seed has diverged the model is not called "
      f"({calls} of {n} strides)")
check(all(len(r["r2"]) == n + 1 and r["r2"][-1] == P.R2_FLOOR
          and r["diverged_at"] is not None and len(r["times"]) == n + 1
          and np.all(np.isfinite(r["mean_true"])) for r in got),
      "yet each series runs to the end of the schedule, scored -1")
check(all(abs(r["times"][-1] - 220.0) < 1e-9 for r in got), "to t = 220")
check(len(bad.dts) == calls
      and abs(sum(float(dt[0]) for dt in bad.dts)
              - (times[calls] - times[0])) < 1e-9,
      "and the dts the model got add up to the time it did cover")

# chunking is by cells: ROLLOUT_CELLS // l^2 seeds
chunks = []
real_eval = P.evaluate_schedule


def spy(model, seeds, *a, **k):
    chunks.append(list(seeds))
    return real_eval(model, seeds, *a, **k)


P.evaluate_schedule = spy
P.ROLLOUT_CELLS = 3 * L * L
P.MODEL_PATH = COND
P.CACHE_RESULTS = False
P.RECORDS = False
P.LATTICE = L
P.load_references = lambda l, off, seeds, device: synthetic(len(seeds))
P._PICKS["+0.0"] = {"alpha": alpha, "dt_max": dt_max, "dt_min": 1.0,
                    "source": "test", "signature": repr(P.tune_signature())}
res = quiet(P.schedule_case, Stand([0.0] * 7), "cpu", 0.0, None,
            list(range(20, 27)), "key")
check([len(c) for c in chunks] == [3, 3, 1] and all(r is not None
                                                    for r in res),
      f"7 seeds at 3 per chunk (ROLLOUT_CELLS // l^2): {[len(c) for c in chunks]}")

# the tuning reads the tune seeds less any test seed (--seed-set fno: 90210)
seen_tune = []


def tune_refs(l, off, seeds, device):
    seen_tune.append(list(seeds))
    return synthetic(len(seeds))


P.load_references = tune_refs
P.EVAL_SEEDS = P.SWEEP_SEEDS = P.make_eval_seeds(50)
P.TUNE_ALPHAS, P.TUNE_DT_MAXS = (0.2,), (40.0,)
quiet(P.tune_schedule, Stand([0.0] * 10), "cpu", 0.0)
check(90210 in P.EVAL_SEEDS and seen_tune
      == [[s for s in fno_gt.VAL_SEEDS if s != 90210]],
      f"tune_schedule asks for the references of the tune seeds without the "
      f"test seed 90210 ({seen_tune})")

# the [tune] lines: the real tuner on a one-candidate grid (every edge), and
# the helper on synthetic tables
_, out_one = captured(P.tune_schedule, Stand([0.0] * 10), "cpu", 0.0)
check("[tune] 0 of 1 candidates had at least one diverged tuning seed"
      in out_one and out_one.count("[tune] warning") == 4
      and "sits at" not in out_one,
      f"the tuner prints the count and a line for each of the four edges "
      f"on a one-value grid: {out_one.count('[tune] warning')} warnings")
GA, GD = (0.1, 0.2, 0.35), (30.0, 40.0, 50.0)


def rows(div):
    return [{"alpha": a, "dt_max": d, "steps": 9, "score": 0.5,
             "n_diverged": div.get((a, d), 0)} for a in GA for d in GD]


for pick, want, text_ in (
        ((0.2, 40.0), (False,) * 4, None),
        ((0.1, 40.0), (True, False, False, False), "smaller alpha"),
        ((0.35, 40.0), (False, True, False, False), "larger alpha"),
        ((0.2, 30.0), (False, False, True, False), "smaller dt_max"),
        ((0.2, 50.0), (False, False, False, True), "larger dt_max")):
    ed = P.tune_edges({"alpha": pick[0], "dt_max": pick[1]},
                      rows({(0.1, 30.0): 2, (0.2, 30.0): 1}), GA, GD)
    got = (ed["alpha_smallest"], ed["alpha_largest"], ed["dt_max_smallest"],
           ed["dt_max_largest"])
    lines = P.tune_lines(ed)
    check(got == want and ed["k"] == 2 and ed["n"] == 9
          and "2 of 9 candidates" in lines[0]
          and len(lines) == 1 + sum(want)
          and (text_ is None or text_ in lines[1]),
          f"pick {pick}: edges {got}, k = {ed['k']} of {ed['n']}")
sig_pick = {"alpha": 0.1, "dt_max": 30.0,
            "signature": repr(("tuned", 100.0, 10000.0, GA, GD, (1, 2)))}
ed_sig = P.tune_edges(sig_pick, rows({}))
check(ed_sig["alpha_smallest"] and ed_sig["dt_max_smallest"]
      and not ed_sig["dt_max_largest"],
      "a stored pick is judged on the grid of its own signature")
P.TUNE_ALPHAS, P.TUNE_DT_MAXS = GA, GD
ed_bad = P.tune_edges({"alpha": 0.35, "dt_max": 50.0, "signature": "(("},
                      rows({}))
check(not ed_bad["alpha_smallest"] and ed_bad["alpha_largest"]
      and ed_bad["dt_max_largest"],
      "an unreadable signature falls back to the current TUNE_ constants")

# a schedule result cut at a view horizon keeps its schedule and steps
cut = P.truncate_results(r0, 140.0, None, (100.0, 140.0))
check(abs(cut["times"][-1] - 140.0) < 1e-9 and cut["t_start"] == 100.0
      and cut["schedule"]["steps"] == len(cut["times"]) - 1
      and cut["fno_steps"] <= len(cut["times"]) - 1 and P.has_start_sample(cut)
      and cut["times"][0] == 100.0,
      "truncate_results: times cut at the view, the start kept, steps "
      "counted per stride")

# a result made before the start sample (its series begin at the first
# stride's end) is read as one that lacks it, and nothing is counted twice
old_r = dict(r0)
for k in ("times", "r2", "mass_drift", "mean_true", "mean_pred"):
    old_r[k] = np.asarray(r0[k])[1:]
check(not P.has_start_sample(old_r) and P.has_start_sample(r0)
      and not P.has_start_sample({"t_start": 100.0, "times": r0["times"]})
      and not P.has_start_sample(dict(old_r, t_start=None)),
      "has_start_sample: a series that begins at t_start, and only a "
      "schedule result's")
fixed_up = P.with_start_sample(old_r)
check(np.array_equal(fixed_up["times"], r0["times"])
      and np.array_equal(fixed_up["r2"], r0["r2"])
      and np.array_equal(fixed_up["mass_drift"], r0["mass_drift"])
      and fixed_up["r2"][0] == 1.0 and fixed_up["mass_drift"][0] == 0.0
      and fixed_up["mean_pred"][0] == fixed_up["mean_true"][0]
      == r0["mean_true"][1]
      and len(fixed_up["mean_true"]) == len(fixed_up["mean_pred"]) == n + 1
      and fixed_up["times"].dtype == old_r["times"].dtype
      and len(old_r["r2"]) == n,
      "with_start_sample puts t_start, R2 = 1 and no drift in front of an "
      "older result's series (and leaves the original as it was)")
check(P.with_start_sample(r0) is r0
      and P.with_start_sample(fixed_up) is fixed_up
      and all(np.array_equal(P.with_start_sample(fixed_up)[k], fixed_up[k])
              for k in ("times", "r2")),
      "and a result that has it comes back as it is: once is enough")
plain_r = dict(old_r, schedule=None)
no_start: Any = P.with_start_sample(dict(old_r, t_start=None))
check(P.with_start_sample(plain_r) is plain_r
      and no_start["times"][0] == old_r["times"][0],
      "a result that is not a schedule result is never given one (a record "
      "of the fixed-step engine has a t_start too)")
for name, res_in in (("older", old_r), ("new", r0)):
    cache_file = os.path.join(BOX, f"cache_{name}.npz")
    P._save_result(cache_file, res_in)
    back = P._load_result(cache_file)
    rec_file = os.path.join(BOX, f"record_{name}.npz")
    flat = P._result_arrays(res_in, film=False)
    flat.update(seed=np.asarray(11), off=np.asarray(0.0),
                lattice=np.asarray(L), fingerprint=np.asarray("fp"),
                cache_key=np.asarray("key"), t_start=np.asarray(100.0),
                t_end=np.asarray(220.0))
    P._write_npz(rec_file, flat, compress=True)
    kept, who = P.load_record(rec_file)
    check(all(np.array_equal(x[k], r0[k]) for x in (back, kept)
              for k in ("times", "r2", "mass_drift"))
          and len(back["mean_true"]) == len(kept["mean_pred"]) == n + 1
          and all(P.has_start_sample(x) for x in (back, kept))
          and back["schedule"]["steps"] == n == kept["schedule"]["steps"],
          f"a {name} result through the rollout cache and through a record "
          f"has the start sample once, and the stride count as before")
# a fixed-step result has no start sample, whatever its record says
fs_r = {**old_r, "schedule": None}
fs_file = os.path.join(BOX, "record_fixed.npz")
flat = P._result_arrays(fs_r, film=False)
flat.update(seed=np.asarray(11), off=np.asarray(0.0), lattice=np.asarray(L),
            fingerprint=np.asarray("fp"), cache_key=np.asarray("key"),
            t_start=np.asarray(100.0), t_end=np.asarray(220.0))
P._write_npz(fs_file, flat, compress=True)
fs_back, _ = P.load_record(fs_file)
check(np.array_equal(fs_back["times"], old_r["times"])
      and len(fs_back["r2"]) == n,
      "a fixed-step record (which names t_start too) is read as it is")
strides_above = int(np.sum(old_r["r2"] >= P.R2_THRESHOLD))
check(P.n_steps_above(r0) == strides_above == P.n_steps_above(old_r)
      and P.seed_summary(r0, 11)["steps_above_threshold"] == strides_above
      and P.seed_summary(r0, 11)["mean_r2"] > P.seed_summary(old_r, 11)[
          "mean_r2"] - 1e-9,
      "the steps above the threshold are strides, with or without the start "
      "(the mean over the curve has it)")

# ---------------------------------------------------------------------------
print("4. a tiny dt-conditioned model end to end")
OUT = os.path.join(BOX, "run")
RESULTS = os.path.join(OUT, "Results_x")
CACHE = os.path.join(OUT, "cache")
SEEDS = P.GROUP_SEEDS[:4]
# the fp.sh run: --views, no --snapshots, to 2500 so that the snapshots of the
# schedule engine (500 among them) differ from the seed engine's
RESULTS_V = os.path.join(OUT, "Results_v{H}")
CACHE_V = os.path.join(OUT, "cache_views")


class Killed(BaseException):
    """A job killed by its time limit: nothing after it runs."""


def run_main(patches=(), views=False):
    restore()
    P.WRITE_LOG, P.WRITE_TIMING = False, True
    P.DIAGNOSTICS = False
    P.LATTICE, P.SIZE_SWEEP = L, (L,)
    P.TUNE_ALPHAS, P.TUNE_DT_MAXS = (0.2, 0.7), (30.0, 50.0)
    P.TUNE_SEEDS = fno_gt.VAL_SEEDS[:2]
    P.ROLLOUT_CELLS = 2 * L * L
    P.HOST_WORKERS = 2
    for name, value in patches:
        setattr(P, name, value)
    argv = ["--model", COND, "--eval-mode", "critical", "--seeds", "4",
            "--start-time", "100", "--end-time", "220", "--snapshots",
            "100,140,220", "--results-dir", RESULTS, "--result-cache", CACHE]
    if views:
        argv = ["--model", COND, "--eval-mode", "critical", "--seeds", "2",
                "--views", "1200,2500", "--results-dir", RESULTS_V,
                "--result-cache", CACHE_V]
    buf, code = io.StringIO(), None
    try:
        with contextlib.redirect_stdout(buf):
            code = P.main(argv)
    except Killed:
        pass
    return code, buf.getvalue()


restore()
tally = {"eval": [], "tune": 0}
real_eval, real_tune = P.evaluate_schedule, P.tune_schedule


def eval_spy(model, seeds, *a, **k):
    tally["eval"].append(list(seeds))
    return real_eval(model, seeds, *a, **k)


def tune_spy(*a, **k):
    tally["tune"] += 1
    return real_tune(*a, **k)


def kill(*a, **k):
    raise Killed()


code, log1 = run_main([("evaluate_schedule", kill),
                       ("tune_schedule", tune_spy)])
check(code is None, "a job killed after its tuning, before the first rollout")
meta = P.read_meta(CACHE)
pick1 = ((meta or {}).get("schedule") or {}).get("+0.0")
check(pick1 is not None and pick1["alpha"] in (0.2, 0.7)
      and pick1["dt_max"] in (30.0, 50.0) and len(pick1["table"]) == 4
      and pick1["source"].startswith("tuned"),
      f"meta.json keeps the pick ({pick1 and (pick1['alpha'], pick1['dt_max'])})"
      f" and the four candidates scored")
check(meta is not None and meta.get("engine") == "schedule"
      and meta.get("seed_set") == "group"
      and meta["eval_seeds"] == list(SEEDS),
      "meta.json: engine schedule, the group seed set (the first four)")
check(tally["tune"] == 1, "tuned once")
files = sorted(os.listdir(REFS)) if os.path.isdir(REFS) else []
want = sorted(f"gt_l{L}_off+0.0_seed{sd}_t100-220_ev4.npz"
              for sd in list(SEEDS) + list(fno_gt.VAL_SEEDS[:2]))
check(files == want,
      f"the references went to FNO_GT_CACHE under the group format's names ({len(files)} "
      f"files)")

code, log2 = run_main([("evaluate_schedule", eval_spy),
                       ("tune_schedule", tune_spy)])
check(code == 0, f"the resumed job finishes (exit {code})")
check(tally["tune"] == 1 and "from an earlier job" in log2,
      "the pick is reused on resume: no second tuning")
meta = P.read_meta(CACHE)
pick2 = ((meta or {}).get("schedule") or {}).get("+0.0")
check(pick2 == pick1, "and meta.json still holds the same pick")
check([len(c) for c in tally["eval"]] == [2, 2],
      f"two chunks of ROLLOUT_CELLS // l^2 = 2 seeds ({tally['eval']})")
assert pick2 is not None
sched = P.schedule_times(pick2["alpha"], pick2["dt_max"])
name = P.case_name(0.0)
recs = [P.record_file(CACHE, name, sd) for sd in SEEDS]
check(all(os.path.isfile(r) for r in recs), "a record per seed")
res, who = P.load_record(recs[0])
check(res["t_start"] == 100.0 and who["t_start"] == 100.0
      and np.allclose(res["times"], sched) and res["times"][0] == 100.0
      and res["r2"][0] == 1.0 and P.has_start_sample(res),
      "its t_start is 100 and its times are the schedule's, absolute, from "
      "the start (R2 = 1)")
check(res["schedule"] is not None
      and abs(res["schedule"]["alpha"] - pick2["alpha"]) < 1e-12
      and res["schedule"]["steps"] == len(sched) - 1,
      f"it carries the schedule it was made with ({res['schedule']})")
check(140.0 in res["times"] and 140.0 in res["snapshots"],
      "the must-hit time 140 is a step and a snapshot")
check(res["video_true"] is None, "no film")

flat = sorted(
    os.path.relpath(os.path.join(b_, f), RESULTS).replace(os.sep, "/")
    for b_, _, fs in os.walk(RESULTS) for f in fs)
tok = S.case_token(0.0)
need = sorted([S.case_file(k, tok) for k in S.FIG_KINDS]
              + [f"l{L}/{f}" for f in S.SIZE_FILES]
              + [f for f in S.RUN_FILES
                 if "r2_sweep" not in f and f != "figure_notes.txt"])
check(flat == need
      and sorted(f for f in os.listdir(RESULTS)
                 if os.path.isdir(os.path.join(RESULTS, f))) == [f"l{L}"],
      f"the run folder is the run's files and one folder for the box size "
      f"({len(flat)} files)")
check(os.path.isdir(os.path.join(CACHE, "extras")),
      "FNO's own outputs are in the cache's extras/")
man = json.dumps(json.load(open(os.path.join(RESULTS, "manifest.json"))))
check('"alpha"' in man and str(pick2["dt_max"]) in man,
      "manifest.json names the schedule")
case_entry = json.load(open(os.path.join(RESULTS, "manifest.json")))[
    "cases"][tok]
check(case_entry["steps"] == len(sched) - 1
      and case_entry["t_start"] == 100.0
      and case_entry["r2_curve"]["t"][0] == 100.0
      and case_entry["r2_curve"]["mean"][0] == 1.0
      and case_entry["schedule"]["steps"] == len(sched) - 1,
      "manifest.json: steps are strides, and the mean R2 curve starts at "
      "t_start with R2 = 1")
mean = np.loadtxt(os.path.join(P.extras_for(RESULTS, os.path.join(
    CACHE, "extras"), False), P.case_name(0.0), P.MEAN_R2_FILE),
    delimiter=",", skiprows=1, ndmin=2)
check(np.allclose(mean[:, 0], sched) and np.all(mean[:, 1] == 4)
      and mean[0, 0] == 100.0 and mean[0, 2] == 1.0,
      "mean_r2.csv: the schedule's times from t_start (the mean R2 there is "
      "1), all four seeds in every mean")

n_eval = len(tally["eval"])
code, log3 = run_main([("evaluate_schedule", eval_spy),
                       ("tune_schedule", tune_spy)])
check(code == 0 and len(tally["eval"]) == n_eval and tally["tune"] == 1
      and "already finished" in log3,
      "a third job finds the case finished: no tuning, no rollout")

# ---------------------------------------------------------------------------
print("4a. a re-prediction keeps the complete cases it already has, drawn "
      "from their records, with no reference left for them")
L_OLD = 24
KEEP_RESULTS = os.path.join(OUT, "Results_keep")
KEEP_CACHE = os.path.join(OUT, "cache_keep")
RESULTS_MAIN, CACHE_MAIN = RESULTS, CACHE
RESULTS, CACHE = KEEP_RESULTS, KEEP_CACHE
code_a, log_a = run_main([("SIZE_SWEEP", (L, L_OLD))])
meta_a = P.read_meta(KEEP_CACHE) or {}
old_name = f"psi0+0.00_l{L_OLD}"
check(code_a == 0 and meta_a.get("planned_cases") == [f"psi0+0.00_l{L}",
                                                       old_name]
      and meta_a["cases"][old_name]["complete"] is True
      and meta_a["cases"][old_name]["lattice"] == L_OLD,
      "a first job over the sizes 16 and 24 completes both cases")
gone = [f for f in os.listdir(REFS) if f.startswith(f"gt_l{L_OLD}_")]
for f in gone:
    os.remove(os.path.join(REFS, f))
check(len(gone) == 4 and not [f for f in os.listdir(REFS)
                              if f.startswith(f"gt_l{L_OLD}_")],
      "the references of the l = 24 case are removed from the cache")
asked = []
real_solve = P.solve_references


def solve_spy(l, off, seeds, device):
    asked.append((l, off))
    return real_solve(l, off, seeds, device)


shutil.rmtree(KEEP_RESULTS)
code_b, log_b = run_main([("SIZE_SWEEP", (L,)),
                          ("solve_references", solve_spy)])
meta_b = P.read_meta(KEEP_CACHE) or {}
check(code_b == 0 and not any(l == L_OLD for l, _ in asked)
      and "kept from an earlier job" in log_b,
      f"the second job, whose grid has the base lattice only, draws the "
      f"l = 24 case from its records and asks for no reference of it "
      f"({asked})")
man_b = json.load(open(os.path.join(KEEP_RESULTS, "manifest.json")))
check(sorted(man_b["cases"]) == sorted(["psi0p0.00", f"l{L_OLD}"])
      and all(os.path.isfile(os.path.join(KEEP_RESULTS, f"l{L_OLD}", n))
              for n in [S.fig_name(k, "psi0p0.00") for k in S.FIG_KINDS]
              + list(S.SIZE_FILES))
      and meta_b["planned_cases"] == [f"psi0+0.00_l{L}", old_name]
      and meta_b["cases"][old_name]["complete"] is True
      and meta_b["cases"][old_name]["cache_key"]
      == meta_a["cases"][old_name]["cache_key"],
      "its folder has the kept case's figures in l24/ and its manifest "
      "both cases; meta.json plans both, and the kept one stays complete "
      "with its entry as it was")
RESULTS, CACHE = RESULTS_MAIN, CACHE_MAIN

# ---------------------------------------------------------------------------
print("4b. the fp.sh run (--views, no --snapshots) and its redraw in a fresh "
      "process")
WANT_V = {1200: [100.0, 500.0, 1000.0, 1200.0],
          2500: [100.0, 500.0, 1000.0, 2000.0, 2500.0]}
# the manifest lists the columns of snapshots_<case>.png: those of
# S.SNAPSHOT_COLUMNS that the run has (the end of a view is not one)
COLS_V = {h: S.snapshot_columns(v) for h, v in WANT_V.items()}
check(SAVED["PER_SEED_FIGURES"] is True,
      "the run keeps the production PER_SEED_FIGURES (the best seed's figures)")
code, logv = run_main(views=True)
check(code == 0, f"the run to the views 1200 and 2500 finishes (exit {code})")
check("4 trajectories ready (4 solved now)" in logv,
      "the prewarm solved the 2 test and the 2 tune references")
live = {h: json.load(open(os.path.join(RESULTS_V.format(H=f"{h:g}"),
                                       "manifest.json"))) for h in WANT_V}
check(all(live[h]["snapshot_times"] == COLS_V[h] for h in WANT_V),
      f"each view has the snapshots of a run to its horizon: "
      f"{ {h: live[h]['snapshot_times'] for h in WANT_V} }")
P.ENGINE = "seed"
seed_set = P.default_snapshots(100.0, 2500.0)
P.ENGINE = "schedule"
check(500.0 not in seed_set and 500.0 in live[2500]["snapshot_times"],
      f"(a fresh process starts as the seed engine, whose set for 2500 is "
      f"{seed_set}, without 500)")
extras_v = os.path.join(CACHE_V, "extras")
bests = [d for h, ds, fs in os.walk(extras_v) for d in ds
         if d.startswith("best_seed")]
check(len(bests) == 2, f"PER_SEED_FIGURES: a best seed's folder per view "
                      f"({bests})")
name_v = P.case_name(0.0)
rec, who = P.load_record(P.record_file(CACHE_V, name_v, P.GROUP_SEEDS[0]))
check(rec["gt_cached"] is False and rec["gt_seconds"] > 0.0
      and who["t_start"] == 100.0,
      f"the record says its reference was solved in this job and what that "
      f"cost ({rec['gt_seconds']:.2f} s), not that it came from the cache")


def timing_of(folder):
    names = [f for f in os.listdir(folder) if f.endswith("_timing.json")]
    if not names:
        return {}, ""
    with open(os.path.join(folder, names[0])) as fh:
        meta = json.load(fh)["meta"]
    with open(os.path.join(folder, names[0][:-5] + ".txt")) as fh:
        return meta, fh.read()


tm, tx = timing_of(os.path.join(extras_v, "Results_v2500"))
check(tm.get("n_ground_truths_solved") == 2
      and tm.get("n_ground_truths_from_cache") == 0 and tm["dt_step"] is None,
      f"the timing report counts the 2 references as solved "
      f"({tm.get('n_ground_truths_solved')} of {tm.get('n_rollouts')})")

REDRAWN = os.path.join(OUT, "redrawn_v{H}")
proc = subprocess.run(
    [sys.executable, os.path.join(ROOT, "make_submission.py"), "--records",
     CACHE_V, "--out", REDRAWN], cwd=BOX, capture_output=True, text=True,
    encoding="utf-8", errors="replace", timeout=900)
check(proc.returncode == 0,
      f"make_submission.py --records redraws the run in a fresh process "
      f"(exit {proc.returncode}) {proc.stderr[-300:] if proc.returncode else ''}")
for h in WANT_V:
    folder = REDRAWN.format(H=f"{h:g}")
    again = (json.load(open(os.path.join(folder, "manifest.json")))
             if os.path.isfile(os.path.join(folder, "manifest.json")) else {})
    same = {k: again.get(k) == live[h][k]
            for k in ("snapshot_times", "snapshot_times_main",
                      "schedules", "t_end", "t_start")}
    check(all(same.values()) and again.get("snapshot_times") == COLS_V[h],
          f"the redrawn manifest of the {h} view has the live run's snapshot "
          f"times {again.get('snapshot_times')} ({[k for k in same if not same[k]]} differ)")
tm2, tx2 = timing_of(os.path.join(extras_v, "redrawn_v2500"))
check(tm2.get("n_ground_truths_solved") == 2 and tm2.get("dt_step") is None
      and "a stride of the schedule" in tx2 and "a stride of the schedule" in tx,
      "and its timing report describes the schedule engine, as the live one")

# ---------------------------------------------------------------------------
print("5. the diagnostics take a dt")
restore()
P.START_TIME, P.END_TIME = 100.0, 220.0
P.DIAG_TILE, P.DIAG_TILE_T = L, 100.0
P.DIAG_LOWK_L, P.DIAG_LOWK_SEEDS, P.DIAG_LOWK_T_END = L, 2, None
stand = Stand([0.0, 0.0])
tile = quiet(P.tile_invariance, stand, "cpu")
check(tile["dt"] == 10.0 and {float(d[0]) for d in stand.dts} == {10.0}
      and tile["t"] == 100.0 and tile["seed"] == fno_gt.VAL_SEEDS[0]
      and tile["max_abs"] == 0.0,
      "the tile test asks a dt-conditioned model for a step of 10")
stand = Stand([0.0, 0.0])
low = quiet(P.lowk_check, stand, "cpu")
check(low["dt"] == 48.0 and {float(d[0]) for d in stand.dts} == {48.0}
      and low["seeds"] == list(fno_gt.VAL_SEEDS[:2]) and "172" in low["gain"],
      f"the low-k gain a step of 48, from the last frame with one "
      f"({list(low['gain'])})")
text = P.diagnostics_text({"tile": tile, "lowk": low, "seconds": 0.0})
check("dt=10" in text and "dt=48" in text, "and the log says so")

seen = {}


def fake_batch(lat, seeds, off, t_end, dt_step, cache_dir=None, device=None):
    seen.update(lat=lat, seeds=list(seeds), t_end=t_end)
    t = np.arange(0.0, t_end + 0.5, dt_step)
    return [(t, np.zeros((len(t), 1, 1))) for _ in seeds]


def fake_one(l, seed, off, t_end, dt_step, cache_dir=None, device=None):
    seen.update(tile_seed=seed)
    t = np.arange(0.0, t_end + 0.5, dt_step)
    g = np.random.default_rng(1).uniform(-0.1, 0.1, (len(t), l, l))
    return t, g


fixed = tiny_model(False)
P.ch_ground_truth_batch, P.ch_ground_truth = fake_batch, fake_one
P.END_TIME = 2000.0
P.DIAG_LOWK_T_END = None
T.lowk_gains = lambda *a, **k: {}
tile = quiet(P.tile_invariance, fixed, "cpu")
low = quiet(P.lowk_check, fixed, "cpu")
check(T.VAL_T_END == 10000.0 and T.VAL_SEEDS == fno_gt.VAL_SEEDS,
      "(fno_train's own window is the dt-conditioned one now)")
check(seen["t_end"] == FS["VAL_T_END"] == 2000.0
      and seen["seeds"] == list(FS["VAL_SEEDS"][:P.DIAG_LOWK_SEEDS])
      and seen["tile_seed"] == FS["VAL_SEEDS"][0],
      "a fixed-step model: the low-k check reads FIXED_STEP's window and "
      "seeds, not fno_train.VAL_T_END = 10000")
check(tile["dt"] is None and low["dt"] == P.DT_STEP == 1.0,
      "and keeps today's steps (the model's own)")
restore()

# ---------------------------------------------------------------------------
print("6. the lock-step engine of a fixed-step model")
restore()
P.START_TIME, P.END_TIME = 5.0, 40.0
P.SNAPSHOT_TIMES = (5.0, 10.0, 40.0)
P.DIAG_EVERY, P.WRITE_VIDEO, P.HOST_WORKERS = 2, False, 1


class Fixed(nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.gain, self.calls = gain, 0

    def forward(self, x):
        self.calls += 1
        return x * self.gain * self.w


calm, blow = Fixed(1.0), Fixed(10.0)
out_calm = quiet(P.evaluate_seeds, calm, [3, 4], "cpu", 0.0, 8)
out_blow = quiet(P.evaluate_seeds, blow, [3, 4], "cpu", 0.0, 8)
check(calm.calls == 35 and all(len(r["r2"]) == 35 for r in out_calm),
      "no divergence: all 35 steps")
check(blow.calls < 35 and all(r["diverged_at"] is not None for r in out_blow)
      and all(len(r["r2"]) == blow.calls for r in out_blow)
      and all(r["r2"][-1] == P.R2_FLOOR for r in out_blow),
      f"every seed diverged: the loop ended after {blow.calls} of 35 steps and "
      f"the series stop there, the last scored -1")
check(all(r["video_true"] is None for r in out_blow), "and no film")
restore()

check(project_state() == PROJECT, "nothing was written into the project")
os.chdir(SCRATCH)
shutil.rmtree(BOX, ignore_errors=True)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
