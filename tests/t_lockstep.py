"""The lock-step engine, horizon views and the long-horizon command line."""
import contextlib
import hashlib
import io as _io
import json
import os
import shutil
import sys
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from _paths import fixed_step_model, ROOT, SCRATCH, project_state

# a fixed-step network: this suite runs the seed engine, and the project's own
# checkpoints are dt-conditioned (see fixed_step_model in _paths.py)
REAL_MODEL = fixed_step_model()

os.chdir(ROOT)
import ch_gpu as C
import fno_predict as P
# this suite is about the temporary cache and what supersedes it; the
# records and meta.json that t_records.py covers would sit in the same
# folder and change what it counts
P.RECORDS = False
import fno_train as T
T.use_fixed_step()

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(_io.StringIO()):
        return fn(*a, **k)


KEYS = ("START_TIME", "END_TIME", "SNAPSHOT_TIMES", "VIDEO_EVERY",
        "DIAG_EVERY", "ENGINE", "EVAL_BATCH", "VIEWS", "RESULTS_DIR",
        "RESULTS_TEMPLATE", "LOG_TIME", "CACHE_END_TIME", "WRITE_VIDEO",
        "EARLY_STOP", "MODEL_PATH", "RESULT_CACHE", "GT_CACHE",
        "ch_ground_truth", "HOST_WORKERS", "LATTICE", "SIZE_SWEEP",
        "EVAL_SEEDS", "SWEEP_SEEDS", "EVAL_MODE", "WRITE_LOG", "WRITE_TIMING",
        "PER_SEED_FIGURES", "KEEP_CACHE", "REDO", "load_checkpoint",
        "CACHE_RESULTS", "WRITE_SUBMISSION",
        "VIDEO_SCHEDULE", "VIDEO_DTYPE", "SNAPSHOTS_EXPLICIT", "_RUN_SNAPSHOTS",
        "figure_sweeps")
SAVED = {k: getattr(P, k) for k in KEYS}
# what the file itself says, before the next lines point it at SCRATCH
DEFAULTS = {k: SAVED[k] for k in ("MODEL_PATH", "RESULTS_DIR", "RESULT_CACHE",
                                  "GT_CACHE")}
# The folders default to names relative to the project, where this suite
# runs: whatever falls back on them lands in SCRATCH instead.
SAVED.update(MODEL_PATH=REAL_MODEL,
             RESULTS_DIR=os.path.join(SCRATCH, "lockstep_results"),
             RESULT_CACHE=os.path.join(SCRATCH, "lockstep_result_cache"),
             GT_CACHE=os.path.join(SCRATCH, "lockstep_gt_cache"))
PROJECT = project_state()


def restore():
    for k, v in SAVED.items():
        setattr(P, k, v)


restore()
# these runs are of stand-in models, and the diagnostics would solve real
# references for them
P.DIAGNOSTICS = False


class Smooth(nn.Module):
    """Elementwise, so a batch computes bitwise what each seed would alone."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.999))

    def forward(self, x):
        return x * self.w - 0.0005 * (x * x * x)


class Blowup(Smooth):
    """Grows each seed once its own |psi| passes a level: seeds diverge at
    different times.  The reduction is within one seed, so still exact."""

    def forward(self, x):
        big = (x.abs().amax(dim=(-2, -1), keepdim=True) > 1.0).to(x.dtype)
        return x * (1.1 + 0.6 * big)


class TurnsBad(Smooth):
    """Smooth for its first n calls, then the sign flips: R2 falls below the
    threshold at a known step, without diverging."""

    def __init__(self, n):
        super().__init__()
        self.n, self.calls = n, 0

    def forward(self, x):
        self.calls += 1
        y = super().forward(x)
        return y if self.calls <= self.n else -y


class Spy(Blowup):
    """Blowup that records, per call, each seed's largest |input|."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):
        self.seen.append(x.abs().amax(dim=(-3, -2, -1)).detach().clone())
        return super().forward(x)


def torch_truth(l, seed, off=0.0, t_end=2000.0, dt_step=1.0, cache_dir=None,
                device="cpu", verbose=True):
    """ch_ground_truth through fno_train.ch_step on this device, so the
    per-seed engine sees exactly the truth the lock-step one streams."""
    every = int(round(dt_step / T.DT))
    n_total = int(round(t_end / T.DT))
    p = torch.as_tensor(T.initial_condition(l, seed, off), dtype=torch.float64)
    frames = [p.numpy().copy()]
    for n in range(1, n_total + 1):
        p = T.ch_step(p)
        if n % every == 0:
            frames.append(p.numpy().copy())
    return np.arange(len(frames)) * dt_step, np.stack(frames)


def same(a, b):
    if isinstance(a, dict):
        return (isinstance(b, dict) and set(a) == set(b)
                and all(same(a[k], b[k]) for k in a))
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(
            same(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a):
        return np.isnan(b)
    return a == b


TIMING = ("gt_seconds", "fno_seconds", "gt_cached")


def differences(a, b):
    return sorted(k for k in set(a) | set(b)
                  if k not in TIMING and not same(a.get(k), b.get(k)))


def cut_like(a, b):
    """The per-seed result ``a`` cut where the batch's ``b`` ends: the
    lock-step loop ends once every seed has diverged (or stopped), the
    per-seed engine goes on to END_TIME with the diverged scored -1."""
    n = len(b["times"])
    if len(a["times"]) <= n:
        return a
    out = dict(a)
    for k in ("times", "r2", "mass_drift", "mean_true", "mean_pred"):
        out[k] = np.asarray(a[k])[:n]
    nd = len(b["diag_times"])
    out["diag_times"] = np.asarray(a["diag_times"])[:nd]
    for k in ("L_true", "L_pred"):
        out[k] = {name: np.asarray(v)[:nd] for name, v in a[k].items()}
    for k in ("energy_true", "energy_pred"):
        out[k] = np.asarray(a[k])[:nd]
    out["snapshots"] = {t: v for t, v in a["snapshots"].items()
                        if t in b["snapshots"]}
    if a.get("video_t") is not None and len(a["video_t"]):
        keep = np.asarray(a["video_t"]) <= b["times"][-1] + 1e-9
        out["video_t"] = np.asarray(a["video_t"])[keep]
        for k in ("video_true", "video_pred"):
            out[k] = np.asarray(a[k])[keep]
    return out


def setup(end=30.0, early=False):
    restore()
    P.START_TIME, P.END_TIME = 5.0, end
    P.SNAPSHOT_TIMES = tuple(t for t in (5.0, 10.0, 20.0, 30.0) if t <= end)
    P.VIDEO_EVERY, P.DIAG_EVERY, P.WRITE_VIDEO = 4, 3, True
    P.EARLY_STOP = early
    P.GT_CACHE = ""
    P.ch_ground_truth = torch_truth


seeds, L = (3, 7, 11), 16

print("1. the streamed truth is the stored truth, bitwise")
p = torch.as_tensor(np.stack([T.initial_condition(L, s, 0.2) for s in seeds]),
                    dtype=torch.float64)
for _ in range(700):
    p = T.ch_step(p)
alone = []
for s in seeds:
    q = torch.as_tensor(T.initial_condition(L, s, 0.2), dtype=torch.float64)
    for _ in range(700):
        q = T.ch_step(q)
    alone.append(q.numpy())
check(np.array_equal(p.numpy(), np.stack(alone)),
      "a batch of seeds advances bitwise as each seed alone")
p0 = np.stack([T.initial_condition(L, s, 0.2) for s in seeds])
kept = C.integrate(torch.as_tensor(p0, dtype=torch.float64), 700, 1)
check(kept is not None and np.array_equal(kept[:, 699], p.numpy()),
      "and bitwise as ch_gpu.integrate, an independent copy of ch.py's update")
if torch.cuda.is_available():
    try:
        ref = T._integrate(T.initial_condition(L, seeds[0], 0.2), 700, 100, 8,
                           "cuda")
        got = torch.as_tensor(T.initial_condition(L, seeds[0], 0.2),
                              dtype=torch.float64, device="cuda")
        for _ in range(700):
            got = T.ch_step(got)
        check(np.array_equal(ref[7], got.cpu().numpy()),
              "on CUDA, ch_step is exactly _integrate's update")
    except torch.cuda.OutOfMemoryError:
        print("      (CUDA busy; skipped)")

print("\n2. the lock-step engine records what evaluate_seed records")
for name, model, early, off in (
        ("smooth", Smooth(), False, 0.2),
        ("smooth, critical composition", Smooth(), False, 0.0),
        ("seeds diverging at different times", Blowup(), False, 0.2),
        ("EARLY_STOP", Blowup(), True, 0.2)):
    setup(early=early)
    per_seed = [quiet(P.evaluate_seed, model, s, "cpu", off, L) for s in seeds]
    batch = quiet(P.evaluate_seeds, model, seeds, "cpu", off, L)
    diffs = [differences(cut_like(a, b), b)
             for a, b in zip(per_seed, batch)]
    check(all(not d for d in diffs),
          f"{name}: every seed's dict matches, key for key "
          f"{[d for d in diffs if d][:1]}")
    if name.startswith("seeds"):
        div = [r["diverged_at"] for r in batch]
        check(len({d for d in div if d is not None}) > 1,
              f"(the seeds really diverged at different times: {div})")
        rate = [r["fno_seconds"] / r["fno_steps"] for r in batch]
        check(np.allclose(rate, rate[0], rtol=1e-9)
              and len({r["fno_steps"] for r in batch}) > 1,
              "each seed is charged for its own model steps, at one rate")
    if name == "EARLY_STOP":
        check(any(len(r["times"]) < 25 for r in batch),
              "(and stopped early)")
setup()
batch1 = quiet(P.evaluate_seeds, Smooth(), seeds, "cpu", 0.2, L)
P.HOST_WORKERS = 1
batch2 = quiet(P.evaluate_seeds, Smooth(), seeds, "cpu", 0.2, L)
check(all(not differences(a, b) for a, b in zip(batch1, batch2)),
      "the host thread pool changes nothing")
check(all(r["fno_steps"] == 25 and r["gt_steps"] == 3000
          and r["gt_seconds"] > 0 and r["fno_seconds"] > 0 for r in batch1),
      "step counts as evaluate_seed's, timings positive and per seed")

setup()
spy = Spy()
res = quiet(P.evaluate_seeds, spy, seeds, "cpu", 0.2, L)
seen = torch.stack(spy.seen)                      # (calls, seeds)
fed = [float(seen[int(round(r["diverged_at"] - P.START_TIME)):, i].max())
       for i, r in enumerate(res) if r["diverged_at"] is not None
       and int(round(r["diverged_at"] - P.START_TIME)) < len(seen)]
check(fed and all(v == 0.0 for v in fed),
      f"a diverged seed's row is zeroed, never fed back to the model {fed}")
check(len(seen) == max(r["fno_steps"] for r in res),
      f"once no seed is running the model is not called "
      f"({len(seen)} calls for at most {max(r['fno_steps'] for r in res)} "
      f"steps)")

setup(early=True)
per_seed = [quiet(P.evaluate_seed, TurnsBad(2), s, "cpu", 0.2, L)
            for s in seeds]
real_step, n_solver = T.ch_step, [0]


def counted(q):
    n_solver[0] += 1
    return real_step(q)


T.ch_step = counted
try:
    batch = quiet(P.evaluate_seeds, TurnsBad(2), seeds, "cpu", 0.2, L)
finally:
    T.ch_step = real_step
check(all(not differences(a, b) for a, b in zip(per_seed, batch))
      and all(r["stopped_early"] for r in batch),
      "with EARLY_STOP and every seed stopped, the dicts still match")
check(n_solver[0] == 500 + 3 * 100,
      f"and the loop ends there: {n_solver[0]} solver steps (the whole "
      f"window is 3000)")

print("\n3. a view is a shorter run")
for name, model in (("smooth", Smooth()), ("diverging", Blowup())):
    setup(end=30.0)
    full = quiet(P.evaluate_seeds, model, seeds, "cpu", 0.2, L)
    for h in (10.0, 20.0, 25.0):
        setup(end=h)
        native = quiet(P.evaluate_seeds, model, seeds, "cpu", 0.2, L)
        cut = [P.truncate_results(r, h) for r in full]
        diffs = [differences(a, b) for a, b in zip(cut, native)]
        check(all(not d for d in diffs),
              f"{name}, view {h:g} of a run to 30 equals a run to {h:g} "
              f"{[d for d in diffs if d][:1]}")
setup(end=30.0)
full = quiet(P.evaluate_seeds, TurnsBad(2), seeds, "cpu", 0.2, L)
setup(end=7.0)
native = quiet(P.evaluate_seeds, TurnsBad(2), seeds, "cpu", 0.2, L)
cut = [P.truncate_results(r, 7.0) for r in full]
diffs = [differences(a, b) for a, b in zip(cut, native)]
check(all(r["stopped_early"] and 7.0 < r["t_stop"] for r in full)
      and all(not r["stopped_early"] and r["t_stop"] == 7.0 for r in cut)
      and all(not d for d in diffs),
      f"R2 falling below the threshold at t = {full[0]['t_stop']:g}, after "
      f"the view to 7, is not a stop in that view {diffs[:1]}")

setup(end=30.0)
full = quiet(P.evaluate_seeds, Smooth(), seeds, "cpu", 0.2, L)
setup(end=30.0)
P.SNAPSHOT_TIMES = (5.0, 20.0, 30.0)
native = quiet(P.evaluate_seeds, Smooth(), seeds, "cpu", 0.2, L)
cut = [P.truncate_results(r, 30.0, snaps=P.SNAPSHOT_TIMES) for r in full]
check(all(sorted(r["snapshots"]) == [5.0, 20.0, 30.0] for r in cut)
      and all(not differences(a, b) for a, b in zip(cut, native)),
      "a view keeps only its own snapshot times, not every view's")

for dtype in (np.float32, np.float16):
    setup(end=30.0)
    P.VIDEO_SCHEDULE, P.VIDEO_DTYPE = ((10.0, 2), (30.0, 4)), dtype
    full = quiet(P.evaluate_seeds, Smooth(), seeds, "cpu", 0.2, L)
    good = True
    for h, e in P.VIDEO_SCHEDULE:
        setup(end=h)
        P.VIDEO_EVERY, P.VIDEO_DTYPE = e, dtype
        native = quiet(P.evaluate_seeds, Smooth(), seeds, "cpu", 0.2, L)
        cut = [P.truncate_results(r, h, e) for r in full]
        good = good and all(not differences(a, b)
                            for a, b in zip(cut, native))
    check(good and full[0]["video_true"].dtype == dtype,
          f"one film with every view's cadence gives each view its own "
          f"({np.dtype(dtype).name} frames)")

print("\n4. the cache key")
restore()
st = os.stat(P.MODEL_PATH)
old = (st.st_size, int(st.st_mtime), 128, 0.0, P.START_TIME, P.END_TIME,
       P.EARLY_STOP, P.R2_THRESHOLD, P.DIAG_EVERY, P.DIVERGE_PSI, P.R2_FLOOR,
       tuple(sorted(P.SNAPSHOT_TIMES)))
legacy = hashlib.md5(repr(old).encode()).hexdigest()[:12]
with mock.patch.object(P, "numerics_now", lambda: dict(P.OLD_NUMERICS)):
    check(P._cache_key(0.0, 128) == legacy,
          "with the numerics of every run before the stamp the key is the "
          "one those runs had, so their caches still hit")
today = P._cache_key(0.0, 128)
check(today != legacy,
      "with today's (TF32 off) it is another, so a rollout of other numerics "
      "is not taken for this run's")
P.ENGINE = "lockstep"
k_lock = P._cache_key(0.0, 128)
P.ENGINE, P.VIDEO_EVERY = "seed", 20
k_video = P._cache_key(0.0, 128)
check(len({k_lock, k_video, today}) == 3,
      "the lock-step engine and another film cadence get their own keys")
restore()
P.END_TIME, P.SNAPSHOT_TIMES = 20000.0, (100.0, 1000.0, 20000.0)
VW = os.path.join(SCRATCH, "views")        # never the project's folders
P.ENGINE, P.RESULTS_TEMPLATE = "lockstep", os.path.join(VW, "r_P{H}")
k_full = P._cache_key(0.0, 128)
with P.view_context(1000.0):
    k_view = P._cache_key(0.0, 128)
    inside = (P.END_TIME, P.SNAPSHOT_TIMES, P.RESULTS_DIR)
check(k_view == k_full
      and inside == (1000.0, (100.0, 500.0, 1000.0),
                     os.path.join(VW, "r_P1000"))
      and P.END_TIME == 20000.0,
      "inside a view the horizon, snapshots (a run to 1000's) and folder "
      "change but the cache key is the full run's")
P.SNAPSHOTS_EXPLICIT = True
with P.view_context(1000.0):
    explicit = P.SNAPSHOT_TIMES
check(explicit == (100.0, 1000.0),
      "with --snapshots a view keeps the given times up to its horizon")
P.SNAPSHOTS_EXPLICIT = False
P.VIDEO_SCHEDULE, P.VIDEO_EVERY = ((1000.0, 10), (20000.0, 100)), 100
before = {k: getattr(P, k) for k in (
    "END_TIME", "SNAPSHOT_TIMES", "RESULTS_DIR", "CACHE_END_TIME",
    "_RUN_SNAPSHOTS", "VIDEO_EVERY")}
film = None
try:
    with P.view_context(1000.0):
        film = P.VIDEO_EVERY
        raise RuntimeError("a view failing half way")
except RuntimeError:
    pass
after = {k: getattr(P, k) for k in before}
check(film == 10 and after == before,
      f"a view films at its own cadence ({film}) and, even when it fails, "
      f"leaves every setting as it found it")
k_sched = P._cache_key(0.0, 128)
P.VIDEO_SCHEDULE = None
k_plain = P._cache_key(0.0, 128)
P.VIDEO_SCHEDULE = ((1000.0, 10), (20000.0, 100))
P.VIDEO_DTYPE = np.float16
k_half = P._cache_key(0.0, 128)
check(len({k_sched, k_plain, k_half}) == 3,
      "a film per view, and its frame type, are part of the cache key")
restore()
P.RESULTS_TEMPLATE = os.path.join(VW, "st_P{H}")
P.START_TIME, P.END_TIME = 5.0, 40.0
P._RUN_STAMPS.clear()
stamps = {}
for h in (10.0, 30.0):
    with P.view_context(h):
        stamps[h] = P.run_stamp()["window"]
check(stamps[10.0]["end"] == 10.0 and stamps[30.0]["end"] == 30.0
      and all(t <= 10.0 for t in stamps[10.0]["snapshots"]),
      f"each view's folder is stamped with its own window {stamps}")
P._RUN_STAMPS.clear()
restore()

print("\n5. the command line")
J = os.path.join
DEFAULT_MODEL = J("Work", "models", "train_t0500", "fno_ch.pt")
check(DEFAULTS == {"MODEL_PATH": DEFAULT_MODEL,
                   "RESULTS_DIR": J("Results", "train_t0500_pred_t02000"),
                   "RESULT_CACHE": J("Work", "rollout_cache",
                                     "train_t0500_pred_t02000"),
                   "GT_CACHE": J("Work", "gt_cache")},
      f"the file's own defaults: the t = 500 model, and the results and "
      f"rollout cache of its run to t = 2000 ({DEFAULTS})")
# The names a run gets come from the checkpoint's own name, so the flags are
# tried against the default one (the file this suite runs on is somewhere
# else, and may not even be named for its training horizon).
BASE5 = dict(SAVED, MODEL_PATH=DEFAULT_MODEL)


def cli(argv, **settings):
    restore()
    P.MODEL_PATH = DEFAULT_MODEL
    for k, v in settings.items():
        setattr(P, k, v)
    # apply_overrides asks the file at MODEL_PATH which engine it needs; in
    # the project that file is a dt-conditioned model (since 2026-10-26),
    # which means the schedule engine, and these are the flags of the
    # fixed-step engines, so the model is taken to be a fixed-step one
    real = P.checkpoint_conditioned
    P.checkpoint_conditioned = lambda path: False
    try:
        quiet(P.apply_overrides, P.parse_args(argv))
    finally:
        P.checkpoint_conditioned = real
    return {k: getattr(P, k) for k in KEYS}


g = cli([])
check(all(g[k] == BASE5[k] for k in KEYS), "no flags change nothing")
g = cli(["--end-time", "20000"])
check(g["ENGINE"] == "lockstep" and g["VIDEO_EVERY"] == 100
      and g["SNAPSHOT_TIMES"] == (100.0, 1000.0, 2000.0, 5000.0, 10000.0,
                                  20000.0)
      and g["RESULTS_DIR"] == J("Results", "train_t0500_pred_t20000")
      and g["RESULT_CACHE"] == J("Work", "rollout_cache",
                                 "train_t0500_pred_t20000"),
      f"--end-time 20000 scales everything and names its folder and cache "
      f"for the run: {g['SNAPSHOT_TIMES']}, every {g['VIDEO_EVERY']}, "
      f"{g['RESULTS_DIR']}, {g['RESULT_CACHE']}")
g = cli(["--end-time", "10000"])
check(g["SNAPSHOT_TIMES"] == (100.0, 1000.0, 2000.0, 5000.0, 10000.0)
      and g["VIDEO_EVERY"] == 50, "and 10000 likewise")
g = cli(["--model", "fno_ch_T1000.pt", "--views", "2000,10000,20000"])
check(g["END_TIME"] == 20000.0 and g["VIEWS"] == (2000.0, 10000.0, 20000.0)
      and g["RESULTS_TEMPLATE"] == J("Results", "train_t1000_pred_t{H:0>5}")
      and g["RESULTS_DIR"] == J("Results", "train_t1000_pred_t20000")
      and g["RESULT_CACHE"] == J("Work", "rollout_cache",
                                 "train_t1000_pred_t20000")
      and {2000.0, 10000.0, 20000.0} <= set(g["SNAPSHOT_TIMES"]),
      "--views runs to the longest, names a folder per view from the "
      "model, and the rollout cache for the longest")
check([g["RESULTS_TEMPLATE"].format(H=f"{h:g}") for h in g["VIEWS"]]
      == [J("Results", f"train_t1000_pred_t{h}")
          for h in ("02000", "10000", "20000")],
      "the template fills in to a five-digit horizon, 02000 for 2000")
check(g["VIDEO_SCHEDULE"] == ((2000.0, 10), (10000.0, 50), (20000.0, 100))
      and g["VIDEO_DTYPE"] == np.float16 and 500.0 in g["SNAPSHOT_TIMES"],
      "and films every view at its own cadence, keeping t = 500 for the 2000 "
      "view")
views_ok = []
n_model = 19900
vt = np.array([100.0 + k + 1 for k in range(n_model)
               if P._video_frame(k, 100.0 + k + 1)])
fake = {"times": np.arange(101.0, 20001.0), "r2": np.ones(n_model),
        "diag_times": np.arange(101.0, 20001.0, 10.0), "L_true": {},
        "L_pred": {}, "energy_true": np.ones(1990),
        "energy_pred": np.ones(1990), "snapshots": {}, "video_t": vt,
        "video_true": np.zeros((vt.size, 1, 1), np.float16),
        "video_pred": np.zeros((vt.size, 1, 1), np.float16),
        "diverged_at": None, "stopped_early": False, "t_stop": 20000.0,
        "fno_steps": n_model, "fno_seconds": 1.0, "gt_steps": 2000000,
        "gt_seconds": 1.0}
P.RESULTS_TEMPLATE = os.path.join(VW, "results_T1000_P{H}")
for h, e in g["VIDEO_SCHEDULE"]:
    with P.view_context(h):
        views_ok.append((h, P.SNAPSHOT_TIMES, P.VIDEO_EVERY))
    got = P.truncate_results(fake, h, e)["video_t"]
    want = [100.0 + k + 1 for k in range(int(h - 100.0))
            if k == 0 or (k + 1) % e == 0]
    views_ok.append(list(got) == want)
shutil.rmtree(VW, ignore_errors=True)
check(views_ok == [(2000.0, (100.0, 500.0, 1000.0, 2000.0), 10), True,
                   (10000.0, (100.0, 1000.0, 2000.0, 5000.0, 10000.0), 50),
                   True,
                   (20000.0, (100.0, 1000.0, 2000.0, 5000.0, 10000.0,
                              20000.0), 100), True],
      "each view has the snapshots and the film (191 frames at 2000) of a "
      "run to its horizon")
gx = cli(["--model", J("Work", "models", "train_t1000", "fno_ch.pt"),
          "--views", "2000,10000"])
check(gx["RESULTS_TEMPLATE"] == J("Results", "train_t1000_pred_t{H:0>5}")
      and gx["RESULTS_DIR"] == J("Results", "train_t1000_pred_t10000")
      and gx["RESULT_CACHE"] == J("Work", "rollout_cache",
                                  "train_t1000_pred_t10000"),
      "a model in its own folder names the run, and a shorter longest view "
      "names the cache")
gx = cli(["--views", "2000,10000", "--end-time", "20000"])
check(gx["RESULT_CACHE"] == J("Work", "rollout_cache",
                              "train_t0500_pred_t20000")
      and gx["RESULTS_TEMPLATE"] == J("Results", "train_t0500_pred_t{H:0>5}"),
      "the cache holds what the rollout ran to (END_TIME), whatever the "
      "views")
gx = cli(["--views", "2000,10000", "--start-time", "200"])
check(gx["RESULTS_TEMPLATE"] == J("Results", "train_t0500_pred_t00200-{H:0>5}")
      and gx["RESULT_CACHE"] == J("Work", "rollout_cache",
                                  "train_t0500_pred_t00200-10000"),
      "views from another start say so in the template and the cache")
g = cli(["--end-time", "2000"])
check(g["ENGINE"] == "seed" and g["VIDEO_EVERY"] == 10
      and g["SNAPSHOT_TIMES"] == (100.0, 500.0, 1000.0, 2000.0),
      "--end-time 2000 keeps today's engine, cadence and snapshots")
for argv, run in ((["--start-time", "200", "--end-time", "4000"],
                   "train_t0500_pred_t00200-04000"),
                  (["--start-time", "100", "--end-time", "4000"],
                   "train_t0500_pred_t04000"),
                  (["--end-time", "4000"], "train_t0500_pred_t04000"),
                  (["--model", J("Work", "models", "train_t1000",
                                 "fno_ch.pt")], "train_t1000_pred_t02000"),
                  (["--model", J("Work", "models", "train_t0300_lap_mine",
                                 "fno_ch.pt"), "--end-time", "10000"],
                   "train_t0300_lap_mine_pred_t10000"),
                  (["--model", "fno_ch_T1000.pt"], "train_t1000_pred_t02000"),
                  (["--model", "fno_ch.pt", "--end-time", "10000"],
                   "fno_ch_pred_t10000"),
                  (["--model", "other.pt", "--end-time", "10000"],
                   "other_pred_t10000"),
                  (["--model", "other.pt"], "other_pred_t02000")):
    g = cli(argv)
    check(g["RESULTS_DIR"] == J("Results", run)
          and g["RESULT_CACHE"] == J("Work", "rollout_cache", run),
          f"{' '.join(argv)} -> {g['RESULTS_DIR']}, {g['RESULT_CACHE']}")
g = cli(["--model", "fno_ch_T1000.pt"])
check(g["SNAPSHOT_TIMES"] == SAVED["SNAPSHOT_TIMES"] and g["ENGINE"] == "seed"
      and g["VIDEO_EVERY"] == 10,
      "--model alone changes only the folders, not the run")
g = cli(["--end-time", "4000", "--result-cache", "rc_x",
         "--results-dir", "r_x"])
check(g["RESULT_CACHE"] == "rc_x" and g["RESULTS_DIR"] == "r_x",
      "and a folder or cache given on the command line wins over the names")
g = cli(["--end-time", "4000", "--results-dir", "r_{H:0>5}"])
check(g["RESULTS_DIR"] == "r_04000" and g["RESULTS_TEMPLATE"] is None,
      "a {H:0>5} in --results-dir without views is filled with the end time")
g = cli(["--views", "2000,20000", "--results-dir", "r_{H:0>5}"])
check(g["RESULTS_TEMPLATE"] == "r_{H:0>5}" and g["RESULTS_DIR"] == "r_20000",
      "and with views it is the template, whatever the {H} is spelt")
g = cli(["--end-time", "4000"], RESULT_CACHE="mine")
check(g["RESULT_CACHE"] == J("Work", "rollout_cache",
                             "train_t0500_pred_t04000"),
      "a horizon flag renames the cache, whatever the file said")
for argv, why in ((["--views", "2000,20000", "--results-dir", "same"],
                   "views without {H} in the folder"),
                  (["--views", "2000,20000", "--results-dir", "same_{{H}}"],
                   "views with the braces escaped, so no {H} to fill"),
                  (["--start-time", "-5"], "a start before t = 0"),
                  (["--views", "2000,20000", "--snapshots", "100,20000"],
                   "a view with no snapshot"),
                  (["--views", "50,2000"], "a view before the start"),
                  (["--end-time", "2000.5"], "an end off the model-step grid"),
                  (["--views", "2000,abc"], "a view that is not a number"),
                  (["--end-time", "4000", "--snapshots", "100,1000"],
                   "an end with no snapshot"),
                  (["--snapshots", "50,2000"], "a snapshot before the start"),
                  (["--end-time", "50"], "an end before the start"),
                  (["--end-time", "100"], "an end at the start"),
                  (["--video-every", "0"], "a film cadence of 0"),
                  (["--results-dir", "r_{X}"], "a folder with stray braces")):
    try:
        cli(argv)
        refused = False
    except SystemExit:
        refused = True
    check(refused, f"refused: {why}")
g = cli(["--engine", "lockstep", "--batch", "8", "--video-every", "7",
         "--diag-every", "5", "--snapshots", "100,700,2000",
         "--result-cache", "rc_x", "--log-time", "on"])
check(g["ENGINE"] == "lockstep" and g["EVAL_BATCH"] == 8
      and g["VIDEO_EVERY"] == 7 and g["DIAG_EVERY"] == 5
      and g["SNAPSHOT_TIMES"] == (100.0, 700.0, 2000.0)
      and g["RESULT_CACHE"] == "rc_x" and g["LOG_TIME"] == "on",
      "every setting can be given explicitly")
g = cli(["--keep-cache", "--redo", "--no-submission"])
check(g["KEEP_CACHE"] is True and g["REDO"] is True
      and g["WRITE_SUBMISSION"] is False
      and SAVED["KEEP_CACHE"] is False and SAVED["REDO"] is False
      and all(g[k] == BASE5[k] for k in KEYS
              if k not in ("KEEP_CACHE", "REDO", "WRITE_SUBMISSION")),
      "--keep-cache, --redo and --no-submission set just their own switch, "
      "each off by default")
restore()

print("\n6. time axes")
fig, ax = plt.subplots()
P._time_axis(ax)
linear = ax.get_xscale()
P.END_TIME = 20000.0
P._time_axis(ax)
check(linear == "linear" and ax.get_xscale() == "log",
      "linear over 100..2000 as today, logarithmic over 100..20000")
plt.close(fig)
restore()

import matplotlib.axes  # noqa: E402

import make_submission as S  # noqa: E402

figs = os.path.join(SCRATCH, "long_figs")
os.makedirs(figs, exist_ok=True)


def grown(end, seed):
    """L ~ t^(1/3): below l/8 = 16 up to 2000, past it by 20000."""
    t = np.arange(101.0, end + 1.0)
    d = t[::10]
    g = np.random.default_rng(seed)
    Lt = 0.8 * d ** (1 / 3) * (1 + 0.01 * g.standard_normal(d.size))
    return {"times": t, "r2": np.linspace(0.99, 0.9, t.size),
            "diag_times": d, "L_true": {P.LENGTH_KEY: Lt},
            "L_pred": {P.LENGTH_KEY: Lt * 1.02}, "diverged_at": None}


# the title of the growth figure is in its notes now (figure_notes.txt)
notes_today: dict = {}
notes_long: dict = {}
today = quiet(S.figure_growth, [grown(2000.0, s) for s in range(3)], 0.0,
              os.path.join(figs, "g2000.png"), 128, None, notes_today)
t_today = "\n".join(notes_today["g2000.png"])
longer = quiet(S.figure_growth, [grown(20000.0, s) for s in range(3)],
               0.0, os.path.join(figs, "g20000.png"), 128, None, notes_long)
t_long = "\n".join(notes_long["g20000.png"])
check("small_L" not in today and longer.get("small_L") is not None
      and longer["small_L"]["until"] < 20000.0,
      f"the small-L fit appears only once L outgrows l/8, and today's "
      f"manifest entry is unchanged (until t = "
      f"{(longer.get('small_L') or {}).get('until')})")
check("while L <=" not in t_today and "\nwhile L <=" in t_long,
      "the growth notes give that fit a line of its own")

scales = []
real_panel = S._sweep_panel


def spy_panel(ax, *a, **k):
    real_panel(ax, *a, **k)
    scales.append(ax.get_xscale())


def sweep_case(end, lat):
    t = np.arange(101.0, end + 1.0)
    return {"times": t, "mean_curve": np.linspace(0.99, 0.8, t.size),
            "lattice": lat, "stats": {"n_seeds": 3, "n_end": 3, "n_div": 0}}


S._sweep_panel = spy_panel
try:
    for end in (2000.0, 20000.0):
        quiet(S.figure_sweeps, {},
              {128: sweep_case(end, 128), 512: sweep_case(end, 512)},
              os.path.join(figs, f"s{end:g}.png"))
finally:
    S._sweep_panel = real_panel
check(scales == ["linear", "log"],
      f"the submission sweeps are linear in time to 2000, log to 20000 "
      f"{scales}")
shutil.rmtree(figs, ignore_errors=True)

print("\n7. a whole run with views, through main()")
out = os.path.join(SCRATCH, "lockstep_main")
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)


class Fake(Smooth):
    config = {"backbone": "physical_k", "hard_mass_conservation": True,
              "residual_prediction": True, "mode_scaling": 1.0,
              "zero_dc": True}

    def n_parameters(self):
        return 1


class Killed(BaseException):
    """What a job killed by its time limit looks like from inside: nothing
    after it runs, no handler catches it."""


fake = Fake()
real_cuda = torch.cuda.is_available


def run_main(argv, eval_seeds=seeds, sweep_seeds=None, sizes=(L,),
             patches=()):
    """main() on the fake model, on the CPU, writing only into SCRATCH.

    The sweep seeds are the eval seeds unless given.  ``patches`` are
    (object, name, value) set for this run only.  Returns (exit status, None
    if the job was killed; what it printed; its stderr).
    """
    restore()
    P.load_checkpoint = lambda path, device=None: (
        fake, {"best": {"epoch": 3, "score": 0.9, "survival": 20.0}})
    P.LATTICE, P.SIZE_SWEEP = L, tuple(sizes)
    P.EVAL_SEEDS = tuple(eval_seeds)
    P.SWEEP_SEEDS = P.EVAL_SEEDS if sweep_seeds is None else tuple(sweep_seeds)
    P.WRITE_LOG, P.WRITE_TIMING, P.PER_SEED_FIGURES = False, True, False
    P.GT_CACHE = ""
    was = [(o, n, getattr(o, n)) for o, n, _ in patches]
    for o, n, v in patches:
        setattr(o, n, v)
    torch.cuda.is_available = lambda: False
    buf, err = _io.StringIO(), _io.StringIO()
    code = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = P.main(argv)
    except Killed:
        pass
    finally:
        torch.cuda.is_available = real_cuda
        for o, n, v in reversed(was):
            setattr(o, n, v)
    return code, buf.getvalue(), err.getvalue()


def load(path):
    with open(path) as fh:
        return json.load(fh)


def raw(path):
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def listing(d):
    return set(os.listdir(d)) if os.path.isdir(d) else set()


def listing_all(d):
    """Every file below d, as a path from d."""
    return {os.path.relpath(os.path.join(b_, f), d).replace(os.sep, "/")
            for b_, _, fs in os.walk(d) for f in fs}


# a second box size, so that both sweep figures have a panel to draw (one
# curve per panel draws none)
L2 = 8
CASE = f"psi0{0.0:+.2f}_l{L}"
CASE2 = f"psi0{0.0:+.2f}_l{L2}"
STEM = {CASE: "fno_ch", CASE2: f"fno_ch_l{L2}"}
view = {h: os.path.join(out, f"results_x_P{h}") for h in (10, 30)}
cache7 = os.path.join(out, "cache")
argv7 = ["--eval-mode", "critical", "--start-time", "5", "--end-time", "40",
         "--views", "10,30", "--no-video", "--engine", "lockstep",
         "--results-dir", os.path.join(out, "results_x_P{H}"),
         "--result-cache", cache7]
sizes7 = (L2, L)
real_sweeps = S.figure_r2_sweeps


def cadence(start, end):
    """A film cadence of its own per view, as real views have (every 10 to
    2000, 100 to 20000): a view short enough to test would otherwise get the
    run's, and a cache key worked out inside a view would pass for the
    rollout's."""
    return {10.0: 2, 30.0: 3}.get(float(end), 5)


per_view = [(P, "default_video_every", cadence)]


def sweeps_fail_at_10(*a, **k):
    if P.END_TIME == 10.0:
        raise RuntimeError("the 10 view's sweep figure fails")
    return real_sweeps(*a, **k)


code, log, _ = run_main(argv7, sizes=sizes7,
                        patches=per_view
                        + [(S, "figure_r2_sweeps", sweeps_fail_at_10),
                           (P, "PER_SEED_FIGURES", True)])
# the run's settings are still in place: the rollouts' keys, and the ones a
# view would work out
k_run = {CASE: P._cache_key(0.0, L), CASE2: P._cache_key(0.0, L2)}
k_view = {}
for h in view:
    with P.view_context(float(h)):
        k_view[h] = P._cache_key(0.0, L)
check(P.VIDEO_SCHEDULE == ((10.0, 2), (30.0, 3))
      and len({k_run[CASE], *k_view.values()}) == 3,
      f"(each view films at its own cadence, so a cache key worked out inside "
      f"a view is not the rollout's: {P.VIDEO_SCHEDULE})")
TOK1, TOK2 = "psi0p0.00", f"l{L2}"
TOK = {CASE: TOK1, CASE2: TOK2}
XV = {h: os.path.join(cache7, "extras", os.path.basename(view[h]))
      for h in view}                       # each view's extras folder
man = {h: os.path.join(view[h], "manifest.json") for h in view}
book_f = {h: os.path.join(XV[h], S.BOOKKEEPING) for h in view}
summ = {(h, c): os.path.join(XV[h], c, f"{STEM[c]}_summary.json")
        for h in view for c in STEM}
check(code == 1 and all(os.path.exists(f) for f in list(man.values())
                        + list(summ.values()) + list(book_f.values()))
      and not any(d.lower().startswith("submission")
                  for d in os.listdir(out)),
      f"one run writes one flat folder per view -- the manifest at its top, "
      f"each case's summary in the view's extras folder -- and no Submission "
      f"folder, and says it failed (exit {code}; {sorted(os.listdir(out))})")
KINDS = ("snapshots", "phase_ordering_kinetics", "domain_growth",
         "psi_distribution", "composition", "free_energy")
TOP = {f"l{n}/{k}_psi0p0.00.png" for n in (L, L2) for k in KINDS} | {
    f"l{n}/{f}" for n in (L, L2) for f in S.SIZE_FILES} | {
    "composition.csv", "manifest.json", "report.txt"}
SWEEP = {"r2_sweep_per_case/psi0p0.00.png",
         "r2_sweep_per_case/figure_notes.txt"}
XTOP = {"submission_manifest.json", "sweep_curves.npz", "fno_ch_timing.json",
        "fno_ch_timing.txt"}
PLAIN = {"psi_distribution.csv", "domain_growth.csv", "mean_r2.csv",
         "composition.csv"}
OWN = ("_summary.json", "_r2_and_snapshots.png", "_conservation.png")
SEED_OWN = ("_r2.csv", "_summary.json", "_r2_and_snapshots.png",
            "_kinetics.png", "_conservation.png")
layout, bad = {}, {}
for h, d in view.items():
    # the 10 view's r2_sweep figure fails: it has none, the 30 view does
    # (no figure_notes.txt at the top either: its blocks are the sweep's)
    want_top = (TOP | SWEEP if h == 30 else TOP)
    bad[(h, "top")] = sorted(listing_all(d) ^ want_top)
    bad[(h, "extras")] = sorted(
        (listing(XV[h]) - set(STEM)) ^ XTOP)
    for c, stem in STEM.items():
        best = (load(summ[(h, c)]).get("best_seed")
                if os.path.exists(summ[(h, c)]) else None)
        bs = f"best_seed{best}"
        want = PLAIN | {stem + e for e in OWN} | {bs}
        bad[(h, c)] = sorted(listing(os.path.join(XV[h], c)) ^ want)
        bad[(h, c, bs)] = sorted(
            listing(os.path.join(XV[h], c, bs))
            ^ {f"{stem}_seed{best}{e}" for e in SEED_OWN})
check(not any(bad.values()),
      f"each view's folder has a folder for each box size with the figures "
      f"of its case and the size's own files, and the run's files at the top (the r2_sweep figure only where it did "
      f"not fail); its extras folder holds the run-wide bookkeeping, curves "
      f"and timing and a folder per case with the data, the fno_ch figures, "
      f"the averages and one best-seed folder with that seed's R2 and "
      f"figures (its kinetics too); no film with --no-video "
      f"{ {k: v for k, v in bad.items() if v} }")
m = {h: load(f) for h, f in man.items() if os.path.exists(f)}
bk = {h: load(f) for h, f in book_f.items() if os.path.exists(f)}
if len(m) == len(man) == len(bk):
    win = {h: bk[h]["compositions"]["+0.00"]["run"]["window"]["end"]
           for h in bk}
    check(m[10].get("pdf_time") == 10.0 and m[30].get("pdf_time") == 30.0
          and m[10].get("t_end") == 10.0 and m[30].get("t_end") == 30.0
          and win == {10: 10.0, 30: 30.0},
          f"each view's manifest is judged at, and windowed to, its horizon "
          f"(not the rollout's 40) {win}")
    cols = {h: m[h]["cases"][TOK1]["snapshots"]["panels"] for h in m}
    check({h: [p["t"] for p in c] for h, c in cols.items()}
          == {10: [5.0, 10.0], 30: [5.0, 30.0]},
          f"each view's figures show a run to that horizon's snapshots, not "
          f"the rollout's union (5, 10, 30, 40): "
          f"{ {h: [p['t'] for p in c] for h, c in cols.items()} }")
    check(len(bk[10].get("failed_last_run") or ()) == 1
          and "r2_sweep" in bk[10]["failed_last_run"][0]
          and bk[30].get("failed_last_run") == [],
          f"a view's failure is recorded in that view only "
          f"({bk[10].get('failed_last_run')}, {bk[30].get('failed_last_run')})")
    check(bk[10]["compositions"]["+0.00"]["cache_key"]
          == bk[30]["compositions"]["+0.00"]["cache_key"],
          "both views name the one cached rollout they came from")
    ents = {h: (bk[h]["compositions"]["+0.00"], bk[h]["sizes"].get(str(L)))
            for h in bk}
    check(all(s is not None and c["folder"] == s["folder"] == TOK1
              and c["files"] == s["files"] and c["files"]
              and all(os.path.exists(os.path.join(view[h], f))
                      for f in c["files"])
              for h, (c, s) in ents.items()),
          f"the critical case is also the l={L} box size: both entries name "
          f"the one case token and its files, in the view's folder -- no "
          f"alias copy "
          f"{ {h: c['folder'] for h, (c, _) in ents.items()} }")
    other = {h: bk[h]["sizes"].get(str(L2)) or {} for h in bk}
    check(all(sorted(bk[h]["sizes"]) == sorted([str(L), str(L2)])
              and other[h].get("folder") == TOK2 and other[h].get("files")
              and all(os.path.exists(os.path.join(view[h], f))
                      for f in other[h]["files"]) for h in bk)
          and all(m[h]["aliases"] == {f"l{L}": TOK1}
                  and sorted(m[h]["cases"]) == sorted([TOK1, TOK2])
                  for h in m),
          f"and the l={L2} box size has an entry and a token of its own "
          f"{ {h: e.get('folder') for h, e in other.items()} }")
    recs = {k: load(f).get("record") or {} for k, f in summ.items()
            if os.path.exists(f)}
    keys = {k: bk[k[0]][kind][name]["cache_key"] for k in recs
            for kind, name in ([("compositions", "+0.00")] if k[1] == CASE
                               else [("sizes", str(L2))])}
    check(len(recs) == len(summ)
          and all(r.get("complete") and r.get("window") == [5.0, k[0]]
                  and r.get("key") == keys[k] == k_run[k[1]]
                  and r.get("seeds") == len(seeds)
                  for k, r in recs.items()),
          f"each view's case summaries record the rollout's key (the one "
          f"worked out outside the views, as the bookkeeping has it), that "
          f"view's window, the seeds, and that they are complete "
          f"{ {k: (r.get('window'), r.get('complete')) for k, r in recs.items()} }")
first_drop = log.find("cached rollout(s) are deleted")
second_roll = log.find("[lockstep] 3 seeds together",
                       log.find("[lockstep] 3 seeds together") + 1)
check(log.count("[lockstep] 3 seeds together") == 2
      and log.count("-> cached") == 2 * len(seeds)
      and log.count("its 3 cached rollout(s) are deleted") == 2
      and sorted(os.listdir(cache7)) == ["extras"],
      f"each case's seeds were rolled out together, once, cached while the "
      f"case ran, and deleted -- only extras/ is left in the cache folder "
      f"-- once its outputs were written ({log.count('-> cached')} cached)")
check(0 <= first_drop < second_roll,
      "a case's cache goes as soon as that case is written, before the next "
      "case is rolled out, not at the end of the run")
timing = {h: os.path.join(XV[h], "fno_ch_timing.json") for h in view}
rows = {h: load(timing[h])["cases"] if os.path.exists(timing[h]) else None
        for h in view}
check(all(os.path.exists(os.path.join(XV[h], "fno_ch_timing.txt"))
          and r and sorted(x["lattice"] for x in r) == sorted(sizes7)
          and all(x["n_rollouts"] == len(seeds) for x in r)
          for h, r in rows.items()),
      f"the timing report is written per view, one line per case "
      f"{ {h: r and [(x['lattice'], x['n_rollouts']) for x in r] for h, r in rows.items()} }")

# the run-wide outputs go, so a second run has to write them again
RUN_WIDE = {(h, f): os.path.join(d, f)
            for h in view
            for d, names in ((view[h], ("r2_sweep_per_case/psi0p0.00.png",
                                        "report.txt", "manifest.json")),
                             (XV[h], ("fno_ch_timing.json",
                                      "fno_ch_timing.txt")))
            for f in names}
for path in RUN_WIDE.values():
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
watch = {**summ, **{(h, c, f): os.path.join(d, f)
                    for h in view for c in STEM
                    for d, f in ((view[h], f"l{L if TOK[c] == TOK1 else L2}"
                                  "/snapshots_psi0p0.00.png"),
                                 (os.path.join(XV[h], c),
                                  f"{STEM[c]}_r2_and_snapshots.png"))}}
kept = {k: raw(f) for k, f in watch.items()}
man_at = {h: None for h in man}   # its manifest is among the run-wide files
code2, log2, _ = run_main(argv7, sizes=sizes7, patches=per_view)
check(code2 == 0 and "[lockstep]" not in log2 and "-> reusing" not in log2
      and "-> cached" not in log2 and log2.count("already finished") == 2
      and sorted(os.listdir(cache7)) == ["extras"],
      f"a second run with the same flags takes every finished case from its "
      f"records: no rollout, no cache, exit {code2}")
check(all(v and raw(watch[k]) == v for k, v in kept.items()),
      "and leaves the finished cases' summaries and figures as the first run "
      "wrote them")
rows2 = {h: load(timing[h])["cases"] if os.path.exists(timing[h]) else None
         for h in view}
check(all(os.path.exists(path) for path in RUN_WIDE.values())
      and rows2 == rows,
      "it still writes each view's run-wide outputs from the records: the "
      "r2_sweep figure (the 10 view's too, which failed the first time), "
      "the report, the manifest and the timing report, with the same numbers")
m2 = {h: load(f) for h, f in man.items() if os.path.exists(f)}
bk2 = {h: load(f) for h, f in book_f.items() if os.path.exists(f)}
if len(m) == len(man) and len(m2) == len(man) == len(bk2):
    entries = (("compositions", "+0.00"), ("sizes", str(L)),
               ("sizes", str(L2)))
    check(all(bk2[h].get("failed_last_run") == []
              and bk2[h]["last_run"]["written"] != bk[h]["last_run"]["written"]
              and m2[h]["cases"] == m[h]["cases"] | {
                  t: m2[h]["cases"][t] for t in m2[h]["cases"]}
              and sorted(m2[h]["cases"]) == sorted(m[h]["cases"])
              and all(bk2[h][kind][k].get("cache_key")
                      == bk[h][kind][k].get("cache_key")
                      and bk2[h][kind][k].get("files")
                      == bk[h][kind][k].get("files")
                      for kind, k in entries)
              for h in view),
          "and each view's bookkeeping, rewritten by this run: the 10 "
          "view's failure cleared and the cases' entries kept; manifest.json "
          "is back with the same cases")

# one view of one case unfinished: that case is not taken from its records
stale = load(summ[(10, CASE)])
stale["record"]["complete"] = False
with open(summ[(10, CASE)], "w") as fh:
    json.dump(stale, fh)
before30 = os.stat(summ[(30, CASE)]).st_mtime_ns
other2 = {h: raw(summ[(h, CASE2)]) for h in view}
code3, log3, _ = run_main(argv7, sizes=sizes7, patches=per_view)
check(code3 == 0 and log3.count("already finished") == 1
      and log3.count("[lockstep] 3 seeds together") == 1
      and (load(summ[(10, CASE)]).get("record") or {}).get("complete")
      and os.stat(summ[(30, CASE)]).st_mtime_ns != before30
      and all(v and raw(summ[(h, CASE2)]) == v for h, v in other2.items())
      and sorted(os.listdir(cache7)) == ["extras"],
      "with one view unfinished that case is rolled out again, every view of "
      "it rewritten and its cache deleted after; the finished case is still "
      "taken from its records")
restore()
shutil.rmtree(out, ignore_errors=True)

print("\n8. the rollout cache is temporary, and a finished case is resumed")
top = os.path.join(SCRATCH, "lockstep_resume")
shutil.rmtree(top, ignore_errors=True)
os.makedirs(top)
M8 = os.path.join(top, "fno_ch.pt")        # its own, so its mtime can move
with open(M8, "wb") as fh:
    fh.write(b"x")


def flags(name, *extra, engine="lockstep", video=False):
    """A small run into top/<name>, its rollouts cached in <name>_cache."""
    return ["--eval-mode", "critical", "--start-time", "5", "--end-time",
            "15", *(() if video else ("--no-video",)), "--engine", engine,
            "--model", M8, "--results-dir", os.path.join(top, name),
            "--result-cache", os.path.join(top, name + "_cache"), *extra]


def case_file(name, f, folder=CASE):
    """A file of the case's extras folder, in the run's rollout cache."""
    return os.path.join(top, name + "_cache", "extras", folder, f)


def flat_file(name, f):
    """A file of the run folder: ``f`` in the folder of the base lattice."""
    return os.path.join(top, name, f"l{L}", f)


def record(name, folder=CASE, stem="fno_ch"):
    """The case summary's record, or None when there is no summary."""
    path = case_file(name, f"{stem}_summary.json", folder)
    return (load(path).get("record") or {}) if os.path.exists(path) else None


def cached(name):
    """The cache folder's own files (not its extras/), or None when it holds
    none."""
    d = os.path.join(top, name + "_cache")
    files = [f for f in os.listdir(d) if f != "extras"] if os.path.isdir(
        d) else []
    return sorted(files) if files else None


code, log, _ = run_main(flags("a", "--keep-cache"))
k1 = P._cache_key(0.0, L)
files, rec = cached("a"), record("a")
check(code == 0 and "[lockstep] 3 seeds together" in log
      and files is not None and len(files) == len(seeds)
      and all(k1 in f for f in files) and "[cache]" not in log,
      f"--keep-cache keeps the case's cached rollouts ({files})")
check(rec is not None and rec.get("complete") is True
      and rec.get("key") == k1 and rec.get("seeds") == len(seeds)
      and rec.get("seed_digest") == P._seed_digest(seeds)
      and rec.get("window") == [5.0, 15.0] and rec.get("submission") is True
      and rec.get("video") is False,
      f"the case summary records what produced it: key, seeds, window, "
      f"outputs, complete {rec}")
ref = raw(case_file("a", "mean_r2.csv"))
kept_a = raw(case_file("a", "fno_ch_summary.json"))

code, log, _ = run_main(flags("a"))
check(code == 0 and "already finished" in log and "(--redo does)" in log
      and "[lockstep]" not in log and "-> reusing" not in log,
      "a second run skips the finished case: no rollout, not even a cache "
      "read")
check(cached("a") is None and "its 3 cached rollout(s) are deleted" in log,
      "and deletes the cache a --keep-cache job left, the emptied folder too")
check(kept_a and raw(case_file("a", "fno_ch_summary.json")) == kept_a,
      "and leaves the case's summary as the earlier job wrote it")

code, log, _ = run_main(flags("a", "--redo"))
check(code == 0 and "already finished" not in log
      and "[lockstep] 3 seeds together" in log
      and ref and raw(case_file("a", "mean_r2.csv")) == ref
      and (record("a") or {}).get("complete") and cached("a") is None,
      "--redo rolls the finished case out again -- the same numbers -- and "
      "deletes its cache after")

st = os.stat(M8)
os.utime(M8, (st.st_atime, st.st_mtime + 100.0))
code, log, _ = run_main(flags("a"))
k2 = P._cache_key(0.0, L)
check(code == 0 and k2 != k1 and "already finished" not in log
      and "[lockstep] 3 seeds together" in log
      and (record("a") or {}).get("key") == k2,
      "a record written with another model file (its mtime) is not reused: "
      "the case is rolled out again and recorded with the new key")
code, log, _ = run_main(flags("a", "--snapshots", "5,10,15"))
k3 = P._cache_key(0.0, L)
check(code == 0 and k3 != k2 and "already finished" not in log
      and "[lockstep] 3 seeds together" in log
      and (record("a") or {}).get("key") == k3,
      "nor one written with other --snapshots")
code, log, _ = run_main(flags("a", "--snapshots", "5,10,15"),
                        eval_seeds=seeds[:2])
check(code == 0 and P._cache_key(0.0, L) == k3
      and "already finished" not in log
      and "[lockstep] 2 seeds together" in log
      and (record("a") or {}).get("seeds") == 2
      and (record("a") or {}).get("seed_digest") == P._seed_digest(seeds[:2]),
      "nor one written for other seeds, though the key is the same")

calls = [0]
real_evaluate = P.evaluate_seeds


def killed_in_second_batch(*a, **k):
    calls[0] += 1
    if calls[0] == 2:
        raise Killed()
    return real_evaluate(*a, **k)


code, log, _ = run_main(flags("k", "--batch", "2"),
                        patches=[(P, "evaluate_seeds",
                                  killed_in_second_batch)])
kk = P._cache_key(0.0, L)
first_two = sorted(os.path.basename(P._result_path(0.0, s, L))
                   for s in seeds[:2])
files = cached("k")
check(code is None and files == first_two and record("k") is None,
      f"a job killed in the middle of a case leaves the seeds it finished "
      f"in the cache, and no summary ({files})")
cdir = os.path.join(top, "k_cache")
foreign = os.path.join(cdir, first_two[0].replace(kk, "0123456789ab"))
if files:
    shutil.copyfile(os.path.join(cdir, first_two[0]), foreign)
    shutil.copyfile(os.path.join(cdir, first_two[0]),
                    os.path.join(cdir, first_two[0]) + ".corrupt")
code, log, _ = run_main(flags("k", "--batch", "2"))
check(code == 0 and log.count("-> reusing") == 2
      and "[lockstep] 1 seeds together" in log
      and "[lockstep] 2 seeds" not in log
      and ref and raw(case_file("k", "mean_r2.csv")) == ref,
      "the next job reuses the seeds cached, rolls out only the rest, and "
      "gets the numbers an unbroken run does")
check(cached("k") == [os.path.basename(foreign)],
      f"then deletes the case's cached rollouts and their set-aside copies, "
      f"but not another key's, so that folder stays ({cached('k')})")


def killed_drawing(*a, **k):
    raise Killed()


code, log, _ = run_main(flags("w", "--keep-cache"))
done_w = (record("w") or {}).get("complete")
code, log, _ = run_main(flags("w", "--redo", "--keep-cache"),
                        patches=[(S, "write_case", killed_drawing)])
check(done_w and code is None and record("w") is None
      and os.path.exists(case_file("w", "mean_r2.csv"))
      and len(cached("w") or ()) == len(seeds),
      "a --redo job killed while writing the case -- at its submission "
      "figures, the last step before the summary -- leaves no summary (the "
      "earlier one goes first), only the cache")
code, log, _ = run_main(flags("w"))
check(code == 0 and "already finished" not in log
      and log.count("-> reusing") == len(seeds) and "[lockstep]" not in log
      and (record("w") or {}).get("complete") and cached("w") is None,
      "so the next job does not take the half-written case for finished: it "
      "writes it again from the cache, then deletes the cache")


def figures_fail(*a, **k):
    raise RuntimeError("the submission figures fail")


code, log, _ = run_main(flags("s"), patches=[(S, "write_case",
                                              figures_fail)])
rec = record("s")
check(code == 1 and rec is not None and rec.get("complete") is False
      and rec.get("submission") is False
      and len(cached("s") or ()) == len(seeds)
      and "kept, as its submission figures failed" in log,
      f"a case whose submission figures fail keeps its cache, and its record "
      f"says it is not complete (exit {code}, {cached('s')})")
code, log, _ = run_main(flags("s"))
rec = record("s") or {}
check(code == 0 and "already finished" not in log
      and log.count("-> reusing") == len(seeds) and "[lockstep]" not in log
      and rec.get("complete") and rec.get("submission")
      and os.path.exists(flat_file("s", "snapshots_psi0p0.00.png"))
      and cached("s") is None,
      "so the next run redoes it from that cache, figures and all, and only "
      "then deletes the cache")

code, log, _ = run_main(flags("n", "--no-submission", "--keep-cache"))
rec = record("n") or {}
check(code == 0 and rec.get("complete") and rec.get("submission") is False
      and not any(os.path.exists(os.path.join(top, "n", f))
                  for f in ("manifest.json", "README.md", "sweeps.png"))
      and not os.path.exists(flat_file("n", "snapshots_psi0p0.00.png"))
      and os.path.exists(case_file("n", "mean_r2.csv")),
      "--no-submission writes no submission figures, manifest or README "
      "(the averages stay), and the record says so")
code, log, _ = run_main(flags("n"))
check(code == 0 and "already finished" not in log
      and log.count("-> reusing") == len(seeds)
      and (record("n") or {}).get("submission") is True
      and os.path.exists(flat_file("n", "snapshots_psi0p0.00.png"))
      and os.path.exists(os.path.join(top, "n", "manifest.json")),
      "a run that wants the submission does not take that record for "
      "finished")
code, log, _ = run_main(flags("n", "--no-submission"))
check(code == 0 and "already finished" in log and "[lockstep]" not in log,
      "while a record with the submission serves a run that does not want "
      "it")


def has_film(name):
    return any(os.path.exists(case_file(name, f))
               for f in ("rollout.mp4", "rollout.gif"))


code, log, _ = run_main(flags("v", "--keep-cache", "--video-every", "10",
                              engine="seed"))
kv = P._cache_key(0.0, L)
rv = record("v") or {}
code, log, _ = run_main(flags("v", "--video-every", "10", engine="seed",
                              video=True))
check(code == 0 and P._cache_key(0.0, L) == kv and rv.get("video") is False
      and "already finished" not in log
      and log.count("-> reusing") == len(seeds),
      "with the seed engine the film is not in the cache key, but a record "
      "without the film does not serve a run that wants it")
rv = record("v") or {}
check(rv.get("complete") and not has_film("v") and rv.get("video") is False
      and cached("v") is None,
      f"the rollouts it reused had no frames, so there is no film, and the "
      f"record does not claim one {rv}")
code, log, _ = run_main(flags("v", "--video-every", "10", engine="seed",
                              video=True))
rv = record("v") or {}
check(code == 0 and "already finished" not in log and "-> reusing" not in log
      and log.count("-> cached") == len(seeds) and has_film("v")
      and rv.get("video") is True,
      "so the next run rolls the case out again and makes the film")
code, log, _ = run_main(flags("v", "--video-every", "10", engine="seed",
                              video=True))
check(code == 0 and "already finished" in log and "-> cached" not in log,
      "and the one after takes the case, film and all, from its record")
code, log, _ = run_main(flags("u", "--no-submission", "--video-every", "10",
                              engine="seed", video=True))
ru = record("u") or {}
code, log, _ = run_main(flags("u", "--no-submission", "--video-every", "10",
                              engine="seed", video=True))
check(code == 0 and ru.get("complete") and ru.get("video") is False
      and not has_film("u") and "already finished" in log,
      f"--no-submission makes no film either, and the record says so; yet it "
      f"serves the next --no-submission run, which wants none {ru}")

TWIN = CASE + "_sweep_seeds"
code, log, _ = run_main(flags("d"), sweep_seeds=seeds[:2])
check(code == 0 and log.count("[lockstep] 3 seeds together") == 1
      and "[lockstep] 2 seeds" not in log and log.count("-> reusing") == 2
      and record("d") is not None
      and record("d", TWIN, f"fno_ch_l{L}") is not None
      and cached("d") is None,
      "with other sweep seeds, the l=16 sweep case reuses the composition "
      "case's cached seeds: the cache goes only once both are written")
code, log, _ = run_main(flags("d"), sweep_seeds=seeds[:2])
check(code == 0 and log.count("already finished") == 2
      and "[lockstep]" not in log,
      "and a second run skips both")

print("\n9. what the review found: shared caches, lost entries, settings")
S = P._submission()
real_write_case = S.write_case


def twin_figures_fail(results, seeds_, off, lattice, out_dir, *a, **k):
    if out_dir.rstrip("\\/").endswith(TWIN):
        raise RuntimeError("the twin's figures fail")
    return real_write_case(results, seeds_, off, lattice, out_dir, *a, **k)


code, log, _ = run_main(flags("e"), sweep_seeds=seeds[:2],
                        patches=[(S, "write_case", twin_figures_fail)])
left = cached("e") or []
key_e = P._cache_key(0.0, L)
shared = sorted(f"fno_ch_l{L}_seed{sd}_{key_e}.npz" for sd in seeds[:2])
check(code == 1 and (record("e", TWIN, f"fno_ch_l{L}") or {}).get("complete")
      is False and sorted(left) == shared,
      f"when the sweep-seed twin's figures fail, the rollouts it shares with "
      f"the finished composition case are kept for it (and only those): "
      f"{left}")
code, log, _ = run_main(flags("e"), sweep_seeds=seeds[:2])
check(code == 0 and log.count("already finished") == 1
      and log.count("-> reusing") == 2 and "[lockstep]" not in log
      and (record("e", TWIN, f"fno_ch_l{L}") or {}).get("complete")
      and cached("e") is None,
      "so the next run redoes the twin from them, rolls nothing out, and "
      "only then deletes them")

code, log, _ = run_main(flags("f"))
man_f = os.path.join(top, "f_cache", "extras", S.BOOKKEEPING)
before = load(man_f)
m = load(man_f)
del m["compositions"]["+0.00"]
del m["sizes"][str(L)]
with open(man_f, "w") as fh:
    json.dump(m, fh)
code, log, _ = run_main(flags("f"))
after = load(man_f)
check(code == 0 and "already finished" in log and "[lockstep]" not in log
      and after["compositions"].get("+0.00") == before["compositions"]["+0.00"]
      and after["sizes"].get(str(L)) == before["sizes"][str(L)]
      and after["sweep"]["missing_curves"] == []
      and after["sweep"]["status"].startswith("r2_sweep drawn"),
      "an entry the manifest lost comes back when its finished case is "
      "resumed -- the same entry, its sweep curve with it")
st_f = load(case_file("f", "fno_ch_summary.json"))
check(all(isinstance(st_f.get(k), dict)
          and set(st_f[k]) == {"mean", "std", "n"}
          and 0 <= st_f[k]["n"] <= len(seeds)
          for k in ("growth_exponent_truth", "growth_exponent_fno",
                    "final_r2", "t_stop_of_stopped"))
      and st_f["final_r2"]["n"] == len(seeds) and "seeds" not in st_f,
      "the summary averages the seeds' growth exponents, final R2 and stop "
      "times (mean, spread, n) instead of listing them")

code, log, _ = run_main(flags("g"), sizes=(L2,))
m1 = load(os.path.join(top, "g_cache", "extras", S.BOOKKEEPING))
code2, log2, _ = run_main(flags("g"), sizes=(L2, L))
m2 = load(os.path.join(top, "g_cache", "extras", S.BOOKKEEPING))
check(code == 0 and code2 == 0 and sorted(m1["sizes"]) == [str(L2)]
      and log2.count("already finished") == 2 and "[lockstep]" not in log2
      and sorted(m2["sizes"], key=int) == [str(L2), str(L)]
      and m2["sizes"][str(L)].get("folder") == TOK1
      and m2["sizes"][str(L)] == m2["compositions"]["+0.00"],
      "a critical case first run without l=16 in the sweep is recorded as "
      "that box size too when a later run's sweep includes it, without a "
      "rollout")


def no_room(path, res):
    raise OSError(122, "Disk quota exceeded")


code, log, _ = run_main(flags("h"), patches=[(P, "_save_result", no_room)])
check(code == 0 and "could not cache" in log and "going on without it" in log
      and (record("h") or {}).get("complete") is True
      and os.path.exists(case_file("h", "mean_r2.csv")),
      "a cache that cannot be written (a full quota) costs only the resume: "
      "the case is still written, whole")

code, log, _ = run_main(flags("i"))
code, log, _ = run_main(flags("i", "--log-time", "on"))
rec_i = record("i") or {}
check(code == 0 and "already finished" not in log
      and "[lockstep] 3 seeds together" in log
      and (rec_i.get("figures") or {}).get("log_time") is True,
      "a run that wants other time axes (--log-time) redraws the case "
      "rather than skip it with the old axes")
code, log, _ = run_main(flags("i", "--log-time", "on"))
check(code == 0 and "already finished" in log,
      "and the next run with the same axes skips it")

import matplotlib.figure as mfig  # noqa: E402

LABELS = {}
real_savefig = mfig.Figure.savefig


def labels_of(self, fname, *a, **k):
    if isinstance(fname, str) and fname.endswith("_conservation.png"):
        LABELS[os.path.basename(os.path.dirname(fname))
               + "/" + os.path.basename(fname)] = [
            [ln.get_label() for ln in ax.get_lines()] for ax in self.axes]
    return real_savefig(self, fname, *a, **k)


code, log, _ = run_main(flags("j"), patches=[(mfig.Figure, "savefig",
                                              labels_of),
                                             (P, "PER_SEED_FIGURES", True)])
case_lab = LABELS.get(f"{CASE}/fno_ch_conservation.png") or [[]]
flat = [lab for axis in case_lab for lab in axis]
best_j = load(case_file("j", "fno_ch_summary.json")).get("best_seed")
own = LABELS.get(f"best_seed{best_j}/fno_ch_seed{best_j}_conservation.png")
check(code == 0 and flat
      and not any(lab.startswith("seed ") for lab in flat)
      and f"mean over {len(seeds)} seeds" in flat
      and "worst seed at each step" in flat
      and sum(len(axis) for axis in case_lab) <= 6,
      f"the case's conservation figure draws averages over the seeds, not "
      f"each seed's own curve: {case_lab}")
check(own is not None
      and any(f"seed {best_j}" in lab for axis in own for lab in axis),
      "while the best seed's own figure still draws that seed")
restore()
shutil.rmtree(top, ignore_errors=True)
new = sorted(project_state() - PROJECT)
check(not new, f"and no run wrote into the project folder, Work/ or Results/ "
               f"{new[:3]}")

shutil.rmtree(VW, ignore_errors=True)
print("\n" + ("LOCKSTEP PASS" if ok else "LOCKSTEP FAIL"))
sys.exit(0 if ok else 1)
