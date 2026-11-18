"""--smoke is a tiny end-to-end run of the real scripts, in its own folder.

A 16 x 16 lean dataset is made here by hand, and fno_train, fno_predict and
make_submission run on it for real, on the CPU, in a box under _scratch whose
Work/ is the project's.  After each run the box is compared with a listing
taken before: nothing may appear outside Work/smoke/, no other checkpoint may
be loaded, and a flag given beside --smoke still wins over it.
"""
import contextlib
import glob
import io
import json
import os
import shutil
import sys
import time
import types
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from _paths import SCRATCH, project_state

BOX = os.path.join(SCRATCH, "smoke_box")
shutil.rmtree(BOX, ignore_errors=True)
os.makedirs(BOX)
os.chdir(BOX)
PROJECT = project_state()

import fno_paths
fno_paths.WORK = os.path.join(BOX, "Work")
import ch_gpu
import fno_predict as P
import fno_runlog
import fno_train as T
T.use_fixed_step()
import make_submission as S

ok = True
t_suite = time.time()


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


class Reached(Exception):
    """Raised by the stand-in for the first real work: main() got that far."""


START = [(m, {k: v for k, v in vars(m).items() if k.isupper()})
         for m in (T, P, S, fno_runlog)]
MONITOR = T.MONITOR_SEEDS[:1], T.MONITOR_OFFS[:1]


def reset():
    """Every module setting back as imported, as a new process has it."""
    for mod, saved in START:
        vars(mod).update(saved)


def run(fn, *argv, stop=None):
    """fn(argv) with its output captured and its log closed as exit would.

    Returns (what it returned, the exception that ended it or None, the
    output).  ``stop`` names a (module, function) that raises Reached when
    main() gets to it, to look at the settings main() arrived with.
    """
    buf, ends = io.StringIO(), []
    value = exc = None
    real = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buf
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(
            fno_runlog, "atexit", types.SimpleNamespace(register=ends.append)))
        if stop:
            stack.enter_context(mock.patch.object(*stop, side_effect=Reached))
        try:
            value = fn(list(argv))
        except (Exception, SystemExit) as err:
            exc = err
        finally:
            for end in ends:
                end()
            sys.stdout, sys.stderr = real
    return value, exc, buf.getvalue()


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def stops(fn, *a):
    """Whether fn(*a) ends in a SystemExit."""
    try:
        quiet(fn, *a)
    except SystemExit:
        return True
    return False


def said(exc):
    return str(exc.code) if isinstance(exc, SystemExit) else repr(exc)


def tree():
    return {os.path.relpath(os.path.join(here, name), BOX)
            for here, dirs, files in os.walk(BOX) for name in dirs + files}


SMOKE = os.path.join("Work", "smoke")


def strays(before):
    """What appeared in the box since ``before`` outside Work/smoke/."""
    return sorted(p for p in tree() - before if p != "Work"
                  and not (p + os.sep).startswith(SMOKE + os.sep))


def tiny_lean(path, rows=4, frames=300, n=16):
    """A lean dataset (ch_lean_v1) of smooth random fields: each row drifts
    between two of them, so consecutive frames are alike and no frame leaves
    [-0.7, 0.7]."""
    rng = np.random.default_rng(1)
    k2 = np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2

    def smooth():
        f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n)))
                         * np.exp(-60.0 * k2)).real
        f -= f.mean()
        return 0.5 * f / np.abs(f).max()

    j = np.arange(frames)[:, None, None] / 40.0
    data = np.stack([np.cos(j) * smooth() + np.sin(j) * smooth()
                     for _ in range(rows)]).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)
    meta = {"format": T.LEAN_FORMAT, "l": n, "dt": 0.01, "gap": 10,
            "n_ahead": 100, "shift": 10, "t_end": frames / 10,
            "n_frames": frames, "dtype": "float32", "n_train": rows - 1,
            "n_val": 1,
            "rows": [{"seed": i, "off": 0.0,
                      "split": "train" if i < rows - 1 else "val"}
                     for i in range(rows)]}
    with open(ch_gpu.lean_sidecar(path), "w") as fh:
        json.dump(meta, fh)


def listing(folder):
    return sorted((f, os.path.getsize(os.path.join(folder, f)),
                   os.stat(os.path.join(folder, f)).st_mtime_ns)
                  for f in os.listdir(folder))


TINY = os.path.join(BOX, "data", "tiny.npy")
tiny_lean(TINY)
OLD_MODEL = os.path.join(BOX, "fno_ch.pt")
with open(OLD_MODEL, "wb") as fh:
    fh.write(b"not a checkpoint")
ELSEWHERE = os.path.join(BOX, "elsewhere")
os.makedirs(ELSEWHERE)
OUTSIDE_MODEL = os.path.join(ELSEWHERE, "fno_ch.pt")
with open(OUTSIDE_MODEL, "wb") as fh:
    fh.write(b"not a checkpoint")

ROOT = fno_paths.smoke_root()
# the Laplacian head is the default, so the smoke model is a _lap one
LABEL = "train_t0020_lap"
RUN = f"{LABEL}_pred_t00120"
MODEL = fno_paths.model_path(LABEL, ROOT)
RESULTS = fno_paths.results_dir(RUN, ROOT)
CACHE = fno_paths.rollout_dir(RUN, ROOT)
LOGS = fno_paths.log_dir(ROOT)

loaded = []
real_load = P.load_checkpoint


def spy(path, device="cpu"):
    loaded.append(os.path.abspath(path))
    return real_load(path, device=device)


P.load_checkpoint = spy

print("1. where is inside Work/smoke")
check(ROOT == os.path.join(BOX, "Work", "smoke"),
      "the root follows fno_paths.WORK, which is read when it is asked for")
for path, inside in ((os.path.join(ROOT, "models", "m", "fno_ch.pt"), True),
                     (ROOT, True),
                     (os.path.join(ROOT, "models", "..", "..", "x"), False),
                     (os.path.join(ROOT, "..", "smoke_old", "x"), False),
                     (ROOT + "_old", False),
                     (os.path.join("Work", "smoke", "x"), True)):
    check(fno_paths.under(path, ROOT) == inside,
          f"{os.path.relpath(path, BOX)} is {'' if inside else 'not '}inside")
helper = os.path.join(BOX, "helper")
for name, age in (("old", 100), ("new", 0), ("older", 200)):
    os.makedirs(os.path.join(helper, "rollout_cache", name))
    stamp = time.time() - age
    os.utime(os.path.join(helper, "rollout_cache", name), (stamp, stamp))
open(os.path.join(helper, "rollout_cache", "file.txt"), "w").close()
check(fno_paths.latest_rollouts(helper)
      == os.path.join(helper, "rollout_cache", "new"),
      "the newest rollout cache is the newest folder, not a file")
shutil.rmtree(helper)
check(fno_paths.latest_rollouts(helper) is None, "none where there is none")

print("\n2. a smoke prediction with no smoke model loads nothing")
before = tree()
quiet(P.use_old_inputs)
check(P.MODEL_PATH == "fno_ch.pt",
      "without --smoke the old-input fallback would take the file in the "
      "working directory")
reset()
_, exc, out = run(P.main, "--smoke")
check(isinstance(exc, SystemExit) and "no smoke model" in said(exc)
      and "fno_train.py --smoke" in said(exc), f"refused: {said(exc)}")
check(not loaded, "and nothing was loaded, the old fno_ch.pt included")
check(not strays(before) and not os.path.exists(ROOT),
      "and nothing was written, not even a log")
_, exc, out = run(S.main, "--smoke")
check(isinstance(exc, SystemExit) and "no smoke rollouts" in said(exc),
      f"make_submission --smoke with no rollouts says so: {said(exc)}")
check(not strays(before) and not os.path.exists(ROOT), "and writes nothing")

print("\n3. fno_train.py --smoke")
before = tree()
t0 = time.time()
reset()
_, exc, out = run(T.main, "--smoke", "--dataset", TINY)
print(f"   ({time.time() - t0:.0f}s)")
check(exc is None, f"it ran to the end ({exc!r})")
state = fno_paths.model_dir(LABEL, ROOT)
check(os.path.isfile(MODEL) and os.path.isfile(
    os.path.join(state, fno_paths.STATE_FILE)),
    f"the best model and its resume state are in "
    f"{os.path.relpath(state, BOX)}")
with open(os.path.splitext(MODEL)[0] + "_history.json") as fh:
    history = json.load(fh)
check([h["epoch"] for h in history] == [1, 2]
      and all(h["unroll"] == 2 for h in history),
      "two epochs, both at unroll 2")
check(len([ln for ln in out.splitlines() if ln.startswith("epoch ")]) == 2,
      "and the log shows two epoch lines")
logs = glob.glob(os.path.join(LOGS, "fno_train_*.log"))
text = open(logs[0], encoding="utf-8").read() if logs else ""
check(len(logs) == 1 and "[smoke]" in text and "elapsed" in text,
      "the log is in Work/smoke/logs, holds the smoke line, and was closed")
gt = sorted(os.path.basename(p) for p in glob.glob(
    os.path.join(fno_paths.gt_dir(ROOT), "*.npz")))
check(len(gt) == 2 and all("_l32_" in f and "_t110_" in f for f in gt),
      f"the references are 32 x 32 to t = 110, in the smoke gt_cache: {gt}")
check(not strays(before),
      f"nothing was written outside Work/smoke: {strays(before)}")
check(out.count("[smoke]") == 1, "one smoke line is printed")

print("\n4. the flags still win, and the settings win over the file")
reset()
T.WRITE_LOG = False
T.OFF_CRITICAL, T.MAX_HOURS, T.RESUME, T.PATIENCE = True, 5.0, True, 8
T.MODEL_OUT, T.CHECKPOINT = "prod.pt", "prod_last.pt"
_, exc, out = run(T.main, "--smoke", "--dataset", TINY, "--epochs", "1",
                  stop=(T, "load_trajectories"))
check(isinstance(exc, Reached), f"main() got as far as the data ({exc!r})")
check((T.EPOCHS, T.STEPS_PER_EPOCH, T.BATCH_SIZE, T.VAL_EVERY) == (1, 4, 2, 1),
      "--epochs 1 wins; the other smoke settings stay")
check(T.OFF_CRITICAL is False and T.MAX_HOURS is None and T.RESUME is False
      and T.PATIENCE == 0, "and smoke turns off what a file may have on")
check(T.T_TRAIN == 20.0, "a lean dataset is sliced to t = 20")
check(len(T.VAL_SEEDS) == 1 and (T.VAL_LATTICE, T.VAL_T_START, T.VAL_T_END,
                                 T.VAL_BATCHES) == (32, 100.0, 110.0, 2)
      and (T.MONITOR_SEEDS, T.MONITOR_OFFS) == MONITOR
      and (T.MONITOR_LATTICE, T.MONITOR_T_END, T.MAX_UNROLL) == (32, 110.0, 2),
      "one validation seed and one monitor trajectory, both small")
quiet(T.resolve_outputs, 190)
check(T.MODEL_OUT == MODEL and T.CHECKPOINT == os.path.join(
    state, fno_paths.STATE_FILE) and T.GT_CACHE == fno_paths.gt_dir(ROOT),
    "the model, its state and the references are under the smoke root, "
    "whatever the file named")
reset()
T.WRITE_LOG = False
_, exc, out = run(T.main, "--smoke", "--dataset", TINY, "--t-train", "30",
                  "--tag", "mine", stop=(T, "load_trajectories"))
quiet(T.resolve_outputs, 290)
check(T.T_TRAIN == 30.0 and T.MODEL_OUT == fno_paths.model_path(
    "train_t0030_lap_mine", ROOT), f"--t-train and --tag win: {T.MODEL_OUT}")
check(not stops(T.check_best_model, MODEL),
      "under --smoke a model of another run is overwritten without a word")
T.ROOT = None
check(stops(T.check_best_model, MODEL),
      "(outside it, that same model is another run's best)")
reset()
T.WRITE_LOG = False
_, exc, out = run(T.main, "--smoke", "--dataset", "never.npz",
                  stop=(T, "load_trajectories"))
check(isinstance(exc, Reached) and T.T_TRAIN is None
      and "read whole" in out, "an npz is read whole, and the log says so")

print("\n5. fno_predict.py --smoke")
before = tree()
t0 = time.time()
reset()
loaded.clear()
_, exc, out = run(P.main, "--smoke")
print(f"   ({time.time() - t0:.0f}s)")
check(exc is None, f"it ran to the end ({exc!r})")
check(loaded == [os.path.abspath(MODEL)],
      f"it loaded the smoke model and nothing else: {loaded}")
cases = sorted(
    os.path.relpath(os.path.join(b_, f), RESULTS).replace(os.sep, "/")
    for b_, _, fs in os.walk(RESULTS) for f in fs) if os.path.isdir(
        RESULTS) else []
FLAT = sorted(
    [f"{size}/{kind}_psi0p0.00.png" for size in ("l32", "l48")
     for kind in ("snapshots", "phase_ordering_kinetics", "domain_growth",
                  "psi_distribution", "composition", "free_energy")]
    + [f"{size}/{f}" for size in ("l32", "l48")
       for f in ("composition_all.png", "r2_time.png", "figure_notes.txt")]
    + ["composition.csv", "manifest.json", "report.txt",
       "r2_sweep_per_case/psi0p0.00.png",
       "r2_sweep_per_case/figure_notes.txt"])
check(cases == FLAT,
      f"the results are Work/smoke/Results/{RUN}: a folder for each of the "
      f"sizes l32 and l48 with the critical case (psi0p0.00) and the size's "
      f"own files, and the run's files at the top: {cases}")
kept = glob.glob(os.path.join(CACHE, "*.npz"))
check(len(kept) == 4, f"the rollouts are kept in the smoke cache: {len(kept)}")
recs = sorted(glob.glob(os.path.join(CACHE, "records", "*", "seed*.npz")))
meta = P.read_meta(CACHE)
check(len(recs) == 4 and meta is not None
      and meta["fingerprint"] == fno_paths.fingerprint(MODEL)
      and len(meta["cases"]) == 2
      and all(c["complete"] for c in meta["cases"].values())
      and not any("vid_true" in np.load(r).files for r in recs),
      "and so are a record per seed, without the film frames, and a "
      "meta.json that calls both cases complete, in the same smoke folder")
films = glob.glob(os.path.join(CACHE, "extras", "**", "rollout.*"),
                  recursive=True)
check(films and not glob.glob(os.path.join(RESULTS, "**", "rollout.*"),
                              recursive=True),
      "and the film was written, into the cache's extras/ and not into the "
      "results folder")
gt = glob.glob(os.path.join(BOX, "**", "gt_*.npz"), recursive=True)
check(len(gt) == 7 and all(os.path.dirname(p) == fno_paths.gt_dir(ROOT)
                           for p in gt),
      f"the references (6, and the one the diagnostics share) are only in "
      f"the smoke gt_cache ({len(gt)} files)")
diag_path = os.path.join(CACHE, "extras", "diagnostics.json")
try:
    with open(diag_path) as fh:
        diag = json.load(fh)
except (OSError, ValueError):
    diag = {}
check(diag.get("tile", {}).get("tiled_l") == 64
      and isinstance(diag.get("tile", {}).get("max_abs"), float)
      and diag.get("lowk", {}).get("lattice") == 32
      and len(diag.get("lowk", {}).get("seeds", ())) == 1,
      "the diagnostics ran tiny, l = 32 tiled to 64 on 1 seed, and wrote "
      "diagnostics.json in the cache's extras/")
logs = glob.glob(os.path.join(LOGS, "fno_predict_*.log"))
text = open(logs[0], encoding="utf-8").read() if logs else ""
check(len(logs) == 1 and "[smoke]" in text and "elapsed" in text,
      "the log is in Work/smoke/logs, holds the smoke line, and was closed")
check(not strays(before),
      f"nothing was written outside Work/smoke: {strays(before)}")

print("\n6. the old file, another model, and the flags beside --smoke")
before = tree()
loaded.clear()
reset()
for name, bad in (("a checkpoint outside", OUTSIDE_MODEL),
                  ("the production model", fno_paths.model_path(
                      P.DEFAULT_MODEL)),
                  ("one through ..", os.path.join(ROOT, "..", "models",
                                                  "train_t0500", "fno_ch.pt")),
                  ("one in a folder sharing the prefix", os.path.join(
                      ROOT + "_old", "models", LABEL, "fno_ch.pt"))):
    reset()
    _, exc, out = run(P.main, "--smoke", "--model", bad)
    check(isinstance(exc, SystemExit)
          and "never loads a production checkpoint" in said(exc),
          f"--model {name} is refused: {said(exc)[:60]}")
check(not loaded and not strays(before),
      "and none of it was loaded or written")
reset()
P.WRITE_LOG = False
_, exc, out = run(P.main, "--smoke", stop=(P, "load_checkpoint"))
check(isinstance(exc, Reached) and P.MODEL_PATH == MODEL,
      "with the old fno_ch.pt beside it, the model is the smoke one")
reset()
P.WRITE_LOG = False
rel = os.path.relpath(os.path.join(ROOT, "models", "..", "models", LABEL,
                                   "fno_ch.pt"))
_, exc, out = run(P.main, "--smoke", "--model", rel, "--seeds", "3",
                  "--sizes", "32,40", "--no-video",
                  stop=(P, "load_checkpoint"))
check(isinstance(exc, Reached) and P.MODEL_PATH == rel
      and P.RESULTS_DIR == RESULTS and P.RESULT_CACHE == CACHE,
      "a --model under the root is taken, and its results stay under it")
check(len(P.EVAL_SEEDS) == 3 and P.SWEEP_SEEDS == P.EVAL_SEEDS
      and P.SIZE_SWEEP == (32, 40) and P.WRITE_VIDEO is False
      and P.EVAL_MODE == "critical" and P.KEEP_CACHE is True
      and P.LATTICE == 32, "--seeds, --sizes and --no-video win over it")
reset()
P.WRITE_LOG = False
_, exc, out = run(P.main, "--smoke", "--end-time", "140",
                  stop=(P, "load_checkpoint"))
run140 = f"{LABEL}_pred_t00140"
check(isinstance(exc, Reached) and P.END_TIME == 140.0
      and P.RESULTS_DIR == fno_paths.results_dir(run140, ROOT)
      and P.RESULT_CACHE == fno_paths.rollout_dir(run140, ROOT)
      and P.GT_CACHE == fno_paths.gt_dir(ROOT),
      f"a horizon flag names its folders under the root: {P.RESULTS_DIR}")
reset()
P.WRITE_LOG = False
_, exc, out = run(P.main, "--smoke", "--results-dir", "mine",
                  stop=(P, "load_checkpoint"))
check(P.RESULTS_DIR == "mine" and P.RESULT_CACHE == CACHE,
      "an explicit --results-dir is the user's to choose")

print("\n7. make_submission.py --smoke")
reset()
os.remove(os.path.join(RESULTS, "report.txt"))
victim = os.path.join(RESULTS, "l32", "snapshots_psi0p0.00.png")
os.remove(victim)
before = tree()
cache_before = listing(CACHE)
_, exc, out = run(S.main, "--smoke")
check(exc is None, f"it ran ({exc!r})")
check(os.path.isfile(os.path.join(RESULTS, "report.txt"))
      and os.path.isfile(victim),
      "the smoke results folder was rebuilt from the cache")
with open(os.path.join(RESULTS, "manifest.json")) as fh:
    manifest = json.load(fh)
check(manifest.get("source", "").startswith("make_submission, regenerated")
      and manifest.get("cache") == CACHE, "and says where from")
check(listing(CACHE) == cache_before, "the cache was only read")
logs = glob.glob(os.path.join(LOGS, "make_submission_*.log"))
check(len(logs) == 1 and "[smoke]" in open(logs[0], encoding="utf-8").read(),
      "its log is in Work/smoke/logs")
check(not strays(before),
      f"nothing was written outside Work/smoke: {strays(before)}")
before = tree()
for what, argv in (
        ("--all", ["--smoke", "--all"]),
        ("a --cache outside", ["--smoke", "--cache", ELSEWHERE]),
        ("a --cache through ..", ["--smoke", "--cache",
                                  os.path.join(CACHE, "..", "..", "..", "x")]),
        ("an --out outside", ["--smoke", "--out",
                              os.path.join(ELSEWHERE, "out")]),
        ("a --model outside", ["--smoke", "--model", OUTSIDE_MODEL])):
    reset()
    loaded.clear()
    _, exc, out = run(S.main, *argv)
    check(isinstance(exc, SystemExit) and exc.code not in (0, None),
          f"{what} is refused: {out.strip().splitlines()[-1][:70]}")
check(not loaded and not strays(before), "and nothing was loaded or written")

print("\n8. a lower-case results/ is the old results folder, not Results/")
# the smoke variant gets a root of its own, as Work/smoke/Results is in use
WORK, WORK2 = fno_paths.WORK, os.path.join(BOX, "Work2")
FAKE = fno_paths.model_path(LABEL, os.path.join(WORK2, "smoke"))
os.makedirs(os.path.dirname(FAKE))
open(FAKE, "w").close()
OLD_RESULTS = (os.path.join(BOX, "results"),
               os.path.join(WORK2, "smoke", "results"))
for folder in OLD_RESULTS:
    os.makedirs(folder)
folds = os.path.isdir(os.path.join(BOX, "RESULTS"))
print(f"   ({'case is ignored' if folds else 'case counts'} on this "
      f"filesystem)")
for where, argv, work, folder in (
        ("Results/", ["--model", os.path.relpath(MODEL)], WORK,
         OLD_RESULTS[0]),
        ("Work/smoke/Results/", ["--smoke"], WORK2, OLD_RESULTS[1])):
    reset()
    P.WRITE_LOG = False
    loaded.clear()
    fno_paths.WORK = work
    _, exc, out = run(P.main, *argv, stop=(P, "load_checkpoint"))
    fno_paths.WORK = WORK
    if folds:
        check(isinstance(exc, SystemExit) and "results/ (the old results "
              "folder)" in said(exc) and "migrate_layout.py" in said(exc),
              f"a run into {where} is refused: {said(exc)[:60]}")
        check(not loaded and not os.listdir(folder),
              "before it loaded a model or wrote a file")
    else:
        check(isinstance(exc, Reached),
              f"where case counts, results/ is no bar to a run into {where}")
for folder in OLD_RESULTS:
    shutil.rmtree(folder)
shutil.rmtree(WORK2)

check(not (project_state() - PROJECT), "the project folder is untouched")
print(f"\n   ({time.time() - t_suite:.0f}s in all)")
if ok:
    shutil.rmtree(BOX, ignore_errors=True)
print("\n" + ("SMOKE PASS" if ok else "SMOKE FAIL"))
sys.exit(0 if ok else 1)
