"""Lean datasets and training flags: ch_gpu --lean, slicing, resume guard."""
import contextlib
import io as _io
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

from _paths import SCRATCH

os.chdir(SCRATCH)
import ch_gpu as C
import fno_train as T
T.use_fixed_step()

IMPORTED = {n: getattr(T, n) for n in ("DATASET", "GT_CACHE", "MODEL_OUT",
                                       "CHECKPOINT")}

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(_io.StringIO()):
        return fn(*a, **k)


dev = torch.device("cpu")
d = os.path.join(SCRATCH, "lean_t")
shutil.rmtree(d, ignore_errors=True)
os.makedirs(d)
C.L, C.WRITE_LOG, T.WRITE_LOG = 16, False, False

print("1. the lean file is ch.py's frames, streamed")
p2, p4 = os.path.join(d, "a2.npy"), os.path.join(d, "a4.npy")
quiet(C.write_lean, p2, 2.0, dev)
quiet(C.write_lean, p4, 4.0, dev)
a2, a4 = np.load(p2), np.load(p4)
check(a2.shape == (25, 20, 16, 16) and a2.dtype == np.float32,
      f"25 rows of 10 T frames, float32: {a2.shape} {a2.dtype}")
check(np.array_equal(a4[:, :20], a2),
      "a longer file is a bitwise prefix extension of a shorter one")
C.END = 3.0
npz = os.path.join(d, "g.npz")
quiet(C.generate, npz, dev)
with np.load(npz) as z:
    tx, ty, vx = z["train_X"], z["train_y"], z["val_X"]
new_npz = os.path.join(d, "fresh_npz", "data", "g.npz")
quiet(C.generate, new_npz, dev)
new_lean = os.path.join(d, "fresh_lean", "data", "a2.npy")
quiet(C.write_lean, new_lean, 2.0, dev)
check(os.path.isfile(new_npz) and os.path.isfile(new_lean)
      and np.array_equal(np.load(new_lean), a2),
      "ch_gpu makes the folder it writes the npz or the lean file into")
check(C.npz_path(1000) == os.path.join("Work", "data", "test_t1000.npz")
      and C.npz_path(300) == os.path.join("Work", "data", "test_t0300.npz"),
      f"and by default it writes Work/data/test_t<END>.npz, as the group protocol does "
      f"({C.npz_path(1000)})")
check(tx.shape[1] == 20
      and np.array_equal(a4[:20, :20], tx.astype(np.float32))
      and np.array_equal(a4[20:, :20], vx.astype(np.float32)),
      "its frames are generate()'s X (ch.py's) to t = 3, cast to float32")
check(np.array_equal(a4[:20, 10:30], ty.astype(np.float32)),
      "and ch.py's y is the same frames shifted by N_AHEAD // GAP")
with open(C.lean_sidecar(p2)) as fh:
    meta = json.load(fh)
check(meta["format"] == T.LEAN_FORMAT and meta["n_frames"] == 20
      and meta["shift"] == 10 and meta["n_train"] == 20
      and [r["split"] for r in meta["rows"]].count("val") == 5,
      "the sidecar records the format, frames, shift and row order")
try:
    quiet(C.write_lean, p2, 2.0, dev)
    refused = False
except FileExistsError:
    refused = True
check(refused, "an existing dataset is never overwritten")
try:
    quiet(C.write_lean, os.path.join(d, "b2.dat"), 2.0, dev)
    refused = False
except ValueError:
    refused = True
check(refused and not os.path.exists(os.path.join(d, "b2.dat"))
      and not os.path.exists(os.path.join(d, "b2.json")),
      "a name without .npy is refused (fno_train would not recognise it)")


class Dies(C._RowWriter):
    """Stops the run at the third frame, as a time limit would."""

    def __call__(self, j, frame):
        super().__call__(j, frame)
        if j == 2:
            raise KeyboardInterrupt("time limit")


real_writer, C._RowWriter = C._RowWriter, Dies
p_dead = os.path.join(d, "dead.npy")
try:
    quiet(C.write_lean, p_dead, 2.0, dev)
    died = False
except KeyboardInterrupt:
    died = True
C._RowWriter = real_writer
left = [f for f in os.listdir(d) if f.startswith("dead")]
check(died and not left,
      f"an interrupted write leaves nothing behind, not even the partial "
      f"file ({left})")

print("\n2. fno_train slices it, and matches the npz path")
unroll = T.MAX_UNROLL
T.MAX_UNROLL = 1                   # t = 3 is 20 frames; 8 steps need 81
T.TRAIN_ON_VAL, T.T_TRAIN = False, None
tr, va = quiet(T.load_trajectories, p4)
check(tr.shape == (20, 30, 16, 16) and va.shape == (5, 30, 16, 16),
      f"the whole horizon: {tr.shape}, {va.shape}")
T.T_TRAIN = 3.0
tr2, va2 = quiet(T.load_trajectories, p4)
T.T_TRAIN = None
nt, nv = quiet(T.load_trajectories, npz)
check(tr2.shape == nt.shape == (20, 20, 16, 16) and np.array_equal(tr2, nt)
      and np.array_equal(va2, nv),
      "sliced to t = 3 it is bitwise the array the npz loader returns")
s_lean = T.WindowSampler(tr2, "cpu", torch.device("cpu"), 1, True, 0)
s_npz = T.WindowSampler(nt, "cpu", torch.device("cpu"), 1, True, 0)
check(all(torch.equal(s_lean.sample(4), s_npz.sample(4)) for _ in range(5)),
      "and the sampler draws the same training windows from both")
T.TRAIN_ON_VAL, T.T_TRAIN = True, 3.0
tr3, va3 = quiet(T.load_trajectories, p4)
check(tr3.shape[0] == 25 and va3.shape[0] == 0
      and np.array_equal(tr3[:20], tr2) and np.array_equal(tr3[20:], va2),
      "with TRAIN_ON_VAL the 25 rows come back merged, no second copy")
check(va3.base is None,
      "and the empty val is its own array, not a view keeping train alive")
for bad, why, unr in ((5.0, "beyond the file's horizon", 1),
                      (3.05, "off the 0.1 frame grid", 1),
                      (3.0, "too short for MAX_UNROLL = 8 (81 frames)", 8)):
    T.T_TRAIN, T.MAX_UNROLL = bad, unr
    try:
        quiet(T.load_trajectories, p4)
        raised = False
    except ValueError:
        raised = True
    check(raised, f"--t-train {bad:g} is refused: {why}")
T.MAX_UNROLL = 1
T.T_TRAIN = None
bare = os.path.join(d, "bare.npy")
shutil.copy(p2, bare)
odd = os.path.join(d, "odd.bin")
shutil.copy(p2, odd)
wrong = os.path.join(d, "wrong.npy")
shutil.copy(p2, wrong)
with open(os.path.join(d, "wrong.json"), "w") as fh:
    json.dump({"format": "something_else"}, fh)
for path, why in ((bare, "an .npy without its .json sidecar"),
                  (odd, "an .npy under another name, no sidecar"),
                  (wrong, "a sidecar of another format")):
    try:
        quiet(T.load_trajectories, path)
        raised = ""
    except ValueError as exc:
        raised = str(exc)
    check("sidecar" in raised,
          f"refused before reading: {why}")
check(T.lean_meta(npz) is None, "an npz is not taken for a lean file")
T.T_TRAIN, T.TRAIN_ON_VAL = 2.0, False
try:
    quiet(T.load_trajectories, npz)
    raised = False
except ValueError:
    raised = True
check(raised, "an npz whose horizon differs from --t-train is refused")
T.T_TRAIN = None
T.MAX_UNROLL = unroll

print("\n3. the command line")
names = ("DATASET", "T_TRAIN", "TAG", "MODEL_OUT", "CHECKPOINT", "RESUME",
         "EPOCHS", "VAL_T_START", "VAL_T_END", "MONITOR_T_END")
saved = {n: getattr(T, n) for n in names}
check(quiet(T.apply_overrides, T.parse_args([])) == []
      and all(getattr(T, n) == saved[n] for n in names),
      "no flags change nothing")
check(IMPORTED == {"DATASET": os.path.join("Work", "data", "test.npz"),
                   "GT_CACHE": os.path.join("Work", "gt_cache"),
                   "MODEL_OUT": None, "CHECKPOINT": None},
      f"the file's defaults: the dataset and references under Work/, and "
      f"the outputs left to be derived ({IMPORTED})")
# outputs somebody set already: a flag that names the run drops them, so
# they are derived from the horizon and tag instead
STALE = ("stale.pt", "stale_last.pt")
T.MODEL_OUT, T.CHECKPOINT = STALE
quiet(T.apply_overrides, T.parse_args(["--dataset", p4, "--t-train", "2",
                                       "--val-t-end", "500"]))
check(T.DATASET == p4 and T.T_TRAIN == 2.0 and T.TAG == "T2"
      and T.MODEL_OUT is None and T.CHECKPOINT is None
      and T.VAL_T_END == 500.0,
      "--t-train sets the tag run_meta records (T2) and leaves the outputs "
      "to be named from the horizon")
T.MODEL_OUT, T.CHECKPOINT = STALE
quiet(T.apply_overrides, T.parse_args(["--tag", "mine", "--no-resume",
                                       "--epochs", "3"]))
check(T.TAG == "mine" and T.MODEL_OUT is None and T.CHECKPOINT is None
      and not T.RESUME and T.EPOCHS == 3,
      "--tag, --no-resume and --epochs")
for n, v in saved.items():
    setattr(T, n, v)
T.MODEL_OUT, T.CHECKPOINT = STALE
quiet(T.apply_overrides, T.parse_args(["--dataset", p4]))
check(T.TAG == "T4" and T.MODEL_OUT is None and T.CHECKPOINT is None,
      f"a lean --dataset alone is tagged by its horizon: {T.TAG}")
for n, v in saved.items():
    setattr(T, n, v)
for bad_tag in ("a_b", "lap", "nohard", "-x", "a b"):
    try:
        quiet(T.apply_overrides, T.parse_args([f"--tag={bad_tag}"]))
        said = None
    except SystemExit as exc:
        said = str(exc.code)
    check(said is not None and "--tag" in said and repr(bad_tag) in said
          and T.TAG == saved["TAG"],
          f"--tag {bad_tag!r} is refused with the rule: {said}")
    for n, v in saved.items():
        setattr(T, n, v)
for argv, why in ((["--dataset", npz], "an npz"),
                  (["--val-t-end", "500"], "a validation window"),
                  (["--monitor-t-end", "500"], "a monitor horizon")):
    try:
        quiet(T.apply_overrides, T.parse_args(argv))
        refused = False
        said = ""
    except SystemExit as exc:
        refused = True
        said = str(exc.code)
    check(refused and T.MODEL_OUT == saved["MODEL_OUT"]
          and "--t-train T or --tag NAME" in said
          and "does not overwrite fno_ch.pt" in said,
          f"{why} without --t-train/--tag is refused, never fno_ch.pt")
    for n, v in saved.items():
        setattr(T, n, v)
check(quiet(T.check_windows) is None, "the file's windows pass")
for vals, why in (({"VAL_T_START": 100.5}, "off the model-step grid"),
                  ({"VAL_T_START": 2000.0}, "an empty validation window"),
                  ({"VAL_T_START": 500.0, "MONITOR_T_END": 500.0,
                    "TRAIN_ON_VAL": True},
                   "monitor trajectories ending at the window start")):
    keep = {k: getattr(T, k) for k in vals}
    for k, v in vals.items():
        setattr(T, k, v)
    try:
        T.check_windows()
        refused = False
    except SystemExit:
        refused = True
    for k, v in keep.items():
        setattr(T, k, v)
    check(refused, f"refused at start-up: {why}")

print("\n4. a too-short reference is refused, not silently clamped")
T.CHANNELS, T.MODES, T.LAYERS = 8, 6, 2
model = T.build_model(n_ref=16, device="cpu")
times = np.arange(5.0)
frames = np.zeros((5, 16, 16))
try:
    T.rollout_score(model, times, frames, "cpu", t_start=0.0, t_end=10.0)
    raised = False
except ValueError:
    raised = True
check(raised, "rollout_score raises when the truth ends before t_end")

print("\n5. a run never resumes another run's checkpoint")
T.EPOCHS, T.STEPS_PER_EPOCH, T.BATCH_SIZE = 2, 4, 2
T.MAX_UNROLL, T.BPTT, T.VAL_BATCHES = 2, 2, 2
T.VAL_SEEDS, T.VAL_LATTICE = (11,), 16
T.MONITOR_SEEDS, T.MONITOR_OFFS, T.MONITOR_LATTICE = (21, 22), (-0.4, 0.4), 16
T.VAL_EVERY, T.PATIENCE, T.MAX_HOURS = 2, 0, 1.0
T.GT_CACHE = os.path.join(d, "gt")
T.TRAIN_ON_VAL, T.OFF_CRITICAL = True, False
base = ["--dataset", p4, "--t-train", "4", "--tag", "lt",
        "--val-t-start", "1", "--val-t-end", "3", "--monitor-t-end", "3"]
# --t-train 4 --tag lt is the model train_t0004_lap_lt, in a folder of its
# own (the Laplacian head is the default)
LT = os.path.join("Work", "models", "train_t0004_lap_lt")
LT_BEST = os.path.join(LT, "fno_ch.pt")
LT_LAST = os.path.join(LT, "fno_ch_last.pt")
LT_HIST = os.path.join(LT, "fno_ch_history.json")
shutil.rmtree(LT, ignore_errors=True)
baseline = {n: getattr(T, n) for n in names}


def train(argv):
    buf = _io.StringIO()
    code = None
    try:
        with contextlib.redirect_stdout(buf):
            T.main(argv)
    except SystemExit as exc:
        code = exc.code
    for n, v in baseline.items():
        if n != "RESUME":
            setattr(T, n, v)
    return buf.getvalue(), code


T.RESUME = True
log, code = train(base + ["--no-resume"])
ck = torch.load(LT_LAST, map_location="cpu", weights_only=False)
check(code is None and os.path.exists(LT_BEST) and os.path.exists(LT_HIST)
      and ck["run_meta"]["t_train"] == 4.0 and ck["run_meta"]["tag"] == "lt",
      f"a tagged run makes its folder {LT}, trains, and records its "
      f"settings: {ck.get('run_meta')}")
check(sorted(os.listdir(LT)) == ["fno_ch.pt", "fno_ch_config.json",
                                 "fno_ch_history.json", "fno_ch_last.pt"],
      f"the folder holds the best model, its config and history and the "
      f"resume state: {sorted(os.listdir(LT))}")
check(sum(LT_BEST in ln for ln in log.splitlines()
          if ln.startswith("[outputs]")) == 1,
      "and the log says once where the model goes")
check("lean dataset" in log and "all 25 rows" in log,
      "it trained from the lean file, all 25 rows merged")
refs = os.listdir(T.GT_CACHE) if os.path.isdir(T.GT_CACHE) else []
check(any(f.startswith("gt_l16_seed11_") for f in refs)
      and any(f.startswith("gt_l16_seed21_") for f in refs)
      and any(f.startswith("gt_l16_seed22_") for f in refs),
      f"the validation and monitor references went to the rebound GT_CACHE, "
      f"not the one the function was defined with: {sorted(refs)}")
T.RESUME = True
log, code = train(base)
check(code is None and f"[resume] {LT_LAST} at epoch 2" in log,
      "the same settings resume it")
T.RESUME = True
log, code = train(base[:-2] + ["--monitor-t-end", "4"])
check(isinstance(code, str) and "belongs to another run" in code,
      "different settings refuse to resume it, instead of mixing runs")
ck.pop("run_meta")
torch.save(ck, LT_LAST)
T.RESUME = True
log, code = train(base)
check(isinstance(code, str) and "no run settings recorded" in code,
      "an old checkpoint with no settings is not resumed by a tagged run")


def untagged(best=LT_BEST, last=LT_LAST):
    """The same run with its settings edited into the file, no flags: the
    outputs are named there too, and main() leaves a name that is set."""
    T.DATASET, T.T_TRAIN = p4, 4.0
    T.VAL_T_START, T.VAL_T_END, T.MONITOR_T_END = 1.0, 3.0, 3.0
    T.MODEL_OUT, T.CHECKPOINT = best, last
    T.RESUME = True


untagged()
log, code = train([])
check(isinstance(code, str) and "no run settings recorded" in code,
      "nor by an untagged run whose settings differ from the file's")
untagged()
file_default = T.DEFAULT_RUN_META
T.DEFAULT_RUN_META = {"dataset": "a4.npy", "t_train": 4.0,
                      "val_t_start": 1.0, "val_t_end": 3.0,
                      "monitor_t_end": 3.0, "tag": ""}
log, code = train([])
T.DEFAULT_RUN_META = file_default
check(code is None and f"[resume] {LT_LAST} at epoch 2" in log,
      "but the file's own run resumes it, as before run settings existed")


def stamp(path):
    return os.path.getsize(path), os.stat(path).st_mtime_ns


# the best model is another run's too: a fresh start must not overwrite it
other = base[:-2] + ["--monitor-t-end", "4"]
os.remove(LT_LAST)
kept = stamp(LT_BEST)
for flags, why in (([], "with no state to resume"),
                   (["--no-resume"], "with --no-resume")):
    T.RESUME = True
    log, code = train(other + flags)
    check(isinstance(code, str) and "best model of another run" in code
          and "--tag NAME" in code and LT_BEST in code
          and stamp(LT_BEST) == kept and not os.path.exists(LT_LAST),
          f"another run's best model is left alone {why}")
ck = torch.load(LT_BEST, map_location="cpu", weights_only=False)
ck.pop("run_meta")
torch.save(ck, LT_BEST)                    # as a migrated fno_ch.pt is
kept = stamp(LT_BEST)
T.RESUME = True
log, code = train(base)
check(isinstance(code, str) and "no run settings recorded" in code
      and "best model of another run" in code and stamp(LT_BEST) == kept,
      "and so is a model with no settings, which is the default run's")
T.DEFAULT_RUN_META = {"dataset": "a4.npy", "t_train": 4.0,
                      "val_t_start": 1.0, "val_t_end": 3.0,
                      "monitor_t_end": 3.0, "tag": "lt"}
T.RESUME = True
log, code = train(base)
T.DEFAULT_RUN_META = file_default
ck = torch.load(LT_BEST, map_location="cpu", weights_only=False)
check(code is None and ck.get("run_meta", {}).get("tag") == "lt",
      "unless this is the default run, which may take it over")
os.remove(LT_LAST)
T.RESUME = True
log, code = train(base)
check(code is None and os.path.exists(LT_LAST),
      "a run may start fresh over its own best model, as before")
T.RESUME = True
log, code = train(other)
check(isinstance(code, str) and "belongs to another run" in code
      and "--tag NAME" in code and "--no-resume keeps the old files as "
      "fno_ch_prev.pt and fno_ch_last_stale.pt" in code,
      "and a refused resume points to --tag, and says --no-resume keeps "
      "the old files")
log, code = train(["--dataset", os.path.join(d, "missing.npz"), "--tag", "x",
                   "--val-t-start", "3", "--val-t-end", "3"])
check(isinstance(code, str) and "empty" in code,
      "a bad window stops the run before any data is read")
T.RESUME = saved["RESUME"]
shutil.rmtree(LT, ignore_errors=True)

# the three files of a run need not share a folder, and none of the three
# writes may rely on another having made it
SPLIT_M = os.path.join(d, "split_m", "fno_ch.pt")
SPLIT_C = os.path.join(d, "split_c", "fno_ch_last.pt")
SPLIT_H = os.path.join(d, "split_m", "fno_ch_history.json")
untagged(SPLIT_M, SPLIT_C)
log, code = train([])
check(code is None and os.path.isfile(SPLIT_M) and os.path.isfile(SPLIT_C)
      and os.path.isfile(SPLIT_H),
      "the best model, the resume state and the history each make the "
      "folder they go into")
T.RESUME = saved["RESUME"]

print("\n6. the outputs are named from the training horizon")


def resolved(n_frames, **settings):
    """resolve_outputs under these settings (the others as a plain run has
    them): (the label, MODEL_OUT, CHECKPOINT, what it printed)."""
    keep = {n: getattr(T, n) for n in ("T_TRAIN", "TAG", "MODEL_OUT",
                                       "CHECKPOINT", "HARD_CONSTRAINTS",
                                       "RESUME", "ROOT", "LAPLACIAN_OUTPUT")}
    plain = dict(T_TRAIN=None, TAG="", MODEL_OUT=None, CHECKPOINT=None,
                 HARD_CONSTRAINTS=True, RESUME=True, ROOT=None,
                 LAPLACIAN_OUTPUT=False)
    for n, v in {**plain, **settings}.items():
        setattr(T, n, v)
    log = _io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            label = T.resolve_outputs(n_frames)
        return label, str(T.MODEL_OUT), str(T.CHECKPOINT), log.getvalue()
    finally:
        for n, v in keep.items():
            setattr(T, n, v)


def models(label):
    folder = os.path.join("Work", "models", label)
    return (label, os.path.join(folder, "fno_ch.pt"),
            os.path.join(folder, "fno_ch_last.pt"))


for n_frames, settings, label in (
        (7, dict(T_TRAIN=1000.0, TAG="T1000"), "train_t1000"),
        (7, dict(T_TRAIN=300.0), "train_t0300"),
        (2990, {}, "train_t0300"),
        (4990, {}, "train_t0500"),
        (2990, dict(TAG="T300"), "train_t0300"),
        (2990, dict(TAG="mine"), "train_t0300_mine"),
        (7, dict(T_TRAIN=500.0, TAG="T1000"), "train_t0500_T1000"),
        (7, dict(T_TRAIN=250.5, TAG="T250.5"), "train_t0250p5"),
        (7, dict(T_TRAIN=2000.0, TAG="T2000"), "train_t2000"),
        (7, dict(T_TRAIN=1000.0, HARD_CONSTRAINTS=False),
         "train_t1000_nohard"),
        (7, dict(T_TRAIN=1000.0, TAG="mine", HARD_CONSTRAINTS=False),
         "train_t1000_nohard_mine"),
        (7, dict(T_TRAIN=1000.0, LAPLACIAN_OUTPUT=True), "train_t1000_lap"),
        (7, dict(T_TRAIN=1000.0, TAG="mine", LAPLACIAN_OUTPUT=True),
         "train_t1000_lap_mine")):
    got = resolved(n_frames, **settings)
    check(got[:3] == models(label),
          f"{n_frames} frames, {settings}: {got[0]} in {got[1]}")
label, best, last, text = resolved(7, T_TRAIN=1000.0)
check(len(text.strip().splitlines()) == 1 and best in text and last in text,
      f"one line says where the model and the resume state go: "
      f"{text.strip()!r}")
for given, settings in (
        ("MODEL_OUT", dict(MODEL_OUT="mine/best.pt")),
        ("CHECKPOINT", dict(CHECKPOINT="mine/state.pt")),
        ("both", dict(MODEL_OUT="mine/best.pt", CHECKPOINT="mine/state.pt"))):
    got = resolved(7, T_TRAIN=1000.0, **settings)
    want = models("train_t1000")
    check(got[0] == "train_t1000"
          and got[1] == settings.get("MODEL_OUT", want[1])
          and got[2] == settings.get("CHECKPOINT", want[2]),
          f"{given} set beforehand is left as it is"
          + ("" if given == "both" else ", the other one derived")
          + f": {got[1]}, {got[2]}")



def refusal(n_frames, **settings):
    """What resolve_outputs says when it refuses these settings, or None."""
    try:
        resolved(n_frames, **settings)
    except SystemExit as exc:
        return str(exc.code)
    return None


# the resume state where the old code kept it, with none at the new place:
# a silent start from epoch 0 is the one thing to avoid
for old, settings in (
        ("fno_ch_T1000_last.pt", dict(T_TRAIN=1000.0, TAG="T1000")),
        ("fno_ch_mine_last.pt", dict(T_TRAIN=1000.0, TAG="mine")),
        ("fno_ch_last.pt", dict(T_TRAIN=500.0)),
        ("fno_ch_T1000_nohard_last.pt",
         dict(T_TRAIN=1000.0, TAG="T1000", HARD_CONSTRAINTS=False)),
        ("fno_ch_nohard_last.pt",
         dict(T_TRAIN=500.0, HARD_CONSTRAINTS=False))):
    with open(old, "w") as fh:
        fh.write("state")
    said = refusal(7, **settings)
    new = resolved(7, RESUME=False, **settings)[2]
    check(said is not None and old in said and new in said
          and "python migrate_layout.py" in said and "--apply" in said
          and "--no-resume" in said,
          f"{old} with nothing at {new}: refused, saying where it goes "
          f"({said and said[:50]})")
    os.remove(old)
OLD_STATE = "fno_ch_T1000_last.pt"
ASK = dict(T_TRAIN=1000.0, TAG="T1000")
NEW_STATE = models("train_t1000")[2]
with open(OLD_STATE, "w") as fh:
    fh.write("state")
check(refusal(7, RESUME=False, **ASK) is None,
      "with --no-resume it starts fresh as asked")
check(refusal(7, ROOT=os.path.join("Work", "smoke"), **ASK) is None,
      "and under --smoke the old state does not matter")
check(refusal(7, CHECKPOINT="mine/state.pt", **ASK) is None,
      "nor when the state's name was set rather than derived")
check(refusal(7, T_TRAIN=1000.0, TAG="other") is None,
      "nor a run of another name")
os.makedirs(os.path.dirname(NEW_STATE), exist_ok=True)
with open(NEW_STATE, "w") as fh:
    fh.write("state")
check(refusal(7, **ASK) is None,
      "nor once there is a state at the new place, which is then the one "
      "resumed")
shutil.rmtree(os.path.dirname(NEW_STATE))
os.remove(OLD_STATE)
check(refusal(7, **ASK) is None, "no old state, no refusal")

fresh = os.path.join(d, "fresh", "deeper")
T.save_model(model, os.path.join(fresh, "fno_ch.pt"), epoch=1)
check(os.path.isfile(os.path.join(fresh, "fno_ch.pt")),
      "save_model makes the folder it saves into")
T.save_model(model, os.path.join(d, "bare_name.pt"), epoch=1)
check(os.path.isfile(os.path.join(d, "bare_name.pt")),
      "and a name with no folder is as good as it was")

print("\n7. inputs from before the folder layout are used where they are")
DEFAULT_DATA = os.path.join("Work", "data", "test.npz")
DEFAULT_GT = os.path.join("Work", "gt_cache")


class Stop(Exception):
    pass


seen = []


def read_dataset(path, *a, **k):
    seen.append((path, T.GT_CACHE))
    raise Stop


real_load = T.load_trajectories
T.load_trajectories = read_dataset
box = os.path.join(d, "old_layout")


def start(*argv):
    """main() as far as the dataset: (what it was asked to read and the
    GT_CACHE it had then, what it printed)."""
    T.DATASET, T.GT_CACHE = DEFAULT_DATA, DEFAULT_GT
    seen.clear()
    log = _io.StringIO()
    here = os.getcwd()
    os.chdir(box)
    try:
        with contextlib.redirect_stdout(log):
            T.main(list(argv))
    except Stop:
        pass
    finally:
        os.chdir(here)
        for n, v in baseline.items():
            setattr(T, n, v)
    return list(seen), log.getvalue()


def layout_lines(text):
    return [ln for ln in text.splitlines() if ln.startswith("[layout]")]


shutil.rmtree(box, ignore_errors=True)
os.makedirs(box)
try:
    got, text = start()
    check(got == [(DEFAULT_DATA, DEFAULT_GT)] and not layout_lines(text),
          f"neither place has them: the new names, nothing said ({got})")
    with open(os.path.join(box, "test.npz"), "wb") as fh:
        fh.write(b"x")
    os.makedirs(os.path.join(box, "gt_cache"))
    got, text = start()
    check(got == [("test.npz", "gt_cache")]
          and len(layout_lines(text)) == 2
          and all("migrate_layout.py" in ln for ln in layout_lines(text)),
          f"only the old test.npz and gt_cache/ are there: they are used, "
          f"and two lines say so ({got}, {layout_lines(text)})")
    got, text = start("--dataset", "other.npz", "--tag", "x")
    check(got == [("other.npz", "gt_cache")],
          f"a dataset named on the command line is not swapped for the old "
          f"test.npz ({got})")
    os.makedirs(os.path.join(box, "Work", "data"))
    os.makedirs(os.path.join(box, "Work", "gt_cache"))
    with open(os.path.join(box, DEFAULT_DATA), "wb") as fh:
        fh.write(b"y")
    got, text = start()
    check(got == [(DEFAULT_DATA, DEFAULT_GT)] and not layout_lines(text),
          f"with both there the new ones win, silently ({got})")
    shutil.rmtree(os.path.join(box, "Work", "gt_cache"))
    got, text = start()
    check(got == [(DEFAULT_DATA, "gt_cache")] and len(layout_lines(text)) == 1,
          f"each is looked for on its own ({got})")
finally:
    T.load_trajectories = real_load
    T.DATASET, T.GT_CACHE = IMPORTED["DATASET"], os.path.join(d, "gt")



def ch_gpu(*args):
    """ch_gpu.py as a script from the sandbox: (its output, exit status)."""
    run = subprocess.run([sys.executable,
                          os.path.join(os.path.dirname(C.__file__),
                                       "ch_gpu.py"), *args],
                         cwd=box, capture_output=True, text=True,
                         timeout=600)
    return run.stdout, run.returncode


shutil.rmtree(box, ignore_errors=True)
os.makedirs(box)
os.makedirs(os.path.join(box, "Work", "data"))
# always with --end: a run without one would generate a whole dataset
for end, args in ((1000.0, ("--end", "1000")), (300.0, ("--end", "300"))):
    have = os.path.join(box, C.npz_path(end))
    with open(have, "wb") as fh:
        fh.write(b"x")
    text, code = ch_gpu(*args)
    with open(have, "rb") as fh:
        kept = fh.read() == b"x"
    check(code == 0 and f"{os.path.basename(have)} already exists" in text
          and kept and os.listdir(os.path.dirname(have))
          == [os.path.basename(have)],
          f"ch_gpu {' '.join(args)} finds "
          f"{C.npz_path(end)} and does not generate another (exit {code})")
    os.remove(have)
with open(os.path.join(box, "another.npz"), "wb") as fh:
    fh.write(b"y")
text, code = ch_gpu("--out", "another.npz")
check(code == 0 and "another.npz already exists" in text,
      "and an --out of its own is only looked for where it says")

shutil.rmtree(d, ignore_errors=True)
print("\n" + ("LEAN PASS" if ok else "LEAN FAIL"))
sys.exit(0 if ok else 1)
