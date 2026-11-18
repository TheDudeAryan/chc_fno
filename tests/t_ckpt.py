"""Checkpoint safety: rotation, atomic saves, the schedule stamp, the config
file, the non-finite gradient skip and the CPU guard inside SLURM.

Every scenario is a tiny CPU run of the real fno_train.main, in a folder of
its own under tests/_scratch.
"""
import collections
import contextlib
import datetime
import hashlib
import io as _io
import json
import os
import re
import shutil
import socket
import sys

import numpy as np
import torch

from _paths import SCRATCH

os.chdir(SCRATCH)
import fno_predict as P
import fno_train as T
T.use_fixed_step()

setattr(torch.cuda, "is_available", lambda: False)

D = os.path.join(SCRATCH, "ckpt_t")
shutil.rmtree(D, ignore_errors=True)
os.makedirs(D)

L, NF = 32, 40
rng = np.random.default_rng(0)
traj = np.tanh(rng.standard_normal((5, NF, L, L)) * 0.5)
traj -= traj.mean(axis=(-2, -1), keepdims=True)
for i, off in enumerate((-0.4, -0.2, 0.0, 0.2, 0.4)):
    traj[i] += off

DEFAULTS = dict(T.config_constants())
DEFAULT_META = T.DEFAULT_RUN_META
BASE = dict(
    WRITE_LOG=False, EPOCHS=2, STEPS_PER_EPOCH=3, BATCH_SIZE=2, MAX_UNROLL=2,
    BPTT=2, VAL_BATCHES=2, VAL_SEEDS=(11,), VAL_LATTICE=32, VAL_T_END=105.0,
    VAL_T_START=100.0, MONITOR_SEEDS=(21, 22), MONITOR_OFFS=(-0.4, 0.4),
    MONITOR_LATTICE=32, MONITOR_T_END=105.0, VAL_EVERY=1, PATIENCE=0,
    CHANNELS=8, MODES=6, LAYERS=2, MAX_HOURS=None, RESUME=False,
    TRAIN_ON_VAL=True, GT_CACHE=os.path.join(D, "gt"), REPLACE_WAIT=0.0)

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield old
    finally:
        setattr(obj, name, old)


def tiny_data(*a, **k):
    dt = T.numpy_dtype()
    return traj[:4].astype(dt), traj[4:].astype(dt)


def no_off_critical(lattice, n_frames, device="cpu", verbose=True):
    return np.zeros((2, n_frames, lattice, lattice), T.numpy_dtype())


setattr(T, "load_trajectories", tiny_data)
setattr(T, "off_critical_trajectories", no_off_critical)


def run(folder, argv=(), **cfg):
    """T.main in ``folder`` under the tiny settings and these overrides:
    (what it printed, the SystemExit code or the exception, or None)."""
    for k, v in {**DEFAULTS, **BASE, **cfg}.items():
        setattr(T, k, v)
    T.DEFAULT_RUN_META = cfg.get("DEFAULT_RUN_META", DEFAULT_META)
    T.MODEL_OUT = os.path.join(D, folder, "fno_ch.pt")
    T.CHECKPOINT = os.path.join(D, folder, "fno_ch_last.pt")
    buf, code = _io.StringIO(), None
    try:
        with contextlib.redirect_stdout(buf):
            T.main(list(argv))
    except SystemExit as exc:
        code = exc.code
    except Exception as exc:
        code = exc
    return buf.getvalue(), code


def path(folder, name):
    return os.path.join(D, folder, name)


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def contents(folder):
    """{file name: (sha, size)} of what a folder holds."""
    top = os.path.join(D, folder)
    names = os.listdir(top) if os.path.isdir(top) else []
    return {n: (sha(os.path.join(top, n)),
                os.path.getsize(os.path.join(top, n))) for n in names}


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def finite(state):
    return all(bool(torch.isfinite(v).all()) for v in state.values()
               if v.is_floating_point())


def epoch_lines(out):
    return [ln for ln in out.splitlines() if ln.startswith("epoch ")]


STAMPED = r"_\d{8}-\d{6}(?:-\d+)?"
CURRENT = {"fno_ch.pt", "fno_ch_config.json", "fno_ch_history.json",
           "fno_ch_last.pt"}

print("1. a fresh start moves this run's own old files aside")
out, code = run("rot", ["--epochs", "2"])
one = contents("rot")
check(code is None and set(one) == CURRENT and "[keep]" not in out,
      f"the first start has nothing to move: {sorted(one)}")
out, code = run("rot", SEED=1)
two = contents("rot")
want = CURRENT | {"fno_ch_prev.pt", "fno_ch_prev_history.json",
                  "fno_ch_prev_config.json", "fno_ch_last_stale.pt"}
check(code is None and set(two) == want,
      f"RESUME False: the best model, history, config and state become "
      f"fno_ch_prev.pt, fno_ch_prev_history.json, fno_ch_prev_config.json "
      f"and fno_ch_last_stale.pt ({sorted(set(two) - CURRENT)})")
check(all(two[new] == one[old] for old, new in (
    ("fno_ch.pt", "fno_ch_prev.pt"),
    ("fno_ch_history.json", "fno_ch_prev_history.json"),
    ("fno_ch_config.json", "fno_ch_prev_config.json"),
    ("fno_ch_last.pt", "fno_ch_last_stale.pt"))),
    "and each kept file is the old one, byte for byte")
check(two["fno_ch.pt"] != one["fno_ch.pt"]
      and two["fno_ch_last.pt"] != one["fno_ch_last.pt"],
      "while the files under the old names are the new run's")
keeps = [ln for ln in out.splitlines() if ln.startswith("[keep]")]
check(len(keeps) == 5 and all("fno_ch_prev" in k or "stale" in k
                              for k in keeps[:-1])
      and sum(bool(re.search(r"\(\d+\.\d (MB|KB)\)", k))
              for k in keeps) == 4
      and "resume is off" in out and "nothing was deleted" in keeps[-1],
      f"it says what was kept, how big it is and why: {keeps[:1]}")

out, code = run("rot", SEED=2)
three = contents("rot")
rotated = set(three) - CURRENT
check(code is None and len(three) == 12
      and all(three[n] == two[n] for n in two
              if n.startswith(("fno_ch_prev.", "fno_ch_prev_history.",
                               "fno_ch_prev_config.", "fno_ch_last_stale."))),
      "a second rotation leaves the first one's files where they were")
check(all(any(re.fullmatch(pat, n) for n in rotated) for pat in (
    rf"fno_ch_prev{STAMPED}\.pt", rf"fno_ch_prev{STAMPED}_history\.json",
    rf"fno_ch_prev{STAMPED}_config\.json",
    rf"fno_ch_last_stale{STAMPED}\.pt")),
    f"and puts the second under a timestamp: "
    f"{sorted(n for n in rotated if re.search(STAMPED, n))}")
before = collections.Counter(two.values())
after = collections.Counter(three.values())
check(not before - after and sum(after.values()) - sum(before.values()) == 4
      and sum(s for _, s in before.elements())
      <= sum(s for _, s in after.elements()),
      "no file is lost: every earlier file is still there, byte for byte, "
      "and only the four new ones are added")
print("\n   free_name on its own")
fixed = "20261014-101010"
with patched(T.time, "strftime", lambda fmt: fixed):
    f = os.path.join(D, "fn.pt")
    check(T.free_name(f) == f, "a free name is returned as it is")
    open(f, "wb").close()
    g = T.free_name(f)
    check(g == os.path.join(D, f"fn_{fixed}.pt"),
          f"a taken name gets _<YYYYmmdd-HHMMSS> before the extension "
          f"({os.path.basename(g)})")
    open(g, "wb").close()
    check(T.free_name(f) == os.path.join(D, f"fn_{fixed}-2.pt"),
          "and in the same second the next one is numbered")

print("\n   no state to resume, a different model, and a refused start")
os.remove(path("rot", "fno_ch_last.pt"))
before = contents("rot")
out, code = run("rot", RESUME=True, SEED=3)
after = contents("rot")
moved = {n for n in after if n not in before}
check(code is None and "[keep] best model" in out
      and any(re.fullmatch(rf"fno_ch_prev{STAMPED}\.pt", n) for n in moved)
      and not any("stale" in n for n in moved),
      "RESUME True with no state file: the best model is moved aside and "
      "there is no state to keep")
before = contents("rot")
out, code = run("rot", RESUME=True, CHANNELS=10)
check(isinstance(code, str) and "written by a different model" in code
      and "channels: 8 -> 10" in code
      and "--channels 8 --keep-folder" in code
      and "--no-resume" in code and contents("rot") == before,
      "a state from a different model is refused, not restarted over: the "
      "message says how to go on with it or start anew, and nothing moves")
out, code = run("rot", RESUME=False, CHANNELS=10)
after = contents("rot")
check(code is None
      and any(re.fullmatch(rf"fno_ch_last_stale{STAMPED}\.pt", n)
              for n in set(after) - set(before))
      and "which will not be resumed (resume is off" in out,
      "with --no-resume the other model's state is kept as a _stale file, "
      "with the reason in the log")

before = contents("rot")
out, code = run("rot", RESUME=True, VAL_T_END=110.0, CHANNELS=10)
check(isinstance(code, str) and "belongs to another run" in code
      and "--no-resume keeps the old files as fno_ch_prev.pt and "
      "fno_ch_last_stale.pt" in code and contents("rot") == before,
      "a state of another run is still refused, nothing is moved, and the "
      "message says --no-resume keeps the old files")
out, code = run("rot", VAL_T_END=110.0)
check(isinstance(code, str) and "best model of another run" in code
      and "--tag NAME" in code and contents("rot") == before,
      "and so is a fresh start over another run's best model")

print("\n2. saves go through <path>.partial")
model = T.build_model(n_ref=L, device="cpu")
for k, v in {**DEFAULTS, **BASE}.items():
    setattr(T, k, v)
good = os.path.join(D, "atomic", "m.pt")
T.save_model(model, good, epoch=1)
old = sha(good)
real_save = torch.save


def half_then_fail(obj, f, *a, **k):
    with open(f, "wb") as fh:
        fh.write(b"half a checkpoint")
    raise OSError("disk full")


with patched(torch, "save", half_then_fail):
    try:
        T.save_model(model, good, epoch=2)
        err = None
    except OSError as exc:
        err = exc
check(err is not None and sha(good) == old
      and not os.path.exists(good + ".partial"),
      "a save that dies half way leaves the old file intact and no partial")
fresh_path = os.path.join(D, "atomic", "new.pt")
with patched(torch, "save", half_then_fail):
    try:
        T.save_model(model, fresh_path)
    except OSError:
        pass
check(not os.path.exists(fresh_path),
      "and a first save that dies leaves nothing under the final name")

calls = []
real_replace = os.replace


def busy_twice(src, dst):
    calls.append(src)
    if len(calls) <= 2:
        raise PermissionError(13, "in use")
    real_replace(src, dst)


with patched(os, "replace", busy_twice):
    T.save_model(model, good, epoch=3)
check(len(calls) == 3 and load(good)["epoch"] == 3
      and not os.path.exists(good + ".partial"),
      "a PermissionError on the rename is tried again and then succeeds")
calls.clear()


def always_busy(src, dst):
    calls.append(src)
    raise PermissionError(13, "in use")


kept = sha(good)
with patched(os, "replace", always_busy):
    try:
        T.save_model(model, good, epoch=4)
        err = None
    except PermissionError as exc:
        err = exc
check(err is not None and len(calls) == 1 + T.REPLACE_RETRIES == 4
      and sha(good) == kept and load(good + ".partial")["epoch"] == 4,
      "a rename that keeps failing is tried 3 more times, then raised, with "
      "the old file intact and the finished new one left as .partial")
os.remove(good + ".partial")

events = []
real_fsync = os.fsync


def spy_fsync(fd):
    events.append(("fsync", os.fstat(fd).st_size))
    real_fsync(fd)


def spy_replace(src, dst):
    events.append(("replace", os.path.getsize(src)))
    real_replace(src, dst)


with patched(os, "fsync", spy_fsync), patched(os, "replace", spy_replace):
    T.save_model(model, good, epoch=5)
    first = list(events)
    events.clear()
    T._write_json(good + ".json", [{"epoch": 5}])
check([e[0] for e in first] == ["fsync", "replace"]
      and first[0][1] == first[1][1] == os.path.getsize(good) > 0
      and [e[0] for e in events] == ["fsync", "replace"],
      "the new file is flushed to disk, whole, before it is renamed into "
      "place, for a model and for a json file alike")


def fsync_fails(fd):
    raise OSError(5, "input/output error")


kept = sha(good)
with patched(os, "fsync", fsync_fails):
    try:
        T.save_model(model, good, epoch=6)
        err = None
    except OSError as exc:
        err = exc
check(err is not None and sha(good) == kept
      and not os.path.exists(good + ".partial"),
      "a flush that fails leaves the old file intact and no partial")


def state_save_dies_at(*dying):
    """A torch.save that dies half way on these state saves (1 = the first)
    and works as usual otherwise."""
    seen = []

    def save(obj, f, *a, **k):
        if str(f).endswith("fno_ch_last.pt.partial"):
            seen.append(f)
            if len(seen) in dying:
                with open(f, "wb") as fh:
                    fh.write(b"half a state")
                raise OSError("disk full")
        real_save(obj, f, *a, **k)
    return save


def warnings_in(out):
    return [ln for ln in out.splitlines() if ln.startswith("[save] WARNING")]


with patched(torch, "save", state_save_dies_at(2)):
    out, code = run("atomic_run", EPOCHS=3)
ck = load(path("atomic_run", "fno_ch_last.pt"))
warned = warnings_in(out)
with open(path("atomic_run", "fno_ch_history.json")) as fh:
    done_epochs = [r["epoch"] for r in json.load(fh)]
check(code is None and len(epoch_lines(out)) == 3 and len(warned) == 1
      and "epoch 2" in warned[0] and "disk full" in warned[0]
      and "tries again" in warned[0],
      f"a state save that dies at epoch 2 is a warning, not the end of the "
      f"run: {warned[:1]}")
check(ck["epoch"] == 3 and finite(ck["state_dict"])
      and done_epochs == [1, 2, 3]
      and not any(n.endswith(".partial") for n in contents("atomic_run")),
      "and the next epoch saves the state again, whole")
with patched(torch, "save", state_save_dies_at(2, 3)):
    out, code = run("atomic_two", EPOCHS=3)
ck = load(path("atomic_two", "fno_ch_last.pt"))
check(code is None and len(epoch_lines(out)) == 3
      and len(warnings_in(out)) == 2 and ck["epoch"] == 1
      and finite(ck["state_dict"])
      and not any(n.endswith(".partial") for n in contents("atomic_two"))
      and os.path.exists(path("atomic_two", "fno_ch.pt")),
      "when every later save dies too, the run still finishes, and the "
      "state it leaves is the epoch-1 one, whole and loadable")


def busy_at_epoch_2(src, dst):
    if str(dst).endswith("fno_ch_last.pt") and load(src)["epoch"] == 2:
        calls.append(src)
        raise PermissionError(13, "in use")
    real_replace(src, dst)


calls.clear()
with patched(os, "replace", busy_at_epoch_2):
    out, code = run("atomic_busy", EPOCHS=3)
ck = load(path("atomic_busy", "fno_ch_last.pt"))
check(code is None and len(calls) == 1 + T.REPLACE_RETRIES
      and out.count("trying again") == T.REPLACE_RETRIES
      and len(warnings_in(out)) == 1 and "epoch 2" in warnings_in(out)[0]
      and ck["epoch"] == 3
      and not any(n.endswith(".partial") for n in contents("atomic_busy")),
      "a rename that is still refused after its retries is a warning too, "
      "and the next epoch's state goes in")
calls.clear()


def best_save_dies(obj, f, *a, **k):
    if str(f).endswith("fno_ch.pt.partial"):
        with open(f, "wb") as fh:
            fh.write(b"half a model")
        raise OSError("disk full")
    real_save(obj, f, *a, **k)


with patched(torch, "save", best_save_dies):
    out, code = run("atomic_best", EPOCHS=3)
check(isinstance(code, OSError) and not warnings_in(out)
      and not os.path.exists(path("atomic_best", "fno_ch.pt"))
      and not os.path.exists(path("atomic_best", "fno_ch.pt.partial")),
      "but a best model that cannot be saved still stops the run")
hist = path("atomic_run", "fno_ch_history.json")


def dump_dies(obj, fh, **k):
    fh.write("[1")
    raise OSError("disk full")


T._write_json(hist, [{"epoch": 1}])
with patched(json, "dump", dump_dies):
    try:
        T._write_json(hist, [{"epoch": 2}])
        err = None
    except OSError as exc:
        err = exc
with open(hist) as fh:
    check(err is not None and json.load(fh) == [{"epoch": 1}]
          and not os.path.exists(hist + ".partial"),
          "the history and config files are written the same way")

print("\n3. the schedule stamp")
STAMP = ["EPOCHS", "STEPS_PER_EPOCH", "LR", "LR_FLOOR", "LR_STAGE_DECAY",
         "BATCH_SIZE", "MAX_UNROLL", "BPTT", "WARM_FRAC", "RAMP_FRAC",
         "WEIGHT_DECAY", "GRAD_CLIP", "H1_WEIGHT", "MASS_WEIGHT",
         "CONTRACTIVITY_WEIGHT", "AMP_TARGET", "AMP_EPS", "SMOOTH_WEIGHT",
         "NOISE_STD", "NOISE_PROB", "TRAIN_ON_VAL", "OFF_CRITICAL",
         "OFF_CRITICAL_OFFS", "OFF_CRITICAL_TRAJ", "OFF_CRITICAL_SEED0",
         "SEED", "PRECISION", "EMA", "SELECT", "AMP", "VAL_SEEDS",
         "VAL_LATTICE", "R2_THRESHOLD", "SURVIVAL_MIN_DELTA", "PSI_MAX",
         "LOSS_SPIKE_FACTOR", "DIVERGE_FACTOR", "MAX_RECOVERIES"]
out, code = run("sched", OFF_CRITICAL=True)
state_file = path("sched", "fno_ch_last.pt")
ck = load(state_file)
check(code is None and set(ck["schedule"]) == set(STAMP),
      f"the state carries a stamp of {len(ck['schedule'])} settings")
check(not {"PATIENCE", "MIN_DELTA", "MAX_HOURS", "VAL_EVERY", "WRITE_LOG",
           "BUDGET_RESERVE_MIN"} & set(ck["schedule"]),
      "and not patience, the minimum gain, the time budget, logging or "
      "VAL_EVERY")
pristine = contents("sched")
out, code = run("sched", RESUME=True, OFF_CRITICAL=True, PATIENCE=5,
                MIN_DELTA=0.5, VAL_EVERY=2, MAX_HOURS=3.0)
check(code is None and f"[resume] {state_file} at epoch 2" in out,
      "settings outside the stamp may change on a resume")


def changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 1.5 + 0.01
    if isinstance(value, tuple):
        return tuple(x + 0.01 for x in value)
    if value in ("score", "survival"):
        return "survival" if value == "score" else "score"
    if value == "off":
        return "bf16"
    return "fp64" if value == "fp32" else "fp32"


refused = []
for key in STAMP:
    pristine = contents("sched")
    kw: dict[str, object] = dict(RESUME=True, OFF_CRITICAL=True)
    kw[key] = changed(ck["schedule"][key])
    out, code = run("sched", **kw)
    refused.append(isinstance(code, str) and f"  {key}: " in code
                   and len(re.findall(r"^  \w+: ", code, re.M)) == 1
                   and "--no-resume" in code and "fno_ch_prev.pt" in code
                   and "[keep]" not in out
                   and contents("sched") == pristine)
    if not refused[-1]:
        print("   |", key, repr(code)[:200])
check(all(refused) and len(refused) == len(STAMP),
      f"each of the {len(STAMP)} settings, changed alone, refuses the "
      f"resume, names itself, and moves and writes nothing "
      f"({refused.count(True)} did)")
out, code = run("sched", RESUME=True, OFF_CRITICAL=True, EPOCHS=5, LR=1e-3)
check(isinstance(code, str) and "EPOCHS: 2 -> 5" in code
      and "LR: 0.002 -> 0.001" in code and "--accept-schedule" in code,
      "and the message lists every difference as old -> new, and says "
      "--accept-schedule goes on anyway")

print("\n   a stamp from before the last nine settings was known")
NEWEST = STAMP[-9:]
out, code = run("oldstamp")
old_state = path("oldstamp", "fno_ch_last.pt")
older = load(old_state)
for key in NEWEST:
    older["schedule"].pop(key)
torch.save(older, old_state)
out, code = run("oldstamp", RESUME=True, AMP="bf16", VAL_SEEDS=(12,),
                R2_THRESHOLD=0.5, SURVIVAL_MIN_DELTA=20.0, PSI_MAX=6.0,
                LOSS_SPIKE_FACTOR=9.0, DIVERGE_FACTOR=4.0, MAX_RECOVERIES=3)
check(NEWEST[0] == "AMP" and code is None
      and f"[resume] {old_state} at epoch 2" in out
      and "another training schedule" not in out,
      "a setting the saved stamp does not have is not compared, so a state "
      "from before the nine were stamped resumes with them changed")

print("\n   --accept-schedule")
out, code = run("accept")
state_file2 = path("accept", "fno_ch_last.pt")
before = contents("accept")
out, code = run("accept", RESUME=True, EPOCHS=3, LR=1e-3)
check(isinstance(code, str) and "--accept-schedule" in code
      and contents("accept") == before,
      "without the flag the changed schedule is refused, as before")
out, code = run("accept", ["--accept-schedule"], RESUME=True, EPOCHS=3,
                LR=1e-3)
st = load(state_file2)
with open(path("accept", "fno_ch_history.json")) as fh:
    hist = json.load(fh)
note = {"EPOCHS": [2, 3], "LR": [0.002, 0.001]}
check(code is None and f"[resume] {state_file2} at epoch 2" in out
      and "  EPOCHS: 2 -> 3" in out and "  LR: 0.002 -> 0.001" in out
      and len(epoch_lines(out)) == 1,
      "with it the resume goes on, prints the differences, and trains the "
      "epoch that was left")
check([r.get("schedule_change") for r in hist] == [None, None, note]
      and [r.get("schedule_change") for r in st["history"]]
      == [None, None, note],
      "the record of the first epoch after the resume holds the "
      "differences, in the history file and in the state, and no other does")
check(st["epoch"] == 3 and st["schedule"]["EPOCHS"] == 3
      and st["schedule"]["LR"] == 1e-3,
      "and the state holds the new schedule")
out, code = run("accept", RESUME=True, EPOCHS=3, LR=1e-3)
check(code is None and f"[resume] {state_file2} at epoch 3" in out,
      "so the next resume needs no flag")
before = contents("accept")
out, code = run("accept", RESUME=True)
check(isinstance(code, str) and "EPOCHS: 3 -> 2" in code
      and contents("accept") == before,
      "while the old schedule is now the one that is refused")
out, code = run("accept", ["--accept-schedule"], RESUME=True, EPOCHS=4,
                LR=1e-3)
with open(path("accept", "fno_ch_history.json")) as fh:
    hist = json.load(fh)
check(code is None and [r.get("schedule_change") for r in hist]
      == [None, None, note, {"EPOCHS": [3, 4]}],
      "a later resume notes only what it changed itself")

out, code = run("accept_same", ["--accept-schedule"], EPOCHS=3, PATIENCE=1,
                MIN_DELTA=10.0)
check(code is None and len(epoch_lines(out)) == 1
      and load(path("accept_same", "fno_ch_last.pt"))["epoch"] == 1,
      "a run stopped after epoch 1 by its patience leaves a state to resume")
out, code = run("accept_same", ["--accept-schedule"], RESUME=True, EPOCHS=3)
with open(path("accept_same", "fno_ch_history.json")) as fh:
    hist = json.load(fh)
check(code is None and len(hist) == 3 and "another training" not in out
      and not any("schedule_change" in r for r in hist),
      "the flag with nothing to accept prints and records nothing")

before = contents("accept")
out, code = run("accept", ["--accept-schedule"], RESUME=True, EPOCHS=4,
                LR=1e-3, CHANNELS=10)
check(isinstance(code, str) and "written by a different model" in code
      and contents("accept") == before,
      "it accepts a schedule and nothing else: another model is refused")
out, code = run("accept", ["--accept-schedule"], RESUME=True, EPOCHS=4,
                LR=1e-3, VAL_T_END=110.0)
check(isinstance(code, str) and "belongs to another run" in code
      and contents("accept") == before,
      "and so is another run")

print("\n   a state from before the stamp resumes as it did")
legacy = load(state_file)
for key in ("schedule", "train_config"):
    legacy.pop(key)
meta = legacy.pop("run_meta")
torch.save(legacy, state_file)
out, code = run("sched", RESUME=True, OFF_CRITICAL=True, EPOCHS=3,
                DEFAULT_RUN_META=meta)
check(code is None and f"[resume] {state_file} at epoch 2" in out
      and "no schedule stamp" in out and len(epoch_lines(out)) == 1,
      "no stamp: it resumes, with a note, whatever the schedule has become")
torch.save(legacy, state_file)
out, code = run("sched", RESUME=True, OFF_CRITICAL=True,
                DEFAULT_RUN_META={"dataset": "other.npy"})
check(isinstance(code, str) and "no run settings recorded" in code,
      "and the rule that only the default run resumes a state with no run "
      "settings still holds")

print("\n4. the config file")
out, code = run("cfg", ["--epochs", "2"])
with open(path("cfg", "fno_ch_config.json")) as fh:
    cfg = json.load(fh)
now = json.loads(json.dumps({"constants": dict(T.config_constants()),
                             "schedule": T.schedule_stamp()}))
check(code is None and set(cfg) == {"label", "seed", "size", "run_meta",
                                    "schedule", "constants", "launches"},
      f"fno_ch_config.json beside the model holds {sorted(cfg)}")
check(cfg["constants"] == now["constants"]
      and cfg["constants"]["EPOCHS"] == 2 and cfg["constants"]["LR"] == T.LR
      and set(cfg["constants"]) == {k for k, _ in T.config_constants()},
      "every upper-case constant, as the log's configuration block lists them")
check(cfg["run_meta"] == T.run_meta()
      and cfg["schedule"] == now["schedule"]
      and cfg["label"] == "train_t0005_lap" and cfg["seed"] == T.SEED
      and cfg["size"] == {"channels": T.CHANNELS, "modes": T.MODES},
      f"the run settings, the schedule, the label ({cfg['label']}), the "
      f"seed and the size")
launch = cfg["launches"][0]
check(len(cfg["launches"]) == 1
      and set(launch) == {"time", "host", "argv", "resumed_from"}
      and launch["argv"] == ["--epochs", "2"]
      and launch["host"] == socket.gethostname()
      and launch["resumed_from"] is None,
      "one launch: the command line, the host, and no epoch resumed from")
check(datetime.datetime.fromisoformat(launch["time"]).tzinfo is not None,
      f"and the time with its zone ({launch['time']})")
best = load(path("cfg", "fno_ch.pt"))
state = load(path("cfg", "fno_ch_last.pt"))
check(json.loads(json.dumps(best["train_config"])) == cfg
      and json.loads(json.dumps(state["train_config"])) == cfg,
      "the same record is in both checkpoints")
del best["train_config"]
torch.save(best, path("cfg", "old.pt"))
check(T.load_checkpoint(path("cfg", "old.pt"))[0] is not None,
      "a checkpoint without it still loads")
before = contents("cfg")
out, code = run("cfg", ["--epochs", "9"], RESUME=True)
check(isinstance(code, str) and contents("cfg") == before,
      "a start the schedule stamp refuses does not rewrite the config file")

print("\n   one entry per launch")
cfg_file = path("cfg", "fno_ch_config.json")


def text(p):
    with open(p) as fh:
        return fh.read()


def launches(p=cfg_file):
    return json.loads(text(p))["launches"]


out, code = run("cfg", ["--epochs", "3", "--accept-schedule"], RESUME=True)
two = launches()
check(code is None and len(two) == 2 and two[0] == launch
      and two[1]["argv"] == ["--epochs", "3", "--accept-schedule"]
      and two[1]["resumed_from"] == 2
      and two[1]["host"] == socket.gethostname()
      and two[1]["time"] >= launch["time"],
      "a resume adds its own entry, with the epoch it took up from, and "
      "keeps the first")
check(load(path("cfg", "fno_ch_last.pt"))["train_config"]["launches"] == two
      and json.loads(text(cfg_file))["constants"]["EPOCHS"] == 3,
      "the state carries the list, and the rest of the file is the latest "
      "launch's")
out, code = run("cfg", ["--epochs", "3"], RESUME=True)
three = launches()
check(code is None and len(three) == 3 and three[:2] == two
      and three[2]["resumed_from"] == 3
      and three[2]["argv"] == ["--epochs", "3"],
      "and so does a third launch, even one with no epoch left to train")

junk = '{"launches": [{"argv": '
with open(cfg_file, "w") as fh:
    fh.write(junk)
out, code = run("cfg", ["--epochs", "3"], RESUME=True)
aside = [n for n in os.listdir(os.path.join(D, "cfg"))
         if n.startswith("fno_ch_config_old")]
new_list = launches()
check(code is None and aside == ["fno_ch_config_old.json"]
      and text(path("cfg", aside[0])) == junk
      and len(new_list) == 1 and new_list[0]["resumed_from"] == 3
      and "[keep]" in out,
      "a config file that cannot be read is moved aside as it was, and a "
      "new list is started")

unit = os.path.join(D, "unit")
os.makedirs(unit)
unit_cfg = os.path.join(unit, "fno_ch_config.json")
sink = _io.StringIO()
with contextlib.redirect_stdout(sink):
    nothing = T.read_launches(unit_cfg)
check(nothing == [] and os.listdir(unit) == [],
      "with no config file there are no launches, and nothing is made")
good_list = [{"argv": ["a"], "resumed_from": None}]
with open(unit_cfg, "w") as fh:
    json.dump({"launches": good_list}, fh)
with contextlib.redirect_stdout(sink):
    got = T.read_launches(unit_cfg)
check(got == good_list and os.listdir(unit) == ["fno_ch_config.json"],
      "a readable list is returned and the file stays")
bodies = ['[1, 2]', '"text"', '{"argv": ["--epochs", "2"], "host": "h"}',
          '{"launches": {"argv": []}}', '{"launches": [1, 2]}', '']
kept_all = {}
for body in bodies:
    with open(unit_cfg, "w") as fh:
        fh.write(body)
    with contextlib.redirect_stdout(sink):
        got = T.read_launches(unit_cfg)
    now_files = {n: text(os.path.join(unit, n)) for n in os.listdir(unit)}
    moved = [n for n in now_files
             if n != "fno_ch_config.json" and n not in kept_all]
    kept_all.update({n: now_files[n] for n in moved})
    check(got == [] and len(moved) == 1 and kept_all[moved[0]] == body
          and "fno_ch_config.json" not in now_files,
          f"not a list of launches, so it is kept aside and a new list "
          f"starts: {body[:34]!r}")
check(len(kept_all) == len(bodies)
      and sum(n.startswith("fno_ch_config_old_") for n in kept_all)
      == len(bodies) - 1,
      "each one under a name of its own, never over an earlier one")

print("\n   a fresh start begins a new list")
out, code = run("cfg", ["--epochs", "2", "--no-resume"])
fresh = launches()
check(code is None and len(fresh) == 1 and fresh[0]["resumed_from"] is None
      and fresh[0]["argv"] == ["--epochs", "2", "--no-resume"]
      and len(launches(path("cfg", "fno_ch_prev_config.json"))) == 1,
      "and the old list goes aside with the old files")

print("\n5. a non-finite gradient skips the step")
flag = {"bad": False, "n": 0}
real_sample = T.WindowSampler.sample
real_build = T.build_model


def counted(self, *a, **k):
    flag["n"] += 1
    flag["bad"] = flag["n"] % 2 == 0
    return real_sample(self, *a, **k)


def hooked(*a, **k):
    m = real_build(*a, **k)
    for p in m.parameters():
        p.register_hook(lambda g: g * float("inf") if flag["bad"] else g)
    return m


with patched(T.WindowSampler, "sample", counted), \
        patched(T, "build_model", hooked):
    out, code = run("nan", STEPS_PER_EPOCH=4, SPIKE_ABORT_FRAC=0.9)
lines = epoch_lines(out)
nan_state = path("nan", "fno_ch_last.pt")
ck = load(nan_state) if os.path.exists(nan_state) else None
steps = ({int(s["step"]) for s in ck["optimizer"]["state"].values()}
         if ck else set())
check(code is None and len(lines) == 2
      and all("2 skipped (2 non-finite gradients)" in ln for ln in lines),
      f"half the batches had an infinite gradient and are counted in the "
      f"skipped number: {lines[0][-52:] if lines else out[-200:]}")
check(ck is not None and finite(ck["state_dict"]) and steps == {4},
      f"the weights stay finite and the optimiser stepped only on the good "
      f"batches ({sorted(steps)})")
check(ck is not None and ck["history"][0]["skipped"] == 2
      and ck["history"][0]["nonfinite_grad"] == 2,
      "and the history records them")
with patched(T.WindowSampler, "sample", counted), \
        patched(T, "build_model", hooked):
    out, code = run("nan_stall", STEPS_PER_EPOCH=4, SPIKE_ABORT_FRAC=0.1)
check("DIVERGED" in out and "batches skipped" in out,
      "the health check reads them: too many and the epoch is a stall")

print("\n5b. a run that stopped on its patience, or ran its EPOCHS, stays stopped")
scripted = []


def scripted_score(times, r2, t_end=T.VAL_T_END, threshold=T.R2_THRESHOLD):
    """The rollout score of each validation in turn (0.1 once the list is
    used up), so that which epoch improves is not left to a random network."""
    return (scripted.pop(0) if scripted else 0.1), 100.0


def stat(folder, name):
    st = os.stat(path(folder, name))
    return st.st_size, st.st_mtime_ns


def snapshot(folder):
    """{file name: ((sha, size), (size, mtime))} of what a folder holds."""
    return {n: (v, stat(folder, n)) for n, v in contents(folder).items()}


def untouched(folder, then):
    """Every file but the config (each start adds its launch there) as it
    was: same bytes, same size, same modification time, nothing new."""
    return ({n: v for n, v in snapshot(folder).items()
             if n != "fno_ch_config.json"}
            == {n: v for n, v in then.items() if n != "fno_ch_config.json"})


STAY = dict(EPOCHS=10, PATIENCE=2, PATIENCE_AFTER_RAMP=False, VAL_EVERY=1)
with patched(T, "score_from_curve", scripted_score):
    scripted[:] = [0.5]
    out, code = run("stay", **STAY)
    ck = load(path("stay", "fno_ch_last.pt"))
    check(code is None and len(epoch_lines(out)) == 3 and ck["epoch"] == 3
          and "[stop] rollout score" in out and ck.get("stale") == 2,
          f"stopped on patience after epoch 3, and the state keeps the "
          f"count: {ck.get('stale')}")
    then = snapshot("stay")

    scripted[:] = [0.9]
    out, code = run("stay", RESUME=True, **STAY)
    check(code is None and not epoch_lines(out) and scripted == [0.9],
          "resuming it trains no epoch")
    check("already stopped on patience (2 validations" in out
          and "nothing to train (raise PATIENCE to continue; PATIENCE is not "
          "part of the schedule stamp)" in out,
          "and says why and what to do")
    check(untouched("stay", then),
          "fno_ch.pt, the state and the history keep their bytes and times, "
          "and no file is rotated or added")
    with open(path("stay", "fno_ch_history.json")) as fh:
        check(len(json.load(fh)) == 3, "no row is added to the history")

    out, code = run("stay", RESUME=True, **{**STAY, "PATIENCE": 3})
    ck = load(path("stay", "fno_ch_last.pt"))
    check(code is None and len(epoch_lines(out)) == 4 and ck["epoch"] == 7
          and "nothing to train" not in out,
          "with PATIENCE raised it trains again, from the count it had")
    check(stat("stay", "fno_ch.pt") != then["fno_ch.pt"][1]
          and load(path("stay", "fno_ch.pt"))["best"]["score"] == 0.9,
          "and the better model it found is saved")

    then = snapshot("stay")
    out, code = run("stay", RESUME=True, **{**STAY, "PATIENCE": 3})
    check(code is None and not epoch_lines(out) and untouched("stay", then),
          "stopped again on the new PATIENCE, it stays stopped")

    print("\n   a state from before the count was saved counts from zero")
    old = load(path("stay", "fno_ch_last.pt"))
    old.pop("stale")
    torch.save(old, path("stay", "fno_ch_last.pt"))
    then = snapshot("stay")
    out, code = run("stay", RESUME=True, **{**STAY, "PATIENCE": 3})
    check(code is None and len(epoch_lines(out)) == 3
          and "nothing to train" not in out,
          "as every start did: PATIENCE more validations are trained")
    check(snapshot("stay")["fno_ch.pt"] == then["fno_ch.pt"],
          "and a worse score leaves fno_ch.pt alone")

    print("\n   a run that reached EPOCHS")
    out, code = run("done", EPOCHS=2, VAL_EVERY=1, PATIENCE=0)
    check(code is None and len(epoch_lines(out)) == 2, "ran its 2 epochs")
    scripted[:] = [0.9]
    then = snapshot("done")
    out, code = run("done", RESUME=True, EPOCHS=2, VAL_EVERY=1, PATIENCE=0)
    check(code is None and not epoch_lines(out) and scripted == [0.9]
          and "at epoch 2 of 2; nothing to train" in out,
          "resuming it trains nothing, and says so")
    check(untouched("done", then),
          "fno_ch.pt and the rest are as they were")

print("\n6. no CPU inside a SLURM job")
envs = {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "FNO_ALLOW_CPU")}


def env(**kv):
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


asked = []


def spy_data(*a, **k):
    asked.append(1)
    return tiny_data()


env(SLURM_JOB_ID="4242", FNO_ALLOW_CPU=None)
with patched(T, "load_trajectories", spy_data):
    out, code = run("guard")
    check(isinstance(code, str) and "4242" in code
          and "--allow-cpu" in code and "FNO_ALLOW_CPU=1" in code
          and not asked and not os.path.exists(os.path.join(D, "guard")),
          f"fno_train refuses, names the job and the way out, and reads and "
          f"writes nothing: {str(code)[:70]}")
    env(FNO_ALLOW_CPU="0")
    out, code = run("guard")
    check(isinstance(code, str) and "4242" in code,
          "FNO_ALLOW_CPU has to be 1")
    env(FNO_ALLOW_CPU=None)
    out, code = run("guard", ["--allow-cpu"])
    check(code is None and len(epoch_lines(out)) == 2,
          "--allow-cpu lets it run")
    shutil.rmtree(os.path.join(D, "guard"))
    env(FNO_ALLOW_CPU="1")
    out, code = run("guard")
    check(code is None and len(epoch_lines(out)) == 2,
          "so does FNO_ALLOW_CPU=1")
    shutil.rmtree(os.path.join(D, "guard"))
    env(SLURM_JOB_ID=None, FNO_ALLOW_CPU=None)
    out, code = run("guard")
    check(code is None and len(epoch_lines(out)) == 2,
          "and outside SLURM nothing changes")
env(SLURM_JOB_ID="4242")
check(T.require_gpu(torch.device("cuda")) is None,
      "a CUDA device is never refused, inside SLURM or not")

P.WRITE_LOG = False
P.MODEL_PATH = os.path.join(D, "none.pt")
before = sorted(os.listdir(D))


def predict(*argv):
    """What P.main does with these flags: the SystemExit message, or the
    FileNotFoundError that says it got as far as looking for the model."""
    try:
        with contextlib.redirect_stdout(_io.StringIO()):
            P.main(list(argv))
    except SystemExit as exc:
        return exc.code
    except FileNotFoundError as exc:
        return exc
    return None


got = predict()
check(isinstance(got, str) and "4242" in got and "--allow-cpu" in got,
      f"fno_predict refuses and names the job: {str(got)[:60]}")
check(isinstance(predict("--allow-cpu"), FileNotFoundError),
      "fno_predict --allow-cpu gets past the guard")
env(FNO_ALLOW_CPU="1")
check(isinstance(predict(), FileNotFoundError), "and so does FNO_ALLOW_CPU=1")
env(SLURM_JOB_ID=None, FNO_ALLOW_CPU=None)
check(isinstance(predict(), FileNotFoundError)
      and sorted(os.listdir(D)) == before,
      "outside SLURM it goes on as before, and none of this wrote a file")
env(**envs)

shutil.rmtree(D, ignore_errors=True)
print("\nCKPT PASS" if ok else "\nCKPT FAIL")
sys.exit(0 if ok else 1)
