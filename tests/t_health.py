"""The epoch-1 health check, tested against the code it replaces.

Each scenario runs the real training loop twice -- once on the pre-fix file
kept beside this test, once on the current one -- so the test shows the bug as
well as its absence.
"""
import contextlib
import importlib.util
import io as _io
import os
import shutil
import sys

import numpy as np

from _paths import ROOT, SCRATCH

HERE = SCRATCH
FIXTURES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures")
os.chdir(HERE)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


NEW = load("health_new", os.path.join(ROOT, "fno_train.py"))
NEW.use_fixed_step()   # the check was written for the fixed-step loop
OLD = load("health_old",
           os.path.join(FIXTURES, "fno_train_pre_health.py"))

L, NF = 32, 40
rng = np.random.default_rng(0)
traj = np.tanh(rng.standard_normal((5, NF, L, L)) * 0.5)
traj -= traj.mean(axis=(-2, -1), keepdims=True)
for i, off in enumerate((-0.4, -0.2, 0.0, 0.2, 0.4)):
    traj[i] += off

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


def run(T, tag, fresh=True, argv=None, **cfg):
    """One short training run, returning its stdout."""
    T.load_trajectories = lambda *a, **k: (traj[:4].astype(T.numpy_dtype()),
                                           traj[4:].astype(T.numpy_dtype()))
    T.EPOCHS, T.STEPS_PER_EPOCH, T.BATCH_SIZE = 3, 4, 2
    T.MAX_UNROLL, T.BPTT, T.VAL_BATCHES = 2, 2, 2
    T.VAL_SEEDS, T.VAL_LATTICE = (11,), 32
    T.VAL_T_END, T.VAL_T_START = 105.0, 100.0
    T.MONITOR_SEEDS, T.MONITOR_OFFS = (21, 22), (-0.4, 0.4)
    T.MONITOR_LATTICE, T.MONITOR_T_END = 32, 105.0
    T.VAL_EVERY, T.PATIENCE = 3, 0
    T.CHANNELS, T.MODES, T.LAYERS = 8, 6, 2
    T.MAX_HOURS, T.RESUME, T.TRAIN_ON_VAL = 1.0, False, True
    T.GT_CACHE = "health_gt"
    T.SPIKE_ABORT_FRAC, T.MAX_DELTA_REL = 0.10, 0.5
    T.DIVERGE_FACTOR, T.MAX_RECOVERIES = 3.0, 8
    for k, v in cfg.items():
        setattr(T, k, v)
    T.MODEL_OUT, T.CHECKPOINT = f"{tag}.pt", f"{tag}_last.pt"
    for f in (T.MODEL_OUT, T.CHECKPOINT):
        if fresh and os.path.exists(f):
            os.remove(f)
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        T.main(argv) if argv is not None else T.main()
    return buf.getvalue()


def epochs_of(out):
    return [l for l in out.splitlines() if l.startswith("epoch ")]


print("=" * 72)
print("1. a healthy run is untouched")
out = run(NEW, "h_ok")
done = [l for l in epochs_of(out) if "DIVERGED" not in l]
print("   |", done[0].strip() if done else "no epoch line")
check(len(done) == 3 and "DIVERGED" not in out,
      "3 clean epochs, nothing tripped")

print("\n2. epoch 1 unphysical (d/psi over the limit)")
new = run(NEW, "h_unphys", MAX_DELTA_REL=1e-12)
old = run(OLD, "h_unphys_old", MAX_DELTA_REL=1e-12)
first_new = epochs_of(new)[0].strip()
first_old = epochs_of(old)[0].strip()
print("   | new:", first_new[:96])
print("   | old:", first_old[:96])
check("DIVERGED" in first_new, "new: epoch 1 is caught")
check("DIVERGED" not in first_old, "old: epoch 1 escaped (the bug)")
check("no healthy epoch to roll back to yet" in new,
      "new: says there is nothing to roll back to")
check("rolled back" not in new, "new: does not roll back to a broken state")
check(len(epochs_of(new)) == 1, "new: stops instead of burning the wall clock")
check(len(epochs_of(old)) == 3, "old: ran on with a broken epoch 1")

print("\n3. epoch 1 stalled (too many batches skipped)")
new = run(NEW, "h_stall", SPIKE_ABORT_FRAC=-1.0)
old = run(OLD, "h_stall_old", SPIKE_ABORT_FRAC=-1.0)
check("DIVERGED" in epochs_of(new)[0], "new: epoch 1 is caught")
check("batches skipped" in new, "new: names the stall as the reason")
check("DIVERGED" not in epochs_of(old)[0], "old: epoch 1 escaped")

print("\n4. the relative criterion still works, and still waits for a median")
new = run(NEW, "h_rel", DIVERGE_FACTOR=1e-9)
lines = epochs_of(new)
print("   |", lines[0].strip()[:96])
print("   |", lines[1].strip()[:96] if len(lines) > 1 else "(no epoch 2)")
check("DIVERGED" not in lines[0],
      "epoch 1 passes: no median exists yet to compare against")
check(len(lines) > 1 and "DIVERGED" in lines[1],
      "epoch 2 trips once a median exists")
check("vs median" in new, "the reason names the median comparison")
check("rolled back to epoch 1" in new, "rolls back to the healthy epoch 1")

print("\n4b. a resumed run's first epoch trips the guard")
run(NEW, "h_res", EPOCHS=1)
check(os.path.exists("h_res_last.pt"), "a 1-epoch run leaves a resume state")
res = run(NEW, "h_res", fresh=False, argv=["--accept-schedule"], EPOCHS=3,
          RESUME=True, MAX_DELTA_REL=1e-12)
lines = epochs_of(res)
print("   |", lines[0].strip()[:96] if lines else "no epoch line")
check("[resume]" in res and "at epoch 1" in res, "the second run resumed")
check(bool(lines) and "DIVERGED" in lines[0]
      and "rolled back to epoch 1" in lines[0],
      "epoch 2 (the first resumed) rolls back to the resumed epoch 1")
check("no healthy epoch to roll back to yet" not in res
      and "stopping." not in lines[0],
      "it does not stop for want of a healthy epoch")
fresh_run = run(NEW, "h_res_fresh", MAX_DELTA_REL=1e-12)
check("no healthy epoch to roll back to yet" in fresh_run,
      "a fresh run still has no healthy state before its first epoch")

print("\n5. diverge_reason never formats a missing median")
for flags in ((True, False, False), (False, True, False),
              (False, False, True)):
    blown, unphys, stall = flags
    msg = NEW.diverge_reason(float("nan") if blown else 0.5, 9.9, 7, None,
                             blown, stall, unphys)
    print("   |", msg)
    check("None" not in msg and msg, f"readable with median=None {flags}")

for f in os.listdir(HERE):
    if f.startswith("h_") and f.endswith(".pt"):
        os.remove(os.path.join(HERE, f))
shutil.rmtree(os.path.join(HERE, "health_gt"), ignore_errors=True)

print("\n" + ("HEALTH PASS" if ok else "HEALTH FAIL"))
sys.exit(0 if ok else 1)
