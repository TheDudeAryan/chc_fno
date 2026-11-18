"""The wall-clock budget comes from the scheduler, or from nowhere."""
import contextlib
import io as _io
import os
import sys
import time

import numpy as np

from _paths import ROOT, SCRATCH

os.chdir(SCRATCH)
import fno_train as T
T.use_fixed_step()

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


def clear_slurm():
    for k in ("SLURM_JOB_ID", "SLURM_JOB_END_TIME", "SLURM_JOB_NODELIST",
              "SLURM_JOB_PARTITION"):
        os.environ.pop(k, None)


print("1. resolve_budget")
clear_slurm()
T.MAX_HOURS = "auto"
d, why = T.resolve_budget(time.time())
print("   |", why)
check(d is None, "off the scheduler there is no deadline")
check("not under a scheduler" in why, "and it says so")

os.environ["SLURM_JOB_ID"] = "123456"
os.environ["SLURM_JOB_END_TIME"] = str(time.time() + 7200)
d, why = T.resolve_budget(time.time())
print("   |", why)
check(d is not None and abs(d - (time.time() + 7200)) < 5,
      "SLURM_JOB_END_TIME is used verbatim")
check("1.99 h" in why or "2.00 h" in why, "reports the time remaining")

os.environ.pop("SLURM_JOB_END_TIME")
d, why = T.resolve_budget(time.time())
print("   |", why)
check(d is None, "no END_TIME and no scontrol falls back to no deadline")

clear_slurm()
T.MAX_HOURS = 3.0
t0 = time.time()
d, why = T.resolve_budget(t0)
print("   |", why)
check(d is not None and abs(d - (t0 + 10800)) < 1,
      "a number still means hours from launch")

T.MAX_HOURS = None
check(T.resolve_budget(time.time())[0] is None, "None means no limit")

T.MAX_HOURS = "forever"
try:
    T.resolve_budget(time.time())
    check(False, "a bad string is rejected")
except ValueError as exc:
    print("   |", exc)
    check("must be a number" in str(exc), "a bad string is rejected clearly")

L, NF = 32, 40
rng = np.random.default_rng(0)
traj = np.tanh(rng.standard_normal((5, NF, L, L)) * 0.5)
traj -= traj.mean(axis=(-2, -1), keepdims=True)
for i, off in enumerate((-0.4, -0.2, 0.0, 0.2, 0.4)):
    traj[i] += off


def run(tag, epochs=6):
    T.load_trajectories = lambda *a, **k: (traj[:4].astype(T.numpy_dtype()),
                                           traj[4:].astype(T.numpy_dtype()))
    T.EPOCHS, T.STEPS_PER_EPOCH, T.BATCH_SIZE = epochs, 4, 2
    T.MAX_UNROLL, T.BPTT, T.VAL_BATCHES = 2, 2, 2
    T.VAL_SEEDS, T.VAL_LATTICE = (11,), 32
    T.VAL_T_END, T.VAL_T_START = 105.0, 100.0
    T.MONITOR_SEEDS, T.MONITOR_OFFS = (21, 22), (-0.4, 0.4)
    T.MONITOR_LATTICE, T.MONITOR_T_END = 32, 105.0
    T.VAL_EVERY, T.PATIENCE = epochs, 0
    T.CHANNELS, T.MODES, T.LAYERS = 8, 6, 2
    T.RESUME, T.TRAIN_ON_VAL, T.WRITE_LOG = False, True, False
    T.GT_CACHE = "budget_gt"
    T.MODEL_OUT, T.CHECKPOINT = f"{tag}.pt", f"{tag}_last.pt"
    for f in (T.MODEL_OUT, T.CHECKPOINT):
        if os.path.exists(f):
            os.remove(f)
    buf = _io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buf):
        try:
            T.main()
        except SystemExit as exc:
            code = exc.code
    return buf.getvalue(), code


print("\n2. the loop honours a deadline that is close")
T.MAX_HOURS = "auto"
T.BUDGET_RESERVE_MIN = 10.0
# the fake job trains on the CPU on purpose: without this, require_gpu (which
# t_ckpt tests) stops a SLURM job that has no CUDA device before the loop runs
os.environ["FNO_ALLOW_CPU"] = "1"
os.environ["SLURM_JOB_ID"] = "999"
os.environ["SLURM_JOB_END_TIME"] = str(time.time() + 300)
out, code = run("bud_near")
epochs = [l for l in out.splitlines() if l.startswith("epoch ")]
line = [l for l in out.splitlines() if "[budget]" in l]
for l in line:
    print("   |", l.strip())
check(any("stopping after epoch" in l for l in line),
      "stops early when the reserve will not fit")
check(len(epochs) == 1, f"stopped after the first epoch, ran {len(epochs)}")
check("Resubmit to continue" in out, "tells you how to carry on")
check(code == 75, f"a budget stop exits with code 75, got {code!r}")
tail = out[out.rindex("CONFIGURATION"):]
check("exiting with code 75" in tail and "afterok" in tail,
      "and says so after the configuration dump (everything is saved first)")
check(os.path.exists("bud_near_last.pt"),
      "the resume state is on disk before the exit")

print("\n3. and runs to the end when there is time")
os.environ["SLURM_JOB_END_TIME"] = str(time.time() + 36000)
out, code = run("bud_far")
epochs = [l for l in out.splitlines() if l.startswith("epoch ")]
print("   |", [l for l in out.splitlines() if "[budget]" in l][0].strip())
check(len(epochs) == 6, f"all 6 epochs ran, got {len(epochs)}")
check("stopping after epoch" not in out, "no early stop")
check(code == 0 and "exiting with code" not in out,
      "finishing on EPOCHS exits 0")

print("\n4. no scheduler, no deadline")
clear_slurm()
out, code = run("bud_free")
epochs = [l for l in out.splitlines() if l.startswith("epoch ")]
print("   |", [l for l in out.splitlines() if "[budget]" in l][0].strip())
check(len(epochs) == 6, "runs all epochs on the cloud PC")
check("not under a scheduler" in out, "log records why there was no limit")
check(code == 0, "and exits 0")

import shutil
for f in os.listdir("."):
    if f.startswith("bud_") and f.endswith(".pt"):
        os.remove(f)
shutil.rmtree("budget_gt", ignore_errors=True)
clear_slurm()
print("\n" + ("BUDGET PASS" if ok else "BUDGET FAIL"))
sys.exit(0 if ok else 1)
