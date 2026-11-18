"""TRAIN_ON_VAL: does it fold the val trajectories in, and stay honest?"""
import contextlib
import io as _io
import os
import sys

import numpy as np
import torch

from _paths import ROOT, SCRATCH

os.chdir(SCRATCH)
import fno_train as T
T.use_fixed_step()

L, NF = 32, 40
rng = np.random.default_rng(0)
# 4 "train" + 1 "val" trajectory, distinguishable by composition
traj = np.tanh(rng.standard_normal((5, NF, L, L)) * 0.5).astype(T.numpy_dtype())
traj -= traj.mean(axis=(-2, -1), keepdims=True)
for i, off in enumerate((-0.4, -0.2, 0.0, 0.2, 0.4)):
    traj[i] += off
T.load_trajectories = lambda *a, **k: (traj[:4], traj[4:])

T.EPOCHS, T.STEPS_PER_EPOCH, T.BATCH_SIZE = 2, 4, 2
T.MAX_UNROLL, T.BPTT, T.VAL_BATCHES = 2, 2, 2
T.VAL_SEEDS, T.VAL_LATTICE, T.VAL_T_END, T.VAL_T_START = (11,), 32, 105.0, 100.0
T.MONITOR_SEEDS, T.MONITOR_OFFS = (21, 22), (-0.4, 0.4)
T.MONITOR_LATTICE, T.MONITOR_T_END = 32, 105.0
T.VAL_EVERY, T.PATIENCE = 2, 0
T.CHANNELS, T.MODES, T.LAYERS = 8, 6, 2
T.MAX_HOURS, T.RESUME = 1.0, False
T.GT_CACHE = "smoke_gt"

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


def run(tag, train_on_val):
    T.TRAIN_ON_VAL = train_on_val
    T.MODEL_OUT, T.CHECKPOINT = f"{tag}.pt", f"{tag}_last.pt"
    for f in (T.MODEL_OUT, T.CHECKPOINT):
        if os.path.exists(f):
            os.remove(f)
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        T.main()
    return buf.getvalue()


print("--- TRAIN_ON_VAL = True ---")
out = run("tv_on", True)
for line in out.splitlines():
    if "[data] TRAIN_ON_VAL" in line or "[val] one-step monitor" in line:
        print("   |", line.strip())
check("training on all 5 trajectories" in out, "folds val into train (4 -> 5)")
check("-0.4: 1" in out and "0.4: 1" in out,
      "composition balance reported, one per offset")
check("held-out l=32 trajectories" in out, "monitor uses its own trajectories")
check("independent of the 1 rollout seeds" in out,
      "monitor is independent of the rollout seeds")
check(not (set(T.MONITOR_SEEDS) & set(T.VAL_SEEDS)), "monitor seeds disjoint from VAL_SEEDS")
epochs = [l for l in out.splitlines() if l.startswith("epoch ")]
print("   |", epochs[0] if epochs else "no epoch line")
check(len(epochs) == T.EPOCHS and "DIVERGED" not in out,
      "trains cleanly to the end")
v1 = float(epochs[0].split("val1")[1].split("|")[0])
dp = float(epochs[0].split("d/psi")[1].split("|")[0])
check(np.isfinite(v1) and v1 > 0, f"val1 finite and non-zero ({v1:.5f})")
check(np.isfinite(dp) and dp > 0, f"d/psi finite and non-zero ({dp:.4f})")

print("\n--- TRAIN_ON_VAL = False (unchanged behaviour) ---")
out = run("tv_off", False)
for line in out.splitlines():
    if "[val] one-step monitor" in line:
        print("   |", line.strip())
check("TRAIN_ON_VAL" not in out.split("CONFIGURATION")[0],
      "no folding message when disabled")
check("test.npz val_X" in out, "monitor falls back to val_X")
epochs = [l for l in out.splitlines() if l.startswith("epoch ")]
print("   |", epochs[0] if epochs else "no epoch line")
check(len(epochs) == T.EPOCHS and "DIVERGED" not in out,
      "still trains cleanly")

print("\nTRAINVAL PASS" if ok else "\nTRAINVAL FAIL")
sys.exit(0 if ok else 1)
