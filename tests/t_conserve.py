"""figure_conservation, end to end on the real checkpoint at reduced scale.

A case keeps no per-seed rows: its summary carries the worst conservation
numbers over every seed (max_mean_error, max_mass_drift), mean_r2.csv the
seed-averaged drift, and only the best seed has a folder of its own.
"""
import json
import os
import shutil
import sys

import numpy as np

from _paths import fixed_step_model, SCRATCH

# a fixed-step network: this suite runs the seed engine, and the project's own
# checkpoints are dt-conditioned (see fixed_step_model in _paths.py)
REAL_MODEL = fixed_step_model()

os.chdir(SCRATCH)
import torch

import fno_predict as P
import fno_train as T
T.use_fixed_step()
import make_submission as S

OUT = os.path.abspath("cons_out")
for d in (OUT, OUT + "_legacy", "cons_cache", "cons_rt"):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
P.WRITE_LOG = False
P.RESULTS_DIR = OUT
P.RESULT_CACHE = os.path.abspath("cons_cache")
P.GT_CACHE = T.GT_CACHE = os.path.abspath("cons_gt")
P.MODEL_PATH = REAL_MODEL
P.LATTICE, P.END_TIME = 64, 200.0
P.SNAPSHOT_TIMES = (100.0, 150.0, 200.0)
P.EVAL_SEEDS = P.SWEEP_SEEDS = (99999, 54321)
P.DIAG_EVERY = 5

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, ckpt = P.load_checkpoint(P.MODEL_PATH, device=device)
P.MODEL_FLAGS.update(hard_constraints=True, hard_mass_conservation=True,
                     zero_dc=True, label="cons", note="")
print(f"model: {model.n_parameters():,} params, "
      f"epoch {ckpt.get('best', {}).get('epoch')}\n")

# The rollouts as write_case_outputs receives them.  The case keeps no
# per-seed rows, so the per-seed numbers are worked out here from these.
captured = []
real_wco = P.write_case_outputs


def capture_wco(results, seeds, *a, **kw):
    captured.append((list(results), tuple(seeds)))
    return real_wco(results, seeds, *a, **kw)


P.write_case_outputs = capture_wco
try:
    res = P.run_case(model, device, 0.0, lattice=64, seeds=P.EVAL_SEEDS)
finally:
    P.write_case_outputs = real_wco
check(len(captured) == 1 and len(captured[0][0]) == 2,
      "run_case handed both seeds' rollouts to write_case_outputs")
if not (len(captured) == 1 and len(captured[0][0]) == 2):
    print("\nCONSERVE FAIL")
    sys.exit(1)
results, seeds = captured[0]
folder = P.case_dir(0.0, 64)
st = res["stats"]

print("\n1. the case's conservation figure")
png = os.path.join(folder, "cons_l64_conservation.png")
check(os.path.exists(png), "conservation figure written")
if os.path.exists(png):
    check(os.path.getsize(png) > 20000,
          f"figure is non-trivial ({os.path.getsize(png) / 1000:.0f} kB)")

print("\n2. the worst error over the seeds, from the case summary")
sp = os.path.join(folder, "cons_l64_summary.json")
check(os.path.exists(sp), "case summary written into the case folder")
disk = json.load(open(sp)) if os.path.exists(sp) else {}
check("seeds" not in st and "seeds" not in disk,
      "the summary keeps no per-seed list")
check("max_mean_error" in st and "max_mass_drift" in st,
      "max_mean_error and max_mass_drift recorded in the summary")
check(disk.get("max_mean_error") == st.get("max_mean_error")
      and disk.get("max_mass_drift") == st.get("max_mass_drift"),
      "the summary on disk carries the same maxima as the returned case")

# straight from the arrays, not through seed_summary
with np.errstate(invalid="ignore"):
    errs = [float(np.nanmax(np.abs(np.asarray(r["mean_pred"])
                                   - np.asarray(r["mean_true"]))))
            for r in results]
drifts = [float(np.max(r["mass_drift"])) for r in results]
print("\n  max |psi_bar_FNO - psi_bar_truth| per seed:")
for sd, e, dr in zip(seeds, errs, drifts):
    print(f"    seed {sd}: {e:.3e}   (max_mass_drift {dr:.3e})")
print(f"  summary: max_mean_error {st.get('max_mean_error')}, "
      f"max_mass_drift {st.get('max_mass_drift')}")
check(st.get("max_mean_error") == max(errs),
      "max_mean_error is the max over the seeds' own mean errors")
check(st.get("max_mass_drift") == max(drifts),
      "max_mass_drift is the max over the seeds' own mass drifts")
rows = [P.seed_summary(r, sd) for r, sd in zip(results, seeds)]
check([row["max_mean_error"] for row in rows] == errs
      and [row["max_mass_drift"] for row in rows] == drifts,
      "seed_summary works out the same per-seed numbers")

eps = float(np.finfo(np.float64 if P.PRECISION == "fp64"
                     else np.float32).eps)
tol = 4.0 * eps
worst = st.get("max_mean_error")
worst = float("inf") if worst is None else float(worst)
print(f"  working precision {P.PRECISION}, eps {eps:.2e}, "
      f"tolerance {tol:.2e}, worst {worst:.3e}")
check(worst < tol,
      f"conservation holds to the precision it runs at ({P.PRECISION})")

print("\n3. the seed-averaged drift in mean_r2.csv")
mp = os.path.join(folder, P.MEAN_R2_FILE)
check(os.path.exists(mp), f"{P.MEAN_R2_FILE} written into the case folder")
if os.path.exists(mp):
    head = open(mp).readline().strip()
    check(head == "t,n_seeds,r2_mean,r2_std,mass_drift_mean",
          f"columns are t,n_seeds,r2_mean,r2_std,mass_drift_mean ({head})")
    data = np.loadtxt(mp, delimiter=",", skiprows=1, ndmin=2)
    stack = np.stack([np.asarray(r["mass_drift"], dtype=np.float64)
                      for r in results])
    same_len = data.shape[0] == stack.shape[1]
    check(same_len and bool(np.all(data[:, 1] == len(results))),
          "one line per step, each averaging both seeds")
    # written as %.18e, which gives a float64 back to the last bit
    check(same_len and np.array_equal(data[:, 4], stack.mean(axis=0)),
          "mass_drift_mean is the average of the seeds' drifts, bit for bit")
    check(float(np.max(data[:, 4])) <= float(st.get("max_mass_drift", -1.0)),
          "the averaged drift never exceeds the summary's max_mass_drift")
    r2s = np.stack([np.asarray(r["r2"], dtype=np.float64) for r in results])
    check(same_len and np.array_equal(data[:, 0], results[0]["times"])
          and np.array_equal(data[:, 2], np.nanmean(r2s, axis=0))
          and np.array_equal(data[:, 3], np.nanstd(r2s, axis=0)),
          "t, r2_mean and r2_std are the steps and the seeds' R2 mean and "
          "spread, bit for bit")

print("\n4. the best seed's own folder")
best = st.get("best_seed")
bi = int(S.best_index(results))
check(best == seeds[bi],
      f"best seed {best} is the one make_submission.best_index picks")
bdir = os.path.join(folder, P.BEST_SEED_DIR.format(seed=best))
bpng = os.path.join(bdir, f"cons_l64_seed{best}_conservation.png")
check(os.path.exists(bpng), "the best seed's own conservation figure written")
bsum = os.path.join(bdir, f"cons_l64_seed{best}_summary.json")
bjs = json.load(open(bsum)) if os.path.exists(bsum) else {}
check(bjs.get("max_mean_error") == errs[bi]
      and bjs.get("max_mass_drift") == drifts[bi],
      "its summary carries that seed's own mean error and drift")
others = [d for d in os.listdir(folder)
          if os.path.isdir(os.path.join(folder, d))
          and d != os.path.basename(bdir) and "seed" in d]
check(not others, f"no other seed has a folder ({others})")

print("\n5. the cache round-trip")
# run_case caches the rollouts and leaves them; only evaluate_all (main)
# deletes a case's cache, once its outputs are written
files = P.case_cache_files(0.0, 64, seeds)
check(len(files) == 2 and all(os.path.exists(f) for f in files),
      "run_case on its own leaves the case's cached rollouts in place")
rt = os.path.join(os.path.abspath("cons_rt"), "roundtrip.npz")
P._save_result(rt, results[0])
back = P._load_result(rt)
b_true, b_pred = back["mean_true"], back["mean_pred"]
check(b_true is not None and b_pred is not None,
      "means survive the cache round-trip")
check(b_true is not None and b_pred is not None
      and np.array_equal(b_true, results[0]["mean_true"])
      and np.array_equal(b_pred, results[0]["mean_pred"])
      and np.array_equal(back["mass_drift"], results[0]["mass_drift"]),
      "bit for bit: mean_true, mean_pred and mass_drift")
if all(os.path.exists(f) for f in files):
    c_pred = P._load_result(files[0])["mean_pred"]
    check(c_pred is not None
          and np.array_equal(c_pred, results[0]["mean_pred"]),
          "the cache run_case wrote holds the same means")

# a finished case's cache goes with its set-aside copies, but never another
# key's rollouts in the same folder
other = os.path.join(P.RESULT_CACHE, "cons_l64_seed99999_000000000000.npz")
if os.path.exists(files[0]):
    shutil.copyfile(files[0], other)
    shutil.copyfile(files[0], files[0] + ".corrupt")
n = P.drop_case_cache(0.0, 64, seeds)
check(n == 3 and not any(os.path.exists(f) for f in files)
      and not os.path.exists(files[0] + ".corrupt"),
      f"drop_case_cache deletes the case's rollouts and .corrupt copy ({n})")
check(os.path.exists(other), "a rollout under another cache key is left")

print("\n6. a cache written before the means existed")
legacy = dict(back)
legacy["mean_true"] = legacy["mean_pred"] = None
P.figure_conservation([legacy], [1], os.path.join(OUT, "legacy.png"), 0.0, 64)
check(not os.path.exists(os.path.join(OUT, "legacy.png")),
      "legacy cache skips the figure instead of crashing")
check(P.seed_summary(legacy, 1)["max_mean_error"] is None,
      "its seed gets no mean error rather than a made-up one")
# a case mixing it with a current seed: the max over the seeds that have one
P.RESULTS_DIR, P.WRITE_SUBMISSION = OUT + "_legacy", False
try:
    mixed = P.write_case_outputs([legacy, results[1]], seeds, 0.0, 64)
    alone = P.write_case_outputs([legacy], seeds[:1], 0.0, 64)
    lsum = os.path.join(P.case_dir(0.0, 64), "cons_l64_summary.json")
    ljs = json.load(open(lsum)) if os.path.exists(lsum) else {}
finally:
    P.RESULTS_DIR, P.WRITE_SUBMISSION = OUT, True
mst = mixed["stats"]
check(mst.get("max_mean_error") == errs[1],
      "a case mixing it in reports the max over the seeds that have means")
check(mst.get("max_mass_drift") == max(drifts),
      "and max_mass_drift still over every seed")
check(alone["stats"].get("max_mean_error", 0.0) is None
      and "max_mean_error" in ljs and ljs["max_mean_error"] is None,
      "a case with no seed that has means reports none, not a zero")
check(alone["stats"].get("max_mass_drift") == drifts[0],
      "while its max_mass_drift is still that seed's")

print("\nCONSERVE PASS" if ok else "\nCONSERVE FAIL")
sys.exit(0 if ok else 1)
