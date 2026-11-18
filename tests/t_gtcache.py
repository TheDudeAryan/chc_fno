"""GT_CACHE off must solve correctly and leave nothing on disk."""
import os
import shutil
import sys

import numpy as np

from _paths import ROOT, SCRATCH

os.chdir(SCRATCH)
import fno_predict as P
import fno_train as T
T.use_fixed_step()

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


CACHE = os.path.join(SCRATCH, "gt_probe")
shutil.rmtree(CACHE, ignore_errors=True)

print("1. with a cache directory")
t1, f1 = T.ch_ground_truth(32, 4242, 0.0, 3.0, T.DT_STEP, cache_dir=CACHE,
                           device="cpu", verbose=False)
files = os.listdir(CACHE)
check(len(files) == 1, f"one reference written: {files}")
size = os.path.getsize(os.path.join(CACHE, files[0]))
check(size > 1000, f"and it is real ({size/1000:.0f} kB)")

t1b, f1b = T.ch_ground_truth(32, 4242, 0.0, 3.0, T.DT_STEP, cache_dir=CACHE,
                             device="cpu", verbose=False)
check(np.array_equal(f1, f1b), "a second call hits the cache identically")

print("\n2. with the cache off")
before = set(os.listdir(SCRATCH))
for empty in ("", None):
    t2, f2 = T.ch_ground_truth(32, 4242, 0.0, 3.0, T.DT_STEP, cache_dir=empty,
                               device="cpu", verbose=False)
    check(np.array_equal(f1, f2),
          f"cache_dir={empty!r} gives bit-identical frames")
    check(np.array_equal(t1, t2), f"cache_dir={empty!r} gives the same times")
after = set(os.listdir(SCRATCH))
check(after == before, f"nothing new on disk: {sorted(after - before)}")
check(not os.path.exists("None") and not os.path.exists(""),
      "no stray directory named after the empty cache")

print("\n3. the CLI flag")
saved = P.GT_CACHE
P.GT_CACHE = "gt_cache"
import contextlib
import io as _io
with contextlib.redirect_stdout(_io.StringIO()) as log:
    P.apply_overrides(P.parse_args(["--no-gt-cache"]))
check(P.GT_CACHE == "", f"--no-gt-cache empties GT_CACHE, got {P.GT_CACHE!r}")
check("references not kept" in log.getvalue(), "and says so in the log")
P.GT_CACHE = saved

print("\n4. the free-space guard of prewarm_references")
import fno_gt
import fno_paths

GUARD = os.path.join(SCRATCH, "guard_cache")
shutil.rmtree(GUARD, ignore_errors=True)
keep = {n: getattr(P, n) for n in ("REF_CACHE", "START_TIME", "END_TIME",
                                    "HOME_CACHE_LIMIT", "grid_cases", "TUNE",
                                    "solve_references", "ref_cache")}
keep_free = fno_paths.free_bytes
keep_env = {n: os.environ.get(n) for n in ("SLURM_JOB_ID", "FNO_GT_CACHE")}
solved_calls = []
P.REF_CACHE = None
P.ref_cache = lambda: GUARD
P.START_TIME, P.END_TIME = 100.0, 200.0
P.TUNE = False
P.grid_cases = lambda: [(0.0, 32, [1, 2, 3])]
P.solve_references = lambda l, off, seeds, dev: (solved_calls.append(
    (l, off, tuple(seeds))) or len(seeds))
one = fno_gt.gt_file_bytes(32, 100.0, 200.0, fno_gt.GT_EVERY)
n_fr = len(fno_gt.gt_times(100.0, 200.0, fno_gt.GT_EVERY))
check(one == n_fr * 32 * 32 * 4 + n_fr * 8 + 2048,
      f"a file is {one} bytes for {n_fr} frames")


def guard(free, slurm=False, env=None, limit=None):
    for n in ("SLURM_JOB_ID", "FNO_GT_CACHE"):
        os.environ.pop(n, None)
    if slurm:
        os.environ["SLURM_JOB_ID"] = "1"
    if env:
        os.environ["FNO_GT_CACHE"] = env
    fno_paths.free_bytes = lambda path: free
    P.HOME_CACHE_LIMIT = keep["HOME_CACHE_LIMIT"] if limit is None else limit
    solved_calls.clear()
    buf = _io.StringIO()
    code = None
    with contextlib.redirect_stdout(buf):
        try:
            P.prewarm_references(None, skip_done=False)
        except SystemExit as exc:
            code = str(exc.code)
    return code, buf.getvalue()


try:
    code, out = guard(free=int(2e9 + 3 * one - 1))
    check(code is not None and "need" in code and not solved_calls
          and not os.path.exists(GUARD),
          "too little free space: refuses, solves nothing, writes nothing")
    check(f"[gt] need {3 * one / 1e9:.1f} GB for 3 files;" in out
          and f"free in {GUARD}" in out, "and prints the need line")
    code, out = guard(free=int(2e9 + 3 * one + 1))
    check(code is None and solved_calls == [(32, 0.0, (1, 2, 3))],
          "enough free space: proceeds as before")
    check("WARNING" not in out, "outside Slurm: no warning")
    code, out = guard(free=int(1e15), slurm=True)
    check(code is None and "WARNING: FNO_GT_CACHE is not set" in out
          and "200 GB" in out and "/scratch/data/<user>/fno_gt_cache" in out
          and solved_calls, "in Slurm without FNO_GT_CACHE: warns, small need runs")
    code, out = guard(free=int(1e15), slurm=True, limit=2.0 * one)
    check(code is not None and "limit" in code and not solved_calls,
          "in Slurm, a need above HOME_CACHE_LIMIT: refuses")
    code, out = guard(free=int(1e15), slurm=True, env=GUARD, limit=2.0 * one)
    check(code is None and "WARNING" not in out and solved_calls,
          "with FNO_GT_CACHE set: no warning and no home limit")
finally:
    fno_paths.free_bytes = keep_free
    for n, v in keep.items():
        setattr(P, n, v)
    for n, v in keep_env.items():
        if v is None:
            os.environ.pop(n, None)
        else:
            os.environ[n] = v
    shutil.rmtree(GUARD, ignore_errors=True)

shutil.rmtree(CACHE, ignore_errors=True)
print("\n" + ("GTCACHE PASS" if ok else "GTCACHE FAIL"))
sys.exit(0 if ok else 1)
