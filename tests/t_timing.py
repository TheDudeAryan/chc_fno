"""Timing report: real solve times, honest cache labelling, sane speedups.

The report has one line per case, built from the sums each case summary
keeps (timing_totals) -- no seed has a row of its own.  Its numbers must be
the ones the old per-seed rows gave once averaged, so they are checked here
against those, worked out from seed_summary rows of the rollouts in memory.
"""
import json
import os
import re
import shutil
import sys

import numpy as np
import torch

from _paths import fixed_step_model, SCRATCH

# a fixed-step network: this suite runs the seed engine, and the project's own
# checkpoints are dt-conditioned (see fixed_step_model in _paths.py)
REAL_MODEL = fixed_step_model()

os.chdir(SCRATCH)
import fno_paths
import fno_predict as P
# records and meta.json (t_records.py) stay in the cache's folder, which
# the checks below expect to be removed once it is empty
P.RECORDS = False
import fno_train as T
T.use_fixed_step()

OUT = os.path.abspath("time_out")
for d in (OUT, "time_cache", "time_gt"):
    shutil.rmtree(d, ignore_errors=True)      # force a genuine solve
os.makedirs(OUT, exist_ok=True)
P.WRITE_LOG = False
P.RESULTS_DIR = OUT
P.RESULT_CACHE = os.path.abspath("time_cache")
P.GT_CACHE = T.GT_CACHE = os.path.abspath("time_gt")
P.MODEL_PATH = REAL_MODEL
P.LATTICE, P.END_TIME = 64, 200.0
P.SNAPSHOT_TIMES = (100.0, 200.0)
P.EVAL_SEEDS = P.SWEEP_SEEDS = (99999, 54321)
P.DIAG_EVERY = 20

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, ckpt = P.load_checkpoint(P.MODEL_PATH, device=device)
P.MODEL_FLAGS.update(hard_constraints=True, hard_mass_conservation=True,
                     zero_dc=True, label="tm", note="")

WINDOW = (P.END_TIME - P.START_TIME) / P.END_TIME
CASE_KEYS = {"off", "lattice", "n_rollouts", "n_solved",
             "gt_euler_steps_same_window", "gt_seconds_same_window_mean",
             "fno_seconds_mean", "fno_ms_per_step_mean", "speedup_mean"}

# the rollouts as write_case_outputs receives them, for the per-seed rows
captured = []
real_wco = P.write_case_outputs


def capture_wco(results, seeds, *a, **kw):
    captured.append((list(results), tuple(seeds)))
    return real_wco(results, seeds, *a, **kw)


def rolled_out(off, lattice, seeds):
    """run_case, and the seed_summary rows of the rollouts it wrote."""
    mark = len(captured)
    P.write_case_outputs = capture_wco
    try:
        case = P.run_case(model, device, off, lattice=lattice, seeds=seeds)
    finally:
        P.write_case_outputs = real_wco
    got = captured[mark:]
    rows = ([P.seed_summary(r, s) for r, s in zip(*got[0])]
            if len(got) == 1 else [])
    return case, rows


def per_seed_means(rows):
    """A case's numbers as the report gave them when it kept one row per
    seed: each seed's solver time scaled to the window, its ms/step and its
    speedup, then the means (over the solved seeds for the solver)."""
    per = []
    for r in rows:
        gw, fno = WINDOW * r["gt_seconds"], r["fno_seconds"]
        per.append({"gt": gw, "fno": fno, "cached": bool(r["gt_cached"]),
                    "ms": 1e3 * fno / r["fno_steps"] if r["fno_steps"] else None,
                    "speedup": gw / fno if fno > 0 else None})
    fresh = [p for p in per if not p["cached"]]

    def mean(v):
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None

    return {"n_rollouts": len(per), "n_solved": len(fresh),
            "gt_seconds_same_window_mean": mean([p["gt"] for p in fresh]),
            "fno_seconds_mean": mean([p["fno"] for p in per]),
            "fno_ms_per_step_mean": mean([p["ms"] for p in per if p["ms"]]),
            "speedup_mean": mean([p["speedup"] for p in fresh]),
            "fno_seconds_solved_mean": mean([p["fno"] for p in fresh])}


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return bool(np.isclose(a, b, rtol=1e-12, atol=0.0))


def agrees(row, want):
    return all(same(row.get(k), want[k]) for k in
               ("n_rollouts", "n_solved", "gt_seconds_same_window_mean",
                "fno_seconds_mean", "fno_ms_per_step_mean", "speedup_mean"))


def shown(text, value, digits):
    """A number printed to ``digits`` decimals is ``value``."""
    try:
        return abs(float(text) - value) <= 0.5 * 10.0 ** -digits + 1e-9
    except (TypeError, ValueError):
        return False


def table(txt):
    return [ln.split() for ln in txt.splitlines()
            if ln.strip().startswith("psi0=")]


OVERALL = re.compile(r"over the (\d+) rollouts whose reference was actually "
                     r"solved: solver ([\d.]+) s, FNO ([\d.]+) s, "
                     r"speedup ([\d.]+)x")

print("=== pass 1: ground truth solved fresh ===")
case, rows1 = rolled_out(0.0, 64, P.EVAL_SEEDS)
check(len(rows1) == 2, "both seeds' rollouts seen by write_case_outputs")
jp = os.path.join(OUT, "tm_timing.json")
tp = os.path.join(OUT, "tm_timing.txt")
P.write_timing_report([case], device, model, jp, tp)

check(os.path.exists(jp) and os.path.exists(tp), "timing files written")
rep = json.load(open(jp))
check("cases" in rep and "rows" not in rep,
      "the report is per case, with no per-seed rows")
check(len(rep.get("cases", ())) == 1
      and all(set(r) == CASE_KEYS for r in rep.get("cases", ())),
      "one line for the one case, with no seed field")
check(rep["meta"]["n_ground_truths_solved"] == 2,
      f"both references counted as solved ({rep['meta']['n_ground_truths_solved']})")
check(rep["meta"]["n_rollouts"] == 2 and rep["meta"]["device"] == device
      and rep["meta"]["model_parameters"] == model.n_parameters(),
      "meta: two rollouts, this device, this model")
check(rep["meta"]["euler_steps_per_model_step"] == 100,
      "one model step = 100 Euler steps")
row = (rep.get("cases") or [{}])[0]
for r in rows1:
    print(f"    seed {r['seed']}: solver {r['gt_seconds']:.2f}s "
          f"(window {WINDOW * r['gt_seconds']:.2f}s), FNO "
          f"{r['fno_seconds']:.2f}s over {r['fno_steps']} steps")
print(f"    case: {row}")
check(row.get("off") == 0.0 and row.get("lattice") == 64,
      "the line names the case (psi0 = 0, l = 64)")
check(len(rows1) == 2 and all(r["fno_steps"] == 100 for r in rows1),
      "FNO step count matches the rollout length (t=100..200)")
check(row.get("gt_euler_steps_same_window") == 10000,
      "solver steps over the same window: 100 of real time at dt = 0.01")
check(len(rows1) == 2 and all(not r["gt_cached"] for r in rows1)
      and row.get("n_solved") == 2 and row.get("n_rollouts") == 2,
      "both seeds solved, and the case line counts them as solved")
want1 = per_seed_means(rows1)
check(agrees(row, want1),
      "json: solver s, FNO s, ms/step and speedup equal the means of the "
      "per-seed values")
sp = row.get("speedup_mean")
check(sp is not None and np.isfinite(sp) and sp > 0,
      "speedup recorded and finite (its value is a hardware fact, not a "
      "requirement)")
ms = row.get("fno_ms_per_step_mean")
check(ms is not None and 0 < ms < 1000, "ms/step is plausible")

sp1 = os.path.join(P.case_dir(0.0, 64), "tm_l64_summary.json")
disk1 = json.load(open(sp1)) if os.path.exists(sp1) else {}
check("seeds" not in disk1 and disk1.get("timing") == P.timing_totals(rows1),
      "the case summary keeps the timing as sums over its seeds, no rows")
tot = disk1.get("timing") or {}
check(tot.get("n") == 2 and tot.get("n_solved") == 2 and tot.get("n_ms") == 2
      and tot.get("n_ratio") == 2, f"sums over both seeds ({tot})")

txt1 = open(tp).read()
lines = table(txt1)
check(len(lines) == 1, f"the text table has one line, the case's ({len(lines)})")
ln = lines[0] if lines else []
check(len(ln) == 7 and ln[:3] == ["psi0=+0.0", "l=64", "2"]
      and shown(ln[3], want1["gt_seconds_same_window_mean"], 1)
      and shown(ln[4], want1["fno_seconds_mean"], 1)
      and shown(ln[5], want1["fno_ms_per_step_mean"], 2)
      and shown(ln[6].rstrip("x"), want1["speedup_mean"], 1),
      f"txt: the case line shows the per-seed means ({' '.join(ln)})")
m = OVERALL.search(txt1)
gt_f, fno_f = (want1["gt_seconds_same_window_mean"],
               want1["fno_seconds_solved_mean"])
check(m is not None and m.group(1) == "2" and gt_f is not None
      and fno_f is not None and shown(m.group(2), gt_f, 1)
      and shown(m.group(3), fno_f, 1) and shown(m.group(4), gt_f / fno_f, 1),
      "txt: the overall line is over the 2 solved rollouts, from their means")
check("ground truths solved      2 of 2" in txt1,
      "txt: ground truths solved 2 of 2")

print("\n=== pass 1b: the same case, from no model (a regeneration) ===")
jn, tn = os.path.join(OUT, "none_timing.json"), os.path.join(OUT,
                                                             "none_timing.txt")
P.write_timing_report([case], None, None, jn, tn)
repn = json.load(open(jn)) if os.path.exists(jn) else {"meta": {}}
check(repn["meta"].get("device") == "unknown"
      and repn["meta"].get("gpu") is None
      and repn["meta"].get("model_parameters") is None
      and repn["meta"].get("model_dtype") is None,
      "no model or device: reported as unknown, not a crash")
check(repn.get("cases") == rep["cases"], "and the case line is unchanged")
txtn = open(tn).read() if os.path.exists(tn) else ""
check("model parameters          unknown" in txtn
      and "device                    unknown" in txtn,
      "txt says the device and the model are unknown")
check(repn["meta"].get("precision") is None
      and "precision                 unknown" in txtn
      and "NOTE: the model is evaluated in" not in txtn,
      "and the precision too: this file's setting says nothing about how "
      "those rollouts were run")

print("\n=== pass 2: same run, everything now cached ===")
shutil.rmtree(P.RESULT_CACHE, ignore_errors=True)   # force re-evaluation,
case2, rows2 = rolled_out(0.0, 64, P.EVAL_SEEDS)    # but the GT is cached now
# run_case rolls out whatever it is given: only evaluate_all (main) takes a
# case an earlier job finished from its summary instead
check(not case2.get("resumed") and len(rows2) == 2,
      "run_case rolled the finished case out again into the same folder")
disk2 = json.load(open(sp1)) if os.path.exists(sp1) else {}
check(disk2.get("timing") == P.timing_totals(rows2)
      and (disk2.get("record") or {}).get("complete") is True,
      "and rewrote its summary, with this rollout's timing sums")
jp2 = os.path.join(OUT, "tm2_timing.json")
tp2 = os.path.join(OUT, "tm2_timing.txt")
P.write_timing_report([case2], device, model, jp2, tp2)
rep2 = json.load(open(jp2))
check(rep2["meta"]["n_ground_truths_from_cache"] == 2
      and rep2["meta"]["n_ground_truths_solved"] == 0,
      "cache hits identified, not passed off as solves")
row2 = (rep2.get("cases") or [{}])[0]
check(len(rows2) == 2 and all(r["gt_cached"] for r in rows2)
      and row2.get("n_solved") == 0 and row2.get("n_rollouts") == 2,
      "both seeds read from cache, and the case line counts no solve")
check(row2.get("gt_seconds_same_window_mean") is None
      and row2.get("speedup_mean") is None,
      "no solver time and no speedup quoted from cache reads")
want2 = per_seed_means(rows2)
check(agrees(row2, want2), "json: FNO s and ms/step still the per-seed means")
gt_fresh = np.mean([r["gt_seconds"] for r in rows1])
gt_cache = np.mean([r["gt_seconds"] for r in rows2])
print(f"    solve {gt_fresh:.2f}s vs cache read {gt_cache:.3f}s "
      f"({gt_fresh / max(gt_cache, 1e-9):.0f}x) -- why cached rows must be "
      f"excluded from the speedup")
check(gt_cache < gt_fresh, "cache read is faster than solving")
txt = open(tp2).read()
check("every reference came from cache" in txt,
      "text report refuses to quote a speedup from cache reads")
lines2 = table(txt)
ln2 = lines2[0] if lines2 else []
check(len(lines2) == 1 and len(ln2) == 7 and ln2[3] == "cached"
      and ln2[6] == "--" and shown(ln2[4], want2["fno_seconds_mean"], 1)
      and shown(ln2[5], want2["fno_ms_per_step_mean"], 2),
      f"txt: the case line says cached, no speedup ({' '.join(ln2)})")

print("\n=== pass 2b: two cases of one composition and lattice ===")
# a sweep-seed twin of the critical case lands on the same line: the sums
# add, so the line is over all their seeds, not a mean of the two means
twin = {"off": 0.0, "lattice": 64,
        "stats": {"timing": P.timing_totals(rows2[:1])}}
jm, tm = os.path.join(OUT, "merge_timing.json"), os.path.join(OUT,
                                                              "merge_timing.txt")
P.write_timing_report([case, twin], device, model, jm, tm)
repm = json.load(open(jm))
wantm = per_seed_means(rows1 + rows2[:1])
check(len(repm["cases"]) == 1 and agrees(repm["cases"][0], wantm)
      and repm["meta"]["n_rollouts"] == 3
      and repm["meta"]["n_ground_truths_solved"] == 2,
      "one line over all 3 rollouts, 2 of them solved, with their means")

print("\n=== pass 3: a later job finds the case finished ===")
# through evaluate_all, as main runs it: the case is taken from its summary,
# not rolled out, and its timing comes from the sums that summary keeps
P.LABEL, P.EVAL_MODE, P.SIZE_SWEEP = "tm", "none", (64,)
P.TIMING_JSON, P.TIMING_TEXT = "{label}3_timing.json", "{label}3_timing.txt"
refused = []


def refuse(*a, **kw):
    refused.append(a[2:])
    raise RuntimeError("a finished case was rolled out again")


def on_disk(path):
    """A file's bytes and mtime, to tell whether anything rewrote it."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return fh.read(), os.stat(path).st_mtime_ns


cache2 = P.case_cache_files(0.0, 64, P.EVAL_SEEDS)
check(len(cache2) == 2 and all(os.path.exists(f) for f in cache2),
      "before it: pass 2's two rollouts are in the cache")
before3 = on_disk(sp1)
status = P.evaluate_all(model, ckpt, device, case_fn=refuse)
check(not refused and status == 0,
      f"the finished case is not rolled out again (status {status})")
check(before3 is not None and on_disk(sp1) == before3,
      "its summary is left exactly as pass 2 wrote it")
XTRAS = os.path.join(P.RESULT_CACHE, P.EXTRAS_NAME)   # evaluate_all's reports
jp3 = os.path.join(XTRAS, "tm3_timing.json")
rep3 = json.load(open(jp3)) if os.path.exists(jp3) else {"meta": {}}
check(rep3.get("cases") == rep2["cases"],
      "its timing line is the one the rolled-out case gave, exactly")
check(all(rep3["meta"].get(k) == rep2["meta"][k] for k in
          ("n_rollouts", "n_ground_truths_solved",
           "n_ground_truths_from_cache")),
      "and so are the counts")
check(not any(os.path.exists(f) for f in
              P.case_cache_files(0.0, 64, P.EVAL_SEEDS))
      and sorted(os.listdir(P.RESULT_CACHE)) == [P.EXTRAS_NAME],
      "the case's cached rollouts are deleted; only extras/ is left in the "
      "cache folder, with the timing report in it")

print("\n=== pass 4: --redo rolls the finished case out again ===")
P.REDO = True
P.TIMING_JSON, P.TIMING_TEXT = "{label}4_timing.json", "{label}4_timing.txt"
redone = []


def redo(*a, **kw):
    redone.append(a[2:])
    return P.run_case(*a, **kw)


try:
    status4 = P.evaluate_all(model, ckpt, device, case_fn=redo)
finally:
    P.REDO = False
check(len(redone) == 1 and status4 == 0,
      f"with --redo the finished case is rolled out again (status {status4})")
check(on_disk(sp1) not in (None, before3),
      "and its summary is written anew")
jp4 = os.path.join(XTRAS, "tm4_timing.json")
rep4 = json.load(open(jp4)) if os.path.exists(jp4) else {"meta": {}}
row4 = (rep4.get("cases") or [{}])[0]
check(row4.get("n_rollouts") == 2 and row4.get("n_solved") == 0
      and rep4["meta"].get("n_ground_truths_from_cache") == 2,
      "its timing line is the new rollouts', both references from cache")
check(not any(os.path.exists(f) for f in
              P.case_cache_files(0.0, 64, P.EVAL_SEEDS))
      and sorted(os.listdir(P.RESULT_CACHE)) == [P.EXTRAS_NAME],
      "and its cached rollouts are deleted once more")

print("\n=== pass 5: the GPU timing mode (FNO_TIMING) ===")
from _paths import REAL_MODEL as DT_MODEL
TKNOBS = {"FNO_TIMING_SIZES": "16:32", "FNO_TIMING_BATCH": "2",
          "FNO_TIMING_SOLVER_T": "8", "FNO_TIMING_CALLS": "2",
          "FNO_TIMING_FULL": "0"}
TENV = ("FNO_TIMING", "FNO_TIMING_OUT", *TKNOBS)
saved_env = {k: os.environ.get(k) for k in TENV}
saved_full_t = P.TIMING_FULL_T


def listing(skip):
    """Every path under SCRATCH, minus the folder ``skip``."""
    keep = set()
    for here, dirs, files in os.walk(SCRATCH):
        for n in dirs + files:
            full = os.path.join(here, n)
            if os.path.abspath(full) != skip and not os.path.abspath(
                    full).startswith(skip + os.sep):
                keep.add(os.path.relpath(full, SCRATCH))
    return keep


def timing_run(spec, out, **more):
    """main() with FNO_TIMING=spec; returns (return value, stdout)."""
    import contextlib
    import io
    os.environ.update(TKNOBS)
    os.environ.update({k: str(v) for k, v in more.items()})
    os.environ["FNO_TIMING"] = spec
    os.environ["FNO_TIMING_OUT"] = out
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rv = P.main(["--allow-cpu"])
    return rv, buf.getvalue()


def timing_checks(tag, spec, names, dt_cond):
    out = os.path.abspath(f"timing_{tag}")
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(out + "_2", ignore_errors=True)
    before = listing(out)
    predicted = []
    real_ea = P.evaluate_all
    P.evaluate_all = lambda *a, **k: predicted.append(1) or 0
    try:
        rv, text = timing_run(spec, out)
    finally:
        P.evaluate_all = real_ea
    check(rv == 0 and not predicted,
          f"{tag}: the mode runs and returns, and no prediction runs")
    check(os.path.isfile(os.path.join(out, "timing.txt"))
          and os.path.isfile(os.path.join(out, "timing.json")),
          f"{tag}: timing.txt and timing.json are in FNO_TIMING_OUT")
    rep = json.load(open(os.path.join(out, "timing.json")))
    txt = open(os.path.join(out, "timing.txt")).read()
    check([r["l"] for r in rep["rows"]] == [16, 32]
          and [m["name"] for m in rep["models"]] == names
          and all(len(r["models"]) == len(names) for r in rep["rows"]),
          f"{tag}: one row per size and per model, keys: "
          f"{sorted(rep['rows'][0])}")
    check(all({"s_per_call", "ms_per_call", "s_per_run", "speedup"}
              <= set(m) for r in rep["rows"] for m in r["models"])
          and {"device", "host", "torch", "cuda", "precision", "batch"}
          <= set(rep["meta"]) and rep["validation"] is None,
          f"{tag}: the model and meta keys are there; no validation when off")
    check(all(abs(m["speedup"] - r["solver_s_per_run"] / m["s_per_run"])
              <= 1e-9 * m["speedup"]
              for r in rep["rows"] for m in r["models"]),
          f"{tag}: speed-up = solver share / model share")
    me = rep["meta"]
    check(me["batch"] == 2 and me["solver_steps"] == 800
          and abs(me["solver_scale"] - 990000 / 800) < 1e-9
          and me["calls_per_run"] == 320
          and all(abs(r["solver_s_per_run"] - r["solver_seconds"] / 2
                      * 990000 / 800) <= 1e-9 * r["solver_s_per_run"]
                  for r in rep["rows"])
          and all(abs(m["s_per_run"] - m["s_per_call"] / r["network_batch"]
                      * 320) <= 1e-9 * m["s_per_run"]
                  for r in rep["rows"] for m in r["models"]),
          f"{tag}: the shares are scaled by 990000/800 and 320 calls")
    check("990,000" in txt and "per run x 1238" in txt
          and "per run x 320" in txt and socket_name() in txt
          and "reference, other job" in txt and "24.2" in txt,
          f"{tag}: the scale factors, host and reference are in the file")
    check(any(m["dt_conditioned"] == dt_cond for m in rep["models"])
          and "CPU timing" in txt,
          f"{tag}: the network kind is noted, and it says CPU timing")
    check(not os.path.exists(os.path.join(out, "_solver_tmp"))
          and sorted(os.listdir(out)) == ["timing.json", "timing.txt"],
          f"{tag}: the temporary solver folder is gone")
    check(listing(out) == before,
          f"{tag}: nothing in SCRATCH outside FNO_TIMING_OUT changed")
    return out, rep


def socket_name():
    import socket
    return socket.gethostname()


try:
    P_MODEL = P.MODEL_PATH
    P.MODEL_PATH = REAL_MODEL
    try:
        out_f, rep_f = timing_checks(
            "fixed", "1", [fno_paths.model_label(REAL_MODEL)], False)
    finally:
        P.MODEL_PATH = P_MODEL
    if DT_MODEL is None:
        print("SKIP the dt-conditioned model: no REAL_MODEL here")
    else:
        P_MODEL = P.MODEL_PATH
        both = f"{DT_MODEL}:{REAL_MODEL}"
        out_d, rep_d = timing_checks(
            "list", both, [fno_paths.model_label(DT_MODEL),
                           fno_paths.model_label(REAL_MODEL)], True)
        check([m["dt_conditioned"] for m in rep_d["models"]] == [True, False],
              "the ':' list gave the dt-conditioned and the fixed-step model, "
              "in order")
    print("\n--- a label that is not a checkpoint is refused ---")
    msg = ""
    try:
        timing_run("no_such_label_xyz", os.path.abspath("timing_bad"))
    except SystemExit as e:
        msg = str(e)
    check("no_such_label_xyz" in msg and "no checkpoint" in msg
          and not os.path.exists(os.path.abspath("timing_bad")),
          f"an unknown label is refused with a message ({msg[:70]!r}...)")
    print("\n--- an existing folder gets a twin ---")
    out2 = os.path.abspath("timing_twin")
    shutil.rmtree(out2, ignore_errors=True)
    shutil.rmtree(out2 + "_2", ignore_errors=True)
    shutil.rmtree(out2 + "_3", ignore_errors=True)
    os.makedirs(out2)
    open(os.path.join(out2, "keep.txt"), "w").write("mine")
    P.MODEL_PATH = REAL_MODEL
    try:
        timing_run("1", out2)
        timing_run("1", out2)
    finally:
        P.MODEL_PATH = P_MODEL
    check(os.listdir(out2) == ["keep.txt"]
          and os.path.isfile(os.path.join(out2 + "_2", "timing.txt"))
          and os.path.isfile(os.path.join(out2 + "_3", "timing.txt")),
          "an existing folder is left alone; the result goes to _2, then _3")
    print("\n--- the validation block ---")
    P.TIMING_FULL_T = 20.0
    P.MODEL_PATH = REAL_MODEL
    try:
        out_v = os.path.abspath("timing_valid")
        shutil.rmtree(out_v, ignore_errors=True)
        timing_run("1", out_v, FNO_TIMING_FULL=1)
    finally:
        P.MODEL_PATH = P_MODEL
        P.TIMING_FULL_T = saved_full_t
    rv_ = json.load(open(os.path.join(out_v, "timing.json")))["validation"]
    check(rv_ and rv_["l"] == 16 and abs(rv_["difference_percent"] - 100 * (
        rv_["full_s_per_run"] - rv_["short_s_per_run"])
        / rv_["short_s_per_run"]) < 1e-9
        and "difference" in open(os.path.join(out_v, "timing.txt")).read(),
        "the validation block: full vs short-span estimate, with the "
        "difference in %")
finally:
    P.TIMING_FULL_T = saved_full_t
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

print("\n" + txt1)
print("TIMING PASS" if ok else "TIMING FAIL")
sys.exit(0 if ok else 1)
