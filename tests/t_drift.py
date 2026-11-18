"""The drift mode (FNO_DRIFT): <psi_pred> - <psi_true> of a rollout, with and
without the final mean removal.

Small CPU models, a 16 x 16 case solved into a reference cache in SCRATCH.

  1. a run: the files, their keys, the entries, the common schedule
  2. +nomean changes the deviation of a model whose raw increment has a
     nonzero mean, and a model with the removal keeps <psi> exactly
  3. +nomean on a model with no mean removal, a missing checkpoint, and the
     CPU refusal
  4. a seed that diverges has nulls from then on
  5. the output folder is never an existing one; nothing is written into the
     project
"""
import contextlib
import io
import json
import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from _paths import ROOT, SCRATCH, project_state

os.chdir(SCRATCH)
PROJECT = project_state()
import fno_predict as P
import fno_train as T

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


D = os.path.join(SCRATCH, "drift_t")
shutil.rmtree(D, ignore_errors=True)
os.makedirs(D)
P.WRITE_LOG = False
P.REF_CACHE = os.path.join(D, "gt")
P.DRIFT_LATTICE = 16
P.DRIFT_PICK_CACHE = os.path.join(D, "no_such_cache")
P.ROLLOUT_CELLS = 4 * 16 * 16
P.EVAL_SEEDS = (99999, 54321, 77777, 13579, 24680)
# the mode takes the group seeds unless --seed-set fno: the same five here
P.GROUP_SEEDS = P.EVAL_SEEDS
SAVED_END = P.END_TIME
for k in ("FNO_DRIFT", "FNO_DRIFT_OUT", "FNO_DRIFT_T_END", "FNO_DRIFT_SEEDS",
          "FNO_ALLOW_CPU"):
    os.environ.pop(k, None)
os.environ["FNO_DRIFT_T_END"] = "140"


def build(name, hard):
    """A small untrained dt-conditioned plain-head model, saved as <name>.pt
    in a folder named for it.  With hard=False it has no final mean removal
    and no zero DC, as the --no-hard-constraints ablation."""
    torch.manual_seed(3)
    model = T.CahnHilliardFNO(
        channels=8, modes=6, n_layers=2, n_ref=16, dt_conditioned=True,
        zero_dc=hard, hard_mass_conservation=hard, laplacian_output=False)
    with torch.no_grad():
        out = model.backbone.project[2]
        assert isinstance(out, torch.nn.Conv2d) and out.bias is not None
        torch.nn.init.normal_(out.weight, std=0.1)
        torch.nn.init.constant_(out.bias, 0.02)
    path = os.path.join(D, "models", name, "fno_ch.pt")
    os.makedirs(os.path.dirname(path))
    T.save_model(model, path, train_config={"constants": {"MASS_WEIGHT": 0.1}})
    return path


HARD = build("train_t0500", True)
SOFT = build("train_t0500_nohard", False)


def drift(spec, out, argv=("--allow-cpu",), **env):
    """main() with FNO_DRIFT=spec; (the return value, what it printed)."""
    os.environ["FNO_DRIFT"] = spec
    os.environ["FNO_DRIFT_OUT"] = out
    os.environ.update({k: str(v) for k, v in env.items()})
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rv = P.main(list(argv))
    finally:
        for k in ("FNO_DRIFT", "FNO_DRIFT_OUT", *env):
            os.environ.pop(k, None)
    return rv, buf.getvalue()


def load(out):
    with open(os.path.join(out, "drift.json")) as fh:
        return json.load(fh)


print("1. a run")
out1 = os.path.join(D, "out1")
predicted = []
real_ea = P.evaluate_all
P.evaluate_all = lambda *a, **k: predicted.append(1) or 0
try:
    rv, text = drift(f"{HARD}:{HARD}+nomean:{SOFT}", out1)
finally:
    P.evaluate_all = real_ea
rep = load(out1)
check(rv == 0 and not predicted and sorted(os.listdir(out1))
      == ["drift.json", "drift.txt"],
      "the mode runs, no prediction runs, and the folder holds the two files")
check({"meta", "times", "mean_true", "entries"} <= set(rep)
      and {"device", "host", "date", "job_id", "off", "lattice", "n_seeds",
           "alpha", "dt_max", "schedule_source", "t_end"} <= set(rep["meta"])
      and rep["meta"]["lattice"] == 16 and rep["meta"]["off"] == 0.0
      and rep["meta"]["n_seeds"] == 5 and rep["meta"]["t_end"] == 140.0
      and rep["meta"]["seed_set"] == "group"
      and rep["meta"]["seeds"] == list(P.GROUP_SEEDS),
      "meta: the case, the group seed set, the node, the device, the date "
      "and the job")
n = len(rep["times"])
KEYS = {"spec", "label", "fingerprint", "laplacian_output",
        "hard_mass_conservation", "zero_dc", "train_mass_weight",
        "deviation", "mean_deviation", "max_abs_per_seed", "max_abs_median",
        "max_abs_max", "n_diverged"}
ent = {e["spec"]: e for e in rep["entries"]}
check(len(ent) == 3 and all(KEYS <= set(e) for e in ent.values())
      and all(len(e["deviation"]) == 5 and len(e["deviation"][0]) == n
              and len(e["mean_deviation"]) == n
              and len(e["max_abs_per_seed"]) == 5
              for e in ent.values())
      and rep["times"][0] == 100.0 and rep["times"][-1] == 140.0
      and len(rep["mean_true"]) == 5,
      f"three entries with every key; deviation is seeds x {n} times")
check([e["hard_mass_conservation"] for e in rep["entries"]]
      == [True, False, False]
      and [e["zero_dc"] for e in rep["entries"]] == [True, True, False]
      and all(e["train_mass_weight"] == 0.1 for e in rep["entries"])
      and ent[HARD]["fingerprint"] != ent[SOFT]["fingerprint"],
      "the flags are after +nomean; the training weight and the "
      "fingerprints are in")
check("alpha 0.1, dt_max 30" in text
      and "DRIFT_ALPHA / DRIFT_DT_MAX" in text
      and rep["meta"]["alpha"] == 0.1 and rep["meta"]["dt_max"] == 30.0,
      "no pick in the cache: alpha 0.1 and dt_max 30, and it says so")
txt = open(os.path.join(out1, "drift.txt")).read()
check(len([ln for ln in txt.splitlines() if ln.startswith(D)]) == 3
      and "diverged 0 of 5" in txt,
      "drift.txt has one line per entry")

print("\n2. +nomean")
a, b = ent[HARD], ent[HARD + "+nomean"]
check(a["max_abs_max"] < 1e-5 and b["max_abs_max"] > 1e-3
      and b["max_abs_median"] > 1e-3,
      f"with the removal <psi> stays at the reference's ({a['max_abs_max']:.1e}); "
      f"without it it moves ({b['max_abs_max']:.1e})")
check(abs(b["mean_deviation"][-1]) > 1e-3 and b["n_diverged"] == 0,
      "and the seed mean of the deviation is not zero at the end")
pick_dir = os.path.join(D, "pick")
os.makedirs(pick_dir)
with open(os.path.join(pick_dir, "meta.json"), "w") as fh:
    json.dump({"cases": {}, "schedule": {"+0.0": {
        "alpha": 0.2, "dt_max": 20.0, "dt_min": 1.0}}}, fh)
P.DRIFT_PICK_CACHE = pick_dir
out_p = os.path.join(D, "out_pick")
rv, text = drift(HARD, out_p)
rp = load(out_p)
check(rp["meta"]["alpha"] == 0.2 and rp["meta"]["dt_max"] == 20.0
      and "the pick in" in rp["meta"]["schedule_source"]
      and "alpha 0.2, dt_max 20" in text,
      "the psi0 = 0 pick of the cache's meta.json is the schedule, and the "
      "run prints which")
P.DRIFT_PICK_CACHE = os.path.join(D, "no_such_cache")

print("\n3. refusals")


def refused(spec, out, **kw):
    called = []
    real = P.drift_rollout
    P.drift_rollout = lambda *a, **k: called.append(1)
    try:
        drift(spec, out, **kw)
        return None, called
    except SystemExit as exc:
        return str(exc), called
    finally:
        P.drift_rollout = real


msg, called = refused(f"{SOFT}+nomean", os.path.join(D, "out_r1"))
check(msg and "no final mean removal" in msg and not called
      and not os.path.exists(os.path.join(D, "out_r1")),
      "+nomean on a model with no mean removal is refused, nothing rolled out")
msg, called = refused(f"{HARD}:{D}/models/none/fno_ch.pt",
                      os.path.join(D, "out_r2"))
check(msg and "no checkpoint" in msg and "nothing was rolled out" in msg
      and not called and not os.path.exists(os.path.join(D, "out_r2")),
      "a missing checkpoint stops the run before anything is rolled out")
msg, called = refused("no_such_label_xyz", os.path.join(D, "out_r3"))
check(msg and "no_such_label_xyz" in msg and not called,
      "an unknown label is refused with its name")
if not torch.cuda.is_available():
    msg, called = refused(HARD, os.path.join(D, "out_r4"), argv=())
    check(msg and "no CUDA device" in msg and not called
          and not os.path.exists(os.path.join(D, "out_r4")),
          "a CPU device is refused without --allow-cpu")
    rv, _ = drift(HARD, os.path.join(D, "out_r5"), argv=(),
                  FNO_ALLOW_CPU=1)
    check(rv == 0 and os.path.isfile(os.path.join(D, "out_r5", "drift.json")),
          "and FNO_ALLOW_CPU=1 allows it")

print("\n4. a diverging seed")
saved_limit = P.DIVERGE_PSI
P.DIVERGE_PSI = 1e-3
try:
    out_d = os.path.join(D, "out_div")
    rv, _ = drift(HARD, out_d)
finally:
    P.DIVERGE_PSI = saved_limit
e = load(out_d)["entries"][0]
check(e["n_diverged"] == 5
      and all(row[0] is not None and all(x is None for x in row[1:])
              for row in e["deviation"])
      and e["mean_deviation"][1:] == [None] * (n - 1)
      and e["max_abs_per_seed"][0] is not None,
      "a seed that diverges has a deviation at the start and nulls from "
      "its divergence on; the means are null where none is alive")

print("\n5. the folder")
out_t = os.path.join(D, "out_twin")
os.makedirs(out_t)
with open(os.path.join(out_t, "keep.txt"), "w") as fh:
    fh.write("mine")
drift(HARD, out_t)
drift(HARD, out_t)
check(os.listdir(out_t) == ["keep.txt"]
      and os.path.isfile(os.path.join(out_t + "_2", "drift.json"))
      and os.path.isfile(os.path.join(out_t + "_3", "drift.json")),
      "an existing folder is left alone; the result goes to _2, then _3")
check(P.END_TIME == SAVED_END,
      "END_TIME is put back after the run")
check(project_state() == PROJECT,
      "nothing was written into the project")

print("\n" + ("DRIFT PASS" if ok else "DRIFT FAIL"))
sys.exit(0 if ok else 1)
