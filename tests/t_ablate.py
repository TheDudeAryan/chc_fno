"""The ablation mode (FNO_ABLATE): the R2 of a model with its low-k fill
and/or its mu clamp switched off, on the real grid's schedules.

Small CPU models; the grid is made small by P.LATTICE = 16 and
P.SIZE_SWEEP = (16, 32), so it has ten cases; the references are solved into
a cache in SCRATCH.

  1. a run: the files, the keys, the entries, the settings as used
  2. the schedules: one per case, the pick of its composition, the same for
     every entry
  3. +nofill and +noclamp change the rollout; a model with no mu channel
  4. the pooled, common-6 and per-size means are the means over the right
     cases
  5. a diverging seed counts -1 from its divergence on
  6. the same numbers as the normal scoring code (evaluate_schedule)
  7. refusals, before anything is rolled out
  8. the folder is never an existing one; nothing is written into the project
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
import fno_gt
import fno_predict as P
import fno_train as T

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


D = os.path.join(SCRATCH, "ablate_t")
shutil.rmtree(D, ignore_errors=True)
os.makedirs(D)
P.WRITE_LOG = False
P.REF_CACHE = os.path.join(D, "gt")
P.LATTICE = 16
P.SIZE_SWEEP = (16, 32)
P.ABLATE_COMMON_BOX = 32
P.ABLATE_TIMES = (120.0, 140.0)
P.ROLLOUT_CELLS = 4 * 16 * 16
P.EVAL_SEEDS = (99999, 54321, 77777, 13579, 24680)
# the mode takes the group seeds unless --seed-set fno: the same five here
P.GROUP_SEEDS = P.EVAL_SEEDS
P.SWEEP_SEEDS = P.EVAL_SEEDS
P.MU_CLIP = 0.05             # small, so that the clamp matters to the rollout
SAVED_END = P.END_TIME
OFFS = (-0.4, -0.2, 0.0, 0.2, 0.4)
PICKS = {"-0.4": (0.1, 30.0), "-0.2": (0.1, 30.0), "+0.0": (0.2, 40.0),
         "+0.2": (0.3, 24.0), "+0.4": (0.1, 28.0)}
pick_dir = os.path.join(D, "pick")
os.makedirs(pick_dir)


def write_meta(picks, snapshots=(100.0, 120.0, 140.0)):
    meta = {"cases": {}, "snapshots": list(snapshots), "schedule": {
        k: {"alpha": a, "dt_max": m, "dt_min": 1.0, "source": "test",
            "signature": "not read"} for k, (a, m) in picks.items()}}
    with open(os.path.join(pick_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh)


write_meta(PICKS)
P.ABLATE_PICK_CACHE = pick_dir
for k in ("FNO_ABLATE", "FNO_ABLATE_OUT", "FNO_ABLATE_T_END",
          "FNO_ABLATE_SEEDS", "FNO_ABLATE_CASES", "FNO_ALLOW_CPU"):
    os.environ.pop(k, None)
os.environ["FNO_ABLATE_T_END"] = "140"
os.environ["FNO_ABLATE_SEEDS"] = "3"


def build(name, mu=True, fill=True, dt=True):
    """A small untrained dt-conditioned model, saved as <name>/fno_ch.pt in
    D/models; its path."""
    torch.manual_seed(5)
    model = T.CahnHilliardFNO(
        channels=8, modes=6, n_layers=2, n_ref=16, dt_conditioned=dt,
        zero_dc=True, hard_mass_conservation=True, laplacian_output=False,
        lowk_fill=fill, mu_channel=mu, mu_clip=1.2 if mu else None)
    with torch.no_grad():
        out = model.backbone.project[2]
        assert isinstance(out, torch.nn.Conv2d) and out.bias is not None
        torch.nn.init.normal_(out.weight, std=0.1)
        torch.nn.init.constant_(out.bias, 0.02)
    path = os.path.join(D, "models", name, "fno_ch.pt")
    os.makedirs(os.path.dirname(path))
    T.save_model(model, path)
    return path


FULL = build("train_t0500_lap")
NOMU = build("train_t0500_lap_nomu", mu=False)
NOFILLM = build("train_t0500_lap_nofillmodel", fill=False)
FIXED = build("train_t0500_lap_fixed", dt=False)


def ablate(spec, out, argv=("--allow-cpu",), **env):
    """main() with FNO_ABLATE=spec; (the return value, what it printed)."""
    os.environ["FNO_ABLATE"] = spec
    os.environ["FNO_ABLATE_OUT"] = out
    os.environ.update({k: str(v) for k, v in env.items()})
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rv = P.main(list(argv))
    finally:
        for k in ("FNO_ABLATE", "FNO_ABLATE_OUT", *env):
            os.environ.pop(k, None)
    return rv, buf.getvalue()


def load(out):
    with open(os.path.join(out, "ablate.json")) as fh:
        return json.load(fh)


print("1. a run")
out1 = os.path.join(D, "out1")
predicted = []
real_ea = P.evaluate_all
P.evaluate_all = lambda *a, **k: predicted.append(1) or 0
SPEC = f"{FULL}:{FULL}+nofill:{FULL}+noclamp:{NOMU}"
try:
    rv, text = ablate(SPEC, out1)
finally:
    P.evaluate_all = real_ea
rep = load(out1)
check(rv == 0 and not predicted
      and sorted(os.listdir(out1)) == ["ablate.json", "ablate.txt"],
      "the mode runs, no prediction runs, and the folder holds the two files "
      "and nothing else")
m = rep["meta"]
check({"device", "host", "date", "job_id", "seeds", "n_seeds", "seed_set",
       "schedule_source", "schedules", "report_times", "t_end"} <= set(m)
      and m["seed_set"] == "group" and m["seeds"] == list(P.GROUP_SEEDS[:3])
      and "meta.json" in m["schedule_source"] and m["n_cases"] == 10
      and m["report_times"] == [120.0, 140.0],
      "meta: the group seeds, the schedule source, the node, the device, "
      "the date and the job")
ent = {e["spec"]: e for e in rep["entries"]}
KEYS = {"spec", "label", "fingerprint", "lowk_fill", "mu_clip", "mu_channel",
        "complete", "cases", "groups"}
CASE = {"r2_at_times", "n_diverged", "n_seeds", "schedule", "off", "lattice",
        "per_seed_r2", "diverged_at"}
names = sorted(ent[FULL]["cases"])
check(len(ent) == 4 and all(KEYS <= set(e) for e in ent.values())
      and all(e["complete"] for e in ent.values())
      and all(len(e["cases"]) == 10 and all(CASE <= set(c)
                                            for c in e["cases"].values())
              for e in ent.values())
      and names == sorted([f"psi0{o:+.2f}_l16" for o in OFFS]
                          + [f"psi0{o:+.2f}_l32" for o in OFFS]),
      f"four entries with every key, ten cases each ({names[:2]} ...)")
check(all(set(c["r2_at_times"]) == {"120", "140"}
          and set(c["r2_at_times"]["140"])
          >= {"mean", "n", "diverged", "n_alive", "mean_alive"}
          and c["n_seeds"] == 3 and c["r2_at_times"]["140"]["n"] == 3
          for c in ent[FULL]["cases"].values()),
      "per case: the all-seeds mean, the survivors-only mean, n diverged, at "
      "each report time")
check((ent[FULL]["lowk_fill"], ent[FULL]["mu_clip"],
       ent[FULL]["mu_channel"]) == (True, 0.05, True)
      and (ent[FULL + "+nofill"]["lowk_fill"],
           ent[FULL + "+nofill"]["mu_clip"]) == (False, 0.05)
      and (ent[FULL + "+noclamp"]["lowk_fill"],
           ent[FULL + "+noclamp"]["mu_clip"]) == (True, None)
      and (ent[NOMU]["lowk_fill"], ent[NOMU]["mu_clip"],
           ent[NOMU]["mu_channel"]) == (True, None, False)
      and len({e["fingerprint"] for e in rep["entries"]}) == 2
      and ent[FULL]["label"] == "train_t0500_lap"
      and ent[NOMU]["label"] == "train_t0500_lap_nomu",
      "the settings as used: the default is the checkpoint's fill and the "
      "module's clamp; +nofill, +noclamp and the no-mu model are what they "
      "say; the labels and fingerprints are in")
txt = open(os.path.join(out1, "ablate.txt")).read()
check(txt.count("fp ") == 4 and "common 6 cases" in txt
      and "l = 32" in txt and "NOT COMPLETE" not in txt,
      "ablate.txt has one block per entry with the pooled, common-6 and "
      "per-size rows")

print("\n2. the schedules")
sch = {n: ent[FULL]["cases"][n]["schedule"] for n in names}
want = {}
for o in OFFS:
    a, mx = PICKS[f"{o:+.1f}"]
    must = [120.0, 140.0]
    n_steps = len(fno_gt.make_schedule(
        100.0, 140.0, a, 1.0, mx, quantum=fno_gt.GT_EVERY,
        must_hit=must)) - 1
    for lat in (16, 32):
        want[f"psi0{o:+.2f}_l{lat}"] = (a, mx, n_steps)
check(all((s["alpha"], s["dt_max"], s["steps"]) == want[n]
          for n, s in sch.items()),
      "each case has the pick of its composition (the box size does not "
      "matter) and the steps that pick makes")
check(len({s["alpha"] for s in sch.values()}) == 3
      and all(ent[e]["cases"][n]["schedule"] == sch[n]
              for e in ent for n in names),
      "the compositions have different picks and every entry has the same "
      "schedule in every case")

print("\n3. the settings change what they should")


def r2s(e, n):
    return ent[e]["cases"][n]["per_seed_r2"]["140"]


big = "psi0+0.00_l32"
check(r2s(FULL, big) != r2s(FULL + "+nofill", big),
      "+nofill changes the R2 of a case at a box that resamples the weights")
check(all(r2s(FULL, n) != r2s(FULL + "+noclamp", n) for n in names),
      "+noclamp changes the R2 of every case (the clamp was in use)")
check(r2s(FULL, big) != r2s(NOMU, big),
      "the model with no mu channel is another model")
rv, text = ablate(f"{NOMU}+nofill", os.path.join(D, "out_nm"),
                  FNO_ABLATE_CASES="psi0+0.00_l16")
nm = load(os.path.join(D, "out_nm"))["entries"][0]
check(rv == 0 and nm["lowk_fill"] is False and nm["mu_clip"] is None
      and nm["mu_channel"] is False and list(nm["cases"]) == ["psi0+0.00_l16"]
      and nm["groups"]["common_6"] is None,
      "+nofill on the no-mu model works (it has a fill); FNO_ABLATE_CASES "
      "keeps one case, and then there is no common-6 row")

print("\n4. the pooled means")
g = ent[FULL]["groups"]
cases = ent[FULL]["cases"]


def mean_over(sel, t="140", key="mean"):
    return float(np.mean([cases[n]["r2_at_times"][t][key] for n in sel]))


common = [n for n in names if n.endswith("_l16")] + ["psi0+0.00_l32"]
check(g["pooled"]["cases"] == 10 and g["pooled"]["seeds"] == 30
      and abs(g["pooled"]["at"]["140"]["mean"] - mean_over(names)) < 1e-12
      and abs(g["pooled"]["at"]["120"]["mean"]
              - mean_over(names, "120")) < 1e-12
      and g["pooled"]["at"]["140"]["n"] == 30,
      "pooled: the mean over all ten cases")
check(g["common_6"] is not None and g["common_6"]["cases"] == 6
      and abs(g["common_6"]["at"]["140"]["mean"] - mean_over(common)) < 1e-12
      and abs(g["common_6"]["at"]["140"]["mean_alive"]
              - mean_over(common, key="mean_alive")) < 1e-12,
      "common 6: the five compositions at the base box and the critical "
      "quench at the other (all-seeds and survivors-only means)")
check(set(g["by_size"]) == {"16", "32"}
      and g["by_size"]["32"]["cases"] == 5
      and abs(g["by_size"]["32"]["at"]["140"]["mean"]
              - mean_over([n for n in names if n.endswith("_l32")])) < 1e-12,
      "per box size: the mean over the cases of that size")

print("\n5. a diverging seed")
real_roll = P.roll_schedule
rng = np.random.default_rng(1)
ts = np.array([100.0, 104.0, 108.0, 112.0, 116.0])
truth = rng.standard_normal((5, 3, 8, 8)) * 0.3


def fake_roll(model, psi0, times, device, on_step):
    """A model that is right for seed 0, blows up at the 2nd stride for seed
    1 and gives nan at the 3rd for seed 2."""
    last = 0
    for i in range(1, len(times)):
        psi = torch.as_tensor(truth[i], dtype=torch.float64).unsqueeze(1)
        psi = psi.clone()
        if i >= 2:
            psi[1] = 9.0
        if i >= 3:
            psi[2] = float("nan")
        last = i
        if not on_step(i, psi):
            break
    return last, last, 0.0


P.roll_schedule = fake_roll
try:
    r2, div = P.ablate_score(None, truth, ts, "cpu")
finally:
    P.roll_schedule = real_roll
check(np.allclose(r2[:, 0], 1.0)
      and r2[0, 1] == 1.0 and r2[1, 1] > -1.0 and (r2[2:, 1] == -1.0).all()
      and r2[0, 2] == 1.0 and r2[1, 2] > -1.0 and r2[2, 2] > -1.0
      and (r2[3:, 2] == -1.0).all(),
      "R2 is 1 at the start; a seed that blows up (|psi| > 5) or goes "
      "non-finite is -1 from its divergence on")
check(np.isnan(div[0]) and div[1] == ts[2] and div[2] == ts[3],
      "diverged_at is the time of the divergence, nan for a seed that "
      "never diverges")
rows = P.ablate_case_numbers(ts, r2, div)
P.ABLATE_TIMES = (112.0, 116.0)
try:
    rows = P.ablate_case_numbers(ts, r2, div)
finally:
    P.ABLATE_TIMES = (120.0, 140.0)
a116 = rows["r2_at_times"]["116"]
check(rows["n_diverged"] == 2 and a116["n"] == 3 and a116["diverged"] == 2
      and abs(a116["mean"] - (1.0 - 1.0 - 1.0) / 3) < 1e-9
      and a116["n_alive"] == 1 and abs(a116["mean_alive"] - 1.0) < 1e-9,
      "the all-seeds mean keeps the diverged seeds at -1, the survivors-only "
      "mean leaves them out, n diverged counts them")
P.DIVERGE_PSI = 1e-3
try:
    out_d = os.path.join(D, "out_div")
    ablate(FULL, out_d, FNO_ABLATE_CASES="psi0+0.00_l16")
finally:
    P.DIVERGE_PSI = 5.0
c = load(out_d)["entries"][0]["cases"]["psi0+0.00_l16"]
check(c["n_diverged"] == 3 and all(x is not None for x in c["diverged_at"])
      and c["r2_at_times"]["140"]["mean"] == -1.0
      and c["r2_at_times"]["140"]["mean_alive"] is None
      and c["r2_at_times"]["140"]["diverged"] == 3,
      "a real run in which every seed diverges: R2 = -1, no survivors")

print("\n6. the same numbers as the normal scoring")
dev = "cpu"
seeds = [int(s) for s in P.GROUP_SEEDS[:3]]
name = "psi0+0.20_l32"
c = ent[FULL]["cases"][name]
a, mx = PICKS["+0.2"]
times = np.asarray(fno_gt.make_schedule(
    100.0, 140.0, a, 1.0, mx, quantum=fno_gt.GT_EVERY,
    must_hit=[120.0, 140.0]), dtype=np.float64)
saved = (P.END_TIME, P.SNAPSHOT_TIMES)


def normal_scoring(model):
    """evaluate_schedule on the case, a chunk at a time as the real run
    does (ROLLOUT_CELLS // l^2 seeds: one at l = 32 here); its results."""
    got = []
    P.END_TIME, P.SNAPSHOT_TIMES = 140.0, (100.0, 120.0, 140.0)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for sd in seeds:
                refs = P.load_references(32, 0.2, [sd], dev)
                r, _ = P.evaluate_schedule(
                    model, [sd], dev, refs, times,
                    {"alpha": a, "dt_min": 1.0, "dt_max": mx})
                got += r
    finally:
        P.END_TIME, P.SNAPSHOT_TIMES = saved
    return got


model, _ = T.load_checkpoint(FULL)
with contextlib.redirect_stdout(io.StringIO()):
    P.apply_lowk_fill(model)
    P.apply_mu_clip(model)
res = normal_scoring(model)
normal = P._submission().r2_at_times(res, (120.0, 140.0))
mine = c["r2_at_times"]
check(all(abs(normal[k][f] - mine[k][f]) < 1e-9
          for k in normal for f in ("mean", "mean_alive"))
      and all(normal[k]["diverged"] == mine[k]["diverged"]
              and normal[k]["n"] == mine[k]["n"] for k in normal)
      and set(normal) == set(mine),
      "run_ablate's R2 at the report times equals evaluate_schedule's, "
      "scored by make_submission.r2_at_times, for the same model, "
      "schedule and seeds")
i140 = int(np.argmin(abs(np.asarray(res[0]["times"]) - 140.0)))
check(np.allclose([r["r2"][i140] for r in res], c["per_seed_r2"]["140"],
                  atol=1e-9, rtol=0),
      "and seed by seed")
# the same with the clamp off and the fill off against a model set so
nm_model, _ = T.load_checkpoint(FULL)
with contextlib.redirect_stdout(io.StringIO()):
    P.apply_lowk_fill(nm_model)
    P.apply_mu_clip(nm_model)
T.set_lowk_fill(nm_model, False)
T.set_mu_clip(nm_model, None)
res2 = normal_scoring(nm_model)
rv, _ = ablate(f"{FULL}+nofill+noclamp", os.path.join(D, "out_both"),
               FNO_ABLATE_CASES=name)
b = load(os.path.join(D, "out_both"))["entries"][0]
check(b["lowk_fill"] is False and b["mu_clip"] is None
      and np.allclose([r["r2"][i140] for r in res2],
                      b["cases"][name]["per_seed_r2"]["140"], atol=1e-9,
                      rtol=0),
      "both suffixes together: the same as the normal scoring of a model "
      "with the fill and the clamp set off")

print("\n7. refusals")


def refused(spec, out, **kw):
    called = []
    real = P.ablate_score
    P.ablate_score = lambda *a, **k: called.append(1) or (None, None)
    try:
        ablate(spec, out, **kw)
        return None, called
    except SystemExit as exc:
        return str(exc), called
    finally:
        P.ablate_score = real


def gone(out):
    return not os.path.exists(out)


o = os.path.join(D, "out_r")
msg, called = refused(f"{NOMU}+noclamp", o + "1")
check(msg and "no mu channel" in msg and not called and gone(o + "1"),
      "+noclamp on a model with no mu channel is refused, nothing rolled out")
msg, called = refused(f"{NOFILLM}+nofill", o + "2")
check(msg and "no low-k weight fill" in msg and not called and gone(o + "2"),
      "+nofill on a model with no fill is refused")
msg, called = refused(f"{FULL}:{D}/models/none/fno_ch.pt", o + "3")
check(msg and "no checkpoint" in msg and "nothing was rolled out" in msg
      and not called and gone(o + "3"),
      "a missing checkpoint stops the run before anything is rolled out")
msg, called = refused(FIXED, o + "4")
check(msg and "not dt-conditioned" in msg and not called and gone(o + "4"),
      "a fixed-step model is refused")
msg, called = refused("no_such_label_xyz", o + "5")
check(msg and "no_such_label_xyz" in msg and not called,
      "an unknown label is refused with its name")
write_meta({k: v for k, v in PICKS.items() if k != "+0.2"})
msg, called = refused(FULL, o + "6")
check(msg and "no schedule pick for psi0 = +0.2" in msg and not called
      and gone(o + "6"),
      "a missing pick stops the run with the composition named")
write_meta(PICKS, snapshots=())
msg, called = refused(FULL, o + "7")
check(msg and "no snapshot" in msg and gone(o + "7"),
      "a meta.json with no snapshot list is refused")
write_meta(PICKS)
P.ABLATE_PICK_CACHE = os.path.join(D, "no_such_cache")
msg, called = refused(FULL, o + "8")
check(msg and "no meta.json" in msg and not called and gone(o + "8"),
      "no meta.json at all is refused")
P.ABLATE_PICK_CACHE = pick_dir
msg, called = refused(FULL, o + "9", FNO_ABLATE_CASES="psi0+0.00_l99")
check(msg and "no case" in msg and not called and gone(o + "9"),
      "an unknown case name is refused")
if not torch.cuda.is_available():
    msg, called = refused(FULL, o + "10", argv=())
    check(msg and "no CUDA device" in msg and not called and gone(o + "10"),
          "a CPU device is refused without --allow-cpu")
    rv, _ = ablate(FULL, o + "11", argv=(), FNO_ALLOW_CPU=1,
                   FNO_ABLATE_CASES="psi0+0.00_l16")
    check(rv == 0 and os.path.isfile(os.path.join(o + "11", "ablate.json")),
          "and FNO_ALLOW_CPU=1 allows it")

print("\n8. the folder")
out_t = os.path.join(D, "out_twin")
os.makedirs(out_t)
with open(os.path.join(out_t, "keep.txt"), "w") as fh:
    fh.write("mine")
for _ in range(2):
    ablate(FULL, out_t, FNO_ABLATE_CASES="psi0+0.00_l16")
check(os.listdir(out_t) == ["keep.txt"]
      and os.path.isfile(os.path.join(out_t + "_2", "ablate.json"))
      and os.path.isfile(os.path.join(out_t + "_3", "ablate.json")),
      "an existing folder is left alone; the result goes to _2, then _3")
check(P.END_TIME == SAVED_END and P.START_TIME == 100.0,
      "END_TIME is put back after the run")
default_out = P.fresh_folder("FNO_ABLATE_OUT_UNSET", "ablate")
check(os.path.join("analysis", "ablate_") in default_out
      and not os.path.exists(default_out),
      "without FNO_ABLATE_OUT the folder is Work/analysis/ablate_<date>")
check(project_state() == PROJECT, "nothing was written into the project")

print("\n" + ("ABLATE PASS" if ok else "ABLATE FAIL"))
sys.exit(0 if ok else 1)
