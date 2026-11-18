"""The prediction run's diagnostics of the loaded model: a step on a field
against a step on its 2 x 2 tiling, and the low-k increment gain, written to
diagnostics.json in the run's extras/ folder of the rollout cache (one per
view when there are several), and read into report.txt.

Everything is tiny and on the CPU: a real CahnHilliardFNO of 8 channels and
4 modes on lattices of 16 and 32, references solved in a few hundred steps,
and runs of the real main() in a box under _scratch with the model loader and
the case function stubbed.

  1. the tile check: the tiny FNO is invariant, a model with a position in it
     or index mode scaling is flagged; the steps are taken by a float64 copy
     on the CPU and the model itself is left as it was
  2. the low-k gain: the identity has gain 0, the exact solver 1, and the
     starts and the helpers are fno_train's
  3. diagnostics(): the record, the summary, a check that fails
  4. a run: diagnostics.json in every view's extras folder, once, and none in
     the Results folders; --no-diagnostics; a regeneration; --smoke
  5. nothing a rollout holds changes
"""
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import types
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from torch import nn

from _paths import SCRATCH, project_state

BOX = os.path.join(SCRATCH, "diag_box")
shutil.rmtree(BOX, ignore_errors=True)
os.makedirs(BOX)
os.chdir(BOX)
PROJECT = project_state()

import fno_paths
fno_paths.WORK = os.path.join(BOX, "Work")
import fno_predict as P
import fno_train as T
T.use_fixed_step()
# a fixed-step model's checks read FIXED_STEP's seeds and window (fno_train's own
# are the dt-conditioned ones now)
FIXED_STEP: dict[str, Any] = T.FIXED_STEP
import make_submission as S

setattr(torch.cuda, "is_available", lambda: False)
ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def printed(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        got = fn(*a, **k)
    return got, buf.getvalue()


START = {k: v for k, v in vars(P).items() if k.isupper()}
CKPT = os.path.join(BOX, "Work", "models", "train_t0001", "fno_ch.pt")
os.makedirs(os.path.dirname(CKPT))
with open(CKPT, "wb") as fh:
    fh.write(b"a checkpoint, as far as the fingerprint can tell " * 20)
T.CHANNELS, T.MODES, T.LAYERS = 8, 4, 2


def reset():
    """The module as it is written, then the tiny sizes of this suite: a
    16 field tiled to 32, gains at l = 32 over 3 seeds from t = 5 to 12."""
    for name, value in START.items():
        setattr(P, name, value)
    P.WRITE_LOG = False
    P.PER_SEED_FIGURES = False
    P.GT_CACHE = os.path.join(BOX, "gt")
    P.RESULT_CACHE = os.path.join(BOX, "rollouts")
    P.MODEL_PATH = CKPT
    P.LATTICE = 32
    setattr(P, "SIZE_SWEEP", (32,))
    P.START_TIME, P.END_TIME = 5.0, 12.0
    P.DIAG_TILE, P.DIAG_TILE_T = 16, 5.0
    P.DIAG_LOWK_L, P.DIAG_LOWK_T_END = 32, 12.0
    FIXED_STEP["VAL_SEEDS"] = (11, 12, 13)
    T.LOWK_T0 = 5.0
    P.DIAG_LOWK_SEEDS = 3


def tiny_model(n_ref=16, **kw):
    """A real CahnHilliardFNO, its zero-initialised head made to do
    something (a model that adds nothing is tile invariant for nothing)."""
    torch.manual_seed(0)
    model = T.CahnHilliardFNO(channels=8, modes=4, n_layers=2, n_ref=n_ref,
                              **kw)
    last = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]
    with torch.no_grad():
        last.weight.normal_(0.0, 0.5)
    return model.eval()


class Positional(nn.Module):
    """The tiny FNO plus 1e-3 times the column: it knows where a pixel is."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.config = base.config

    def forward(self, x):
        y = self.base(x)
        return y + 1e-3 * torch.arange(x.shape[-1], dtype=y.dtype)


class Seen(nn.Module):
    """A float32 model that notes the dtype and device of every field it is
    given, in SEEN (a copy of it writes in the same list)."""
    SEEN: list[tuple[torch.dtype, str, torch.dtype]] = []

    def __init__(self):
        super().__init__()
        self.config = {"n_ref": 16}
        self.p = nn.Parameter(torch.ones(1))

    def forward(self, x):
        Seen.SEEN.append((x.dtype, x.device.type, self.p.dtype))
        return x * self.p


class Plain(nn.Module):
    """A float64 stand-in that records the training lattice it claims."""

    def __init__(self, n_ref=16):
        super().__init__()
        self.config = {"n_ref": n_ref}
        self.p = nn.Parameter(torch.zeros(1, dtype=torch.float64))


class Identity(Plain):
    def forward(self, x):
        return x


class Exact(Plain):
    """The true next frame: the solver's own steps from the one it is given."""

    def forward(self, x):
        for _ in range(int(round(T.DT_STEP / T.DT))):
            x = T.ch_step(x)
        return x


reset()
model = tiny_model()
psi0 = quiet(T.ch_ground_truth, 16, 11, 0.0, 5.0, T.DT_STEP,
             cache_dir=P.GT_CACHE)[1][-1]
with torch.no_grad():
    x0 = torch.as_tensor(psi0, dtype=torch.float32)[None, None]
    moved = float((model(x0) - x0).abs().max())

print("1. the tile check")
check(moved > 1e-4, f"the tiny model does something: it moves psi by "
                    f"{moved:.1e}")
tile = quiet(P.tile_invariance, model, "cpu")
check(tile["invariant"] and tile["max_abs"] < 1e-5 and tile["rel_l2"] < 1e-5,
      f"the tiny FNO is tile invariant: largest difference "
      f"{tile['max_abs']:.1e}, relative L2 {tile['rel_l2']:.1e}")
check((tile["l"], tile["tiled_l"], tile["seed"], tile["t"])
      == (16, 32, 11, 5.0) and tile["tolerance"] == P.DIAG_TILE_TOL,
      "on the first validation seed's field at t = 5, 16 tiled to 32")
bent = quiet(P.tile_invariance, Positional(model), "cpu")
check(not bent["invariant"] and bent["max_abs"] > 1e-3
      and bent["rel_l2"] > 1e-4,
      f"a model that adds 1e-3 x column is not: largest difference "
      f"{bent['max_abs']:.1e}")
by_index = quiet(P.tile_invariance, tiny_model(mode_scaling="index"), "cpu")
check(not by_index["invariant"],
      f"nor is one with index mode scaling ({by_index['max_abs']:.1e})")
by_fill = quiet(P.tile_invariance, tiny_model(lowk_fill=True), "cpu")
check(by_fill["invariant"] and by_fill["max_abs"] < 1e-5,
      f"and one with the low-k weight fill on still is "
      f"({by_fill['max_abs']:.1e}): the odd modes the fill touches are "
      f"empty in a tiling")
line = P.diagnostics_text({"tile": tile, "lowk": {"skipped": "x"},
                           "seconds": 0.0})
check((tile["device"], tile["dtype"]) == ("cpu", "float64")
      and "float64 on cpu" in line,
      "the record says the steps were taken in float64 on the CPU, and so "
      "does the printed line")
seen = Seen()
Seen.SEEN.clear()
quiet(P.tile_invariance, seen, "cpu")
check(Seen.SEEN and set(Seen.SEEN) == {(torch.float64, "cpu", torch.float64)}
      and seen.p.dtype == torch.float32,
      "a float32 model is asked for its steps in float64 (its copy's "
      "weights too), and is itself still float32")
train_mode = tiny_model().train()
weights = {k: v.clone() for k, v in train_mode.state_dict().items()}
quiet(P.tile_invariance, train_mode, "cpu")
check(train_mode.training and all(
        v.dtype == torch.float32 and torch.equal(v, weights[k])
        for k, v in train_mode.state_dict().items()),
      "the model given is not changed: still in training mode if it was, "
      "weights and dtype the same")
loose = P.DIAG_TILE_TOL
P.DIAG_TILE_TOL = 1.0
check(quiet(P.tile_invariance, Positional(model), "cpu")["invariant"],
      "the verdict follows DIAG_TILE_TOL")
P.DIAG_TILE_TOL = loose
P.DIAG_TILE_T = 3.0
check(quiet(P.tile_invariance, model, "cpu")["t"] == 3.0,
      "and the field's time follows DIAG_TILE_T")
P.DIAG_TILE_T = 5.0

print("2. the low-k gain")
lk0 = quiet(P.lowk_check, Identity(), "cpu")
gains0 = [v for g in lk0["gain"].values() for v in g.values()]
check(list(lk0["gain"]) == ["5", "11"] and len(gains0) == 8
      and all(v == 0.0 for v in gains0),
      f"a model that leaves psi alone has gain 0 in every shell, at "
      f"t0 = {list(lk0['gain'])}")
lk1 = quiet(P.lowk_check, Exact(), "cpu")
gains1 = [v for g in lk1["gain"].values() for v in g.values()]
check(len(gains1) == 8 and all(v is not None and abs(v - 1.0) < 1e-6
                               for v in gains1),
      f"one that returns the true next frame has gain 1 "
      f"({min(gains1):.9f}..{max(gains1):.9f})")
check(lk1["lattice"] == 32 and lk1["n_ref"] == 16 and lk1["dt"] == T.DT_STEP
      and lk1["seeds"] == [11, 12, 13]
      and lk1["shells"] == list(T.LOWK_SHELLS)
      and set(lk1["gain"]["5"]) == set(T.LOWK_SHELLS),
      "the record names the lattice, the training lattice, the seeds and "
      "the shells")
real = quiet(P.lowk_check, model, "cpu")
check(all(isinstance(v, float) and abs(v) < 5.0
          for g in real["gain"].values() for v in g.values())
      and real["text"] == T.lowk_text({float(k): g for k, g
                                       in real["gain"].items()}),
      f"the tiny FNO's gains are numbers, in fno_train's text: "
      f"{real['text']}")
P.DIAG_LOWK_SEEDS = 2
check(quiet(P.lowk_check, Identity(), "cpu")["seeds"] == [11, 12],
      "DIAG_LOWK_SEEDS takes the first validation seeds")
P.DIAG_LOWK_SEEDS = 3
P.END_TIME = 8.0
check(list(quiet(P.lowk_check, Identity(), "cpu")["gain"]) == ["5", "7"],
      "the last start is END_TIME - DT_STEP")
P.END_TIME = 30.0
check(list(quiet(P.lowk_check, Identity(), "cpu")["gain"]) == ["5", "11"],
      "and the last frame of the reference when that ends sooner")
P.END_TIME, T.LOWK_T0 = 12.0, 50.0
check(list(quiet(P.lowk_check, Identity(), "cpu")["gain"]) == ["11"],
      "a start the reference does not hold is left out")
T.LOWK_T0 = 5.0
real_gains = T.lowk_gains
setattr(T, "lowk_gains", lambda *a, **k: {5.0: dict.fromkeys(T.LOWK_SHELLS,
                                                              123.0)})
try:
    liar = quiet(P.lowk_check, Identity(), "cpu")
finally:
    setattr(T, "lowk_gains", real_gains)
check(liar["gain"] == {"5": dict.fromkeys(T.LOWK_SHELLS, 123.0)},
      "the gains are fno_train.lowk_gains, not a copy of it")

print("3. diagnostics()")
before = {k: v.clone() for k, v in model.state_dict().items()}
rec, out = printed(P.diagnostics, model, "cpu")
check({"written", "model_path", "fingerprint", "tile", "lowk", "seconds"}
      <= set(rec) and rec["model_path"] == CKPT
      and rec["fingerprint"] == fno_paths.fingerprint(CKPT)
      and rec["seconds"] >= 0.0,
      f"the record names the model, its fingerprint and the time it took "
      f"({rec['seconds']} s)")
check(json.loads(json.dumps(rec)) == rec, "and is plain JSON")
check("tile invariance" in out and "low-k gain" in out
      and "NOT INVARIANT" not in out and "took" in out,
      "the summary says both")
after = model.state_dict()
check(all(torch.equal(before[k], after[k]) for k in before)
      and not model.training,
      "the model's weights and mode are as they were")
_, bad_out = printed(P.diagnostics, Positional(model), "cpu")
check("NOT INVARIANT" in bad_out, "a model that is not invariant is said so")


def boom(model, device):
    raise RuntimeError("no field")


real_tile = P.tile_invariance
setattr(P, "tile_invariance", boom)
try:
    partial, partial_out = printed(P.diagnostics, model, "cpu")
finally:
    setattr(P, "tile_invariance", real_tile)
check(partial["tile"] == {"error": "RuntimeError: no field"}
      and "gain" in partial["lowk"] and "failed" in partial_out,
      "a check that fails is recorded and reported, and the other still runs")

print("4. a run")
CALLS = []
real_diagnostics = P.diagnostics


def spy(model, device):
    CALLS.append(device)
    return real_diagnostics(model, device)


ROLLS = []


def rollout_case(model, device, off, lattice=None, seeds=None):
    """A case of real rollouts; its summary is in a file beside the others."""
    lat = P.LATTICE if lattice is None else lattice
    sd = P.seeds_for(off) if seeds is None else tuple(seeds)
    res = [P.evaluate_seed(model, s, device, off, lat) for s in sd]
    ROLLS.append(res)
    doc = {"seeds": [int(s) for s in sd],
           "r2": [r["r2"].tolist() for r in res],
           "snapshots": [[hashlib.sha256(np.ascontiguousarray(a).tobytes())
                          .hexdigest() for pair in r["snapshots"].values()
                          for a in pair] for r in res]}
    os.makedirs(P.RESULTS_DIR, exist_ok=True)
    P._write_json_atomic(os.path.join(P.RESULTS_DIR,
                                      f"case_{off:+.2f}_l{lat}.json"), doc)
    return None


def go(*flags, net=None):
    """P.main on a tiny critical run: (its exit, what it printed)."""
    reset()
    CALLS.clear()
    ROLLS.clear()
    net = model if net is None else net
    setattr(P, "load_checkpoint", lambda path, device=None: (
        net, {"best": {"epoch": 1, "score": 0.9, "survival": 12.0}}))
    setattr(P, "run_case", rollout_case)
    setattr(P, "diagnostics", spy)
    argv = ["--eval-mode", "critical", "--sizes", "32", "--seeds", "2",
            "--no-video", "--start-time", "5", "--end-time", "12", *flags]
    buf, code = io.StringIO(), None
    try:
        with contextlib.redirect_stdout(buf):
            code = P.main(argv)
    except SystemExit as exc:
        code = exc.code
    return code, buf.getvalue()


def files(folder):
    return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


def read(path):
    with open(path) as fh:
        return json.load(fh)


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


WRITES = []
real_write = P._write_json_atomic
setattr(P, "_write_json_atomic",
        lambda path, obj: (WRITES.append(path), real_write(path, obj))[1])
VIEWS_DIR = os.path.join(BOX, "views_on", "h{H:0>3}")
V8, V12 = (os.path.join(BOX, "views_on", f"h{h:03d}") for h in (8, 12))
# FNO's own outputs of a run are in <rollout cache>/extras/ (one folder per
# view when there are several), not in the Results folders
_, out_on = go("--views", "8,12", "--results-dir", VIEWS_DIR)
XROOT = os.path.join(P.RESULT_CACHE, P.EXTRAS_NAME)   # as main set it
X8, X12 = (os.path.join(XROOT, os.path.basename(v)) for v in (V8, V12))
check(len(CALLS) == 1, f"two views: the diagnostics ran once ({CALLS})")
check(all("diagnostics.json" in files(x) for x in (X8, X12))
      and not any("diagnostics.json" in files(v) for v in (V8, V12)),
      f"and diagnostics.json is in each view's extras folder, not in the "
      f"Results folders: {files(X8)[:4]}")
rec8, rec12 = read(os.path.join(X8, "diagnostics.json")), read(
    os.path.join(X12, "diagnostics.json"))
check(rec8 == rec12 and rec8["tile"]["invariant"]
      and rec8["fingerprint"] == fno_paths.fingerprint(CKPT)
      and rec8["lowk"]["lattice"] == 32 and "gain" in rec8["lowk"],
      "the same record in both, from this run's model")
check(sorted(w for w in WRITES if w.endswith("diagnostics.json"))
      == sorted(os.path.join(x, "diagnostics.json") for x in (X8, X12))
      and not [f for x in (X8, X12) for f in files(x) if f.endswith(".tmp")],
      "written with _write_json_atomic, and no partial file left")
check("diagnostics" in out_on and "diagnostics.json" in out_on,
      "and the log says where it went")

PLAIN = os.path.join(BOX, "plain_on")
go("--results-dir", PLAIN)
check(len(CALLS) == 1 and "diagnostics.json" in files(XROOT)
      and "diagnostics.json" not in files(PLAIN),
      "a run without views writes it in extras/ itself, not in its one "
      "results folder")
shutil.rmtree(XROOT)

OFF_DIR = os.path.join(BOX, "views_off", "h{H:0>3}")
O8, O12 = (os.path.join(BOX, "views_off", f"h{h:03d}") for h in (8, 12))
_, out_off = go("--views", "8,12", "--no-diagnostics", "--results-dir",
                OFF_DIR)
check(not CALLS and not any("diagnostics.json" in files(x) for x in (
        XROOT, os.path.join(XROOT, "h008"), os.path.join(XROOT, "h012")))
      and not any("diagnostics.json" in files(v) for v in (O8, O12))
      and "diagnostics" not in out_off.replace("DIAGNOSTICS", ""),
      "--no-diagnostics runs none and writes none")
check(P.parse_args(["--no-diagnostics"]).no_diagnostics
      and not P.parse_args([]).no_diagnostics
      and "no tile or low-k checks" in out_off,
      "the flag is parsed, and the log says it moved a setting")

REGEN = os.path.join(BOX, "regen")
reset()
CALLS.clear()
setattr(P, "diagnostics", spy)
P.RESULTS_DIR = REGEN
P.RESULT_CACHE = os.path.join(BOX, "regen_cache")
_, regen_out = printed(P.evaluate_all, model, {}, "cpu",
                       case_fn=lambda *a, **k: None, resume=False,
                       clean=False)
check(not CALLS and "diagnostics.json" not in files(REGEN)
      and not os.path.exists(os.path.join(XROOT, "diagnostics.json"))
      and "[diagnostics]" not in regen_out,
      "a regeneration (a case_fn) makes no diagnostics")

reset()
CALLS.clear()
P.RESULTS_DIR = os.path.join(BOX, "stand_in")
setattr(P, "diagnostics", spy)


class StandIn:
    config = {"backbone": "physical_k", "hard_mass_conservation": True,
              "residual_prediction": True, "mode_scaling": 1.0,
              "zero_dc": True}

    def n_parameters(self):
        return 0

    def parameters(self):
        yield types.SimpleNamespace(dtype="float32")


setattr(P, "run_case", lambda *a, **k: None)
printed(P.evaluate_all, StandIn(), {}, "cpu")
check(not CALLS, "a stand-in that is not a torch module has nothing to check")
setattr(P, "diagnostics", real_diagnostics)

print("5. nothing a rollout holds changes")
check(files(O12) and files(V12) == files(O12),
      f"a run with diagnostics writes the Results files one without does "
      f"(its diagnostics.json went to extras/): {files(O12)}")
case = "case_+0.00_l32.json"
check(case in files(V12) and sha(os.path.join(V12, case))
      == sha(os.path.join(O12, case)),
      "and its case summary (every R2 and snapshot of the rollouts) is "
      "byte for byte the same")
reset()
setattr(P, "diagnostics", real_diagnostics)
setattr(P, "SNAPSHOT_TIMES", (5.0, 8.0, 12.0))
P.WRITE_VIDEO = False
net = tiny_model()
first = quiet(P.evaluate_seed, net, 5, "cpu", 0.0, 32)
quiet(P.diagnostics, net, "cpu")
again = quiet(P.evaluate_seed, net, 5, "cpu", 0.0, 32)
check(np.array_equal(first["r2"], again["r2"])
      and np.array_equal(first["mass_drift"], again["mass_drift"])
      and all(np.array_equal(a, b) for t in first["snapshots"]
              for a, b in zip(first["snapshots"][t], again["snapshots"][t])),
      "a seed rolled out before the diagnostics and after them agrees to "
      "the bit")

print("6. --smoke, and the README")
reset()
quiet(P.smoke_cases)
check((P.DIAG_TILE, P.DIAG_LOWK_SEEDS, P.DIAG_LOWK_L)
      == (P.LATTICE, 1, P.LATTICE) == (32, 1, 32)
      and P.DIAG_LOWK_T_END == P.START_TIME == P.DIAG_TILE_T,
      "--smoke sets the diagnostics tiny: l = 32 tiled to 64, 1 seed")
FIXED_STEP["VAL_SEEDS"] = (11, 12, 13)
T.LOWK_T0 = 5.0
small, small_out = printed(P.diagnostics, tiny_model(), "cpu")
check(small["tile"]["tiled_l"] == 64 and small["tile"]["invariant"]
      and small["lowk"]["seeds"] == [11] and small["lowk"]["lattice"] == 32
      and not os.path.relpath(P.GT_CACHE, BOX).startswith(".."),
      f"and they run so, in the smoke root: tile {small['tile']['l']} -> "
      f"{small['tile']['tiled_l']}, seed(s) {small['lowk']['seeds']}")
reset()
rec = rec12
man = {"label": "x", "written": "now", "cases": {}, "diverge_abs": 5.0,
       "threshold": 0.8}
rep_path = os.path.join(BOX, "report_diag.txt")
quiet(S.write_report, rep_path, man, {}, rec)
with open(rep_path) as fh:
    text = fh.read()
quiet(S.write_report, rep_path, man, {}, None)
with open(rep_path) as fh:
    text_none = fh.read()
check("6  DIAGNOSTICS" in text and (f"tile invariance: l = {rec['tile']['l']} tiled to "
                                f"{rec['tile']['tiled_l']}") in text
      and "7  LOW-k INCREMENT RESPONSE" in text
      and rec["lowk"]["text"] in text
      and "no diagnostics.json for this run" in text_none,
      "report.txt reads the diagnostics into its sections 5 and 6, and says "
      "so when there is none")
old_rec = json.loads(json.dumps(rec))
for key in ("device", "dtype"):
    del old_rec["tile"][key]
quiet(S.write_report, rep_path, man, {}, old_rec)
with open(rep_path) as fh:
    text_old = fh.read()
check("float64 on cpu" in text and "float64" not in text_old
      and f"tiled to {rec['tile']['tiled_l']} at t = " in text_old,
      "and its tile line says the steps were float64 on the CPU (a "
      "diagnostics.json from before that says nothing, and still reads)")

print("7. a rebuild into a folder that holds only diagnostics.json")
cache = os.path.join(BOX, "empty_cache")
os.makedirs(cache)
solo, stray = (os.path.join(BOX, "build_solo"), os.path.join(BOX,
                                                              "build_stray"))
for folder, name in ((solo, "diagnostics.json"), (stray, "notes.txt")):
    os.makedirs(folder)
    with open(os.path.join(folder, name), "w") as fh:
        fh.write("{}")


def refused(folder):
    try:
        quiet(S.build, cache, folder)
    except SystemExit as exc:
        return "already holds files" in str(exc.code)
    except Exception:
        return False
    return False


check(not refused(solo) and refused(stray),
      "make_submission's rebuild accepts it, and still refuses a folder "
      "with other files and no manifest")
reset()

os.chdir(SCRATCH)
shutil.rmtree(BOX, ignore_errors=True)
new = sorted(project_state() - PROJECT)
check(not new, f"nothing was written in the project folder, Work/ or "
               f"Results/ {new[:3]}")
print("\n" + ("DIAG PASS" if ok else "DIAG FAIL"))
sys.exit(0 if ok else 1)
