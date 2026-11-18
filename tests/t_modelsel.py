"""Model and selection: the 38-channel size, the Laplacian-output head, the
averaged weights (EMA), survival selection, the low-wavenumber gain, the
flags the job scripts pass, the low-k fill of the resampled spectral weights,
the precision flags and the chemical-potential input channel.

The runs are tiny CPU runs of the real fno_train.main, each in a folder of
its own under tests/_scratch.
"""
import contextlib
import hashlib
import importlib.util
import io as _io
import json
import os
import shutil
import subprocess
import sys
import types
import warnings
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from _paths import REAL_MODEL, ROOT, SCRATCH

os.chdir(SCRATCH)
import fno_paths as F
import fno_train as T
MU_TRAINING_DEFAULT = T.MU_CHANNEL          # before --fixed-step's values go in
MU_CLIP_DEFAULT = T.MU_CLIP
FIXED_NOTES = T.use_fixed_step()

setattr(torch.cuda, "is_available", lambda: False)

D = os.path.join(SCRATCH, "modelsel_t")
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
sink = _io.StringIO()


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


def reset(**cfg):
    for k, v in {**DEFAULTS, **BASE, **cfg}.items():
        setattr(T, k, v)
    T.DEFAULT_RUN_META = DEFAULT_META
    T.MODEL_OUT = T.CHECKPOINT = None


def run(folder, argv=(), **cfg):
    """T.main in ``folder`` under the tiny settings and these overrides:
    (what it printed, the SystemExit code or the exception, or None)."""
    reset(**cfg)
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


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def epoch_lines(out):
    return [ln for ln in out.splitlines() if ln.startswith("epoch ")]


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def rolls(q):
    return (np.roll(q, 1, -2) + np.roll(q, -1, -2) + np.roll(q, 1, -1)
            + np.roll(q, -1, -1) - 4.0 * q)


print("1. the model has 38 channels, and an old checkpoint its own")
check(DEFAULTS["CHANNELS"] == 38, f"CHANNELS is {DEFAULTS['CHANNELS']}")
reset()
T.CHANNELS, T.MODES, T.LAYERS = (DEFAULTS["CHANNELS"], DEFAULTS["MODES"],
                                 DEFAULTS["LAYERS"])
full = T.build_model(n_ref=64, device="cpu")
check(17_000_000 < full.n_parameters() < 18_000_000
      and full.config["channels"] == 38,
      f"build_model gives {full.n_parameters():,} parameters (64 channels "
      f"were 50,394,241)")
del full
reset()
old = T.CahnHilliardFNO(channels=12, modes=6, n_layers=2, n_ref=16)
old_cfg = {k: v for k, v in old.config.items() if k != "laplacian_output"}
old.config = dict(old_cfg)
T.save_model(old, os.path.join(D, "old.pt"), epoch=3)
back, _ = T.load_checkpoint(os.path.join(D, "old.pt"))
check(back.config["channels"] == 12 and 38 != 12
      and back.config["laplacian_output"] is False,
      "a checkpoint with no laplacian_output key loads with its own size, "
      "as the base head")
check(T.model_drift(old_cfg, back.config) == {}
      and T.model_drift(old_cfg, dict(back.config, laplacian_output=True))
      == {"laplacian_output": (False, True)}
      and T.model_drift(old_cfg, dict(back.config, channels=13))
      == {"channels": (12, 13)},
      "and the resume comparison reads the missing key as the base head")

print("\n2. the Laplacian-output head")
head = T.CahnHilliardFNO(channels=8, modes=6, n_layers=2, n_ref=16,
                         laplacian_output=True).double()
out_conv = head.backbone.project[2]
assert isinstance(out_conv, nn.Conv2d)
with torch.no_grad():
    nn.init.normal_(out_conv.weight, std=0.3)
g = torch.Generator().manual_seed(1)
psi = torch.randn(3, 1, 16, 16, dtype=torch.float64, generator=g)
inc = (head(psi) - psi).detach()
feed = torch.cat([psi, psi.mean(dim=(-2, -1), keepdim=True).expand_as(psi)],
                 dim=1)
m_out = head.backbone(feed).detach().numpy()
want = rolls(m_out)
err = float(np.abs(inc.numpy() - (want - want.mean(axis=(-2, -1),
                                                    keepdims=True))).max())
mean = float(inc.mean(dim=(-2, -1)).abs().max())
check(err < 1e-12 and float(inc.abs().max()) > 1e-3,
      f"in float64 the increment is lap5 of the head's output ({err:.1e}), "
      f"and not a trivial zero")
check(mean < 1e-12, f"and its spatial mean is {mean:.1e}")
flat = T.CahnHilliardFNO(channels=8, modes=6, n_layers=2, n_ref=16).double()
flat.load_state_dict(head.state_dict())
other_inc = (flat(psi) - psi).detach().numpy()
check(flat.config["laplacian_output"] is False
      and np.abs(other_inc - inc.numpy()).max() > 1e-3,
      "the same weights with the key off give another increment")
p = np.random.default_rng(5).standard_normal((2, 16, 16))
want_step = T.ch_step(torch.as_tensor(p)).numpy() - p
check(np.abs(rolls(p ** 3 - p - rolls(p)) * T.DT - want_step).max() < 1e-12,
      "lap5 is the stencil of ch_step (DX = 1)")
try:
    T.CahnHilliardFNO(channels=8, modes=6, n_layers=2, n_ref=16,
                      residual_prediction=False, laplacian_output=True)
    err_msg = None
except ValueError as exc:
    err_msg = str(exc)
check(err_msg is not None and "increment" in err_msg,
      "a head that predicts the whole next frame cannot be a Laplacian")

print("\n   the switches")
reset()
with contextlib.redirect_stdout(sink):
    moved = T.apply_overrides(T.parse_args([]))
check(moved == [] and (T.LAPLACIAN_OUTPUT, T.EMA, T.SELECT)
      == (True, False, "score"),
      "no flag: nothing changes, and the file's defaults are the Laplacian "
      "head, no average, score")
with contextlib.redirect_stdout(sink):
    moved = T.apply_overrides(T.parse_args(
        ["--variant", "base", "--ema", "--select", "survival"]))
check(T.LAPLACIAN_OUTPUT is False and T.EMA is True
      and T.SELECT == "survival" and len(moved) == 3
      and any("LAPLACIAN_OUTPUT True -> False" in m for m in moved),
      "--variant base, --ema and --select survival set the constants")
with contextlib.redirect_stdout(sink):
    T.apply_overrides(T.parse_args(["--variant", "lap"]))
check(T.LAPLACIAN_OUTPUT is True and T.EMA is True,
      "--variant lap is still accepted, turns the head on and leaves the "
      "rest")
bad = _io.StringIO()
with contextlib.redirect_stderr(bad):
    try:
        T.parse_args(["--variant", "fancy"])
        refused = False
    except SystemExit:
        refused = True
check(refused, "another variant is refused")

print("\n   the label, the refusal, and a tiny run")
reset(T_TRAIN=1000.0, ROOT=os.path.join(D, "root"))
with contextlib.redirect_stdout(sink):
    default = T.resolve_outputs(NF)
    T.MODEL_OUT = T.CHECKPOINT = None
    T.LAPLACIAN_OUTPUT = False
    plain = T.resolve_outputs(NF)
    T.MODEL_OUT = T.CHECKPOINT = None
    T.LAPLACIAN_OUTPUT = True
    label = T.resolve_outputs(NF)
check(default == "train_t1000_lap" and plain == "train_t1000"
      and label == "train_t1000_lap"
      and label == F.train_label(1000.0, lap=True)
      and os.path.join("train_t1000_lap", "fno_ch.pt") in str(T.MODEL_OUT)
      and os.path.join("train_t1000_lap", "fno_ch_last.pt")
      in str(T.CHECKPOINT),
      f"the folder carries _lap: {label}, {T.MODEL_OUT}")
reset(LAPLACIAN_OUTPUT=True, HARD_CONSTRAINTS=False)
try:
    T.check_variant()
    why = None
except SystemExit as exc:
    why = str(exc)
check(why is not None and "HARD_CONSTRAINTS" in why
      and "--variant base" in why,
      "lap with HARD_CONSTRAINTS False is refused, with what to do instead")
reset(LAPLACIAN_OUTPUT=False, HARD_CONSTRAINTS=False)
T.check_variant()
reset(SELECT="best")
try:
    T.check_variant()
    why = None
except SystemExit as exc:
    why = str(exc)
check(why is not None and "SELECT" in why, "an unknown selection is refused")
out, code = run("nohard", ["--variant", "lap"], HARD_CONSTRAINTS=False)
check(isinstance(code, str) and "HARD_CONSTRAINTS" in code
      and not os.path.exists(path("nohard", "fno_ch.pt")),
      "and main stops with it before it trains anything")

out, code = run("nohard_auto", [], HARD_CONSTRAINTS=False)
auto = [ln for ln in out.splitlines() if "no-hard-constraints" in ln]
nh = load(path("nohard_auto", "fno_ch.pt")) if code is None else {}
check(code is None and len(epoch_lines(out)) == 2 and len(auto) == 1
      and auto[0].startswith("[cli] LAPLACIAN_OUTPUT True -> False")
      and nh["config"]["laplacian_output"] is False,
      "the ablation takes the base head by itself, with one printed line, "
      "instead of refusing")
out, code = run("nohard_base", ["--variant", "base"], HARD_CONSTRAINTS=False)
check(code is None and "no-hard-constraints" not in out,
      "and says nothing when --variant base was asked for already")

print("\n   --no-hard-constraints and --mass-weight")


def switch(argv):
    """apply_overrides on these flags from the defaults: (the log lines,
    the SystemExit text or None)."""
    reset(T_TRAIN=500.0)
    try:
        with contextlib.redirect_stdout(sink):
            moved = T.apply_overrides(T.parse_args(argv))
            T.check_variant()
        return moved, None
    except SystemExit as exc:
        return [], str(exc)


moved, why = switch(["--no-hard-constraints"])
check(why is None and (T.HARD_CONSTRAINTS, T.ZERO_DC,
                       T.HARD_MASS_CONSERVATION) == (False, False, False)
      and T.LAPLACIAN_OUTPUT is False and T.MASS_WEIGHT == 0.1
      and T.label_for(500.0) == "train_t0500_nohard"
      and sum("HARD_CONSTRAINTS True -> False" in m for m in moved) == 1,
      "--no-hard-constraints sets the three constants, takes the plain head, "
      "keeps the penalty 0.1, gives train_t0500_nohard, one [cli] line")
moved, why = switch(["--variant", "base", "--no-hard-constraints",
                     "--mass-weight", "0", "--tag", "free"])
check(why is None and T.MASS_WEIGHT == 0.0
      and T.label_for(500.0) == "train_t0500_nohard_free"
      and any("MASS_WEIGHT 0.1 -> 0" in m for m in moved),
      "--mass-weight 0 --tag free: the penalty is 0 and the label "
      "train_t0500_nohard_free")
moved, why = switch(["--mass-weight", "0.3"])
check(why is not None and "--no-hard-constraints" in why
      and T.MASS_WEIGHT == 0.1 and T.HARD_CONSTRAINTS is True,
      "--mass-weight without --no-hard-constraints is refused, and "
      "nothing changes")
moved, why = switch(["--variant", "lap", "--no-hard-constraints"])
check(why is not None and "--variant lap" in why,
      "--variant lap with --no-hard-constraints is refused")
bad = _io.StringIO()
with contextlib.redirect_stderr(bad):
    try:
        T.parse_args(["--no-hard-constraints", "--mass-weight", "-1"])
        refused = False
    except SystemExit:
        refused = True
check(refused, "a negative --mass-weight is refused")
out, code = run("nohard_free", ["--no-hard-constraints", "--mass-weight",
                                "0.25", "--variant", "base"])
nf = load(path("nohard_free", "fno_ch.pt")) if code is None else {}
nfs = load(path("nohard_free", "fno_ch_last.pt")) if code is None else {}
ncfg = nf.get("config", {})
check(code is None and len(epoch_lines(out)) == 2
      and (ncfg.get("laplacian_output"), ncfg.get("zero_dc"),
           ncfg.get("hard_mass_conservation")) == (False, False, False)
      and nf["train_config"]["constants"]["MASS_WEIGHT"] == 0.25
      and nf["train_config"]["constants"]["HARD_CONSTRAINTS"] is False
      and nfs["train_config"]["schedule"]["MASS_WEIGHT"] == 0.25,
      "a tiny run: the checkpoint config says laplacian_output, zero_dc and "
      "hard_mass_conservation False, and the run facts hold MASS_WEIGHT 0.25")
nm, _ = T.load_checkpoint(path("nohard_free", "fno_ch.pt"))
check(nm.hard_mass_conservation is False and nm.config["zero_dc"] is False,
      "and the checkpoint loader builds a model with both off")
out, code = run("nohard_free", ["--no-hard-constraints", "--mass-weight",
                                "0.5", "--variant", "base"], RESUME=True)
check(isinstance(code, str) and "MASS_WEIGHT" in code,
      "a resume with another weight is caught by the schedule stamp")

print("\n   --no-mu-channel")


def switch_mu(argv, **cfg):
    """apply_overrides on these flags with the channel on, as a real
    training has it: (the log lines, the SystemExit text or None)."""
    reset(T_TRAIN=500.0, MU_CHANNEL=True, **cfg)
    try:
        with contextlib.redirect_stdout(sink):
            moved = T.apply_overrides(T.parse_args(argv))
            T.check_variant()
        return moved, None
    except SystemExit as exc:
        return [], str(exc)


moved, why = switch_mu(["--no-mu-channel"])
check(why is not None and "--tag" in why and T.MU_CHANNEL is True,
      "--no-mu-channel without --tag is refused, and nothing changes")
moved, why = switch_mu(["--no-mu-channel", "--tag", "nomu"])
check(why is None and T.MU_CHANNEL is False and T.LAPLACIAN_OUTPUT is True
      and T.HARD_CONSTRAINTS is True
      and T.label_for(500.0) == "train_t0500_lap_nomu"
      and sum("MU_CHANNEL True -> False" in m for m in moved) == 1
      and len(moved) == 2,
      "--no-mu-channel --tag nomu: the channel is off, nothing else moves, "
      "the folder is train_t0500_lap_nomu, one [cli] line (and the tag's)")
moved, why = switch_mu(["--variant", "base", "--no-hard-constraints",
                        "--no-mu-channel", "--tag", "nomu"])
check(why is None and T.MU_CHANNEL is False and T.HARD_CONSTRAINTS is False
      and T.label_for(500.0) == "train_t0500_nohard_nomu",
      "with --no-hard-constraints it is accepted: train_t0500_nohard_nomu")
moved, why = switch_mu([])
check(why is None and T.MU_CHANNEL is True and moved == [],
      "without the flag the channel stays on")
out, code = run("mu_off", ["--no-mu-channel", "--tag", "nomu"],
                MU_CHANNEL=True, EPOCHS=1, ROOT=os.path.join(D, "root_nomu"))
mu_best = os.path.join(D, "root_nomu", "models", "train_t0005_lap_nomu",
                       "fno_ch.pt")
if code is None and os.path.isfile(mu_best):
    nomu = load(mu_best)
else:
    nomu = {"config": {}, "train_config": {}, "state_dict": {}}
    print(code, out[-600:])
check(code is None and nomu["config"].get("mu_channel") is False
      and nomu["config"].get("mu_clip") is None
      and nomu["state_dict"]["backbone.lift.0.weight"].shape[1] == 2
      and nomu["train_config"]["constants"]["MU_CHANNEL"] is False
      and "[cli] MU_CHANNEL True -> False" in out and epoch_lines(out)
      and "clamped to" not in out,
      "a tiny run: the model has no mu channel and no clamp, its constants "
      "say MU_CHANNEL False, and nothing is written outside its own folder")
out, code = run("mu_off_hard", ["--no-mu-channel", "--tag", "nomu",
                                "--no-hard-constraints", "--variant", "base"],
                MU_CHANNEL=True, EPOCHS=1,
                ROOT=os.path.join(D, "root_nomu_hard"))
hb = os.path.join(D, "root_nomu_hard", "models",
                  "train_t0005_nohard_nomu", "fno_ch.pt")
hard = load(hb) if code is None and os.path.isfile(hb) else {"config": {}}
check(code is None and hard["config"].get("mu_channel") is False
      and hard["config"].get("zero_dc") is False
      and hard["config"].get("hard_mass_conservation") is False
      and epoch_lines(out),
      "--no-mu-channel with --no-hard-constraints also trains and saves "
      "both off")

out, code = run("lap", [])
ck = load(path("lap", "fno_ch.pt"))
st = load(path("lap", "fno_ch_last.pt"))
with open(path("lap", "fno_ch_config.json")) as fh:
    cfg_json = json.load(fh)
check(code is None and len(epoch_lines(out)) == 2
      and ck["config"]["laplacian_output"] is True
      and st["config"]["laplacian_output"] is True
      and cfg_json["constants"]["LAPLACIAN_OUTPUT"] is True
      and "[cli]" not in out,
      "a tiny run with no flag trains, and the checkpoints and the config "
      "say lap")
lapped, _ = T.load_checkpoint(path("lap", "fno_ch.pt"))
x32 = torch.as_tensor(traj[:2, 7]).float()[:, None]
step = (lapped(x32) - x32).detach()
check(lapped.config["laplacian_output"] is True
      and float(step.mean(dim=(-2, -1)).abs().max()) < 1e-6,
      "the saved model loads as lap and conserves mass")
out, code = run("lap", ["--variant", "lap"], RESUME=True)
check(code is None and "at epoch 2" in out,
      "a lap state resumes as lap")
out, code = run("lap", ["--variant", "base"], RESUME=True)
base_ck = load(path("lap", "fno_ch_last.pt"))
check(isinstance(code, str) and "laplacian_output: True -> False" in code
      and "different model" in code
      and base_ck["config"]["laplacian_output"] is True,
      "and a base run refuses to resume it: the model differs, and the lap "
      "state is left as it was")
base_ck["config"].pop("laplacian_output")
torch.save(base_ck, path("lap", "fno_ch_last.pt"))
out, code = run("lap", ["--variant", "base"], RESUME=True)
check(code is None and "different model" not in out and "at epoch 2" in out,
      "a state from before the key existed resumes as base")
out, code = run("lap", RESUME=True)
check(isinstance(code, str) and "laplacian_output: False -> True" in code
      and "--variant base" in code,
      "and the default, the Laplacian head, refuses it and names the flag "
      "that goes on with it")

print("\n3. averaged weights (EMA)")
check(DEFAULTS["EMA"] is False and DEFAULTS["EMA_DECAY"] == 0.999,
      "off by default, decay 0.999")
reset()
check(T.ema_decay(0) == 0.1 and abs(T.ema_decay(9) - 10 / 19) < 1e-15
      and abs(T.ema_decay(90) - 91 / 100) < 1e-15
      and T.ema_decay(10 ** 6) == 0.999 and T.ema_decay(20000) == 0.999,
      "the decay is (1+n)/(10+n) while that is below 0.999, then 0.999")
T.EMA_DECAY = 0.9
check(T.ema_decay(10 ** 6) == 0.9 and T.ema_decay(0) == 0.1,
      "and never above EMA_DECAY")
reset()
net = T.CahnHilliardFNO(channels=8, modes=6, n_layers=2, n_ref=16).double()
ema = T.EmaWeights(net)
start = {k: v.clone() for k, v in ema.shadow.items()}
check(ema.n == 0 and same(start, {k: v for k, v in net.state_dict().items()
                                  if k in start}) and len(start) > 5,
      "the average starts as the weights")
with torch.no_grad():
    for prm in net.parameters():
        prm.add_(0.5)
ema.update(net)
ema.update(net)
d0, d1 = T.ema_decay(0), T.ema_decay(1)
sd = net.state_dict()
want = {k: (v * d0 + sd[k] * (1 - d0)) * d1 + sd[k] * (1 - d1)
        for k, v in start.items()}
check(ema.n == 2 and all(torch.allclose(ema.shadow[k], want[k], atol=1e-14)
                         for k in want),
      "two updates blend the weights in with those decays")
raw = {k: v.clone() for k, v in net.state_dict().items()}
net.eval()
x16 = torch.randn(2, 1, 32, 32, dtype=torch.float64, generator=g)
before = net(x16).detach()
inside = None
with ema.swapped(net):
    held = {k: v.clone() for k, v in net.state_dict().items()}
    inside = net(x16).detach()
twin = T.CahnHilliardFNO(channels=8, modes=6, n_layers=2, n_ref=16).double()
twin.load_state_dict(held)
twin.eval()
check(all(torch.equal(held[k], ema.shadow[k]) for k in ema.shadow)
      and same(net.state_dict(), raw) and torch.equal(net(x16), before),
      "inside swapped() the model holds the average, afterwards its own "
      "weights again, bit for bit")
check(torch.equal(inside, twin(x16).detach())
      and not torch.equal(inside, before),
      "and a lattice other than n_ref (resampled weights) sees the average, "
      "not a stale cached copy")
ema.reset(net)
check(ema.n == 0 and all(torch.equal(ema.shadow[k], net.state_dict()[k])
                         for k in ema.shadow),
      "reset() starts the average again from the model's weights")
ema.update(net)
other = T.EmaWeights(net)
other.load(ema.shadow, 77)
check(other.n == 77 and same(other.shadow, ema.shadow),
      "load() takes the saved average and its count")

print("\n   in a run")
real_ema = T.EmaWeights
made = []


class Spy(T.EmaWeights):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        made.append(self)


seen = {}


def spy_curve(model, *a, **k):
    seen["held"] = {kk: v.clone() for kk, v in model.state_dict().items()}
    return real_curve(model, *a, **k)


real_curve = T.rollout_curve
setattr(T, "rollout_curve", spy_curve)
real_monitor = T.holdout_monitor
batches = []


def spy_monitor(*a, **k):
    batches[:] = real_monitor(*a, **k)
    return batches


setattr(T, "holdout_monitor", spy_monitor)
out, code = run("ema", ["--ema"], EPOCHS=1, LR=0.05)
ck = load(path("ema", "fno_ch.pt"))
st = load(path("ema", "fno_ch_last.pt"))
shadow = st["ema"]
check(code is None and st["ema_n"] == 3
      and set(shadow) <= set(st["state_dict"])
      and any(not torch.equal(shadow[k], st["state_dict"][k]) for k in shadow),
      "the state keeps the raw weights, the average and its count of 3")
check(all(torch.equal(ck["state_dict"][k], shadow[k]) for k in shadow)
      and ck["train_config"]["constants"]["EMA"] is True,
      "the best model saved is the average, not the raw weights")
check(all(torch.equal(seen["held"][k], shadow[k]) for k in shadow),
      "the rollout validation ran on the average")
avg_net = T.build_model(n_ref=L, device="cpu")
avg_net.load_state_dict(ck["state_dict"])
raw_net = T.build_model(n_ref=L, device="cpu")
raw_net.load_state_dict(st["state_dict"])


def val1(net_):
    net_.eval()
    with torch.no_grad():
        return float(np.mean([float(T.loss_fn(net_(x), y))
                              for x, y in batches]))


printed = float(epoch_lines(out)[0].split("val1")[1].split("|")[0])
check(abs(printed - val1(avg_net)) < 6e-6
      and abs(printed - val1(raw_net)) > 1e-4,
      f"and so did the monitor: val1 {printed:.5f} is the average's "
      f"({val1(avg_net):.5f}), not the raw weights' ({val1(raw_net):.5f})")
setattr(T, "rollout_curve", real_curve)
setattr(T, "holdout_monitor", real_monitor)

out, code = run("noema", EPOCHS=1)
ck = load(path("noema", "fno_ch.pt"))
st = load(path("noema", "fno_ch_last.pt"))
check(code is None and "ema" not in st and same(ck["state_dict"],
                                                st["state_dict"])
      and "EMA_DECAY" not in st["schedule"] and st["schedule"]["EMA"] is False,
      "without --ema there is no average, the best is the raw weights, and "
      "the stamp has no decay")

print("\n   the stamp and a resume")
out, code = run("ema2", ["--ema"])
st2 = load(path("ema2", "fno_ch_last.pt"))
check(code is None and st2["schedule"]["EMA"] is True
      and st2["schedule"]["EMA_DECAY"] == 0.999 and st2["ema_n"] == 6,
      "with --ema the stamp holds EMA and EMA_DECAY, and 6 steps were counted")
out, code = run("ema2", ["--ema"], RESUME=True, EMA_DECAY=0.99)
check(isinstance(code, str) and "  EMA_DECAY: 0.999 -> 0.99" in code,
      "a resume with another decay is refused")
out, code = run("ema2", RESUME=True)
check(isinstance(code, str) and "  EMA: True -> False" in code,
      "and so is one with the average switched off")
sentinel = load(path("ema2", "fno_ch_last.pt"))
sentinel["epoch"] = 1
sentinel["ema"] = {k: torch.zeros_like(v) for k, v in sentinel["ema"].items()}
sentinel["ema_n"] = 10 ** 6
torch.save(sentinel, path("ema2", "fno_ch_last.pt"))
out, code = run("ema2", ["--ema"], RESUME=True)
st3 = load(path("ema2", "fno_ch_last.pt"))
top = max(float(v.abs().max()) for v in st3["state_dict"].values())
low = max(float(v.abs().max()) for v in st3["ema"].values())
check(code is None and "at epoch 1" in out and "holds no averaged" not in out
      and st3["ema_n"] == 10 ** 6 + 3 and low < 0.01 * top,
      f"a resume picks the saved average up where it was: count "
      f"{st3['ema_n']}, size {low:.1e} against the weights' {top:.1e}")
plain = load(path("noema", "fno_ch_last.pt"))
plain["epoch"] = 0
for key in ("EMA", "SELECT"):
    plain["schedule"].pop(key)
torch.save(plain, path("noema", "fno_ch_last.pt"))
out, code = run("noema", ["--ema"], EPOCHS=1, RESUME=True)
st4 = load(path("noema", "fno_ch_last.pt"))
check(code is None and "holds no averaged weights" in out
      and st4["ema_n"] == 3,
      "a state from before the average, with no stamp for it, resumes and "
      "starts the average from its weights")

print("\n   a rollback")
made.clear()
setattr(T, "EmaWeights", Spy)
out, code = run("roll", ["--ema"], EPOCHS=3, DIVERGE_FACTOR=1e-9)
setattr(T, "EmaWeights", real_ema)
st = load(path("roll", "fno_ch_last.pt"))
check(code is None and out.count("rolled back to epoch 1") == 2
      and st["epoch"] == 1 and len(made) == 1,
      "two epochs diverge and roll back to epoch 1")
check(made[0].n == 0 and all(torch.equal(v, st["state_dict"][k])
                             for k, v in made[0].shadow.items()),
      "and the average was started again from the restored weights")

print("\n4. survival selection")
check(DEFAULTS["SELECT"] == "score" and DEFAULTS["SURVIVAL_MIN_DELTA"] == 10.0,
      "score by default, a minimum gain of 10 time units")
t4 = np.array([0.0, 1.0, 2.0, 3.0])
check(abs(T.interp_survival(t4, [1.0, 0.9, 0.7, 0.5], 0.8, 3.0) - 1.5) < 1e-12
      and abs(T.interp_survival(t4, [1.0, 0.9, 0.8, 0.7], 0.8, 3.0) - 2.0)
      < 1e-12,
      "the first crossing is interpolated between the two steps, and a curve "
      "that only touches the threshold survives to that step")
check(T.interp_survival(t4, [1.0, 0.9, 0.85, 0.81], 0.8, 99.0) == 99.0
      and T.interp_survival(t4, [1.0, 0.9, 0.7, 0.9], 0.8, 99.0)
      < 2.0,
      "a seed that stays above is cut off by the window end; a later "
      "recovery does not matter")
t100 = np.array([100.0, 101.0, 102.0])
check(abs(T.interp_survival(t100, [1.0, 0.4, 0.3], 0.8, 102.0)
          - (100 + 1 / 3)) < 1e-12,
      "a fall at the first step is interpolated from the start, where R2 is 1")
check(T.interp_survival(t4, [1.0, np.nan, 0.9, 0.9], 0.8, 3.0) == 1.0
      and T.interp_survival(t4, [1.0, 0.9, -np.inf, 0.9], 0.8, 3.0) == 2.0,
      "an R2 that is not a number fails at its own step")
r2 = [1.0, 0.5, 1.2, -0.3, 0.9, np.nan]
t6 = np.arange(6.0)
check(abs(T.in_range_accuracy(t6, r2, 4.0) - 0.6) < 1e-12
      and abs(T.in_range_accuracy(t6, r2, 5.0) - 0.48) < 1e-12
      and T.in_range_accuracy(t6, r2, 0.5) == 0.0,
      "the in-range accuracy clips R2 to [0, 1], counts a non-number as 0, "
      "skips the start and stops at the limit")
rule = T.better_epoch
check(rule(120, 0.1, 100, 0.9, 10) and rule(105, 0.9, 100, 0.5, 10)
      and rule(95, 0.9, 100, 0.5, 10) and rule(110, 0.9, 100, 0.5, 10),
      "a clear gain in survival wins whatever the accuracy; within the "
      "minimum gain, a higher accuracy wins")
check(not rule(105, 0.5, 100, 0.9, 10) and not rule(105, 0.9, 100, 0.9, 10)
      and not rule(85, 0.99, 100, 0.5, 10)
      and not rule(110, 0.4, 100, 0.5, 10),
      "within it, a lower or equal accuracy loses; far below the best loses "
      "whatever the accuracy; exactly the minimum gain is not more than it")
walk, acc = [1000.0], 0.5
best_s, best_a, peak = 1000.0, 0.5, 1000.0
for s in (991.0, 982.0, 973.0):
    acc += 0.01
    if rule(s, acc, best_s, best_a, 10, peak):
        best_s, best_a, peak = s, acc, max(peak, s)
        walk.append(s)
check(walk == [1000.0, 991.0] and rule(991, 0.9, 991, 0.5, 10)
      and not rule(991, 0.9, 991, 0.5, 10, 1005.0),
      f"the band is measured from the longest survival a best has had, so "
      f"slightly worse but more accurate epochs cannot walk the best down "
      f"({walk})")

print("\n   against the score the old code gave")
spec = importlib.util.spec_from_file_location(
    "sel_old", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fixtures", "fno_train_pre_health.py"))
assert spec is not None and spec.loader is not None
OLD = importlib.util.module_from_spec(spec)
sys.modules["sel_old"] = OLD
spec.loader.exec_module(OLD)
r = np.random.default_rng(3)
a_f, b_f = r.standard_normal((2, 16, 16)), 0.05 * r.standard_normal((16, 16))
frames = np.stack([a_f[0] + j * b_f for j in range(26)])
times = 100.0 + np.arange(26.0)
ident = T.CahnHilliardFNO(channels=8, modes=6, n_layers=2, n_ref=16)
now = T.rollout_score(ident, times, frames, "cpu", t_start=100.0,
                      t_end=125.0)
then = OLD.rollout_score(ident, times, frames, "cpu", t_start=100.0,
                         t_end=125.0)
t_c, r_c = T.rollout_curve(ident, times, frames, "cpu", t_start=100.0,
                           t_end=125.0)
check(now == then and 100.0 < now[1] < 125.0 and 0.0 < now[0] < 1.0,
      f"rollout_score gives what it gave before: {now}")
si = T.interp_survival(t_c, r_c, 0.8, 125.0)
check(len(t_c) == 26 and t_c[0] == 100.0 and r_c[0] == 1.0
      and now[1] - 1.0 < si <= now[1],
      f"the curve starts at R2 = 1 and the interpolated survival {si:.2f} "
      f"lies in the step before the first one below 0.8 ({now[1]:g})")

print("\n   in a run")
rules = []
real_rule = T.better_epoch


def spy_rule(*a):
    rules.append(a)
    return real_rule(*a)


setattr(T, "better_epoch", spy_rule)
cfg = dict(T_TRAIN=52.0, VAL_T_END=110.0, MONITOR_T_END=110.0)
out, code = run("surv", ["--select", "survival"], **cfg)
ck = load(path("surv", "fno_ch.pt"))
st = load(path("surv", "fno_ch_last.pt"))
lines = epoch_lines(out)
with open(path("surv", "fno_ch_history.json")) as fh:
    hist = json.load(fh)
check(code is None and len(rules) == 2
      and rules[0][2:] == (-1.0, -1.0, 10.0, None),
      "SELECT survival asks the rule at every validation, with the minimum "
      "gain of 10")
check(all("interpolated survival" in ln and "in-range R2" in ln
          and "(t <= 104)" in ln for ln in lines)
      and "<- best" in lines[0] and "in-range R2" in out.split("[done]")[1],
      "both numbers are logged, with the range it is judged over (2 x the "
      "training horizon), and the end line gives them")
check(ck["best"]["accuracy"] >= 0.0
      and ck["best"]["epoch"] in (1, 2)
      and abs(ck["best"]["survival"]
              - float(np.mean(ck["best"]["per_seed"]))) < 1e-9
      and hist[0]["survival_interp"] > 100.0
      and 0.0 <= hist[0]["in_range"] <= 1.0
      and st["schedule"]["SELECT"] == "survival",
      "the best carries the interpolated survival and the accuracy, the "
      "history keeps them, the stamp has SELECT")
out, code = run("surv", RESUME=True, **cfg)
check(isinstance(code, str) and "  SELECT: survival -> score" in code,
      "a resume under the other rule is refused")
rules.clear()
out, code = run("score", **cfg)
check(code is None and not rules
      and all("interpolated" not in ln for ln in epoch_lines(out)),
      "the score rule does not ask it, and its lines are as before")
force = {"yes": True}
setattr(T, "better_epoch", lambda *a: force["yes"])
out, code = run("force", ["--select", "survival"], **cfg)
check(out.count("<- best") == 2, "the rule's answer decides what is saved")
force["yes"] = False
out, code = run("noforce", ["--select", "survival"], **cfg)
check(out.count("<- best") == 0
      and not os.path.exists(path("noforce", "fno_ch.pt")),
      "a validation that does not win saves nothing")
setattr(T, "better_epoch", real_rule)

print("\n5. the low-wavenumber gain")
masks = T.lowk_masks(32, 2.0)
union = np.sum([m.astype(int) for m in masks.values()], axis=0)
check(union.max() == 1 and union.sum() == 32 * 32 - 1 and not union[0, 0]
      and masks["below"].sum() == 8 and masks["first"].sum() == 16,
      "the four shells at a 2x lattice share every mode but n = 0; 8 modes "
      "lie below the training box's fundamental")
check(not T.lowk_masks(16, 1.0)["below"].any()
      and T.lowk_masks(16, 1.0)["first"].sum() == 8,
      "at the training lattice there is nothing below it, and the first "
      "shell starts at |n| = 1")
check(T.lowk_times(np.arange(11.0)) == [9.0]
      and T.lowk_times(np.arange(2001.0)) == [1000.0, 1999.0]
      and T.lowk_times(np.arange(1002.0)) == [1000.0]
      and T.lowk_times(np.arange(1001.0)) == [999.0]
      and T.lowk_times(np.arange(5000.0)) == [1000.0, 4998.0],
      "the start times are 1000 and the last frame with a next one (VAL_T_END "
      "- DT_STEP), only those the references cover")
rr = np.random.default_rng(9)
syn = [(np.arange(6.0), rr.standard_normal((6, 32, 32))) for _ in range(3)]
table = {fr[j].tobytes(): fr[j + 1] for _, fr in syn for j in range(5)}


class Step(nn.Module):
    """x plus alpha times the true change, which it looks up."""

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.anchor = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, x):
        nxt = np.stack([table[xi[0].numpy().tobytes()] for xi in x.detach()])
        nxt_t = torch.as_tensor(nxt)[:, None]
        return x + self.alpha * (nxt_t - x)


got = {}
for alpha in (0.0, 0.5, 1.0, 2.0):
    got[alpha] = T.lowk_gains(Step(alpha), syn, [2.0, 4.0], 2.0, "cpu")
check(all(abs(v) < 1e-12 for t0 in got[0.0].values() for v in t0.values()),
      "a model that leaves psi alone has gain 0 in every shell")
check(all(abs(v - 1.0) < 1e-12 for t0 in got[1.0].values()
          for v in t0.values()),
      "a model that returns the true next frame has gain 1 in every shell")
check(all(abs(v - 0.5) < 1e-12 for v in got[0.5][2.0].values())
      and all(abs(v - 2.0) < 1e-12 for v in got[2.0][4.0].values())
      and list(got[1.0]) == [2.0, 4.0] and len(got[1.0][2.0]) == 4,
      "half the true change gives 0.5 and twice gives 2, at both times")
same_size = T.lowk_gains(Step(1.0), syn, [2.0], 1.0, "cpu")[2.0]
check(same_size["below"] is None and abs(same_size["first"] - 1.0) < 1e-12,
      "with no modes below the training box that shell has no gain (None)")
flat_t = [(np.arange(3.0), np.zeros((3, 16, 16)))]
table.update({np.zeros((16, 16)).tobytes(): np.zeros((16, 16))})
quiet = T.lowk_gains(Step(1.0), flat_t, [1.0], 1.0, "cpu")
check(all(v is None for v in quiet[1.0].values()),
      "and a state that does not change at all has none anywhere")
check(T.lowk_text({1000.0: {"below": 0.8, "first": 1.0, "mid": 1.014,
                            "high": None}}) == "0.80/1.00/1.01/-@1000"
      and T.lowk_text({4.0: dict(below=None, first=0.5, mid=0.5, high=2.0),
                       9.0: dict(below=1, first=1, mid=1, high=1)})
      == "-/0.50/0.50/2.00@4 1.00/1.00/1.00/1.00@9",
      "the log shows below/first/mid/high at each start time")

print("\n   in a run")
real_gain = T.lowk_gains
out, code = run("lowk", ["--select", "survival"], **cfg)
with open(path("lowk", "fno_ch_history.json")) as fh:
    hist = json.load(fh)
lk = hist[0]["lowk"]
check(code is None and all("| lowk g " in ln for ln in epoch_lines(out))
      and list(lk) == ["109"] and set(lk["109"]) == set(T.LOWK_SHELLS)
      and lk["109"]["below"] is None and lk["109"]["first"] is not None
      and T.lowk_text({109.0: lk["109"]}) in out,
      f"every validation prints it and the history keeps it ({lk})")


def liar(*a, **k):
    return {109.0: {s: 123.0 for s in T.LOWK_SHELLS}}


setattr(T, "lowk_gains", liar)
out2, code2 = run("lowk2", ["--select", "survival"], **cfg)
setattr(T, "lowk_gains", real_gain)


def verdict(text):
    return [(ln.split("| rollout")[1].split("| lowk")[0], "<- best" in ln)
            for ln in epoch_lines(text)]


check(code2 is None and "lowk g 123.00" in out2
      and verdict(out) == verdict(out2) and len(verdict(out)) == 2,
      "and what it says changes neither the scores nor which epoch is best")

print("\n6. the job scripts pass their variables on: tested in t_protocol,"
      " section 5")

reset()
with contextlib.redirect_stdout(sink):
    T.apply_overrides(T.parse_args(
        ["--dataset", "Work/data/ch_l64_T2000_f32.npy", "--t-train", "1000",
         "--variant", "lap", "--ema"]))
    lab = T.resolve_outputs(NF)
check(lab == "train_t1000_lap" and T.EMA and T.LAPLACIAN_OUTPUT,
      f"and the flags of that job give the label {lab}")

print("\n7. variants that share a folder")


def snapshot(folder):
    """{file name: sha} of what a folder holds."""
    top = os.path.join(D, folder)
    shot = {}
    for name in sorted(os.listdir(top)):
        with open(os.path.join(top, name), "rb") as fh:
            shot[name] = hashlib.sha256(fh.read()).hexdigest()
    return shot


def refused_rows(code, rows):
    """True when ``code`` is the refusal naming exactly these rows."""
    return (isinstance(code, str) and all(r in code for r in rows)
            and code.count("\n  ") == len(rows) and "--tag NAME" in code
            and "folder of its own" in code)


PLAIN = dict(LAPLACIAN_OUTPUT=False)


def run7(folder, argv=(), **cfg):
    """run() with the plain head, as these checks were made."""
    return run(folder, argv, **{**PLAIN, **cfg})


surv = dict(T_TRAIN=52.0, VAL_T_END=110.0, MONITOR_T_END=110.0)
check(T.best_variant({}) == {"EMA": False, "SELECT": "score",
                             "LAPLACIAN_OUTPUT": False,
                             "DT_CONDITIONED": False, "MU_CHANNEL": False}
      and T.best_variant({"train_config": {}}) == T.best_variant({})
      and T.best_variant({"train_config": {"constants": {
          "EMA": True, "SELECT": "survival", "LAPLACIAN_OUTPUT": True,
          "DT_CONDITIONED": True, "MU_CHANNEL": True}}})
      == {"EMA": True, "SELECT": "survival", "LAPLACIAN_OUTPUT": True,
          "DT_CONDITIONED": True, "MU_CHANNEL": True},
      "a best model's variant is read from its train_config, and one with "
      "none is EMA off, score, the base head, a fixed step and no mu channel")

out, code = run7("share", EPOCHS=1)
before = snapshot("share")
check(code is None and "fno_ch.pt" in before,
      "a base run leaves its best model in the folder")
for flags, rows in ((["--ema"], ["  EMA: False -> True"]),
                    (["--select", "survival"],
                     ["  SELECT: score -> survival"]),
                    (["--variant", "lap"],
                     ["  LAPLACIAN_OUTPUT: False -> True"]),
                    (["--ema", "--select", "survival"],
                     ["  EMA: False -> True",
                      "  SELECT: score -> survival"])):
    out, code = run7("share", flags)
    check(refused_rows(code, rows) and snapshot("share") == before
          and "[keep]" not in out and not epoch_lines(out),
          f"{' '.join(flags)} over that best is refused, names only what "
          f"differs, says --tag NAME, and moves and trains nothing")

out, code = run7("share", MU_CHANNEL=True)
check(refused_rows(code, ["  MU_CHANNEL: False -> True"])
      and snapshot("share") == before and "[keep]" not in out
      and not epoch_lines(out),
      "and so is a best model without the mu channel by a run with it")

out, code = run7("share_clip", EPOCHS=1, MU_CHANNEL=True, MU_CLIP=None)
out, code = run7("share_clip", EPOCHS=1, MU_CHANNEL=True, MU_CLIP=1.2)
check(code is None and "[keep] best model" in out
      and "MU_CLIP" not in T.best_variant({"train_config": {"constants": {
          "MU_CLIP": 1.2}}}),
      "MU_CLIP is not part of a best model's variant: it does not change "
      "training, so a run with another clip may start fresh over its best")

out, code = run7("share_ema", ["--ema"], EPOCHS=1)
before = snapshot("share_ema")
out, code = run7("share_ema", [])
check(refused_rows(code, ["  EMA: True -> False"])
      and snapshot("share_ema") == before,
      "and an averaged best is not overwritten by a plain run either")
out, code = run7("share_ema", ["--ema"], EPOCHS=1)
check(code is None and "[keep] best model" in out
      and "fno_ch_prev.pt" in snapshot("share_ema"),
      "while the same variant may start fresh over its own best, which "
      "moves aside as before")

out, code = run7("share_surv", ["--select", "survival"], EPOCHS=1, **surv)
before = snapshot("share_surv")
out, code = run7("share_surv", [], **surv)
check(refused_rows(code, ["  SELECT: survival -> score"])
      and snapshot("share_surv") == before,
      "a survival best is not overwritten by a score run")
out, code = run7("share_surv", ["--ema"], **surv)
check(refused_rows(code, ["  EMA: False -> True",
                          "  SELECT: survival -> score"])
      and snapshot("share_surv") == before,
      "and every difference is listed at once")

out, code = run7("share_lap", ["--variant", "lap"], EPOCHS=1)
before = snapshot("share_lap")
out, code = run7("share_lap", [])
check(refused_rows(code, ["  LAPLACIAN_OUTPUT: True -> False"])
      and snapshot("share_lap") == before,
      "a Laplacian-head best is not overwritten by a base run")

out, code = run7("share_old", EPOCHS=1)
old_best = path("share_old", "fno_ch.pt")
ck = load(old_best)
ck.pop("train_config")
torch.save(ck, old_best)
before = snapshot("share_old")
for flags, rows in ((["--ema"], ["  EMA: False -> True"]),
                    (["--select", "survival"],
                     ["  SELECT: score -> survival"]),
                    (["--variant", "lap"],
                     ["  LAPLACIAN_OUTPUT: False -> True"])):
    out, code = run7("share_old", flags)
    check(refused_rows(code, rows) and snapshot("share_old") == before,
          f"a best from before train_config counts as the plain run, so "
          f"{' '.join(flags)} is refused over it")
out, code = run7("share_old", [])
check(code is None and "[keep] best model" in out
      and "fno_ch_prev.pt" in snapshot("share_old"),
      "and a plain run may start fresh over it")

other = os.path.join(D, "share_smoke", "fno_ch.pt")
os.makedirs(os.path.dirname(other))
torch.save({"run_meta": T.run_meta(), "train_config": {"constants": {
    "EMA": True, "SELECT": "survival", "LAPLACIAN_OUTPUT": True}}}, other)
reset(**PLAIN)
try:
    T.check_best_model(other)
    msg = None
except SystemExit as exc:
    msg = str(exc)
T.ROOT = os.path.join(D, "share_smoke")
try:
    T.check_best_model(other)
    smoke_msg = None
except SystemExit as exc:
    smoke_msg = str(exc)
check(msg is not None and msg.count("\n  ") == 3 and smoke_msg is None,
      "under --smoke's root nothing is refused, whatever the variant")

print("\n8. the low-k fill of the spectral weights, and the precision flags")


def lowk_net(fill, n_ref=16):
    """A tiny float64 net, the same weights whatever ``fill`` is: random
    spectral weights, so the entries next to DC are not small, and a head
    that does something."""
    torch.manual_seed(3)
    net = T.CahnHilliardFNO(channels=6, modes=4, n_layers=2, n_ref=n_ref,
                            lowk_fill=fill).double()
    with torch.no_grad():
        for mod in net.modules():
            if isinstance(mod, T.SpectralConv2d):
                mod.weight.normal_()
        last = net.backbone.project[2]
        assert isinstance(last, nn.Conv2d)
        nn.init.normal_(last.weight, std=0.3)
    return net.eval()


def spectral(net):
    return [m for m in net.modules() if isinstance(m, T.SpectralConv2d)]


def old_resampled(self, n):
    """_resampled_weight as it was before the fill existed."""
    m_n = self.modes_for(n)
    if m_n == self.modes:
        return self.effective_weight
    w = self.effective_weight.permute(0, 1, 4, 2, 3).reshape(
        1, self.in_channels * self.out_channels * 2,
        2 * self.modes + 1, self.modes + 1)
    w = torch.nn.functional.interpolate(
        w, size=(2 * m_n + 1, m_n + 1), mode="bilinear", align_corners=True)
    return w.reshape(self.in_channels, self.out_channels, 2,
                     2 * m_n + 1, m_n + 1).permute(0, 1, 3, 4, 2)


fill_on, fill_off = lowk_net(True), lowk_net(False)
fill_off.load_state_dict(fill_on.state_dict())
plain = T.CahnHilliardFNO(channels=6, modes=4, n_layers=2, n_ref=16)
check(fill_on.config["lowk_fill"] is True and fill_off.config["lowk_fill"]
      is False and plain.config["lowk_fill"] is False
      and T.lowk_fill_of(fill_on) and not T.lowk_fill_of(fill_off)
      and all(m.lowk_fill for m in spectral(fill_on))
      and T.LOWK_FILL is True,
      "lowk_fill is in the model's config and on every spectral layer; off "
      "by default in the model class (as an older checkpoint is), on in "
      "fno_train's training default")
g = torch.Generator().manual_seed(4)
psi = torch.randn(2, 1, 16, 16, dtype=torch.float64, generator=g)
tiled2, tiled4 = psi.repeat(1, 1, 2, 2), psi.repeat(1, 1, 4, 4)
check(torch.equal(fill_on(psi), fill_off(psi)),
      "at n = n_ref there is no resampling: the output is bitwise the same")
for name, net in (("on", fill_on), ("off", fill_off)):
    err2 = float((net(tiled2) - net(psi).repeat(1, 1, 2, 2)).abs().max())
    err4 = float((net(tiled4) - net(psi).repeat(1, 1, 4, 4)).abs().max())
    check(err2 < 1e-12 and err4 < 1e-12,
          f"fill {name}: a 2 x 2 and a 4 x 4 tiling gives the tiled output "
          f"in float64 ({err2:.1e}, {err4:.1e})")
check(float((fill_on(tiled2) - fill_off(tiled2)).abs().max()) < 1e-12,
      "a tiling has no odd modes, so it does not see the fill")
rough = torch.randn(2, 1, 32, 32, dtype=torch.float64, generator=g)
check(float((fill_on(rough) - fill_off(rough)).abs().max()) > 1e-6,
      "a field that is not a tiling does see the fill")
with mock.patch.object(T.SpectralConv2d, "_resampled_weight", old_resampled):
    before = fill_off(rough)
check(torch.equal(before, fill_off(rough)),
      "with the fill off the output is bitwise what it was before the fill")

layer_on, layer_off = spectral(fill_on)[0], spectral(fill_off)[0]
dense = layer_on.weight.detach() * layer_on.dc_mask     # the masked weights
m = layer_on.modes
fill = (dense[:, :, m - 1, 0] + dense[:, :, m + 1, 0]
        + dense[:, :, m, 1]) / 3.0
w_on, w_off = layer_on._resampled_weight(32), layer_off._resampled_weight(32)
check(torch.equal(w_off, old_resampled(layer_off, 32)),
      "the off weights are the old ones, bitwise")
m2 = layer_on.modes_for(32)


def close(a, b):
    return bool(torch.allclose(a, b, rtol=0.0, atol=1e-14))


check(m2 == 2 * m and w_on.shape[2:4] == (2 * m2 + 1, m2 + 1)
      and float(w_on[:, :, m2, 0].abs().max()) == 0.0
      and float(w_off[:, :, m2, 0].abs().max()) == 0.0,
      "on the doubled grid the DC entry is 0 again after the resampling")
check(close(w_off[:, :, m2 + 1, 0], 0.5 * dense[:, :, m + 1, 0])
      and close(w_on[:, :, m2 + 1, 0],
                0.5 * (fill + dense[:, :, m + 1, 0]))
      and close(w_on[:, :, m2 - 1, 0],
                0.5 * (fill + dense[:, :, m - 1, 0]))
      and close(w_on[:, :, m2, 1], 0.5 * (fill + dense[:, :, m, 1])),
      "the lowest new modes: half the trained neighbour without the fill, "
      "the mean of the fill and the neighbour with it")
check(close(w_on[:, :, m2 + 1, 1], 0.25 * (
    fill + dense[:, :, m + 1, 0] + dense[:, :, m, 1]
    + dense[:, :, m + 1, 1])),
      "and the diagonal one blends the fill, two neighbours and itself")
diff = (w_on - w_off).abs().amax(dim=(0, 1, 4)) > 0
want_diff = {(m2 - 1, 0), (m2 + 1, 0), (m2 - 1, 1), (m2, 1), (m2 + 1, 1)}
check({(int(i), int(j)) for i, j in diff.nonzero()} == want_diff,
      f"only those entries differ from the unfilled grid, every other one "
      f"(the even indices, the trained points) is bitwise the same: "
      f"{sorted(want_diff)}")
m4 = layer_on.modes_for(64)
w4_on, w4_off = layer_on._resampled_weight(64), layer_off._resampled_weight(64)
check(m4 == 4 * m and float(w4_on[:, :, m4, 0].abs().max()) == 0.0
      and all(close(w4_off[:, :, m4 + j, 0], j / 4 * dense[:, :, m + 1, 0])
              and close(w4_on[:, :, m4 + j, 0],
                        (1 - j / 4) * fill + j / 4 * dense[:, :, m + 1, 0])
              for j in (1, 2, 3)),
      "on the 4 x grid modes 1, 2, 3 are 1/4, 1/2, 3/4 of the trained one "
      "without the fill, and blend towards it from the fill with it")
w_down = layer_on._resampled_weight(8)
check(float(w_down[:, :, layer_on.modes_for(8), 0].abs().max()) == 0.0
      and torch.equal(w_down, layer_off._resampled_weight(8)),
      "a smaller lattice samples the old grid's points: the same weights, "
      "the DC entry 0")
no_zero = T.SpectralConv2d(2, 2, 4, 16, "physical", zero_dc=False,
                           lowk_fill=True).double().eval()
check(torch.equal(no_zero._resampled_weight(32),
                  old_resampled(no_zero, 32)),
      "without zero_dc there is no masked entry to fill: nothing changes")

print("\n   set_lowk_fill")
net = lowk_net(False)
base_out = net(rough)
check(any(len(mm._cache) for mm in spectral(net)),
      "an eval pass keeps the resampled weights")
T.set_lowk_fill(net, True)
check(all(mm.lowk_fill and not mm._cache for mm in spectral(net))
      and net.config["lowk_fill"] is True and T.lowk_fill_of(net),
      "set_lowk_fill sets every spectral layer and the config, and clears "
      "their caches")
filled = net(rough)
check(float((filled - base_out).abs().max()) > 1e-6
      and torch.equal(net(psi), fill_off(psi)),
      "the lattices that need resampling now differ, n = n_ref does not")
T.set_lowk_fill(net, False)
check(torch.equal(net(rough), base_out) and net.config["lowk_fill"] is False,
      "and turning it off again gives back the same output, bitwise")

print("\n   the setting in a checkpoint and in build_model")
reset(LOWK_FILL=False)
built_off = T.build_model(n_ref=16, device="cpu")
reset()
built = T.build_model(n_ref=16, device="cpu")
check(built.config["lowk_fill"] is True and T.lowk_fill_of(built)
      and built_off.config["lowk_fill"] is False
      and not T.lowk_fill_of(built_off),
      "build_model takes LOWK_FILL (on by default): a training with it on "
      "makes a model with it, and its validation (the model itself) uses it; "
      "with it off, a model without")
reset(PRECISION="fp64")
path_on = os.path.join(D, "fill_on.pt")
T.save_model(fill_on, path_on)
with contextlib.redirect_stdout(sink):
    back_on, _ = T.load_checkpoint(path_on)
check(back_on.config["lowk_fill"] is True and T.lowk_fill_of(back_on)
      and torch.equal(back_on(rough), fill_on(rough)),
      "a checkpoint with the key loads with the fill on")
legacy_cfg = {k: v for k, v in fill_off.config.items() if k != "lowk_fill"}
fill_off.config = dict(legacy_cfg)
path_old = os.path.join(D, "fill_old.pt")
T.save_model(fill_off, path_old)
with contextlib.redirect_stdout(sink):
    back_old, _ = T.load_checkpoint(path_old)
check(back_old.config["lowk_fill"] is False and not T.lowk_fill_of(back_old)
      and torch.equal(back_old(rough), base_out)
      and T.model_drift(legacy_cfg, back_old.config) == {}
      and T.model_drift(legacy_cfg, dict(back_old.config, lowk_fill=True))
      == {"lowk_fill": (False, True)},
      "one without it loads with the fill off, gives the output it always "
      "did, and a resume reads the missing key as off")
reset()

print("\n   the precision flags")
flags_before = (torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision())
try:
    for prec, tf32, level in (("fp32", False, "highest"),
                              ("tf32", True, "high"),
                              ("fp64", False, "highest")):
        # the opposite first, so the call has something to undo
        torch.backends.cuda.matmul.allow_tf32 = not tf32
        torch.backends.cudnn.allow_tf32 = not tf32
        torch.set_float32_matmul_precision("highest" if tf32 else "high")
        reset(PRECISION=prec)
        got = T.set_precision_flags()
        check(got is tf32 and torch.backends.cuda.matmul.allow_tf32 is tf32
              and torch.backends.cudnn.allow_tf32 is tf32
              and torch.get_float32_matmul_precision() == level,
              f"PRECISION {prec}: matmul and cudnn allow_tf32 {tf32}, "
              f"matmul precision {level}")
    torch.backends.cudnn.allow_tf32 = True          # PyTorch's default
    reset(PRECISION="fp32")
    T.device_and_precision()
    check(torch.backends.cudnn.allow_tf32 is False,
          "device_and_precision sets them through it")
    torch.backends.cudnn.allow_tf32 = True
    reset(PRECISION="bf16")
    try:
        T.set_precision_flags()
        why = None
    except ValueError as exc:
        why = str(exc)
    check(why is not None and "PRECISION" in why
          and torch.backends.cudnn.allow_tf32 is True,
          "a PRECISION that is none of them is refused, flags untouched")
finally:
    (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
     level) = flags_before
    torch.set_float32_matmul_precision(level)
    reset()

print("\n9. the chemical-potential input channel (mu_channel)")
MU_CH = 6


def mu_net(mu, dt_conditioned=False, mean=True, laplacian=True,
           clip=None):
    """A tiny float64 net: random spectral weights and FiLM, and a head that
    does something."""
    torch.manual_seed(5)
    net = T.CahnHilliardFNO(
        channels=MU_CH, modes=4, n_layers=2, n_ref=16,
        mode_scaling="physical", use_mean_channel=mean,
        laplacian_output=laplacian, dt_conditioned=dt_conditioned,
        lowk_fill=True, mu_channel=mu, mu_clip=clip).double()
    with torch.no_grad():
        for mod in net.modules():
            if isinstance(mod, T.SpectralConv2d):
                mod.weight.normal_()
        last = net.backbone.project[2]
        assert isinstance(last, nn.Conv2d)
        nn.init.normal_(last.weight, std=0.3)
        if net.dt_conditioned:
            for block in net.backbone.blocks:
                film = block.film
                assert isinstance(film, nn.Linear)
                film.weight.normal_(0.0, 0.5)
    return net.eval()


def lift_conv(net):
    conv = net.backbone.lift[0]
    assert isinstance(conv, nn.Conv2d)
    return conv


def lift_input(net, psi, dt=None):
    """The tensor the backbone is given for ``psi``."""
    seen = []
    hook = net.backbone.register_forward_pre_hook(
        lambda mod, args: seen.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            T.step(net, psi, dt)
    finally:
        hook.remove()
    return seen[0]


def before_mu(net, psi, dt=None):
    """CahnHilliardFNO.forward as it was before mu_channel, written out (with
    hard mass conservation and the residual step, as the models here have
    them)."""
    x = torch.cat([psi, psi.mean(dim=(-2, -1), keepdim=True).expand_as(psi)],
                  dim=1)
    if net.dt_conditioned:
        assert (net.cond is not None and net.log_gain is not None
                and net.gain_exp is not None and dt is not None)
        out = net.backbone(x, net.cond(T.dt_embedding(dt, net.cond_in - 1)))
        gain = torch.exp(net.log_gain + net.gain_exp
                         * torch.log(dt.clamp_min(1e-6)))
        out = out * gain[:, None, None, None]
    else:
        out = net.backbone(x)
    if net.laplacian_output:
        out = T._lap_t(out)
    out = out - out.mean(dim=(-2, -1), keepdim=True)
    return psi + out


g = torch.Generator().manual_seed(7)
psi = torch.randn(2, 1, 16, 16, dtype=torch.float64, generator=g)
dts = torch.tensor([3.0, 40.0], dtype=torch.float64)
mu_on, mu_off = mu_net(True), mu_net(False)
x_on, x_off = lift_input(mu_on, psi), lift_input(mu_off, psi)
bare = T.CahnHilliardFNO(channels=MU_CH, modes=4, n_layers=2, n_ref=16)
check(x_on.shape == (2, 3, 16, 16) and x_off.shape == (2, 2, 16, 16)
      and lift_conv(mu_on).in_channels == 3
      and lift_conv(mu_off).in_channels == 2
      and mu_on.config["mu_channel"] is True and mu_on.mu_channel
      and mu_off.config["mu_channel"] is False and not mu_off.mu_channel
      and bare.config["mu_channel"] is False,
      "mu_channel adds one input channel to the lift and is in the model's "
      "config; off by default in the model class (as an older checkpoint is)")
no_mean = mu_net(True, mean=False)
check(lift_input(no_mean, psi).shape[1] == 2
      and lift_conv(no_mean).in_channels == 2
      and lift_conv(mu_net(False, mean=False)).in_channels == 1,
      "without the mean channel it is the second input, not the third")
extra = MU_CH * 1 * 1 * 1       # out channels x one more input x a 1 x 1 kernel
check(mu_on.n_parameters() - mu_off.n_parameters() == extra
      and lift_conv(mu_on).weight.numel() - lift_conv(mu_off).weight.numel()
      == extra,
      f"the parameters grow by the lift's extra input weights ({extra}), and "
      f"by nothing else")
n_off, n_on = (T.CahnHilliardFNO(
    channels=38, modes=24, n_layers=5, laplacian_output=True,
    dt_conditioned=True, mu_channel=m).n_parameters() for m in (False, True))
print(f"   the default model (38 channels, 24 modes, 5 layers, dt-conditioned,"
      f" Laplacian head): {n_off:,} without, {n_on:,} with the channel")
check(n_off == 17_799_303 and n_on == 17_799_341 and n_on - n_off == 38,
      "the full-size model: 17,799,303 as before, 17,799,341 with the "
      "channel (+38)")

p = psi.numpy()
want = p ** 3 - p - T._lap_np(p)
check(x_on.dtype == torch.float64
      and float(np.abs(x_on[:, 2:].numpy() - want).max()) < 1e-13
      and float(np.abs(want).max()) > 1.0
      and torch.equal(x_on[:, :1], psi)
      and torch.equal(x_on[:, 1:2], psi.mean(dim=(-2, -1), keepdim=True)
                      .expand_as(psi)),
      "the channel is psi^3 - psi - lap(psi) with the numpy 5-point periodic "
      "Laplacian (float64), after psi and its mean")
x32 = lift_input(mu_net(True).float(), psi.float())
check(x32.dtype == torch.float32
      and float(np.abs(x32[:, 2:].numpy() - want).max()) < 1e-4,
      "in float32 the channel is float32")
flat_in = lift_input(mu_on, torch.cat([torch.ones(1, 1, 16, 16),
                                       -torch.ones(1, 1, 16, 16)]).double())
check(float(flat_in[:, 2].abs().max()) == 0.0,
      "a bulk phase (psi = 1 or -1 everywhere) has mu exactly 0")

with torch.no_grad():
    state = mu_off.state_dict()
    w3 = torch.zeros_like(lift_conv(mu_on).weight)
    w3[:, :2] = state["backbone.lift.0.weight"]
    state["backbone.lift.0.weight"] = w3
    mu_on.load_state_dict(state)
    zero_w = float((mu_on(psi) - mu_off(psi)).abs().max())
    w3[:, 2] = 0.5
    mu_on.load_state_dict(state)
    live_w = float((mu_on(psi) - mu_off(psi)).abs().max())
check(zero_w < 1e-12 and live_w > 1e-6,
      f"the channel is wired in: with zero weights for it the output is the "
      f"old one ({zero_w:.1e}), with weights it is not ({live_w:.1e})")

tiled_net = mu_net(True, dt_conditioned=True)
base = T.step(tiled_net, psi, dts)
errs = [float((T.step(tiled_net, psi.repeat(1, 1, k, k), dts)
               - base.repeat(1, 1, k, k)).abs().max()) for k in (2, 4)]
check(max(errs) < 1e-12 and float((base - psi).abs().max()) > 1e-3
      and tiled_net.config["mu_channel"] is True,
      f"a dt-conditioned model with physical mode scaling and the channel: "
      f"a 2 x 2 and a 4 x 4 tiling gives the tiled output in float64 "
      f"({errs[0]:.1e}, {errs[1]:.1e}), and the step is not trivial")

for name, net, dt in (("fixed-step", mu_net(True), None),
                      ("conditioned", mu_net(True, dt_conditioned=True), dts),
                      ("plain head", mu_net(True, laplacian=False), None)):
    moved = float((T.step(net, psi + 0.3, dt).mean(dim=(-2, -1))
                   - (psi + 0.3).mean(dim=(-2, -1))).abs().max())
    check(moved < 1e-12, f"mass is conserved with the channel on, {name} "
          f"model ({moved:.1e})")

print("\n   a checkpoint without the key, and with it")
reset(PRECISION="fp64")
for name, dt_cond, dt in (("fixed-step", False, None),
                          ("conditioned", True, dts)):
    net = mu_net(False, dt_conditioned=dt_cond)
    legacy_cfg = {k: v for k, v in net.config.items() if k != "mu_channel"}
    net.config = dict(legacy_cfg)
    path_old = os.path.join(D, f"mu_old_{name}.pt")
    T.save_model(net, path_old)
    with contextlib.redirect_stdout(sink):
        back_old, ck_old = T.load_checkpoint(path_old)
    check("mu_channel" not in ck_old["config"]
          and back_old.config["mu_channel"] is False
          and not back_old.mu_channel and lift_conv(back_old).in_channels == 2
          and torch.equal(T.step(back_old, psi, dt), T.step(net, psi, dt))
          and torch.equal(T.step(back_old, psi, dt), before_mu(net, psi, dt)),
          f"a {name} checkpoint whose config lacks the key loads with the "
          f"channel off and gives bitwise the output of the code before it")
    check(T.model_drift(legacy_cfg, back_old.config) == {}
          and T.model_drift(legacy_cfg, dict(back_old.config, mu_channel=True))
          == {"mu_channel": (False, True)},
          f"and a resume reads the missing key as off ({name})")
path_on = os.path.join(D, "mu_on.pt")
T.save_model(mu_on, path_on)
with contextlib.redirect_stdout(sink):
    back_on, _ = T.load_checkpoint(path_on)
check(back_on.config["mu_channel"] is True and back_on.mu_channel
      and lift_conv(back_on).in_channels == 3
      and torch.equal(back_on(psi), mu_on(psi)),
      "a checkpoint with the key loads with the channel on")
if REAL_MODEL is not None:
    with contextlib.redirect_stdout(sink):
        real, real_ck = T.load_checkpoint(REAL_MODEL)
    if real_ck["config"].get("dt_conditioned"):
        # every checkpoint of the project is dt-conditioned since 2026-10-23,
        # and the key is exactly what this check is about (a checkpoint from
        # before it existed), so there is nothing to run it on
        print("SKIP  the checkpoint to test against is dt-conditioned: no "
              "trained fixed-step checkpoint (from before the mu channel) "
              "to load with the channel off")
    else:
        big = psi.repeat(1, 1, 2, 2)
        check("mu_channel" not in real_ck["config"] and not real.mu_channel
              and lift_conv(real).in_channels == 2
              and torch.equal(real(big), before_mu(real, big)),
              "the trained fixed-step checkpoint of this project loads with "
              "the channel off and steps as it did before")
reset()

print("\n   mu_clip, the clamp on psi before the channel")
bounded = psi.clamp(-1.0, 1.0)
spike = bounded.clone()
spike[0, 0, 5, 7] = 3.0
spike[1, 0, 9, 2] = -2.5
plain_net, clip_net = mu_net(True), mu_net(True, clip=1.2)
check(plain_net.mu_clip is None and plain_net.config["mu_clip"] is None
      and clip_net.mu_clip == 1.2 and clip_net.config["mu_clip"] == 1.2
      and bare.config["mu_clip"] is None,
      "mu_clip is in the model's config, None by default in the model class")
for name, dt_cond, dt in (("fixed-step", False, None),
                          ("conditioned", True, dts)):
    a_net, b_net = (mu_net(True, dt_conditioned=dt_cond),
                    mu_net(True, dt_conditioned=dt_cond, clip=1.2))
    check(torch.equal(T.step(a_net, bounded, dt), T.step(b_net, bounded, dt))
          and torch.equal(lift_input(a_net, bounded, dt),
                          lift_input(b_net, bounded, dt)),
          f"with |psi| <= the clip everywhere the clip changes nothing: the "
          f"{name} model's input and output are bitwise the same")
x_spike = lift_input(clip_net, spike)
c = np.clip(spike.numpy(), -1.2, 1.2)
want_c = c ** 3 - c - T._lap_np(c)
raw = spike.numpy() ** 3 - spike.numpy() - T._lap_np(spike.numpy())
check(float(np.abs(x_spike[:, 2:].numpy() - want_c).max()) < 1e-13
      and float(np.abs(want_c).max()) < 0.5 * float(np.abs(raw).max())
      and torch.equal(x_spike[:, :1], spike)
      and torch.equal(x_spike[:, 1:2], spike.mean(dim=(-2, -1), keepdim=True)
                      .expand_as(spike)),
      f"a pixel at 3.0: the channel is that of psi clamped to +-1.2 "
      f"(largest {float(np.abs(want_c).max()):.2f}, not "
      f"{float(np.abs(raw).max()):.0f}); psi and its mean are not clamped")
check(not torch.equal(T.step(clip_net, spike), T.step(plain_net, spike)),
      "and the step from that state differs from the unclipped model's")
for name, net, dt in (("fixed-step", mu_net(True, clip=1.2), None),
                      ("conditioned",
                       mu_net(True, dt_conditioned=True, clip=1.2), dts)):
    moved = float((T.step(net, spike, dt).mean(dim=(-2, -1))
                   - spike.mean(dim=(-2, -1))).abs().max())
    check(moved < 1e-12, f"mass is still conserved with the clip and a "
          f"spike, {name} model ({moved:.1e})")
no_mu_clip = mu_net(False, clip=1.2)
check(torch.equal(T.step(no_mu_clip, spike), T.step(mu_net(False), spike))
      and T.mu_clip_of(no_mu_clip) is None and T.mu_clip_of(clip_net) == 1.2
      and T.mu_clip_of(plain_net) is None,
      "a model without the channel ignores the clip (mu_clip_of says None)")
T.set_mu_clip(plain_net, 1.2)
check(plain_net.mu_clip == 1.2 and plain_net.config["mu_clip"] == 1.2
      and torch.equal(T.step(plain_net, spike), T.step(clip_net, spike)),
      "set_mu_clip sets the model and its config, and then steps as one "
      "built with the clip")
T.set_mu_clip(plain_net, None)
check(plain_net.mu_clip is None and plain_net.config["mu_clip"] is None,
      "and None takes it off")
reset(PRECISION="fp64")
path_clip = os.path.join(D, "mu_clip.pt")
T.save_model(clip_net, path_clip)
with contextlib.redirect_stdout(sink):
    back_clip, ck_clip = T.load_checkpoint(path_clip)
check(ck_clip["config"]["mu_clip"] == 1.2 and back_clip.mu_clip == 1.2
      and back_clip.config["mu_clip"] == 1.2
      and torch.equal(T.step(back_clip, spike), T.step(clip_net, spike)),
      "a checkpoint round trip keeps the clip")
legacy_clip = {k: v for k, v in clip_net.config.items() if k != "mu_clip"}
old_net = mu_net(True)
old_net.config = dict(legacy_clip)
path_noclip = os.path.join(D, "mu_noclip.pt")
T.save_model(old_net, path_noclip)
with contextlib.redirect_stdout(sink):
    back_noclip, ck_noclip = T.load_checkpoint(path_noclip)
check("mu_clip" not in ck_noclip["config"] and back_noclip.mu_clip is None
      and back_noclip.config["mu_clip"] is None
      and torch.equal(T.step(back_noclip, spike), T.step(old_net, spike))
      and T.CONFIG_DEFAULTS["mu_clip"] is None,
      "a checkpoint whose config lacks the key loads with no clip, and steps "
      "as it did")
check(T.model_drift(legacy_clip, back_noclip.config) == {}
      and T.model_drift(legacy_clip, clip_net.config) == {},
      "and a resume of it with the clip is not refused (the clip does not "
      "change training)")
reset()

print("\n   the validation rollout of a window the horizon does not reach")
refs_t = types.SimpleNamespace(
    times=np.array([100.0, 112.0]),
    truth=np.stack([psi[:, 0].numpy()] * 2))
val_net = mu_net(False, dt_conditioned=True)
with warnings.catch_warnings():
    warnings.simplefilter("error")
    far = T.vdt_rollout(val_net, refs_t, "cpu", horizon=20.0)
    whole = T.vdt_rollout(val_net, refs_t, "cpu")
check(far["accuracy"] == 0.0 and np.isfinite(far["score"])
      and 0.0 < whole["accuracy"] <= 1.0 and far["score"] == whole["score"],
      "a horizon of 20 against a window from t = 100 has no time in range: "
      "accuracy 0 (as in_range_accuracy), with no empty-mean warning")

print("\n   build_model, --fixed-step and a training run")
reset(MU_CHANNEL=False)
built_off = T.build_model(n_ref=16, device="cpu")
reset(MU_CHANNEL=True)
built_on = T.build_model(n_ref=16, device="cpu")
check(built_on.config["mu_channel"] is True and built_on.mu_channel
      and lift_conv(built_on).in_channels == 3
      and built_off.config["mu_channel"] is False
      and lift_conv(built_off).in_channels == 2,
      "build_model takes MU_CHANNEL: a training with it on makes a model "
      "with the channel, with it off one without")
reset(MU_CHANNEL=True, MU_CLIP=None)
built_free = T.build_model(n_ref=16, device="cpu")
check(MU_CLIP_DEFAULT == 1.2 and built_on.mu_clip == 1.2
      and built_on.config["mu_clip"] == 1.2
      and built_off.config["mu_clip"] is None
      and built_free.config["mu_clip"] is None and built_free.mu_clip is None
      and "MU_CLIP" not in T.FIXED_STEP,
      "build_model takes MU_CLIP (1.2 by default) for a model with the "
      "channel, and None for one without; --fixed-step has nothing to put "
      "back")
reset(MU_CHANNEL=True)
check(MU_TRAINING_DEFAULT is True
      and T.FIXED_STEP["MU_CHANNEL"] is False
      and any(n.startswith("MU_CHANNEL True -> False") for n in FIXED_NOTES)
      and DEFAULTS["MU_CHANNEL"] is False
      and T.CONFIG_DEFAULTS["mu_channel"] is False,
      "the training default is on; --fixed-step puts it off (and says so), "
      "and a config without the key is read as off")

out, code = run("mu_run", EPOCHS=1, MU_CHANNEL=True)
best = load(path("mu_run", "fno_ch.pt"))
check(code is None and best["config"]["mu_channel"] is True
      and best["state_dict"]["backbone.lift.0.weight"].shape[1] == 3
      and best["train_config"]["constants"]["MU_CHANNEL"] is True
      and best["config"]["mu_clip"] == 1.2
      and best["train_config"]["constants"]["MU_CLIP"] == 1.2
      and "[model] MU_CHANNEL" in out and "clamped to +-1.2" in out
      and epoch_lines(out),
      "a tiny training run with the channel trains, prints it (and the "
      "clamp), and records both in the model's config and constants")
before = snapshot("mu_run")
out, code = run("mu_run", EPOCHS=2, RESUME=True, MU_CHANNEL=False)
check(isinstance(code, str) and "written by a different model" in code
      and "mu_channel: True -> False" in code
      and "only be set back at the top of fno_train.py" in code
      and snapshot("mu_run") == before,
      "resuming it without the channel is refused, and nothing moves")
out, code = run("mu_run", ["--accept-schedule"], EPOCHS=2, RESUME=True,
                MU_CHANNEL=True)
check(code is None and "[resume]" in out and len(epoch_lines(out)) == 1,
      "and with the channel it resumes")

reset()
print("\n" + ("MODELSEL PASS" if ok else "MODELSEL FAIL"))
sys.exit(0 if ok else 1)
