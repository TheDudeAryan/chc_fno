"""The variable-step (dt-conditioned) model, the group protocol's training and
its reference cache (fno_gt).

  1. an old fixed-step checkpoint loads and gives the output it gave before
  2. the conditioned model: dt only through the gain at init, FiLM, errors
  3. the chain sampler: dt range and grid, chains fit, start time
  4. make_schedule: grid, must-hit times, step counts
  5. fno_gt file names and members; writer / loader round trip
  6. a tiny dt-conditioned run end to end: best checkpoint, resume
  7. --fixed-step trains as the old code did (same epoch lines)

Everything is written under tests/_scratch.  The old project (E:/Clo) is
only read.
"""
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import torch

from _paths import ROOT, SCRATCH, project_state

BOX = os.path.join(SCRATCH, "vdt_box")
shutil.rmtree(BOX, ignore_errors=True)
os.makedirs(BOX)
os.chdir(BOX)
PROJECT = project_state()

import fno_gt
import fno_paths
fno_paths.WORK = os.path.join(BOX, "Work")
import fno_runlog
import fno_train as T

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


OLD_ROOT = "E:/Clo"
OLD_MODEL = os.path.join(OLD_ROOT, "Work", "models", "train_t0300_lap",
                         "fno_ch.pt")


def load_foreign(name, folder, file):
    """A module of another project, read only: no bytecode, own name."""
    if not os.path.exists(os.path.join(folder, file)):
        return None
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(folder, file))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        print(f"   ({file} of {folder} could not be imported: {exc!r})")
        return None
    finally:
        sys.path.remove(folder)


def tiny_model(dt_conditioned=False):
    torch.manual_seed(0)
    return T.CahnHilliardFNO(channels=8, modes=4, n_ref=16,
                             laplacian_output=True,
                             dt_conditioned=dt_conditioned).double()


def field(b=2, n=16, seed=3):
    g = torch.Generator().manual_seed(seed)
    return (0.3 * torch.randn(b, 1, n, n, generator=g)).double()


def quiet(fn, *a, **kw):
    buf, real = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        return fn(*a, **kw)
    finally:
        sys.stdout = real


# ---------------------------------------------------------------------------
print("1. an old fixed-step checkpoint")
old_train = load_foreign("old_fno_train", OLD_ROOT, "fno_train.py")
x = field(2, 64).float()
if os.path.exists(OLD_MODEL):
    model, ck = T.load_checkpoint(OLD_MODEL)
    check(not model.dt_conditioned and "dt_conditioned" not in ck["config"],
          "the real checkpoint has no dt_conditioned key and loads fixed-step")
    with torch.no_grad():
        y = model(x)
        same = bool(torch.equal(y, model(x, 7.0))
                    and torch.equal(y, T.step(model, x, 7.0)))
    check(same, "a fixed-step model ignores dt (forward and step)")
    if old_train is not None:
        old_model, _ = old_train.load_checkpoint(OLD_MODEL)
        with torch.no_grad():
            check(torch.equal(old_model(x), y),
                  "its output equals the old code's, bit for bit")
    n_old = model.n_parameters()
else:
    print("   (no real checkpoint here; a tiny saved one stands in)")
    m0 = tiny_model()
    path = os.path.join(BOX, "old.pt")
    T.save_model(m0, path)
    ck0 = torch.load(path, weights_only=False)
    for key in ("dt_conditioned", "cond_dim"):
        ck0["config"].pop(key)
    torch.save(ck0, path)
    model, _ = T.load_checkpoint(path)
    check(not model.dt_conditioned, "a config without the key loads fixed")
    n_old = model.n_parameters()
cfg = dict(channels=8, modes=4, n_layers=5, n_ref=16)
drift = T.model_drift({k: v for k, v in tiny_model().config.items()
                       if k not in ("dt_conditioned", "cond_dim")},
                      tiny_model(dt_conditioned=True).config)
check(drift == {"dt_conditioned": (False, True)},
      f"a saved config without the key drifts only in dt_conditioned: {drift}")
msg = T.drift_refusal("x.pt", drift, "a and b")
check("--fixed-step" in msg and "Archive" in msg and "--tag" in msg,
      "the refusal names --fixed-step, Archive/ and --tag")

# ---------------------------------------------------------------------------
print("\n2. the conditioned model")
m = tiny_model(dt_conditioned=True)
psi = field()
with torch.no_grad():
    y1 = m(psi, 5.0)
check(torch.equal(y1, psi), "at init the zero projection leaves psi alone")
with torch.no_grad():
    for p in m.backbone.project[2].parameters():
        p.normal_(0.0, 0.3)
    a = m(psi, torch.tensor([1.0, 1.0]).double()) - psi
    b = m(psi, torch.tensor([3.0, 3.0]).double()) - psi
    c = m(psi, 3.0) - psi
check(torch.allclose(b, 3.0 * a, rtol=1e-9, atol=1e-12) and bool(a.abs().max() > 1e-6),
      "at init the increment depends on dt only through the gain (gain = dt)")
check(torch.equal(b, c), "a float dt is the same as a tensor of it")
mixed = m(psi, torch.tensor([1.0, 3.0]).double()) - psi
check(torch.allclose(mixed[0], a[0]) and torch.allclose(mixed[1], b[1]),
      "dt is per sample")
check(float((m(psi, 4.0).mean(dim=(-2, -1)) - psi.mean(dim=(-2, -1))).abs().max())
      < 1e-12, "mass is conserved")
with torch.no_grad():
    film = m.backbone.blocks[0].film
    assert isinstance(film, torch.nn.Linear)
    film.weight.normal_(0.0, 0.5)
    d = m(psi, 3.0) - psi
check(not torch.allclose(d, 3.0 * a, rtol=1e-3),
      "with a trained FiLM the dt dependence is more than the gain")
try:
    m(psi)
    check(False, "a conditioned model called without dt is refused")
except ValueError:
    check(True, "a conditioned model called without dt is refused")
try:
    T.step(m, psi)
    check(False, "step() without dt is refused for it")
except ValueError:
    check(True, "step() without dt is refused for it")
check(sum(p.numel() for n, p in m.named_parameters()
          if "film" not in n and n not in ("cond.0.weight", "cond.0.bias",
                                           "cond.2.weight", "cond.2.bias",
                                           "log_gain", "gain_exp"))
      == tiny_model().n_parameters(),
      "the conditioned model is the fixed one plus FiLM, MLP and gain")
full_fixed = T.CahnHilliardFNO(laplacian_output=True).n_parameters()
full_vdt = T.CahnHilliardFNO(laplacian_output=True,
                             dt_conditioned=True).n_parameters()
print(f"   parameters: fixed {full_fixed:,}, conditioned {full_vdt:,} "
      f"(+{full_vdt - full_fixed:,})")
check(17_700_000 < full_vdt < 18_100_000, "about 17.8-18.0 million")

# ---------------------------------------------------------------------------
print("\n3. the chain sampler")
n_fr, n_tr, side = 900, 3, 4
frames = np.zeros((n_tr, n_fr, side, side), dtype=np.float32)
frames += np.arange(n_fr, dtype=np.float32)[None, :, None, None]
smp = T.ChainSampler(frames, torch.device("cpu"), torch.device("cpu"),
                     torch.float32, 7)
check(smp.f_min == 350 and smp.df_min == 10 and smp.df_max == 500,
      "T_MIN 35 is frame 350, dt 1.0 is 10 frames, 50.0 is 500")
check(abs(smp.max_dt(4) - (899 - 350) // 4 * 0.1) < 1e-9,
      f"a chain of 4 caps dt at {smp.max_dt(4):.1f}")
all_dt, good, n_chains = [], True, 0
for n_steps in (1, 2, 4):
    for _ in range(40):
        chain, dt = smp.sample(64, n_steps)
        f = [c[:, 0, 0, 0].numpy() for c in chain]
        df = np.rint(dt.numpy() / 0.1).astype(int)
        good &= bool(np.all(f[0] >= 350) and np.all(df >= 10)
                     and np.all(df <= smp.df_hi(n_steps))
                     and np.all(np.abs(dt.numpy() - df * 0.1) < 1e-5)
                     and np.all(f[-1] <= n_fr - 1))
        for k in range(1, n_steps + 1):
            good &= bool(np.all(f[k] - f[k - 1] == df))
        if n_steps == 1:
            all_dt.append(dt.numpy())
        n_chains += 64
check(good, f"{n_chains} chains: start >= T_MIN, equal strides on the 0.1 "
            f"grid, dt in [1.0, cap], every chain fits")
all_dt = np.concatenate(all_dt)
check(1.0 <= all_dt.min() and all_dt.max() <= 50.0, "dt in [1, 50]")
check(2.0 < np.median(all_dt) < 8.5,
      f"dt is log-uniform (median {np.median(all_dt):.2f} against the "
      f"geometric middle of the cap)")
fixed, dtf = smp.sample(8, 1, dt=12.0)
check(np.allclose(dtf.numpy(), 12.0), "dt can be fixed for a check")
ch2, _ = smp.sample(16, 2)
aug = smp.augment(ch2, torch.Generator().manual_seed(1))
check(len(aug) == 3 and all(a.shape == c.shape for a, c in zip(aug, ch2)),
      "augment keeps the chain")

# ---------------------------------------------------------------------------
print("\n4. make_schedule")
must = (500, 1000, 2000, 4000, 6000, 8000)
cases = [(0.2, 40.0), (0.1, 50.0), (0.35, 30.0), (0.2, 50.0)]
for alpha, dtm in cases:
    mine = fno_gt.make_schedule(100.0, 10000.0, alpha, 1.0, dtm, 4.0, must)
    ok_grid = bool(np.all(np.abs(mine / 4.0 - np.round(mine / 4.0)) < 1e-9))
    check(ok_grid and mine[0] == 100.0 and mine[-1] == 10000.0
          and set(must) <= set(mine.tolist()),
          f"alpha {alpha}, dt_max {dtm:g}: on the 4.0 grid, hits the "
          f"must-hit times")
expect = {(0.2, 40.0): 255, (0.1, 50.0): 222, (0.35, 30.0): 316,
          (0.2, 50.0): 215}
got = {c: len(fno_gt.make_schedule(100.0, 10000.0, c[0], 1.0, c[1], 4.0, must))
       - 1 for c in cases}
check(got == expect, f"step counts {got} against the survey's {expect}")
check(len(fno_gt.make_schedule(100.0, 10000.0, 0.2, 1.0, 40.0, 4.0)) - 1 == 249,
      "the validation schedule (0.2, 40) is 249 steps")

# ---------------------------------------------------------------------------
print("\n5. fno_gt")
CACHE = os.path.join(BOX, "refs")
args = (16, 0.0)
span = (100.0, 108.0, 4.0)
solved = quiet(fno_gt.ground_truth_batch, CACHE, *args, [11, 22], *span,
               0.01, "cpu")
check(solved == 2, "two missing seeds are solved together")
paths = [fno_gt.find_gt(CACHE, *args, s, *span) for s in (11, 22)]
check(all(paths) and os.path.basename(paths[0] or "") ==
      "gt_l16_off+0.0_seed11_t100-108_ev4.npz", "the file is named in the group format")
t, f = fno_gt.load_gt(paths[0] or "")
check(isinstance(f, np.memmap) and f.shape == (3, 16, 16)
      and f.dtype == np.float32 and list(t) == [100.0, 104.0, 108.0],
      "it loads as a float32 memmap with the times")
with np.load(paths[0] or "") as z:
    names = sorted(z.files)
check(names == sorted(["times", "fields", "seed", "off", "l", "device",
                       "solver_dtype", "solver_dt", "torch_version"]),
      f"members as in the group format: {names}")
quiet(fno_gt.ground_truth_batch, os.path.join(BOX, "refs1"), *args, [11],
      *span, 0.01, "cpu")
one = fno_gt.load_gt(fno_gt.find_gt(os.path.join(BOX, "refs1"), *args, 11,
                                    *span) or "")[1]
check(np.array_equal(f, one), "a seed solved in a batch equals the single "
                              "solve, bit for bit")
check(quiet(fno_gt.ground_truth_batch, CACHE, *args, [11, 22], *span, 0.01,
            "cpu") == 0, "present seeds are not solved again")
long_span = (100.0, 112.0, 4.0)
quiet(fno_gt.ground_truth_batch, CACHE, *args, [11], *long_span, 0.01, "cpu")
check(fno_gt.find_gt(CACHE, *args, 11, 100.0, 108.0, 4.0) ==
      fno_gt.gt_path(CACHE, *args[:1], 0.0, 11, 100.0, 108.0, 4.0)
      and fno_gt.find_gt(CACHE, *args, 22, 100.0, 104.0, 4.0) is not None,
      "an exact name wins, and a longer solve serves a shorter span")
tl, fl = fno_gt.load_gt(fno_gt.find_gt(CACHE, *args, 11, *long_span) or "",
                        108.0)
check(len(tl) == 3 and np.array_equal(fl[:], f[:]),
      "load_gt(path, t1) slices; the first frames are the same solve")
check(not [p for p in os.listdir(CACHE) if "partial" in p],
      "no partial file is left")
os.environ["FNO_GT_CACHE"] = os.path.join(BOX, "elsewhere")
check(fno_paths.ref_cache_dir() == os.path.join(BOX, "elsewhere")
      and fno_paths.ref_cache_dir("smoke") == fno_paths.gt_dir("smoke"),
      "FNO_GT_CACHE moves the cache, and --smoke ignores it")
del os.environ["FNO_GT_CACHE"]
check(fno_paths.ref_cache_dir() == fno_paths.gt_dir(), "default: gt_cache")

# ---------------------------------------------------------------------------
print("\n6. a tiny dt-conditioned run")
DATA = os.path.join(BOX, "data", "tiny.npy")


def tiny_lean(path, rows=4, nfr=300, n=16):
    rng = np.random.default_rng(1)
    k2 = np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2

    def smooth():
        g = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n)))
                         * np.exp(-60.0 * k2)).real
        g -= g.mean()
        return 0.5 * g / np.abs(g).max()

    j = np.arange(nfr)[:, None, None] / 40.0
    data = np.stack([np.cos(j) * smooth() + np.sin(j) * smooth()
                     for _ in range(rows)]).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)
    meta = {"format": T.LEAN_FORMAT, "l": n, "dt": 0.01, "gap": 10,
            "n_ahead": 100, "shift": 10, "t_end": nfr / 10, "n_frames": nfr,
            "dtype": "float32", "n_train": rows - 1, "n_val": 1,
            "rows": [{"seed": i, "off": 0.0,
                      "split": "train" if i < rows - 1 else "val"}
                     for i in range(rows)]}
    with open(os.path.splitext(path)[0] + ".json", "w") as fh:
        json.dump(meta, fh)


tiny_lean(DATA)
START = {k: v for k, v in vars(T).items() if k.isupper()}


def reset():
    globals_ = T.__dict__
    globals_.update(START)
    fno_runlog.LOG_DIR = os.path.join(BOX, "logs")


def run_main(*argv):
    buf, real = io.StringIO(), (sys.stdout, sys.stderr)
    sys.stdout = sys.stderr = buf
    exc = None
    try:
        T.main(list(argv))
    except BaseException as e:
        exc = e
    finally:
        sys.stdout, sys.stderr = real
        getattr(fno_runlog, "stop_log", lambda: None)()
    return exc, buf.getvalue()


reset()
T.CHANNELS, T.MODES = 8, 4
T.WRITE_LOG = False
exc, out = run_main("--smoke", "--dataset", DATA, "--allow-cpu")
check(exc is None, f"--smoke trains the conditioned model: {exc!r}")
check("val1" in out and "survives" in out and "<- best" in out
      and "chains of" in out, "the log has val1, a rollout and a best")
best = os.path.join(BOX, "Work", "smoke", "models")
found = [os.path.join(d, f) for d, _, fs in os.walk(best) for f in fs
         if f == "fno_ch.pt"]
check(len(found) == 1, "a best checkpoint was written under the smoke root")
if found:
    mdl, ck = T.load_checkpoint(found[0])
    check(mdl.dt_conditioned and ck["config"]["dt_conditioned"] is True
          and ck["run_meta"].get("dt_conditioned") is True,
          "it is dt-conditioned and says so (config and run_meta)")
    check(ck["train_config"]["constants"]["DT_CONDITIONED"] is True
          and "VDT_UNROLL_STAGES" in ck["train_config"]["schedule"],
          "train_config and the schedule stamp carry the vdt settings")
    with torch.no_grad():
        yy = mdl(field(2, 16).float(), torch.tensor([4.0, 20.0]))
    check(bool(torch.isfinite(yy).all()), "it runs at two dts")

# resume: stop after epoch 1 of 2, then continue
reset()
T.CHANNELS, T.MODES, T.WRITE_LOG = 8, 4, False
T.STEPS_PER_EPOCH, T.BATCH_SIZE, T.VAL_EVERY = 3, 4, 1
T.VAL_SEEDS, T.VAL_LATTICE, T.VAL_T_END, T.T_MIN = (5,), 16, 112.0, 0.0
T.VAL1_WINDOWS = 4
calls = []
real_epoch = T.vdt_epoch


def stopping(*a, **k):
    calls.append(1)
    if len(calls) == 2:
        raise KeyboardInterrupt
    return real_epoch(*a, **k)


T.vdt_epoch = stopping
argv = ["--dataset", DATA, "--t-train", "30", "--epochs", "2", "--tag", "vdt",
        "--allow-cpu"]
exc, out = run_main(*argv)
check(isinstance(exc, KeyboardInterrupt), "the first run is interrupted")
state = [os.path.join(d, f) for d, _, fs in os.walk(os.path.join(BOX, "Work", "models"))
         for f in fs if f == "fno_ch_last.pt"]
check(len(state) == 1 and torch.load(state[0], weights_only=False)["epoch"] == 1,
      "the resume state is at epoch 1")
T.vdt_epoch = real_epoch
exc, out = run_main(*argv)
check(exc is None and "[resume]" in out and "at epoch 1" in out
      and "epoch   2/2" in out and "epoch   1/2" not in out,
      f"the second run resumes at epoch 1 and finishes epoch 2: {exc!r}")
st = torch.load(state[0], weights_only=False)
check(st["epoch"] == 2 and st["config"]["dt_conditioned"] is True
      and st.get("ema") is not None, "state at epoch 2, with the EMA")
exc, out = run_main("--dataset", DATA, "--t-train", "30", "--epochs", "2",
                    "--tag", "vdt", "--allow-cpu", "--fixed-step")
check(isinstance(exc, SystemExit) and ("dt_conditioned" in out + str(exc.code)),
      "a --fixed-step run in that folder is refused and names dt_conditioned")

# ---------------------------------------------------------------------------
print("\n7. --fixed-step against the old code")
reset()


def subproc(script, cwd, *flags):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    os.makedirs(cwd, exist_ok=True)
    p = subprocess.run([sys.executable, script, "--smoke", "--dataset", DATA,
                        "--allow-cpu", *flags], cwd=cwd, env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return p.returncode, p.stdout + p.stderr


def epoch_lines(text):
    rows = [ln for ln in text.splitlines() if re.match(r"epoch +\d+/", ln)]
    return [re.sub(r"\d+\.\ds \(\s*\d+ ms/step\)", "T", ln) for ln in rows]


def same_lines(a, b, tol=2e-5):
    """Equal text, numbers equal to ``tol`` (CPU reductions are threaded, so
    the last digit of a loss can differ between two runs of the same code)."""
    num = re.compile(r"-?\d+\.\d+(?:e[+-]?\d+)?")
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if num.sub("#", x) != num.sub("#", y):
            return False
        if any(abs(float(u) - float(v)) > tol
               for u, v in zip(num.findall(x), num.findall(y))):
            return False
    return True


if os.path.exists(os.path.join(OLD_ROOT, "fno_train.py")):
    c_new, o_new = subproc(os.path.join(ROOT, "fno_train.py"),
                           os.path.join(BOX, "new"), "--fixed-step")
    c_old, o_old = subproc(os.path.join(OLD_ROOT, "fno_train.py"),
                           os.path.join(BOX, "old"))
    check(c_new == 0 and c_old == 0, f"both run ({c_new}, {c_old})")
    ln_new, ln_old = epoch_lines(o_new), epoch_lines(o_old)
    check(ln_new and same_lines(ln_new, ln_old),
          "the epoch lines (loss, val1, d/psi, rollout, survival) are "
          "those of the old code (to the last digit)")
    if not same_lines(ln_new, ln_old):
        print("   new:", *ln_new, sep="\n     ")
        print("   old:", *ln_old, sep="\n     ")
else:
    c_new, o_new = subproc(os.path.join(ROOT, "fno_train.py"),
                           os.path.join(BOX, "new"), "--fixed-step")
    check(c_new == 0 and epoch_lines(o_new), "--fixed-step trains")
check("unroll" in o_new and "DT_CONDITIONED" in o_new,
      "the fixed-step log is the old loop's")

# ---------------------------------------------------------------------------
check(project_state() == PROJECT, "nothing was written into the project")
print("\n" + ("ALL PASS" if ok else "FAILURES"))
sys.exit(0 if ok else 1)
