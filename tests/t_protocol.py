"""The shared test protocol: the training seed, the model size, the folder
labels they make, continuing a model of another size, and the job scripts
(ft.sh, fp.sh, ftp.sh, ch_gpu.sh) that take them as the group's do.

The runs are tiny CPU runs of the real fno_train.main, each in a folder of
its own under tests/_scratch.  The scripts are run through bash with stub
python files that log their arguments, or skipped where there is no bash.
"""
import contextlib
import io as _io
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import torch

from _paths import ROOT, SCRATCH

os.chdir(SCRATCH)
import fno_paths as F
import fno_predict as P
import fno_train as T
T.use_fixed_step()

setattr(torch.cuda, "is_available", lambda: False)

D = os.path.join(SCRATCH, "protocol_t")
shutil.rmtree(D, ignore_errors=True)
os.makedirs(D)

ok = True
sink = _io.StringIO()


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


def refused(fn, *a):
    """The message fn raises (ValueError or SystemExit) for these
    arguments, or None."""
    try:
        fn(*a)
    except (ValueError, SystemExit) as exc:
        return str(exc.code if isinstance(exc, SystemExit) else exc)
    return None


print("1. the tag slot of a label")
check(F.compose_tag() == "" and F.compose_tag(None, 0, "") == "",
      "no size, seed 0 and no name: nothing, so a plain label is as it was")
check(F.compose_tag(seed=1) == "s1" and F.compose_tag((24, 8)) == "c24m8"
      and F.compose_tag(name="mine") == "mine",
      "each part alone")
check(F.compose_tag((24, 8), 1, "mine") == "c24m8-s1-mine"
      and F.compose_tag((24, 8), 0, "mine") == "c24m8-mine"
      and F.compose_tag(None, 2, "mine") == "s2-mine",
      "together the order is size, seed, name")
rows = []
for size in (None, (24, 8), (64, 24)):
    for seed in (0, 1, 12):
        for name in ("", "mine", "x-y", "s1x"):
            rows.append((size, seed, name))
back = [F.split_tag(F.compose_tag(*r)) for r in rows]
check(back == [{"size": s, "seed": sd, "name": n} for s, sd, n in rows],
      f"split_tag takes the tag apart again, for {len(rows)} combinations")
fails = []
for size, seed, name in rows:
    lab = F.run_label(F.train_label(1000, lap=True,
                                    tag=F.compose_tag(size, seed, name)),
                      10000)
    got = F.parse_label(lab)
    if (got is None or got["tag"] != F.compose_tag(size, seed, name)
            or not got["lap"] or got["t_train"] != 1000.0
            or got["end"] != 10000.0):
        fails.append(lab)
check(not fails, f"parse_label parses every one of them as a run label "
                 f"{fails[:2]}")
check(F.parse_label("train_t1000_lap_c24m8-s1-mine")
      == {"train": "train_t1000_lap_c24m8-s1-mine", "t_train": 1000.0,
          "lap": True, "nohard": False, "tag": "c24m8-s1-mine",
          "start": None, "end": None},
      "and as a train label, with the old keys only")
bad = [t for t in ("s1", "c24m8", "c24m8-x", "s3-x", "c1m1-s2")
       if refused(F.check_user_tag, t) is None]
fine = [t for t in ("mine", "s1x", "stable", "c24", "m8", "seed1", "c24m8x")
        if refused(F.check_user_tag, t) is not None]
check(not bad and not fine,
      f"a --tag that looks like a size or seed token is refused, the rest "
      f"are not ({bad}, {fine})")
check("--seed" in (refused(F.compose_tag, None, 0, "s1") or "")
      and refused(F.compose_tag, None, -1, "") is not None,
      "compose_tag refuses such a name and a negative seed")

print("\n2. the flags and the label")
DEFAULTS = dict(T.config_constants())
CH0, MO0 = DEFAULTS["CHANNELS"], DEFAULTS["MODES"]


def labelled(argv, **cfg):
    """(the label, what apply_overrides printed) for these flags, with the
    file's settings and these."""
    for k, v in {**DEFAULTS, **cfg}.items():
        setattr(T, k, v)
    T.MODEL_OUT = T.CHECKPOINT = None
    T.ROOT = os.path.join(D, "root")
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        T.apply_overrides(T.parse_args(["--t-train", "1000", *argv]))
        label = T.resolve_outputs(7)
    return label, buf.getvalue()


check(labelled([])[0] == "train_t1000_lap",
      "no flag: train_t1000_lap, the Laplacian head being the default")
lab, out = labelled(["--seed", "1"])
check(lab == "train_t1000_lap_s1" and T.SEED == 1 and T.TAG == "s1"
      and "[cli] SEED 0 -> 1" in out,
      f"--seed 1 adds s1 and sets SEED: {lab}")
check(labelled(["--seed", "0"])[0] == "train_t1000_lap" and T.SEED == 0,
      "--seed 0 changes nothing, so an existing label stays")
check(labelled(["--seed", "2", "--tag", "mine"])[0]
      == "train_t1000_lap_s2-mine",
      "with --tag the seed goes before the name")
lab, out = labelled(["--channels", "24", "--modes", "8"])
check(lab == "train_t1000_lap_c24m8" and (T.CHANNELS, T.MODES) == (24, 8)
      and "[cli] CHANNELS %d -> 24" % CH0 in out
      and "[cli] MODES %d -> 8" % MO0 in out,
      f"--channels and --modes set the size and add c24m8: {lab}")
check(labelled(["--channels", str(CH0), "--modes", str(MO0)])[0]
      == "train_t1000_lap",
      "the file's own size adds no token")
check(labelled(["--modes", "8"])[0] == f"train_t1000_lap_c{CH0}m8"
      and labelled(["--channels", "24"])[0] == f"train_t1000_lap_c24m{MO0}",
      "one flag alone still writes both numbers")
check(labelled(["--channels", "24", "--modes", "8", "--seed", "1", "--tag",
                "mine", "--variant", "base"])[0]
      == "train_t1000_c24m8-s1-mine",
      "all of them: c<C>m<M>-s<N>-<tag> after the variant")
lab, out = labelled(["--channels", "64", "--keep-folder"])
check(lab == "train_t1000_lap" and T.CHANNELS == 64 and "--keep-folder" in out,
      "--keep-folder leaves the size out of the name and says so")
check(labelled(["--channels", "64", "--keep-folder", "--seed", "1"])[0]
      == "train_t1000_lap_s1",
      "and only the size: a seed still goes in")
check("c<C>m<M> and s<N>" in (refused(labelled, ["--tag", "s1"]) or ""),
      "a --tag that is a seed token is refused")
errs = []
for flags in (["--seed", "-1"], ["--channels", "0"], ["--modes", "0"],
              ["--modes", "x"]):
    with contextlib.redirect_stderr(_io.StringIO()):
        errs.append(refused(T.parse_args, flags))
check(all(e == "2" for e in errs), f"a negative seed and a count that is "
                                   f"not above 0 are refused ({errs})")
check(T.max_modes(64) == 31 and T.max_modes(32) == 15 and T.max_modes(2) == 1,
      "a box of n points holds (n - 1) // 2 modes")
T.MODES = 31
check(refused(T.check_modes, 64) is None, "31 modes fit a 64-point box")
T.MODES = 32
msg = refused(T.check_modes, 64) or ""
check("at most 31" in msg and "64" in msg,
      f"and 32 do not: {msg}")
buf = _io.StringIO()
with contextlib.redirect_stdout(buf):
    try:
        T.parse_args(["-h"])
    except SystemExit:
        pass
helptext = re.sub(r"\s+", " ", buf.getvalue())
check("(file: lap)" in helptext and "--seed N" in helptext
      and "--channels C" in helptext and "--modes M" in helptext
      and "--keep-folder" in helptext and "ablation" in helptext,
      "the help text knows the new flags and that lap is the default")

print("\n3. the seed is a training seed, and the config shows it")
L, NF = 32, 40
rng = np.random.default_rng(0)
traj = np.tanh(rng.standard_normal((5, NF, L, L)) * 0.5)
traj -= traj.mean(axis=(-2, -1), keepdims=True)
for i, off in enumerate((-0.4, -0.2, 0.0, 0.2, 0.4)):
    traj[i] += off


def tiny_data(*a, **k):
    dt = T.numpy_dtype()
    return traj[:4].astype(dt), traj[4:].astype(dt)


def no_off_critical(lattice, n_frames, device="cpu", verbose=True):
    return np.zeros((2, n_frames, lattice, lattice), T.numpy_dtype())


setattr(T, "load_trajectories", tiny_data)
setattr(T, "off_critical_trajectories", no_off_critical)
BASE = dict(
    WRITE_LOG=False, EPOCHS=2, STEPS_PER_EPOCH=3, BATCH_SIZE=2, MAX_UNROLL=2,
    BPTT=2, VAL_BATCHES=2, VAL_SEEDS=(11,), VAL_LATTICE=32, VAL_T_END=105.0,
    VAL_T_START=100.0, MONITOR_SEEDS=(21, 22), MONITOR_OFFS=(-0.4, 0.4),
    MONITOR_LATTICE=32, MONITOR_T_END=105.0, VAL_EVERY=1, PATIENCE=0,
    CHANNELS=8, MODES=6, LAYERS=2, MAX_HOURS=None, RESUME=False,
    TRAIN_ON_VAL=True, GT_CACHE=os.path.join(D, "gt"), REPLACE_WAIT=0.0)
DEFAULT_META = T.DEFAULT_RUN_META


def run(where, argv=(), **cfg):
    """T.main in the folder ``where`` under the tiny settings: (what it
    printed, the SystemExit code or the exception, or None)."""
    folder = os.path.join(D, "runs", where)
    os.makedirs(folder, exist_ok=True)
    for k, v in {**DEFAULTS, **BASE, **cfg}.items():
        setattr(T, k, v)
    T.DEFAULT_RUN_META = DEFAULT_META
    T.MODEL_OUT = T.CHECKPOINT = T.ROOT = None
    buf, code = _io.StringIO(), None
    os.chdir(folder)
    try:
        with contextlib.redirect_stdout(buf):
            T.main(list(argv))
    except SystemExit as exc:
        code = exc.code
    except Exception as exc:
        code = exc
    finally:
        os.chdir(SCRATCH)
    return buf.getvalue(), code


def models(where):
    top = os.path.join(D, "runs", where, "Work", "models")
    return sorted(os.listdir(top)) if os.path.isdir(top) else []


def best(where, label):
    p = os.path.join(D, "runs", where, "Work", "models", label, "fno_ch.pt")
    return torch.load(p, map_location="cpu", weights_only=False)


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def only(where):
    got = models(where)
    return got[0] if len(got) == 1 else None


out_a, code_a = run("a")
out_a2, code_a2 = run("a2", ["--seed", "0"])
out_b, code_b = run("b", ["--seed", "1"])
out_c, code_c = run("c", ["--seed", "1"])
la, lb = only("a"), only("b")
check(all(c is None for c in (code_a, code_a2, code_b, code_c))
      and la is not None and lb == f"{la}_s1" and only("c") == lb
      and only("a2") == la,
      f"tiny runs train: seed 0 and no flag give {la}, seed 1 gives {lb}")
if la is not None and lb is not None:
    wa, wa2 = best("a", la)["state_dict"], best("a2", la)["state_dict"]
    wb, wc = best("b", lb)["state_dict"], best("c", lb)["state_dict"]
    check(same(wa, wa2) and same(wb, wc),
          "the same seed trains the same weights, however it is given")
    check(not same(wa, wb),
          "and another seed trains other weights: the flag reaches the "
          "training")
    ck = best("b", lb)
    with open(os.path.join(D, "runs", "b", "Work", "models", lb,
                           "fno_ch_config.json")) as fh:
        cfg = json.load(fh)
    check(cfg["seed"] == 1 and cfg["constants"]["SEED"] == 1
          and cfg["schedule"]["SEED"] == 1 and cfg["run_meta"]["tag"] == "s1"
          and cfg["label"] == lb and cfg["size"] == {"channels": 8,
                                                     "modes": 6}
          and ck["train_config"]["seed"] == 1 and "SEED 0 -> 1" in out_b,
          "the training config, the schedule stamp and the checkpoint all "
          "say seed 1")
    check(P._ckpt_train(ck) == {"seed": 1, "channels": 8, "modes": 6,
                                "laplacian_output": True,
                                "mu_channel": bool(ck["config"]["mu_channel"]),
                                "lowk_fill": bool(ck["config"]["lowk_fill"])}
          and ck["config"]["lowk_fill"] and not ck["config"]["mu_channel"]
          and P._ckpt_train(best("a", la))["seed"] == 0,
          "and the prediction side reads the seed, the size, the head and "
          "the two model options (here the fill on, the mu channel off, as "
          "this suite trains) from the checkpoint")
old = {"config": {"channels": 64, "modes": 24}}
check(P._ckpt_train(old) == {"seed": None, "channels": 64, "modes": 24,
                             "laplacian_output": False, "mu_channel": False,
                             "lowk_fill": False}
      and P._ckpt_train({}) == {"seed": None, "channels": None,
                                "modes": None, "laplacian_output": False,
                                "mu_channel": False, "lowk_fill": False},
      "a checkpoint from before the seed was recorded has no seed, and "
      "loads as the base head, without the two options")
on = P._ckpt_train({"config": {"mu_channel": True, "lowk_fill": True}})
check(on["mu_channel"] is True and on["lowk_fill"] is True,
      "and the two options are read when the checkpoint has them on")

print("\n4. a model of another size, in the folder it has")
out, code = run("k", [], CHANNELS=8)
label = only("k")
check(code is None and label is not None and "_c" not in label,
      f"the old model: CHANNELS 8 in the file, no flag, folder {label}")
out, code = run("k", [], CHANNELS=10, RESUME=True)
check(isinstance(code, str) and "channels: 8 -> 10" in code
      and "--channels 8 --keep-folder" in code
      and "folder of its own" in code and "--no-resume" in code
      and models("k") == [label],
      "the file's new size refuses its state, and the message gives the "
      "flags that go on with it")
out, code = run("k", ["--channels", "8", "--keep-folder"], CHANNELS=10,
                RESUME=True)
check(code is None and "at epoch 2" in out and models("k") == [label],
      "--channels 8 --keep-folder resumes it in the folder it is in")
shot = {}
if label is not None:
    top = os.path.join(D, "runs", "k", "Work", "models", label)
    shot = {n: os.path.getsize(os.path.join(top, n)) for n in os.listdir(top)}
out, code = run("k", ["--channels", "8"], CHANNELS=10, RESUME=True)
new = [m for m in models("k") if m != label]
check(code is None and new == [f"{label}_c8m6"] and "[resume]" not in out,
      f"without --keep-folder the size is in the name and the run starts "
      f"in a folder of its own: {new}")
if label is not None and new:
    top = os.path.join(D, "runs", "k", "Work", "models", label)
    check({n: os.path.getsize(os.path.join(top, n))
           for n in os.listdir(top)} == shot,
          "and the first folder is not touched")
    check(best("k", new[0])["config"]["channels"] == 8,
          "that model has 8 channels")
out, code = run("m", ["--modes", "16"])
check(isinstance(code, str) and "at most 15" in code
      and models("m") == [],
      "--modes beyond what the training boxes hold is refused before "
      "anything is written")

print("\n5. the job scripts: ft.sh, fp.sh, ftp.sh and ch_gpu.sh, as in the group's runs")
JOBS = ("ft.sh", "fp.sh", "ftp.sh", "ch_gpu.sh")
for name in JOBS + ("hpc_storage.sh",):
    with open(os.path.join(ROOT, name), "rb") as fh:
        raw = fh.read()
    sh = raw.decode()
    check(b"\r" not in raw and sh.startswith("#!/bin/bash\n")
          and "set -euo pipefail\n" in sh
          and re.search(r"^#SBATCH --output=\w+_%j\.out$", sh, re.M)
          and re.search(r"^#SBATCH --error=\w+_%j\.err$", sh, re.M),
          f"{name}: LF, set -euo pipefail, its logs next to it as the "
          f"group's are")
# the time limits started at ft.sh 6 h, ftp.sh 10 h and fp.sh 4 h;
# since 2026-10-22 each is set after a job to what it used plus a margin, so
# only their form is checked; fp.sh and ftp.sh end by refreshing the Results
# index
for name in ("ft.sh", "ftp.sh", "fp.sh"):
    with open(os.path.join(ROOT, name)) as fh:
        sh = fh.read()
    last = [ln for ln in sh.splitlines() if ln.startswith("python ")][-1]
    check(re.search(r"^#SBATCH --time=\d{2}:\d{2}:\d{2}$", sh, re.M)
          and (last == "python make_submission.py --overview")
          == (name != "ft.sh") and "val-t-end" not in sh,
          f"{name}: a --time=HH:MM:SS limit, "
          + ("ends with python make_submission.py --overview"
             if name != "ft.sh" else "no results step")
          + ", no --val-t-end (validation is to 10000 whatever END is)")
gone = ("ft_long.sh", "fp_long.sh", "ch_lean.sh", "protocol.sh", "sweep.sh",
        "submit_common.sh")
check(not [n for n in gone if os.path.exists(os.path.join(ROOT, n))],
      "the retired submission scripts are gone from the project")

box = os.path.join(SCRATCH, "protocol_jobs")
shutil.rmtree(box, ignore_errors=True)
os.makedirs(box)


def printed(*argv):
    """(exit code, stdout, stderr) of fno_train.py --print-label."""
    p = subprocess.run([sys.executable, os.path.join(ROOT, "fno_train.py"),
                        *argv, "--print-label"], cwd=box,
                       capture_output=True, text=True, errors="replace")
    return p.returncode, p.stdout.strip(), p.stderr


for argv, want in ((["--t-train", "1000"], "train_t1000_lap"),
                   (["--t-train", "300", "--variant", "base"], "train_t0300"),
                   (["--t-train", "500", "--channels", str(CH0), "--modes",
                     str(MO0), "--seed", "0"], "train_t0500_lap"),
                   (["--t-train", "1000", "--channels", "24", "--modes", "8"],
                    "train_t1000_lap_c24m8"),
                   (["--t-train", "1000", "--seed", "1"], "train_t1000_lap_s1"),
                   (["--t-train", "1000", "--variant", "base", "--channels",
                     "24", "--modes", "8", "--seed", "2"],
                    "train_t1000_c24m8-s2")):
    code, out, err = printed(*argv)
    same_as_training = (argv[1] != "1000"
                        or labelled(argv[2:])[0] == out)
    check(code == 0 and out == want and same_as_training,
          f"--print-label {' '.join(argv)}: {out} (the folder the training "
          f"writes)")
check(os.listdir(box) == [], "--print-label reads no data and writes nothing")
code, out, err = printed()
check(code != 0 and out == "" and "needs --t-train" in err,
      "--print-label without --t-train stops with a message")

bash = shutil.which("bash") or ""
if not bash or "system32" in bash.lower():
    print("      (no bash here; the scripts are not run)")
else:
    for name in JOBS:
        p = subprocess.run([bash, "-n", name], cwd=ROOT,
                           capture_output=True, text=True, errors="replace")
        check(p.returncode == 0, f"bash -n {name}: {p.stderr.strip()}")
    stubs = os.path.join(box, "stubs")
    os.makedirs(stubs)
    os.makedirs(os.path.join(box, "env", "bin"))
    open(os.path.join(box, "env", "bin", "activate"), "w").close()

    def write(name, text, folder=box):
        with open(os.path.join(folder, name), "w", newline="\n") as fh:
            fh.write(text)

    # the cluster's module command, a python that skips the torch banner,
    # and stand-ins that log their arguments; --print-label runs the real
    # fno_train.py, so the label is the one the training would write
    write("module", "#!/bin/bash\nexit 0\n", stubs)
    write("python", '#!/bin/bash\ncase "$*" in *"import torch"*) exit 0;;'
          ' esac\nexec "$REAL_PYTHON" "$@"\n', stubs)
    for name in ("module", "python"):
        os.chmod(os.path.join(stubs, name), 0o755)
    logger = ("import os, sys\n"
              "with open('calls.txt', 'a') as fh:\n"
              "    fh.write(' '.join([os.path.basename(sys.argv[0])]"
              " + sys.argv[1:]) + '\\n')\n")
    write("fno_train.py",
          "import importlib.util, os, sys\n"
          "if '--print-label' in sys.argv:\n"
          "    spec = importlib.util.spec_from_file_location('real_train', "
          "os.path.join(os.environ['REAL_ROOT'], 'fno_train.py'))\n"
          "    assert spec is not None and spec.loader is not None\n"
          "    real = importlib.util.module_from_spec(spec)\n"
          "    spec.loader.exec_module(real)\n"
          "    real.main(sys.argv[1:])\n"
          "    sys.exit(0)\n"
          + logger.replace("import os, sys\n", "")
          + "sys.exit(int(os.environ.get('TRAIN_EXIT', '0')))\n")
    write("fno_predict.py", logger)
    write("make_submission.py", logger)
    write("ch_gpu.py", logger)
    for name in JOBS:
        shutil.copy(os.path.join(ROOT, name), box)

    def job(name, **env):
        """(exit code, the python calls it made, its output)."""
        calls = os.path.join(box, "calls.txt")
        if os.path.exists(calls):
            os.remove(calls)
        # no exported shell functions and no BASH_ENV: on a login shell
        # `module` is one (Lmod's, exported, and defined again through
        # BASH_ENV in every bash a script starts), and it would run in place
        # of the stub, put the cluster's own python first on PATH and bypass
        # every stand-in below
        run_env = {k: v for k, v in os.environ.items()
                   if not k.startswith("BASH_FUNC_")
                   and k not in ("BASH_ENV", "ENV")}
        run_env.update(env, PATH=stubs + os.pathsep + os.environ["PATH"],
                       PYTHONPATH=ROOT, REAL_PYTHON=sys.executable,
                       REAL_ROOT=ROOT)
        p = subprocess.run([bash, name], cwd=box, env=run_env,
                           capture_output=True, text=True, errors="replace")
        got = []
        if os.path.exists(calls):
            with open(calls) as fh:
                got = fh.read().splitlines()
        return p.returncode, got, p.stdout + p.stderr

    def data(end):
        return f"--dataset Work/data/test_t{end}.npz"

    size = f"--channels {CH0} --modes {MO0}"
    code, got, text = job("ft.sh")
    check(code == 0 and got == [f"fno_train.py {data('0500')} --t-train 500 "
                                f"--variant lap {size} --seed 0"],
          f"ft.sh: END 500, lap, the default size, seed 0 ({got})")
    code, got, text = job("ft.sh", END="2000", VARIANT="base", SEED="1")
    check(code == 0 and got == [f"fno_train.py {data('2000')} --t-train 2000 "
                                f"--variant base {size} --seed 1"],
          f"ft.sh END=2000,VARIANT=base,SEED=1: no --val-t-end, the "
          f"validation is to 10000 as for every END ({got})")
    code, got, text = job("ft.sh", END="500", VARIANT="base", NOHARD="1")
    check(code == 0 and got == [f"fno_train.py {data('0500')} --t-train 500 "
                                f"--variant base {size} --seed 0 "
                                f"--no-hard-constraints"],
          f"ft.sh NOHARD=1 adds --no-hard-constraints only ({got})")
    code, got, text = job("ft.sh", END="500", VARIANT="base", NOHARD="1",
                          MASS_WEIGHT="0", TAG="free")
    check(code == 0 and got == [f"fno_train.py {data('0500')} --t-train 500 "
                                f"--variant base {size} --seed 0 "
                                f"--no-hard-constraints --mass-weight 0 "
                                f"--tag free"],
          f"ft.sh NOHARD, MASS_WEIGHT, TAG add their three flags ({got})")
    code, got, text = job("ft.sh", END="500", NOMU="1", TAG="nomu")
    check(code == 0 and got == [f"fno_train.py {data('0500')} --t-train 500 "
                                f"--variant lap {size} --seed 0 --tag nomu "
                                f"--no-mu-channel"],
          f"ft.sh NOMU=1,TAG=nomu adds --tag nomu --no-mu-channel ({got})")
    code, got, text = job("fp.sh", END="300", VARIANT="base")
    check(code == 0 and got == ["fno_predict.py --model "
                                "Work/models/train_t0300/fno_ch.pt "
                                "--views 10000",
                                "make_submission.py --overview"],
          f"fp.sh END=300,VARIANT=base predicts train_t0300 to 10000 ({got})")
    code, got, text = job("fp.sh", CHANNELS="24", MODES="8", SEED="1",
                          T_END="20000")
    check(code == 0 and got == ["fno_predict.py --model "
                                "Work/models/train_t0500_lap_c24m8-s1/"
                                "fno_ch.pt --views 20000",
                                "make_submission.py --overview"],
          f"fp.sh picks the model ft.sh wrote for that size and seed ({got})")
    code, got, text = job("ftp.sh", END="500")
    check(code == 0 and got == [f"fno_train.py {data('0500')} --t-train 500 "
                                f"--variant lap {size} --seed 0",
                                "fno_predict.py --model "
                                "Work/models/train_t0500_lap/fno_ch.pt "
                                "--views 10000",
                                "make_submission.py --overview"],
          f"ftp.sh trains, predicts that model, refreshes the index ({got})")
    code, got, text = job("ftp.sh", TRAIN_EXIT="75")
    check(code == 75 and len(got) == 1 and got[0].startswith("fno_train.py"),
          f"a training stopped by the time limit (exit 75) ends ftp.sh "
          f"before the prediction and the index (exit {code}, {got})")
    code, got, text = job("ch_gpu.sh")
    check(code == 0 and got == ["ch_gpu.py --end 500"],
          f"ch_gpu.sh makes the dataset to END = 500, test_t0500.npz ({got})")
    code, got, text = job("ch_gpu.sh", END="1000")
    check(code == 0 and got == ["ch_gpu.py --end 1000"],
          f"and END=1000 the one to 1000, test_t1000.npz ({got})")
    code, got, text = job("ch_gpu.sh", END="300")
    check(code == 0 and got == ["ch_gpu.py --end 300"],
          f"and END=300 the one to 300, test_t0300.npz ({got})")
shutil.rmtree(box, ignore_errors=True)

print("\n" + ("PROTOCOL PASS" if ok else "PROTOCOL FAIL"))
sys.exit(0 if ok else 1)
