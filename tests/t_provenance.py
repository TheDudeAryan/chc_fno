"""Which checkpoint wrote a result, and which seeds it was scored on.

Checkpoints here are a few bytes in a file and every rollout is synthetic, so
nothing is trained or predicted.  main() runs in a box under _scratch (its
own Results/, Work/ and Archive/) with the model loader and the case function
stubbed, and the box is compared with a listing taken before each run: a run
that must not write anything may not add a file, and nothing may appear in
the project folder.

  1. the checkpoint fingerprint: its value, and where it is recorded
  2. a finished case is kept or redone by the fingerprint in its record
  3. a results folder of another checkpoint is moved to Archive/superseded;
     so is one of the same checkpoint made in other numerics (a stamp of
     precision, TF32 and the low-k fill), or with no stamp
  4. the seed sets
"""
import contextlib
import errno
import hashlib
import io
import json
import os
import re
import shutil
import sys
import types
from typing import Any
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from _paths import SCRATCH, project_state

BOX = os.path.join(SCRATCH, "provenance_box")
STASH = os.path.join(SCRATCH, "provenance_stash")
for _d in (BOX, STASH):
    shutil.rmtree(_d, ignore_errors=True)
    os.makedirs(_d)
os.chdir(BOX)
PROJECT = project_state()

import fno_paths
fno_paths.WORK = os.path.join(BOX, "Work")
import fno_predict as P
# this suite is about the temporary cache and what supersedes it; the
# records and meta.json that t_records.py covers would sit in the same
# folder and change what it counts
P.RECORDS = False
import fno_train as T
T.use_fixed_step()

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


START = {k: v for k, v in vars(P).items() if k.isupper()}
# the checks assume the checkpoint's own config decides the fill, whatever
# fno_predict's LOWK_FILL is set to at the time
START["LOWK_FILL"] = None
# and the checkpoint's own config the mu clamp (it only counts for a model
# with a mu channel, which these stand-ins lack unless a check gives one)
START["MU_CLIP"] = None
CALLS = []
LABEL = "train_t0100_pred_t02000"
CKPT = fno_paths.model_path("train_t0100")
RESULTS = os.path.join("Results", LABEL)
CACHE = os.path.join(fno_paths.rollout_root(), LABEL)
# FNO's own outputs of a run (case summaries, timing report, bookkeeping)
# are in the cache's extras/, not in the Results folder


def XT(*parts):
    return os.path.join(CACHE, "extras", *parts)


BOOKKEEPING = "submission_manifest.json"
ARCHIVE = os.path.join("Archive", "superseded")
OLD_ROLLOUT = "fno_ch_l128_seed99999_0123456789ab.npz"


class FakeModel:
    config = {"backbone": "physical_k", "hard_mass_conservation": True,
              "residual_prediction": True, "mode_scaling": 1.0,
              "zero_dc": True}

    def n_parameters(self):
        return 0

    def parameters(self):
        yield types.SimpleNamespace(dtype="float32")


P.load_checkpoint = lambda path, device=None: (
    FakeModel(), {"best": {"epoch": 1, "score": 0.9, "survival": 2000.0}})
rng = np.random.default_rng(0)


def fake_results(off, lat, seeds):
    """Synthetic rollouts, shaped as evaluate_seed returns them."""
    t = np.arange(P.START_TIME + 1, P.END_TIME + 1)
    res = []
    for i, _ in enumerate(seeds):
        r2 = np.clip(1.0 - (t - t[0]) / 9000.0 - 0.002 * i, -1, 1)
        res.append({
            "times": t, "r2": r2, "mass_drift": np.full(t.size, 1e-9),
            "mean_true": np.full(t.size, off),
            "mean_pred": np.full(t.size, off) + 1e-9,
            "diag_times": t[::200],
            "energy_true": np.linspace(1, 0.5, t[::200].size),
            "energy_pred": np.linspace(1, 0.5, t[::200].size),
            "t_stop": float(t[-1]), "stopped_early": False,
            "diverged_at": None,
            "L_true": {k: np.linspace(4, 20, t[::200].size)
                       for k in ("L_zero", "L_peak", "L_moment", "L_grad")},
            "L_pred": {k: np.linspace(4, 20, t[::200].size)
                       for k in ("L_zero", "L_peak", "L_moment", "L_grad")},
            "snapshots": {tt: (rng.standard_normal((lat, lat)),
                               rng.standard_normal((lat, lat)))
                          for tt in P.SNAPSHOT_TIMES},
            "gt_seconds": 1.0, "gt_cached": False, "gt_steps": 1,
            "fno_seconds": 0.5, "fno_steps": 1900,
            "video_t": np.asarray([]), "video_true": None,
            "video_pred": None})
    return res


def note(off, lattice, seeds):
    """Remember a case run.  A case with nothing to show is asked for again
    as the l = 128 sweep point, so only the composition case is kept."""
    if lattice is None:
        CALLS.append((off, lattice, seeds))


def real_run_case(model, device, off, lattice=None, seeds=None):
    """A case written by the real write_case_outputs, with its cache key."""
    lat = P.LATTICE if lattice is None else lattice
    seeds = P.seeds_for(off) if seeds is None else tuple(seeds)
    note(off, lattice, seeds)
    return P.write_case_outputs(fake_results(off, lat, seeds), seeds, off,
                                lattice, key=P._cache_key(off, lat))


def lite_run_case(model, device, off, lattice=None, seeds=None):
    """A case that only notes it was asked for, and the seeds it was given."""
    note(off, lattice, P.seeds_for(off) if seeds is None else tuple(seeds))
    return None


def reset():
    for name, value in START.items():
        setattr(P, name, value)
    P.WRITE_LOG = False
    P.PER_SEED_FIGURES = False


def run(*flags, case=lite_run_case, seeds="2", model=CKPT, lowk=None,
        clip=None):
    """main() on the critical case at l = 128: (its status or the SystemExit
    that ended it, its log).  ``lowk`` is P.LOWK_FILL and ``clip`` P.MU_CLIP
    for the run."""
    reset()
    P.LOWK_FILL = lowk
    P.MU_CLIP = clip
    P.run_case = case
    CALLS.clear()
    argv = ["--model", model, "--eval-mode", "critical", "--sizes", "128",
            "--no-video"]
    if seeds:
        argv += ["--seeds", seeds]
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            code = P.main(argv + list(flags))
    except SystemExit as exc:
        code = exc
    return code, log.getvalue()


def said(code):
    return str(code.code) if isinstance(code, SystemExit) else ""


def overrides(*argv):
    reset()
    with contextlib.redirect_stdout(io.StringIO()):
        P.apply_overrides(P.parse_args(list(argv)))


def digest(folder):
    """Every file under folder by its sha256, and every folder as None."""
    out = {}
    for here, dirs, files in os.walk(folder):
        for d in dirs:
            out[os.path.relpath(os.path.join(here, d), folder)] = None
        for f in files:
            path = os.path.join(here, f)
            with open(path, "rb") as fh:
                out[os.path.relpath(path, folder)] = hashlib.sha256(
                    fh.read()).hexdigest()
    return out


def files_only(folder):
    return {k: v for k, v in digest(folder).items() if v is not None}


def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def put_model(payload, when, path=CKPT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(payload)
    os.utime(path, (when, when))
    return hashlib.sha256(payload).hexdigest()[:12]


def archived():
    return sorted(os.listdir(ARCHIVE)) if os.path.isdir(ARCHIVE) else []


def manifest_of(folder):
    return read_json(os.path.join(folder, "manifest.json"))


def forge(folder, fingerprint=None, nbytes=None, mtime=None,
          numerics: Any = True):
    """A manifest.json that names a checkpoint, as an earlier run left it:
    with the numerics of a run of the present settings (True), with a stamp
    given, or with none (None), as a run from before the stamp."""
    run_ = {"fingerprint": fingerprint, "model_bytes": nbytes,
            "model_mtime": mtime}
    if numerics is not None:
        run_["numerics"] = P.numerics_of() if numerics is True else numerics
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "manifest.json"), "w") as fh:
        json.dump({"last_run": run_, "compositions": {"+0.00": {
            "run": run_}}, "sizes": {}}, fh)
    with open(os.path.join(folder, "kept.txt"), "w") as fh:
        fh.write("an earlier run's file")


def wipe():
    for d in ("Results", "Archive", fno_paths.rollout_root(), "elsewhere"):
        shutil.rmtree(d, ignore_errors=True)


def old_cache(folder=CACHE, name=OLD_ROLLOUT):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name), "wb") as fh:
        fh.write(b"a rollout of the old model")


A_BYTES, A_TIME = b"checkpoint A weights " * 50, 1_700_000_000
B_BYTES, B_TIME = b"checkpoint B, retrained " * 60, 1_700_100_000
FP_A = put_model(A_BYTES, A_TIME)

print("1. the checkpoint fingerprint")
check(fno_paths.fingerprint(CKPT) == hashlib.sha256(A_BYTES).hexdigest()[:12]
      and re.fullmatch("[0-9a-f]{12}", FP_A) is not None,
      f"the first 12 hex digits of the file's sha256: {FP_A}")
plain = os.path.join(BOX, "plain.pt")
real_sha = hashlib.sha256
with mock.patch.object(fno_paths.hashlib, "sha256", wraps=real_sha) as sha:
    put_model(b"one", 1_600_000_000, plain)
    sha.reset_mock()
    first = fno_paths.fingerprint(plain)
    again = fno_paths.fingerprint(plain)
    check(first == again and sha.call_count == 1,
          f"hashed once, then kept ({sha.call_count} hash for 2 asks)")
    put_model(b"another", 1_600_000_100, plain)
    sha.reset_mock()
    check(fno_paths.fingerprint(plain) != first and sha.call_count == 1,
          "a file that changed is hashed again")
os.remove(plain)

wipe()
with mock.patch.object(fno_paths.hashlib, "sha256", wraps=real_sha) as sha:
    code, log = run(case=real_run_case)
check(sha.call_count == 1,
      f"a whole run, its case records, manifest and timing report, hashes "
      f"the checkpoint once ({sha.call_count})")
check(code == 0 and CALLS == [(0.0, None, P.make_eval_seeds(2))],
      f"a first run with checkpoint A rolls the case out: {code}, {CALLS}")
man = manifest_of(RESULTS)
book = read_json(XT(BOOKKEEPING))
entry = (book.get("compositions") or {}).get("+0.00") or {}
check(man.get("checkpoint_fingerprint") == FP_A
      and (book.get("last_run") or {}).get("fingerprint") == FP_A
      and (entry.get("run") or {}).get("fingerprint") == FP_A,
      "the run folder's manifest.json names it, and so do last_run and the "
      "case's own run record in the bookkeeping in extras/")
check((book.get("checkpoints") or {}).get("by_fingerprint")
      == {FP_A: ["compositions +0.00", "sizes 128"]},
      f"and the bookkeeping groups the cases by it: "
      f"{(book.get('checkpoints') or {}).get('by_fingerprint')}")
case_dir = XT("psi0+0.00_l128")
rec = read_json(os.path.join(case_dir, "fno_ch_summary.json")).get("record")
check((rec or {}).get("fingerprint") == FP_A,
      "the case record in its summary holds it")
timing = read_json(XT("fno_ch_timing.json"))
with open(XT("fno_ch_timing.txt")) as fh:
    timing_text = fh.read()
check(not any(f.startswith("fno_ch_timing") for f in os.listdir(RESULTS)),
      "the timing report is in extras/, not in the Results folder")
check((timing.get("meta") or {}).get("fingerprint") == FP_A
      and f"checkpoint fingerprint    {FP_A}" in timing_text,
      "the timing report gives it, in the json and in the text")


def value_column(line):
    head = re.match(r"  \S+(?: \S+)* {2,}", line)
    return head.end() if head else -1


columns = {value_column(ln)
           for ln in timing_text.splitlines()
           if ln.startswith(("  device", "  checkpoint fingerprint",
                             "  model parameters", "  one model step"))}
check(len(columns) == 1, f"and its line lines up with the others {columns}")
shutil.copytree(RESULTS, os.path.join(STASH, "results"))
shutil.copytree(XT(), os.path.join(STASH, "extras"))


def stage(fingerprint_gone=False):
    """Results of checkpoint A as the run above left them (the run folder
    and the extras of its cache), and A's file."""
    wipe()
    shutil.copytree(os.path.join(STASH, "results"), RESULTS)
    shutil.copytree(os.path.join(STASH, "extras"), XT())
    put_model(A_BYTES, A_TIME)
    if fingerprint_gone:
        # a manifest from before fingerprints: the old schema, whose runs
        # name the file by size and time only
        man = read_json(XT(BOOKKEEPING))
        runs = [man["last_run"]] + [e["run"] for k in ("compositions", "sizes")
                                    for e in man[k].values()]
        for r in runs:
            r.pop("fingerprint")
        for path in (os.path.join(RESULTS, "manifest.json"),
                     XT(BOOKKEEPING)):
            with open(path, "w") as fh:
                json.dump(man, fh)


print("\n2. a finished case: kept, or redone when its fingerprint differs")
stage()
code, log = run()
check(code == 0 and CALLS == [] and "already finished" in log,
      f"the same checkpoint, window and seeds: not rolled out again {CALLS}")
summary = os.path.join(case_dir, "fno_ch_summary.json")
stats = read_json(summary)
stats["record"]["fingerprint"] = "0" * 12
with open(summary, "w") as fh:
    json.dump(stats, fh)
code, log = run()
check(code == 0 and CALLS == [(0.0, None, P.make_eval_seeds(2))]
      and "already finished" not in log,
      f"a record that names another fingerprint is rolled out again {CALLS}")
stage()
stats = read_json(summary)
stats["record"].pop("fingerprint")
with open(summary, "w") as fh:
    json.dump(stats, fh)
code, log = run()
check(code == 0 and CALLS == [] and "already finished" in log,
      f"a record from before fingerprints still counts {CALLS}")
stage()
check(read_json(summary)["record"]["key"] == P._cache_key(0.0, 128),
      "the cache key is still made from the file's size and time")
put_model(b"checkpoint X weights " * 50, A_TIME)
fno_paths.forget_fingerprints()
check(P._cache_key(0.0, 128) == read_json(summary)["record"]["key"]
      and fno_paths.fingerprint(CKPT) != FP_A,
      "so another file of the same size and time has the same key: the "
      "fingerprint is what tells them apart")

print("\n3. a results folder of another checkpoint is superseded")
FP_B = put_model(B_BYTES, B_TIME)
check(FP_B != FP_A, f"checkpoint B has its own fingerprint {FP_B}")
stage()
before_results = digest(RESULTS)
before_extras = files_only(XT())
old_cache()
put_model(B_BYTES, B_TIME)
code, log = run()
dirs = archived()
want = re.compile(rf"{LABEL}_{FP_A}_\d{{8}}-\d{{6}}")
check(code == 0 and len(dirs) == 1 and want.fullmatch(dirs[0]) is not None,
      f"B takes over: the old folder goes to Archive/superseded/<label>_"
      f"<old fingerprint>_<time>: {dirs}")
home = os.path.join(ARCHIVE, dirs[0]) if dirs else ARCHIVE
check(digest(os.path.join(home, "Results", LABEL)) == before_results,
      "under Results/<label>, every file and folder with its bytes kept")
check(files_only(os.path.join(home, "rollout_cache", LABEL))
      == {OLD_ROLLOUT: hashlib.sha256(
          b"a rollout of the old model").hexdigest(),
          **{os.path.join("extras", k): v for k, v in before_extras.items()}}
      and not os.path.exists(os.path.join(CACHE, OLD_ROLLOUT)),
      "the old model's rollout cache goes with it, under rollout_cache/, "
      "with its extras/ (case summaries, bookkeeping) as they were")
book = read_json(XT(BOOKKEEPING))
check((book.get("last_run") or {}).get("fingerprint") == FP_B
      and not os.path.exists(case_dir) and CALLS
      == [(0.0, None, P.make_eval_seeds(2))],
      "and the run starts fresh: only B's bookkeeping, no old case, the case "
      "rolled out again")
check("SUPERSEDED" in log and f"moved {RESULTS} -> " in log
      and f"{os.path.join(home, 'Results', LABEL)}" in log,
      "the log says loudly what moved where")

stage()
old_cache()
put_model(B_BYTES, B_TIME)
reset()
P.MODEL_PATH = CKPT
mine = f"fno_ch_l128_seed99999_{P._cache_key(0.0, 128)}.npz"
old_cache(name=mine)
code, log = run()
check(len(archived()) == 1
      and os.listdir(os.path.join(ARCHIVE, archived()[0], "rollout_cache",
                                  LABEL)) == ["extras"]
      and sorted(f for f in os.listdir(CACHE) if f.endswith(".npz"))
      == sorted([OLD_ROLLOUT, mine]),
      "a cache holding rollouts of this model stays, with the old ones in "
      "it; only its extras/ goes with the old results")

stage()
old_cache("elsewhere")
put_model(B_BYTES, B_TIME)
code, log = run("--result-cache", "elsewhere")
check(len(archived()) == 1
      and os.path.exists(os.path.join("elsewhere", OLD_ROLLOUT))
      and "stay" in log,
      "a cache outside Work/rollout_cache is never moved")

stage()
old_cache()
code, log = run()
check(code == 0 and archived() == [] and CALLS == []
      and digest(case_dir) == digest(
          os.path.join(STASH, "extras", "psi0+0.00_l128"))
      and os.path.exists(os.path.join(CACHE, OLD_ROLLOUT)),
      "the same checkpoint is a top-up: nothing moves, the finished case "
      "and the cache are left as they were")

print("\n   a manifest from before fingerprints is judged by size and time")
stage(fingerprint_gone=True)
code, log = run()
check(code == 0 and archived() == [] and CALLS == [],
      "the same file by size and time: left alone")
stage(fingerprint_gone=True)
put_model(B_BYTES, A_TIME)
code, log = run()
check(len(archived()) == 1 and archived()[0].startswith(f"{LABEL}_unknown_"),
      f"another size: moved, and 'unknown' stands for the fingerprint: "
      f"{archived()}")
stage(fingerprint_gone=True)
put_model(b"checkpoint X weights " * 50, A_TIME + 5)
code, log = run()
check(len(archived()) == 1 and archived()[0].startswith(f"{LABEL}_unknown_"),
      f"another time with the same size: moved too: {archived()}")

print("\n   a fingerprint outranks size and time")
stage()
put_model(b"checkpoint X weights " * 50, A_TIME)
code, log = run()
check(len(archived()) == 1 and archived()[0].startswith(f"{LABEL}_{FP_A}_"),
      "the same size and time but other bytes is another checkpoint")

print("\n   the same checkpoint in other numerics")
DEFAULT_NUM = {"precision": "fp32", "tf32": False, "lowk_fill": False,
               "mu_clip": None}
FILL_NUM = dict(DEFAULT_NUM, lowk_fill=True)
CLIP_NUM = dict(DEFAULT_NUM, mu_clip=1.2)
stashed_man = read_json(os.path.join(STASH, "results", "manifest.json"))
stashed_book = read_json(os.path.join(STASH, "extras", BOOKKEEPING))
stashed_entry = stashed_book["compositions"]["+0.00"]["run"]
check(P.numerics_of() == DEFAULT_NUM and P.numerics_of(FakeModel()) == DEFAULT_NUM
      and P.OLD_NUMERICS == {"precision": "fp32", "tf32": True,
                             "lowk_fill": False, "mu_clip": None},
      "a run is stamped fp32, TF32 off, no fill, no mu clamp; a run before "
      "the stamp is read as fp32 with TF32 on")
check(stashed_man.get("checkpoint_numerics") == DEFAULT_NUM
      and stashed_book["last_run"].get("numerics") == DEFAULT_NUM
      and stashed_entry.get("numerics") == DEFAULT_NUM
      and read_json(os.path.join(STASH, "extras", "psi0+0.00_l128",
                                 "fno_ch_summary.json"))["record"].get(
                                     "numerics") == DEFAULT_NUM,
      "the run folder's manifest, last_run, the case's run record and its "
      "summary carry the stamp")
check(P.numerics_of(types.SimpleNamespace(config={"lowk_fill": True}))
      ["lowk_fill"] is True, "a model whose config has the fill is stamped so")
P.LOWK_FILL = True
check(P.numerics_of(FakeModel()) == FILL_NUM,
      "and LOWK_FILL overrides the model's own")
P.LOWK_FILL = None
P.MODEL_PATH = CKPT
key_default = P._cache_key(0.0, 128)
P.LOWK_FILL = True
key_fill = P._cache_key(0.0, 128)
P.LOWK_FILL = None
check(key_default != key_fill,
      "the cache key follows the numerics, so a cached rollout of other "
      "numerics is never reused")

stage()
old_cache()
before_results = digest(RESULTS)
torch.backends.cudnn.allow_tf32 = True    # PyTorch's own default
code, log = run(case=real_run_case, lowk=True)
check(torch.backends.cudnn.allow_tf32 is False
      and torch.backends.cuda.matmul.allow_tf32 is False
      and "[setup] precision=fp32, TF32 off" in log
      and "[setup] low-k weight fill on (LOWK_FILL)" in log,
      "main() sets the precision flags before the model runs, and says so "
      "in a [setup] line, with the fill")
dirs = archived()
want = re.compile(rf"{LABEL}_{FP_A}-fp32_\d{{8}}-\d{{6}}")
check(code == 0 and len(dirs) == 1 and want.fullmatch(dirs[0]) is not None,
      f"the fill turned on: the old folder goes to Archive/superseded/"
      f"<label>_<fingerprint>-<old numerics>_<time>: {dirs}")
home = os.path.join(ARCHIVE, dirs[0]) if dirs else ARCHIVE
check(digest(os.path.join(home, "Results", LABEL)) == before_results
      and os.path.isfile(os.path.join(home, "rollout_cache", LABEL,
                                      OLD_ROLLOUT)),
      "with every file and its bytes, and the old cache beside it")
check("SUPERSEDED" in log and "other numerics" in log
      and "low-k fill off" in log and "low-k fill on" in log
      and "of checkpoint" not in log,
      "the log says the numerics differ, not that the checkpoint does")
check(manifest_of(RESULTS).get("checkpoint_numerics") == FILL_NUM
      and CALLS == [(0.0, None, P.make_eval_seeds(2))]
      and "already finished" not in log,
      "the run starts fresh: the case rolled out again, the new manifest "
      "stamped with the fill")
code, log = run(lowk=True)
check(code == 0 and len(archived()) == 1 and CALLS == []
      and "already finished" in log,
      "the same stamp again resumes: nothing moves, the case is kept")

stage()
for path in (os.path.join(RESULTS, "manifest.json"),):
    man = read_json(path)
    man.pop("checkpoint_numerics")
    with open(path, "w") as fh:
        json.dump(man, fh)
code, log = run()
dirs = archived()
check(code == 0 and len(dirs) == 1
      and re.fullmatch(rf"{LABEL}_{FP_A}-fp32_tf32_\d{{8}}-\d{{6}}", dirs[0])
      and "fp32, TF32 on" in log and CALLS == [(0.0, None,
                                                 P.make_eval_seeds(2))],
      f"a manifest with no stamp is a run with TF32 on: superseded by a "
      f"run with it off ({dirs})")
wipe()
put_model(A_BYTES, A_TIME)
forge(RESULTS, FP_A, numerics=None)
code, log = run()
check(len(archived()) == 1 and archived()[0].startswith(f"{LABEL}_{FP_A}-")
      and CALLS == [(0.0, None, P.make_eval_seeds(2))],
      "so is a last_run and case run with none")
wipe()
put_model(A_BYTES, A_TIME)
forge(RESULTS, FP_A, numerics={"tf32": False})
code, log = run()
check(archived() == [] and os.path.isfile(os.path.join(RESULTS, "kept.txt")),
      "a stamp that lacks a key is read with the old value there, so "
      "{'tf32': False} is the present numerics less the precision")
wipe()
put_model(A_BYTES, A_TIME)
forge(RESULTS, FP_A, numerics=P.OLD_NUMERICS)
code, log = run()
check(len(archived()) == 1,
      "and the stamp of the old numerics itself is superseded")

stage()
snap = digest(BOX)
code, log = run("--no-supersede", lowk=True)
check(isinstance(code, SystemExit) and "other numerics" in said(code)
      and RESULTS in said(code) and "--no-supersede" in said(code)
      and digest(BOX) == snap and CALLS == [],
      f"--no-supersede stops for it too, and moves and writes nothing: "
      f"{said(code)[:100]}")

print("\n   the same checkpoint with another mu clamp")


class MuModel(FakeModel):
    """A model with a mu channel; its own clamp, if any, is in its config."""

    def __init__(self, clip=None):
        self.config = dict(FakeModel.config, mu_channel=True, mu_clip=clip)


P.LOWK_FILL = P.MU_CLIP = None
check(P.numerics_of(MuModel()) == DEFAULT_NUM
      and P.numerics_of(MuModel(1.0))["mu_clip"] == 1.0
      and P.numerics_of(FakeModel()) == DEFAULT_NUM,
      "the stamp has the model's own clamp when it has a mu channel, and "
      "none for a model without one")
P.MU_CLIP = 1.2
check(P.numerics_of(MuModel()) == CLIP_NUM
      and P.numerics_of(MuModel(1.0)) == CLIP_NUM
      and P.numerics_of(FakeModel()) == DEFAULT_NUM
      and P.numerics_of()["mu_clip"] is None,
      "MU_CLIP, when set, decides it for a model with a mu channel, and "
      "changes nothing for one without, or for settings with no model")
P.MU_CLIP = None
check("mu clip" not in P.numerics_text(DEFAULT_NUM)
      and P.numerics_text(CLIP_NUM).endswith(", mu clip 1.2")
      and P.numerics_tag(DEFAULT_NUM) == "fp32"
      and P.numerics_tag(CLIP_NUM) == "fp32_clip1.2"
      and P.numerics_tag({"precision": "fp32", "tf32": False,
                          "lowk_fill": True, "mu_clip": 1.2})
      == "fp32_lowk_clip1.2"
      and P.numerics_text({"precision": "fp32", "tf32": False,
                           "lowk_fill": False}).endswith("low-k fill off"),
      "the text and the folder-name tag name the clamp only when there is "
      "one, and a stamp with no such key still reads")
P.MODEL_PATH = CKPT
keys = {}
for name, num in (("none", DEFAULT_NUM), ("clip", CLIP_NUM),
                  ("other", dict(DEFAULT_NUM, mu_clip=1.0)),
                  ("before", {k: v for k, v in FILL_NUM.items()
                              if k != "mu_clip"}),
                  ("fill", FILL_NUM)):
    with mock.patch.object(P, "numerics_now", lambda n=num: dict(n)):
        keys[name] = P._cache_key(0.0, 128)
check(len(set(keys[k] for k in ("none", "clip", "other"))) == 3
      and keys["before"] == keys["fill"],
      "the cache key follows the clamp, and a stamp with none has the key it "
      "had before there was one")

P.load_checkpoint = lambda path, device=None: (
    MuModel(), {"best": {"epoch": 1, "score": 0.9, "survival": 2000.0}})
stage()
old_cache()
before_results = digest(RESULTS)
code, log = run(case=real_run_case, clip=1.2)
dirs = archived()
check(code == 0 and len(dirs) == 1
      and re.fullmatch(rf"{LABEL}_{FP_A}-fp32_\d{{8}}-\d{{6}}", dirs[0])
      and digest(os.path.join(ARCHIVE, dirs[0], "Results", LABEL))
      == before_results
      and "SUPERSEDED" in log and "other numerics" in log
      and "mu clip 1.2" in log and "[setup] mu-channel clamp |psi| <= 1.2 "
      "(MU_CLIP)" in log and "of checkpoint" not in log,
      f"the clamp turned on: the old results go to Archive/superseded as "
      f"another numerics, and the log says so ({dirs})")
check(manifest_of(RESULTS).get("checkpoint_numerics") == CLIP_NUM
      and CALLS == [(0.0, None, P.make_eval_seeds(2))]
      and "already finished" not in log,
      "the run starts fresh, stamped with the clamp")
code, log = run(clip=1.2)
check(code == 0 and len(archived()) == 1 and CALLS == []
      and "already finished" in log,
      "the same clamp again resumes: nothing moves, the case is kept")
code, log = run(case=real_run_case, clip=1.0)
dirs = archived()
check(code == 0 and len(dirs) == 2
      and any(re.fullmatch(rf"{LABEL}_{FP_A}-fp32_clip1.2_\d{{8}}-\d{{6}}", d)
              for d in dirs)
      and manifest_of(RESULTS).get("checkpoint_numerics")
      == dict(DEFAULT_NUM, mu_clip=1.0),
      f"another clamp supersedes it again, the folder named for the old one "
      f"({dirs})")
code, log = run(case=real_run_case)
check(code == 0 and len(archived()) == 3
      and manifest_of(RESULTS).get("checkpoint_numerics") == DEFAULT_NUM
      and "[setup] mu-channel clamp off (the checkpoint's config)" in log,
      "and none (the checkpoint's config) supersedes that")
snap = digest(BOX)
code, log = run("--no-supersede", clip=1.2)
check(isinstance(code, SystemExit) and "other numerics" in said(code)
      and digest(BOX) == snap and CALLS == [],
      "--no-supersede stops for a clamp too, and moves and writes nothing")
P.load_checkpoint = lambda path, device=None: (
    FakeModel(), {"best": {"epoch": 1, "score": 0.9, "survival": 2000.0}})
stage()
code, log = run(clip=1.2)
check(code == 0 and archived() == [] and CALLS == []
      and "[setup] mu-channel clamp not used (the model has no mu channel)"
      in log,
      "a model without a mu channel is not superseded by it, and the log "
      "says the clamp is not used")

print("\n   --no-supersede refuses")
stage()
old_cache()
put_model(B_BYTES, B_TIME)
snap = digest(BOX)
code, log = run("--no-supersede")
check(isinstance(code, SystemExit) and FP_A in said(code)
      and RESULTS in said(code) and "--no-supersede" in said(code),
      f"the run stops, naming the folder and the fingerprint: {said(code)}")
check(digest(BOX) == snap and CALLS == [],
      "and nothing was moved, written or rolled out")
stage()
code, log = run("--no-supersede")
check(code == 0 and CALLS == [] and archived() == [],
      "with the same checkpoint it runs as usual")

print("\n   a move that fails stops the run before it writes")
stage()
old_cache()
put_model(B_BYTES, B_TIME)
snap = digest(BOX)
with mock.patch.object(os, "replace", side_effect=PermissionError(
        errno.EACCES, "in use by another process")):
    code, log = run()
check(isinstance(code, SystemExit) and "could not move" in said(code)
      and "before writing anything" in said(code),
      f"a clear message: {said(code)[:110]}")
check(files_only(BOX) == {k: v for k, v in snap.items() if v is not None}
      and CALLS == [] and not archived(),
      "no file written or moved, no case run, no Archive left behind")

stage()
old_cache()
put_model(B_BYTES, B_TIME)
real_replace = os.replace
tries = []


def second_fails(src, dst):
    tries.append(src)
    if len(tries) == 2:
        raise PermissionError(errno.EACCES, "in use by another process")
    return real_replace(src, dst)


with mock.patch.object(os, "replace", side_effect=second_fails):
    code, log = run()
check(isinstance(code, SystemExit) and "Already moved" in said(code)
      and CALLS == [] and os.path.isdir(CACHE)
      and not os.path.exists(RESULTS),
      "if the cache is what fails, it says what was already moved, and "
      "still writes nothing")

print("\n   across devices the folder is copied over, not lost")
stage()
put_model(B_BYTES, B_TIME)
tries.clear()


def other_device(src, dst):
    if os.path.abspath(src) != os.path.abspath(RESULTS):
        return real_replace(src, dst)
    tries.append(src)
    raise OSError(errno.EXDEV, "cross-device link")


with mock.patch.object(os, "replace", side_effect=other_device):
    with mock.patch.object(P.shutil, "move", wraps=shutil.move) as mv:
        code, log = run()
check(code == 0 and mv.call_count >= 1 and len(archived()) == 1
      and digest(os.path.join(ARCHIVE, archived()[0], "Results", LABEL))
      == before_results,
      "an EXDEV from os.replace falls back to shutil.move, bytes kept")

print("\n   a damaged manifest says nothing, so the folder stays")
stage()
with open(os.path.join(RESULTS, "manifest.json"), "w") as fh:
    fh.write("{not json")
put_model(B_BYTES, B_TIME)
code, log = run()
check(code == 0 and archived() == [],
      "no readable manifest, no other checkpoint named: not moved")

print("\n   with views, every view's folder is looked at")
VIEWS = ("--views", "1000,2000")
t1000 = os.path.join("Results", "train_t0100_pred_t01000")
t2000 = os.path.join("Results", "train_t0100_pred_t02000")
t0500 = os.path.join("Results", "train_t0100_pred_t00500")
wipe()
put_model(B_BYTES, B_TIME)
for folder in (t1000, t2000, t0500):
    forge(folder, FP_A)
old_cache()
code, log = run(*VIEWS)
names = archived()
check(len(names) == 2 and all(f"_{FP_A}_" in n for n in names),
      f"both views' folders were another checkpoint's: {names}")
moved_dirs = sorted(d for n in names for d in os.listdir(
    os.path.join(ARCHIVE, n, "Results")))
check(moved_dirs == ["train_t0100_pred_t01000", "train_t0100_pred_t02000"],
      f"each under its own label: {moved_dirs}")
with_cache = [n for n in names if os.path.isdir(
    os.path.join(ARCHIVE, n, "rollout_cache", LABEL))]
check(len(with_cache) == 1 and with_cache[0].startswith(LABEL),
      f"the cache goes with the folder it is named for: {with_cache}")
check(os.path.isfile(os.path.join(t0500, "kept.txt"))
      and (manifest_of(t0500).get("last_run") or {}).get("fingerprint")
      == FP_A,
      "a folder of the same family that this run does not write stays")
wipe()
forge(t1000, FP_A)
forge(t2000, FP_B)
code, log = run(*VIEWS)
names = archived()
check(len(names) == 1 and names[0].startswith("train_t0100_pred_t01000_")
      and os.path.isfile(os.path.join(t2000, "kept.txt")),
      f"only the view whose manifest names another checkpoint moves: "
      f"{names}")

print("\n   --smoke archives under its own root")
wipe()
smoke_model = os.path.join(fno_paths.smoke_root(), "models", "train_t0001",
                           "fno_ch.pt")
put_model(B_BYTES, B_TIME, smoke_model)
smoke_run = os.path.join(fno_paths.smoke_root(), "Results",
                         "train_t0001_pred_t00120")
forge(smoke_run, "ffffffffffff")
outside = digest(".")
reset()
P.run_case = lite_run_case
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    P.main(["--smoke"])
inside = os.path.join(fno_paths.smoke_root(), "Archive", "superseded")
names = sorted(os.listdir(inside)) if os.path.isdir(inside) else []
check(len(names) == 1
      and names[0].startswith("train_t0001_pred_t00120_ffffffffffff_")
      and os.path.isfile(os.path.join(inside, names[0], "Results",
                                      "train_t0001_pred_t00120", "kept.txt")),
      f"the old smoke results are under Work/smoke/Archive: {names}")
now = digest(".")
strays = sorted(k for k in set(now) - set(outside)
                if not k.startswith(os.path.join("Work", "smoke")))
check(not strays and not os.path.exists("Archive"),
      f"and nothing outside Work/smoke, no Archive/ in the box: {strays}")

print("\n4. the seed sets")
group = P.GROUP_SEEDS
check(len(group) == 50 and len(set(group)) == 50 and group[:4] ==
      (99999, 54321, 77777, 31415),
      "50 distinct seeds, the original four first")
check(min(group) >= 20000 and not set(group) & (
    set(T.VAL_SEEDS) | set(T.MONITOR_SEEDS)),
      "all clear of the dataset's training and validation seeds (below 20000) and "
      "of VAL_SEEDS and MONITOR_SEEDS")
check(set(group) & set(P.make_eval_seeds(50))
      == {99999, 54321, 77777, 31415},
      "four of them are also in this project's own set")

overrides()
check(P.SEED_SET == "fno" and P.EVAL_SEEDS == P.make_eval_seeds(50)
      and P.SWEEP_SEEDS == P.EVAL_SEEDS,
      "without a flag: this project's 50 seeds, as before")
overrides("--seed-set", "fno")
check(P.SEED_SET == "fno" and P.EVAL_SEEDS == P.make_eval_seeds(50),
      "--seed-set fno is the same")
overrides("--seed-set", "group")
check(P.SEED_SET == "group" and P.EVAL_SEEDS == group and P.SWEEP_SEEDS == group,
      "--seed-set group: the group's 50, and the sweep follows")
overrides("--seed-set", "group", "--seeds", "7")
check(P.EVAL_SEEDS == group[:7] and P.SWEEP_SEEDS == group[:7],
      "--seeds 7 with group: the first 7 of its list")
overrides("--seeds", "7")
check(P.SEED_SET == "fno" and P.EVAL_SEEDS == P.make_eval_seeds(7),
      "--seeds 7 alone: this project's first 7, as before")
reset()
P.SWEEP_SEEDS = (11, 22, 33)
with contextlib.redirect_stdout(io.StringIO()):
    P.apply_overrides(P.parse_args(["--seed-set", "group"]))
check(P.EVAL_SEEDS == group and P.SWEEP_SEEDS == (11, 22, 33),
      "a SWEEP_SEEDS changed by hand is left alone")
for argv, what in ((["--seed-set", "group", "--seeds", "51"], "51 of 50"),
                   (["--seed-set", "group", "--seeds", "-3"], "-3")):
    try:
        overrides(*argv)
        refused = False
    except SystemExit:
        refused = True
    check(refused, f"--seeds {what} with group is refused")
try:
    with contextlib.redirect_stderr(io.StringIO()):
        P.parse_args(["--seed-set", "nonsense"])
    refused = False
except SystemExit as exc:
    refused = exc.code == 2
check(refused, "an unknown seed set is refused")

wipe()
put_model(A_BYTES, A_TIME)
code, log = run("--seed-set", "group", seeds="3")
book = read_json(XT(BOOKKEEPING))
check(CALLS == [(0.0, None, group[:3])]
      and book.get("seed_set") == "group"
      and (book.get("last_run") or {}).get("seed_set") == "group",
      f"a run on group seeds scores them, and the bookkeeping names the set "
      f"({CALLS}, {book.get('seed_set')})")
wipe()
code, log = run()
check(read_json(XT(BOOKKEEPING)).get("seed_set") == "fno",
      "and a run without the flag records fno")

reset()
shutil.rmtree(STASH, ignore_errors=True)
os.chdir(SCRATCH)
shutil.rmtree(BOX, ignore_errors=True)
new = sorted(project_state() - PROJECT)
check(not new, f"no run wrote into the project folder, Work/ or Results/ "
               f"{new[:3]}")
print("\n" + ("PROVENANCE PASS" if ok else "PROVENANCE FAIL"))
sys.exit(0 if ok else 1)
