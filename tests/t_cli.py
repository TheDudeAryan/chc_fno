"""The one-off CLI overrides change the run without touching the file."""
import contextlib
import io as _io
import json
import os
import shutil
import sys
import types

import numpy as np

from _paths import fixed_step_model, ROOT, SCRATCH, project_state

# a fixed-step network: this suite runs the seed engine, and the project's own
# checkpoints are dt-conditioned (see fixed_step_model in _paths.py)
REAL_MODEL = fixed_step_model()

os.chdir(ROOT)
PROJECT = project_state()
import fno_predict as P
import make_submission as S

TMP = os.path.join(SCRATCH, "cli_results")
CACHE = os.path.join(SCRATCH, "cli_cache")
# the file's RESULT_CACHE is relative, and this suite runs in the project
P.RESULT_CACHE = CACHE
P.GT_CACHE = os.path.join(SCRATCH, "cli_gt_cache")
P.MODEL_PATH = REAL_MODEL
P.WRITE_LOG = False
# t_records.py is about records; kept here they would keep the cache folder
# that these checks expect to be removed once it is empty
P.RECORDS = False
ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


SAVED = {k: getattr(P, k) for k in
         ("EVAL_MODE", "SIZE_SWEEP", "EVAL_SEEDS", "SWEEP_SEEDS",
          "WRITE_VIDEO", "RESULTS_DIR", "WRITE_TIMING", "PER_SEED_FIGURES",
          "WRITE_SUBMISSION", "RESULT_CACHE", "GT_CACHE", "KEEP_CACHE",
          "REDO", "MODEL_PATH", "RECORDS")}


def restore():
    for k, v in SAVED.items():
        setattr(P, k, v)


def overrides(argv):
    """apply_overrides on these flags; returns what it logged."""
    with contextlib.redirect_stdout(_io.StringIO()) as log:
        P.apply_overrides(P.parse_args(argv))
    return log.getvalue()


print("1. apply_overrides on its own")
restore()
out = overrides([])
check("no overrides" in out, "no flags leaves the file alone")
check(P.EVAL_MODE == SAVED["EVAL_MODE"], "EVAL_MODE untouched")
check(P.KEEP_CACHE is False and P.REDO is False
      and P.WRITE_SUBMISSION is True,
      "by default the cache is cleaned, a finished case is resumed and the "
      "submission files are written")

restore()
overrides(["--eval-mode", "critical"])
check(P.EVAL_MODE == "critical", "--eval-mode critical takes effect")
check(P.eval_offsets() == (P.CRITICAL_OFF,),
      f"only the critical composition is evaluated: {P.eval_offsets()}")

restore()
overrides(["--seeds", "12"])
check(len(P.EVAL_SEEDS) == 12, f"--seeds 12 gives 12, got {len(P.EVAL_SEEDS)}")
check(P.SWEEP_SEEDS == P.EVAL_SEEDS,
      "SWEEP_SEEDS follows when it was following before")
check(P.EVAL_SEEDS == P.make_eval_seeds(50)[:12],
      "a smaller count is a prefix of the larger one, so caches still hit")

restore()
P.SWEEP_SEEDS = (11, 22, 33)
overrides(["--seeds", "12"])
check(P.SWEEP_SEEDS == (11, 22, 33),
      "an independent SWEEP_SEEDS is left alone")

restore()
overrides(["--sizes", "128,512"])
check(P.SIZE_SWEEP == (128, 512), f"--sizes parsed: {P.SIZE_SWEEP}")

restore()
overrides(["--no-video"])
check(P.WRITE_VIDEO is False, "--no-video turns recording off")

restore()
P.RECORDS = True
out = overrides([])
check(P.RECORDS is True, "records are kept by default")
out = overrides(["--no-records"])
check(P.RECORDS is False and "RECORDS True -> False" in out,
      "--no-records turns the per-seed records and meta.json off")

restore()
out = overrides(["--no-submission"])
check(P.WRITE_SUBMISSION is False and "WRITE_SUBMISSION True -> False" in out,
      "--no-submission turns the submission figures, film and manifest off")

restore()
out = overrides(["--keep-cache"])
check(P.KEEP_CACHE is True and "KEEP_CACHE -> True" in out,
      "--keep-cache keeps the cached rollouts, and says so")
check(P.REDO is False, "--keep-cache alone still resumes finished cases")

restore()
out = overrides(["--redo"])
check(P.REDO is True and "REDO -> True" in out,
      "--redo rolls finished cases out again, and says so")
check(P.KEEP_CACHE is False, "--redo alone still cleans the cache")

restore()
elsewhere = os.path.join(SCRATCH, "cli_cache_elsewhere")
out = overrides(["--result-cache", elsewhere])
check(P.RESULT_CACHE == elsewhere and "RESULT_CACHE ->" in out,
      "--result-cache moves the rollout cache")

print("\n2. end to end through main()")
restore()
P.MODEL_FLAGS.update(hard_constraints=True, hard_mass_conservation=True,
                     zero_dc=True, label="fno_ch", note="")


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
calls = []
BEST = {}
# film frames each rollout records: none by default, as a cache made without
# the film; section 4 records some
FRAMES = [0]


def film_frames(t, lat):
    n = FRAMES[0]
    if not n:
        return {"video_t": np.asarray([]), "video_true": None,
                "video_pred": None}
    return {"video_t": np.linspace(t[0], t[-1], n),
            "video_true": rng.standard_normal((n, lat, lat))
            .astype(np.float32),
            "video_pred": rng.standard_normal((n, lat, lat))
            .astype(np.float32)}


def fake_results(off, lat, seeds):
    """Synthetic rollouts, ranked in a shuffled order."""
    t = np.arange(P.START_TIME + 1, P.END_TIME + 1)
    order = rng.permutation(len(seeds))
    res = []
    for i, _ in enumerate(seeds):
        r2 = np.clip(1.0 - (t - t[0]) / 9000.0 - 0.002 * order[i], -1, 1)
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
            "fno_seconds": 0.5, "fno_steps": 1900, **film_frames(t, lat)})
    BEST[(off, lat)] = int(seeds[S.best_index(res)])
    return res


def fake_run_case(model, device, off, lattice=None, seeds=None):
    calls.append((off, lattice))
    lat = P.LATTICE if lattice is None else lattice
    seeds = P.seeds_for(off) if seeds is None else tuple(seeds)
    return P.write_case_outputs(fake_results(off, lat, seeds), seeds, off,
                                lattice)


P.run_case = fake_run_case
shutil.rmtree(TMP, ignore_errors=True)
P.RESULTS_DIR = TMP
P.WRITE_TIMING = False
P.PER_SEED_FIGURES = False

with contextlib.redirect_stdout(_io.StringIO()) as log:
    P.main(["--eval-mode", "critical", "--seeds", "6",
            "--sizes", "128,512", "--no-video", "--no-submission"])

print("   |", [ln for ln in log.getvalue().splitlines()
               if ln.startswith("[cli]")])
check(calls == [(0.0, None), (0.0, 512)],
      f"critical once, then only the extra lattice: {calls}")
check(not any(abs(off) > 1e-12 for off, _ in calls),
      "no off-critical case was run")
# FNO's own outputs are in the rollout cache's extras/, one folder per case
XD = P.extras_dir()
dirs = sorted(d for d in os.listdir(XD) if os.path.isdir(f"{XD}/{d}"))
check(dirs == ["psi0+0.00_l128", "psi0+0.00_l512"],
      f"one extras folder per critical case: {dirs}")
seed_dirs = sorted(d for d in os.listdir(f"{XD}/{dirs[0]}")
                   if os.path.isdir(f"{XD}/{dirs[0]}/{d}"))
want = P.BEST_SEED_DIR.format(seed=BEST.get((0.0, 128)))
check(seed_dirs == [want],
      f"of the 6 seeds only the best keeps a folder, {want}/: {seed_dirs}")
check(not any(f.endswith(".mp4") or f.endswith(".gif")
              for root in (TMP, CACHE) for _, _, fs in os.walk(root)
              for f in fs),
      "--no-video left no videos behind")
top = (sorted(os.listdir(TMP)) if os.path.isdir(TMP) else [])
check(top == [],
      f"--no-submission wrote nothing into the Results folder (no figure, "
      f"manifest, report or sweep): {top}")
loose = sorted(f for d in dirs for f in os.listdir(f"{XD}/{d}")
               if f in S.CASE_FILES + S.CASE_DATA or f.startswith("rollout."))
check(not loose, f"and no submission data or film in a case's extras: "
                 f"{loose}")
check(all(os.path.isfile(f"{XD}/{d}/{f}")
          for d, sfx in zip(dirs, ("", "_l512"))
          for f in (f"fno_ch{sfx}_r2_and_snapshots.png",
                    f"fno_ch{sfx}_summary.json", P.MEAN_R2_FILE)),
      "while the fno_ch figures, summaries and averages are all there")

print("\n3. --eval-mode none: only the lattice sweep, for a top-up run")
restore()
P.MODEL_FLAGS.update(hard_constraints=True, hard_mass_conservation=True,
                     zero_dc=True, label="fno_ch", note="")
calls.clear()
shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(CACHE, ignore_errors=True)
P.RESULTS_DIR = TMP
P.WRITE_TIMING = False
P.WRITE_SUBMISSION = False
P.PER_SEED_FIGURES = False

saved_mode = P.EVAL_MODE
P.EVAL_MODE = "none"
check(P.eval_offsets() == (), "EVAL_MODE 'none' yields no composition")
P.EVAL_MODE = "nonsense"
try:
    P.eval_offsets()
    check(False, "a bad EVAL_MODE is rejected")
except ValueError as exc:
    check("none" in str(exc), "the error message lists 'none'")
P.EVAL_MODE = saved_mode

with contextlib.redirect_stdout(_io.StringIO()) as log3:
    P.main(["--eval-mode", "none", "--sizes", "256", "--seeds", "4",
            "--no-video"])
print("   |", [ln for ln in log3.getvalue().splitlines()
               if ln.startswith("[cli]")])
check(calls == [(0.0, 256)],
      f"exactly one case, the requested lattice: {calls}")
XD3 = P.extras_dir()
dirs3 = sorted(d for d in os.listdir(XD3) if os.path.isdir(f"{XD3}/{d}"))
check(dirs3 == ["psi0+0.00_l256"], f"one extras folder only: {dirs3}")
check(not any(f.startswith(("r2_sweep", "fno_ch_sweeps"))
              for f in (os.listdir(TMP) if os.path.isdir(TMP) else [])),
      "no sweep figure from a single curve")

print("\n4. --keep-cache and --redo: the rollout cache and finished cases")


def caching_run_case(model, device, off, lattice=None, seeds=None):
    """fake_run_case the way run_case does it: every rollout cached, and the
    case written with its cache key, so a later job can take it as done."""
    calls.append((off, lattice))
    lat = P.LATTICE if lattice is None else lattice
    seeds = P.seeds_for(off) if seeds is None else tuple(seeds)
    for sd in seeds:
        cached(off, sd, lat)
    return P.write_case_outputs(fake_results(off, lat, seeds), seeds, off,
                                lattice, key=P._cache_key(off, lat))


def cached(off, seed, lat):
    path = P._result_path(off, seed, lat)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"a cached rollout")
    return path


# The cache key is made from the model file's size and time: this section
# has one of its own to touch, never the project's checkpoint.
MODEL = os.path.join(SCRATCH, "cli_model.pt")


def run4(*flags, seeds=3):
    """One job over the critical case, the submission files on unless the
    flags say --no-submission; returns (status, log, its cache files)."""
    restore()
    P.MODEL_FLAGS.update(hard_constraints=True, hard_mass_conservation=True,
                         zero_dc=True, label="fno_ch", note="")
    P.MODEL_PATH = MODEL
    P.RESULTS_DIR = TMP
    P.WRITE_TIMING = False
    P.PER_SEED_FIGURES = False
    calls.clear()
    with contextlib.redirect_stdout(_io.StringIO()) as log4:
        code = P.main(["--eval-mode", "critical", "--seeds", str(seeds),
                       "--sizes", "128", "--result-cache", CACHE, *flags])
    files = P.case_cache_files(0.0, None, P.EVAL_SEEDS)
    return code, log4.getvalue(), files


with open(MODEL, "wb") as fh:
    fh.write(b"m")
P.run_case = caching_run_case
shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(CACHE, ignore_errors=True)
os.makedirs(CACHE)
foreign = os.path.join(CACHE, "fno_ch_l128_seed99999_0123456789ab.npz")
with open(foreign, "wb") as fh:
    fh.write(b"another model's rollout")
summary = os.path.join(CACHE, "extras", "psi0+0.00_l128",
                       "fno_ch_summary.json")

code, text, own = run4("--no-video", "--no-submission")
check(code == 0 and calls == [(0.0, None)],
      f"a first job rolls the case out: {calls}")
check(len(own) == 3 and not any(os.path.exists(f) for f in own),
      "its outputs written, the case's cached rollouts are deleted")
check(os.path.exists(foreign),
      "a rollout of another cache key in the same folder is left alone")
check(os.path.isdir(CACHE), "and the cache folder, not empty, stays")
before = open(summary, "rb").read() if os.path.exists(summary) else b""

for sd in P.make_eval_seeds(3):
    cached(0.0, sd, 128)
code, text, own = run4("--no-video", "--no-submission")
check(code == 0 and calls == [], f"a second job does not roll the finished "
                                 f"case out again: {calls}")
check("already finished" in text and "--redo" in text,
      "the log says it was finished earlier, and how to redo it")
check((open(summary, "rb").read() if os.path.exists(summary) else None)
      == before, "its summary is left as the first job wrote it")
check(not any(os.path.exists(f) for f in own),
      "a resumed case's leftover cached rollouts are deleted too")

code, text, own = run4("--no-video", "--no-submission", "--redo",
                       "--keep-cache")
check(code == 0 and calls == [(0.0, None)],
      f"--redo rolls it out again: {calls}")
check(all(os.path.exists(f) for f in own),
      "--keep-cache keeps its cached rollouts")

os.remove(foreign)
code, text, own = run4("--no-video", "--no-submission", "--redo")
check(code == 0 and calls == [(0.0, None)]
      and os.listdir(CACHE) == [P.EXTRAS_NAME],
      "without --keep-cache they go, and only extras/ is left in the cache "
      "folder")

def record():
    """The record the case summary keeps of the job that wrote it."""
    try:
        with open(summary) as fh:
            return json.load(fh).get("record") or {}
    except (OSError, ValueError):
        return {}


def films():
    """The case's film, if it has one."""
    here = os.path.dirname(summary)
    return sorted(f for f in (os.listdir(here) if os.path.isdir(here) else ())
                  if f.startswith("rollout."))


# from here on the rollouts carry film frames, so only the flags decide
# whether a film is drawn
FRAMES[0] = 4
code, text, own = run4("--no-video")
check(code == 0 and calls == [(0.0, None)],
      "a job that wants the submission files does not take a case finished "
      "without them")
rec = record()
check(os.path.isfile(os.path.join(TMP, "l128", "snapshots_psi0p0.00.png"))
      and not films() and rec.get("submission") is True
      and rec.get("video") is False,
      f"--no-video with the submission on: its figures but no film, and the "
      f"record says so (films {films()}, submission {rec.get('submission')}, "
      f"video {rec.get('video')})")


def manifest_files():
    """The files manifest.json names (the flat schema: the run folder's)."""
    try:
        with open(os.path.join(TMP, "manifest.json")) as fh:
            m = json.load(fh)
    except (OSError, ValueError):
        return []
    return m.get("files", [])


listed = manifest_files()
code, text, own = run4("--no-video")
check(code == 0 and calls == [] and "already finished" in text,
      f"the case finished with them is then taken as finished: {calls}")
again = manifest_files()
check(bool(listed) and again == listed
      and all(os.path.isfile(os.path.join(TMP, f)) for f in again),
      f"and the manifest still names its submission files: {again}")

code, text, own = run4()
check(code == 0 and calls == [(0.0, None)],
      "nor does one that wants the film take a case finished without it")
check(len(films()) == 1 and record().get("video") is True,
      f"that job's case has its film, and its record says so: {films()}")
code, text, own = run4()
check(code == 0 and calls == [] and "already finished" in text,
      f"a job that wants the film takes a case finished with it: {calls}")
code, text, own = run4("--no-video", "--no-submission")
check(code == 0 and calls == [],
      "a case finished with more than a job wants is taken as finished too")

# rollouts cached without frames (the film was off when they were made) give
# no film even with the film on
FRAMES[0] = 0
code, text, own = run4("--redo")
rec = record()
check(code == 0 and calls == [(0.0, None)] and not films()
      and rec.get("submission") is True and rec.get("video") is False,
      f"rollouts without frames: no film, and the record says none (films "
      f"{films()}, video {rec.get('video')})")
code, text, own = run4()
check(code == 0 and calls == [(0.0, None)],
      "so a job that wants the film does not take that case as finished")

print("\n5. make_submission's command line")
# the options that cannot go together are refused before anything is
# written; no arguments at all is the default mode (every complete run
# redrawn, then the overview), tried at the end of section 6 in a folder
# of its own
for argv, what, words in (
        (["--cache", CACHE], "--cache without --out", ("--out", "--cache")),
        (["--partial"], "--partial without --records", ("--records",)),
        (["--records", CACHE], "--records without --out", ("--out",)),
        (["--overview", "--cache", CACHE], "--overview with --cache",
         ("--overview", "--cache")),
        (["--index", "--run", "x"], "--index (the old --overview) with "
         "--run", ("--overview", "--run")),
        (["--run", "x", "--cache", CACHE], "--run with --cache",
         ("--run", "--cache"))):
    err = _io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            S.main(argv)
        check(False, f"make_submission with {what} is refused")
    except SystemExit as exc:
        said = err.getvalue().partition("error:")[2]
        check(exc.code == 2 and all(w in said for w in words),
              f"make_submission with {what} is refused, naming "
              f"{' and '.join(words)} (exit {exc.code}: {said.strip()!r})")

print("\n6. the model, and the inputs from before the folder layout")
J = os.path.join
DEFAULT_MODEL = J("Work", "models", "train_t0500", "fno_ch.pt")
DEFAULT_GT = J("Work", "gt_cache")
SB = os.path.join(SCRATCH, "cli_layout")


class Stop(Exception):
    """Raised by the stand-in for load_checkpoint: main() got that far."""


loaded = []


def stop_at_load(path, device=None):
    loaded.append((path, P.GT_CACHE))
    raise Stop


def sandbox(*files):
    """A fresh folder holding just these (a trailing / makes a folder)."""
    shutil.rmtree(SB, ignore_errors=True)
    os.makedirs(SB)
    for name in files:
        path = J(SB, name)
        os.makedirs(os.path.dirname(path) or SB, exist_ok=True)
        if name.endswith("/"):
            os.makedirs(path, exist_ok=True)
        else:
            with open(path, "wb") as fh:
                fh.write(b"m")


def start_main(*flags, **settings):
    """main() in the sandbox as far as the checkpoint, with the model and
    the references as the file's own defaults: (what it was asked to load
    and the GT_CACHE it had then, its log, RESULTS_DIR and RESULT_CACHE
    as the flags left them, the error that stopped it)."""
    restore()
    P.MODEL_PATH, P.GT_CACHE = DEFAULT_MODEL, DEFAULT_GT
    for k, v in settings.items():
        setattr(P, k, v)
    P.load_checkpoint = stop_at_load
    loaded.clear()
    err = None
    log = _io.StringIO()
    os.chdir(SB)
    try:
        with contextlib.redirect_stdout(log):
            P.main(list(flags))
    except (Stop, FileNotFoundError) as exc:
        err = exc
    finally:
        os.chdir(ROOT)
    return list(loaded), log.getvalue(), (P.RESULTS_DIR, P.RESULT_CACHE), err


def layout_lines(text):
    return [ln for ln in text.splitlines() if ln.startswith("[layout]")]


sandbox("fno_ch.pt", "gt_cache/")
got, text, (res, rc), err = start_main()
check(got == [("fno_ch.pt", "gt_cache")] and isinstance(err, Stop)
      and len(layout_lines(text)) == 2
      and all("migrate_layout.py" in ln for ln in layout_lines(text)),
      f"the model and the references still where they were: used, and two "
      f"lines say so ({got}, {layout_lines(text)})")
got, text, (res, rc), err = start_main("--end-time", "4000")
check(got == [("fno_ch.pt", "gt_cache")]
      and res == J("Results", "train_t0500_pred_t04000")
      and rc == J("Work", "rollout_cache", "train_t0500_pred_t04000"),
      f"and a run to another horizon is still named for the t = 500 model, "
      f"not for fno_ch: {res}, {rc}")
got, text, (res, rc), err = start_main(RESULTS_DIR=J(SB, "r"))
check(res == J(SB, "r"), "a results folder the file names is left as it is")

sandbox("fno_ch.pt", "gt_cache/", "Work/models/train_t0500/fno_ch.pt",
        "Work/gt_cache/")
got, text, _, err = start_main()
check(got == [(DEFAULT_MODEL, DEFAULT_GT)] and not layout_lines(text),
      f"with the new places there, they win and nothing is said ({got})")
sandbox("Work/models/train_t0500/fno_ch.pt", "gt_cache/")
got, text, _, err = start_main()
check(got == [(DEFAULT_MODEL, "gt_cache")] and len(layout_lines(text)) == 1,
      f"each input is looked for on its own ({got})")

sandbox("fno_ch.pt", "gt_cache/")
got, text, _, err = start_main("--model", "other.pt")
check(isinstance(err, FileNotFoundError) and "other.pt" in str(err)
      and len(layout_lines(text)) == 1 and "references" in text,
      "a model named on the command line is never swapped for fno_ch.pt")
got, text, _, err = start_main("--gt-cache-dir", "mine")
check(got == [("fno_ch.pt", "mine")],
      f"nor a reference folder it named ({got})")
got, text, _, err = start_main("--no-gt-cache")
check(got == [("fno_ch.pt", "")], f"nor --no-gt-cache ({got})")

sandbox("Work/models/train_t0300/fno_ch.pt",
        "Work/models/train_t1000/fno_ch.pt",
        "Work/models/train_t1000/fno_ch_last.pt")
got, text, _, err = start_main()
said = str(err)
check(isinstance(err, FileNotFoundError) and DEFAULT_MODEL in said
      and "fno_train.py" in said and "--model" in said
      and J("Work", "models", "train_t0300", "fno_ch.pt") in said
      and J("Work", "models", "train_t1000", "fno_ch.pt") in said
      and "fno_ch_last.pt" not in said,
      f"a missing model is named, with what to do and the models that "
      f"are there: {said}")
sandbox()
got, text, _, err = start_main()
check(isinstance(err, FileNotFoundError) and str(err).endswith(": none."),
      f"and with none there it says so: {err}")

# no arguments: the default mode.  In a folder with no run in it there is
# nothing to redraw and no Results/ to index; it says so and writes nothing
# but its log, and the log stays in the folder it ran in.
sandbox()
real_log = S.fno_runlog.start_log
S.fno_runlog.start_log = lambda *a, **k: None
os.chdir(SB)
try:
    with contextlib.redirect_stdout(_io.StringIO()) as quiet_out:
        try:
            S.main([])
            said = ""
        except SystemExit as exc:
            said = str(exc.code)
finally:
    os.chdir(ROOT)
    S.fno_runlog.start_log = real_log
check("Results" in said and not os.path.exists(J(SB, "Results"))
      and "[redraw]" not in quiet_out.getvalue(),
      f"make_submission with no arguments is the default mode, not an "
      f"error of usage: in a folder without runs it ends with {said!r}")
shutil.rmtree(SB, ignore_errors=True)

restore()
shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(CACHE, ignore_errors=True)
new = sorted(project_state() - PROJECT)
check(not new, f"no run wrote into the project folder, Work/ or Results/ "
               f"{new[:3]}")
print("\n" + ("CLI PASS" if ok else "CLI FAIL"))
sys.exit(0 if ok else 1)
