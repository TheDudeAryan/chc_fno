"""The permanent per-seed records, meta.json, and the redraw from them.

A run keeps one compact record per seed and a meta.json beside the temporary
cache, and the cache's cleaning never touches them; the cache folder also
holds extras/ (FNO's own outputs of the run), which no scan of the cache or
loader of the records ever looks into.  make_submission --records
redraws a results folder from them with no model and no solver.  Every
rollout here is synthetic (evaluate_seed is stubbed), the checkpoint is a
small file of this suite's own, and nothing is written outside _scratch.
"""
import contextlib
import difflib
import io as _io
import json
import os
import re
import shutil
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from _paths import ROOT, SCRATCH, project_state

os.chdir(ROOT)
PROJECT = project_state()
import fno_paths
import fno_predict as P
import fno_runlog

S = P._submission()
ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


BASE = os.path.join(SCRATCH, "records")
shutil.rmtree(BASE, ignore_errors=True)
os.makedirs(BASE)
CKPT = os.path.join(BASE, "models", "train_t0500", "fno_ch.pt")
CKPT2 = os.path.join(BASE, "models2", "train_t0500", "fno_ch.pt")
for _path, _text in ((CKPT, b"a checkpoint of this suite, not a network"),
                     (CKPT2, b"another checkpoint of this suite")):
    os.makedirs(os.path.dirname(_path))
    with open(_path, "wb") as fh:
        fh.write(_text)
FP, FP2 = fno_paths.fingerprint(CKPT), fno_paths.fingerprint(CKPT2)

KEYS = ("EVAL_MODE", "SIZE_SWEEP", "EVAL_SEEDS", "SWEEP_SEEDS", "WRITE_VIDEO",
        "RESULTS_DIR", "RESULTS_TEMPLATE", "VIEWS", "WRITE_TIMING",
        "PER_SEED_FIGURES", "PER_SEED_OUTPUT", "WRITE_SUBMISSION", "WRITE_LOG",
        "START_TIME", "END_TIME", "SNAPSHOT_TIMES", "MODEL_PATH", "REDO",
        "KEEP_CACHE", "load_checkpoint", "evaluate_seed", "RESULT_CACHE",
        "GT_CACHE", "CACHE_RESULTS", "RECORDS", "ROOT", "NO_SUPERSEDE",
        "SEED_SET", "LATTICE", "LABEL", "AUTO_LABEL", "LOG_TIME",
        "SNAPSHOTS_EXPLICIT", "RECORDED_FINGERPRINT", "CACHE_END_TIME",
        "_RUN_SNAPSHOTS", "DIAG_EVERY", "VIDEO_EVERY", "VIDEO_SCHEDULE",
        "evaluate_seeds", "EVAL_BATCH",
        "VIDEO_DTYPE", "DIAG_TILE", "DIAG_TILE_T", "DIAG_LOWK_SEEDS",
        "DIAG_LOWK_L", "DIAG_LOWK_T_END", "ENGINE", "DIAGNOSTICS",
        "LOWK_FILL", "MU_CLIP", "RECORDED_NUMERICS", "PLAN_ONLY",
        "solve_references", "TUNE")
SAVED = {k: getattr(P, k) for k in KEYS}
# Nothing here may land in the project: every folder is inside _scratch.  The
# checkpoint's own config decides the fill, whatever fno_predict's LOWK_FILL
# is set to at the time.
SAVED.update(WRITE_LOG=False, MODEL_PATH=CKPT, GT_CACHE="", LOWK_FILL=None,
             MU_CLIP=None,
             RESULTS_DIR=os.path.join(BASE, "results_default"),
             RESULT_CACHE=os.path.join(BASE, "cache_default"),
             ROOT=os.path.join(BASE, "root"))
LOG_DIR = fno_runlog.LOG_DIR
WORK = fno_paths.WORK
FLAGS = dict(hard_constraints=True, hard_mass_conservation=True,
             zero_dc=True, label="fno_ch", note="")
BEST = {"epoch": 7, "score": 0.91, "survival": 2000.0}


class FakeModel:
    config = {"backbone": "physical_k", "hard_mass_conservation": True,
              "residual_prediction": True, "mode_scaling": 1.0,
              "zero_dc": True}

    def n_parameters(self):
        return 1234

    def parameters(self):
        yield types.SimpleNamespace(dtype="float32")


ROLLED = []
META_AT = []


def fake_rollout(seed, off, lat, frames=3):
    """The same synthetic rollout for the same seed, case and box, with a
    short film, and float32 predictions in the odd seeds' snapshots."""
    g = np.random.default_rng([int(seed), int(round(off * 10)) + 10, lat])
    t = np.arange(P.START_TIME + 1, P.END_TIME + 1)
    diag = t[::max(1, t.size // 19)]
    r2 = np.clip(1.0 - (t - t[0]) / 9000.0 - 0.01 * g.random(), -1, 1)
    snaps = {}
    for tt in P.SNAPSHOT_TIMES:
        f = np.tanh(3 * g.standard_normal((24, 24))) + off
        q = np.clip(f + 0.05 * g.standard_normal(f.shape), -1.05, 1.05)
        snaps[tt] = (f, q.astype(np.float32) if seed % 2 else q)
    L = np.linspace(5, 14, diag.size) * (1 + 0.1 * g.random())
    return {
        "times": t, "r2": r2, "mass_drift": np.full(t.size, 1e-9),
        "mean_true": np.full(t.size, off),
        "mean_pred": np.full(t.size, off) + 1e-9,
        "diag_times": diag,
        "energy_true": np.linspace(1, 0.5, diag.size),
        "energy_pred": np.linspace(1, 0.5, diag.size),
        "t_stop": float(t[-1]), "stopped_early": False, "diverged_at": None,
        "L_true": dict({k: L for k in ("L_zero", "L_peak", "L_moment",
                                       "L_grad")}, L_half=0.5 * L),
        "L_pred": dict({k: L * 1.01 for k in ("L_zero", "L_peak", "L_moment",
                                              "L_grad")}, L_half=0.5 * L),
        "snapshots": snaps,
        "gt_seconds": 1.0, "gt_cached": False, "gt_steps": 1,
        "fno_seconds": 0.5, "fno_steps": 1900,
        "video_t": np.linspace(t[0], t[-1], frames),
        "video_true": g.standard_normal((frames, 24, 24)).astype(np.float32),
        "video_pred": g.standard_normal((frames, 24, 24)).astype(np.float32)}


def fake_evaluate_seed(model, seed, device, off=0.0, lattice=None):
    lat = P.LATTICE if lattice is None else lattice
    ROLLED.append((off, lat, seed))
    META_AT.append((lat, seed, P.read_meta(P.RESULT_CACHE)))
    return fake_rollout(seed, off, lat)


BATCHES = []


def fake_evaluate_seeds(model, seeds, device, off=0.0, lattice=None):
    """The lock-step engine's batch, made up: one rollout per seed."""
    lat = P.LATTICE if lattice is None else lattice
    BATCHES.append(tuple(seeds))
    return [fake_rollout(s, off, lat) for s in seeds]


def no_rollout(*a, **k):
    raise AssertionError("a rollout was asked for")


def setup():
    for k, v in SAVED.items():
        setattr(P, k, v)
    fno_runlog.LOG_DIR = LOG_DIR
    fno_paths.WORK = WORK
    P.MODEL_FLAGS.update(FLAGS)
    P.load_checkpoint = lambda path, device=None: (FakeModel(),
                                                   {"best": dict(BEST)})
    P.evaluate_seed = fake_evaluate_seed
    P.evaluate_seeds = fake_evaluate_seeds
    ROLLED.clear()
    BATCHES.clear()
    META_AT.clear()


def run_main(argv):
    buf = _io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            code = P.main(argv)
    except SystemExit as exc:
        code = exc.code
    return code, buf.getvalue()


def flags(results, cache, *more, model=CKPT, sizes="128,384", seeds="3"):
    return ["--model", model, "--results-dir", results, "--result-cache",
            cache, "--eval-mode", "critical", "--sizes", sizes,
            "--seeds", seeds, "--no-gt-cache", "--no-diagnostics", *more]


def files_under(d):
    return sorted(os.path.relpath(os.path.join(b, f), d).replace(os.sep, "/")
                  for b, _, fs in os.walk(d) for f in fs)


def cache_only(files):
    """The cache folder's own files: everything but extras/ (FNO's outputs
    of the run, which no scan of the cache or loader of the records looks
    into)."""
    return [f for f in files if not f.startswith("extras/")]


def read(p):
    with open(p, "rb") as fh:
        return fh.read()


def meta_of(cache):
    """meta.json of a records folder; the suite stops where there is none."""
    meta = P.read_meta(cache)
    if meta is None:
        raise SystemExit(f"FAIL  no readable meta.json in {cache}")
    return meta


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def redraw(cache, out, *more):
    """make_submission --records, quietly; returns (exit code, log)."""
    real = S.fno_runlog.start_log
    S.fno_runlog.start_log = lambda *a, **k: None
    code, buf = None, _io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            S.main(["--records", cache, "--out", out, *more])
    except SystemExit as exc:
        code = exc.code
    finally:
        S.fno_runlog.start_log = real
    return code, buf.getvalue()


TIME = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d")


def norm(obj, folders):
    """A JSON value with the folders and the times it names made alike, and
    the film's files left out of file lists."""
    if isinstance(obj, str):
        for f in folders:
            obj = obj.replace(f, "<out>")
        return TIME.sub("<time>", obj)
    if isinstance(obj, dict):
        return {k: norm(v, folders) for k, v in obj.items()}
    if isinstance(obj, list):
        return [norm(v, folders) for v in obj
                if not (isinstance(v, str) and v.endswith((".mp4", ".gif")))]
    return obj


def flat(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flat(v, f"{path}/{k}")
    else:
        yield path, obj


# What may differ between a run's own JSON and its redraw: the film, the
# checkpoint file (a redraw never opens one), and who wrote it.  Every other
# value, the times and folders aside, must be the same.
JSON_MAY_DIFFER = (r"/label$", r".*/film(/.*)?$", r".*/video$",
                   r".*/(model_path|model_bytes|model_mtime)$",
                   r"/checkpoints(/.*)?$", r"/source$", r"/records(/.*)?$",
                   r"/meta/(device|gpu|precision)$")
TEXT_MAY_DIFFER = (r"- report of ", r"^  drawn by ", r"^Model `", r"^Redrawn from ", r"^The film cannot ",
                   r"rollout\.mp4", r"^  playing ", r"^  prediction and ",
                   r"^  which keep no film", r"^  device ", r"^  precision ")


def json_differences(a, b, dir_a, dir_b):
    da = dict(flat(norm(jload(a), [dir_a, dir_b])))
    db = dict(flat(norm(jload(b), [dir_a, dir_b])))
    return sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k)
                  and not any(re.fullmatch(p, k) for p in JSON_MAY_DIFFER))


def text_differences(a, b, dir_a, dir_b):
    def lines(p):
        with open(p, encoding="utf-8") as fh:
            return str(norm(fh.read(), [dir_a, dir_b])).splitlines()
    out = [ln[1:] for ln in difflib.unified_diff(lines(a), lines(b), n=0,
                                                 lineterm="")
           if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    return [ln for ln in out
            if not any(re.search(p, ln) for p in TEXT_MAY_DIFFER)]


def same_outputs(orig, new):
    """Compare a results folder with its redraw; returns a list of what
    differs, which must be empty.  Figures and CSVs must match byte for byte;
    a JSON, README or text file only where it names the film, the checkpoint
    file or the writer."""
    fo, fn = files_under(orig), files_under(new)
    bad = []
    films = [f for f in fo if f.endswith((".mp4", ".gif"))]
    if sorted(set(fo) - set(fn)) != sorted(films) or set(fn) - set(fo):
        bad.append(f"file lists differ: only original "
                   f"{sorted(set(fo) - set(fn) - set(films))}, only redraw "
                   f"{sorted(set(fn) - set(fo))}")
    for f in sorted(set(fo) & set(fn)):
        a, b = os.path.join(orig, f), os.path.join(new, f)
        if f.endswith((".png", ".csv", ".pdf")):
            if read(a) != read(b):
                bad.append(f"{f}: bytes differ")
        elif f.endswith(".json"):
            bad += [f"{f}: {k}" for k in json_differences(a, b, orig, new)]
        elif f.endswith((".md", ".txt")):
            bad += [f"{f}: {ln!r}" for ln in text_differences(a, b, orig, new)]
        elif f.endswith(".npz"):
            with np.load(a) as za, np.load(b) as zb:
                if (sorted(za.files) != sorted(zb.files) or not all(
                        np.array_equal(za[k], zb[k]) for k in za.files)):
                    bad.append(f"{f}: arrays differ")
        else:
            bad.append(f"{f}: not a kind this compares")
    return bad


def pngs_and_csvs(d):
    return {f: read(os.path.join(d, *f.split("/"))) for f in files_under(d)
            if f.endswith((".png", ".csv"))}


EXPECTED = {"times", "r2", "mass_drift", "mean_true", "mean_pred",
            "diag_times", "energy_true", "energy_pred", "t_stop",
            "gt_seconds", "gt_cached", "gt_steps", "fno_seconds",
            "fno_steps", "stopped_early", "diverged_at", "snap_t",
            "snap_true", "snap_pred", "Ltrue_L_zero", "Ltrue_L_peak",
            "Ltrue_L_moment", "Ltrue_L_grad", "Ltrue_L_half", "Lpred_L_zero",
            "Lpred_L_peak", "Lpred_L_moment", "Lpred_L_grad", "Lpred_L_half"}
IDENTITY = {"seed", "off", "lattice", "fingerprint", "cache_key", "t_start",
            "t_end"}
ARRAY_KEYS = ("times", "r2", "mass_drift", "mean_true", "mean_pred",
              "diag_times", "energy_true", "energy_pred")


print("1. a run keeps a record of every seed, and meta.json")
ORIG = os.path.join(BASE, "results_orig")
CACHE = os.path.join(BASE, "rollout_cache", "train_t0500_pred_t02000")
setup()
code, log = run_main(flags(ORIG, CACHE))
SEEDS = P.EVAL_SEEDS
LABEL = os.path.basename(CACHE)
KEY = {lat: P._cache_key(0.0, lat) for lat in (128, 384)}
cases = ("psi0+0.00_l128", "psi0+0.00_l384")
check(code == 0 and len(ROLLED) == 6,
      f"a clean run rolled out 3 seeds of 2 cases: {code}, {len(ROLLED)}")
want = sorted(["meta.json"] + [f"records/{c}/seed{s}.npz" for c in cases
                               for s in SEEDS])
in_cache = [f for f in files_under(CACHE) if not f.startswith("extras/")]
check(in_cache == want and not [f for f in files_under(CACHE)
                                if f.endswith(".tmp")],
      f"the cache folder holds meta.json and a record per seed, and no "
      f"temporary cache file or .tmp, beside extras/: {in_cache}")
xs = sorted(os.listdir(os.path.join(CACHE, "extras")))
check(xs == sorted(["psi0+0.00_l128", "psi0+0.00_l384", "sweep_curves.npz",
                    "submission_manifest.json", "fno_ch_timing.json",
                    "fno_ch_timing.txt"]),
      f"and extras/ holds a folder per case and the run's bookkeeping, "
      f"curves and timing report: {xs}")
check("stays: it holds the records" in log
      and os.path.isdir(CACHE),
      "the temporary cache is cleaned but the folder stays, as it holds the "
      "records, and the log says why")

bad = []
for c, lat in zip(cases, (128, 384)):
    for s in SEEDS:
        res = fake_rollout(s, 0.0, lat)
        with np.load(os.path.join(CACHE, "records", c, f"seed{s}.npz")) as z:
            names = set(z.files)
            if names != EXPECTED | IDENTITY:
                odd = sorted(names ^ (EXPECTED | IDENTITY))
                bad.append(f"{c}/{s}: keys {odd}")
            for k in ARRAY_KEYS:
                if (z[k].dtype != res[k].dtype
                        or not np.array_equal(z[k], res[k])):
                    bad.append(f"{c}/{s}: {k}")
            order = sorted(res["snapshots"])
            for j, name in enumerate(("snap_true", "snap_pred")):
                held = np.stack([res["snapshots"][t][j] for t in order])
                if z[name].dtype != held.dtype or not np.array_equal(
                        z[name], held):
                    bad.append(f"{c}/{s}: {name} {z[name].dtype}")
            for grp, name in (("L_true", "Ltrue_"), ("L_pred", "Lpred_")):
                for k, v in res[grp].items():
                    if not np.array_equal(z[name + k], v):
                        bad.append(f"{c}/{s}: {name}{k}")
            who = (int(z["seed"]), float(z["off"]), int(z["lattice"]),
                   str(z["fingerprint"]), str(z["cache_key"]),
                   float(z["t_start"]), float(z["t_end"]))
            if who != (s, 0.0, lat, FP, KEY[lat], 100.0, 2000.0):
                bad.append(f"{c}/{s}: identity {who}")
            if not (float(z["t_stop"]) == res["t_stop"]
                    and bool(z["stopped_early"]) is False
                    and np.isnan(z["diverged_at"])
                    and float(z["gt_seconds"]) == 1.0):
                bad.append(f"{c}/{s}: scalars")
check(not bad, f"every record holds all the cache's keys but the film, each "
               f"array with the dtype the rollout had (float32 and float64 "
               f"predictions both), and who made it: {bad[:3]}")
check(all(not k.startswith("vid") and "video" not in k
          for c in cases for s in SEEDS for k in np.load(
              os.path.join(CACHE, "records", c, f"seed{s}.npz")).files),
      "no record carries a video key, though the rollouts had frames")
res0 = fake_rollout(SEEDS[0], 0.0, 128)
check(res0["video_true"].nbytes > 0
      and "vid_true" in P._result_arrays(res0)
      and "vid_true" not in P._result_arrays(res0, film=False),
      "the temporary cache still keeps the frames; only the record leaves "
      "them out")

meta = meta_of(CACHE)
check(meta["label"] == LABEL
      and meta["fingerprint"] == FP and meta["checkpoint"] == CKPT
      and meta["model"]["config"] == FakeModel.config
      and meta["model"]["n_parameters"] == 1234
      and meta["model"]["best"] == BEST,
      "meta.json names the label, the checkpoint fingerprint and path and "
      "the model's config")
DEFAULT_NUM = {"precision": "fp32", "tf32": False, "lowk_fill": False,
               "mu_clip": None}
FILL_NUM = dict(DEFAULT_NUM, lowk_fill=True)
check(meta["numerics"] == DEFAULT_NUM,
      f"and the numerics it ran in: precision, TF32 off, the low-k fill "
      f"({meta['numerics']})")
check(meta["start"] == 100.0 and meta["end"] == 2000.0 and meta["views"] == []
      and meta["snapshots"] == [100.0, 500.0, 1000.0, 2000.0]
      and meta["lattice"] == 128 and meta["size_sweep"] == [128, 384]
      and meta["eval_mode"] == "critical"
      and meta["eval_seeds"] == list(SEEDS),
      "and the window, views and the settings that decide the cases")
check(meta["planned_cases"] == list(cases)
      and all(meta["cases"][c]["seeds"] == list(SEEDS)
              and meta["cases"][c]["complete"] is True
              and meta["cases"][c]["cache_key"] == KEY[lat]
              for c, lat in zip(cases, (128, 384))),
      "the planned cases, each with its seeds, its cache key and a complete "
      "flag that is true once it has finished")
check(all(TIME.fullmatch(meta[k]) for k in ("created", "updated"))
      and meta["created"] <= meta["updated"],
      f"created and updated times: {meta['created']}, {meta['updated']}")
first = META_AT[0][2]
second = next(m for lat, sd, m in META_AT if lat == 384)
check(first is not None
      and [first["cases"][c]["complete"] for c in cases] == [False, False]
      and second is not None
      and [second["cases"][c]["complete"] for c in cases] == [True, False],
      "meta.json is there before the first seed rolls out with nothing "
      "complete, and is updated as cases finish: the first case is complete "
      "while the second rolls out")
check(not [f for f in files_under(CACHE) if f.endswith(".tmp")],
      "records and meta.json are written under a temporary name and renamed")

print("\n2. a redraw from the records alone is the same")
REDRAW = os.path.join(BASE, "results_redraw")
setup()
P.evaluate_seed = no_rollout
P.load_checkpoint = no_rollout
before_cache = {f: read(os.path.join(CACHE, *f.split("/")))
                for f in files_under(CACHE)}
code, log = redraw(CACHE, REDRAW)
check(code is None, f"it exits cleanly: {code}")
films = [f for f in files_under(CACHE) if f.endswith((".mp4", ".gif"))]
films_before = [f for f in before_cache if f.endswith((".mp4", ".gif"))]
check(len(films_before) == 2 and films == films_before
      and not any(f.endswith((".mp4", ".gif")) for f in files_under(ORIG))
      and not any(f.endswith((".mp4", ".gif")) for f in files_under(REDRAW))
      and all(read(os.path.join(CACHE, *f.split("/"))) == before_cache[f]
              for f in films),
      f"the original has a film per case, in the cache's extras/; the redraw "
      f"makes none, in the run folder or anywhere, and leaves those alone: "
      f"{films_before}")
diffs = same_outputs(ORIG, REDRAW)
check(not diffs, f"the redraw is the original's outputs, every figure and "
                 f"CSV byte for byte: {diffs[:4]}")
n_exact = len(pngs_and_csvs(REDRAW))
FLAT = sorted(
    [S.case_file(kind, tok) for tok in ("psi0p0.00", "l384")
     for kind in S.FIG_KINDS]
    + [f"l{n}/{f}" for n in (128, 384) for f in S.SIZE_FILES]
    + list(S.RUN_FILES)
    + ["r2_sweep_per_case/psi0p0.00.png", "r2_sweep_per_case/figure_notes.txt"])
check(n_exact >= 11 and pngs_and_csvs(REDRAW) == pngs_and_csvs(ORIG)
      and files_under(REDRAW) == files_under(ORIG) == FLAT,
      f"{n_exact} figures and CSVs, none left out of the comparison; the "
      f"redraw is the set of {len(FLAT)} files the run wrote")
# a records folder that still holds a case of a dropped size (here with links
# that lead nowhere, as after the user removed the records): the redraw does
# not open them, draws the other cases, and names nothing of l = 192
setup()
LEFT = os.path.join(BASE, "cache_leftover")
shutil.copytree(CACHE, LEFT)
with open(os.path.join(LEFT, "meta.json")) as fh:
    meta_left = json.load(fh)
meta_left["cases"]["psi0+0.00_l192"] = {
    "off": 0.0, "lattice": 192, "seeds": list(SEEDS), "cache_key": "k",
    "complete": True}
meta_left["planned_cases"] = list(meta_left["planned_cases"]) + [
    "psi0+0.00_l192"]
meta_left["size_sweep"] = [128, 384, 192]
with open(os.path.join(LEFT, "meta.json"), "w") as fh:
    json.dump(meta_left, fh)
os.makedirs(os.path.join(LEFT, "records", "psi0+0.00_l192"))
for sd in SEEDS:
    os.symlink(os.path.join(BASE, "nowhere", f"seed{sd}.npz"),
               os.path.join(LEFT, "records", "psi0+0.00_l192",
                            f"seed{sd}.npz"))
out = os.path.join(BASE, "out_leftover")
code, log = redraw(LEFT, out)
got = files_under(out)
ml = jload(os.path.join(out, "manifest.json"))
check(code is None and not [f for f in got if "l192" in f]
      and not os.path.exists(os.path.join(out, "l192"))
      and sorted(ml["cases"]) == sorted(
          jload(os.path.join(REDRAW, "manifest.json"))["cases"])
      and "l192" not in json.dumps(ml) and "l192" not in log
      and "l192" not in open(os.path.join(out, "report.txt")).read()
      and "l = 192" not in open(os.path.join(out, "report.txt")).read(),
      f"a leftover case of a dropped size is not drawn, not read and not "
      f"named: {code}, {[f for f in got if 'l192' in f]}, "
      f"{[l for l in log.splitlines() if 'l192' in l][:2]}, "
      f"{sorted(ml['cases'])} {log[-600:]!r}")

now_cache = {f: read(os.path.join(CACHE, *f.split("/")))
             for f in files_under(CACHE)}
changed = sorted(f for f in now_cache if now_cache.get(f) != before_cache.get(f))
check(sorted(now_cache) == sorted(before_cache)
      and all(f.startswith("extras/") for f in changed)
      and {f: v for f, v in now_cache.items() if not f.startswith("extras/")}
      == {f: v for f, v in before_cache.items()
          if not f.startswith("extras/")},
      f"meta.json and records/ are untouched (the redraw rewrites the "
      f"extras: {changed[:3]})")
report = open(os.path.join(REDRAW, "report.txt"), encoding="utf-8").read()
check(FP in report and "CONVENTIONS" in report
      and "- report of results_redraw" in report,
      "report.txt names the checkpoint's fingerprint")
m = jload(os.path.join(REDRAW, "manifest.json"))
book = jload(os.path.join(CACHE, "extras", S.BOOKKEEPING))
check(m["source"].startswith("make_submission, redrawn from records in ")
      and m["cache"] == CACHE and m["checkpoint_fingerprint"] == FP
      and m["checkpoint_numerics"] == meta["numerics"] == DEFAULT_NUM
      and book["records"]["folder"] == CACHE
      and book["records"]["partial"] is False
      and book["records"]["fingerprint"] == FP
      and all(e["film"] is None for e in book["compositions"].values())
      and book["compositions"]["+0.00"]["run"]["epoch"] == 7,
      "manifest.json names where it was drawn from; the bookkeeping names "
      "the records, has no film and keeps the checkpoint's epoch")
check(P.RESULTS_DIR == SAVED["RESULTS_DIR"] and P.LABEL == "fno_ch"
      and P.RECORDED_FINGERPRINT is None and P.RECORDED_NUMERICS is None
      and P.WRITE_VIDEO
      == SAVED["WRITE_VIDEO"] and P.EVAL_MODE == SAVED["EVAL_MODE"]
      and P.MODEL_PATH == SAVED["MODEL_PATH"],
      "fno_predict's settings are put back afterwards")

print("\n3. --no-records")
NOREC = os.path.join(BASE, "results_norec")
NOREC_CACHE = os.path.join(BASE, "rollout_cache", "norec")
setup()
code, log = run_main(flags(NOREC, NOREC_CACHE, "--no-records"))
check(code == 0 and P.RECORDS is False and len(ROLLED) == 6,
      "the flag turns RECORDS off for a clean run")
check(os.listdir(NOREC_CACHE) == ["extras"],
      "no records, no meta.json, and the emptied cache folder keeps only "
      "extras/")
check(pngs_and_csvs(NOREC) == pngs_and_csvs(ORIG),
      "the outputs are the same with and without records")

print("\n4. drop_case_cache keeps the records")
DROP = os.path.join(BASE, "rollout_cache", "drop")
setup()
P.RESULT_CACHE = DROP
seeds = (11, 12)
for s in seeds:
    P._save_result(P._result_path(0.0, s, 128), fake_rollout(s, 0.0, 128))
    P.save_record(0.0, None, s, fake_rollout(s, 0.0, 128), "k")
    with open(P._result_path(0.0, s, 128) + ".corrupt", "wb") as fh:
        fh.write(b"old")
n_before = len(files_under(DROP))
with contextlib.redirect_stdout(_io.StringIO()):
    n = P.drop_case_cache(0.0, None, seeds)
left = files_under(DROP)
check(n == 4 and n_before == 6 and sorted(left) == sorted(
    [f"records/psi0+0.00_l128/seed{s}.npz" for s in seeds]),
      f"the cached rollouts and their set-aside copies go, the records "
      f"stay: {left}")
real_files = P.case_cache_files
P.case_cache_files = lambda *a: [P.record_path(0.0, None, s) for s in seeds]
try:
    with contextlib.redirect_stdout(_io.StringIO()):
        n = P.drop_case_cache(0.0, None, seeds)
finally:
    P.case_cache_files = real_files
check(n == 0 and sorted(files_under(DROP)) == sorted(left),
      "and even a list that names the records is not deleted from")
with contextlib.redirect_stdout(_io.StringIO()) as buf:
    P.remove_empty_cache()
check(os.path.isdir(DROP) and sorted(files_under(DROP)) == sorted(left)
      and "stays" in buf.getvalue(),
      "and the final removal of the cache folder leaves it, and them")
for s in seeds:
    os.remove(os.path.join(DROP, "records", "psi0+0.00_l128", f"seed{s}.npz"))
with contextlib.redirect_stdout(_io.StringIO()):
    P.remove_empty_cache()
check(not os.path.exists(DROP),
      "with nothing left of the records, not even empty folders, it is "
      "removed")

print("\n4b. a record is written whole or not at all")
setup()
P.RESULT_CACHE = os.path.join(BASE, "rollout_cache", "atomic")
P.save_record(0.0, None, 21, fake_rollout(21, 0.0, 128), "k")
path = P.record_path(0.0, None, 21)
old_bytes = read(path)
real_save = np.savez_compressed


def broken(fh, **arrays):
    fh.write(b"half a file")
    raise OSError("disk full")


setattr(np, "savez_compressed", broken)
try:
    with contextlib.redirect_stdout(_io.StringIO()) as buf:
        P.keep_record(0.0, None, 21, fake_rollout(22, 0.0, 128), "k2")
        P.keep_record(0.0, None, 23, fake_rollout(23, 0.0, 128), "k2")
finally:
    setattr(np, "savez_compressed", real_save)
check(read(path) == old_bytes
      and files_under(P.RESULT_CACHE) == ["records/psi0+0.00_l128/seed21.npz"]
      and buf.getvalue().count("could not keep the record") == 2,
      "a write that fails leaves the earlier record whole, no new file and "
      "no .tmp, and says so instead of stopping the run")

print("\n4c. a seed whose record cannot be written leaves its case "
      "incomplete")
UNWRITABLE = os.path.join(BASE, "rollout_cache", "unwritable")
setup()
real_save_record = P.save_record


def save_unless_wide(off, lattice, seed, res, key):
    if lattice == 384:
        raise OSError("disk full")
    real_save_record(off, lattice, seed, res, key)


P.save_record = save_unless_wide
try:
    code, log = run_main(flags(os.path.join(BASE, "results_unwritable"),
                               UNWRITABLE, seeds="2"))
finally:
    P.save_record = real_save_record
mu = meta_of(UNWRITABLE)
check(code == 0 and log.count("could not keep the record") == 2
      and os.path.isfile(os.path.join(BASE, "results_unwritable",
                                      "l384", "snapshots_psi0p0.00.png")),
      "the run goes on and draws both cases, saying which records it could "
      "not keep")
check([mu["cases"][c]["complete"] for c in cases] == [True, False],
      "and meta.json calls the case without its records incomplete, though "
      "its outputs are written")
out = os.path.join(BASE, "out_unwritable")
code, log = redraw(UNWRITABLE, out)
check(isinstance(code, str) and cases[1] in code and cases[0] not in code
      and not os.path.exists(out),
      "so a redraw refuses the folder, naming it")
code, log = redraw(UNWRITABLE, out, "--partial")
check(code is None and os.path.isfile(os.path.join(
    out, "l128", "snapshots_psi0p0.00.png"))
      and not os.path.exists(os.path.join(out, "l384")),
      "and --partial draws the other")

print("\n5. a redraw refuses what is not whole")
PART = os.path.join(BASE, "cache_incomplete")
NOMETA = os.path.join(BASE, "cache_nometa")
shutil.copytree(CACHE, PART)
shutil.copytree(CACHE, NOMETA)
os.remove(os.path.join(NOMETA, "meta.json"))
out = os.path.join(BASE, "out_nometa")
code, log = redraw(NOMETA, out)
check(isinstance(code, str) and "meta.json" in code
      and not os.path.exists(out),
      f"no meta.json: refused, and nothing written: {code!r:.100}")
gone = os.path.join(PART, "records", cases[1], f"seed{SEEDS[1]}.npz")
os.remove(gone)
out = os.path.join(BASE, "out_incomplete")
code, log = redraw(PART, out)
check(isinstance(code, str) and "incomplete" in code and cases[1] in code
      and cases[0] not in code and "--partial" in code
      and not os.path.exists(out),
      f"a record gone: the folder is refused, naming the case: {code!r:.140}")
meta_part = jload(os.path.join(PART, "meta.json"))
meta_part["cases"][cases[0]]["complete"] = False
with open(os.path.join(PART, "meta.json"), "w") as fh:
    json.dump(meta_part, fh)
code, log = redraw(PART, out)
check(isinstance(code, str) and cases[0] in code and cases[1] in code
      and not os.path.exists(out),
      "a case meta.json calls incomplete is refused though its records are "
      "all there")
meta_part["cases"][cases[0]]["complete"] = True
with open(os.path.join(PART, "meta.json"), "w") as fh:
    json.dump(meta_part, fh)
out = os.path.join(BASE, "out_partial")
code, log = redraw(PART, out, "--partial")
got = files_under(out)
check(code is None and not [f for f in got if f.endswith("_l384.png")]
      and [f for f in got if f.endswith("_psi0p0.00.png")]
      and all(read(os.path.join(out, f)) == read(os.path.join(ORIG, f))
              for f in got if f.endswith("_psi0p0.00.png")),
      "--partial draws only the complete case, as the full run drew it")
mp = jload(os.path.join(out, "manifest.json"))
check(mp["records"]["partial"] is True
      and mp["records"]["left_out"] == [cases[1]]
      and sorted(mp["cases"]) == ["psi0p0.00"],
      "and the manifest says which case was left out")
code, log = redraw(NOMETA, os.path.join(BASE, "out_x"), "--partial")
check(isinstance(code, str) and "meta.json" in code,
      "--partial does not excuse a missing meta.json")
mixed = os.path.join(BASE, "cache_mixed")
shutil.copytree(CACHE, mixed)
bad_rec = os.path.join(mixed, "records", cases[0], f"seed{SEEDS[2]}.npz")
with np.load(bad_rec) as z:
    arrays = {k: z[k] for k in z.files}
arrays["cache_key"] = np.asarray("0" * 12)
np.savez_compressed(bad_rec, **arrays)
setup()
real = S.fno_runlog.start_log
S.fno_runlog.start_log = lambda *a, **k: None
try:
    with contextlib.redirect_stdout(_io.StringIO()):
        with contextlib.redirect_stderr(_io.StringIO()):
            mm = S.redraw(mixed, os.path.join(BASE, "out_mixed"))
finally:
    S.fno_runlog.start_log = real
check("composition +0.00" in mm["failed"]
      and os.path.isfile(os.path.join(BASE, "out_mixed", "l384",
                                  "snapshots_psi0p0.00.png")),
      f"records of one case that disagree on their cache key fail that case, "
      f"and the other is still drawn: {mm['failed']}")

print("\n6. records of another checkpoint are put aside, not mixed")
setup()
OTHER = os.path.join(BASE, "results_other")
code, log = run_main(flags(OTHER, CACHE, model=CKPT2, sizes="128",
                           seeds="2"))
meta2 = meta_of(CACHE)
arch = os.path.join(SAVED["ROOT"], "Archive", "superseded")
moved = sorted(os.listdir(arch)) if os.path.isdir(arch) else []
aside = [os.path.join(b, f) for b, _, fs in os.walk(arch) for f in fs] \
    if moved else []
check(code == 0 and meta2["fingerprint"] == FP2
      and len(moved) == 1 and moved[0].startswith(f"{LABEL}_{FP}_")
      and len([a for a in aside if a.endswith("meta.json")]) == 1
      and len([a for a in aside if a.endswith(".npz")]) == 6,
      f"the old records and meta.json moved to Archive/superseded: {moved}")
check(sorted(f for f in cache_only(files_under(CACHE))
             if f.endswith(".npz"))
      == sorted(f"records/{cases[0]}/seed{s}.npz" for s in P.EVAL_SEEDS)
      and meta2["planned_cases"] == [cases[0]],
      "and the folder now holds only this run's records")
setup()
code, log = run_main(flags(os.path.join(BASE, "results_other2"), CACHE,
                           "--no-supersede", sizes="128", seeds="2"))
check(isinstance(code, str) and "--no-supersede" in code
      and meta_of(CACHE)["fingerprint"] == FP2,
      "--no-supersede stops instead, and moves nothing")

print("\n6b. the same checkpoint in other numerics is put aside too")
NUM = os.path.join(BASE, "rollout_cache", "numerics")
NUM_RESULTS = os.path.join(BASE, "results_numerics")
NUM_LABEL = os.path.basename(NUM)


def put_aside(label):
    return sorted(d for d in os.listdir(arch) if d.startswith(label + "_")
                  ) if os.path.isdir(arch) else []


def under_archive(name):
    return sorted(os.path.relpath(os.path.join(b, f), os.path.join(arch, name))
                  .replace(os.sep, "/") for b, _, fs in os.walk(
                      os.path.join(arch, name)) for f in fs)


setup()
code, log = run_main(flags(NUM_RESULTS, NUM, sizes="128", seeds="2"))
m_a = meta_of(NUM)
check(code == 0 and m_a["numerics"] == DEFAULT_NUM and m_a["fingerprint"] == FP
      and put_aside(NUM_LABEL) == [] and "[setup] precision=fp32, TF32 off"
      in log and "numerics: fp32, TF32 off, low-k fill off" in log,
      "a first run stamps meta.json, and the log says the precision and "
      "that TF32 is off")
setup()
P.evaluate_seed = no_rollout
code, log = run_main(flags(NUM_RESULTS, NUM, sizes="128", seeds="2"))
check(code == 0 and not ROLLED and put_aside(NUM_LABEL) == []
      and meta_of(NUM)["created"] == m_a["created"]
      and "already finished" in log,
      "the same stamp again resumes: nothing moved, nothing rolled out")

stampless = os.path.join(BASE, "cache_stampless")
shutil.copytree(NUM, stampless)
meta_s = jload(os.path.join(stampless, "meta.json"))
meta_s.pop("numerics")
with open(os.path.join(stampless, "meta.json"), "w") as fh:
    json.dump(meta_s, fh)
setup()
P.evaluate_seed = P.load_checkpoint = no_rollout
out_s, out_n = os.path.join(BASE, "out_stampless"), os.path.join(BASE,
                                                                 "out_stamped")
code_s, _ = redraw(stampless, out_s)
code_n, _ = redraw(NUM, out_n)
check(code_s is None and code_n is None
      and jload(os.path.join(out_s, "manifest.json"))["checkpoint_numerics"]
      == P.OLD_NUMERICS
      and jload(os.path.join(out_n, "manifest.json"))["checkpoint_numerics"]
      == DEFAULT_NUM,
      "a redraw gives the manifest the stamp of its meta.json, and the old "
      "numerics where there is none")

setup()
P.LOWK_FILL = True
code, log = run_main(flags(NUM_RESULTS, NUM, sizes="128", seeds="2"))
kept = put_aside(NUM_LABEL)
meta_b = meta_of(NUM)
check(code == 0 and len(kept) == 1
      and re.fullmatch(rf"{NUM_LABEL}_{FP}-fp32_\d{{8}}-\d{{6}}", kept[0])
      and sorted(f for f in under_archive(kept[0])
                 if f.endswith(".npz")) == sorted(
          f"rollout_cache/{NUM_LABEL}/records/{cases[0]}/seed{s}.npz"
          for s in P.EVAL_SEEDS)
      and any(f.endswith("/meta.json") for f in under_archive(kept[0])),
      f"the fill turned on: the records and meta.json of the same checkpoint "
      f"go to Archive/superseded/<label>_<fingerprint>-<old numerics>_"
      f"<time>: {kept}")
check(meta_b["numerics"] == FILL_NUM and meta_b["fingerprint"] == FP
      and sorted(ROLLED) == sorted((0.0, 128, s) for s in P.EVAL_SEEDS)
      and meta_b["cases"][cases[0]]["complete"] is True
      and "other numerics" in log and "records of checkpoint" not in log,
      "the run rolls the seeds out again, writes a fresh meta.json with the "
      "new stamp, and says the numerics differ, not the checkpoint")
check(len(put_aside(os.path.basename(NUM_RESULTS))) == 1
      and "other numerics" in log,
      "the results folder of the old stamp went the same way")

NUM2 = os.path.join(BASE, "rollout_cache", "numerics2")
setup()
code, log = run_main(flags(os.path.join(BASE, "results_numerics2"), NUM2,
                           sizes="128", seeds="2"))
meta_c = jload(os.path.join(NUM2, "meta.json"))
meta_c.pop("numerics")
with open(os.path.join(NUM2, "meta.json"), "w") as fh:
    json.dump(meta_c, fh)
setup()
code, log = run_main(flags(os.path.join(BASE, "results_numerics3"), NUM2,
                           sizes="128", seeds="2"))
kept2 = put_aside(os.path.basename(NUM2))
check(code == 0 and len(kept2) == 1
      and re.fullmatch(rf"numerics2_{FP}-fp32_tf32_\d{{8}}-\d{{6}}", kept2[0])
      and len(ROLLED) == 2 and meta_of(NUM2)["numerics"] == DEFAULT_NUM
      and "TF32 on" in log,
      f"a meta.json with no stamp is the old numerics (TF32 on): put aside "
      f"by a run with it off: {kept2}")

setup()
P.LOWK_FILL = True
before = {f: read(os.path.join(NUM2, *f.split("/")))
          for f in files_under(NUM2) if not f.startswith("extras/")}
code, log = run_main(flags(os.path.join(BASE, "results_numerics4"), NUM2,
                           "--no-supersede", sizes="128", seeds="2"))
check(isinstance(code, str) and "--no-supersede" in code
      and "other numerics" in code and not ROLLED
      and {f: read(os.path.join(NUM2, *f.split("/")))
           for f in files_under(NUM2) if not f.startswith("extras/")}
      == before and len(put_aside(os.path.basename(NUM2))) == 1,
      "--no-supersede stops on it, and moves and writes nothing")

# the tuned schedules are part of meta.json, so they go with it
SCHED = os.path.join(BASE, "rollout_cache", "schedules")
setup()
P.RESULT_CACHE, P.ENGINE = SCHED, "schedule"
pick = {"alpha": 0.35, "dt_max": 45.0, "dt_min": 1.0, "source": "tuned",
        "signature": "x", "score": 0.9, "table": []}


def start_and_picks(lowk):
    P.LOWK_FILL = lowk
    P._PICKS.clear()
    with contextlib.redirect_stdout(_io.StringIO()):
        P.start_records(FakeModel(), {})
    got = dict(P._PICKS)
    P._META.clear()
    P._PICKS.clear()
    return got


start_and_picks(None)
meta_p = jload(os.path.join(SCHED, "meta.json"))
meta_p["schedule"] = {"+0.0": pick}
with open(os.path.join(SCHED, "meta.json"), "w") as fh:
    json.dump(meta_p, fh)
check(start_and_picks(None) == {"+0.0": pick},
      "a resubmission with the same stamp goes on with the tuned schedules")
check(start_and_picks(True) == {} and meta_of(SCHED)["schedule"] == {}
      and meta_of(SCHED)["numerics"] == FILL_NUM
      and any(f.endswith("meta.json") for f in under_archive(
          put_aside("schedules")[0])),
      "and one with another stamp tunes them again; the old picks are in "
      "the archive with the old meta.json")

print("\n7. records are also made from rollouts found in the temporary cache")
KEPT = os.path.join(BASE, "rollout_cache", "kept")
setup()
code, log = run_main(flags(os.path.join(BASE, "results_kept1"), KEPT,
                           "--no-records", "--keep-cache", sizes="128",
                           seeds="2"))
flat_files = sorted(f for f in cache_only(files_under(KEPT))
                    if f.endswith(".npz"))
check(code == 0 and len(flat_files) == 2 and not os.path.exists(
    os.path.join(KEPT, "records")),
      f"a first job with --keep-cache and --no-records leaves rollouts only: "
      f"{flat_files}")
setup()
P.evaluate_seed = no_rollout
code, log = run_main(flags(os.path.join(BASE, "results_kept2"), KEPT,
                           sizes="128", seeds="2"))
mk = meta_of(KEPT)
check(code == 0 and not ROLLED and log.count("-> reusing") == 2,
      "a second job rolls nothing out: it finds both rollouts in the cache")
check(sorted(cache_only(files_under(KEPT))) == sorted(
          ["meta.json"] + [f"records/{cases[0]}/seed{s}.npz"
                           for s in P.EVAL_SEEDS])
      and mk["cases"][cases[0]]["complete"] is True,
      "and keeps a record of each, the case complete; the cache's own files "
      "are cleaned away as usual")

print("\n7b. the lock-step engine keeps them too")
LOCK = os.path.join(BASE, "rollout_cache", "lockstep")
setup()
more = ("--engine", "lockstep", "--batch", "2", "--no-records",
        "--keep-cache")
code, log = run_main(flags(os.path.join(BASE, "results_lock1"), LOCK, *more,
                           sizes="128", seeds="3"))
held = sorted(f for f in cache_only(files_under(LOCK))
              if f.endswith(".npz"))
check(code == 0 and BATCHES == [tuple(P.EVAL_SEEDS[:2]),
                                tuple(P.EVAL_SEEDS[2:])] and len(held) == 3
      and not os.path.exists(os.path.join(LOCK, "records")),
      f"a first lock-step job rolls the seeds out in batches of 2 and "
      f"caches them: {BATCHES}")
os.remove(os.path.join(LOCK, held[0]))
setup()
code, log = run_main(flags(os.path.join(BASE, "results_lock2"), LOCK,
                           "--engine", "lockstep", "--batch", "2",
                           sizes="128", seeds="3"))
missing = [s for s in P.EVAL_SEEDS if f"_seed{s}_" in held[0]]
check(code == 0 and BATCHES == [tuple(missing)] and log.count(
    "-> reusing") == 2,
      f"a second finds two in the cache and rolls out only the third: "
      f"{BATCHES}")
check(sorted(cache_only(files_under(LOCK))) == sorted(
          ["meta.json"] + [f"records/{cases[0]}/seed{s}.npz"
                           for s in P.EVAL_SEEDS])
      and meta_of(LOCK)["cases"][cases[0]]["complete"] is True,
      "and a record of all three, those from the cache and the one rolled "
      "out")

print("\n8. --smoke keeps its records under its own root")
BOX = os.path.join(BASE, "smoke_box")
os.makedirs(os.path.join(BOX, "Work", "smoke", "models", "train_t0020"))
with open(os.path.join(BOX, "Work", "smoke", "models", "train_t0020",
                       "fno_ch.pt"), "wb") as fh:
    fh.write(b"a smoke checkpoint of this suite")
setup()
os.chdir(BOX)
fno_paths.WORK = "Work"
tree_before = {os.path.relpath(os.path.join(b, f), BOX)
               for b, ds, fs in os.walk(BOX) for f in fs + ds}
code, log = run_main(["--smoke", "--no-gt-cache", "--no-diagnostics"])
os.chdir(ROOT)
fno_paths.WORK = WORK
smoke_cache = os.path.join(BOX, "Work", "smoke", "rollout_cache")
runs = sorted(os.listdir(smoke_cache)) if os.path.isdir(smoke_cache) else []
everything = files_under(BOX)
recs = [f for f in everything if "/records/" in f or f.endswith("/meta.json")]
new = {os.path.relpath(os.path.join(b, f), BOX)
       for b, ds, fs in os.walk(BOX) for f in fs + ds} - tree_before
strays = sorted(p for p in new if not (p + os.sep).startswith(
    os.path.join("Work", "smoke") + os.sep) and p != "Work")
check(len(runs) == 1 and recs and all(
    r.startswith(f"Work/smoke/rollout_cache/{runs[0]}/") for r in recs)
      and any(r.endswith("/meta.json") for r in recs)
      and sum(r.endswith(".npz") for r in recs) == 4 and not strays,
      f"the records and meta.json are under Work/smoke/rollout_cache, and "
      f"nothing is written outside Work/smoke: {recs[:2]}, {strays}")

print("\n9. a run with views")
VIEWS_OUT = os.path.join(BASE, "views_{H}")
VCACHE = os.path.join(BASE, "rollout_cache", "views")
setup()
code, log = run_main(["--model", CKPT, "--results-dir", VIEWS_OUT,
                      "--result-cache", VCACHE, "--eval-mode", "critical",
                      "--sizes", "128", "--seeds", "2", "--no-gt-cache",
                      "--no-diagnostics", "--views", "1000,2000"])
mv = meta_of(VCACHE)
check(code == 0 and mv["views"] == [1000.0, 2000.0]
      and mv["end"] == 2000.0 and mv["cases"][cases[0]]["complete"] is True,
      f"meta.json lists the views of the run: {code}")
VREDRAW = os.path.join(BASE, "vredraw_{H}")
code, log = redraw(VCACHE, os.path.join(BASE, "vredraw"))
check(isinstance(code, str) and "{H}" in code,
      "records of a run with views want {H} in --out, as the run did")
code, log = redraw(VCACHE, VREDRAW)
bad = []
for h in ("1000", "2000"):
    bad += same_outputs(VIEWS_OUT.format(H=h), VREDRAW.format(H=h))
check(code is None and not bad,
      f"each view is redrawn as the run drew it: {bad[:3]}")

print("\n10. what a record weighs, and which cases a run plans")
sizes = {}
for lat in (128, 256):
    setup()
    P.RESULT_CACHE = os.path.join(BASE, f"size{lat}")
    g = np.random.default_rng(lat)
    res = fake_rollout(1, 0.0, lat)
    res["snapshots"] = {t: (g.standard_normal((lat, lat)),
                            g.standard_normal((lat, lat)))
                        for t in P.SNAPSHOT_TIMES}
    P.save_record(0.0, None, 1, res, "k")
    path = P.record_path(0.0, None, 1)
    with np.load(path) as z:
        raw = sum(z[k].nbytes for k in z.files)
        frames = z["snap_true"].nbytes + z["snap_pred"].nbytes
    sizes[lat] = (raw / 1e6, frames / 1e6, os.path.getsize(path) / 1e6)
check(1.0 < sizes[128][1] < 1.1 and 1.05 < sizes[128][0] < 1.2
      and sizes[128][2] <= sizes[128][0] * 1.01
      and 4.1 < sizes[256][1] < 4.3 and 4.2 < sizes[256][0] < 4.4
      and sizes[256][2] <= sizes[256][0] * 1.01,
      "the size the comment states: about 1.1 MB at l = 128 and 4.3 MB at "
      "l = 256 for four snapshot times (frames 1.05 and 4.2 MB), the "
      f"worst case of noise that does not compress: {sizes}")

setup()
OFFS5 = [-0.4, -0.2, 0.0, 0.2, 0.4]
P.EVAL_MODE, P.SIZE_SWEEP = "all", (128, 256, 512)
P.EVAL_SEEDS = P.SWEEP_SEEDS = P.make_eval_seeds(3)
plan = [(o, l, tuple(s)) for o, l, s in P.planned_cases()]
check([(o, l) for o, l, _ in plan]
      == [(o, None) for o in OFFS5] + [(o, 256) for o in OFFS5]
      + [(o, 512) for o in OFFS5]
      and all(s == tuple(P.EVAL_SEEDS) for _, _, s in plan),
      "every composition at every size: the base lattice (lattice None), "
      "then 256, then 512, five compositions each")
P.SWEEP_SEEDS = P.make_eval_seeds(2)
plan = P.planned_cases()
check([(o, l) for o, l, _ in plan][:15]
      == [(o, None) for o in OFFS5] + [(o, 256) for o in OFFS5]
      + [(o, 512) for o in OFFS5]
      and [(o, l) for o, l, _ in plan][15:] == [(0.0, 128)]
      and plan[15][2] == tuple(P.SWEEP_SEEDS)
      and [s for o, l, s in plan[:5]]
      == [tuple(P.EVAL_SEEDS) if o == 0.0 else tuple(P.SWEEP_SEEDS)
          for o in OFFS5]
      and all(s == tuple(P.SWEEP_SEEDS) for _, l, s in plan[5:15]),
      "sweep seeds that differ: the base lattice uses seeds_for(off), the "
      "other sizes the sweep seeds, and only the critical case gets a twin "
      "at the base lattice")
P.SIZE_SWEEP = (384,)
check([(o, l) for o, l, _ in P.planned_cases()][5:]
      == [(o, 384) for o in OFFS5] + [(0.0, 128)],
      "the twin is planned when the sweep does not include the base "
      "lattice, as before")
P.SIZE_SWEEP = (128, 256)
P.SWEEP_SEEDS = P.EVAL_SEEDS
P.EVAL_MODE = "critical"
check([(o, l) for o, l, _ in P.planned_cases()] == [(0.0, None), (0.0, 256)],
      "EVAL_MODE picks the compositions: the critical mode gives the "
      "critical case at each size")
P.EVAL_MODE = "off_critical"
check([(o, l) for o, l, _ in P.planned_cases()]
      == [(-0.4, None), (0.4, None), (-0.4, 256), (0.4, 256)],
      "and the off-critical mode the two off-critical ones")
P.EVAL_MODE = "none"
check([(o, l) for o, l, _ in P.planned_cases()] == [(0.0, 128), (0.0, 256)],
      "a top-up run plans the critical case at its sizes only")

print("\n7. a re-prediction keeps the complete cases it already has")
setup()
KEEP = os.path.join(BASE, "keep_cache")
shutil.rmtree(KEEP, ignore_errors=True)
P.RESULT_CACHE = KEEP
P.RECORDS = True
P.EVAL_MODE, P.SIZE_SWEEP = "critical", (128, 256)
P.EVAL_SEEDS = P.SWEEP_SEEDS = P.make_eval_seeds(2)
old_names = ["psi0+0.00_l384", "psi0+0.00_l512", "psi0+0.00_l192"]
for name, lat in zip(old_names, (384, 512, 192)):
    for sd in P.EVAL_SEEDS:
        rf = P.record_file(KEEP, name, sd)
        os.makedirs(os.path.dirname(rf), exist_ok=True)
        open(rf, "wb").close()
meta_keep = {"cases": {
    old_names[0]: {"off": 0.0, "lattice": 384, "seeds": list(P.EVAL_SEEDS),
                   "cache_key": "k", "complete": True},
    old_names[1]: {"off": 0.0, "lattice": 512, "seeds": list(P.EVAL_SEEDS),
                   "cache_key": "k", "complete": False},
    # a dropped size (192), complete and with its records: still ignored
    old_names[2]: {"off": 0.0, "lattice": 192, "seeds": list(P.EVAL_SEEDS),
                   "cache_key": "k", "complete": True},
    "psi0+0.00_l256": {"off": 0.0, "lattice": 256,
                       "seeds": list(P.EVAL_SEEDS), "cache_key": "k",
                       "complete": True}}}
grid = P.grid_cases()
kept = P.kept_cases(meta_keep, grid)
check(kept == [(0.0, 384, tuple(P.EVAL_SEEDS))],
      f"a complete case the grid does not plan is kept (l = 384); one not "
      f"marked complete (l = 512), one the grid plans anyway (l = 256) and "
      f"a complete one of a dropped size (l = 192, records on disk) are "
      f"not: {kept}")
check(P.planned_cases(meta_keep) == grid + kept,
      "and planned_cases is the grid, then the kept cases")
os.remove(P.record_file(KEEP, old_names[0], P.EVAL_SEEDS[1]))
check(P.kept_cases(meta_keep, grid) == [],
      "a case with a record missing is not kept")
P.RECORDS = False
open(P.record_file(KEEP, old_names[0], P.EVAL_SEEDS[1]), "wb").close()
check(P.kept_cases(meta_keep, grid) == [],
      "and nothing is kept when RECORDS is off")
P.RECORDS = True
asked = []
P.solve_references = lambda l, off, seeds, device: asked.append(
    (l, off, tuple(seeds))) or 0
P.TUNE = False
P._META.clear()
P._META.update(meta_keep)
with contextlib.redirect_stdout(_io.StringIO()):
    P.prewarm_references(None, skip_done=False)
check(asked and all(l in (128, 256) for l, _, _ in asked),
      f"prewarm_references asks for the references of the grid only, none "
      f"for the kept case: {asked}")
P._META.clear()
P.PLAN_ONLY = [(0.0, 384, (1, 2))]
check(P.planned_cases() == [(0.0, 384, (1, 2))],
      "PLAN_ONLY (a redraw) plans exactly the cases it names")
P.PLAN_ONLY = None
check(P.records_status(KEEP, meta_keep).keys()
      == {old_names[0], old_names[1], "psi0+0.00_l256"}
      and P.is_dropped(192) and P.is_dropped(160) and not P.is_dropped(256)
      and not P.is_dropped(None),
      "records_status leaves the dropped size out")

setup()
check(project_state() == PROJECT,
      "nothing was written into the project, outside _scratch")
print("\nALL PASSED" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
