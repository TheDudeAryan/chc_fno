"""What the three submission suites share: t_submission (sections 1 to 16),
t_submission_regen (17, 18, 24) and t_submission_publish (19 to 23, 25, 10).

The one-file t_submission took about 16 min; the sections are split so the
runner can run the three side by side.  Importing this module does what the
top of that file did: it imports fno_predict, stubs matplotlib's title methods
to record the titles, takes the snapshots of the project the last check of
every suite compares with, and defines the fake rollouts and the helpers more
than one suite needs.  It writes nothing; the name starts with an underscore
so run_all.py does not run it as a suite."""
import contextlib
import io as _io
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import types

import matplotlib.axes
import matplotlib.figure
import numpy as np

from _paths import REAL_MODEL as _REAL_MODEL, ROOT, SCRATCH

if _REAL_MODEL is None:
    sys.exit("no trained checkpoint to run against: put one at "
             "Work/models/train_t0500/fno_ch.pt (see REAL_MODEL in _paths.py)")
REAL_MODEL: str = _REAL_MODEL      # narrowed: importers see a str

os.chdir(ROOT)
import fno_predict as P

S = P._submission()
ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


TITLES = []
_suptitle = matplotlib.figure.Figure.suptitle
_set_title = matplotlib.axes.Axes.set_title


def _record_suptitle(self, t, *a, **k):
    TITLES.append(("sup", str(t)))
    return _suptitle(self, t, *a, **k)


def _record_title(self, label, *a, **k):
    TITLES.append(("ax", str(label)))
    return _set_title(self, label, *a, **k)


_text = matplotlib.axes.Axes.text


def _record_text(self, x, y, s, *a, **k):
    # the paper-style sweep labels its panels "(a)", "(b)" with ax.text
    if str(s)[:3] in ("(a)", "(b)"):
        TITLES.append(("ax", str(s)))
    return _text(self, x, y, s, *a, **k)


matplotlib.figure.Figure.suptitle = _record_suptitle
matplotlib.axes.Axes.set_title = _record_title
matplotlib.axes.Axes.text = _record_text


def _record_notes(kind, name):
    """What a figure used to say in its titles is in its notes now
    (figure_notes.txt): record them as the title was, as ``kind``."""
    fn = getattr(S, name)

    def wrapper(*a, **k):
        if k.get("notes") is None:
            k["notes"] = {}
        got = fn(*a, **k)
        for lines in k["notes"].values():
            TITLES.append((kind, " ".join(lines)))
        return got
    setattr(S, name, wrapper)


_record_notes("sup", "figure_snapshots")
for _name in ("figure_psi_pdf", "figure_growth", "figure_composition",
              "figure_free_energy"):
    _record_notes("ax", _name)


def snapshot_titles():
    return [t for kind, t in TITLES if kind == "sup" and "lattice sites" in t]


def panel_titles():
    return [t for kind, t in TITLES if kind == "ax" and t[:3] in ("(a)", "(b)")]


KEYS = ("EVAL_MODE", "SIZE_SWEEP", "EVAL_SEEDS", "SWEEP_SEEDS", "WRITE_VIDEO",
        "RESULTS_DIR", "RESULTS_TEMPLATE", "VIEWS", "WRITE_TIMING",
        "PER_SEED_FIGURES", "PER_SEED_OUTPUT", "WRITE_SUBMISSION", "WRITE_LOG",
        "START_TIME", "END_TIME", "SNAPSHOT_TIMES", "MODEL_PATH", "REDO",
        "KEEP_CACHE", "run_case", "load_checkpoint", "evaluate_seed",
        "write_case_outputs", "RESULT_CACHE", "GT_CACHE", "CACHE_RESULTS",
        "ch_ground_truth", "rollout_frames", "EXTRAS_ROOT", "KEEP_FILM")
SAVED = {k: getattr(P, k) for k in KEYS}
# Nothing this suite does may land in the project: no run log, and every
# results folder and cache inside SCRATCH.  restore() puts these back rather
# than the module's project-relative defaults ("result_cache" would be the
# project's own, since this suite runs from ROOT).
SAVED.update(WRITE_LOG=False, MODEL_PATH=REAL_MODEL,
             RESULTS_DIR=os.path.join(SCRATCH, "results_default"),
             RESULT_CACHE=os.path.join(SCRATCH, "result_cache_sub"),
             GT_CACHE=os.path.join(SCRATCH, "gt_cache_sub"))


# main() asks checkpoint_conditioned() which engine the model needs, and that
# reads the file REAL_MODEL names -- a dt-conditioned model since 2026-10-26,
# which would send every run here to the schedule engine, and leave ENGINE
# there for the sections that call the real run_case with evaluate_seed
# stubbed (section 22 died in tune_schedule on model None).  The model this
# suite stubs is a fixed-step one, so say so.
P.checkpoint_conditioned = lambda path: False


def restore():
    for k, v in SAVED.items():
        setattr(P, k, v)


restore()


def fake_model(hard=True):
    class FakeModel:
        config = {"backbone": "physical_k", "hard_mass_conservation": hard,
                  "residual_prediction": True, "mode_scaling": 1.0,
                  "zero_dc": hard}

        def n_parameters(self):
            return 0

        def parameters(self):
            yield types.SimpleNamespace(dtype="float32")
    return FakeModel()


rng = np.random.default_rng(0)
calls = []
# every rollout's mean psi differs a little from every other's, so a case's
# composition.csv changes whenever the case is written again
JITTER = itertools.count()


def fake_results(off, n, diverge=()):
    """n synthetic rollouts; indices in `diverge` stop halfway, as a real
    diverged rollout does (shorter diagnostics, NaN predictions after)."""
    t = np.arange(P.START_TIME + 1, P.END_TIME + 1)
    diag = t[::100]
    out = []
    for i in range(n):
        r2 = np.clip(1.0 - (t - t[0]) / 9000.0 - 0.003 * i, -1, 1)
        L = np.linspace(5, 14, diag.size)
        snaps = {}
        for tt in P.SNAPSHOT_TIMES:
            f = np.tanh(3 * rng.standard_normal((24, 24))) + off
            q = np.clip(f + 0.05 * rng.standard_normal(f.shape), -1.05, 1.05)
            if i in diverge and tt > P.START_TIME + 0.5 * (P.END_TIME
                                                           - P.START_TIME):
                q = np.full_like(f, np.nan)
            snaps[tt] = (f, q)
        cut = diag.size // 2 if i in diverge else diag.size
        out.append({
            "times": t, "r2": r2, "mass_drift": np.full(t.size, 1e-9),
            "mean_true": np.full(t.size, off),
            "mean_pred": np.full(t.size, off) + 1e-9
            + 1e-12 * next(JITTER),
            "diag_times": diag[:cut],
            "energy_true": np.linspace(1, 0.5, cut),
            "energy_pred": np.linspace(1, 0.5, cut),
            "t_stop": float(t[-1]), "stopped_early": False,
            "diverged_at": float(diag[cut - 1]) if i in diverge else None,
            "L_true": {k: L[:cut] for k in ("L_zero", "L_peak", "L_moment",
                                            "L_grad")},
            "L_pred": {k: L[:cut] * 1.01 for k in ("L_zero", "L_peak",
                                                   "L_moment", "L_grad")},
            "snapshots": snaps,
            "gt_seconds": 1.0, "gt_cached": False, "gt_steps": 1,
            "fno_seconds": 0.5, "fno_steps": 1900,
            "video_t": np.asarray([]), "video_true": None,
            "video_pred": None})
    return out


def fake_run_case(model, device, off, lattice=None, seeds=None):
    """run_case with made-up rollouts: the same cache key, the same outputs,
    so a later run takes the case for finished exactly as it would a real
    one."""
    calls.append((off, lattice))
    seeds = P.seeds_for(off) if seeds is None else tuple(seeds)
    lat = P.LATTICE if lattice is None else lattice
    key = P._cache_key(off, lat) if os.path.exists(P.MODEL_PATH) else None
    return P.write_case_outputs(fake_results(off, len(seeds)), seeds, off,
                                lattice, key=key)


def xr():
    """FNO's extras folder of the run being tested (in its rollout cache)."""
    return os.path.join(P.RESULT_CACHE, P.EXTRAS_NAME)


def bookkeeping(d=None):
    """The bookkeeping manifest of the extras folder (the old manifest.json
    of a results folder: cases, run stamps, last_run)."""
    with open(os.path.join(xr() if d is None else d, S.BOOKKEEPING)) as fh:
        return json.load(fh)


def setup(results_dir, hard=True):
    restore()
    # every results folder has a rollout cache of its own, as every run's
    # label gives it one
    P.RESULT_CACHE = os.path.join(SCRATCH, "rc_" + os.path.basename(
        results_dir))
    if not os.path.exists(results_dir):    # a new results folder, a new cache
        shutil.rmtree(P.RESULT_CACHE, ignore_errors=True)
    P.MODEL_FLAGS.update(hard_constraints=True, hard_mass_conservation=True,
                         zero_dc=True, label="fno_ch", note="")
    model = fake_model(hard)
    P.load_checkpoint = lambda path, device=None: (
        model, {"best": {"epoch": 7, "score": 0.91, "survival": 2000.0}})
    P.run_case = fake_run_case
    P.RESULTS_DIR = results_dir
    P.WRITE_TIMING = False
    P.PER_SEED_FIGURES = False
    calls.clear()
    TITLES.clear()


EXIT = []


def run_main(argv):
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        EXIT.append(P.main(argv))
    return buf.getvalue()


def manifest(d):
    with open(os.path.join(d, "manifest.json")) as fh:
        return json.load(fh)


def clean(*dirs):
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


def read(p):
    with open(p, "rb") as fh:
        return fh.read()


def files_under(d):
    """Every file below d, as a path relative to d with / separators."""
    return sorted(os.path.relpath(os.path.join(b, f), d).replace(os.sep, "/")
                  for b, _, fs in os.walk(d) for f in fs)


def tree(d):
    """Every file below d with its bytes."""
    return {f: read(os.path.join(d, *f.split("/"))) for f in files_under(d)}


def leftovers_tree(d):
    """Scratch folders, holds and temporaries anywhere below d."""
    return [os.path.join(b, x) for b, dirs, fs in os.walk(d) for x in dirs + fs
            if x.startswith(".partial_") or x.endswith((".tmp", ".tmp.npz"))
            or ".partial." in x]


def old_names(d):
    """Alias copies: a file of the l = 128 box named for it (snapshots_l128
    .png ...), anywhere below d.  The base box is the case psi0p0.00."""
    return [f for f in files_under(d) if "_l128" in f.split("/")[-1]
            and "sweep_seeds" not in f]


def flat_figs(tokens, kinds=None):
    """The run-folder figures of these case tokens: the kinds a run to
    t_end <= 2000 draws, as paths from the run folder: the box size's
    folder, then <kind>_<composition>.png (S.case_file)."""
    kinds = kinds or list(S.FIG_KINDS)
    return [S.case_file(k, t) for t in tokens for k in kinds]


def sweep_files(sweep_sizes, comps):
    """The files of the two sweep folders: r2_sweep_per_size/l<size>.png for
    each size in ``sweep_sizes`` and r2_sweep_per_case/<token>.png for each
    composition token in ``comps``, with the figure_notes.txt of each folder
    that has an image."""
    out = []
    if sweep_sizes:
        out += [f"r2_sweep_per_size/l{n}.png" for n in sweep_sizes]
        out.append("r2_sweep_per_size/figure_notes.txt")
    if comps:
        out += [f"r2_sweep_per_case/{c}.png" for c in comps]
        out.append("r2_sweep_per_case/figure_notes.txt")
    return out


def run_files(sizes, sweep_sizes=None,
              comps=("psi0m0.40", "psi0p0.00", "psi0p0.40")):
    """The run-wide files of a run folder that has the box sizes ``sizes``:
    those at the top, in each size folder and in the two sweep folders (an
    image for each of ``sweep_sizes``, default all sizes, and for each
    composition token in ``comps``)."""
    return (list(S.RUN_FILES) + [f"l{n}/{f}" for n in sizes
                                 for f in S.SIZE_FILES]
            + sweep_files(sizes if sweep_sizes is None else sweep_sizes,
                          comps))


def submission_dirs():
    """Submission* folders at the project's top and anywhere in SCRATCH."""
    top = {os.path.join(ROOT, d) for d in os.listdir(ROOT)
           if d.startswith("Submission") and os.path.isdir(os.path.join(ROOT,
                                                                       d))}
    deep = {os.path.join(b, d) for b, dirs, _ in os.walk(SCRATCH)
            for d in dirs if d.startswith("Submission")}
    return top | deep


PLAIN = ("psi_distribution.csv", "domain_growth.csv", "composition.csv")
TOKENS = ("psi0m0.40", "psi0p0.00", "psi0p0.40", "l512", "psi0m0.40_l512",
          "psi0p0.40_l512")

def wcase(results, seeds, off, lattice, out, key=None, videos=None,
          root=None, **kw):
    """S.write_case into the folder ``out`` itself (sizedir ""), the way the
    critical twin is written, with the case's extras in a folder of their own
    beside it (``out`` + "_x").  The suites that test publishing and
    recovery work on one folder of plain file names; the box size's folder
    of a run is tested through a run."""
    kw.setdefault("sizedir", "")
    return S.write_case(results, seeds, off, lattice, out, key=key,
                        videos=videos, root=out if root is None else root,
                        extras=kw.pop("extras", out + "_x"), **kw)


FLATF = tuple(S.fig_name(k, "psi0p0.00") for k in S.FIG_KINDS)   # one case's figures

RES = os.path.join(SCRATCH, "results_sub")
clean(RES, RES + "_nohard", os.path.join(SCRATCH, "rc_results_sub"),
      os.path.join(SCRATCH, "rc_results_sub_nohard"))
LOGS = os.path.join(ROOT, "Work", "logs")
logs_before = set(os.listdir(LOGS)) if os.path.isdir(LOGS) else set()
SUB_BEFORE = submission_dirs()
# What a run with the file's default folders would make in the project, now
# and before the folder layout: the folders, and whatever is in them.
PROJECT_OWN = ("result_cache", "results", "gt_cache", "Work", "Results")


def own_state():
    state = set()
    for name in PROJECT_OWN:
        top = os.path.join(ROOT, name)
        if os.path.exists(top):
            state.add(name)
        for here, dirs, files in os.walk(top):
            state.update(os.path.relpath(os.path.join(here, f), ROOT)
                         for f in dirs + files)
    return state


OWN_BEFORE = own_state()


# --- helpers more than one of the three suites needs -------------------------
# Each sat in the middle of the section of the one-file suite that first used
# it (named in the comment); the sections now only call it.

# section 5, used again by 17: the figure the stubs put back
real_growth = S.figure_growth


# section 14, used again by 19
def stopped(r):
    """r as EARLY_STOP leaves a seed that stopped halfway."""
    r = dict(r)
    n = r["times"].size // 2
    cut = r["diag_times"].size // 2
    r.update({k: np.asarray(r[k])[:n] for k in ("times", "r2", "mass_drift",
                                                "mean_true", "mean_pred")})
    r.update(diag_times=r["diag_times"][:cut],
             energy_true=r["energy_true"][:cut],
             energy_pred=r["energy_pred"][:cut],
             L_true={k: v[:cut] for k, v in r["L_true"].items()},
             L_pred={k: v[:cut] for k, v in r["L_pred"].items()},
             snapshots={tt: v for tt, v in r["snapshots"].items()
                        if tt <= r["times"][-1]},
             t_stop=float(r["times"][-1]), stopped_early=True)
    return r


# section 18, used again by 19 and 20
def flat(lat):
    return {"lattice": lat, "times": np.arange(101.0, 2001.0),
            "mean_curve": np.linspace(1, 0.9, 1900), "stats": {"n_seeds": 3}}


# section 9, used by 11, 12, 20, 21 and 23
def contents(d):
    return {f: read(os.path.join(d, f)) for f in os.listdir(d)
            if os.path.isfile(os.path.join(d, f))}


def leftovers(d):
    return [f for f in os.listdir(d)
            if f.startswith(".partial_") or f.endswith((".tmp", ".tmp.npz"))
            or ".partial." in f]


real_replace = os.replace


def replace_failing(when, exc: type[BaseException] = PermissionError):
    def rep(src, dst, *a, **k):
        if when(os.fspath(src), os.fspath(dst)):
            raise exc(13, "locked (test)", dst) if exc is PermissionError \
                else exc()
        return real_replace(src, dst, *a, **k)
    return rep


def parent(p):
    return os.path.basename(os.path.dirname(p))


STG = S.stage_name("psi0p0.00")      # the run folder's stage for the case
HOLD = STG + ".old"                  # and its holding folder
XSTG = S.STAGE                       # the extras' stage


# section 12, used again by 20 and 23
real_revert = S._revert


def kill_mid_publish(folder, at="psi_distribution_psi0p0.00.png", off=0.0):
    """Leave a publish as a killed run would: half placed, journal kept."""
    os.replace = replace_failing(
        lambda s_, d_: parent(s_).startswith(".partial_psi")
        and os.path.basename(d_) == at, KeyboardInterrupt)
    S._revert = lambda hold, out: False
    try:
        wcase(fake_results(off, 3), (1, 2, 3), off, 128, folder,
                     videos=False)
    except KeyboardInterrupt:
        pass
    finally:
        os.replace = real_replace
        S._revert = real_revert


def project_untouched():
    """The last two checks of every suite: nothing it ran wrote a run log or a
    Submission folder, or anything else, into the project (the end of the old
    section 10, which came last in the one-file suite)."""
    logs_after = set(os.listdir(LOGS)) if os.path.isdir(LOGS) else set()
    check(logs_after == logs_before,
          f"no run log written into the project's Work/logs/ "
          f"({len(logs_after - logs_before)} new)")
    check(submission_dirs() == SUB_BEFORE and own_state() <= OWN_BEFORE,
          f"the whole suite made no Submission folder anywhere, and nothing new "
          f"in the project's {', '.join(PROJECT_OWN)} "
          f"({sorted(own_state() - OWN_BEFORE)[:3]})")


def finish():
    """Say how the suite went and leave with its exit status."""
    print("\n" + ("SUBMISSION PASS" if ok else "SUBMISSION FAIL"))
    sys.exit(0 if ok else 1)
