"""Where the project is, so no suite has to hardcode it.

The tests run on a cluster and on a Windows cloud PC, where the project sits at
different absolute paths, so everything is derived from this file's own
location instead.  Importing this also puts the project on sys.path, which is
what lets a suite say ``import fno_train`` from inside tests/.

SCRATCH is where suites do their damage: checkpoints, ground-truth caches, log
files and throwaway figures.  It is inside tests/ rather than the project root
so a failed run never leaves debris next to the real results.  The environment
variable FNO_TEST_SCRATCH names another folder (created if missing): run_all.py
gives every suite its own, so suites running side by side never share one.

REAL_MODEL is the user's trained checkpoint, which a few suites run against
(a short rollout needs a real network).  It is looked for where the project
keeps it now, where it sat before the folder layout, and on the laptop; None
on a machine with none of them, and those suites then say so and stop.
FNO_TEST_MODEL (an environment variable) names one to use before any of those.

REAL_RECORDS is a rollout-cache folder of a finished run (its meta.json and
records/), which t_flat copies and redraws.  FNO_TEST_RECORDS names one; the
laptop's is the fallback; None where there is neither, and t_flat then skips
that part.  A folder of the project's own (Work/rollout_cache/<label>) can be
named: the suite only reads it.

REAL_MODEL is a dt-conditioned model (every checkpoint of the project is one
since 2026-10-23): it is run by the schedule engine, takes a step length in
every call and, with --seed-set group, scores on the 50 group seeds.  The suites
written for the fixed-step engines (seed, lock-step) call fixed_step_model()
instead: a small untrained fixed-step network, built on first use in SCRATCH
(FNO_TEST_FIXED_MODEL names a trained one to use in its place).

The training figure of a run folder (make_submission.WRITE_TRAINING) is off
for every suite: they check a run folder file by file, and some roll out the
real checkpoint, which has its history beside it.  t_flat and t_report, which
test the figure, turn it on themselves.
"""
import os
import sys

os.environ["FNO_NO_TRAINING_FIGURE"] = "1"
# Figures are saved at half the DPI with light PNG compression (see
# fno_paths.save_figure); no suite depends on a figure's pixel size except
# through fno_paths.figure_dpi.
os.environ["FNO_FAST_FIGURES"] = "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.abspath(
    os.environ.get("FNO_TEST_SCRATCH")
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), "_scratch"))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.makedirs(SCRATCH, exist_ok=True)

REAL_MODEL = next((p for p in (
    os.environ.get("FNO_TEST_MODEL", ""),
    os.path.join(ROOT, "Work", "models", "train_t0500", "fno_ch.pt"),
    os.path.join(ROOT, "fno_ch.pt"),
    "E:/HPC/fno_ch.pt") if os.path.exists(p)), None)

REAL_RECORDS = next((p for p in (
    os.environ.get("FNO_TEST_RECORDS", ""),
    "E:/CH_FNO/Work/rollout_cache/train_t0300_lap_pred_t10000")
    if os.path.isdir(p)), None)


def project_state():
    """What is in the project folder now: its top-level names, and every
    file and folder under Work/ and Results/, where runs write.  A suite
    takes this before and after, and anything new in between is a run that
    wrote into the project instead of SCRATCH."""
    state = {name for name in os.listdir(ROOT) if name != "__pycache__"}
    for top in ("Work", "Results"):
        for here, dirs, files in os.walk(os.path.join(ROOT, top)):
            state.update(os.path.relpath(os.path.join(here, f), ROOT)
                         for f in dirs + files)
    return state


def fixed_step_model():
    """The path of a fixed-step checkpoint for the suites that run the seed or
    lock-step engine on a real network: FNO_TEST_FIXED_MODEL when it names a
    file, else a small untrained one built once under SCRATCH (16 channels, 8
    modes, the kind of model that takes one DT_STEP per call, with the
    constraints the trained ones have, and neither the mu channel nor the
    low-k fill, which came with the dt-conditioned retrain).  It sits in a folder called
    train_t0500, so the run label a suite derives from it is the usual one.
    fno_train's constants are put back after it is built."""
    named = os.environ.get("FNO_TEST_FIXED_MODEL", "")
    if named and os.path.exists(named):
        return named
    path = os.path.join(SCRATCH, "fixed_model", "train_t0500", "fno_ch.pt")
    if os.path.exists(path):
        return path
    import torch
    import fno_train as T
    names = ("CHANNELS", "MODES", "DT_CONDITIONED", "MU_CHANNEL", "LOWK_FILL")
    kept = {k: getattr(T, k) for k in names}
    T.CHANNELS, T.MODES, T.DT_CONDITIONED = 16, 8, False
    T.MU_CHANNEL = T.LOWK_FILL = False
    try:
        torch.manual_seed(0)
        model = T.build_model()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        T.save_model(model, path, best={
            "epoch": 1, "score": 0.9, "survival": 2000.0, "accuracy": 0.9,
            "peak_survival": 2000.0})
    finally:
        for k, v in kept.items():
            setattr(T, k, v)
    return path
