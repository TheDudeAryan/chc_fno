"""The submission figures, and results folders redrawn from records and caches.

A run's folder, Results/<run label>/, is flat.  For each case
(psi0m0.40 ... psi0p0.40, and the box sizes l256, l512; l128 is
psi0p0.00) it holds <kind>_<case>.png for snapshots (t = 100 to 10000 in one
figure), phase_ordering_kinetics, domain_growth (L_half against the
Lifshitz-Slyozov t^(1/3)), psi_distribution (P(psi) at the final time, Solver
magenta / FNO blue), composition and free_energy; and for the run
composition.csv, composition_all.png, r2_sweep_per_size/l<size>.png (mean
R2(t) at one box size, a line for each composition) and
r2_sweep_per_case/<composition>.png (one composition, a line for each box
size), each of those two folders with a figure_notes.txt of its own,
manifest.json (the flat schema) and report.txt, and, where
the model's history is in Work/models/, training_curves.png (its training and
validation curves against epoch).  The figures carry no titles and no
numbers beyond their axes and short legends; figure_notes.txt has, for each
figure file, what it shows and the numbers that would have been on it.  What
only the FNO has (per-case summaries and CSVs, the best seed's folder, the film,
diagnostics.json, sweep_curves.npz, the bookkeeping manifest) is kept in
<rollout cache>/extras/, never in Results/.  The root of Results/ has the
overview: README.md, comparison.png (the critical quench of every run),
comparison_<case>.png (the same for each other case, when the runs hold more
than the critical quench), comparison_t<END>.png (the model sizes of one
training horizon, one panel for each number of modes), r2_vs_model_size.png
(pooled R2 against the parameters of the 24-mode runs), laplacian_on_off.png
(pooled R2(t) with and without the Laplacian output) and figure_notes.txt
(the same notes for these).

fno_predict draws a case's figures into the run folder as the case finishes,
by calling write_case here, and refreshes the overview when the run ends.
From the command line this redraws results folders through the very code a
prediction run uses (fno_predict.evaluate_all), so a redrawn folder is laid out
and drawn exactly as a live one.  No model is run and no trajectory solved;
the records and caches are only read.

Usage:
    python make_submission.py
        every complete run in Work/rollout_cache, redrawn from its per-seed
        records into Results/<run label>/, then the overview
    python make_submission.py --run LABEL
        one run, Work/rollout_cache/LABEL, into Results/LABEL, then the
        overview
    python make_submission.py --overview
        only the overview: Results/README.md (a table of the run folders that
        have a manifest.json, the others listed apart and left alone) and
        comparison.png (the critical quench of every run on one log-t axis),
        its per-case and per-size forms, and the training_curves.png of each
        run folder whose model has a history (the one thing it draws in a run
        folder).  A prediction run does this itself when it ends (--index is
        the old name)
    python make_submission.py --records Work/rollout_cache/LABEL --out DIR
        one run, from the per-seed records it kept (fno_predict --no-records
        keeps none), with no model, no solver and no cache.  Refused when
        meta.json is missing or a case is incomplete; --partial draws the
        complete cases.  The film needs frames a record does not keep, so
        there is none.
    python make_submission.py --cache CACHE --out DIR
        one run, from a folder of cached rollouts such as
        Work/rollout_cache/result_cache_50_seed_300 (a run that kept no
        records)
    python make_submission.py --all
        every run in RUNS (the older caches) -> ALL_OUT, which is
        Results/results_runs_2026-09-30/, with its own comparison.png and
        README.md
    python make_submission.py --smoke
        the newest python fno_predict.py --smoke run, in Work/smoke
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import shutil
import socket
import textwrap
import time
import traceback
import types
import warnings
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatter, MaxNLocator, NullFormatter

import fno_paths
import fno_predict as P
import fno_runlog

ALL_OUT = os.path.join(fno_paths.RESULTS, "results_runs_2026-09-30")
DISPLAY_NAME = "Cahn-Hilliard FNO"

# The runs --all regenerates, each from its cache into ALL_OUT/<key>/.
# "model" is the checkpoint the cache was made with: provenance only.
RUNS = (
    {"key": "A_train_t300_eval_10seed",
     "cache": fno_paths.rollout_dir("result_cache_10_seed"),
     "model": fno_paths.model_path("train_t0300"),
     "title": "trained to t = 300, evaluated on 10 seeds",
     "note": "the original evaluation"},
    {"key": "B_train_t300_eval_50seed",
     "cache": fno_paths.rollout_dir("result_cache_50_seed_300"),
     "model": fno_paths.model_path("train_t0300"),
     "title": "trained to t = 300, evaluated on 50 seeds",
     "note": "the same checkpoint as A, evaluated on fifty seeds instead of "
             "ten"},
    {"key": "C_train_t500_eval_50seed",
     "cache": fno_paths.rollout_dir("result_cache_50_seed_500"),
     "model": fno_paths.model_path("train_t0500"),
     "title": "trained to t = 500, evaluated on 50 seeds",
     "note": "dataset extended to t = 500"},
)

COMPOSITIONS = (-0.4, -0.2, 0.0, 0.2, 0.4)
BASE_LATTICE = 128
SIZE_SWEEP = (128, 256, 512)
# The six cases every run has: five compositions at 128 and the critical
# quench at 256.  Runs made before the box-size grid (2026-11-09) have only
# these; their pooled R2 is the headline that older runs can be compared on.
# The box sizes of that protocol are 160, 192 and 256; 160 and 192 are
# dropped (P.DROPPED_SIZES), so a legacy cache is read at 256 only.
LEGACY_SIZES = tuple(s for s in (160, 192, 256) if s not in P.DROPPED_SIZES)
COMMON_CASES = ("psi0m0.40", "psi0m0.20", "psi0p0.00", "psi0p0.20",
                "psi0p0.40", "l256")
SNAPSHOT_TIMES = (100.0, 500.0, 1000.0, 2000.0)
START_TIME = 100.0
PDF_TIME = 2000.0
R2_THRESHOLD = 0.80
LONG_SPAN = 20.0
# About this many log-spaced times in a case's composition.csv.
COMPOSITION_POINTS = 200
# The times whose R2 the run README and the index report, pooled and per
# group, as the group's other architectures do.
R2_REPORT_TIMES = (2000.0, 5000.0, 10000.0)

C_TRUTH = "#C51B8A"
C_PRED = "#2B5B84"
C_LS = "#222222"
SWEEP_COLOURS = ("#08519C", "#4292C6", "#C51B8A", "#41AB5D", "#005A32")
SWEEP_MARKS = ("o", "s", "^", "D", "v")
RUN_COLORS = ("#1b4f9c", "#c2188b", "#3aa655", "#c07000", "#00868b",
              "#7b3fa0")

WRITE_VIDEOS = True
# The model's training curves (TRAINING_FIG) go into the run folders and the
# overview; False, or FNO_NO_TRAINING_FIGURE in the environment, leaves them
# out.  The suites turn it off (tests/_paths.py): they check a run folder file
# by file, and some roll out the real checkpoint, which has its history beside
# it.  The two that test the figure turn it on again.
WRITE_TRAINING = not os.environ.get("FNO_NO_TRAINING_FIGURE")

PDF_BINS = 25
PDF_RANGE = (-1.1, 1.1)
PDF_YLIM = (1e-2, 1e1)
FIG_DPI = 200

PHASE_CMAP, ERR_CMAP = "RdBu_r", "magma"

# --- the run folder: run-wide files at the top, one folder per box size ----
# Per case, <size folder>/<kind>_<composition token>.png, e.g.
# l256/snapshots_psi0p0.20.png.
FIG_KINDS = ("snapshots", "phase_ordering_kinetics", "domain_growth",
             "psi_distribution", "composition", "free_energy")
# (a run's top has no figure_notes.txt: each sweep folder and each size
# folder has its own)
RUN_FILES = ("composition.csv", "manifest.json", "report.txt")
# The two folders of single R2 sweep images at the top of a run folder:
# r2_sweep_per_size/l<size>.png and r2_sweep_per_case/<composition token>.png,
# each with a figure_notes.txt of its own.
SWEEP_DIRS = ("r2_sweep_per_size", "r2_sweep_per_case")
# What each box-size folder holds besides the case figures.
SIZE_FILES = ("composition_all.png", "r2_time.png", "figure_notes.txt")
# Everything FNO keeps beyond the flat layout (per-case summaries and CSVs, the
# best seed's folder, the film, diagnostics, the internal bookkeeping) lives
# in <rollout cache>/extras/, beside the records and never in Results/.
EXTRAS_DIR = "extras"
# The columns of snapshots_<case>.png; a run uses those of its records that it
# has.
SNAPSHOT_COLUMNS = (100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)
# The group protocol's snapshot times; a run uses those of its records that it has.
GROUP_SNAPSHOTS = (100.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0,
                 10000.0)
CSV_COLS = ("case", "psi0", "t", "psi_mean_truth", "psi_mean_pred",
            "A_pct_truth", "B_pct_truth", "A_pct_pred", "B_pct_pred",
            "drift_pred", "n_seeds", "l")
CURVE_POINTS = 400
DIVERGE_ABS = 5.0


# The kinds of the bookkeeping manifest (extras/submission_manifest.json):
# a composition at the base lattice, the critical quench at another size, and
# any other composition at another size.
KINDS = ("compositions", "sizes", "grid")


def comp_token(off):
    """The composition part of a case's names: psi0m0.40 ... psi0p0.40."""
    return f"psi0{'m' if off < -1e-12 else 'p'}{abs(off):.2f}"


def case_token(off, lattice=None):
    """The name of a case in manifest.json.  A composition at the base box is
    psi0m0.40 ... psi0p0.40; the critical quench at another size is l160,
    l256, l512; any other composition at another size is psi0p0.20_l256.
    ``lattice`` is None for a case at the base box.  (The internal case name
    of records and meta.json, fno_predict.case_name, is not this.)"""
    if lattice is None:
        return comp_token(off)
    if abs(off) < 1e-12:
        return f"l{int(lattice)}"
    return f"{comp_token(off)}_l{int(lattice)}"


def size_folder(lattice):
    """The folder of a box size in the run folder: l128, l256, ..."""
    return f"l{int(lattice)}"


def fig_name(kind, stem):
    """A case figure's file name; ``stem`` is the composition token."""
    return f"{kind}_{stem}.png"


def token_place(token):
    """(size folder, composition token) of a manifest token, or None when it
    is not one: psi0p0.20 -> (l128, psi0p0.20), l160 -> (l160, psi0p0.00),
    psi0p0.20_l256 -> (l256, psi0p0.20)."""
    m = re.fullmatch(r"(psi0[mp]\d+\.\d+)(?:_l(\d+))?", token)
    if m:
        return size_folder(int(m.group(2) or P.LATTICE)), m.group(1)
    m = re.fullmatch(r"l(\d+)", token)
    if m:
        return size_folder(int(m.group(1))), comp_token(0.0)
    return None


# --- time grids: any ascending schedule, not a fixed step ------------------
def grid_index(t, want):
    """Index of the sample of the ascending grid ``t`` that is the time
    ``want``, or None when no sample is that time.  A sample is it when it
    lies within half the spacing to its nearest neighbour, so a fixed grid
    behaves as before and a schedule that hits ``want`` exactly is exact."""
    t = np.asarray(t, dtype=float)
    if not t.size:
        return None
    k = int(np.searchsorted(t, want))
    i = min((j for j in (k - 1, k) if 0 <= j < t.size),
            key=lambda j: abs(t[j] - want))
    gaps = [abs(t[i] - t[j]) for j in (i - 1, i + 1) if 0 <= j < t.size]
    tol = 0.5 * min(gaps) if gaps else 0.5
    return i if abs(t[i] - want) <= tol + 1e-9 else None


def grid_slack(t):
    """Half the last spacing of a grid: how far past its last sample a time
    may lie and still count as the end of the window."""
    t = np.asarray(t, dtype=float)
    return 0.5 * float(t[-1] - t[-2]) if t.size > 1 else 0.5


def result_start(res):
    """The time a rollout was seeded at: its ``t_start`` when the result
    carries one (a schedule engine stores it), else one spacing before its
    first sample (a fixed-step rollout, whose first sample is the first
    step)."""
    if res.get("t_start") is not None:
        return float(res["t_start"])
    t = np.asarray(res["times"], dtype=float)
    return float(t[0] - (t[1] - t[0])) if t.size > 1 else float(t[0]) - 1.0


def comp_label(off):
    return "critical quench" if abs(off) < 1e-12 else "off-critical quench"


def comp_math(off):
    return rf"$\bar\psi_0 = {off:+.1f}$"


def comp_slug(off):
    return comp_token(off)


def cache_group(cache, off, lattice):
    """The newest reasonably complete generation of cached rollouts.

    A cache accumulates several generations, one per model file it was built
    against; only rollouts sharing a 12-hex key came from the same model, so
    mixing them would silently average two different models.

    Picking the largest group is not enough: result_cache_10_seed holds two
    generations of exactly ten for every case, one from 08-25 and one from
    09-08, so size alone leaves the choice to dict ordering.  Nor is picking
    the newest enough on its own -- a stray one-seed group from a local test
    would win.  So: keep the generations at least half the size of the biggest,
    then take the most recent of those.  That prefers a fresh but short run
    (l = 256 stopped at 44) over a complete but superseded one.
    """
    tag = "" if abs(off) < 1e-12 else f"_off{off:+.2f}"
    pattern = os.path.join(cache, f"fno_ch{tag}_l{lattice}_seed*_*.npz")
    groups = collections.defaultdict(list)
    for path in glob.glob(pattern):
        m = re.search(r"_seed(\d+)_([0-9a-f]{12})\.npz$", path)
        if m:
            groups[m.group(2)].append((int(m.group(1)), path))
    if not groups:
        return (), [], None

    biggest = max(len(v) for v in groups.values())
    viable = {k: v for k, v in groups.items() if len(v) >= 0.5 * biggest}
    key = max(viable, key=lambda k: max(os.path.getmtime(p)
                                        for _, p in viable[k]))
    best = groups[key]
    order = {s: i for i, s in enumerate(P.EVAL_SEEDS)}
    best.sort(key=lambda it: (order.get(it[0], 10 ** 6), it[0]))
    if len(groups) > 1:
        dropped = ", ".join(f"{k}({len(v)})" for k, v in groups.items()
                            if k != key)
        print(f"  [cache] psi{off:+.1f} l{lattice}: using {key} "
              f"({len(best)} seeds), ignoring {dropped}", flush=True)
    return tuple(s for s, _ in best), [p for _, p in best], key


def load_case(cache, off, lattice):
    seeds, files, key = cache_group(cache, off, lattice)
    if not files:
        return (), [], None
    return seeds, [P._load_result(f) for f in files], key


def best_index(results):
    """The realisation the snapshots and the film both show.

    Keeping them on the same seed matters: a reader who has just looked at
    four still frames should be watching those same frames move, not a
    different trajectory with the same label.  Preferred, in order: a seed
    that never diverged and has a prediction at the final time; any seed with
    a finite prediction there (one that diverged later); any seed.  A late
    divergence can still have the best mean, and the figure would otherwise
    caption a blank final column.
    """
    scores = [float(np.nanmean(r["r2"])) for r in results]
    t_end = rollout_times(results)[1]
    at_end = [i for i, r in enumerate(results) if t_end in r["snapshots"]
              and np.all(np.isfinite(r["snapshots"][t_end][1]))]
    healthy = [i for i in at_end if results[i].get("diverged_at") is None]
    return max(healthy or at_end or range(len(results)),
               key=lambda i: scores[i])


def seed_ranks(results, seeds):
    """The best, median and worst seed of a case, each with its scores.

    One score ranks them, the mean R2 over the rollout that best_index uses,
    and best is best_index's own pick, so a seed that diverged cannot be it
    and it need not have the highest score.  Median and worst come from the
    scores alone: with an even count the median is the lower of the middle
    two, and ties keep the order of the seeds.  r2_at_end is the seed's R2 at
    the time the case is judged, a diverged seed at P.R2_FLOOR, None if the
    seed has no snapshot there.
    """
    scores = np.array([float(np.nanmean(r["r2"])) for r in results])
    order = np.argsort(np.where(np.isfinite(scores), scores, -np.inf),
                       kind="stable")
    t_end = rollout_times(results)[1]
    picks = {"best": best_index(results),
             "median": int(order[(len(order) - 1) // 2]),
             "worst": int(order[0])}
    out = {}
    for name, i in picks.items():
        snap = results[i]["snapshots"].get(t_end)
        end = None if snap is None else float(P.r2_score(*snap))
        if end is not None and not np.isfinite(end):
            end = P.R2_FLOOR
        out[name] = {"seed": int(seeds[i]), "mean_r2": float(scores[i]),
                     "r2_at_end": end}
    return out


def rank_line(ranks):
    """The line that names the best, median and worst seed."""
    return ("by mean R2 over the rollout: "
            + ", ".join(f"{name} seed {ranks[name]['seed']} "
                        f"({ranks[name]['mean_r2']:.3f})"
                        for name in ("best", "median", "worst")))


# figure_notes.txt: what the figures of a folder show and the numbers that
# are not on the images.  One block per figure file, a blank line between:
# the file name, then its lines, each indented two spaces.
NOTES_FILE = "figure_notes.txt"


def plain_text(text):
    """Text drawn with mathtext, as plain text for figure_notes.txt."""
    return (text.replace(r"\bar\psi_0", "psi0").replace(r"\bar\psi", "psi")
            .replace(r"\geq", ">=")
            .replace(r"\leq", "<=").replace("$", ""))


def notes_text(blocks):
    """The text of figure_notes.txt for {file name: [line, ...]}; a long line
    is wrapped, its continuation indented four spaces."""
    out = []
    for name in sorted(blocks):
        body = [textwrap.fill(plain_text(line), width=78,
                              initial_indent="  ",
                              subsequent_indent="    ")
                for line in blocks[name]]
        out.append("\n".join([name] + body))
    return "\n\n".join(out) + "\n"


def read_notes(path):
    """The blocks of a figure_notes.txt (notes_text read back); empty when
    there is none."""
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return {}
    blocks = {}
    for chunk in text.split("\n\n"):
        lines = chunk.strip("\n").split("\n")
        if not lines[0] or lines[0].startswith(" "):
            continue
        body = []
        for line in lines[1:]:
            if line.startswith("    ") and body:
                body[-1] += " " + line.strip()
            else:
                body.append(line.strip())
        blocks[lines[0]] = body
    return blocks


def write_figure_notes(folder, blocks):
    """Put ``blocks`` into folder/figure_notes.txt: the blocks of the other
    figures are kept, those of figures no longer in the folder dropped.  No
    file is left when no block is."""
    path = os.path.join(folder, NOTES_FILE)
    have = read_notes(path)
    have.update(blocks)
    have = {n: v for n, v in have.items()
            if os.path.isfile(os.path.join(folder, n))}
    if have:
        write_text(path, notes_text(have), shared=True)
        print(f"[saved] {path}")
    else:
        _discard(path)


MANIFEST = "manifest.json"
# FNO's own record of what a run folder holds (cache keys, curve digests, the
# run stamp): in extras/, beside sweep_curves.npz.  manifest.json, in the run
# folder, is the flat schema and is built from it.
BOOKKEEPING = "submission_manifest.json"
CURVES_FILE = "sweep_curves.npz"
CURVES_TMP = "sweep_curves.tmp.npz"
CURVE_PARTS = ("times", "curve", "n", "lattice")
CURVE_OPTIONAL = ("n_end", "n_div", "sd")
PUBLISH_SETTLED = "_settled.json"
SWEEP_TMP = "sweeps.partial.png"
COMPOSITION_FILE = "composition.csv"
COMPOSITION_FIG = "composition_all.png"
COMPOSITION_TMP = "composition_all.partial.png"
INDEX_FILE = "README.md"
INDEX_MARK = ("<!-- written by python make_submission.py --overview; edits "
              "are overwritten -->")
COMPARISON_FILE = "comparison.png"
SIZE_FIG = "r2_vs_model_size.png"
LAPLACIAN_FIG = "laplacian_on_off.png"
END_FIG = "end500_vs_end1000.png"
# The per-case comparison figures go to Results/comparisons/l<size>/.
COMPARISONS_DIR = "comparisons"
# A run folder's figure of the model's training and validation curves.  It is
# not in RUN_FILES: it is drawn only where the model's history file can be
# found (Work/models/<train label>/), by the run that writes the folder or by
# the overview, so a folder without a history keeps exactly its flat set.
TRAINING_FIG = "training_curves.png"
LEFTOVERS = (MANIFEST + ".tmp", BOOKKEEPING + ".tmp", CURVES_TMP, SWEEP_TMP, COMPOSITION_TMP,
             COMPOSITION_FILE + ".tmp")
# The files of the index of all runs (Results/README.md, comparison.png and
# its per-case and per-size forms) and the training figure of a run folder
# are written by whichever predict job finishes, and two may finish together,
# so each job writes them through a temporary of its own, named by host and
# pid: README.md.<host>-<pid>.tmp, comparison.<host>-<pid>.partial.png,
# comparison_<case or t<END>>.<host>-<pid>.partial.png and
# training_curves.<host>-<pid>.partial.png.  A job killed in the middle leaves
# its temporary; one this old is no job's any more.
SHARED_TMP = re.compile(
    re.escape(INDEX_FILE) + r"\.\w+-\d+\.tmp|"
    r"figure_notes\.txt\.\w+-\d+\.tmp|"
    r"(?:r2_vs_model_size|laplacian_on_off|end500_vs_end1000)"
    r"\.\w+-\d+\.partial\.png|"
    + re.escape(os.path.splitext(COMPARISON_FILE)[0])
    + r"(?:_[\w.]+)?\.\w+-\d+\.partial\.png|"
    + re.escape(os.path.splitext(TRAINING_FIG)[0])
    + r"\.\w+-\d+\.partial\.png")
SHARED_TMP_STALE = 3600.0
PUBLISH_JOURNAL = "_publish.json"
PUBLISH_DONE = "_published.json"


def shared_tmp(path):
    """A temporary name beside ``path`` that no other process uses, for a
    file that several jobs write: README.md -> README.md.<host>-<pid>.tmp,
    comparison.png -> comparison.<host>-<pid>.partial.png."""
    host = re.sub(r"[^A-Za-z0-9_]", "_", socket.gethostname())
    who = f"{host}-{os.getpid()}"
    root, ext = os.path.splitext(path)
    if ext == ".png":
        return f"{root}.{who}.partial{ext}"
    return f"{path}.{who}.tmp"


def clear_shared_tmp(folder, older_than=SHARED_TMP_STALE):
    """Remove the shared temporaries in ``folder`` that are over
    ``older_than`` seconds old: a live job's is seconds old, so it is never
    taken from under it."""
    try:
        names = os.listdir(folder)
    except OSError:
        return
    for d in names:
        path = os.path.join(folder, d)
        try:
            if (SHARED_TMP.fullmatch(d)
                    and time.time() - os.path.getmtime(path) > older_than):
                _discard(path)
        except OSError:
            pass


def _discard(path):
    """Remove a file if it is there; a leftover temporary is not an error."""
    try:
        os.remove(path)
    except OSError:
        pass


def write_json(path, obj):
    """Write JSON so that a kill mid-write leaves the old file, not a stub.

    A failure part-way (an unserialisable value, a full disk) leaves the old
    file untouched and removes the temporary, so nothing half-written is left
    in a folder that gets zipped and sent.
    """
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1)
        os.replace(tmp, path)
    except BaseException:
        _discard(tmp)
        raise


def _set_aside(path, what):
    """Move an unreadable file to <name>.corrupt so it is kept, not reused."""
    try:
        os.replace(path, path + ".corrupt")
        where = f"moved aside to {os.path.basename(path)}.corrupt"
    except OSError as exc:
        where = (f"could not move it aside ({type(exc).__name__}), so it "
                 f"will be overwritten")
    print(f"[submission] {path} is unreadable ({what}); {where}", flush=True)


def dropped_key(kind, k):
    """True when the bookkeeping entry ``k`` of ``kind`` is a case of a box
    size in P.DROPPED_SIZES: ("sizes", "160") or ("grid", "+0.20_l160")."""
    try:
        if kind == "sizes":
            return int(k) in P.DROPPED_SIZES
        if kind == "grid":
            return int(str(k).rsplit("_l", 1)[1]) in P.DROPPED_SIZES
    except (IndexError, ValueError):
        pass
    return False


def dropped_label(text):
    """The same for a label of the manifest's "sweep" block: "sizes 160"."""
    kind, _, k = str(text).partition(" ")
    return dropped_key(kind, k)


def without_dropped(manifest):
    """The bookkeeping manifest without any entry of a dropped box size,
    in its "sizes" and "grid" entries and in the lists of its "sweep"
    block.  A manifest that was written before the sizes were dropped still
    holds them; they are ignored, and the next write leaves them out."""
    for kind in KINDS:
        if isinstance(manifest.get(kind), dict):
            manifest[kind] = {k: v for k, v in manifest[kind].items()
                              if not dropped_key(kind, k)}
    sweep = manifest.get("sweep")
    if isinstance(sweep, dict):
        for name in ("cases", "missing_curves", "undocumented_curves",
                     "not_stored"):
            if isinstance(sweep.get(name), list):
                sweep[name] = [x for x in sweep[name]
                               if not dropped_label(x)]
    return manifest


def load_manifest(out) -> dict:
    """The folder's manifest, or a fresh one if it has none or it is damaged.

    A damaged manifest is kept as manifest.json.corrupt rather than silently
    overwritten, and the fresh one says so, since the history it held is not
    in the new one.
    """
    path = os.path.join(out, BOOKKEEPING)
    manifest: dict = {}
    if os.path.exists(path):
        try:
            with open(path) as fh:
                manifest = json.load(fh)
            if not isinstance(manifest, dict):
                raise ValueError("not a JSON object")
        except (OSError, ValueError) as exc:
            _set_aside(path, type(exc).__name__)
            manifest = {"restarted": "the previous manifest was unreadable "
                                     f"and is kept as {BOOKKEEPING}.corrupt"}
    for kind in KINDS:
        if not isinstance(manifest.get(kind), dict):
            manifest[kind] = {}
    return without_dropped(manifest)


def curve_record(case):
    """What sweep_curves.npz keeps for one case."""
    st = case["stats"]
    return {"times": np.asarray(case["times"], dtype=float),
            "curve": np.asarray(case["mean_curve"], dtype=float),
            "n": int(st["n_seeds"]),
            "n_end": int(st.get("n_end", st["n_seeds"])),
            "n_div": int(st.get("n_div", 0)),
            "lattice": int(case["lattice"]),
            **({"sd": np.asarray(case["sd_curve"], dtype=float)}
               if case.get("sd_curve") is not None else {})}


def curve_digest(rec):
    """A short fingerprint of one stored curve.

    Each manifest entry carries the digest of the curve it was drawn with,
    so the sweep can tell a stored curve that belongs to the entry from an
    older one left behind when a save failed -- rather than trusting a flag
    that is itself only as good as the write that set it.
    """
    h = hashlib.md5()
    for part in CURVE_PARTS + CURVE_OPTIONAL:
        if part in rec:
            h.update(part.encode())
            h.update(np.ascontiguousarray(
                np.asarray(rec[part], dtype=float)).tobytes())
    return h.hexdigest()[:16]


def load_curves(out):
    """Every case's mean R2 curve stored in a submission folder.

    A damaged file -- a truncated zip, or an empty one from an interrupted
    save -- is moved aside and treated as absent.  np.load reports those as
    BadZipFile or EOFError, not the OSError or ValueError one would guess, so
    everything is caught: a sidecar that cannot be read must not stop the
    manifest being written.

    The file is opened here rather than by np.load, which on a truncated zip
    raises without closing its handle; on Windows that open handle then
    blocked moving the file aside.
    """
    path = os.path.join(out, CURVES_FILE)
    parts = {}
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh, np.load(fh) as z:
                for name in z.files:
                    kind, k, part = name.split("__")
                    parts.setdefault((kind, k), {})[part] = np.array(z[name])
        except Exception as exc:
            _set_aside(path, type(exc).__name__)
            parts = {}
    return {key: v for key, v in parts.items()
            if set(CURVE_PARTS) <= set(v) and not dropped_key(*key)}


def save_curves(out, curves):
    """Write sweep_curves.npz through a temporary file, like the manifest."""
    arrays = {}
    for (kind, k), v in curves.items():
        for part in CURVE_PARTS + CURVE_OPTIONAL:
            if part in v:
                arrays[f"{kind}__{k}__{part}"] = np.asarray(v[part])
    tmp = os.path.join(out, CURVES_TMP)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, os.path.join(out, CURVES_FILE))
    except BaseException:
        _discard(tmp)
        raise


def curve_case(v):
    """One sweep_curves.npz entry in the shape the figures take."""
    n = int(v["n"])
    return {"lattice": int(v["lattice"]), "times": v["times"],
            "mean_curve": v["curve"],
            "sd_curve": v["sd"] if "sd" in v else None,
            "stats": {"n_seeds": n,
                      "n_end": int(v["n_end"]) if "n_end" in v else n,
                      "n_div": int(v["n_div"]) if "n_div" in v else 0}}


def curve_key(kind, k):
    """(composition, lattice) of a bookkeeping entry, from its kind and key:
    ("compositions", "+0.20") -> (0.2, None), ("sizes", "256") -> (0.0, 256),
    ("grid", "+0.20_l256") -> (0.2, 256)."""
    if kind == "compositions":
        return float(k), None
    if kind == "sizes":
        return float(P.CRITICAL_OFF), int(k)
    off, lat = k.split("_l")
    return float(off), int(lat)


def _size_order(kind, k):
    """Sort key of a bookkeeping entry: box size, then composition."""
    off, lat = curve_key(kind, k)
    return (P.LATTICE if lat is None else lat, off)


def curves_as_cases(curves):
    """sweep_curves.npz entries as {(composition, lattice): case}, the
    lattice being that of the box (the base lattice for a composition case).
    The "sizes" entry at the base lattice is the critical quench again (its
    alias, or the twin on the sweep seeds); it is used only where there is no
    composition case for it."""
    out = {}
    for (kind, k), v in sorted(curves.items(),
                               key=lambda kv: kv[0][0] != "compositions"):
        off, lat = curve_key(kind, k)
        lat = int(v["lattice"]) if lat is None else lat
        if kind == "sizes" and (off, lat) in out:
            continue
        out[(off, lat)] = curve_case(v)
    return out


def data_window(results):
    """The window a cached case was rolled out over, read off its rollouts.

    A rebuild has no config to read it from -- the cache key is a hash -- so
    this is what build() records as the entry's "window", in the same shape
    as fno_predict's run stamp.  "truncated" says whether EARLY_STOP cut
    every surviving seed short: evaluate_seed records t_stop and
    stopped_early whether or not EARLY_STOP is on, but ends the rollout
    there only when it is.  Only such a window may legitimately end before
    a longer one from the same config (folder_times).
    """
    longest = max((r for r in results if len(r["times"])),
                  key=lambda r: float(np.asarray(r["times"], dtype=float)[-1]))
    end = float(np.asarray(longest["times"], dtype=float)[-1])
    slack = grid_slack(longest["times"])
    snaps = sorted({float(s) for r in results for s in r["snapshots"]
                    if s <= end + slack})
    alive = [r for r in results if r.get("diverged_at") is None]
    truncated = bool(alive) and all(
        r.get("stopped_early") and r.get("t_stop") is not None
        and len(r["times"])
        and abs(float(np.asarray(r["times"])[-1]) - float(r["t_stop"]))
        < grid_slack(r["times"])
        for r in alive)
    return {"start": result_start(results[0]), "end": end,
            "snapshots": snaps, "truncated": truncated}


def folder_times(entries):
    """Folder-level snapshot times, start and final time, from its entries.

    One value where every entry agrees, the sorted distinct values where not.
    Whether the entries were rolled out over the same window is judged by
    their windows: a live run's is its config (the stamp's "window"); a
    rebuild's (data_window) and an older entry's are read off the data, so
    they end where the data end.  Two windows agree if their start and
    snapshot times are the same.  A rebuilt window that EARLY_STOP cut short
    (data_window's "truncated") also agrees with a longer one when it has
    that window's start, is a prefix of its snapshot times, and its data end
    before the longer window's next snapshot.  Anything else that is
    shorter -- a live window, a rebuild that simply ran to an earlier
    END_TIME, an older entry -- is a genuinely different window.
    """
    snaps, starts, ends, windows = set(), set(), set(), []
    for e in entries:
        if not isinstance(e, dict):
            continue
        sn = e.get("snapshots") or {}
        if sn.get("times") is not None:
            snaps.add(tuple(float(t) for t in sn["times"]))
        if sn.get("t_start") is not None:
            starts.add(float(sn["t_start"]))
        if sn.get("t_end") is not None:
            ends.add(float(sn["t_end"]))
        run = e.get("run") or {}
        w = run.get("window")
        if w and w.get("start") is not None:
            windows.append((float(w["start"]),
                            tuple(float(t) for t in w["snapshots"]),
                            float(w["end"]) if w.get("end") is not None
                            else None,
                            run.get("source") == "make_submission"
                            and bool(w.get("truncated"))))
        elif sn.get("t_start") is not None and sn.get("times") is not None:
            windows.append((float(sn["t_start"]),
                            tuple(float(t) for t in sn["times"]),
                            float(sn["t_end"]) if sn.get("t_end") is not None
                            else None, False))

    def agree(a, b):
        if a[0] != b[0]:
            return False
        if a[1] == b[1]:
            return True
        short, long_ = (a, b) if len(a[1]) < len(b[1]) else (b, a)
        if not short[3] or len(short[1]) == len(long_[1]):
            return False
        n = len(short[1])
        return (long_[1][:n] == short[1] and short[2] is not None
                and short[2] < long_[1][n])

    def one(vals):
        vals = sorted(vals)
        return vals[0] if len(vals) == 1 else (vals or None)

    times = (list(next(iter(snaps))) if len(snaps) == 1
             else sorted({t for sn in snaps for t in sn}) or None)
    return {"snapshot_times": times,
            "rollout_start": one({w[0] for w in windows} or starts),
            "pdf_time": one(ends),
            "times_agree": all(agree(a, b) for i, a in enumerate(windows)
                               for b in windows[i + 1:])}


def record_cases(out, entries, curve):
    """Merge one case's manifest entries and its sweep curve into a folder.

    fno_predict calls this the moment a case's files are published, so the
    manifest never lags the figures: a run killed three cases in leaves those
    three documented, not new files the manifest knows nothing about.  The
    curve store is written first and the manifest last, each through a
    temporary file, so the manifest is the point at which a case counts.
    """
    os.makedirs(out, exist_ok=True)
    manifest = load_manifest(out)
    curves = load_curves(out)
    for (kind, k), entry in entries.items():
        manifest[kind][k] = entry
        curves[(kind, k)] = curve_record(curve)
    save_curves(out, curves)
    write_json(os.path.join(out, BOOKKEEPING), manifest)
    return manifest


def write_video(results, seeds, off, lattice, path, name=DISPLAY_NAME):
    """One film per case, from frames the rollout already recorded.

    fno_predict stores every VIDEO_EVERY-th step in the result cache, so this
    costs a render and no solving.  Runs made before that feature have no
    frames; figure_video says so and skips rather than failing.

    figure_video titles itself from MODEL_FLAGS, which during a prediction run
    holds the internal label ("fno_ch").  The submission film should carry the
    readable name like every other submission figure, so the label is swapped
    for the render and put back afterwards -- the run's own fno_ch outputs
    are still being written with the original.
    """
    idx = best_index(results)
    saved = dict(P.MODEL_FLAGS)
    P.MODEL_FLAGS.update(label=name, note="")
    try:
        P.figure_video(results[idx], seeds[idx], path, off, lattice)
    finally:
        P.MODEL_FLAGS.clear()
        P.MODEL_FLAGS.update(saved)
    vt = results[idx].get("video_t")
    if results[idx].get("video_true") is None or vt is None or not len(vt):
        return None
    return {"t_first": float(vt[0]), "t_last": float(vt[-1])}


def r2_at(results, t):
    """Every seed's R2 at one snapshot time."""
    return np.array([float(P.r2_score(*r["snapshots"][t]))
                     for r in results if t in r["snapshots"]])


def r2_at_times(results, times=R2_REPORT_TIMES):
    """Each seed's R2 at the reporting ``times`` the rollouts reach, as
    {"2000": {"mean": ..., "n": seeds that reach it, "diverged": seeds
    that had diverged by then}}.

    A seed that diverged is scored at P.R2_FLOOR from then on (a series
    that is not finite is read the same way), and it stays in the mean: the
    mean is over every seed that reaches the time, not over the survivors.
    "mean_alive" (over "n_alive" seeds) is the group protocol's rule instead: the seeds
    that had not diverged, None when none had not.  A seed that stopped
    early is not counted past its stop.  A time no seed reaches is left out.
    """
    out = {}
    for want in times:
        got, alive, diverged = [], [], 0
        for r in results:
            t = np.asarray(r["times"], dtype=float)
            v = np.asarray(r["r2"], dtype=float)
            n = min(t.size, v.size)
            if n == 0:
                continue
            i = grid_index(t[:n], want)
            if i is None:
                continue
            got.append(float(v[i]) if np.isfinite(v[i]) else P.R2_FLOOR)
            div = r.get("diverged_at")
            if div is not None and float(div) <= want + 1e-9:
                diverged += 1
            elif np.isfinite(v[i]):
                alive.append(float(v[i]))
        if got:
            out[format(want, "g")] = {
                "mean": float(np.mean(got)), "n": len(got),
                "diverged": diverged, "n_alive": len(alive),
                "mean_alive": float(np.mean(alive)) if alive else None}
    return out


def group_r2(m, times=R2_REPORT_TIMES):
    """The R2 at the reporting times, pooled and per group, from a
    manifest: {"pooled" | "compositions" | "sizes": {"cases", "seeds",
    "at": {"2000": {"mean", "n", "diverged"}}}}.  A group's mean is over
    every seed of its cases that reach the time.  The critical quench at
    l = 128 is in both groups and counted once in the pooled one (cases are
    told apart by their folder).  None when no case has recorded R2 at a
    reporting time (a manifest from before they were)."""
    groups = {"compositions": [], "sizes": []}
    pooled, seen = [], set()
    for kind in groups:
        for e in _dict(m.get(kind)).values():
            if not isinstance(e, dict) or not isinstance(e.get("seeds"), int):
                continue
            groups[kind].append(e)
            key = e.get("folder") or id(e)
            if key not in seen:
                seen.add(key)
                pooled.append(e)

    def pool(entries):
        at = {}
        for want in times:
            k = format(want, "g")
            rows = [_dict(_dict(e.get("r2_at_times")).get(k))
                    for e in entries]
            rows = [r for r in rows if r.get("n")]
            n = sum(r["n"] for r in rows)
            if n:
                at[k] = {"mean": sum(r["mean"] * r["n"] for r in rows) / n,
                         "n": n, "diverged": sum(r["diverged"] for r in rows)}
        return {"cases": len(entries),
                "seeds": sum(e["seeds"] for e in entries), "at": at}
    out = {"pooled": pool(pooled),
           "compositions": pool(groups["compositions"]),
           "sizes": pool(groups["sizes"])}
    return out if out["pooled"]["at"] else None


def rollout_times(results):
    """(snapshot times, final time, start time), read from the rollouts.

    Read from the data rather than from this module's constants.  A live run
    follows fno_predict's config and a rebuild follows whatever the cache was
    produced with; either can differ from the defaults here.  The times are
    the union over seeds: with EARLY_STOP on, a seed that stopped early lacks
    the later snapshots, and an intersection would have let that one seed
    drop columns from the whole case.  Each figure uses the seeds that have
    what it needs and says how many did not.

    The final time is where the case is judged.
      * If any seed that did not diverge ran the whole window -- always, with
        EARLY_STOP off, unless every seed diverged -- or if every seed
        diverged, it is the window's last snapshot.  A seed that has
        diverged by then is scored there at the floor: a divergence is a
        failure to be shown, not a reason to judge the case earlier, where
        it still looked good.  One that diverged after the last snapshot
        (END_TIME not itself a snapshot time) is reported by the figures as
        having diverged later.
      * Otherwise EARLY_STOP stopped every surviving seed early, and it is
        the latest snapshot any seed reached while still valid (a diverged
        seed up to its divergence) -- but never the input frame, which is a
        copy of the truth with R2 = 1 by construction, unless nothing got
        further; figure_snapshots then says so.
    Snapshots past the end of the window are ignored: rollouts cached before
    evaluate_seed stopped filling them in labelled END_TIME frames as later.
    """
    union = set().union(*(set(r["snapshots"]) for r in results))
    if not union:
        raise ValueError("the rollouts carry no snapshots")
    t_start = result_start(results[0])
    longest = max((r for r in results if len(r["times"])),
                  key=lambda r: float(np.asarray(r["times"], dtype=float)[-1]))
    t_last = float(np.asarray(longest["times"], dtype=float)[-1])
    slack = grid_slack(longest["times"])
    snaps = [s for s in sorted(float(s) for s in union)
             if s <= t_last + slack] or [min(float(s) for s in union)]
    alive = [r for r in results if r.get("diverged_at") is None]
    full = [r for r in alive if len(r["times"])
            and float(np.asarray(r["times"])[-1]) >= t_last - slack]
    if full or not alive:
        t_end = snaps[-1]
    else:
        valid = {float(s) for r in results
                 for s, (_, q) in r["snapshots"].items()
                 if s <= t_last + slack
                 and (r.get("diverged_at") is None or np.all(np.isfinite(q)))}
        t_end = max(valid) if valid else snaps[-1]
    return [s for s in snaps if s <= t_end], t_end, t_start


def snapshot_columns(snaps):
    """The columns of snapshots_<case>.png: the SNAPSHOT_COLUMNS the rollout
    has (a plain-head record may lack t = 100, a short run the late ones) and
    its last time, so a run that ends between them still shows its end.  With
    fewer than two of them, every snapshot time it has."""
    got = [t for t in snaps
           if any(abs(t - c) < 1e-9 for c in SNAPSHOT_COLUMNS)]
    if len(got) < 2:
        return list(snaps)
    return got if got[-1] == snaps[-1] else got + [snaps[-1]]


def figure_snapshots(results, seeds, off, path, name=DISPLAY_NAME,
                     notes=None):
    """Solver, FNO and error at each snapshot time -- no R2 curve, no title.

    One figure with a column for each of SNAPSHOT_COLUMNS (those the rollout
    has).  The first column is the frame the rollout is seeded with when
    that is a snapshot time, so its "prediction" is a verbatim copy of the
    truth and its R2 is exactly 1 by construction.  It is scored like the
    other columns because the first panel is the initial condition and a
    reader expects it to match.

    The grid is one realisation because averaging fields across independent
    initial conditions gives mush rather than a morphology.  The best-scoring
    seed is shown.  A diverged seed has no final R2; it is scored at
    P.R2_FLOOR, the value the rollout gives every step after a divergence.
    If every seed diverged, the one shown says when it did.

    The figure keeps the column headers (t), the row labels and the colour
    bars.  What used to be in the caption goes into ``notes``: the R2 of
    each column, the shown seed and how it was chosen, the median and worst
    R2 at the end, the divergence counts, and the best, median and worst
    seed by mean R2 over the rollout (seed_ranks).
    """
    all_snaps, t_end, t_start = rollout_times(results)
    scores = np.array([float(np.nanmean(r["r2"])) for r in results])
    idx = best_index(results)
    res = results[idx]
    snaps = snapshot_columns([t for t in all_snaps if t in res["snapshots"]])
    ncol = len(snaps)
    lat = res["snapshots"][snaps[0]][0].shape[0]

    def floored(v):
        return np.where(np.isfinite(v), v, P.R2_FLOOR)

    raw = r2_at(results, t_end)
    n_floor = int((~np.isfinite(raw)).sum())
    final = floored(raw)
    shown_t = t_end if t_end in res["snapshots"] else snaps[-1]
    shown_final = float(floored(P.r2_score(*res["snapshots"][shown_t])))
    med, worst = float(np.median(final)), float(np.min(final))
    late = [r["diverged_at"] for r in results
            if r.get("diverged_at") is not None
            and r["diverged_at"] > t_end + 1e-6]
    at_input = t_end <= t_start + 1e-6

    fig, axes = plt.subplots(3, ncol, figsize=(2.9 * ncol + 1.0, 9.0),
                             facecolor="white", layout="constrained")
    axes = np.asarray(axes).reshape(3, ncol)
    rows = (("Solver", 0, PHASE_CMAP, (-1, 1), r"$\psi$"),
            ("FNO", 1, PHASE_CMAP, (-1, 1), r"$\psi$"),
            ("absolute error", 2, ERR_CMAP, (0, 1),
             r"$|\psi_{\rm Solver} - \psi_{\rm FNO}|$"))
    col_r2 = []
    for ri, (row_name, which, cmap, vlim, cbar_label) in enumerate(rows):
        for ci, t in enumerate(snaps):
            true, pred = res["snapshots"][t]
            img = (true, pred, np.abs(true - pred))[which]
            ax = axes[ri, ci]
            im = ax.imshow(img, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                           interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                r2c = P.r2_score(true, pred)
                col_r2.append(f"{t:.0f}: " + (
                    f"{r2c:.3f}" if np.isfinite(r2c) else
                    "diverged" if not np.all(np.isfinite(pred))
                    else "undefined"))
                ax.set_title(f"$t$ = {t:.0f}", fontsize=12, color="black")
            if ri == 1:
                r2c = P.r2_score(true, pred)
                ax.set_xlabel(
                    "$R^2$ = " + (
                        f"{r2c:.3f}" if np.isfinite(r2c) else
                        "diverged" if not np.all(np.isfinite(pred))
                        else "undefined"),
                    fontsize=14, fontweight="bold", color="black", labelpad=4)
            if ci == 0:
                ax.set_ylabel(row_name, fontsize=12, fontweight="bold")
            if ci == ncol - 1:
                cb = fig.colorbar(im, ax=axes[ri, :].tolist(), fraction=0.03,
                                  pad=0.01)
                cb.set_label(cbar_label, fontsize=10)
    fno_paths.save_figure(fig, path, FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")
    if notes is not None:
        ranks = seed_ranks(results, seeds)
        diverged = (f"; diverged at t = {res['diverged_at']:.0f}"
                    if res.get("diverged_at") is not None else "")
        notes[os.path.basename(path)] = [
            "Solver, FNO and absolute error of one seed at each snapshot "
            "time (columns), colour bars for psi and for the error",
            f"case: {comp_label(off)}, psi0 = {off:+.1f}, {lat} x {lat} "
            "lattice sites, dx = 1",
            f"autoregressive rollout seeded at t = {t_start:.0f}"
            + ("; the first column is the seed frame, so its R2 = 1 by "
               "construction" if abs(snaps[0] - t_start) < 1e-6 else ""),
            f"seed shown: {seeds[idx]}, the best of {len(results)} (highest "
            f"mean R2 over the rollout among those that did not diverge)"
            + diverged,
            "R2 of the shown seed at t = " + ", ".join(col_r2),
            f"R2({shown_t:.0f}) of the shown seed = {shown_final:.3f}; over "
            f"the {final.size} seed{'s' if final.size != 1 else ''} that "
            f"reach{'es' if final.size == 1 else ''} t = {t_end:.0f}: "
            f"median {med:.3f}, worst {worst:.3f}",
            f"seeds diverged at t = {t_end:.0f}: {n_floor} (scored at R2 = "
            f"{P.R2_FLOOR:g}); diverged later: {len(late)}"
            + (f" (by t = {max(late):.0f})" if late else "")
            + (f"; {len(results) - final.size} seed"
               f"{'s' if len(results) - final.size != 1 else ''} stopped "
               "before it" if final.size < len(results) else ""),
            rank_line(ranks), f"model: {name}"]
        if at_input:
            notes[os.path.basename(path)].append(
                "no seed reached a snapshot after the input frame")
    return {"seed": int(seeds[idx]), "mean_r2": float(scores[idx]),
            "r2_at_end_shown": shown_final,
            "r2_at_end_median": med,
            "r2_at_end_worst": worst,
            "r2_at_end_diverged": n_floor,
            "r2_at_end_diverged_later": len(late),
            "at_input_frame": bool(at_input),
            "times": [float(t) for t in snaps],
            "t_start": t_start, "t_end": t_end}


def pdf_ylim(*hists):
    """The y-range of a P(psi) plot, from what is drawn: half the smallest
    non-zero density to twice the largest, so both peaks and the thinnest
    tail are in view whatever the horizon.  PDF_YLIM when nothing is."""
    pos = [h[h > 0] for h in hists]
    pos = np.concatenate(pos) if pos else np.empty(0)
    if not pos.size:
        return PDF_YLIM
    return 0.5 * float(pos.min()), 2.0 * float(pos.max())


def figure_psi_pdf(results, off, path, lattice=BASE_LATTICE, data=None,
                   notes=None):
    """P(psi) at the final snapshot, pooled over every seed.

    The y-range follows the data (pdf_ylim).  The figure has the two
    histograms, the axis labels and a short legend; the lever-rule fractions
    for the mean composition beside the measured ones, the seed counts and
    the other facts go into ``notes``.

    The double-well potential puts the equilibrium phases near psi = +-1
    whatever the composition, so the peaks barely move and it is their weights
    that change: the lever rule puts the +1-phase fraction at (1 + psi0) / 2.

    The measured fraction of sites with psi > 0 departs from that by a couple
    of percent, and the sign of the departure follows the sign of psi0 -- the
    majority phase always takes slightly more area than the ideal lever rule
    allows, because the lever rule assumes the phases sit at exactly +-1 while
    curvature and the finite interface width shift them.

    A diverged seed's prediction is NaN from the divergence on.  It is left
    out of the prediction histogram, and the notes say how many were, rather
    than letting one bad seed empty the figure.  A seed that stopped before
    the final time (EARLY_STOP) has neither a truth nor a prediction frame
    there, so both histograms pool the seeds that reached it, and the notes
    say how many of all the seeds that is.
    """
    _, t_end, t_start = rollout_times(results)
    have = [r for r in results if t_end in r["snapshots"]]
    late = [r["diverged_at"] for r in results
            if r.get("diverged_at") is not None
            and r["diverged_at"] > t_end + 1e-6]
    at_input = t_end <= t_start + 1e-6
    true = np.concatenate([r["snapshots"][t_end][0].ravel() for r in have])
    per_seed = [r["snapshots"][t_end][1].ravel() for r in have]
    short = len(results) - len(have)
    diverged = sum(1 for q in per_seed if not np.all(np.isfinite(q)))
    pred = np.concatenate(per_seed)
    pred = pred[np.isfinite(pred)]
    bins = np.linspace(PDF_RANGE[0], PDF_RANGE[1], PDF_BINS + 1)
    centres = 0.5 * (bins[1:] + bins[:-1])
    h_true, _ = np.histogram(true, bins=bins, density=True)
    h_pred, _ = np.histogram(pred, bins=bins, density=True)
    ylim = pdf_ylim(h_true, h_pred)
    clipped = int(((true < bins[0]) | (true > bins[-1])).sum()
                  + ((pred < bins[0]) | (pred > bins[-1])).sum())

    f_true = float((true > 0).mean())
    f_pred = float((pred > 0).mean()) if pred.size else float("nan")
    lever = 0.5 * (1.0 + off)
    shift = f_true - lever
    pct = {"lever": 100.0 * lever, "true": 100.0 * f_true,
           "pred": 100.0 * f_pred}

    fig, ax = plt.subplots(figsize=(7.4, 5.2), facecolor="white")
    ax.set_yscale("log")
    ax.stairs(h_true, bins, fill=True, color=C_TRUTH, alpha=0.35,
              label="Solver")
    ax.stairs(h_true, bins, color=C_TRUTH, lw=1.8)
    ax.plot(centres, h_pred, color=C_PRED, lw=1.5, marker="o", ms=5.5,
            mfc="none", mew=1.3, label="FNO")

    for x in (-1.0, 1.0):
        ax.axvline(x, color="0.6", ls=":", lw=1.1, zorder=0)

    ax.set_xlim(*PDF_RANGE)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\psi$", fontsize=13)
    ax.set_ylabel(r"$P(\psi)$", fontsize=13)
    ax.grid(alpha=0.25, ls=":", which="both")
    ax.legend(loc="upper center", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fno_paths.save_figure(fig, path, FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")
    if notes is not None:
        notes[os.path.basename(path)] = [
            f"distribution P(psi) of the order parameter at t = {t_end:.0f}, "
            "Solver (filled) and FNO (circles), log y axis; dotted lines at "
            "psi = -1 and +1",
            f"case: psi0 = {off:+.1f}, l = {lattice}",
            f"{len(have)} of {len(results)} seeds pooled "
            f"({true.size:,} sites)"
            + (f"; {short} stopped earlier" if short else ""),
            f"lever rule at psi0 = {off:+.1f}: {pct['lever']:.1f} % A-rich "
            f"/ {100 - pct['lever']:.1f} % B-rich (A-rich: psi > 0)",
            f"measured: Solver {pct['true']:.1f} / {100 - pct['true']:.1f} "
            f"%, FNO {pct['pred']:.1f} / {100 - pct['pred']:.1f} %"]
        if diverged:
            notes[os.path.basename(path)].append(
                f"the FNO histogram excludes {diverged} diverged seed"
                f"{'s' if diverged != 1 else ''}")
        if not pred.size:
            notes[os.path.basename(path)].append(
                "no finite prediction to plot")
        if late:
            notes[os.path.basename(path)].append(
                f"{len(late)} seed{'s' if len(late) != 1 else ''} diverged "
                f"later, by t = {max(late):.0f}")
        if at_input:
            notes[os.path.basename(path)].append(
                "no seed reached a snapshot after the input frame "
                "(prediction = truth by construction)")
    if data is not None:
        data["psi_distribution"] = (centres, h_true, h_pred,
                                    (pct["lever"], pct["true"], pct["pred"]))
    return {"n_sites": int(true.size), "frac_pos_true": f_true,
            "frac_pos_pred": f_pred, "lever_rule": lever,
            "lever_shift": shift, "clipped_points": clipped,
            "pred_seeds_excluded": diverged,
            "seeds_not_reaching_t_end": short, "t": t_end,
            "pred_seeds_diverged_later": len(late),
            "at_input_frame": bool(at_input),
            "peak_density_true": float(h_true.max()),
            "peak_density_pred": float(h_pred.max()),
            "y_range": [float(ylim[0]), float(ylim[1])],
            "lever_shift_pred": f_pred - lever,
            "lever_A_pct": pct["lever"], "lever_B_pct": 100 - pct["lever"],
            "A_pct_true": pct["true"], "B_pct_true": 100 - pct["true"],
            "A_pct_pred": pct["pred"], "B_pct_pred": 100 - pct["pred"]}


def _exponent_line(label, w):
    """"label, t = a-b: n = x Solver, y FNO" for an exponent window of
    fno_predict.growth_windows."""
    return (f"{label}, t = {w['window'][0]:.0f}-{w['window'][1]:.0f}: n = "
            f"{w['n_truth']:.3f} Solver, {w['n_fno']:.3f} FNO")


def figure_growth(results, off, path, lattice=BASE_LATTICE, data=None,
                  notes=None):
    """L_half(t) against the Lifshitz-Slyozov law.

    Conserved-order-parameter phase separation coarsens as L ~ t^(1/3) once the
    domains are large compared with the interface.  The reference line has no
    free exponent, only a prefactor, and that prefactor is anchored to the
    ground-truth curve at the geometric-mean time of the late half -- so the
    line tests the slope of the data, not its magnitude.  The quoted n is a
    straight power-law fit over the whole rollout; the window is stated
    because the early part of that window is still in the crossover, which is
    why the off-critical exponents come out below 1/3.

    A diverged seed stops being measured at the divergence, so its length
    series is shorter than the others.  They are padded with NaN and averaged
    with nanmean, which keeps the mean over the seeds still running at each
    time rather than failing on arrays of unequal length.

    Marked on the time axis: the training horizon (a dashed line, when the
    model's is known) and the finite-size regime (shaded from the first time
    the mean L is past l / P.FINITE_SIZE).  The figure draws the
    half-maximum length L_half (the first zero of C(r, t) when the rollouts
    have no L_half) for the Solver and the FNO, mean +-1 sd over the seeds,
    and the 1/3 guide.  The zero-crossing length, the exponents and the
    fit windows go into ``notes``.
    """
    t, Lt, Lp = P.length_stacks(results)
    if t.size == 0 or not np.isfinite(Lt).any():
        raise ValueError("no seed has a single length measurement to plot")
    full = max(float(np.asarray(r["times"], dtype=float)[-1])
               for r in results if len(r["times"]))
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0

    def ended_early(r):
        d = np.asarray(r["diag_times"], dtype=float)
        return (r.get("diverged_at") is not None or d.size == 0
                or float(d[-1]) < full - 1.5 * dt)

    ended = [r for r in results if ended_early(r)]
    short = len(ended)
    why = ("diverged seed" if all(r.get("diverged_at") is not None
                                  for r in ended) else "seed")
    mt, mp = P.seed_mean(Lt), P.seed_mean(Lp)

    n_true = P.fit_growth_exponent(t, mt)
    n_pred = P.fit_growth_exponent(t, mp)
    rec = P.length_summary(results, lattice)
    horizon, onset = rec["train_horizon"], rec["finite_size"]["onset"]
    half = rec["L_half"]

    # Past t ~ 5000 at l = 128 the domains are a sizeable fraction of the box
    # and L(t) bends away from any power law.  Once the mean L exceeds l / 8,
    # the exponent is also fitted over the times before that, where finite
    # size does not yet matter, and the Lifshitz-Slyozov line is anchored
    # there.  Below l / 8 -- every case out to t = 2000 -- nothing changes.
    over = np.nonzero(np.nan_to_num(mt, nan=0.0)
                      > lattice / P.FINITE_SIZE)[0]
    cut = int(over[0]) if over.size else t.size
    small = None
    if over.size and cut >= 3:
        small = {"until": float(t[cut - 1]),
                 "n_truth": P.fit_growth_exponent(t[:cut], mt[:cut]),
                 "n_fno": P.fit_growth_exponent(t[:cut], mp[:cut])}

    # The main curve is the half-maximum length L_1/2 (Puri's Fig. 1.7, as
    # the group's domain_growth), with +-1 sd bands over the seeds; the first zero
    # of C(r, t) is in the kinetics figure.  A rollout without L_1/2 draws
    # the zero-crossing length as the main curve, as before.
    half_curves = None
    if half is not None:
        half_curves = P.length_means(results, "L_half")
        _, hst, hsp = P.length_stacks(results, "L_half")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sd_t, sd_p = np.nanstd(hst, axis=0), np.nanstd(hsp, axis=0)
        main_t, main_p = half_curves[1], half_curves[2]
        sym = r"L_{1/2}"
    else:
        main_t, main_p = mt, mp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sd_t, sd_p = np.nanstd(Lt, axis=0), np.nanstd(Lp, axis=0)
        sym = "L"
    late_lo = float(t[-1]) / 4.0
    n_true_m = P.fit_growth_exponent(t, main_t, late_lo, float(t[-1]))
    n_pred_m = P.fit_growth_exponent(t, main_p, late_lo, float(t[-1]))
    win = P.growth_windows(t, main_t, main_p, horizon)["exponent_train"]

    long_span = t[-1] / max(t[0], 1e-9) > LONG_SPAN + 1e-9
    ev = max(1, len(t) // 14)
    fig, ax = plt.subplots(figsize=(7.4, 5.2), facecolor="white")
    for mean, sd, col in ((main_p, sd_p, C_PRED), (main_t, sd_t, C_TRUTH)):
        lo = np.maximum(mean - sd, 1e-9)
        ax.fill_between(t, lo, mean + sd, color=col, alpha=0.13, lw=0)
    ax.loglog(t, main_p, color=C_PRED, lw=1.8, marker="o", ms=5.5,
              markevery=0.07 if long_span else ev, label="FNO")
    ax.loglog(t, main_t, color=C_TRUTH, lw=1.5, ls="--", marker="o", ms=6.5,
              mfc="none", mew=1.4,
              markevery=(0.035, 0.07) if long_span else (ev // 2, ev),
              label="Solver")

    seg = (t[(t >= 0.45 * small["until"]) & (t <= small["until"])]
           if small is not None else t[t >= 0.45 * t[-1]])
    t0, t1 = float(seg[0]), float(seg[-1])
    base = float(np.interp(t0, t, main_t)) * 1.22
    ax.loglog([t0, t1], [base, base * (t1 / t0) ** (1.0 / 3.0)],
              color="k", ls="--", lw=2.0, zorder=5)
    ax.annotate("1/3", xy=(t1, base * (t1 / t0) ** (1.0 / 3.0)),
                xytext=(7, -3), textcoords="offset points", fontsize=14,
                fontweight="bold")

    # the regime is measured with the first zero of C(r, t), also when the
    # axis is L_1/2: the legend says so, or the band seems to start too early
    P.mark_time(ax, t[0], t[-1], horizon, onset, short=True,
                symbol=r"L_{\rm zero}" if half is not None else "L")

    ax.set_xlabel(r"$t$", fontsize=13)
    ax.set_ylabel(rf"${sym}(t)$", fontsize=13)
    # plain numbers on the length axis: 3, 4, 6, 10, not 3 x 10^0
    ax.yaxis.set_major_formatter(LogFormatter())
    ax.yaxis.set_minor_formatter(LogFormatter())
    ax.grid(alpha=0.25, which="both", ls=":")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    fig.tight_layout()
    fno_paths.save_figure(fig, path, FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")
    if notes is not None:
        zero = "L_zero" if half is not None else "L"
        zero_at = []
        for want in (100.0, 1000.0, float(t[-1])):
            k = grid_index(t, want)
            if k is not None:
                zero_at.append(f"t = {t[k]:.0f}: {mt[k]:.2f} Solver, "
                               f"{mp[k]:.2f} FNO")
        lines = [
            ("domain size against time on log-log axes: "
             + ("the half-maximum length L_half of C(r,t)"
                if half is not None else
                "the first zero of C(r,t), L (no L_half in these records)")
             + f", Solver and FNO, mean +- 1 sd over {len(results)} seed"
             f"{'s' if len(results) != 1 else ''}; the dashed black line is "
             "t^(1/3) (Lifshitz-Slyozov), anchored to the Solver curve"),
            f"case: psi0 = {off:+.1f}, l = {lattice}",
            f"late exponent, t = {late_lo:.0f}-{t[-1]:.0f}: n = "
            f"{n_true_m:.3f} Solver, {n_pred_m:.3f} FNO"]
        if small is not None:
            lines.append(
                f"while {zero} <= l/{P.FINITE_SIZE:g} "
                f"(t <= {small['until']:.0f})"
                f": n = {small['n_truth']:.3f} Solver, "
                f"{small['n_fno']:.3f} FNO (the 1/3 line is anchored there)")
        for label, w in (("training window", win),
                         ("beyond training", rec["exponent_extrap"]
                          if half is None else None)):
            if w is not None:
                lines.append(_exponent_line(label, w))
        if horizon is not None:
            lines.append(f"dashed vertical line: training horizon t = "
                         f"{horizon:g}")
        if onset is not None:
            lines.append(f"grey region: {zero} > l/{P.FINITE_SIZE:g} from "
                         f"t = {onset:g}")
        if half is not None:
            lines += [
                "first zero of C(r,t), L_zero (drawn in the kinetics figure): "
                f"n = {n_true:.3f} Solver, {n_pred:.3f} FNO over t = "
                f"{t[0]:.0f}-{t[-1]:.0f}",
                "L_zero, seed mean: " + "; ".join(zero_at)]
        if short:
            lines.append(f"{short} {why}{'s' if short != 1 else ''} end"
                         f"{'s' if short == 1 else ''} early")
        notes[os.path.basename(path)] = lines
    if data is not None:
        data["domain_growth"] = {
            "t": t, "L_true": mt, "L_pred": mp,
            "n_true": np.sum(np.isfinite(Lt), axis=0),
            "n_pred": np.sum(np.isfinite(Lp), axis=0),
            "half": None if half_curves is None else half_curves[1:],
            "horizon": horizon,
            "train": rec["exponent_train"], "extrap": rec["exponent_extrap"]}
    out = {"main": "L_half" if half is not None else "L_zero",
           "n_late_truth": n_true_m, "n_late_fno": n_pred_m,
           "late_window": [late_lo, float(t[-1])],
           "n_truth": n_true, "n_fno": n_pred, "fit_window": [float(t[0]),
                                                              float(t[-1])],
           "L_start": float(mt[0]), "L_end": float(mt[-1]),
           "L_end_fno": float(mp[-1]), "seeds_ending_early": short}
    out.update({k: rec[k] for k in ("train_horizon", "exponent_train",
                                    "exponent_extrap", "finite_size",
                                    "L_half")})
    if small is not None:
        # only when L outgrew l/8, so manifests of today's runs are unchanged
        out["small_L"] = small
    return out


def _sweep_panel(ax, curves, labels, counts, title, ends=None, log=False):
    """Curves carry no seed counts; the title carries them.

    The legend should say which curve is which and nothing else.  But the
    count cannot simply be dropped: one case can be short -- l = 256 in the
    t = 500 run once stopped at 44 -- and a bare "n = 50 seeds" over a 44-seed
    curve is a false caption.  So a uniform count is stated once, and any case
    that departs from it is named.
    """
    marks = 0.07 if log else max(1, len(curves[0][0]) // 14)
    if log:
        ax.set_xscale("log")
    counts = list(counts)
    for i, ((tt, cc), lab) in enumerate(zip(curves, labels)):
        ax.plot(tt, cc, color=SWEEP_COLOURS[i % len(SWEEP_COLOURS)],
                marker=SWEEP_MARKS[i % len(SWEEP_MARKS)], markevery=marks,
                ms=5, lw=1.9, label=lab)
    ax.set_ylabel(r"$R^2$ vs. Solver", fontsize=11)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(loc="lower left", fontsize=9.5)

    common = max(set(counts), key=counts.count)
    odd = [f"{n} at {lab}" for lab, n in zip(labels, counts) if n != common]
    note = rf"   ($n$ = {common} seeds"
    note += ("; " + ", ".join(odd) + ")") if odd else ")"
    fewer = [f"{e} of {n} at {lab}"
             + (f", {dv} of them diverged" if dv else "")
             for lab, n, (e, dv)
             in zip(labels, counts, ends or [(None, 0)] * len(counts))
             if e is not None and e < n]
    if fewer:
        note += "\nstill running at the end: " + ", ".join(fewer)
    ax.set_title(title + note, fontsize=11, fontweight="bold")


def figure_sweeps(by_off, by_size, path, name=DISPLAY_NAME, start=None):
    """R2 against time, per composition and per box size.

    Written here rather than reused from fno_predict for two reasons: that
    version prints one seed count for every curve, which is false when a case
    is short, and it labels the box size L, which collides with L(t) for the
    domain size in the growth figures.  Here each curve carries its own count
    and the box is l.  The composition panel's lattice and the start time come
    from the curves themselves, not from this module's defaults.
    """
    every = list(by_off.values()) + list(by_size.values())
    if not every:
        return
    if start is None:
        def begins(c):
            tt = np.asarray(c["times"], dtype=float)
            return float(tt[0] - (tt[1] - tt[0])) if tt.size > 1 else 0.0
        starts = sorted({begins(c) for c in every})
        start_for_span = starts[0]
        when = (f"$t = {starts[0]:.0f}$" if len(starts) == 1 else
                "$t = " + ", ".join(f"{b:.0f}" for b in starts)
                + "$ (the curves differ)")
    else:
        when = f"$t = {start:.0f}$"
        start_for_span = float(start)
    log = any(float(np.asarray(c["times"], dtype=float)[-1])
              / max(start_for_span, 1e-9) > LONG_SPAN + 1e-9 for c in every)
    panels = []
    if len(by_off) > 1:
        offs = sorted(by_off)
        lats = [int(by_off[o].get("lattice") or BASE_LATTICE) for o in offs]
        if len(set(lats)) == 1:
            labels = [rf"$\bar\psi_0 = {o:+.1f}$" for o in offs]
            title = (f"(a)  different mixture compositions, "
                     f"{lats[0]} x {lats[0]} sites")
        else:
            labels = [rf"$\bar\psi_0 = {o:+.1f}$, $l = {la}$"
                      for o, la in zip(offs, lats)]
            title = "(a)  different mixture compositions (lattices differ)"
        panels.append((
            [(by_off[o]["times"], by_off[o]["mean_curve"]) for o in offs],
            labels, [by_off[o]["stats"]["n_seeds"] for o in offs], title,
            [(by_off[o]["stats"].get("n_end"),
              by_off[o]["stats"].get("n_div", 0)) for o in offs]))
    if len(by_size) > 1:
        lats = sorted(by_size)
        panels.append((
            [(by_size[l]["times"], by_size[l]["mean_curve"]) for l in lats],
            [rf"$l = {l}$" for l in lats],
            [by_size[l]["stats"]["n_seeds"] for l in lats],
            "(b)  different box sizes $l$, critical quench",
            [(by_size[l]["stats"].get("n_end"),
              by_size[l]["stats"].get("n_div", 0)) for l in lats]))
    if not panels:
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 4.1 * len(panels)),
                             sharex=True, facecolor="white")
    axes = np.atleast_1d(axes)
    for ax, (curves, labels, counts, title, ends) in zip(axes, panels):
        _sweep_panel(ax, curves, labels, counts, title, ends=ends, log=log)
    axes[-1].set_xlabel(r"time  $t$", fontsize=11)
    fig.suptitle(f"{name} - rollout accuracy against time\n"
                 f"mean over unseen seeds, autoregressive from {when}",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    root, ext = os.path.splitext(path)
    tmp = f"{root}.partial{ext}"
    try:
        # sweeps.png -> sweeps.partial.png, which recover_interrupted clears
        fno_paths.save_figure(fig, tmp, FIG_DPI, bbox_inches="tight")
        os.replace(tmp, path)
    finally:
        plt.close(fig)
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"[saved] {path}")


def mean_curve(results):
    n_max = max(len(r["r2"]) for r in results)
    stack = np.full((len(results), n_max), np.nan)
    for i, r in enumerate(results):
        stack[i, :len(r["r2"])] = r["r2"]
    longest = results[int(np.argmax([len(r["r2"]) for r in results]))]
    return longest["times"], np.nanmean(stack, axis=0), float(np.nanmean(stack))


def mean_sd_curve(results):
    """(times, mean, sd, overall mean) of R2 over the seeds still running at
    each time.  The seeds share one schedule (a seed that stopped or
    diverged keeps a prefix of it), so the series are aligned by position,
    not by a step: the grid may be a fixed one or any schedule.  A series
    that is not finite counts as P.R2_FLOOR, as everywhere else."""
    n_max = max(len(r["r2"]) for r in results)
    stack = np.full((len(results), n_max), np.nan)
    for i, r in enumerate(results):
        v = np.asarray(r["r2"], dtype=float)
        stack[i, :len(v)] = np.where(np.isfinite(v), v, P.R2_FLOOR)
    longest = results[int(np.argmax([len(r["r2"]) for r in results]))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = (np.nanstd(stack, axis=0))
        return (np.asarray(longest["times"], dtype=float)[:n_max],
                np.nanmean(stack, axis=0), np.nan_to_num(sd),
                float(np.nanmean(stack)))


def composition_series(results):
    """The seed-mean order parameter of the truth and the FNO, as a mixture.

    c_A = (1 + psi) / 2 and c_B = 100 - c_A, in percent (P.psi_to_a).  The mean
    psi of a Cahn-Hilliard rollout is conserved, so the truth stays on its
    starting composition and the FNO should; how far it leaves it is what is
    measured.  The FNO's mean is over the seeds still running at each time
    (n_seeds says how many).  The series is thinned to about
    COMPOSITION_POINTS log-spaced times plus every snapshot time, so a long
    horizon costs no more rows than a short one; the drift rate is fitted to
    all of it.  None when a rollout has no means (cached before they were
    kept).
    """
    if any(r.get("mean_true") is None or r.get("mean_pred") is None
           for r in results):
        return None
    t, st = P._r2_stack(results, "mean_true")
    _, sp = P._r2_stack(results, "mean_pred")
    mt, mp = P.seed_mean(st), P.seed_mean(sp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sdt, sdp = np.nanstd(st, axis=0), np.nanstd(sp, axis=0)
    keep = {0, t.size - 1}
    keep.update(int(round(x)) - 1
                for x in np.geomspace(1, t.size, COMPOSITION_POINTS))
    for s in rollout_times(results)[0]:
        k = grid_index(t, s)
        if k is not None:
            keep.add(k)
    at = np.array(sorted(keep))
    ok = np.isfinite(mp)
    slope = (float(np.polyfit(t[ok] - t[ok][0], mp[ok], 1)[0])
             if ok.sum() > 1 else float("nan"))
    return {"t": t[at], "psi_true": mt[at], "psi_pred": mp[at],
            "A_true": P.psi_to_a(mt[at]), "A_pred": P.psi_to_a(mp[at]),
            "drift_pred": mp[at] - mt[0],
            "n_seeds": np.sum(np.isfinite(sp), axis=0)[at],
            "sd_true": sdt[at], "sd_pred": sdp[at],
            "n_full": np.sum(np.isfinite(sp), axis=0),
            "full": (t, mt, mp), "slope": slope}


def _num(x):
    """A float for JSON: None for what is not finite."""
    x = float(x)
    return x if np.isfinite(x) else None


def composition_entry(comp, off, lattice, n_seeds):
    """A case's composition in the manifest: the start, the end, the shift of
    the FNO's A fraction from the starting one in percentage points (pp), the
    largest |psi - psi0| of the FNO, and the drift rate of its seed-mean psi,
    per unit t and in pp per 1000 t."""
    t, mt, mp = comp["full"]
    alive = np.nonzero(np.isfinite(mp))[0]
    k = int(alive[-1]) if alive.size else None

    def ab(psi):
        a = float(P.psi_to_a(psi))
        return _num(a), _num(100.0 - a)

    a0t, b0t = ab(mt[0])
    a0p, b0p = ab(mp[0])
    a1t, b1t = ab(mt[-1])
    a1p, b1p = ab(mp[k]) if k is not None else (None, None)
    dev = np.abs(mp - mt[0])
    slope = comp["slope"]
    return {
        "off": float(off), "lattice": int(lattice), "n_seeds": int(n_seeds),
        "psi0": _num(mt[0]),
        "start": {"t": float(t[0]), "A_pct_truth": a0t, "B_pct_truth": b0t,
                  "A_pct_pred": a0p, "B_pct_pred": b0p},
        "end": {"t_truth": float(t[-1]),
                "t_pred": None if k is None else float(t[k]),
                "A_pct_truth": a1t, "B_pct_truth": b1t,
                "A_pct_pred": a1p, "B_pct_pred": b1p},
        "shift_pp": (None if a1p is None or a0t is None else a1p - a0t),
        "max_abs_drift_pred": (_num(np.nanmax(dev)) if np.isfinite(dev).any()
                               else None),
        "drift_rate_psi_per_t": _num(slope),
        "drift_rate_pp_per_1000t": _num(5e4 * slope)}


def figure_composition_all(series, path, name=DISPLAY_NAME, seeds=None,
                           notes=None):
    """c_A (%) against time of every composition on one axis, the Solver
    solid and the FNO dashed.  Conservation makes every line flat, at 100
    times (1 + psi0) / 2, so this is a one-glance certificate of it.
    ``series`` is [(psi0, t, A_truth, A_pred)]; the training horizon is
    marked.  The legend has a colour for each composition and a line style
    for each source.  The c_A values at the start and the end go into
    ``notes``."""
    fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor="white")
    for i, (psi0, t, a_true, a_pred) in enumerate(series):
        colour = SWEEP_COLOURS[i % len(SWEEP_COLOURS)]
        ax.plot(t, a_true, color=colour, lw=2.2,
                label=rf"$\bar\psi_0 = {psi0:+.1f}$")
        ax.plot(t, a_pred, color=colour, lw=1.8, ls="--")
    lo = min(float(np.min(t)) for _, t, _, _ in series)
    hi = max(float(np.max(t)) for _, t, _, _ in series)
    P.mark_time(ax, lo, hi, P.train_horizon(), short=True)
    ax.set_xscale("log")
    ax.set_xlabel(r"$t$  (log)", fontsize=12)
    ax.set_ylabel(r"material A content  $c_A$  (%)", fontsize=12)
    ax.grid(alpha=0.3, which="both", ls=":")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [
                  Line2D([], [], color=INK, lw=2.2, label="Solver"),
                  Line2D([], [], color=INK, lw=1.8, ls="--", label="FNO")],
              fontsize=9, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    tmp = os.path.join(os.path.dirname(path), COMPOSITION_TMP)
    try:
        fno_paths.save_figure(fig, tmp, FIG_DPI, bbox_inches="tight")
        os.replace(tmp, path)
    finally:
        plt.close(fig)
        _discard(tmp)
    print(f"[saved] {path}")
    if notes is not None:
        lines = ["material A content c_A = 100 (1 + psi)/2 (%) against time "
                 "on a log t axis, Solver solid and FNO dashed, one colour "
                 "per composition; conservation makes each line flat at "
                 "100 (1 + psi0)/2",
                 "seed mean" if seeds is None else f"mean over {seeds} seeds"]
        for psi0, t, a_true, a_pred in series:
            ok = np.nonzero(np.isfinite(a_pred))[0]
            end = (f"FNO {a_pred[ok[-1]]:.2f} % at t = {t[ok[-1]]:.0f}"
                   if ok.size else "FNO none")
            lines.append(f"psi0 = {psi0:+.1f}: c_A = {a_true[0]:.2f} % at "
                         f"t = {t[0]:.0f}; at the end Solver "
                         f"{a_true[-1]:.2f} % at t = {t[-1]:.0f}, {end}")
        notes[os.path.basename(path)] = lines


# What a case's folder in extras/ holds from write_case (the figures are in
# the run folder, as FIG_KINDS and the case token say).
CASE_FILES = ("rollout.mp4", "rollout.gif")
CASE_DATA = ("psi_distribution.csv", "domain_growth.csv", COMPOSITION_FILE)
STAGE = ".partial_case"


def stage_name(token):
    """The scratch folder a case draws into inside the run folder."""
    return f".partial_{token}"


def _revert(hold, out):
    """Undo a publish from its journal.  True if everything went back.

    The journal lists the files being published and which of them already
    existed.  An existing one is put back from the holding folder if it got
    there; a new one is removed.  A name that existed but was never moved is
    still the original, so it is left alone.
    """
    try:
        with open(os.path.join(hold, PUBLISH_JOURNAL)) as fh:
            journal = json.load(fh)
    except (OSError, ValueError):
        return True
    ok = True
    existing = set(journal["existing"])
    for n in journal["names"]:
        if n not in existing:
            try:
                os.remove(os.path.join(out, n))
            except FileNotFoundError:
                pass
            except OSError:
                ok = False
    for n in journal["existing"]:
        held = os.path.join(hold, n)
        if os.path.exists(held):
            try:
                os.replace(held, os.path.join(out, n))
            except OSError:
                ok = False
    return ok


def _remove_dir(d):
    """Remove a folder; if something in it will not go, move it aside.

    On NFS a file still open elsewhere leaves a .nfsXXXX entry that rmtree
    cannot delete.  A holding folder left like that would make the next
    publish of the case fail, so it is renamed out of the way for a later
    recover_interrupted to remove.
    """
    shutil.rmtree(d, ignore_errors=True)
    if not os.path.exists(d):
        return
    stale = f"{d}.stale{os.getpid()}_{time.time_ns()}"
    try:
        os.replace(d, stale)
    except OSError:
        left = sorted(os.listdir(d)) if os.path.isdir(d) else []
        raise RuntimeError(
            f"cannot remove {d} (it still holds {', '.join(left) or 'files'});"
            f" close any program that has those files open and rerun")
    print(f"[submission] could not delete {os.path.basename(d)} yet; moved it "
          f"to {os.path.basename(stale)} for a later run to remove", flush=True)


def _settle(hold):
    """Mark a reverted publish as settled, so it is never undone twice."""
    try:
        os.replace(os.path.join(hold, PUBLISH_JOURNAL),
                   os.path.join(hold, PUBLISH_SETTLED))
    except OSError:
        pass


def _clear_hold(hold):
    """Remove a settled hold; if it will not go, say so and carry on.

    Its files are either back in place or superseded, so nothing depends on
    removing it now.  A hold left behind blocks only a later publish of its
    own case, which then says why (_publish), not every other case.
    """
    try:
        _remove_dir(hold)
    except RuntimeError as exc:
        print(f"[submission] WARNING: {exc}", flush=True)


def _recover(hold, out):
    """Finish or undo a publish whose run was killed part-way through.

    Undone only if its journal was never marked done or settled: with no
    journal nothing had moved yet.  Only a failed undo raises -- the
    originals are then still in the hold and must not be thrown away.
    """
    if not os.path.isdir(hold):
        return
    if os.path.exists(os.path.join(hold, PUBLISH_JOURNAL)):
        print(f"[submission] undoing an interrupted publish in {out} "
              f"({os.path.basename(hold)})", flush=True)
        if not _revert(hold, out):
            raise RuntimeError(
                f"could not undo the interrupted publish held in {hold}; "
                f"close any program that has those files open and rerun")
        _settle(hold)
    _clear_hold(hold)


def recover_run(root, extras=None):
    """recover_interrupted for a run folder and, when given, the extras
    folder with the case folders in it.

    Each case publishes into the run folder (its scratch folder is
    .partial_<token>) and into its own folder in extras/ (.partial_case), so a
    run killed mid-publish leaves its hold in one of them; this settles all
    of them before a run writes.
    """
    if os.path.isdir(root):
        recover_interrupted(root)
        for d in sorted(os.listdir(root)):
            if ((re.fullmatch(r"l\d+", d) or d in SWEEP_DIRS)
                    and os.path.isdir(os.path.join(root, d))):
                recover_interrupted(os.path.join(root, d))
    if extras and os.path.isdir(extras):
        recover_interrupted(extras)
        for d in sorted(os.listdir(extras)):
            sub_dir = os.path.join(extras, d)
            if not d.startswith(".") and os.path.isdir(sub_dir):
                recover_interrupted(sub_dir)


def recover_interrupted(out):
    """Settle and tidy a folder before writing to it.

    Every interrupted publish is undone first (each case would settle its own
    on its next publish anyway; doing all of them up front means a case this
    run does not touch is not left half one run's and half another's).  Then
    what a hard-killed run leaves behind is cleared: a scratch folder with
    half-drawn figures, a holding folder moved aside by _remove_dir, and the
    temporaries of the manifest, curve store and sweep -- none of which
    should end up in a zip that is sent off.

    A folder takes one writer at a time: two jobs writing one results
    folder at once could undo each other's publish here.  Run top-ups one
    after another (sbatch --dependency=afterany).
    """
    if not os.path.isdir(out):
        return
    names = sorted(os.listdir(out))
    for d in names:
        if d.startswith(".partial_") and d.endswith(".old"):
            _recover(os.path.join(out, d), out)
    for d in names:
        path = os.path.join(out, d)
        if (d.startswith(".partial_") and not d.endswith(".old")
                and os.path.isdir(path)):
            shutil.rmtree(path, ignore_errors=True)
        elif d in LEFTOVERS or (d.endswith(".partial.png")
                                and os.path.basename(out) in SWEEP_DIRS):
            _discard(path)
    clear_shared_tmp(out)


def _publish(stage, out, names, drop=()):
    """Move stage/<name> onto out/<name> for every name -- all or none.

    Every file about to be replaced, and every file in `drop` that is to be
    deleted, is first moved into a holding folder; only once every new file
    is in place is that folder deleted.  If any step fails, what was placed
    is taken back out and the originals are put back.  On Windows a PNG open
    in a viewer makes os.replace fail with PermissionError, which is the case
    this is for: a case's files are either all this run's or all the last's.

    A journal written before anything moves records what the publish was
    doing, so a run killed in the middle is undone by the next one
    (_recover) rather than leaving a mixture.
    """
    hold = stage + ".old"
    _recover(hold, out)
    if os.path.isdir(hold):
        raise RuntimeError(
            f"cannot publish {os.path.basename(stage)[len('.partial_'):]}: "
            f"{hold} is still in use; close whatever has its files open "
            f"and rerun")
    os.makedirs(hold)
    todo = list(dict.fromkeys(list(names) + list(drop)))
    existing = [n for n in todo if os.path.exists(os.path.join(out, n))]
    write_json(os.path.join(hold, PUBLISH_JOURNAL),
               {"names": list(names), "existing": existing,
                "pid": os.getpid(), "host": socket.gethostname(),
                "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    try:
        for n in existing:
            os.replace(os.path.join(out, n), os.path.join(hold, n))
        for n in names:
            os.replace(os.path.join(stage, n), os.path.join(out, n))
    except BaseException:
        if _revert(hold, out):
            _settle(hold)
            _clear_hold(hold)
        raise
    os.replace(os.path.join(hold, PUBLISH_JOURNAL),
               os.path.join(hold, PUBLISH_DONE))
    _clear_hold(hold)


def thin_curve(t, mean, sd, must=()):
    """A mean +- sd curve cut to about CURVE_POINTS log-spaced samples plus
    the times in ``must`` (the snapshot and report times), for manifest.json:
    enough to draw comparison.png from the run folder alone."""
    t = np.asarray(t, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    n = t.size
    keep = {0, n - 1}
    keep.update(int(round(x)) - 1 for x in np.geomspace(1, n, CURVE_POINTS))
    for s in must:
        k = grid_index(t, s)
        if k is not None:
            keep.add(k)
    at = np.array(sorted(k for k in keep if 0 <= k < n))
    return {"t": [float(x) for x in t[at]],
            "mean": [_num_or_none(x) for x in mean[at]],
            "sd": [_num_or_none(x) for x in sd[at]]}


def _num_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None


def figure_composition(comp, off, lattice, path, name=DISPLAY_NAME,
                       n_seeds=None, notes=None):
    """<psi>(t) - psi0 of the Solver and the FNO, seed mean +-1 sd, on a log t
    axis.  The axis is the departure from the starting mean, because <psi>
    itself would need an offset such as 1e-8+4e-1 on its ticks.  The material
    A content c_A = 50 (1 + psi) % at the start and the end, and the largest
    departures, are in ``notes``."""
    t = np.asarray(comp["t"], dtype=float)
    mt, mp = comp["psi_true"], comp["psi_pred"]
    gap_t = np.asarray(mt, dtype=float) - off
    gap_p = np.asarray(mp, dtype=float) - off
    fig, ax = plt.subplots(figsize=(7.4, 5.2), facecolor="white")
    for mean, sd, col in ((gap_t, comp.get("sd_true"), C_TRUTH),
                          (gap_p, comp.get("sd_pred"), C_PRED)):
        if sd is not None:
            ax.fill_between(t, mean - sd, mean + sd, color=col, alpha=0.15,
                            lw=0)
    ax.plot(t, gap_t, color=C_TRUTH, lw=1.6, ls="--", label="Solver")
    ax.plot(t, gap_p, color=C_PRED, lw=1.6, label="FNO")
    ax.set_xscale("log")
    ax.set_xlabel(r"$t$", fontsize=13)
    ax.set_ylabel(r"$\langle\psi\rangle(t) - \bar\psi_0$", fontsize=13)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                        useMathText=True)
    ax.grid(alpha=0.25, ls=":", which="both")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fno_paths.save_figure(fig, path, FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")
    if notes is not None:
        a0, a1 = float(P.psi_to_a(mt[0])), float(P.psi_to_a(mt[-1]))
        ap = np.asarray(mp, dtype=float)
        k = (int(np.nonzero(np.isfinite(ap))[0][-1]) if np.isfinite(ap).any()
             else 0)

        def largest(gap):
            return f"{float(np.nanmax(np.abs(gap))):.1e}"

        notes[os.path.basename(path)] = [
            "departure of the mean order parameter from its start, <psi>(t) - "
            "psi0, Solver dashed and FNO solid, seed mean +- 1 sd"
            + ("" if n_seeds is None else f" of {n_seeds} seeds")
            + ", log t axis",
            f"case: psi0 = {off:+.1f}, l = {lattice}",
            "material A content c_A = 50 (1 + psi) %: "
            f"{a0:.2f} % at t = {t[0]:.0f}, Solver {a1:.2f} % and FNO "
            f"{float(P.psi_to_a(ap[k])):.2f} % at t = {t[k]:.0f}",
            f"largest departure of the seed mean: Solver {largest(gap_t)}, "
            f"FNO {largest(gap_p)}"]


def figure_free_energy(results, off, path, lattice=BASE_LATTICE, notes=None):
    """Free energy per site against t, the Solver and the FNO, seed means: the
    data and the monotone test of figure_kinetics' panel (e).  The count of
    seeds whose FNO free energy never rises (P.free_energy_monotone) goes
    into ``notes``.  Nothing when the rollouts have no diagnostics."""
    if not any(len(r["diag_times"]) for r in results):
        return None
    t, e_true = P.seed_series(results, lambda r: r["energy_true"])
    _, e_pred = P.seed_series(results, lambda r: r["energy_pred"])
    mt, mp = P.seed_mean(e_true), P.seed_mean(e_pred)
    horizon, onset = P.case_marks(results, lattice)
    falling = P.free_energy_monotone(results)
    fig, ax = plt.subplots(figsize=(7.4, 5.2), facecolor="white")
    ax.plot(t, mt, color=C_TRUTH, lw=1.8, ls="--", label="Solver")
    ax.plot(t, mp, color=C_PRED, lw=1.8, label="FNO")
    P.mark_time(ax, t[0], t[-1], horizon, onset, short=True)
    ax.set_xscale("log")
    ax.set_xlabel(r"$t$", fontsize=13)
    ax.set_ylabel(r"free energy per site  $\langle f \rangle$", fontsize=13)
    ax.grid(alpha=0.25, ls=":", which="both")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fno_paths.save_figure(fig, path, FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")
    n = len(results)
    if notes is not None:
        ok = np.nonzero(np.isfinite(mp))[0]
        lines = [
            "free energy per site <f> against t (log axis), Solver dashed and "
            f"FNO solid, mean over {n} seed{'s' if n != 1 else ''}",
            f"case: psi0 = {off:+.1f}, l = {lattice}",
            f"FNO monotone: {sum(falling)} of {n} seeds (the FNO free energy "
            "never rises, np.all(np.diff(energy_pred) <= 1e-9))",
            f"<f> at t = {t[0]:.0f}: Solver {mt[0]:.5f}, FNO {mp[0]:.5f}; at "
            f"t = {t[ok[-1]]:.0f}: Solver {mt[ok[-1]]:.5f}, FNO "
            f"{mp[ok[-1]]:.5f}" if ok.size else "no finite FNO mean"]
        if horizon is not None:
            lines.append(f"dashed vertical line: training horizon t = "
                         f"{horizon:g}")
        if onset is not None:
            lines.append(f"grey region: L > l/{P.FINITE_SIZE:g} from t = "
                         f"{onset:g} (L: the first zero of C(r,t))")
        notes[os.path.basename(path)] = lines
    return {"monotone": int(sum(falling)), "n_seeds": n}


def _floored(v):
    v = np.asarray(v, dtype=float)
    return np.where(np.isfinite(v), v, P.R2_FLOOR)


def _sd(v):
    v = np.asarray(v, dtype=float)
    return float(v.std(ddof=1)) if v.size > 1 else 0.0


def composition_rows(results, comp, token):
    """The flat schema's composition_series for one case: the starting
    composition and every snapshot time, as dicts of CSV_COLS (without case and psi0)."""
    t, mt, mp = comp["full"]
    n_full = comp.get("n_full")
    snaps, _, t0 = rollout_times(results)
    ref = float(mt[0])

    def row(tt, pt, pp, n):
        a_t, a_p = float(P.psi_to_a(pt)), float(P.psi_to_a(pp))
        return {"t": float(tt), "psi_mean_truth": _num_or_none(pt),
                "psi_mean_pred": _num_or_none(pp),
                "A_pct_truth": _num_or_none(a_t),
                "B_pct_truth": _num_or_none(100.0 - a_t),
                "A_pct_pred": _num_or_none(a_p),
                "B_pct_pred": _num_or_none(100.0 - a_p),
                "drift_pred": _num_or_none(pp - ref), "n_seeds": int(n)}

    rows = [row(t0, ref, ref, len(results))]
    for s in snaps:
        k = grid_index(t, s)
        if k is None or abs(s - t0) < 1e-9:
            continue
        n = int(n_full[k]) if n_full is not None else len(results)
        rows.append(row(s, mt[k], mp[k], n))
    return rows


def _schedule_of(results):
    """How the rollout stepped: the schedule a result carries (a schedule
    engine stores ``schedule``), else read off its times."""
    got = results[0].get("schedule")
    if isinstance(got, dict):
        return {k: (v.item() if hasattr(v, "item") else v)
                for k, v in got.items()}
    t = np.asarray(results[0]["times"], dtype=float)
    gaps = np.diff(np.concatenate([[result_start(results[0])], t]))
    if gaps.size and np.allclose(gaps, gaps[0]):
        return {"kind": "fixed step", "dt": float(gaps[0]),
                "steps": int(t.size)}
    return {"kind": "schedule", "dt_min": float(gaps.min()) if gaps.size
            else None, "dt_max": float(gaps.max()) if gaps.size else None,
            "steps": int(t.size)}


def _sum_or_none(vals):
    got = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(sum(got)) if got else None


def _snap_panels(results, shown):
    """{"rule", "seed", "panels": [{"t", "r2"}]} for a snapshots figure."""
    if not shown:
        return None
    res = results[best_index(results)]
    panels = []
    for t in shown.get("times", ()):
        if t in res["snapshots"]:
            panels.append({"t": float(t),
                           "r2": _num_or_none(P.r2_score(
                               *res["snapshots"][t]))})
    return {"rule": "best seed (highest trajectory-mean R2 among those that "
                    "did not diverge)", "seed": shown.get("seed"),
            "panels": panels}


def flat_case(results, seeds, off, lattice, token, entry, comp):
    """The flat manifest.json entry of one case,
    from the rollouts in memory; kept in the case's bookkeeping entry so a
    run folder can be written without them (write_run_outputs)."""
    snaps, t_end, t_start = rollout_times(results)
    thr = float(P.R2_THRESHOLD)
    times, mean, sd, _ = mean_sd_curve(results)
    end = _floored(r2_at(results, t_end))
    traj = np.array([float(np.mean(_floored(r["r2"]))) for r in results])
    surv = []
    for r in results:
        tt = np.asarray(r["times"], dtype=float)
        v = _floored(r["r2"])
        n = min(tt.size, v.size)
        below = np.nonzero(v[:n] < thr)[0]
        surv.append(float(tt[below[0]]) if below.size else float(t_end))
    surv = np.array(surv)
    never = int(sum(1 for r in results
                    if not np.any(_floored(r["r2"]) < thr)))
    crosses = np.nonzero(mean < thr)[0]
    ranks = seed_ranks(results, seeds)

    def pick(kind):
        k = ranks[kind]
        return {"seed": k["seed"], "r2_trajectory_mean": k["mean_r2"],
                "r2_at_end": k["r2_at_end"]}

    horizon = P.train_horizon()

    def growth(key):
        try:
            t, mt, mp = P.length_means(results, key)
        except Exception:
            return None
        if not t.size or not np.isfinite(mt).any():
            return None
        lo = float(t[-1]) / 4.0
        out = {"late_window": [lo, float(t[-1])],
               "n_truth": _num_or_none(P.fit_growth_exponent(
                   t, mt, lo, float(t[-1]))),
               "n_pred": _num_or_none(P.fit_growth_exponent(
                   t, mp, lo, float(t[-1]))),
               "L_start_truth": _num_or_none(mt[0]),
               "L_end_truth": _num_or_none(mt[-1]),
               "L_end_pred": _num_or_none(mp[-1])}
        if horizon is not None and t[0] < horizon < t[-1]:
            out["train_window"] = [float(t[0]), float(horizon)]
            out["n_truth_train"] = _num_or_none(P.fit_growth_exponent(
                t, mt, float(t[0]), float(horizon)))
            out["n_pred_train"] = _num_or_none(P.fit_growth_exponent(
                t, mp, float(t[0]), float(horizon)))
        return out

    comp_e = entry.get("composition") or {}
    s0, e0 = comp_e.get("start") or {}, comp_e.get("end") or {}
    fs = entry.get("growth") or {}
    out = {
        # a composition case is at the base box; every other size is a box
        "kind": ("composition" if re.fullmatch(r"psi0[mp]\d+\.\d+", token)
                 else "box"),
        "token": token, "lattice": int(lattice), "psi0": float(off),
        "n_seeds": len(results), "seed_list": [int(x) for x in seeds],
        "steps": int(max(len(r["r2"]) - int(P.has_start_sample(r))
                         for r in results)),
        "t_start": t_start, "t_end": float(t_end),
        "n_diverged": int(sum(1 for r in results
                              if r.get("diverged_at") is not None)),
        "r2_at_end": {"mean": float(end.mean()), "sd": _sd(end),
                      "median": float(np.median(end)),
                      "best": float(end.max()), "worst": float(end.min())},
        "r2_trajectory_mean": {"mean": float(traj.mean()), "sd": _sd(traj)},
        "survival": {"threshold": thr, "mean": float(surv.mean()),
                     "sd": _sd(surv), "n_survive_to_end": never,
                     "mean_curve_crosses": (float(times[crosses[0]])
                                            if crosses.size else None)},
        "seeds": {k: pick(k) for k in ("best", "median", "worst")},
        "growth": {"half_max": growth("L_half"),
                   "zero_crossing": growth(None)},
        "composition": ({
            "A_percent_start_truth": s0.get("A_pct_truth"),
            "A_percent_start_pred": s0.get("A_pct_pred"),
            "A_percent_end_truth": e0.get("A_pct_truth"),
            "A_percent_end_pred": e0.get("A_pct_pred"),
            "shift_pp": comp_e.get("shift_pp"),
            "max_abs_psi_drift_pred": comp_e.get("max_abs_drift_pred"),
            "psi_drift_rate_pred": comp_e.get("drift_rate_psi_per_t")}
            if comp_e else None),
        "finite_size": {
            "L_limit": (fs.get("finite_size") or {}).get("limit"),
            "first_time_mean_L_zero_above":
                (fs.get("finite_size") or {}).get("onset")},
        "composition_series": (composition_rows(results, comp, token)
                               if comp is not None else None),
        "snapshots": _snap_panels(results, entry.get("snapshots")),
        "psi_pdf": entry.get("psi_pdf"),
        "timing": {"gt_seconds": _sum_or_none(
                       [r.get("gt_seconds") for r in results]),
                   "fno_seconds": _sum_or_none(
                       [r.get("fno_seconds") for r in results]),
                   "fno_steps": _sum_or_none(
                       [r.get("fno_steps", r.get("steps"))
                        for r in results]),
                   "gt_cached": sum(1 for r in results
                                    if r.get("gt_cached")),
                   "n_seeds": len(results)},
        "r2_at_times": r2_at_times(results),
        "schedule": _schedule_of(results),
        "r2_curve": thin_curve(times, mean, sd,
                               must=tuple(snaps) + R2_REPORT_TIMES),
    }
    return out


def write_case(results, seeds, off, lattice, out, key=None,
               name=DISPLAY_NAME, videos=None, root=None, extras=None,
               token=None, keep_film=True, stem=None, sizedir=None):
    """The submission figures of one case, into the run folder ``out``, and
    their data and the film into the case's folder ``extras`` (extras/<case>/
    of the rollout cache); returns its bookkeeping entry and mean R2 curve.

    This is the only place submission figures are drawn.  fno_predict calls
    it from write_case_outputs while that case's rollouts are still in
    memory, so the run never holds more than one case at a time -- at
    l = 256 with video frames, fifty rollouts are already about five
    gigabytes.

    The figures go into the box size's folder of the run folder,
    <sizedir>/<kind>_<stem>.png for the kinds in FIG_KINDS (snapshots,
    phase_ordering_kinetics, domain_growth with L_half as the main curve,
    psi_distribution, composition, free_energy); ``sizedir`` is
    size_folder(lattice) and ``stem`` the composition token, unless given
    (the critical twin on other seeds is drawn into ``out`` itself, sizedir
    "").  The text they do not carry is in entry["notes"], which
    write_run_outputs puts into the folder's figure_notes.txt.  ``token`` is
    case_token(), the name of the case in manifest.json; when it is not
    given it is made from ``off`` and ``lattice`` (a lattice equal to the
    base box is the composition case).

    Everything is drawn into scratch folders and then published in two
    all-or-nothing steps (_publish), extras first: a failure anywhere --
    drawing, or a locked file at the very end -- leaves the previous run's
    files for this case whole.  A figure this call did not make (no later
    snapshots, no means) is removed from the run folder in the same step.  A
    film this call did not make stays (``keep_film``): it lives in extras/
    and a redraw from records, which makes none, must leave an earlier run's
    film alone.  A live run, whose new rollouts make none, passes
    keep_film=False, so the film of an earlier run does not outlive them.

    ``root`` is the run folder the entry's file names are relative to.
    """
    if not results:
        raise ValueError(f"no rollouts for psi0={off:+.2f}, l={lattice}")
    rollout_times(results)
    videos = WRITE_VIDEOS if videos is None else videos
    if token is None:
        token = case_token(off, None if int(lattice) == P.LATTICE
                           else lattice)
    stem = comp_slug(off) if stem is None else stem
    sizedir = size_folder(lattice) if sizedir is None else sizedir
    if extras is None:
        extras = P.case_dir(off, None if int(lattice) == P.LATTICE
                            else lattice)
    where = os.path.join(out, sizedir) if sizedir else out
    os.makedirs(where, exist_ok=True)
    os.makedirs(extras, exist_ok=True)
    recover_interrupted(where)
    recover_interrupted(extras)
    stage = os.path.join(where, stage_name(stem))
    xstage = os.path.join(extras, STAGE)
    for d in (stage, xstage):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    data = {}
    fig = {k: fig_name(k, stem) for k in FIG_KINDS}
    try:
        entry = {"seeds": len(results), "cache_key": key, "token": token,
                 "seed_list": [int(s) for s in seeds]}
        notes: dict = {}
        entry["snapshots"] = figure_snapshots(
            results, seeds, off, os.path.join(stage, fig["snapshots"]),
            name=name, notes=notes)
        entry["seed_ranks"] = seed_ranks(results, seeds)
        comp = composition_series(results)
        if comp is not None:
            data["composition"] = comp
            entry["composition"] = composition_entry(comp, off, lattice,
                                                     len(results))
            figure_composition(comp, off, lattice,
                               os.path.join(stage, fig["composition"]),
                               name=name, n_seeds=len(results), notes=notes)
        entry["psi_pdf"] = figure_psi_pdf(
            results, off, os.path.join(stage, fig["psi_distribution"]),
            lattice, data=data, notes=notes)
        entry["growth"] = figure_growth(
            results, off, os.path.join(stage, fig["domain_growth"]),
            lattice, data=data, notes=notes)
        P.figure_kinetics(
            results, seeds,
            os.path.join(stage, fig["phase_ordering_kinetics"]),
            off, lattice, notes=notes)
        figure_free_energy(results, off,
                           os.path.join(stage, fig["free_energy"]), lattice,
                           notes=notes)
        entry["film"] = (write_video(
            results, seeds, off, lattice,
            os.path.join(xstage, "rollout.mp4"), name=name)
            if videos else None)
        write_case_data(xstage, data)
        names = [fig[k] for k in FIG_KINDS
                 if os.path.exists(os.path.join(stage, fig[k]))]
        xnames = [f for f in CASE_FILES + CASE_DATA
                  if os.path.exists(os.path.join(xstage, f))]
        entry["flat"] = flat_case(results, seeds, off, lattice, token, entry,
                              comp)
        entry["notes"] = {k: v for k, v in notes.items() if k in names}
        _publish(xstage, extras, xnames,
                 [f for f in (COMPOSITION_FILE,) if f not in xnames]
                 + ([] if keep_film else
                    [f for f in CASE_FILES if f not in xnames]))
        _publish(stage, where, names,
                 [fig[k] for k in FIG_KINDS if fig[k] not in names])
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(xstage, ignore_errors=True)
    entry["folder"] = token
    entry["size_folder"] = sizedir
    entry["stem"] = stem
    entry["files"] = [f"{sizedir}/{n}" if sizedir else n for n in names]
    entry["extras"] = {"folder": os.path.basename(os.path.normpath(extras)),
                       "files": list(xnames)}
    print(f"[saved] {len(names)} figures -> {where}, {len(xnames)} data files "
          f"-> {extras}")
    times, curve, sd, overall = mean_sd_curve(results)
    entry["mean_r2"] = overall
    entry["r2_at_times"] = r2_at_times(results)
    n_end = sum(1 for r in results if len(r["r2"]) >= len(curve))
    n_div = sum(1 for r in results if r.get("diverged_at") is not None)
    out_curve = {"off": off, "lattice": lattice, "times": times,
                 "mean_curve": curve, "sd_curve": sd,
                 "stats": {"n_seeds": len(results), "n_end": n_end,
                           "n_div": n_div}}
    entry["curve_digest"] = curve_digest(curve_record(out_curve))
    return entry, out_curve


def write_case_data(folder, data):
    """The averages over every seed that the figures are drawn from, as CSV."""
    if "composition" in data:
        comp = data["composition"]
        np.savetxt(os.path.join(folder, COMPOSITION_FILE),
                   np.column_stack([comp["t"], comp["psi_true"],
                                    comp["psi_pred"], comp["A_true"],
                                    100.0 - comp["A_true"], comp["A_pred"],
                                    100.0 - comp["A_pred"],
                                    comp["drift_pred"], comp["n_seeds"]]),
                   delimiter=",", comments="",
                   header="t,psi_mean_truth,psi_mean_pred,A_pct_truth,"
                          "B_pct_truth,A_pct_pred,B_pct_pred,drift_pred,"
                          "n_seeds",
                   fmt=("%.10g", "%.12e", "%.12e", "%.10f", "%.10f",
                        "%.10f", "%.10f", "%.12e", "%d"))
    if "psi_distribution" in data:
        c, h_true, h_pred, lever = data["psi_distribution"]
        # the lever-rule fractions are one value each, repeated down the rows
        # so that numpy and pandas read the first three columns as before
        np.savetxt(os.path.join(folder, "psi_distribution.csv"),
                   np.column_stack([c, h_true, h_pred]
                                   + [np.full(c.size, v) for v in lever]),
                   delimiter=",", comments="",
                   header="psi,density_true,density_pred,lever_A_pct,"
                          "A_rich_true_pct,A_rich_pred_pct")
    if "domain_growth" in data:
        g = data["domain_growth"]
        t = g["t"]
        none = np.full(t.size, np.nan)
        half = g["half"] if g["half"] is not None else (none, none)
        fits = [np.full(t.size, np.nan if g[w] is None else g[w][side])
                for w in ("train", "extrap") for side in ("n_truth", "n_fno")]
        horizon = np.full(t.size, np.nan if g["horizon"] is None
                          else g["horizon"])
        np.savetxt(os.path.join(folder, "domain_growth.csv"),
                   np.column_stack([t, g["L_true"], g["L_pred"], g["n_true"],
                                    g["n_pred"], *half, horizon, *fits]),
                   delimiter=",",
                   header="t,L_true_mean,L_pred_mean,n_seeds_true,"
                          "n_seeds_pred,L_half_true_mean,L_half_pred_mean,"
                          "train_horizon,n_train_true,n_train_pred,"
                          "n_extrap_true,n_extrap_pred",
                   comments="", fmt=("%.18e", "%.18e", "%.18e", "%d", "%d")
                   + ("%.18e",) * 7)


PAPER_C = ("#1f2fd0", "#e11ec4", "#137a2e", "#c07000", "#00868b")
PAPER_M = ("o", "s", "^", "D", "v")
PAPER_RC: Any = {
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": 10, "axes.labelsize": 12,
    "legend.fontsize": 8, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 0.9, "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
    "ytick.right": True, "xtick.minor.visible": True,
    "ytick.minor.visible": True, "xtick.major.size": 5.0,
    "ytick.major.size": 5.0, "xtick.minor.size": 2.8,
    "ytick.minor.size": 2.8, "legend.frameon": False,
    "legend.handlelength": 1.9, "legend.labelspacing": 0.35,
    "savefig.bbox": "tight"}
SWEEP_DPI = 300


def _sweep_panel_paper(ax, series, horizon, label, band=0.13):
    """One panel of r2_time: mean R2(t) +-1 sd per curve, the training
    horizon dotted.  The R2 axis follows the data: from the lowest point
    drawn (the -1 sd edge of a band; 0 when nothing is drawn) up to R2 = 1,
    with 4 % of that span as a margin, so a run whose curves stay near 1
    fills the panel."""
    lows, t0, t1 = [], np.inf, -np.inf
    for i, (lab, t, mean, sd) in enumerate(series):
        c = PAPER_C[i % len(PAPER_C)]
        if sd is not None:
            # R2 cannot exceed 1: the band's upper edge is clipped there.
            ax.fill_between(t, mean - sd, np.minimum(mean + sd, 1.0), color=c, alpha=band,
                            lw=0)
            lows.append(np.nanmin(mean - sd))
        else:
            lows.append(np.nanmin(mean))
        ax.plot(t, mean, color=c, marker=PAPER_M[i % len(PAPER_M)],
                markevery=max(1, len(t) // 14), label=lab)
        t0, t1 = min(t0, float(t[0])), max(t1, float(t[-1]))
    if horizon and t0 < horizon < t1:
        ax.axvline(horizon, color="0.25", ls=":", lw=1.0,
                   label="training horizon")
    lo = float(np.nanmin(lows)) if lows else 0.0
    pad = max(0.005, 0.04 * (1.0 - lo))
    ax.set_ylim(lo - pad, 1.0 + pad)
    ax.set_xlim(t0, t1)
    ax.set_ylabel("$R^2$")
    if label:
        ax.text(0.97, 0.95, label, transform=ax.transAxes, ha="right",
                va="top")
    ax.legend(loc="lower left", borderaxespad=0.6, ncol=2,
              columnspacing=1.0)


def _curve_notes(series, counts, horizon):
    """The lines of figure_notes.txt for R2 curves: each curve's mean R2 at
    the reporting times, the seeds behind it and the horizon."""
    lines = []
    for (lab, t, mean, sd), n in zip(series, counts):
        at = []
        for want in R2_REPORT_TIMES:
            k = grid_index(t, want)
            if k is not None and np.isfinite(mean[k]):
                at.append(f"t = {want:g}: {mean[k]:.3f}")
        lines.append(plain_text(lab)
                     + ("" if n is None else f" ({n} seeds)")
                     + ", mean R2 at " + ", ".join(at))
    lines.append((f"dotted line: training horizon t = {horizon:g}; "
                  if horizon else "") + "bands: +-1 sd over the seeds")
    return lines


def _one_panel(series, horizon, path):
    """One panel of r2_time as a figure of its own, at the size and in the
    style of a case's figures: no panel letter, and lighter bands, since five
    of them overlap."""
    with plt.rc_context({"lines.linewidth": 1.8, "lines.markersize": 5.5}):
        fig, ax = plt.subplots(figsize=(7.4, 5.2), facecolor="white")
        _sweep_panel_paper(ax, series, horizon, None, band=0.08)
        ax.set_xlabel("$t$", fontsize=13)
        ax.set_ylabel("$R^2$", fontsize=13)
        ax.grid(alpha=0.25, ls=":")
        fig.tight_layout()
        tmp = f"{os.path.splitext(path)[0]}.partial.png"
        try:
            fno_paths.save_figure(fig, tmp, FIG_DPI, bbox_inches="tight")
            os.replace(tmp, path)
        finally:
            plt.close(fig)
            _discard(tmp)
    print(f"[saved] {path}")


# --- the R2 sweeps: R2(t) per box size, and per composition ----------------
# A box size has one colour and one marker in every panel and every run.
BOX_COLORS = {128: "#440154", 256: "#5dc863", 512: "#e0b400"}
BOX_MARKS = {128: "o", 256: "D", 512: "v"}
# A composition keeps the colour PAPER_C gives it in the order of the five.
COMP_LIST = (-0.4, -0.2, 0.0, 0.2, 0.4)
SWEEP_NOTE_TIMES = (1000.0, 2000.0, 5000.0, 10000.0)
# One image size for every sweep image (inches).
SWEEP_FIGSIZE = (4.4, 3.2)
# Drawn before 2026-11-09 or 2026-11-11; no longer written.  A run folder may
# still hold them.
STALE_SWEEP_FILES = ("r2_sweep.png", "r2_sweep.pdf", "r2_compositions.png",
                     "r2_boxes.png",
                     "r2_sweep_per_size.png", "r2_sweep_per_size.pdf",
                     "r2_sweep_per_case.png", "r2_sweep_per_case.pdf")


def _comp_style(off, i):
    """(colour, marker) of a composition; by position when psi0 is not one of
    the five."""
    k = next((j for j, v in enumerate(COMP_LIST) if abs(v - off) < 1e-9), i)
    return PAPER_C[k % len(PAPER_C)], PAPER_M[k % len(PAPER_M)]


def sweep_panels(curves):
    """The panels of the two sweep figures, from sweep_curves.npz entries:
    (per_size, per_case).  per_size is [(box size, rows)], one panel for each
    size that has two or more compositions, in order of size; per_case is
    [(psi0, rows)], one for each composition that has two or more box sizes.
    A row is (psi0, l, t, mean, sd, seeds); rows go by composition in a
    per_size panel and by size in a per_case panel."""
    by_size: dict = {}
    by_comp: dict = {}
    for (off, lat), c in sorted(curves_as_cases(curves).items(),
                                key=lambda kv: (kv[0][1], kv[0][0])):
        row = (off, lat, np.asarray(c["times"], dtype=float),
               np.asarray(c["mean_curve"], dtype=float),
               None if c.get("sd_curve") is None
               else np.asarray(c["sd_curve"], dtype=float),
               c["stats"]["n_seeds"])
        by_size.setdefault(lat, []).append(row)
        by_comp.setdefault(off, []).append(row)
    return ([(lat, v) for lat, v in sorted(by_size.items()) if len(v) >= 2],
            [(off, v) for off, v in sorted(by_comp.items()) if len(v) >= 2])


def _sweep_lines(panels, what):
    """For each panel its image name, its description and the drawn lines
    [(label, t, mean, sd, seeds, colour, marker, heavy)].  what is "size" or
    "comp".  The image is l<size>.png or <composition token>.png."""
    out = []
    for key, rows in panels:
        if what == "size":
            name, tag = f"l{int(key)}.png", rf"$l = {key}$"
        else:
            name, tag = comp_token(key) + ".png", comp_math(key)
        lines = []
        for i, (off, lat, t, mean, sd, n) in enumerate(rows):
            if what == "size":
                col, mk = _comp_style(off, i)
                lab, heavy = comp_math(off), abs(off) < 1e-12
            else:
                col, mk = BOX_COLORS.get(lat, "0.3"), BOX_MARKS.get(lat, "o")
                lab, heavy = rf"$l = {lat}$", False
            lines.append((lab, t, mean, sd, n, col, mk, heavy))
        out.append((name, tag, lines))
    return out


def _sweep_notes(what, item, horizon):
    """The lines of a sweep folder's figure_notes.txt for one image: what is
    plotted, then each line's seed count and its mean R2 +- sd at the
    reporting times that lie within the run.  ``item`` is one entry of
    _sweep_lines."""
    _, tag, rows = item
    lines = [("mean R2 against time t, linear t axis, the same t and R2 "
              "range in every image of this folder; "
              + ("one image for each box size, one line for each "
                 "composition psi0 (psi0 = 0 heaviest)" if what == "size"
                 else "one image for each composition psi0, one line for "
                 "each box size l")
              + "; each line is the mean over the seeds with a band of +-1 sd "
              "over the seeds"
              + (f"; dotted line: training horizon t = {horizon:g}"
                 if horizon else "")), tag]
    for lab, t, mean, sd, n, *_ in rows:
        at = []
        for want in SWEEP_NOTE_TIMES:
            k = grid_index(t, want)
            if k is None or not np.isfinite(mean[k]):
                continue
            s = ("" if sd is None or not np.isfinite(sd[k])
                 else f" +- {sd[k]:.3f}")
            at.append(f"t = {want:g}: {mean[k]:.3f}{s}")
        lines.append(plain_text(lab) + f" ({n} seeds), mean R2 +- sd at "
                     + (", ".join(at) if at else "no reporting time"))
    return lines


def _draw_sweep(drawn, horizon, folder):
    """Draw one image for each entry of ``drawn`` into ``folder``: R2(t)
    with one linear t range and one R2 range for the whole set, the
    training horizon dotted and in every legend.  No title, no panel letter.
    PNG at SWEEP_DPI, written through a temporary file."""
    os.makedirs(folder, exist_ok=True)
    t0, t1, lo = np.inf, -np.inf, 1.0
    for _, _, rows in drawn:
        for _, t, mean, sd, *_ in rows:
            t0, t1 = min(t0, float(t[0])), max(t1, float(t[-1]))
            lo = min(lo, float(np.nanmin(mean if sd is None
                                         else mean - sd)))
    pad = max(0.005, 0.04 * (1.0 - lo))
    for name, _, rows in drawn:
        with plt.rc_context(PAPER_RC):
            fig, ax = plt.subplots(figsize=SWEEP_FIGSIZE)
            for lab, t, mean, sd, _, col, mk, heavy in rows:
                if sd is not None:
                    # R2 cannot exceed 1: the band's upper edge is clipped there.
                    ax.fill_between(t, mean - sd, np.minimum(mean + sd, 1.0), color=col,
                                    alpha=0.09, lw=0)
                ax.plot(t, mean, color=col, marker=mk,
                        markevery=max(1, len(t) // 14), label=lab,
                        lw=2.6 if heavy else 1.1, ms=5.0 if heavy else 3.5,
                        zorder=4 if heavy else 3)
            if horizon and t0 < horizon < t1:
                ax.axvline(horizon, color="0.25", ls=":", lw=1.0,
                           label="training horizon")
            ax.set_xlim(t0, t1)
            ax.set_ylim(lo - pad, 1.0 + pad)
            ax.set_xlabel("$t$")
            ax.set_ylabel("$R^2$")
            ax.legend(loc="lower left", borderaxespad=0.6, ncol=2,
                      columnspacing=1.0, fontsize=7)
            path = os.path.join(folder, name)
            tmp = os.path.join(folder, os.path.splitext(name)[0]
                               + ".partial.png")
            try:
                fno_paths.save_figure(fig, tmp, SWEEP_DPI)
                os.replace(tmp, path)
            finally:
                plt.close(fig)
                _discard(tmp)
        print(f"[saved] {path}")


def figure_r2_sweeps(curves, root, horizon=None):
    """The single R2 sweep images in ``root``: the mean R2(t) with a band of
    +-1 sd over the seeds, r2_sweep_per_size/l<size>.png (one for each box
    size that has two or more compositions, a line for each composition) and
    r2_sweep_per_case/<composition token>.png (one for each composition that
    has two or more box sizes, a line for each size).  ``curves`` are the
    sweep_curves.npz entries.  A folder with no image is not made.  Each
    folder gets a figure_notes.txt of its own.  An image of an earlier draw
    that is no longer made is left and reported by stale_sweep_files.
    Returns the file names written, relative to ``root``."""
    per_size, per_case = sweep_panels(curves)
    names = []
    for stem, what, panels in (("r2_sweep_per_size", "size", per_size),
                               ("r2_sweep_per_case", "comp", per_case)):
        if not panels:
            continue
        drawn = _sweep_lines(panels, what)
        folder = os.path.join(root, stem)
        _draw_sweep(drawn, horizon, folder)
        names += [f"{stem}/{name}" for name, _, _ in drawn]
        write_figure_notes(folder, {item[0]: _sweep_notes(what, item, horizon)
                                    for item in drawn})
    return names


def stale_sweep_files(root, drawn=None):
    """What ``root`` still holds that is no longer drawn: the sweep figures
    of earlier forms, the images of the sweep folders that this draw did not
    make (``drawn``, names relative to root; None: not known, then none is
    named), and the folders of dropped
    box sizes (l160/).  Names relative to root."""
    out = [f for f in STALE_SWEEP_FILES
           if os.path.isfile(os.path.join(root, f))]
    for d in SWEEP_DIRS if drawn is not None else ():
        try:
            have = sorted(f for f in os.listdir(os.path.join(root, d))
                          if f.endswith(".png")
                          and not f.endswith(".partial.png"))
        except OSError:
            continue
        out += [f"{d}/{f}" for f in have if f"{d}/{f}" not in drawn]
    try:
        out += [d + "/" for d in sorted(os.listdir(root))
                if re.fullmatch(r"l\d+", d) and int(d[1:]) in P.DROPPED_SIZES
                and os.path.isdir(os.path.join(root, d))]
    except OSError:
        pass
    return out


def size_curves(curves, lattice):
    """The cases of one box size in sweep_curves.npz as r2_time's series
    [(label, t, mean, sd)] by composition."""
    cs = curves_as_cases(curves)
    return [(rf"$\bar\psi_0 = {off:+.1f}$",
             np.asarray(c["times"], dtype=float),
             np.asarray(c["mean_curve"], dtype=float),
             None if c.get("sd_curve") is None
             else np.asarray(c["sd_curve"], dtype=float),
             c["stats"]["n_seeds"])
            for (off, lat), c in sorted(cs.items()) if lat == lattice]


def figure_r2_time(series, path, horizon=None, notes=None):
    """r2_time.png of one box size: the mean R2(t) of each composition, with
    the spread over the seeds (+-1 sd) as a band.  ``series`` is
    [(label, t, mean, sd, seeds)]."""
    _one_panel([x[:4] for x in series], horizon, path)
    if notes is not None:
        notes[os.path.basename(path)] = [
            "mean R2 against time, one line per composition at this box "
            "size, +-1 sd over the seeds as a band, linear t axis"
        ] + _curve_notes([x[:4] for x in series], [x[4] for x in series],
                         horizon)


def write_composition_csv(path, cases):
    """The flat layout's composition.csv: CSV_COLS, one row per snapshot time and the
    start of every case (cases = flat_case dicts); the last column is the
    box size l."""
    lines = [",".join(CSV_COLS)]
    for c in cases:
        for r in c.get("composition_series") or ():
            vals = [c["token"], f"{c['psi0']:+.2f}", format(r["t"], ".10g")]
            for k in CSV_COLS[3:-1]:
                v = r.get(k)
                vals.append("" if v is None else
                            (str(int(v)) if k == "n_seeds"
                             else format(v, ".10g")))
            vals.append(str(int(c["lattice"])))
            lines.append(",".join(vals))
    write_text(path, "\n".join(lines) + "\n")
    print(f"[saved] {path}")


def flat_cases(internal, root=None):
    """The cases of a run for manifest.json, from the bookkeeping manifest:
    ({token: flat case}, {alias: token}, [skipped notes]).  The box size
    l = 128 is an alias of the critical quench at the base lattice,
    psi0p0.00, when the run has it; a case drawn apart (the twin on other
    seeds) and an entry with no stored case are skipped."""
    cases, aliases, skipped = {}, {}, []
    for kind in KINDS:
        for k, e in sorted(_dict(internal.get(kind)).items(),
                           key=lambda kv: _size_order(kind, kv[0])):
            u = fno_paths.flat_entry(e)
            if not isinstance(u, dict):
                skipped.append(f"{kind} {k}: no stored case (rewritten by a "
                               f"newer run)")
                continue
            tok = u["token"]
            if tok.endswith("_sweep_seeds"):
                skipped.append(f"{kind} {k}: drawn apart in extras/")
                continue
            if kind == "sizes" and tok != case_token(0.0, int(k)):
                aliases[case_token(0.0, int(k))] = tok
            cases.setdefault(tok, u)
    crit = case_token(P.CRITICAL_OFF)
    if crit in cases:
        aliases.setdefault(case_token(P.CRITICAL_OFF, P.LATTICE), crit)
    return cases, aliases, skipped


def best_record(best):
    """The checkpoint selection of a run (the checkpoint's "best": epoch,
    score, survival, accuracy, peak_survival) as plain numbers for
    manifest.json, or None when it says nothing.  A value that is not a finite
    number is left out, so a numpy scalar or a NaN cannot break the write."""
    out = {}
    for k, v in _dict(best).items():
        if isinstance(v, bool) or not isinstance(v, (int, float, np.number)):
            continue
        if not np.isfinite(float(v)):
            continue
        out[str(k)] = int(v) if isinstance(v, (int, np.integer)) else float(v)
    return out or None


def case_file(kind, token):
    """The path of a case figure in the run folder, from the case's token:
    l256/snapshots_psi0p0.20.png."""
    place = token_place(token)
    if place is None:
        return fig_name(kind, token)
    return f"{place[0]}/{fig_name(kind, place[1])}"


def pooled_end(cases):
    """The R2 at the end of a set of cases (flat_case dicts) pooled: the mean
    over every seed of every case (a diverged seed at P.R2_FLOOR), as
    {"cases": n, "seeds": n, "mean": ...}; None for no case."""
    rows = [(int(c["n_seeds"]), float(c["r2_at_end"]["mean"]))
            for c in cases if _dict(c.get("r2_at_end")).get("mean") is not None
            and c.get("n_seeds")]
    n = sum(w for w, _ in rows)
    if not n:
        return None
    return {"cases": len(rows), "seeds": n,
            "mean": sum(w * m for w, m in rows) / n}


def pooled_groups(cases):
    """The pooled R2 at the end of a run's cases ({token: flat case}), for
    manifest.json and report.txt:
    "common_6_cases": the five compositions at l = 128 and the critical quench
        at l = 256 (COMMON_CASES), only when the run has all six: the
        headline that runs made before the box-size grid can be compared on;
    "all_cases": every case of the run;
    "per_size": {"l128": ...} by box size."""
    out: dict = {"common_6_cases": None,
                 "all_cases": pooled_end(cases.values()), "per_size": {}}
    common = (pooled_end([cases[t] for t in COMMON_CASES])
              if all(t in cases for t in COMMON_CASES) else None)
    if common is not None:
        out["common_6_cases"] = {**common, "tokens": list(COMMON_CASES)}
    for lat in sorted({int(c["lattice"]) for c in cases.values()}):
        got = pooled_end([c for c in cases.values()
                          if int(c["lattice"]) == lat])
        if got is not None:
            out["per_size"][size_folder(lat)] = got
    return out


def sweep_images(root, folder):
    """The images and the notes file that a sweep folder of a run holds."""
    try:
        return sorted(f for f in os.listdir(os.path.join(root, folder))
                      if f.endswith(".png") and not f.endswith(".partial.png")
                      or f == NOTES_FILE)
    except OSError:
        return []


def run_manifest(root, internal, facts, cases, aliases, skipped, files):
    """manifest.json of a run folder, the flat schema: the
    run-level facts, then cases{token}.  ``facts`` comes from
    fno_predict.run_facts."""
    crit = cases.get(case_token(P.CRITICAL_OFF)) or next(
        iter(cases.values()), {})
    sizes = sorted({size_folder(c["lattice"]) for c in cases.values()},
                   key=lambda d: int(d[1:]))
    snaps = sorted({p["t"] for c in cases.values()
                    for p in (c.get("snapshots") or {}).get("panels", ())})
    m = {
        "label": facts.get("label") or os.path.basename(
            os.path.normpath(root)),
        "source": facts.get("source", "fno_predict"),
        "cache": facts.get("cache"),
        "checkpoint": facts.get("checkpoint"),
        "checkpoint_fingerprint": facts.get("fingerprint"),
        "checkpoint_numerics": facts.get("numerics"),
        "checkpoint_best": best_record(facts.get("best")),
        "n_parameters": facts.get("n_parameters"),
        "train_t_end": facts.get("train_horizon"),
        "t_start": crit.get("t_start"), "t_end": crit.get("t_end"),
        "pdf_time": crit.get("t_end"),
        "lattice": facts.get("lattice", BASE_LATTICE),
        "box_off": float(P.CRITICAL_OFF),
        "threshold": float(P.R2_THRESHOLD),
        "diverge_abs": DIVERGE_ABS,
        "diverged_kept_in_means_at": float(P.R2_FLOOR),
        "snapshot_times": snaps or facts.get("snapshots"),
        "snapshot_times_main": {
            t: [p["t"] for p in (c.get("snapshots") or {}).get("panels", ())]
            for t, c in cases.items()},
        "seeds": crit.get("seed_list"), "n_seeds": crit.get("n_seeds"),
        "n_diverged": sum(c.get("n_diverged", 0) for c in cases.values()),
        "skipped_cases": skipped,
        "schedules": {t: c.get("schedule") for t, c in cases.items()},
        "best_seed_rule": "highest trajectory-mean R2 among the seeds that "
                          "did not diverge and have a prediction at t_end",
        "figures": {
            "best_seed": [case_file("snapshots", t) for t in cases],
            "seed_average": [case_file(k, t) for t in cases
                             for k in ("phase_ordering_kinetics",
                                       "domain_growth", "psi_distribution",
                                       "composition", "free_energy")],
            "data": ["composition.csv"]
            + [f"{d}/{f}" for d in sizes for f in SIZE_FILES]
            + [f"{d}/{f}" for d in SWEEP_DIRS
               for f in sweep_images(root, d)]},
        "size_folders": sizes,
        "pooled_r2_at_end": pooled_groups(cases),
        "aliases": aliases, "cases": cases,
        "extras": "Work/rollout_cache/<label>/extras/ (FNO-only outputs)",
        "files": files,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return m


def _prose(text, first="  ", rest="    ", width=78):
    """A paragraph of report.txt, wrapped to the flat layout's 78 columns."""
    return textwrap.wrap(text, width=width, initial_indent=first,
                         subsequent_indent=rest)


def _fmt(x, spec=".4f"):
    return "-" if x is None else format(x, spec)


def r2_group_table(cases):
    """report.txt's R2 at the reporting times, pooled and per case group
    (compositions at the base box; box sizes, whose l = 128 is the critical
    quench), as lines; then the 6 cases every run has, when the run has all of
    them and more (the headline older runs are compared on), and each box
    size.  Each cell is the mean over every seed (a
    diverged one at P.R2_FLOOR), in brackets the mean over the seeds that
    had not diverged (the group protocol's rule), then the number diverged.  Nothing when
    no case records R2 at those times."""
    crit = case_token(P.CRITICAL_OFF)
    sizes = sorted({int(c["lattice"]) for c in cases.values()})
    groups = [("pooled", list(cases)),
              ("compositions", [t for t, c in cases.items()
                                if c.get("kind") == "composition"]),
              ("box sizes", [t for t, c in cases.items()
                             if t == crit or c.get("kind") == "box"])]
    if (all(t in cases for t in COMMON_CASES)
            and len(cases) > len(COMMON_CASES)):
        groups.append(("common 6 cases", list(COMMON_CASES)))
    if len(sizes) > 1:
        groups += [(f"l = {lat}", [t for t, c in cases.items()
                                   if int(c["lattice"]) == lat])
                   for lat in sizes]
    keys = sorted({k for c in cases.values()
                   for k in _dict(c.get("r2_at_times"))}, key=float)
    if not keys:
        return []
    L = [""] + _prose(
        f"R2 at fixed times. Each cell: the all-seeds mean over the "
        f"group's cases (a diverged seed counts R2 = {P.R2_FLOOR:g}); in "
        f"brackets the survivors-only mean (diverged seeds left out); then "
        f"the number of diverged seeds. The critical quench ({crit}, "
        f"l = 128) is in both groups and once in the pooled row. "
        f"'common 6 cases' (shown when the run has more than those) are "
        f"the five compositions at l = 128 and the critical quench at "
        f"l = 256: the set every run has, so the one to compare runs "
        f"on.", rest="  ")
    L += ["", f"  {'group':<15}{'cases':>5}"
          + "".join(f"{'t = ' + k:>19}" for k in keys)]
    for name, members in groups:
        if not members:
            continue
        line = f"  {name:<15}{len(members):>5}"
        for k in keys:
            rows = [_dict(_dict(cases[t].get("r2_at_times")).get(k))
                    for t in members]
            rows = [r for r in rows if r.get("n")]
            n = sum(r["n"] for r in rows)
            if not n:
                line += f"{'-':>19}"
                continue
            mean = sum(r["mean"] * r["n"] for r in rows) / n
            alive = [r for r in rows
                     if r.get("n_alive") and r.get("mean_alive") is not None]
            na = sum(r["n_alive"] for r in alive)
            ma = (sum(r["mean_alive"] * r["n_alive"] for r in alive) / na
                  if na else None)
            div = sum(r.get("diverged", 0) for r in rows)
            cell = f"{mean:.3f} ({_fmt(ma, '.3f')}) {div}"
            line += f"{cell:>19}"
        L.append(line)
    return L


def _checkpoint_options(facts):
    """The model options the checkpoint has, from the training setup (a
    record from before it held them: from the model's config, where an option
    that is absent is off), or "not recorded"."""
    setup, arch = facts.get("train_setup") or {}, facts.get("arch") or {}
    if not (setup or arch):
        return "not recorded"
    return ", ".join(f"{k}={bool(setup.get(k, arch.get(k, False)))}"
                     for k in ("mu_channel", "lowk_fill"))


def _speedup_text(timings):
    """Solver time over FNO time, over the cases whose references were all solved.

    A case with any trajectory from the cache has network time but no (or
    only part of the) solver time, so it is left out of both sums.  The text
    says over how many of the cases the ratio is taken, and is "not
    available" when no case was solved.
    """
    solved = [x for x in timings
              if x.get("n_seeds") and not x.get("gt_cached")
              and x.get("gt_seconds") is not None
              and x.get("fno_seconds") is not None]
    gt = sum(x["gt_seconds"] for x in solved)
    fn = sum(x["fno_seconds"] for x in solved)
    if not solved or not gt or not fn:
        return "not available (no case had all its references solved)"
    return (f"{gt / fn:.1f}x over {len(solved)} of {len(timings)} cases "
            f"(those whose references were all solved)")


def step_schedule_lines(picks):
    """Section 5 of report.txt, two lines per quench: the pick, then how many
    candidates diverged and the grid edges it sits on.  One line says so when the run
    has no pick or no candidate table."""
    picks = {k: v for k, v in (picks or {}).items() if isinstance(v, dict)}
    if not picks:
        return ["  no step schedule was picked: a fixed-step run"]
    out = []
    for key in sorted(picks, key=float):
        pick = picks[key]
        table = pick.get("table") or []
        head = (f"  psi0 {key}: alpha {float(pick['alpha']):g}, dt_max "
                f"{float(pick['dt_max']):g}")
        if not table:
            how = ("fixed by hand (--no-tune), no candidates were tried"
                   if str(pick.get("source", "")).startswith("SCHED_")
                   else "an old meta.json with no candidate table")
            out.append(f"{head}; {how}")
            continue
        ed = P.tune_edges(pick, table)
        flags = [f"{n} {w}" for n, w in (
            ("dt_max", "smallest"), ("dt_max", "largest"),
            ("alpha", "smallest"), ("alpha", "largest"))
            if ed[f"{n}_{w}"]]
        score = pick.get("score")
        steps = next((r.get("steps") for r in table
                      if abs(r["alpha"] - pick["alpha"]) < 1e-9
                      and abs(r["dt_max"] - pick["dt_max"]) < 1e-9), None)
        out.append(
            f"{head}, {steps} steps, score "
            + (f"{score:.4f}" if isinstance(score, (int, float)) else "n/a")
            + f" ({pick.get('source')})")
        out.append(
            " " * len(f"  psi0 {key}: ")
            + f"{ed['k']} of {ed['n']} candidates had diverged tuning seeds; "
            + ("on the grid edge: " + ", ".join(flags) if flags
               else "inside the grid"))
    out += _prose("A pick on the smallest dt_max or alpha of the grid means "
                  "a smaller step was not tried.  The table of every "
                  "candidate is in the run's meta.json (key \"schedule\").",
                  first="  note: ", rest="        ")
    return out


def write_report(path, man, facts, diag=None, picks=None):
    """report.txt: the flat layout's sections (model, composition and its conservation,
    solve time, summary, step schedule, diagnostics, low-k response) filled from the
    manifest, the run facts and diagnostics.json, plus FNO's conventions.
    ``picks`` is the step schedule of each quench (fno_predict._PICKS, the
    "schedule" of meta.json), for section 5."""
    bar = "=" * 78
    cases = man["cases"]
    L = [bar, f"  {DISPLAY_NAME} - report of {man['label']}", bar,
         f"  written {man['written']}", ""]

    def head(n, title):
        L.extend(["", bar, f"  {n}  {title}", bar])

    head(1, "TRAINED MODEL")
    arch = facts.get("arch") or {}
    num = facts.get("numerics")
    L += [f"  drawn by             {man.get('source')}",
          f"  checkpoint           {man.get('checkpoint')}",
          f"  fingerprint          {man.get('checkpoint_fingerprint')}",
          f"  parameters           {facts.get('n_parameters')}",
          f"  precision            {facts.get('dtype')}",
          f"  numerics             "
          f"{P.numerics_text(num) if num else 'not recorded'}",
          f"  trained to t         {man.get('train_t_end')}",
          f"  variant / size       "
          f"{json.dumps(facts.get('train_setup'), default=str)}",
          f"  checkpoint options   {_checkpoint_options(facts)}",
          f"  lattice, box sizes   {man.get('lattice')}, "
          f"{', '.join(man.get('size_folders') or []) or '-'}",
          f"  rolled out           t = {man.get('t_start')} to "
          f"{man.get('t_end')}, snapshots "
          f"{facts.get('snapshots') or man.get('snapshot_times')}",
          f"  test seeds           {man.get('n_seeds')} ("
          + ("fixed list" if fno_paths.seed_set_name(
              facts.get("seed_set")) == "group"
             else str(facts.get("seed_set")))
          + "); diverged seeds in all cases: "
          f"{man.get('n_diverged')} (|psi| > {man['diverge_abs']:g})",
          f"  device               {facts.get('device') or 'not recorded'}",
          "", "  architecture"]
    L += [f"    {k:<24} {v}" for k, v in sorted(arch.items())] or ["    -"]
    L += ["", "  checkpoint selection"]
    best = facts.get("best") or {}
    L += [f"    {k:<24} {v}" for k, v in sorted(best.items())
          if not isinstance(v, (list, dict))] or ["    -"]
    L += ["", "  training configuration"]
    L += [f"    {k:<24} {v}" for k, v in sorted(
        (facts.get("train_config") or {}).items())] or ["    -"]

    head(2, "MIXTURE COMPOSITION AND ITS CONSERVATION")
    L += ["  c_A = 100 (1 + psi) / 2 percent of material A, c_B = 100 - c_A.",
          "  Mass is conserved, so the mean psi (and c_A) stays at its start;",
          "  the shift is the prediction's c_A at the end minus the start.",
          f"  hard_mass_conservation={facts.get('hard_mass_conservation')}, "
          f"zero_dc={facts.get('zero_dc')}, "
          f"laplacian_output={(facts.get('train_setup') or {}).get('laplacian_output')}",
          "", f"  {'case':<15}{'A start':>9}{'A end tr':>10}{'A end pr':>10}"
          f"{'shift pp':>10}{'max |dpsi|':>12}{'drift/t':>11}"]
    for t, c in cases.items():
        k = c.get("composition") or {}
        L.append(f"  {t:<15}{_fmt(k.get('A_percent_start_truth'), '9.3f')}"
                 f"{_fmt(k.get('A_percent_end_truth'), '10.3f')}"
                 f"{_fmt(k.get('A_percent_end_pred'), '10.3f')}"
                 f"{_fmt(k.get('shift_pp'), '10.4f')}"
                 f"{_fmt(k.get('max_abs_psi_drift_pred'), '12.2e')}"
                 f"{_fmt(k.get('psi_drift_rate_pred'), '11.2e')}")
    for t, c in cases.items():
        rows = c.get("composition_series")
        if not rows:
            continue
        L += ["", f"  {t}: c_A (%) over time", f"  {'t':>8}{'A truth':>10}"
              f"{'A FNO':>10}{'B truth':>10}{'B FNO':>10}{'n seeds':>9}"]
        for r in rows:
            L.append(f"  {r['t']:>8.0f}{_fmt(r['A_pct_truth'], '10.3f')}"
                     f"{_fmt(r['A_pct_pred'], '10.3f')}"
                     f"{_fmt(r['B_pct_truth'], '10.3f')}"
                     f"{_fmt(r['B_pct_pred'], '10.3f')}{r['n_seeds']:>9}")

    head(3, "SOLVE TIME")
    tm = [c.get("timing") or {} for c in cases.values()]
    gt, fn = _sum_or_none([x.get("gt_seconds") for x in tm]), _sum_or_none(
        [x.get("fno_seconds") for x in tm])
    n_tr = sum(x.get("n_seeds", 0) for x in tm)
    steps = _sum_or_none([x.get("fno_steps") for x in tm])
    L += [f"  trajectories         {n_tr}",
          f"  reference solver     {_fmt(gt, '.1f')} s in total "
          f"({sum(x.get('gt_cached', 0) for x in tm)} of the trajectories "
          f"came from the cache, which costs no solve)",
          f"  FNO                  {_fmt(fn, '.1f')} s in total, "
          f"{_fmt(steps, '.0f')} model steps",
          f"  speedup              "
          + _speedup_text(tm)]

    head(4, "SUMMARY")
    L += _prose(f"R2 at the end of the rollout over the seeds (diverged seeds "
                f"count R2 = {P.R2_FLOOR:g}), mean R2 over the trajectory, "
                f"time until R2 < {man['threshold']:.2f}.", rest="  ")
    L += ["", f"  {'case':<15}{'steps':>6}{'R2 end':>8}{'sd':>7}{'median':>8}"
          f"{'best':>7}{'worst':>7}{'traj':>7}{'survives':>9}{'never':>7}"
          f"{'n pred':>8}{'n div':>6}"]
    for t, c in cases.items():
        e, tr, sv = c["r2_at_end"], c["r2_trajectory_mean"], c["survival"]
        g = (c.get("growth") or {}).get("half_max") or {}
        L.append(f"  {t:<15}{c['steps']:>6}{e['mean']:>8.3f}{e['sd']:>7.3f}"
                 f"{e['median']:>8.3f}{e['best']:>7.3f}{e['worst']:>7.3f}"
                 f"{tr['mean']:>7.3f}{sv['mean']:>9.0f}"
                 f"{sv['n_survive_to_end']:>3}/{c['n_seeds']:<3}"
                 f"{_fmt(g.get('n_pred'), '8.3f')}{c['n_diverged']:>6}")
    L += [""] + _prose(
        f"survives = mean over the seeds of the first time R2 < "
        f"{man['threshold']:.2f} (a seed that never drops counts the end "
        "time); never = seeds that stay above it; n pred = growth exponent "
        "of the predicted half-maximum length L_half over the late third "
        "of the rollout [t_end/4, t_end].", rest="  ")
    pooled = man.get("pooled_r2_at_end") or {}
    L += ["", "  end-of-rollout R2 pooled (mean over every seed of the "
          "cases):"]
    common = pooled.get("common_6_cases")
    if common:
        L.append(f"    common 6 cases (in every run)          "
                 f"{common['mean']:.4f}  ({common['cases']} cases, "
                 f"{common['seeds']} seeds)")
    else:
        L.append("    common 6 cases                         not all "
                 "six are in this run")
    for name, got in (("all cases", pooled.get("all_cases")),
                      *((f"box size {d}", g) for d, g in
                        _dict(pooled.get("per_size")).items())):
        if got:
            L.append(f"    {name:<38} {got['mean']:.4f}  ({got['cases']} "
                     f"cases, {got['seeds']} seeds)")
    L += r2_group_table(cases)

    head(5, "STEP SCHEDULE")
    L += step_schedule_lines(picks)

    diag = diag if isinstance(diag, dict) else {}
    tile = diag.get("tile") or {}
    head(6, "DIAGNOSTICS")
    if not tile:
        L.append("  no diagnostics.json for this run")
    elif "error" in tile:
        L.append(f"  tile invariance failed: {tile['error']}")
    else:
        # the precision and device the steps were taken in (older
        # diagnostics.json files do not say)
        how = (f", {tile['dtype']} on {tile.get('device')}"
               if tile.get("dtype") else "")
        L += [f"  tile invariance: l = {tile.get('l')} tiled to "
              f"{tile.get('tiled_l')} at t = {tile.get('t')}{how}: largest "
              f"difference {tile.get('max_abs')}, relative L2 "
              f"{tile.get('rel_l2')} ("
              + ("invariant" if tile.get("invariant") else "NOT invariant")
              + ")"]
    head(7, "LOW-k INCREMENT RESPONSE")
    lowk = diag.get("lowk") or {}
    if not lowk:
        L.append("  no diagnostics.json for this run")
    elif "text" in lowk:
        L.append(f"  l = {lowk.get('lattice')}: {lowk['text']}")
    else:
        L.append(f"  {lowk.get('skipped') or lowk.get('error') or lowk}")

    head(8, "CONVENTIONS")
    L += _prose(f"A seed counts as diverged when |psi| exceeds "
                f"{man['diverge_abs']:g}. Such a seed stays in every mean at "
                f"R2 = {P.R2_FLOOR:g} from then on (the all-seeds mean); the "
                f"survivors-only mean, which leaves diverged seeds out, is "
                f"used only in brackets in the R2 table of section 4.",
                first="  * ", rest="    ")
    L += _prose("Times of this report are the rollout's own; R2 at 2000 / "
                "5000 / 10000 is read where the rollout has such a sample.",
                first="  * ", rest="    ")
    L += _prose("FNO-only outputs (per-case summaries, CSVs, best-seed "
                "figures, film, sweep curves) are in Work/rollout_cache/"
                "<label>/extras/, not in this folder.", first="  * ", rest="    ")
    write_text(path, "\n".join(L) + "\n")
    print(f"[saved] {path}")


FLAT_FIGURE = re.compile(
    r"(?:%s)_(?:psi0[mp]\d+\.\d+|l\d+)(?:_sweep_seeds)?\.png"
    % "|".join(FIG_KINDS))


def flat_case_files(folder):
    """The per-case figures a run folder of the old flat layout holds at its
    top (snapshots_psi0p0.20.png, snapshots_l160.png ...); [] for a folder
    that is missing or not of that layout."""
    try:
        return sorted(f for f in os.listdir(folder)
                      if FLAT_FIGURE.fullmatch(f)
                      and os.path.isfile(os.path.join(folder, f)))
    except OSError:
        return []


def refuse_flat(folders):
    """Stop, before anything is written, when one of the run folders still
    holds the per-case files of the old flat layout.  This never moves or
    deletes them; the user moves the folder to Archive/."""
    for folder in folders:
        old = flat_case_files(folder)
        if old:
            raise SystemExit(
                f"{folder} still holds {len(old)} per-case figure(s) of the "
                f"old flat layout (e.g. {old[0]}).  Nothing was written.  "
                f"Move the folder to Archive/ first (mv {folder} Archive/), "
                f"then run again; the per-case figures now go into one "
                f"folder per box size (l128/, l256/ ...).")


def lost_curves(internal, curves, size=None):
    """The documented cases ("<kind> <key>", of one box size or of all) that
    have no stored R2 curve.  The critical case at the base lattice that the
    "sizes" entry repeats is not counted."""
    mine = {(kind, k) for kind in KINDS for k in _dict(internal.get(kind))
            if (size is None or _size_order(kind, k)[0] == size)
            and not (kind == "sizes"
                     and _size_order(kind, k)[0] == P.LATTICE)}
    return sorted(f"{kind} {k}" for kind, k in mine if (kind, k) not in curves)


def write_size_outputs(root, size, cases, internal, curves, horizon, part):
    """The files of one box size's folder besides the case figures:
    composition_all.png, r2_time.png and figure_notes.txt.  ``cases`` are
    the flat_case dicts of that size, ``part`` the error catcher of
    write_run_outputs."""
    folder = os.path.join(root, size_folder(size))
    os.makedirs(folder, exist_ok=True)
    notes: dict = {}
    for kind in KINDS:
        for e in _dict(internal.get(kind)).values():
            e = _dict(e)
            flat = _dict(fno_paths.flat_entry(e))
            place = token_place(flat.get("token", ""))
            if place is not None and place[0] == size_folder(size):
                notes.update(_dict(e.get("notes")))
    comp = sorted((c for c in cases if c.get("composition_series")),
                  key=lambda c: c["psi0"])
    if comp:
        series = []
        for c in comp:
            rows = c["composition_series"]
            series.append((c["psi0"], np.array([r["t"] for r in rows]),
                           np.array([r["A_pct_truth"] for r in rows],
                                    dtype=float),
                           np.array([r["A_pct_pred"] if r["A_pct_pred"]
                                     is not None else np.nan
                                     for r in rows], dtype=float)))
        part(f"{size_folder(size)}/composition_all.png",
             figure_composition_all, series,
             os.path.join(folder, COMPOSITION_FIG), P.submission_name(),
             comp[0]["n_seeds"] if len({c["n_seeds"] for c in comp}) == 1
             else None, notes)
    # a plot drawn without the curve of a documented case would silently
    # lose a line: then the old figure stays and the log says why
    lost = lost_curves(internal, curves, size)
    series = size_curves(curves, size)
    if series and lost:
        print(f"[submission] {size_folder(size)}/r2_time.png NOT redrawn: "
              f"{', '.join(lost)} have no stored curve; rerun the prediction "
              f"over those cases (cached rollouts make it quick)", flush=True)
    elif series:
        part(f"{size_folder(size)}/r2_time.png", figure_r2_time, series,
             os.path.join(folder, "r2_time.png"), horizon, notes)
    part(f"{size_folder(size)}/figure_notes.txt", write_figure_notes, folder,
         notes)


def write_run_outputs(root, extras, facts=None):
    """The top of a run folder -- composition.csv, r2_sweep_per_size/ and
    r2_sweep_per_case/ (single images, each with a figure_notes.txt),
    manifest.json, report.txt -- and the
    files of each box-size folder besides the case figures (composition_all
    .png, r2_time.png, figure_notes.txt), from the bookkeeping the cases left
    in ``extras``; no rollouts are needed.  Each file is written through a
    temporary, and a part that fails is reported in the returned dict and
    raised at the end, after the rest was written."""
    facts = facts or {}
    os.makedirs(root, exist_ok=True)
    refuse_flat([root])
    recover_run(root, None)
    internal = load_manifest(extras)
    curves = load_curves(extras)
    cases, aliases, skipped = flat_cases(internal, root)
    out = {"files": [], "errors": []}
    notes: dict = {}
    sweep_made: list | None = None

    def part(what, fn, *a):
        try:
            return fn(*a)
        except Exception as exc:
            out["errors"].append(f"{what}: {type(exc).__name__}: {exc}")
            print(f"[error] {what} FAILED:", flush=True)
            traceback.print_exc()
            return None

    allc = sorted(cases.values(), key=lambda c: (int(c["lattice"]),
                                                 float(c["psi0"])))
    comp = [c for c in allc if c.get("composition_series")]
    if comp:
        part("composition.csv", write_composition_csv,
             os.path.join(root, COMPOSITION_FILE), comp)
    horizon = facts.get("train_horizon")
    for size in sorted({int(c["lattice"]) for c in allc}):
        write_size_outputs(root, size,
                           [c for c in allc if int(c["lattice"]) == size],
                           internal, curves, horizon, part)
    # the two sweeps hold a line for every documented case: with a curve
    # missing they keep their old files, as each r2_time.png does
    lost = lost_curves(internal, curves)
    if curves and lost:
        print(f"[submission] the images of r2_sweep_per_size/ and "
              f"r2_sweep_per_case/ NOT redrawn: {', '.join(lost)} have no "
              f"stored curve; rerun the prediction over those cases (cached "
              f"rollouts make it quick)", flush=True)
    elif curves:
        sweep_made = part("r2_sweep", figure_r2_sweeps, curves, root,
                          horizon)
    old = stale_sweep_files(root, sweep_made)
    if old:
        print(f"[submission] {root} still holds {', '.join(old)}: these "
              f"are no longer drawn and can be moved to Archive/",
              flush=True)
    # the model's training curves, before the folder's files are listed (the
    # manifest holds the figure); none where the history is not at hand, and
    # a failure here is reported but never fails the run
    if cases:
        draw_training_curves(
            root,
            facts.get("label") or os.path.basename(os.path.normpath(root)),
            facts.get("checkpoint"), best_record(facts.get("best")),
            facts.get("n_parameters"), P.ROOT, P.submission_name())
    part("figure_notes.txt", write_figure_notes, root, notes)
    present = sorted(f for f in os.listdir(root)
                     if os.path.isfile(os.path.join(root, f)))
    for d in sorted(os.listdir(root)):
        if ((re.fullmatch(r"l\d+", d) and int(d[1:]) not in P.DROPPED_SIZES
             or d in SWEEP_DIRS)
                and os.path.isdir(os.path.join(root, d))):
            present += [f"{d}/{f}" for f in sorted(os.listdir(
                os.path.join(root, d)))
                if os.path.isfile(os.path.join(root, d, f))
                and not f.endswith(".partial.png")]
    man = run_manifest(root, internal, facts, cases, aliases, skipped,
                       present)
    if cases:
        diag = _read_json(os.path.join(extras, P.DIAG_FILE))
        part("report.txt", write_report, os.path.join(root, "report.txt"),
             man, facts, diag, dict(P._PICKS))
        man["files"] = sorted(set(man["files"]) | {"manifest.json",
                                                   "report.txt"})
        part("manifest.json", write_json, os.path.join(root, MANIFEST), man)
    out["files"] = man["files"]
    if out["errors"]:
        raise RuntimeError("; ".join(out["errors"]))
    return man


def build(cache, out, model_path=None):
    """Regenerate one run's results folder from its result cache.

    The same code a prediction run writes a folder with (fno_predict's
    evaluate_all and write_case_outputs) is fed the cached rollouts instead
    of new ones, so the folder is laid out and drawn exactly as a live run's:
    every case's figures, averages and best seed, then the manifest, README
    and sweeps.  The cache is only read -- nothing in it is deleted, nothing
    is rolled out or solved -- and the window is taken from the rollouts.
    ``model_path`` names the checkpoint the cache was made with, for the
    manifest's and timing report's provenance only.

    A case the cache does not hold is listed under "missing"; one that fails
    under "failed" (the manifest's failed_last_run), and the rest are built.
    Returns the folder's manifest.
    """
    if (os.path.isdir(out) and set(os.listdir(out)) - {P.DIAG_FILE}
            and not os.path.exists(os.path.join(out, MANIFEST))):
        raise SystemExit(f"{out} already holds files and is not a results "
                         f"folder of this layout; choose a new --out")
    names = ("RESULTS_DIR", "RESULTS_TEMPLATE", "VIEWS", "START_TIME",
             "END_TIME", "SNAPSHOT_TIMES", "MODEL_PATH", "CACHE_END_TIME",
             "_RUN_SNAPSHOTS", "EXTRAS_ROOT", "RUN_SOURCE", "RUN_CACHE",
             "KEEP_FILM", "ENGINE", "CONDITIONED", "RECORDED_NUMERICS",
             "PLAN_ONLY")
    saved = {n: getattr(P, n) for n in names}
    picks = dict(P._PICKS)
    missing = []
    made: dict = {}
    try:
        _, files, _ = cache_group(cache, 0.0, BASE_LATTICE)
        for f in files:
            # the first readable one: a damaged file costs only its own
            # case (from_cache below), not the whole rebuild
            try:
                res = P._load_result(f)
                w = data_window([res])
            except Exception:
                continue
            P.START_TIME, P.END_TIME = w["start"], w["end"]
            P.SNAPSHOT_TIMES = tuple(w["snapshots"])
            # a schedule run says so in its rollouts, and in the meta.json
            # of its folder (with its picks) when the folder has one
            made = P.read_meta(cache) or {}
            if res.get("schedule"):
                made.setdefault("engine", "schedule")
            restore_engine(made)
            break
        # the numerics the cache was made in, as its meta.json says (a
        # cache with none is from before the stamp)
        P.RECORDED_NUMERICS = P.full_numerics(made.get("numerics"))
        P.RESULTS_DIR, P.RESULTS_TEMPLATE, P.VIEWS = out, None, None
        P.CACHE_END_TIME, P._RUN_SNAPSHOTS = None, ()
        # a legacy cache holds the cases of the protocol before the box-size
        # grid: the compositions at the base box and the critical quench at
        # the other kept sizes it has rollouts of, and at the default lattice
        # the kept size of that protocol (LEGACY_SIZES), listed as missing
        # when it lacks it; the dropped sizes are never planned
        found = {int(m.group(1)) for f in (os.listdir(cache)
                                           if os.path.isdir(cache) else ())
                 for m in [re.match(
                     r"fno_ch_l(\d+)_seed\d+_[0-9a-f]{12}\.npz$", f)] if m}
        lats = sorted((found | (set(LEGACY_SIZES) if P.LATTICE == BASE_LATTICE
                                else set())) - {P.LATTICE}
                      - set(P.DROPPED_SIZES))
        P.PLAN_ONLY = (
            [(off, None, tuple(P.seeds_for(off))) for off in P.eval_offsets()]
            + [(P.CRITICAL_OFF, lat, tuple(P.SWEEP_SEEDS)) for lat in lats])
        # a legacy cache is only read: the extras go beside the figures
        P.EXTRAS_ROOT = os.path.join(out, EXTRAS_DIR)
        P.KEEP_FILM = True
        P.RUN_SOURCE = f"make_submission, regenerated from {cache}"
        P.RUN_CACHE = cache
        model, ckpt = None, {}
        if model_path and os.path.exists(model_path):
            model, ckpt = P.load_checkpoint(model_path, device="cpu")
        P.MODEL_PATH = model_path or ""

        def from_cache(model, device, off, lattice=None, seeds=None):
            lat = P.LATTICE if lattice is None else lattice
            sd, results, key = load_case(cache, off, lat)
            if not results:
                missing.append(f"psi0={off:+.2f} l={lat}")
                print(f"[miss] no cached rollouts for psi0={off:+.2f}, "
                      f"l={lat} in {cache}", flush=True)
                return None
            print(f"\n--- psi0 = {off:+.2f}, l = {lat}: {len(results)} "
                  f"seeds, cache key {key} ---", flush=True)
            return P.write_case_outputs(results, sd, off, lattice, key=key)

        P.evaluate_all(model, ckpt, None, case_fn=from_cache, resume=False,
                       clean=False)
    finally:
        for n, v in saved.items():
            setattr(P, n, v)
        P._PICKS.clear()
        P._PICKS.update(picks)
    ex = os.path.join(out, EXTRAS_DIR)
    m = load_manifest(ex)
    if os.path.exists(os.path.join(ex, BOOKKEEPING)):
        m.update(source=f"make_submission, regenerated from {cache}",
                 cache=cache)
        write_json(os.path.join(ex, BOOKKEEPING), m)
    m["missing"] = missing
    m["failed"] = list(m.get("failed_last_run") or ())
    return m


def records_home(folder, meta, status):
    """The folder a run's records really live in, which the manifest and the
    report name.  A scratch copy made to redraw a run without touching its
    cache (meta.json and extras/ copied, records/ linked with cp -as) goes
    away; the cache it links to stays.  Any other folder is its own home."""
    for name, ok in status.items():
        if not ok:
            continue
        first = P.record_file(folder, name, meta["cases"][name]["seeds"][0])
        if not os.path.islink(first):
            return folder
        # <home>/records/<case>/seed<n>.npz
        home = os.path.realpath(first)
        for _ in range(3):
            home = os.path.dirname(home)
        known = fno_paths.rollout_dir(os.path.basename(home))
        return known if os.path.realpath(known) == home else home
    return folder


def recorded_device(folder, meta):
    """The device a run's rollouts took their steps on, from the timing file
    in its extras: meta.json and the records do not hold it.  None when there
    is no such file."""
    name = f"{meta.get('output_label') or P.LABEL}_timing.json"
    return _dict(_dict(_read_json(os.path.join(folder, EXTRAS_DIR, name)))
                 .get("meta")).get("device")


class RecordedModel:
    """What a records folder kept of its model: enough for the outputs to
    carry its flags, size and precision.  It runs nothing."""

    def __init__(self, info):
        self.config = info.get("config") or {}
        self.size = int(info.get("n_parameters") or 0)
        self.dtype = info.get("dtype") or "unknown"

    def n_parameters(self):
        return self.size

    def parameters(self):
        yield types.SimpleNamespace(dtype=self.dtype)


REDRAW_GLOBALS = ("RECORDED_CHECKPOINT", "EXTRAS_ROOT", "RUN_SOURCE", "RUN_CACHE",
                  "KEEP_FILM", "RESULTS_DIR", "RESULTS_TEMPLATE", "VIEWS", "START_TIME",
                  "END_TIME", "SNAPSHOT_TIMES", "MODEL_PATH",
                  "CACHE_END_TIME", "_RUN_SNAPSHOTS", "LATTICE", "EVAL_MODE",
                  "SIZE_SWEEP", "EVAL_SEEDS", "SWEEP_SEEDS", "SEED_SET",
                  "SNAPSHOTS_EXPLICIT", "LOG_TIME", "WRITE_VIDEO", "LABEL",
                  "AUTO_LABEL", "RECORDED_FINGERPRINT", "RECORDED_HORIZON",
                  "RECORDED_TRAIN", "RECORDED_NUMERICS", "RECORDED_DEVICE",
                  "ENGINE", "CONDITIONED", "PLAN_ONLY")


def restore_engine(meta):
    """Put fno_predict back in the engine the run was made with: the default
    snapshots of a view, the timing report and the run facts all depend on
    it, and a fresh process has the seed engine.  meta.json names it for a
    schedule run (and keeps its picks); a run without the key is a
    fixed-step one."""
    P.ENGINE = meta.get("engine") or "seed"
    P.CONDITIONED = P.ENGINE == "schedule"
    P._PICKS.clear()
    P._PICKS.update(meta.get("schedule") or {})


def redraw(folder, out, partial=False):
    """Redraw one run's results folder from its records alone.

    ``folder`` is the rollout-cache folder of the run, with meta.json and
    records/ in it.  No model, solver or temporary cache is needed: the
    settings come from meta.json, the rollouts from the records, and the
    outputs are written by fno_predict's evaluate_all and write_case_outputs,
    as for a live run or a regeneration.  The film needs frames a record does
    not keep, so there is none, and the README says so.

    A folder with no readable meta.json, or with a planned case that is not
    complete (meta.json says so, or a record is gone), is refused before
    anything is written; ``partial`` draws just the complete cases instead.
    A case whose records disagree about their checkpoint or cache key fails
    on its own.  Returns the manifest, as build does.
    """
    meta = P.read_meta(folder)
    if meta is None:
        raise SystemExit(
            f"{folder} has no readable {P.META_FILE}, so it is not a folder "
            f"of records this can redraw (a run that was killed before its "
            f"first case, or one made with --no-records, leaves none)")
    status = P.records_status(folder, meta)
    left = sorted(n for n, ok in status.items() if not ok)
    if left and not partial:
        raise SystemExit(
            f"{folder} is incomplete: {len(left)} of {len(status)} planned "
            f"cases have no complete records ({', '.join(left)}).  Nothing "
            f"was written.  Let the run finish, or give --partial to draw "
            f"only the {len(status) - len(left)} complete case(s).")
    if not any(status.values()):
        raise SystemExit(f"{folder} holds no complete case; nothing to draw")
    extra = sorted(n for n in set(meta["cases"]) - set(status)
                   if not P.is_dropped(meta["cases"][n].get("lattice")))
    if extra:
        print(f"[records] note: {', '.join(extra)} are in {folder} but not "
              f"in the last run's plan, so they are not drawn", flush=True)
    views = tuple(float(h) for h in meta.get("views") or ())
    try:
        per_view = out.format(H="1") != out.format(H="2")
    except (KeyError, IndexError, ValueError):
        raise SystemExit(f"--out {out!r} may contain {{H}} and no other "
                         f"braces")
    if len(views) > 1 and not per_view:
        raise SystemExit(f"these records are of a run with views "
                         f"{', '.join(f'{h:g}' for h in views)}, one results "
                         f"folder each: put {{H}} in --out")
    dirs = [out.format(H=f"{h:g}") for h in views] if views else [out]
    for d in dirs:
        if (os.path.isdir(d) and set(os.listdir(d)) - {P.DIAG_FILE}
                and not os.path.exists(os.path.join(d, MANIFEST))):
            raise SystemExit(f"{d} already holds files and is not a results "
                             f"folder of this layout; choose a new --out")

    # the bookkeeping and sweep curves in extras/ describe the run folder
    # they were made for; a folder that has no manifest.json yet starts
    # from the records alone
    for d in dirs:
        if not os.path.exists(os.path.join(d, MANIFEST)):
            ex = P.extras_for(d, os.path.join(folder, EXTRAS_DIR),
                              len(views) > 1)
            for f in (BOOKKEEPING, CURVES_FILE):
                _discard(os.path.join(ex, f))

    home = records_home(folder, meta, status)
    saved = {n: getattr(P, n) for n in REDRAW_GLOBALS}
    flags, picks = dict(P.MODEL_FLAGS), dict(P._PICKS)
    missing = []
    try:
        restore_engine(meta)
        P.START_TIME, P.END_TIME = float(meta["start"]), float(meta["end"])
        # FNO's extras go into the records folder, beside what they extend;
        # meta.json and records/ are not touched
        P.EXTRAS_ROOT = os.path.join(folder, EXTRAS_DIR)
        P.KEEP_FILM = True       # records hold no frames: an old film stays
        P.RUN_SOURCE = f"make_submission, redrawn from records in {home}"
        P.RUN_CACHE = home
        P.RECORDED_DEVICE = recorded_device(folder, meta)
        P.RECORDED_CHECKPOINT = meta.get("checkpoint")
        P.SNAPSHOT_TIMES = tuple(float(t) for t in meta["snapshots"])
        P.SNAPSHOTS_EXPLICIT = bool(meta.get("snapshots_explicit"))
        P.LOG_TIME = meta.get("log_time", "auto")
        P.LATTICE = int(meta["lattice"])
        P.EVAL_MODE = meta["eval_mode"]
        P.SIZE_SWEEP = tuple(int(x) for x in meta["size_sweep"]
                             if int(x) not in P.DROPPED_SIZES)
        # the cases of the run, as its meta.json lists them: whatever the
        # present settings would plan, an older run's cases are drawn
        P.PLAN_ONLY = [
            (float(meta["cases"][n]["off"]), meta["cases"][n]["lattice"],
             tuple(int(x) for x in meta["cases"][n]["seeds"]))
            for n in (meta.get("planned_cases") or list(meta["cases"]))
            if not P.is_dropped(meta["cases"][n]["lattice"])]
        P.EVAL_SEEDS = tuple(int(x) for x in meta["eval_seeds"])
        P.SWEEP_SEEDS = tuple(int(x) for x in meta["sweep_seeds"])
        P.SEED_SET = fno_paths.seed_set_name(
            meta.get("seed_set", P.SEED_SET))
        P.VIEWS = views or None
        if views:
            P.RESULTS_TEMPLATE = out
            P.RESULTS_DIR = P.view_dir(max(views))
        else:
            P.RESULTS_DIR, P.RESULTS_TEMPLATE = out, None
        P.CACHE_END_TIME, P._RUN_SNAPSHOTS = None, ()
        # never a checkpoint file: the records name it, and are believed
        P.MODEL_PATH = ""
        P.RECORDED_FINGERPRINT = meta.get("fingerprint")
        P.RECORDED_NUMERICS = P.full_numerics(meta.get("numerics"))
        P.RECORDED_HORIZON = (meta["train_horizon"] if "train_horizon" in meta
                              else P.horizon_of(meta.get("checkpoint")))
        P.RECORDED_TRAIN = meta.get("train_setup")
        P.LABEL, P.AUTO_LABEL = meta.get("output_label") or P.LABEL, False
        P.WRITE_VIDEO = False
        info = meta.get("model") or {}
        ckpt = {"best": info["best"]} if info.get("best") else {}

        def from_records(model, device, off, lattice=None, seeds=None):
            name = os.path.basename(P.case_dir(off, lattice))
            if not status.get(name):
                missing.append(name)
                print(f"[miss] no complete records for {name} in {folder}",
                      flush=True)
                return None
            sd = tuple(int(x) for x in meta["cases"][name]["seeds"])
            got = [P.load_record(P.record_file(folder, name, s)) for s in sd]
            keys = {w["cache_key"] for _, w in got}
            prints = {w["fingerprint"] for _, w in got}
            if (len(keys) > 1 or len(prints) > 1
                    or [w["seed"] for _, w in got] != list(sd)
                    or (meta.get("fingerprint") and prints
                        != {meta["fingerprint"]})):
                raise ValueError(
                    f"the records of {name} do not belong together: cache "
                    f"keys {sorted(map(str, keys))}, checkpoints "
                    f"{sorted(map(str, prints))}, meta.json names "
                    f"{meta.get('fingerprint')}")
            print(f"\n--- {name}: {len(got)} seeds from records, cache key "
                  f"{next(iter(keys))} ---", flush=True)
            return P.emit_case([r for r, _ in got], sd, off, lattice,
                               key=next(iter(keys)))

        P.evaluate_all(RecordedModel(info), ckpt, None, case_fn=from_records,
                       resume=False, clean=False)
    finally:
        for n, v in saved.items():
            setattr(P, n, v)
        P.MODEL_FLAGS.clear()
        P.MODEL_FLAGS.update(flags)
        P._PICKS.clear()
        P._PICKS.update(picks)
    m = {}
    for d in dirs:
        ex = P.extras_for(d, os.path.join(folder, EXTRAS_DIR),
                          len(views) > 1)
        m = load_manifest(ex)
        if os.path.exists(os.path.join(ex, BOOKKEEPING)):
            m.update(source=f"make_submission, redrawn from records in "
                            f"{home}",
                     records={"folder": home,
                              "checkpoint": meta.get("checkpoint"),
                              "fingerprint": meta.get("fingerprint"),
                              "partial": bool(left), "left_out": left})
            write_json(os.path.join(ex, BOOKKEEPING), m)
        # the run folder's manifest.json says where it was drawn from, and
        # which complete cases a --partial redraw left out
        shown = _read_json(os.path.join(d, MANIFEST))
        if isinstance(shown, dict) and shown.get("cases") is not None:
            shown["records"] = {"folder": home,
                                "fingerprint": meta.get("fingerprint"),
                                "partial": bool(left), "left_out": left}
            write_json(os.path.join(d, MANIFEST), shown)
    m["missing"] = missing
    m["failed"] = list(m.get("failed_last_run") or ())
    return m


def run_status(folder):
    """(complete, why not) of a records folder: Work/rollout_cache/<label>
    is complete when its meta.json lists cases and every planned case has
    all its records (P.records_status)."""
    meta = P.read_meta(folder)
    if meta is None:
        return False, "no readable meta.json"
    status = P.records_status(folder, meta)
    left = sorted(n for n, ok in status.items() if not ok)
    if not status:
        return False, "meta.json plans no case"
    if left:
        return False, (f"{len(left)} of {len(status)} planned cases have no "
                       f"complete records ({', '.join(left)})")
    return True, ""


def discover(root=None):
    """{label: (folder, complete, why not)} for every folder of
    Work/rollout_cache with a meta.json, by name."""
    base = fno_paths.rollout_root(root)
    found = {}
    if os.path.isdir(base):
        for label in sorted(os.listdir(base)):
            folder = os.path.join(base, label)
            if (os.path.isdir(folder) and not label.startswith((".", "_"))
                    and os.path.isfile(P.meta_file(folder))):
                found[label] = (folder, *run_status(folder))
    return found


def results_target(label, meta, root=None):
    """Where a run's folder goes: Results/<label>, or for a run with views
    the template with {H}, so each view gets its own folder."""
    parsed = fno_paths.parse_label(label)
    if meta.get("views") and parsed and parsed["end"] is not None:
        return fno_paths.results_dir(
            fno_paths.run_template(parsed["train"], parsed["start"]), root)
    return fno_paths.results_dir(label, root)


def redraw_runs(only=None, root=None):
    """The default mode: redraw every complete run of the rollout cache from
    its records into Results/<label>/ (``only``: just that label), then the
    overview.  An incomplete run is skipped with the reason.  Returns the
    list of parts that failed."""
    found = discover(root)
    if only is not None:
        if only not in found:
            raise SystemExit(f"no records folder {only!r} with a meta.json "
                             f"in {fno_paths.rollout_root(root)}")
        if not found[only][1]:
            raise SystemExit(f"{only} is not complete: {found[only][2]}")
        found = {only: found[only]}
    failed = []
    for label, (folder, ok, why) in found.items():
        if not ok:
            print(f"[skip] {label}: {why}", flush=True)
            continue
        meta = P.read_meta(folder) or {}
        out = results_target(label, meta, root)
        print(f"\n[redraw] {label}: records in {folder} -> {out}", flush=True)
        try:
            m = redraw(folder, out)
            failed += [f"{label}: {f}" for f in m.get("failed", ())]
        except SystemExit as exc:
            failed.append(f"{label}: {exc}")
        except Exception as exc:
            failed.append(f"{label}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    write_results_index(fno_paths.results_root(root), root)
    return failed


def summarise(m, out):
    bar = "=" * 78
    print(f"\n{bar}\n  RESULTS  ({m.get('cache', out)})\n{bar}")
    print(f"  {'psi_0':>6} {'seeds':>6} {'mean R2':>9} {'n truth':>9} "
          f"{'n FNO':>7} {'psi>0':>8} {'lever':>7} {'shift':>8}")
    for key, e in m["compositions"].items():
        g, d = e["growth"], e["psi_pdf"]
        print(f"  {key:>6} {e['seeds']:>6} {e['mean_r2']:>9.4f} "
              f"{g['n_truth']:>9.3f} {g['n_fno']:>7.3f} "
              f"{d['frac_pos_true']:>8.4f} {d['lever_rule']:>7.4f} "
              f"{d['lever_shift']:>+8.4f}")
    if m["sizes"]:
        print(f"\n  {'box l':>8} {'seeds':>6} {'mean R2':>9}")
        for key, e in m["sizes"].items():
            print(f"  {key:>8} {e['seeds']:>6} {e['mean_r2']:>9.4f}")
    clipped = sum(e["psi_pdf"]["clipped_points"]
                  for e in m["compositions"].values())
    print(f"\n  points outside the plotted psi range: {clipped}")
    if m.get("missing"):
        print(f"  NOT DRAWN (not in the cache or records): "
              f"{', '.join(m['missing'])}")
    for f in m.get("failed", ()):
        print(f"  FAILED: {f}")
    files = glob.glob(os.path.join(out, "**", "*.png"), recursive=True)
    films = [f for ext in ("mp4", "gif")
             for f in glob.glob(os.path.join(out, "**", f"*.{ext}"),
                                recursive=True)]
    print(f"  {len(files)} figures and {len(films)} films in {out}/")
    print(bar, flush=True)


def critical_curve(cache):
    seeds, results, _ = load_case(cache, 0.0, BASE_LATTICE)
    if not results:
        return None
    t, c, _ = mean_curve(results)
    return t, c


def paired_deltas(cache_a, cache_b):
    """Mean-R2 difference per composition, on the seeds both runs share.

    The runs were evaluated on different seed counts, and the ten seeds the
    small run used are a prefix of the fifty the large ones used.  Comparing
    headline means therefore confounds the model change with the seed sample --
    which is exactly the trap that makes 0.9247 -> 0.9544 look like a training
    effect when almost all of it is sampling.  Pairing on shared seeds removes
    the sample, so what is left is the model.
    """
    rows = []
    for off in COMPOSITIONS:
        sa, ra, _ = load_case(cache_a, off, BASE_LATTICE)
        sb, rb, _ = load_case(cache_b, off, BASE_LATTICE)
        if not ra or not rb:
            continue
        ma = {s: float(np.nanmean(r["r2"])) for s, r in zip(sa, ra)}
        mb = {s: float(np.nanmean(r["r2"])) for s, r in zip(sb, rb)}
        shared = [s for s in sa if s in mb]
        if len(shared) < 3:
            continue
        a = np.array([ma[s] for s in shared])
        b = np.array([mb[s] for s in shared])
        d = b - a
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        rows.append({"off": off, "n": len(shared), "a": float(a.mean()),
                     "b": float(b.mean()), "delta": float(d.mean()),
                     "se": se, "t": float(d.mean() / se) if se else 0.0})
    return rows


def same_model(cache_a, cache_b):
    """Largest per-step R2 difference on every seed two caches share.

    Zero means the two were rolled out by the same checkpoint: the rollouts
    are deterministic, so one different weight would show up here.  Returns
    (max difference, compositions compared) or None when nothing is shared.
    """
    worst, n = 0.0, 0
    for off in COMPOSITIONS:
        sa, ra, _ = load_case(cache_a, off, BASE_LATTICE)
        sb, rb, _ = load_case(cache_b, off, BASE_LATTICE)
        mb = {s: r["r2"] for s, r in zip(sb, rb)}
        shared = [(r["r2"], mb[s]) for s, r in zip(sa, ra) if s in mb]
        if not shared:
            continue
        n += 1
        diffs = [float(np.nanmax(np.abs(x - y)))
                 for x, y in shared if x.shape == y.shape]
        worst = (float("inf") if len(diffs) < len(shared)
                 else max([worst] + diffs))
    return (worst, n) if n else None


def paired_verdict(rows):
    """The headline and the reading of a paired table, from its numbers.

    This used to be written in by hand, and went stale the day more seeds
    arrived: at ten seeds nothing crossed two standard errors, at fifty one
    composition did.  With k compositions tested at once, one |t| >= 2 turns
    up by chance with probability 1 - 0.9545^k even if nothing changed --
    about one in five for k = 5 -- so the text says that instead of calling
    it an effect.  Every sentence, including whether the sign is consistent,
    is worked out from the rows.
    """
    mean_d = sum(r["delta"] for r in rows) / len(rows)
    sig = [r for r in rows if abs(r["t"]) >= 2.0]
    chance = 1.0 - 0.9545 ** len(rows)
    pos = sum(1 for r in rows if r["delta"] > 0)
    neg = sum(1 for r in rows if r["delta"] < 0)
    if not sig:
        if neg == len(rows):
            sign = ("Every change is negative, though none reaches two "
                    "standard errors.")
        elif pos == len(rows):
            sign = ("Every change is positive, though none reaches two "
                    "standard errors.")
        else:
            sign = "The sign is not consistent across compositions."
        return "No, not measurably.", mean_d, [
            "No composition moves by as much as two standard errors.", sign,
            "The honest claim for a write-up is that accuracy is unchanged "
            "-- not that it improved."]

    ups = [r for r in sig if r["delta"] > 0]
    downs = [r for r in sig if r["delta"] < 0]
    desc = ", ".join(f"psi_0 = {r['off']:+.1f} ({r['delta']:+.4f}, "
                     f"t = {r['t']:+.2f})" for r in sig)
    verb = "moves" if len(sig) == 1 else "move"
    caution = (f"With {len(rows)} compositions tested, at least one result "
               f"that large turns up by chance about {chance:.0%} of the "
               f"time even if nothing changed, so read it as a lead to check "
               f"rather than as a finding.")
    if downs and not ups:
        head = "No -- and possibly slightly worse at some compositions."
        tail = "Nowhere does the longer training window help by a clear margin."
    elif ups and not downs:
        head = "Possibly, at some compositions."
        tail = (f"The largest gain is {max(r['delta'] for r in ups):+.4f}; "
                f"weigh it against that chance rate before calling it an "
                f"improvement.")
    else:
        head = "Mixed."
        tail = (f"It helps at {len(ups)} and hurts at {len(downs)} of the "
                f"compositions that moved, so there is no single direction "
                f"to report.")
    return head, mean_d, [
        f"{len(sig)} of {len(rows)} compositions {verb} by two standard "
        f"errors or more: {desc}.", caution, tail]


def figure_compare(built, path, notes=None):
    """The three runs on one axis, which is the only way to read them together.

    A and B share a model and differ only in how many seeds were rolled out, so
    the gap between them measures sampling noise rather than the model.  C is
    the model trained on the longer dataset, so the step from B to C is the
    effect of extending the training window -- the one comparison that says
    whether the extra data was worth the compute.
    """
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.6, 5.0), facecolor="white")
    cols = ("#8C6BB1", "#4292C6", "#C51B8A")
    facts = []
    bars_at = []

    for i, (run, m) in enumerate(built):
        curve = m.get("_critical_curve")
        if curve is None:
            continue
        t, c = curve
        n = m["compositions"].get("+0.00", {}).get("seeds", "?")
        a.plot(t, c, color=cols[i % len(cols)], lw=2.4, label=run["key"][0])
        facts.append(f"{run['key'][0]}: {run['title']}, {n} seeds, "
                     f"R2({t[-1]:.0f}) = {c[-1]:.3f}")
    a.set_xlabel(r"time  $t$", fontsize=11)
    a.set_ylabel(r"$R^2$ vs. Solver", fontsize=11)
    a.grid(alpha=0.3, ls=":")
    a.legend(loc="lower left", fontsize=8)
    a.set_title("(a)", loc="left", fontsize=11, fontweight="bold")

    lats = sorted({int(k) for _, m in built for k in m["sizes"]})
    width = 0.8 / max(len(built), 1)
    for i, (run, m) in enumerate(built):
        vals = [m["sizes"].get(str(l), {}).get("mean_r2", np.nan)
                for l in lats]
        pos = np.arange(len(lats)) + (i - (len(built) - 1) / 2.0) * width
        bars = b.bar(pos, vals, width=width, color=cols[i % len(cols)],
                     label=run["key"][0])
        bars_at.append(f"{run['key'][0]}: " + ", ".join(
            f"l = {l}: {v:.3f}" for l, v in zip(lats, vals)
            if np.isfinite(v)))
    b.set_xticks(np.arange(len(lats)))
    b.set_xticklabels([rf"$l$ = {l}" for l in lats])
    b.set_ylim(0.80, 1.02)
    b.set_ylabel(r"mean $R^2$ over the rollout", fontsize=11)
    b.grid(alpha=0.3, ls=":", axis="y")
    b.legend(fontsize=9, title="run", loc="lower left")
    b.set_title("(b)", loc="left", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fno_paths.save_figure(fig, path, FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")
    if notes is not None:
        notes[os.path.basename(path)] = [
            "the three evaluation runs compared: (a) mean R2 against time "
            "of the critical quench, 128 x 128 sites, one curve for each "
            "run; (b) "
            "mean R2 over the rollout of the critical quench by box size, "
            "one bar for each run"] + facts + ["(b) " + x for x in bars_at]


def write_text(path, text, shared=False):
    """Write a text file so that a failure leaves the old one whole and no
    temporary file behind.  ``shared`` is for a file other jobs write too (the
    index): the temporary then has a name of this process's own."""
    tmp = shared_tmp(path) if shared else path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        _discard(tmp)
        raise


def _read_json(path):
    """A JSON object from a file, or None.  Read only: a folder the index
    looks at is not its to move aside or repair, as load_manifest would."""
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _dict(obj):
    """obj if it is a dict, else an empty one: a manifest read from a folder
    that is not ours may hold anything."""
    return obj if isinstance(obj, dict) else {}


def _curve_of(case):
    """(t, mean, sd) arrays of a case's stored r2_curve, or None."""
    c = _dict(case.get("r2_curve")) if isinstance(case, dict) else {}
    if not c.get("t") or not c.get("mean"):
        return None
    nan = float("nan")
    return (np.array(c["t"], dtype=float),
            np.array([nan if x is None else x for x in c["mean"]],
                     dtype=float),
            np.array([0.0 if x is None else x for x in c.get("sd") or
                      [0.0] * len(c["t"])], dtype=float))


def run_size(label, work_root=None):
    """(channels, modes) the model of a run label was trained with when its
    label does not say (the default size): from the setup file beside its
    checkpoint (Work/models/<train label>/fno_ch_config.json), else the
    default of fno_train.  None for a label that is not one."""
    p = fno_paths.parse_label(label)
    if p is None:
        return None
    cfg = _read_json(fno_paths.config_path(p["train"], work_root))
    size = _dict(cfg).get("size")
    if isinstance(size, dict):
        c, m = size.get("channels"), size.get("modes")
        if isinstance(c, int) and isinstance(m, int):
            return c, m
    return P.fno_train.CHANNELS, P.fno_train.MODES


def case_groups(cases, aliases):
    """The cases of a manifest.json in the shape group_r2 takes: the cases
    at the base box as "compositions", the others as "sizes", and each alias
    (l128) as a size that is the case it names."""
    pseudo: dict = {"compositions": {}, "sizes": {}}
    for tok, c in cases.items():
        if not isinstance(c, dict):
            continue
        e = {"seeds": c.get("n_seeds"), "r2_at_times": c.get("r2_at_times"),
             "folder": tok}
        pseudo["compositions" if c.get("kind") == "composition"
               else "sizes"][tok] = e
    for alias, tok in _dict(aliases).items():
        if tok in pseudo["compositions"]:
            pseudo["sizes"][alias] = pseudo["compositions"][tok]
    return pseudo


def common_tokens(rows):
    """The case tokens every one of ``rows`` (run_row dicts) has, in the
    order of the first row's manifest."""
    if not rows:
        return []
    common = set.intersection(*(set(r["cases"]) for r in rows))
    return [t for t in rows[0]["cases"] if t in common]


def restrict_row(row, tokens):
    """A run_row with only the cases ``tokens``: its pooled R2 and curves
    then cover those cases alone.  Runs with different cases are compared on
    the cases they all have."""
    keep = set(tokens)
    out = dict(row)
    out["cases"] = {t: c for t, c in row["cases"].items() if t in keep}
    out["curves"] = {t: v for t, v in row["curves"].items() if t in keep}
    out["case_info"] = {t: v for t, v in row["case_info"].items()
                        if t in keep}
    out["r2"] = group_r2(case_groups(out["cases"], row["aliases"]))
    out["compositions"] = sum(1 for c in out["cases"].values()
                              if _dict(c).get("kind") == "composition")
    return out


def dropped_token(token):
    """True for the manifest token of a case of a dropped box size (l160,
    psi0p0.20_l192): an older run folder still holds them."""
    place = token_place(token) if isinstance(token, str) else None
    return place is not None and int(place[0][1:]) in P.DROPPED_SIZES


def run_row(folder, work_root=None):
    """What the overview says of one run folder with a manifest.json (the flat
    schema): label, checkpoint, horizons, case counts, the critical quench's
    numbers and curve, every case's curve, the files worth linking.  Read
    only; a manifest that cannot be read leaves the fields empty.
    ``work_root`` is where Work/models is (the smoke root, or None)."""
    label = os.path.basename(os.path.normpath(folder))
    m = _read_json(os.path.join(folder, MANIFEST)) or {}
    cases = {t: c for t, c in _dict(m.get("cases")).items()
             if not dropped_token(t)}
    parsed = fno_paths.parse_label(label)
    trained = m.get("train_t_end")
    if not isinstance(trained, (int, float)):
        trained = parsed["t_train"] if parsed else None
    crit = _dict(cases.get(case_token(P.CRITICAL_OFF)))
    aliases = {a: t for a, t in _dict(m.get("aliases")).items()
               if not dropped_token(a) and not dropped_token(t)}
    end = m.get("pdf_time")
    tag_size = fno_paths.split_tag(parsed["tag"])["size"] if parsed else None
    return {
        "label": label, "readable": bool(m),
        "fingerprints": [m["checkpoint_fingerprint"]]
        if m.get("checkpoint_fingerprint") else [],
        "trained": trained, "end": end if isinstance(end, (int, float))
        else None, "r2": group_r2(case_groups(cases, aliases)),
        "aliases": aliases,
        "compositions": sum(1 for c in cases.values()
                            if _dict(c).get("kind") == "composition"),
        "sizes": len(case_groups(cases, aliases)["sizes"]),
        "curve": _curve_of(crit), "seeds": crit.get("n_seeds"),
        "crit": crit, "cases": cases, "parsed": parsed,
        # every case's mean R2 curve and what the case is, by token, in the
        # order of the manifest
        "curves": {tok: _curve_of(c) for tok, c in cases.items()},
        "case_info": {tok: {"kind": _dict(c).get("kind"),
                            "psi0": _dict(c).get("psi0"),
                            "lattice": _dict(c).get("lattice"),
                            "seeds": _dict(c).get("n_seeds")}
                      for tok, c in cases.items()},
        "checkpoint": m.get("checkpoint"),
        "size": tag_size or run_size(label, work_root),
        "n_parameters": m.get("n_parameters"),
        "n_diverged": m.get("n_diverged"),
        "files": [f for f in ("report.txt", MANIFEST, COMPOSITION_FILE)
                  if os.path.isfile(os.path.join(folder, f))]
        + [d + "/" for d in SWEEP_DIRS
           if os.path.isdir(os.path.join(folder, d))]}


def run_order(row):
    """Runs by the horizon they were trained to, then rolled out to, as the
    labels say; a folder whose name is not a label comes after, by name."""
    p = row["parsed"]
    if p is None or p["end"] is None:
        return (1, 0.0, 0.0, row["label"])
    return (0, p["t_train"], p["end"], row["label"])


def overview_runs(rows):
    """The runs of comparison.png: the untagged ones (default size, training
    seed 0) when there are two or more, else every run."""
    plain = [r for r in rows if r["parsed"] and not r["parsed"]["tag"]]
    return plain if len(plain) >= 2 else rows


def size_ends(rows):
    """The training horizons that have a tagged Laplacian run, ascending:
    each gets a comparison_t<END>.png."""
    return sorted({r["parsed"]["t_train"] for r in rows
                   if r["parsed"] and r["parsed"]["lap"]
                   and r["parsed"]["tag"]})


def size_runs(rows, end):
    """Every Laplacian-output run of one training horizon, default size
    included."""
    return [r for r in rows if r["parsed"] and r["parsed"]["lap"]
            and r["parsed"]["t_train"] == end]


def comparison_name(end):
    return f"comparison_t{int(end)}.png"


def comparison_case_name(token):
    """comparison.png for one case of the runs, named by its composition:
    comparison_psi0m0.40.png ... (in comparisons/l<size>/, which says the
    box size)."""
    place = token_place(token)
    return f"comparison_{place[1] if place else token}.png"


def comparison_case_path(root, token, lattice=None):
    """Where the comparison of one case goes: root/comparisons/l256/
    comparison_psi0p0.00.png.  ``lattice`` is the case's box size as the
    manifests give it; without it the token says (token_place)."""
    place = token_place(token)
    size = (size_folder(lattice) if lattice is not None
            else place[0] if place else "")
    return os.path.join(root, COMPARISONS_DIR, size,
                        comparison_case_name(token))


def case_order(rows):
    """The case tokens of the rows, each once, in the order the first row
    that has it lists them (the manifest's: compositions, then box sizes)."""
    seen: dict = {}
    for r in rows:
        for tok in r["curves"]:
            seen.setdefault(tok, None)
    return list(seen)


def case_phrase(info):
    """How a figure title names a case, from its entry in run_row's
    case_info: "off-critical quench, psi0 = -0.4, l = 128"."""
    off, lat = info.get("psi0"), info.get("lattice")
    if not isinstance(off, (int, float)):
        return "case"
    return (f"{comp_label(off)}, {comp_math(off)}"
            + (f", $l$ = {lat}" if isinstance(lat, int) else ""))


def _save_shared(fig, path):
    tmp = shared_tmp(path)
    try:
        fno_paths.save_figure(fig, tmp, FIG_DPI, bbox_inches="tight")
        os.replace(tmp, path)
    finally:
        plt.close(fig)
        _discard(tmp)
    print(f"[saved] {path}")


# --- the model's training and validation curves (training_curves.png) -------
# One colour per series, from the validated categorical palette (checked with
# the data-viz validator, adjacent pairs; RUN_COLORS are not used here because
# the blue next to the magenta fails its colour-blind check): the loss pair
# of panel (a), the rollout pair of (b), and the survival time of (c).
TRAIN_COLORS = {"train": "#2a78d6", "val1": "#eb6834", "score": "#1baf7a",
                "in_range": "#6250d6", "survival": "#e34948"}
INK, MUTED = "#0b0b0b", "#52514e"
# The longest window the validation rollout runs over (fno_train.VAL_T_END);
# the in-range accuracy covers twice the training horizon, up to this.
VAL_WINDOW = 10000.0


def read_history(path):
    """The entries of a training history file (one per epoch), by epoch, or
    None when it cannot be read or holds none."""
    try:
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(rows, list):
        return None
    rows = [r for r in rows if isinstance(r, dict)
            and isinstance(r.get("epoch"), (int, float))
            and not isinstance(r.get("epoch"), bool)]
    return sorted(rows, key=lambda r: r["epoch"]) or None


def history_file(label, checkpoint=None, work_root=None):
    """The training history of the model a run was rolled out from: the file
    beside its checkpoint (fno_ch.pt -> fno_ch_history.json), else the one of
    its train label in Work/models/.  None when neither is there."""
    found = []
    if isinstance(checkpoint, str) and checkpoint:
        found.append(os.path.splitext(checkpoint)[0] + "_history.json")
    p = fno_paths.parse_label(label)
    if p is not None:
        found.append(fno_paths.history_path(p["train"], work_root))
    return next((f for f in found if os.path.isfile(f)), None)


def report_selection(folder):
    """The "checkpoint selection" block of a run folder's report.txt
    (section 1) as {key: text}, {} when it has none: a run from before
    manifest.json held the selection says there which epoch it was."""
    out: dict = {}
    try:
        with open(os.path.join(folder, "report.txt"), encoding="utf-8",
                  errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return out
    inside = False
    for ln in lines:
        if not inside:
            inside = ln.strip() == "checkpoint selection"
            continue
        m = re.fullmatch(r"    (\S+)\s+(\S.*?)\s*", ln)
        if m is None:
            break
        out[m[1]] = m[2]
    return out


def selected_checkpoint(folder, manifest):
    """{"epoch", "score", "survival"} of the checkpoint a run was rolled out
    from, as the run recorded it: manifest.json's checkpoint_best, else the
    "checkpoint selection" block of report.txt.  None when neither says which
    epoch it was."""
    for found in (_dict(manifest).get("checkpoint_best"),
                  report_selection(folder)):
        src = _dict(found)
        try:
            epoch = int(float(src["epoch"]))
        except (KeyError, TypeError, ValueError):
            continue
        got: dict = {"epoch": epoch}
        for k in ("score", "survival"):
            try:
                got[k] = float(src[k])
            except (KeyError, TypeError, ValueError):
                pass
        return got
    return None


def chosen_row(rows, best):
    """The history entry of the selected epoch, or None when the history does
    not agree with the selection: that epoch was not validated in it, or its
    score is another (the history of some other training)."""
    epoch = _dict(best).get("epoch")
    if not isinstance(epoch, (int, float)):
        return None
    row = next((r for r in rows if r["epoch"] == epoch
                and not r.get("diverged")
                and isinstance(r.get("score"), (int, float))), None)
    score = _dict(best).get("score")
    if (row is not None and isinstance(score, (int, float))
            and abs(row["score"] - score) > 1e-6 * max(1.0, abs(score))):
        return None
    return row


def figure_training_curves(rows, path, name=DISPLAY_NAME, train_label="",
                           n_parameters=None, chosen=None, horizon=None,
                           notes=None):
    """The model's training and validation curves against epoch, in three
    panels that share the epoch axis: (a) the training loss and the one-step
    validation loss, (b) the rollout validation (score and in-range R2, at
    the epochs it was run), (c) the survival time of the validation rollouts
    with their range over the seeds.  The epoch the checkpoint was selected
    at (``chosen``, its history entry) is the vertical line, with a dot on
    every curve; the dotted lines are where the unroll depth steps up.  An
    epoch the training flagged as diverged is left out of the curves (its loss
    is a blow-up that was rolled back) and marked with a cross.  The legends
    are short; the parameter count, the selected epoch's scores, the unroll
    steps and the diverged epochs go into ``notes``."""
    col = TRAIN_COLORS
    good = [r for r in rows if not r.get("diverged")]
    bad = [r for r in rows if r.get("diverged")]
    last = max(r["epoch"] for r in rows)

    def series(key, positive=False):
        pts = [(r["epoch"], float(r[key])) for r in good
               if isinstance(r.get(key), (int, float))
               and not isinstance(r.get(key), bool)
               and np.isfinite(r[key]) and (r[key] > 0 or not positive)]
        return (np.array([p[0] for p in pts], dtype=float),
                np.array([p[1] for p in pts], dtype=float))

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9.6, 9.6),
                             facecolor="white",
                             gridspec_kw={"height_ratios": [1.15, 1.0, 0.85],
                                          "hspace": 0.36})
    ax_loss, ax_roll, ax_surv = axes
    keys: dict = {id(ax): [] for ax in axes}   # each panel's legend entries
    shown: dict = {}

    def line(ax, key, label, positive=False):
        x, y = series(key, positive)
        if not x.size:
            return
        shown[key] = y
        h, = ax.plot(x, y, color=col[key], lw=1.9, label=label, zorder=3)
        keys[id(ax)].append(h)
        v = None if chosen is None else chosen.get(key)
        if (chosen is not None and isinstance(v, (int, float))
                and np.isfinite(v) and (v > 0 or not positive)):
            ax.plot([chosen["epoch"]], [v], ls="none", marker="o", ms=6.5,
                    color=col[key], mfc=col[key], mec="white", mew=1.4,
                    zorder=6)

    line(ax_loss, "train", "training loss", True)
    line(ax_loss, "val1", "validation loss", True)
    ax_loss.set_yscale("log")
    line(ax_roll, "score", "rollout score")
    window = min(2.0 * horizon, VAL_WINDOW) if horizon else None
    line(ax_roll, "in_range", "in-range $R^2$")
    line(ax_surv, "survival", "survival time")
    spread = [(r["epoch"], min(v), max(v)) for r in good
              for v in [[float(x) for x in r["per_seed"]
                         if isinstance(x, (int, float)) and np.isfinite(x)]
                        if isinstance(r.get("per_seed"), list) else []] if v]
    if spread:
        ax_surv.fill_between([s[0] for s in spread], [s[1] for s in spread],
                             [s[2] for s in spread], color=col["survival"],
                             alpha=0.14, lw=0, zorder=2)
        keys[id(ax_surv)].append(Patch(facecolor=col["survival"], alpha=0.14,
                                       label="seed range"))
    if chosen is not None:
        for ax in axes:
            ax.axvline(chosen["epoch"], color=INK, lw=1.0, alpha=0.75,
                       zorder=2)
        keys[id(ax_loss)].append(Line2D(
            [], [], color=INK, lw=1.0, alpha=0.75, marker="o", ms=5,
            mfc=INK, mec="white",
            label="selected epoch"))
    eps, blown = "", []
    if bad:
        # the cross sits on the training curve where the epoch was left out
        # (the line runs straight over the gap, in log loss)
        tx, ty = series("train", True)
        for r in bad:
            at = r["epoch"]
            if tx.size:
                y = float(np.exp(np.interp(at, tx, np.log(ty))))
                ax_loss.plot([at], [y], ls="none", marker="x", ms=9, mew=2.4,
                             color=INK, zorder=7)
        eps = ", ".join(f"{int(r['epoch'])}" for r in bad)
        blown = [r["train"] for r in bad
                 if isinstance(r.get("train"), (int, float))]
        keys[id(ax_loss)].append(Line2D(
            [], [], ls="none", marker="x", ms=7, mew=2.0, color=INK,
            label="diverged epoch"))
    depth = [(r["epoch"], int(r["unroll"])) for r in good
             if isinstance(r.get("unroll"), (int, float))]
    steps = [(e, k) for (e, k), (_, before) in zip(depth[1:], depth)
             if k != before]
    for e, k in steps:
        for ax in axes:
            ax.axvline(e - 0.5, color="0.55", ls=":", lw=1.2, zorder=1)

    lo = min([float(v.min()) for k, v in shown.items()
              if k in ("train", "val1")] or [1e-6])
    hi = max([float(v.max()) for k, v in shown.items()
              if k in ("train", "val1")] or [1.0])
    ax_loss.set_ylim(lo / 1.8, hi * 2.6)
    rolls = np.concatenate([shown[k] for k in ("score", "in_range")
                            if k in shown] or [np.array([0.0])])
    floor = float(rolls.min())
    ax_roll.set_ylim(floor - 0.06 * (1.0 - floor), 1.03)
    top = max([float(shown["survival"].max()) if "survival" in shown else 0.0]
              + [s[2] for s in spread] + [1.0])
    ax_surv.set_ylim(0.0, 1.06 * top)
    ax_surv.set_xlim(0.5, last + 0.5)
    ax_surv.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    for ax, title in zip(axes, ("(a)", "(b)", "(c)")):
        ax.grid(alpha=0.3, ls=":")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax_loss.set_ylabel("loss", fontsize=11)
    ax_roll.set_ylabel(r"$R^2$ vs. Solver", fontsize=11)
    ax_surv.set_ylabel("survival time  $t$", fontsize=11)
    ax_surv.set_xlabel("epoch", fontsize=11)
    for ax in axes:
        # each panel's key sits above it, right of its title
        ax.legend(handles=keys[id(ax)], loc="lower right",
                  bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2, fontsize=8.5,
                  ncol=2 if len(keys[id(ax)]) > 3 else 3, frameon=False,
                  columnspacing=1.3, handlelength=1.8, handletextpad=0.5)

    facts = []
    if n_parameters:
        facts.append(f"{int(n_parameters):,} parameters")
    if chosen is not None:
        stats = [f"rollout score {chosen['score']:.3f}"]
        if isinstance(chosen.get("in_range"), (int, float)):
            stats.append(f"in-range R2 {chosen['in_range']:.3f}")
        if isinstance(chosen.get("survival"), (int, float)):
            stats.append(f"survives to t = {chosen['survival']:.0f}")
        facts.append(f"selected epoch {int(chosen['epoch'])} of "
                     f"{int(last)}: " + ", ".join(stats))
    else:
        facts.append(f"{int(last)} epochs, the selected one is not marked")
    fig.subplots_adjust(left=0.09, right=0.975, top=0.93, bottom=0.065)
    _save_shared(fig, path)
    if notes is not None:
        lines = [f"training curves of {train_label} against epoch, three "
                 "panels sharing the epoch axis: (a) training loss and "
                 "one-step validation loss, (b) rollout score and in-range "
                 "R2 of the validation rollouts, (c) survival time with its "
                 "range over the validation seeds"]
        lines += facts
        if window:
            lines.append(f"in-range R2 covers t <= {window:g}")
        lines.append("survival time: the time R2 first falls below "
                     f"{R2_THRESHOLD:.2f}")
        if steps:
            lines.append("dotted vertical lines: the unroll depth steps up "
                         "at " + ", ".join(f"epoch {e} (unroll {k})"
                                           for e, k in steps))
        if bad:
            lines.append(f"diverged epoch {eps}, loss "
                         + (f"{max(blown):.2g}" if blown else "not recorded")
                         + ", not drawn, rolled back (cross on the loss "
                         "curve)")
        lines.append("training loss uses k unrolled steps; the vertical "
                     "line and dots mark the selected epoch")
        notes[os.path.basename(path)] = lines


def draw_training_curves(folder, label, checkpoint, best, n_parameters,
                         work_root=None, name=DISPLAY_NAME):
    """training_curves.png in a run folder, from the history of the model the
    run was rolled out from (history_file).  True when it was drawn.  A model
    without a history gets no figure, and a figure that fails is reported and
    does not fail the run: nothing else depends on it.  Nothing is drawn
    while WRITE_TRAINING is off."""
    if not WRITE_TRAINING:
        return False
    try:
        hist = history_file(label, checkpoint, work_root)
        if hist is None:
            print(f"[training] {label}: no history file "
                  + (f"beside {checkpoint} or " if checkpoint else "")
                  + f"in {fno_paths.models_dir(work_root)}, so no "
                  f"{TRAINING_FIG}", flush=True)
            return False
        rows = read_history(hist)
        if rows is None:
            print(f"[training] {label}: {hist} holds no epochs, so no "
                  f"{TRAINING_FIG}", flush=True)
            return False
        chosen = chosen_row(rows, best)
        if best and chosen is None:
            print(f"[training] {label}: {hist} does not agree with the "
                  f"selected epoch {best.get('epoch')} (it was not validated "
                  f"there, or with another score), so no epoch is marked",
                  flush=True)
        parsed = fno_paths.parse_label(label)
        notes: dict = {}
        figure_training_curves(
            rows, os.path.join(folder, TRAINING_FIG), name,
            parsed["train"] if parsed else label, n_parameters, chosen,
            parsed["t_train"] if parsed else None, notes)
        write_figure_notes(folder, notes)
        return True
    except Exception as exc:
        print(f"[training] {label}: {TRAINING_FIG} not drawn "
              f"({type(exc).__name__}: {exc})", flush=True)
        return False


def draw_row_training(folder, row, work_root=None):
    """Draw the training curves of a run folder the overview looks at, and
    list the figure in the folder's manifest.json (files) when it was not
    there.  The selection comes from the manifest, or from report.txt for a
    run from before the manifest held it.  A manifest that cannot be read
    leaves the folder alone.  True when the figure was drawn."""
    path = os.path.join(folder, MANIFEST)
    m = _read_json(path) if WRITE_TRAINING else None
    if m is None:
        return False
    clear_shared_tmp(folder)
    name = (f"{DISPLAY_NAME} [hard constraints off]"
            if _dict(row["parsed"]).get("nohard") else DISPLAY_NAME)
    if not draw_training_curves(folder, row["label"], m.get("checkpoint"),
                                selected_checkpoint(folder, m),
                                m.get("n_parameters"), work_root, name):
        return False
    files = m.get("files")
    if isinstance(files, list) and TRAINING_FIG not in files:
        m["files"] = sorted(set(files) | {TRAINING_FIG})
        write_json(path, m)
    return True


def figure_run_comparison(rows, path, name=DISPLAY_NAME, case=None,
                          horizons=None, notes=None):
    """The mean R2 against time of one case for every run in ``rows`` (those
    with a curve), on one log t axis.  ``case`` None is the critical quench
    (comparison.png, the runs' ``curve``); a case token draws that case from
    the runs' ``curves`` (comparison_<case>.png), the title saying which.
    Colour is the horizon a run was trained to, with a dotted vertical line
    of that colour; solid is a Laplacian-output run (_lap), dashed the base
    model, dash-dot an ablation.  ``horizons`` are the horizons of all the
    runs compared, so that a horizon keeps its colour in every file even
    where a run has no curve for the case.  The legend is below the axes and
    names the runs; the training horizons, seed counts and R2 at the end go
    into ``notes``."""
    colours = RUN_COLORS
    if horizons is None:
        horizons = sorted({r["trained"] for r in rows if r["trained"]})
    what = "critical quench"
    if case is not None:
        what = case_phrase(next(r["case_info"][case] for r in rows
                                if case in r["case_info"]))
    fig, ax = plt.subplots(figsize=(9.6, 5.6), facecolor="white")
    lows = []
    facts = []
    for r in rows:
        if case is None:
            t, c, _ = r["curve"]
            n_seeds = r["seeds"]
        else:
            t, c, _ = r["curves"][case]
            n_seeds = r["case_info"][case].get("seeds")
        p = r["parsed"] or {}
        colour = (colours[horizons.index(r["trained"]) % len(colours)]
                  if r["trained"] in horizons else "0.4")
        style = "-." if p.get("nohard") else "-" if p.get("lap") else "--"
        trained = (f"trained to $t$ = {r['trained']:g}" if r["trained"]
                   else "training horizon unknown")
        seeds = f", {n_seeds} seeds" if n_seeds else ""
        ax.plot(t, c, lw=2.0, color=colour, ls=style, label=r["label"])
        facts.append(plain_text(f"{r['label']}: {trained}{seeds}, "
                                f"R2({t[-1]:.0f}) = {c[-1]:.3f}"))
        if r["trained"] and t[0] < r["trained"] < t[-1]:
            ax.axvline(r["trained"], color=colour, ls=":", lw=1.3)
        lows.append(float(np.nanmin(c)))
    low = min([0.0] + lows)
    ax.set_xscale("log")
    ax.set_ylim(low - 0.05 * (1.02 - low), 1.02)
    ax.set_xlabel(r"time  $t$  (log)", fontsize=12)
    ax.set_ylabel(r"mean $R^2$ vs. Solver", fontsize=12)
    ax.grid(alpha=0.3, ls=":", which="both")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8.5)
    fig.tight_layout()
    _save_shared(fig, path)
    if notes is not None:
        notes[os.path.basename(path)] = [
            f"mean R2 against time of the {plain_text(what)}, one curve for "
            "each run, log t axis",
            "colour: the horizon a run was trained to, with a dotted "
            "vertical line of that colour; solid: Laplacian-output run, "
            "dashed: base model, dash-dot: ablation"] + facts


# --- the model sizes of one training horizon (comparison_t<END>.png) --------
# One-hue ramp for the width, light to dark (checked with the validator's
# ordinal mode), set at the widths of the sweep.  A width between two of them
# is mixed from its neighbours, so a width has the same colour whichever
# widths a figure shows.  The default size is black.
SIZE_WIDTHS = (16, 24, 32, 38, 45, 64)
SIZE_RAMP = ("#86b6ef", "#609fe9", "#3886e4", "#276fc8", "#1b5aa8", "#134687")
SEED_STYLES = ("-", "--", ":", "-.")


def width_colour(width):
    """The colour of a model width (see SIZE_RAMP)."""
    rgb = np.array([to_rgb(c) for c in SIZE_RAMP])
    red, green, blue = (float(np.interp(width, SIZE_WIDTHS, rgb[:, i]))
                        for i in range(3))
    return to_hex((red, green, blue))


def pooled_curve(row):
    """(t, mean R2, number of cases) of a run pooled over its cases: the mean
    of the cases' stored mean curves, each case weighted by its seeds, on the
    union of their time grids (a curve is read between its samples on a log t
    axis) over the window every case covers.  None when the run has no
    curve.  At the last time this is the mean over every seed of every case,
    as the index's pooled R2 is."""
    parts = []
    for tok, cv in row["curves"].items():
        if cv is None:
            continue
        w = _dict(row["case_info"].get(tok)).get("seeds")
        t, m = cv[0], cv[1]
        keep = t > 0
        if keep.sum() > 1:
            parts.append((t[keep], m[keep],
                          float(w) if isinstance(w, (int, float)) and w > 0
                          else 1.0))
    if not parts:
        return None
    lo, hi = max(p[0][0] for p in parts), min(p[0][-1] for p in parts)
    grid = np.unique(np.concatenate([p[0] for p in parts]))
    grid = grid[(grid >= lo) & (grid <= hi)]
    if grid.size < 2:
        return None
    total = np.zeros(grid.size)
    for t, m, w in parts:
        total += w * np.interp(np.log(grid), np.log(t), m)
    return grid, total / sum(p[2] for p in parts), len(parts)


def param_count(n):
    """A model's parameter count in a few characters: 17.8M, 425k."""
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}k"
    return str(n)


def figure_size_comparison(rows, end, path, name=DISPLAY_NAME,
                           notes=None):
    """All Laplacian-output runs trained to ``end`` (default size included),
    pooled over their cases: one panel for each number of Fourier modes
    (sharing the R2 axis), a curve for each width coloured by the width
    (SIZE_RAMP), the default size black, another training seed dashed or
    dotted.  The legend of each panel is below it, with the pooled R2 at the
    end of each curve.  The figure has short legends only; the pooled R2 at
    the end, the parameter counts, the case mix and the colour key go into
    ``notes``."""
    default = (P.fno_train.CHANNELS, P.fno_train.MODES)

    def tag_of(r):
        return fno_paths.split_tag(r["parsed"]["tag"])

    def size_of(r):
        return tag_of(r)["size"] or r.get("size") or default

    pooled = [(r, pooled_curve(r)) for r in rows]
    pooled = [(r, pc) for r, pc in pooled if pc is not None]
    if not pooled:
        raise ValueError("no run of this training horizon has a curve")
    cases = max(pc[2] for _, pc in pooled)
    modes = sorted({size_of(r)[1] for r, _ in pooled})
    seeds = sorted({tag_of(r)["seed"] for r, _ in pooled})
    fullest = max(pooled, key=lambda x: x[1][2])[0]["case_info"].values()
    mix = ", ".join(f"{n} {what}{'s' if n != 1 else ''}" for n, what in (
        (sum(i.get("kind") == "composition" for i in fullest), "composition"),
        (sum(i.get("kind") == "box" for i in fullest), "box size")) if n)
    n_seeds = {r["case_info"][tok].get("seeds") for r, _ in pooled
               for tok in r["curves"]}
    t_lo = min(float(pc[0][0]) for _, pc in pooled)
    t_hi = max(float(pc[0][-1]) for _, pc in pooled)
    low = min(float(np.nanmin(pc[1])) for _, pc in pooled)
    floor = min(0.7, np.floor((low - 0.02) / 0.05) * 0.05
                if np.isfinite(low) else 0.7)
    fig, axes = plt.subplots(1, len(modes), sharey=True, squeeze=False,
                             figsize=(4.8 * len(modes) + 0.7, 4.8),
                             facecolor="white")
    facts = []
    for pi, (ax, m) in enumerate(zip(axes[0], modes)):
        facts.append(f"({'abcdefgh'[pi]}) {m} Fourier modes"
                     if len(modes) > 1 else f"{m} Fourier modes")
        mine = sorted(((r, pc) for r, pc in pooled if size_of(r)[1] == m),
                      key=lambda x: (size_of(x[0])[0], tag_of(x[0])["seed"]))
        for r, (t, c, k) in mine:
            width, sd = size_of(r)[0], tag_of(r)["seed"]
            plain = tag_of(r)["size"] is None
            label = (f"{width} ch" + (" (default)" if plain else "")
                     + (f", seed {sd}" if sd else ""))
            facts.append(
                f"{m} modes, {label}: "
                + (f"{param_count(r['n_parameters'])} parameters, "
                   if r.get("n_parameters") else "")
                + f"pooled R2({t[-1]:.0f}) = {c[-1]:.3f}"
                + (f" ({k} of {cases} cases)" if k < cases else ""))
            # the default is the reference: black, a little heavier, on top
            ax.plot(t, c, lw=2.3 if plain else 1.7,
                    color=INK if plain else width_colour(width),
                    ls=SEED_STYLES[seeds.index(sd) % len(SEED_STYLES)],
                    zorder=4 if plain else 3, label=label)
        ax.axvline(end, color="0.3", ls=":", lw=1.2, zorder=1,
                   label="training horizon")
        ax.set_xscale("log")
        ax.set_xlim(t_lo / 1.1, t_hi * 1.1)
        ax.set_ylim(floor, 1.01)
        ax.grid(alpha=0.3, ls=":", which="both")
        if len(modes) > 1:
            ax.set_title(f"({'abcdefgh'[pi]})", loc="left", fontsize=11,
                         fontweight="bold")
        ax.set_xlabel(r"time  $t$  (log)", fontsize=11)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17),
                  fontsize=8.5, frameon=False, handlelength=2.4,
                  borderaxespad=0.0)
    axes[0][0].set_ylabel(r"mean $R^2$ vs. Solver, pooled", fontsize=11)
    seeds_txt = (f", {next(iter(n_seeds))} seeds each"
                 if len(n_seeds) == 1 and None not in n_seeds else "")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.30,
                        wspace=0.08)
    _save_shared(fig, path)
    if notes is not None:
        notes[os.path.basename(path)] = [
            f"mean R2 against time of the model sizes trained to t = "
            f"{end:g}, pooled over the {cases} cases ({mix}){seeds_txt}, "
            "log t axis",
            f"colour: width in channels; black: the default size "
            f"({default[0]} channels, {default[1]} modes)"
            + ("; dashed: another training seed" if len(seeds) > 1 else "")
            + f"; dotted line: training horizon t = {end:g}"] + facts


# --- the model-size and Laplacian figures of the top level -------------------
# The Laplacian runs of this training horizon with the default number of
# Fourier modes are the ones r2_vs_model_size.png compares.
SIZE_END = 500.0
SIZE_MODES = 24


def run_modes(row):
    """The number of Fourier modes of a run's model."""
    tag = fno_paths.split_tag(row["parsed"]["tag"])
    size = tag["size"] or row.get("size") or (P.fno_train.CHANNELS,
                                              P.fno_train.MODES)
    return size[1]


def pooled_r2_end(row):
    """The pooled R2 at the last reporting time (R2_REPORT_TIMES) of a run,
    from its run_row; None when it has none."""
    at = _dict(_dict(_dict(row.get("r2")).get("pooled")).get("at"))
    got = _dict(at.get(format(R2_REPORT_TIMES[-1], "g")))
    return got.get("mean")


def model_size_runs(rows):
    """The runs of r2_vs_model_size.png: Laplacian output, trained to
    SIZE_END, SIZE_MODES modes, with a parameter count and a pooled R2 at
    the last reporting time; by parameter count."""
    got = [r for r in rows if r["parsed"] and r["parsed"]["lap"]
           and r["parsed"]["t_train"] == SIZE_END
           and not r["parsed"].get("nohard") and r.get("n_parameters")
           and pooled_r2_end(r) is not None and run_modes(r) == SIZE_MODES]
    return sorted(got, key=lambda r: r["n_parameters"])


def figure_r2_vs_model_size(rows, path, notes=None):
    """The pooled R2 at the last reporting time against the parameter count
    (log axis) of the runs ``model_size_runs`` picked; the points of the
    second training seed have their own marker.  The table of runs goes into
    ``notes``."""
    seed_of = {r["label"]: fno_paths.split_tag(r["parsed"]["tag"])["seed"]
               for r in rows}
    seeds = sorted({seed_of[r["label"]] for r in rows})
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor="white")
    for i, sd in enumerate(seeds):
        mine = [r for r in rows if seed_of[r["label"]] == sd]
        x = [r["n_parameters"] for r in mine]
        y = [float(pooled_r2_end(r) or np.nan) for r in mine]
        ax.plot(x, y, color=RUN_COLORS[i % len(RUN_COLORS)],
                marker=SWEEP_MARKS[i % len(SWEEP_MARKS)], ms=7,
                lw=1.6 if i == 0 else 0, mfc="none" if i else None,
                mew=1.6, label=f"training seed {sd}")
    ax.set_xscale("log")
    ticks = [t for t in (3e6, 5e6, 1e7, 2e7, 5e7, 1e8)
             if 0.8 * min(r["n_parameters"] for r in rows) <= t
             <= 1.25 * max(r["n_parameters"] for r in rows)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([param_count(t) for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("parameters", fontsize=12)
    ax.set_ylabel(rf"pooled $R^2(t = {R2_REPORT_TIMES[-1]:g})$", fontsize=12)
    ax.grid(alpha=0.3, which="both", ls=":")
    if len(seeds) > 1:
        ax.legend(fontsize=10)
    fig.tight_layout()
    _save_shared(fig, path)
    if notes is not None:
        notes[os.path.basename(path)] = [
            f"pooled R2 at t = {R2_REPORT_TIMES[-1]:g} (mean over every seed "
            "of every case) against the number of parameters, log axis, for "
            f"the Laplacian-output runs trained to t = {SIZE_END:g} with "
            f"{SIZE_MODES} Fourier modes; the second training seed has its "
            "own marker (hollow squares)"] + [
            f"{r['label']}: {int(r['n_parameters']):,} parameters, "
            f"{r['size'][0] if r.get('size') else '?'} channels, training "
            f"seed {seed_of[r['label']]}, pooled R2 = "
            f"{pooled_r2_end(r):.3f}" for r in rows]


def laplacian_pair(rows):
    """(Laplacian on, off): the plain runs trained to SIZE_END, with and
    without the Laplacian output, each with a pooled curve; None when
    either is missing."""
    def pick(lap):
        got = [r for r in rows if r["parsed"] and not r["parsed"]["tag"]
               and r["parsed"]["lap"] == lap
               and r["parsed"]["t_train"] == SIZE_END
               and not r["parsed"].get("nohard")
               and pooled_curve(r) is not None]
        return got[-1] if got else None
    on, off = pick(True), pick(False)
    return None if on is None or off is None else (on, off)


def figure_laplacian_on_off(pair, path, notes=None):
    """The pooled R2(t) of the plain run with the Laplacian output and the
    one without (pooled_curve: the mean over the cases, each on the union of
    the time grids, read between samples on a log t axis), log t, with the
    training horizon.  The values at the reporting times go into ``notes``."""
    fig, ax = plt.subplots(figsize=(7.4, 4.8), facecolor="white")
    facts = []
    lows = []
    t_lo, t_hi = np.inf, 0.0
    for r, label, colour, style in zip(
            pair, ("Laplacian on", "Laplacian off"),
            (RUN_COLORS[0], RUN_COLORS[1]), ("-", "--")):
        curve = pooled_curve(r)
        if curve is None:
            continue
        t, c, k = curve
        t_lo, t_hi = min(t_lo, float(t[0])), max(t_hi, float(t[-1]))
        ax.plot(t, c, lw=2.0, color=colour, ls=style, label=label)
        lows.append(float(np.nanmin(c)))
        at = [f"t = {want:g}: {float(np.interp(np.log(want), np.log(t), c)):.3f}"
              for want in R2_REPORT_TIMES if t[0] <= want <= t[-1]]
        facts.append(f"{label} ({r['label']}, {k} cases): pooled R2 at "
                     + ", ".join(at))
    horizon = pair[0]["trained"]
    if horizon and t_lo < horizon < t_hi:
        ax.axvline(horizon, color="0.3", ls=":", lw=1.3,
                   label="training horizon")
    low = min([0.0] + lows)
    ax.set_xscale("log")
    ax.set_ylim(low - 0.05 * (1.02 - low), 1.02)
    ax.set_xlabel(r"time  $t$  (log)", fontsize=12)
    ax.set_ylabel(r"pooled $R^2$", fontsize=12)
    ax.grid(alpha=0.3, ls=":", which="both")
    ax.legend(loc="lower left", fontsize=10)
    fig.tight_layout()
    _save_shared(fig, path)
    if notes is not None:
        notes[os.path.basename(path)] = [
            f"pooled R2 against time of the plain runs trained to t = "
            f"{SIZE_END:g} with the Laplacian output (solid) and without "
            "(dashed), log t axis; the mean over the cases both runs have, "
            "each case's mean curve read on the union of the time grids "
            "(interpolated on log "
            "t); dotted line: training horizon "
            f"t = {horizon:g}"] + facts


def draw_case_comparisons(root, plain, name=DISPLAY_NAME):
    """comparisons/l<size>/comparison_<composition>.png for every case but
    the critical quench at the base box that two or more of the plain runs
    have; each folder gets a figure_notes.txt of its own.  The critical
    quench at l = 128 is comparison.png itself, so it gets no second file.
    A figure that cannot be drawn is reported and left out.  Returns the
    tokens drawn."""
    have_of = {tok: [r for r in plain if r["curves"].get(tok) is not None]
               for tok in case_order(plain)}
    crit = case_token(P.CRITICAL_OFF)
    shared = [tok for tok, have in have_of.items()
              if len(have) >= 2 and tok != crit and token_place(tok)]
    if not shared:
        return []
    horizons = sorted({r["trained"] for r in plain if r["trained"]})
    made = []
    notes_of: dict = {}
    for tok in shared:
        have = have_of[tok]
        lat = _dict(have[0]["case_info"].get(tok)).get("lattice")
        path = comparison_case_path(
            root, tok, lat if isinstance(lat, int) else None)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            figure_run_comparison(
                have, path, name, case=tok, horizons=horizons,
                notes=notes_of.setdefault(os.path.dirname(path), {}))
            made.append(tok)
        except Exception as exc:
            print(f"[index] {os.path.relpath(path, root)} not drawn "
                  f"({type(exc).__name__}: {exc})", flush=True)
    for folder, notes in notes_of.items():
        write_figure_notes(folder, notes)
    return made


# --- END = 500 against END = 1000 (end500_vs_end1000.png) -------------------
END_PAIR = (500.0, 1000.0)


def end_pair(rows):
    """(trained to 500, trained to 1000): the plain Laplacian runs rolled out
    to R2_REPORT_TIMES[-1], found as laplacian_pair finds its two, each with
    cases; None when either is missing."""
    def pick(end):
        got = [r for r in rows if r["parsed"] and not r["parsed"]["tag"]
               and r["parsed"]["lap"] and r["parsed"]["t_train"] == end
               and not r["parsed"].get("nohard")
               and r["parsed"]["end"] == R2_REPORT_TIMES[-1] and r["cases"]]
        return got[-1] if got else None
    a, b = pick(END_PAIR[0]), pick(END_PAIR[1])
    return None if a is None or b is None else (a, b)


def end_points(row):
    """{psi0: {box size: (mean, standard error, seeds)}} of a run: R2 at the
    last time of the rollout, from the manifest's cases."""
    out: dict = {}
    for c in row["cases"].values():
        c = _dict(c)
        e = _dict(c.get("r2_at_end"))
        if (e.get("mean") is None or c.get("psi0") is None
                or c.get("lattice") is None):
            continue
        n = int(c.get("n_seeds") or 0)
        se = (float(e["sd"]) / np.sqrt(n)
              if e.get("sd") is not None and n > 1 else 0.0)
        out.setdefault(float(c["psi0"]), {})[int(c["lattice"])] = (
            float(e["mean"]), se, n)
    return out


def figure_end_pair(pair, path, notes=None):
    """R2 at t = 10000 against the box size (log2 axis), one panel for each
    composition, the two models as two lines with the seed mean and its
    standard error, only the sizes both runs have.  Returns False (nothing
    drawn) when the runs share no size."""
    pts = [end_points(r) for r in pair]
    offs = sorted(set(pts[0]) & set(pts[1]))
    panels = []
    for off in offs:
        sizes = sorted(set(pts[0][off]) & set(pts[1][off]))
        if sizes:
            panels.append((off, sizes))
    if not panels:
        return False
    colours = (RUN_COLORS[0], RUN_COLORS[1])
    marks = ("o", "s")
    allsizes = sorted({lat for _, sizes in panels for lat in sizes})
    fig, axes = plt.subplots(1, len(panels), sharey=True, squeeze=False,
                             figsize=(2.9 * len(panels) + 0.6, 3.4),
                             facecolor="white")
    facts = []
    for k, (ax, (off, sizes)) in enumerate(zip(axes[0], panels)):
        for i in range(2):
            y = np.array([pts[i][off][lat][0] for lat in sizes])
            e = np.array([pts[i][off][lat][1] for lat in sizes])
            ax.errorbar(sizes, y, yerr=e, color=colours[i], marker=marks[i],
                        ms=5, lw=1.6, capsize=3,
                        label=f"trained to $t$ = {END_PAIR[i]:g}")
            facts.append(f"psi0 = {off:+.1f}, trained to {END_PAIR[i]:g}: "
                         + ", ".join(
                             f"l = {lat}: {pts[i][off][lat][0]:.3f} +- "
                             f"{pts[i][off][lat][1]:.3f} "
                             f"({pts[i][off][lat][2]} seeds)"
                             for lat in sizes))
        ax.set_xscale("log", base=2)
        ax.set_xticks(allsizes)
        ax.set_xticklabels([str(lat) for lat in allsizes])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("$l$", fontsize=11)
        ax.set_title(f"({'abcdefgh'[k]})", fontsize=10, loc="left")
        ax.text(0.97, 0.05, comp_math(off), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10)
        ax.grid(alpha=0.3, ls=":", which="both")
    axes[0][0].set_ylabel(rf"$R^2(t = {R2_REPORT_TIMES[-1]:g})$", fontsize=11)
    axes[0][0].legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    _save_shared(fig, path)
    if notes is not None:
        notes[os.path.basename(path)] = [
            f"R2 at t = {R2_REPORT_TIMES[-1]:g} against the box size l (log2 "
            "axis), one panel for each composition (panels (a), (b), ... in "
            "the order of psi0, the composition in each panel's corner); the "
            "model "
            f"trained to t = {END_PAIR[0]:g} ({pair[0]['label']}) and the one "
            f"trained to t = {END_PAIR[1]:g} ({pair[1]['label']}) as two "
            "lines; points are the seed mean, bars its standard error; only "
            "the sizes both runs have"] + facts
    return True


def index_cells(r):
    """The cells of one run's row in the README table."""
    if not r["readable"]:
        return [f"`{r['label']}/`", "-", "-", "-", "-", "-",
                "manifest.json cannot be read"]
    crit = r["crit"].get("r2_at_end") or {}
    score = ("-" if crit.get("mean") is None else
             f"{crit['mean']:.4f} +- {crit.get('sd', 0.0):.4f}")
    trained = "-" if r["trained"] is None else "t = " + format(r["trained"],
                                                               "g")
    end = "-" if r["end"] is None else "t = " + format(r["end"], "g")
    par = "-" if not r["n_parameters"] else f"{int(r['n_parameters']):,}"
    return [f"`{r['label']}/`", par, trained, end, str(r["seeds"] or "-"),
            str(r["n_diverged"] if r["n_diverged"] is not None else "-"),
            score]


def _also(cases, ends):
    """The end of the sentence that says what the comparison figures are:
    the other cases drawn the same way, and the model sizes."""
    more = []
    if cases:
        more.append("the same for each other case in "
                    "`comparisons/l<size>/comparison_<composition>.png` "
                    f"(`{cases[0]}` ... `{cases[-1]}`)")
    if ends:
        more.append(", ".join(f"`{comparison_name(e)}`" for e in ends)
                    + " the model sizes of one training horizon, pooled "
                      "over the cases every run of the figure has")
    return "".join(", " + ("and " if i == len(more) - 1 else "") + s
                   for i, s in enumerate(more))


def index_text(rows, others, drawn, ends, cases=(), extra=()):
    """Results/README.md: the runs, and what the folders hold.  ``cases`` are
    the tokens that have a comparison_<case>.png; ``extra`` are the other
    top-level figures drawn (r2_vs_model_size.png, laplacian_on_off.png)."""
    L = [f"# Results - {DISPLAY_NAME}", "", INDEX_MARK, "",
         "One folder per run, named `train_t<trained to>[_<tag>][_lap]_pred_"
         "t<rolled out to>`. At its top it holds the files of the whole run: "
         "`manifest.json`, `report.txt`, `composition.csv` "
         "(one row per case and snapshot time, with the box size `l` in the "
         "last column), the folder `r2_sweep_per_size/` (one image "
         "`l<size>.png` for each box size: R2 against time, one line for each "
         "composition, psi_0 = 0 the heaviest), the folder "
         "`r2_sweep_per_case/` (one image `<psi_0>.png` for each composition: "
         "R2 against time, one line for each box size; both with the seed "
         "spread as a band, and each folder with a `figure_notes.txt` of its "
         "own) and "
         "`training_curves.png` (the "
         "model's training and validation curves against epoch) where the "
         "model's history is in `Work/models/`. Each box size has a folder, "
         "`l128/`, `l256/`, `l512/`. A folder holds, for each composition at "
         "that size, `snapshots_<psi_0>.png`, "
         "`phase_ordering_kinetics_<psi_0>.png`, "
         "`domain_growth_<psi_0>.png`, `psi_distribution_<psi_0>.png`, "
         "`composition_<psi_0>.png` and `free_energy_<psi_0>.png`, with "
         "`<psi_0>` as in `psi0m0.40` ... `psi0p0.40`; and for the size as "
         "a whole `composition_all.png`, `r2_time.png` (R2 against time of "
         "each composition, seed mean with the seed spread as a band) and a "
         "`figure_notes.txt` of its own. In `manifest.json` the case keys are "
         "`psi0m0.40` ... `psi0p0.40` for l = 128, `l256`, "
         "`l512` for psi_0 = 0 at the other sizes, and `psi0p0.20_l256` for "
         "any other composition at another size (`l128` is `psi0p0.00`). The "
         "figures have no titles; the `figure_notes.txt` of a folder says "
         "what each of its figures shows and gives the numbers that are not "
         "on the image. A run folder written before this layout, with its "
         "per-case figures at its top, is moved to `Archive/` and written "
         "again. This page and the comparison figures are rebuilt from the "
         "run folders by `python make_submission.py "
         "--overview` and at the end of every prediction job, so anything "
         "typed into them is overwritten.", "", "## Runs", ""]
    if not rows:
        L += ["No folder here has a `manifest.json` yet.", ""]
    else:
        L += ["| run | parameters | trained to | predicted to | seeds "
              "| diverged | critical R2(t_end), mean +- sd |",
              "|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append("| " + " | ".join(index_cells(r)[:7]) + " |")
        L += ["", "R2(t_end) is the all-seeds mean of the critical quench "
              "(psi_0 = 0, l = 128) at the last time of that run's rollout: "
              "a seed whose rollout diverged (|psi| > 5) counts R2 = -1 from "
              "then on and stays in the mean. The survivors-only mean "
              "(diverged seeds left out) is in brackets in the R2 table of "
              "each run's `report.txt`. Runs with "
              "different horizons are not comparable at their end points; "
              + ("`comparison.png` puts their critical curves on one log-t "
                 "axis" if drawn else "`comparison.png` is drawn once two "
                 "runs have a curve")
              + _also(cases, ends) + "."]
        if extra:
            L += ["", "Also here: " + ", ".join(f"`{n}`" for n in extra)
                  + ". Figures that pool over cases (`comparison_t<END>.png`, "
                  "`r2_vs_model_size.png`, `laplacian_on_off.png`) pool over "
                  "the cases every run in the figure has, so a run with more "
                  "cases is compared on the common ones. "
                  f"`{END_FIG}` has one panel for each composition: R2 at "
                  "t = 10000 against the box size (log2 axis) for the model "
                  "trained to t = 500 and the one trained to t = 1000, the "
                  "seed mean with its standard error, at the sizes both "
                  "runs have. Each figure of this folder is described in "
                  "`figure_notes.txt`; those of `comparisons/l<size>/` in "
                  "the `figure_notes.txt` beside them."]
    for r in rows:
        if not r["readable"]:
            continue
        L += ["", f"## {r['label']}", "",
              "| case | R2(t_end) +- sd | median | max | min "
              "| trajectory-mean R2 | best seed (mean R2) | diverged |",
              "|---|---|---|---|---|---|---|---|"]
        for tok, c in r["cases"].items():
            e, tr = c.get("r2_at_end") or {}, c.get("r2_trajectory_mean") or {}
            b = (c.get("seeds") or {}).get("best") or {}
            L.append(f"| {tok} | {_fmt(e.get('mean'))} +- "
                     f"{_fmt(e.get('sd'))} | {_fmt(e.get('median'))} "
                     f"| {_fmt(e.get('best'))} | {_fmt(e.get('worst'))} "
                     f"| {_fmt(tr.get('mean'))} | {b.get('seed', '-')} "
                     f"({_fmt(b.get('r2_trajectory_mean'), '.3f')}) "
                     f"| {c.get('n_diverged', '-')} |")
    L += ["", "## What each run folder holds", "",
          "Every run's figures are drawn by the same code from its per-seed "
          "records; the best seed is shown in the snapshots (one figure, "
          "t = 100 to 10000), the growth figure draws the half-maximum "
          "length L_half with its mean +-1 sd, the kinetics figure has panels "
          "(a) to (f) and `free_energy_<case>.png` the free energy per site. "
          "The `figure_notes.txt` of a folder has, for each of its figures, "
          "what it shows and the numbers that are not on the image. "
          "`manifest.json` "
          "has every number the figures are drawn from; `report.txt` has the "
          "model, the composition and its conservation, the solve time, the "
          "summary table, the step schedule and the diagnostics.", "",
          "FNO's own extras (per-case summaries and CSVs, the best seed's "
          "folder, the film, diagnostics.json, sweep_curves.npz, the "
          "bookkeeping manifest) are not here: they are in "
          "`Work/rollout_cache/<run>/extras/`.", "",
          "## Finite-size regime", "",
          "Past L > l / 8 (mean zero-crossing length) the domains feel the "
          "box; the figures shade that regime and the report's exponents "
          "are fitted over the late third of the rollout, so read late "
          "numbers at l = 128 with that in mind.", "",
          "## Regenerate", "",
          "`python make_submission.py` redraws every complete run from its "
          "records and rebuilds this page; `--run LABEL` one run; "
          "`--overview` only this page and the comparison figures. No model "
          "is run and no trajectory solved.", ""]
    if others:
        L += ["## Other folders", "",
              "These have no `manifest.json` at their top, so they are not "
              "runs of this layout. They are listed here and never "
              "changed.", ""]
        L += [f"- `{n}/` - {first}" for n, first in others]
    return "\n".join(L) + "\n"


def first_line(folder):
    """The first line of a folder's own README, without its heading marks;
    a note when it has none.  Read only."""
    try:
        with open(os.path.join(folder, INDEX_FILE), encoding="utf-8",
                  errors="replace") as fh:
            for line in fh:
                if line.strip():
                    return line.strip().lstrip("#").strip() or "(no title)"
    except OSError:
        pass
    return "(no README)"


def write_results_index(root, work_root=None):
    """The overview of a results folder: README.md, comparison.png, its form
    for each other case (comparisons/l<size>/comparison_<composition>.png),
    comparison_t<END>.png, r2_vs_model_size.png, laplacian_on_off.png and
    end500_vs_end1000.png, from the run folders (those with a manifest.json)
    directly under ``root``.  Figures that pool over cases pool over the
    cases every run in the figure has.  The only thing drawn in a run folder is its
    training_curves.png, from the model's history in Work/models/
    (``work_root`` is the smoke root under --smoke), listed in the folder's
    manifest.json when it was not.

    Any other folder (but comparisons/, the per-case comparison figures) is
    listed under "Other folders" and left as it is.  Each
    file is written through a temporary of this process's own; the page
    carries no time, so the same folders give the same bytes.  An existing
    README.md that this index did not write is kept, and said so.  Returns
    what it found."""
    if not os.path.isdir(root):
        raise SystemExit(f"no results folder {root} to index")
    clear_shared_tmp(root)
    rows, others = [], []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if (not os.path.isdir(path) or name.startswith((".", "_"))
                or name == COMPARISONS_DIR):
            continue
        if os.path.isfile(os.path.join(path, MANIFEST)):
            rows.append(run_row(path, work_root))
        else:
            others.append((name, first_line(path)))
    rows.sort(key=run_order)
    notes: dict = {}
    trained = []
    for r in rows:
        if r["readable"] and draw_row_training(
                os.path.join(root, r["label"]), r, work_root):
            trained.append(r["label"])
    curved = [r for r in rows if r["curve"] is not None]
    plain = overview_runs(curved)
    drawn = len(plain) >= 2
    if drawn:
        # first, and not caught: a figure that cannot be drawn leaves the
        # old pair as it was, rather than a page that describes a new one
        figure_run_comparison(plain, os.path.join(root, COMPARISON_FILE),
                              name=P.submission_name(), notes=notes)
    cases = (draw_case_comparisons(root, plain, P.submission_name())
             if drawn else [])
    ends = size_ends(curved)
    made = []
    for end in ends:
        group = size_runs(curved, end)
        if len(group) >= 2:
            # runs with different cases are compared on the cases they all
            # have
            common = common_tokens(group)
            group = [restrict_row(r, common) for r in group]
            figure_size_comparison(group, end,
                                   os.path.join(root, comparison_name(end)),
                                   name=P.submission_name(), notes=notes)
            made.append(end)
    sized = model_size_runs(curved)
    common = common_tokens(sized)
    sized = [restrict_row(r, common) for r in sized]
    if len(sized) >= 2:
        figure_r2_vs_model_size(sized, os.path.join(root, SIZE_FIG),
                                notes=notes)
    pair = laplacian_pair(curved)
    if pair is not None:
        common = common_tokens(pair)
        pair = (restrict_row(pair[0], common), restrict_row(pair[1], common))
        figure_laplacian_on_off(pair, os.path.join(root, LAPLACIAN_FIG),
                                notes=notes)
    ends_pair = end_pair(rows)
    ended = bool(ends_pair) and figure_end_pair(
        ends_pair, os.path.join(root, END_FIG), notes=notes)
    write_figure_notes(root, notes)
    page = os.path.join(root, INDEX_FILE)
    mine = True
    if os.path.exists(page):
        with open(page, encoding="utf-8", errors="replace") as fh:
            mine = INDEX_MARK in fh.read()
    if mine:
        extra = [n for n, ok in ((SIZE_FIG, len(sized) >= 2),
                                 (LAPLACIAN_FIG, pair is not None),
                                 (END_FIG, ended)) if ok]
        write_text(page, index_text(rows, others, drawn, made, cases, extra),
                   shared=True)
        print(f"[saved] {page}")
    else:
        print(f"[index] {page} was not written by this index, so it is kept "
              f"as it is; move it aside to have the index written",
              flush=True)
    return {"runs": [r["label"] for r in rows],
            "other": [n for n, _ in others], "comparison": drawn,
            "cases": cases, "training": trained,
            "sizes": made, "written": mine,
            "model_size": len(sized) >= 2, "laplacian": pair is not None,
            "end_pair": ended}


def write_index(built, out, paired=None):
    """README.md for the folder of runs, so the three stay readable together."""
    L = [f"# Results - {DISPLAY_NAME}, three runs", "",
         "Each run's folder was regenerated from its result cache by",
         "`python make_submission.py --all`, through the code a prediction",
         "run writes its folder with. Everything the caches held that is",
         "worth keeping -- figures, averages over every seed, the best",
         "seed's data -- is in these folders, so the caches are no longer",
         "needed.", "",
         "## The three runs", "",
         "| run | training data | eval seeds | critical mean R2 | folder |",
         "|---|---|---|---|---|"]
    for run, m in built:
        crit = m["compositions"].get("+0.00", {})
        score = crit.get("mean_r2")
        L.append(f"| {run['key'][0]} "
                 f"| to t = {'500' if 't500' in run['key'] else '300'} "
                 f"| {crit.get('seeds', '-')} "
                 f"| {'-' if score is None else format(score, '.4f')} "
                 f"| `{run['key']}/` |")
    L += ["",
          "`comparison.png` puts all three on one axis, and",
          "`comparison.json` holds its numbers.", ""]
    if paired:
        base, rows, same = paired
        head, mean_d, body = paired_verdict(rows)
        n = min(r["n"] for r in rows)
        L += ["## Did extending the training data help?", "", f"**{head}**",
              ""]
        if same is not None and same[0] == 0.0:
            L += ["A and B are the *same model*: their per-step R2 agrees to",
                  f"0.00e+00 on every seed they share, across {same[1]}",
                  "compositions, and A's ten seeds are a prefix of B's fifty.",
                  "So any gap between A and B is the seed sample, not the",
                  "model.", ""]
        L += [f"Pairing {base} against C on the {n} seeds they share removes",
              "the seed sample and leaves only the effect of training to",
              "t = 500:", "",
              "| psi_0 | seeds | t = 300 | t = 500 | change | s.e. | t |",
              "|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['off']:+.1f} | {r['n']} | {r['a']:.4f} "
                     f"| {r['b']:.4f} | {r['delta']:+.4f} | {r['se']:.4f} "
                     f"| {r['t']:+.2f} |")
        L += ["", f"Mean change across the {len(rows)} compositions: "
                  f"**{mean_d:+.4f}**.", ""] + body + [""]
    for run, m in built:
        L += [f"## {run['key']}", "", f"{run['title']}. {run['note']}.", "",
              f"- regenerated from: `{run['cache']}`",
              "- compositions: "
              + (", ".join(sorted(m["compositions"])) or "none"),
              "- box sizes: "
              + (", ".join(sorted(m["sizes"], key=int)) or "none")]
        if m.get("missing"):
            L.append(f"- not in its cache: {', '.join(m['missing'])}")
        if m.get("failed"):
            L.append(f"- FAILED to build: {'; '.join(m['failed'])}")
        L += ["", "Files: `manifest.json` and `report.txt` in the folder "
              "list them; FNO's extras are in `extras/`.", ""]
    write_text(os.path.join(out, "README.md"), "\n".join(L))
    print(f"[saved] {os.path.join(out, 'README.md')}")


def build_all(out, failures=None):
    """Every run in RUNS into its own subfolder, plus the comparison.

    A run that cannot be built, or whose comparison numbers cannot be read,
    is reported in `failures` (when given) and the others go ahead.
    """
    failures = [] if failures is None else failures

    def guarded(what, fn, *a):
        try:
            return fn(*a)
        except Exception as exc:
            failures.append(f"{what}: {type(exc).__name__}: {exc}")
            print(f"[error] {what} FAILED -- continuing with the rest:",
                  flush=True)
            traceback.print_exc()
            return None

    built, gone = [], []
    for run in RUNS:
        if not os.path.isdir(run["cache"]):
            print(f"[skip] {run['key']}: no directory {run['cache']}")
            if os.path.exists(os.path.join(out, run["key"], MANIFEST)):
                gone.append(run["key"])
            continue
        bar = "#" * 78
        print(f"\n{bar}\n#  {run['key']} - {run['title']}\n{bar}",
              flush=True)
        m = guarded(run["key"], build, run["cache"],
                    os.path.join(out, run["key"]), run.get("model"))
        if m is None:
            continue
        curve = guarded(f"{run['key']} critical curve", critical_curve,
                        run["cache"])
        if curve is not None:
            m["_critical_curve"] = curve
        summarise(m, os.path.join(out, run["key"]))
        built.append((run, m))
    if built and gone:
        print(f"[build] kept the existing comparison.png, comparison.json and "
              f"README.md: {', '.join(gone)} has its folder here but no cache "
              f"any more, and redrawing them without it would drop it",
              flush=True)
    elif built:
        print()
        notes: dict = {}
        figure_compare(built, os.path.join(out, "comparison.png"), notes)
        write_figure_notes(out, notes)
        paired = None
        keys = {r["key"]: r["cache"] for r, _ in built}
        c = next((v for k, v in keys.items() if k.startswith("C")), None)
        best = None
        for k, v in keys.items():
            if c is None or k[0] not in "AB":
                continue
            rows = guarded(f"paired {k[0]} vs C", paired_deltas, v, c)
            if rows and (best is None or (len(rows), min(r["n"] for r in rows))
                         > (len(best[1]), min(r["n"] for r in best[1]))):
                best = (k[0], rows)
        if best is not None:
            a_cache = keys.get(next((k for k in keys if k[0] == "A"), ""), "")
            b_cache = keys.get(next((k for k in keys if k[0] == "B"), ""), "")
            same = (guarded("same-model check", same_model, a_cache,
                            b_cache) if a_cache and b_cache else None)
            paired = (best[0], best[1], same)
            head, mean_d, _ = paired_verdict(best[1])
            print(f"\n  paired {best[0]} vs C on the seeds they share "
                  f"(the seed sample cancels):")
            for r in best[1]:
                print(f"    psi0 {r['off']:+.1f} ({r['n']} seeds): "
                      f"{r['a']:.4f} -> {r['b']:.4f}   {r['delta']:+.4f} "
                      f"+/- {r['se']:.4f}  (t = {r['t']:+.2f})")
            print(f"    mean change {mean_d:+.4f} -- {head}", flush=True)
        write_index(built, out, paired)
        write_json(os.path.join(out, "comparison.json"), {
            "runs": [{"key": run["key"], "cache": run["cache"],
                      "critical_mean_r2": (m["compositions"].get("+0.00")
                                           or {}).get("mean_r2"),
                      "sizes_mean_r2": {k: e.get("mean_r2")
                                        for k, e in m["sizes"].items()}}
                     for run, m in built],
            "paired": (None if paired is None else
                       {"base": paired[0], "rows": paired[1],
                        "same_model": paired[2]})})
    return built


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--cache",
                    help="one result cache to regenerate a folder from")
    ap.add_argument("--out",
                    help=f"the results folder to write; with --all, the "
                         f"folder the runs go into (default {ALL_OUT})")
    ap.add_argument("--records", metavar="DIR",
                    help="redraw a results folder from the per-seed records "
                         "of a run, the rollout-cache folder holding its "
                         "meta.json and records/ (no model, no solver; no "
                         "film); give --out")
    ap.add_argument("--partial", action="store_true",
                    help="with --records: draw the complete cases of a "
                         "folder whose run is not finished, instead of "
                         "refusing it")
    ap.add_argument("--model",
                    help="the checkpoint the cache was made with "
                         "(provenance only)")
    ap.add_argument("--all", action="store_true",
                    help="regenerate every run in RUNS into its own folder, "
                         "plus comparison.png and README.md")
    ap.add_argument("--smoke", action="store_true",
                    help="rebuild the newest python fno_predict.py --smoke "
                         "run from its cache (the defaults of --cache and "
                         "--out); everything stays under Work/smoke, and "
                         "--all is refused")
    ap.add_argument("--overview", "--index", dest="index",
                    action="store_true",
                    help="only the overview of Results/: README.md, "
                         "comparison.png and comparison_t<END>.png, from the "
                         "run folders in it (with --smoke, the smoke root's; "
                         "--out names another results folder); no run "
                         "folder is drawn.  --index is the old name")
    ap.add_argument("--run", metavar="LABEL",
                    help="redraw one run, Work/rollout_cache/LABEL, into "
                         "Results/LABEL from its records, then the overview")
    a = ap.parse_args(argv)
    smoke = []
    if a.index:
        if a.all or a.cache or a.records or a.partial or a.model or a.run:
            ap.error("--overview only rebuilds the overview; it takes no "
                     "--all, --cache, --records, --partial, --run or "
                     "--model")
        root = fno_paths.smoke_root() if a.smoke else None
        if a.out and a.smoke and not fno_paths.under(a.out, root):
            ap.error(f"--smoke only reads and writes under {root}; --out "
                     f"{a.out} is outside it")
        folder = a.out or fno_paths.results_root(root)
        if a.smoke:
            fno_runlog.LOG_DIR = fno_paths.log_dir(root)
        fno_runlog.start_log("make_submission")
        found = write_results_index(folder, root)
        print(f"[overview] {len(found['runs'])} run folder(s) and "
              f"{len(found['other'])} other folder(s) in {folder}; "
              f"training curves of {len(found['training'])} run(s), "
              f"{len(found['cases'])} case comparison(s)")
        return
    if a.smoke:
        root = fno_paths.smoke_root()
        if a.all:
            ap.error("--smoke rebuilds one run; it cannot be combined with "
                     "--all")
        for flag, path in (("--cache", a.cache), ("--out", a.out),
                           ("--model", a.model), ("--records", a.records)):
            if path and not fno_paths.under(path, root):
                ap.error(f"--smoke only reads and writes under {root}; "
                         f"{flag} {path} is outside it")
        if not a.records:
            a.cache = a.cache or fno_paths.latest_rollouts(root)
        source = a.records or a.cache
        if not source:
            raise SystemExit("no smoke rollouts; run python fno_predict.py "
                             "--smoke first")
        a.out = a.out or fno_paths.results_dir(
            os.path.basename(os.path.normpath(source)), root)
        fno_runlog.LOG_DIR = fno_paths.log_dir(root)
        smoke = [f"writing only under {root}: {P.smoke_cases()}"]
    if a.records and (a.cache or a.all or a.model or a.run):
        ap.error("--records is a run of its own: it takes no --cache, --all, "
                 "--run or --model (the records name their model)")
    if a.partial and not a.records:
        ap.error("--partial goes with --records")
    if a.records and not a.out:
        ap.error("give --out with --records")
    if a.run and (a.cache or a.all or a.model):
        ap.error("--run takes no --cache, --all or --model")
    default = not (a.all or a.records or a.cache or a.run)
    if a.cache and not a.out:
        ap.error("give --out with --cache")
    fno_runlog.start_log("make_submission")
    for line in smoke:
        print(f"[smoke] {line}", flush=True)
    if a.all:
        out = a.out or ALL_OUT
        print(f"[build] {len(RUNS)} runs -> {out}", flush=True)
        run_failures = []
        built = build_all(out, run_failures)
        if not built and not run_failures:
            raise SystemExit("found no cache directory for any run")
        failed = run_failures + [f"{run['key']}: {f}" for run, m in built
                                 for f in m.get("failed", ())]
    elif a.run or default:
        failed = redraw_runs(a.run, fno_paths.smoke_root() if a.smoke
                             else None)
    elif a.records:
        if not os.path.isdir(a.records):
            raise SystemExit(f"no such records directory: {a.records}")
        print(f"[redraw] reading the records in {a.records} -> {a.out}",
              flush=True)
        m = redraw(a.records, a.out, a.partial)
        summarise(m, a.out)
        failed = list(m.get("failed", ()))
    else:
        if not a.cache or not os.path.isdir(a.cache):
            raise SystemExit(f"no such cache directory: {a.cache}")
        print(f"[build] reading {a.cache} -> {a.out}", flush=True)
        m = build(a.cache, a.out, a.model)
        summarise(m, a.out)
        failed = list(m.get("failed", ()))
    if failed:
        raise SystemExit(f"{len(failed)} part(s) FAILED to build; "
                         f"everything else was written:\n  "
                         + "\n  ".join(failed))


if __name__ == "__main__":
    main()
