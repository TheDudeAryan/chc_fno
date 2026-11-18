"""The flat Results layout, end to end.

What a run folder holds and what it must not: the file set and names per case
token (psi0m0.40 ... psi0p0.40, l256 ... l512), report.txt and its sections
(the Conventions note included), the single sweep images in r2_sweep_per_size/
and r2_sweep_per_case/ (PNG only), the manifest.json
schema, FNO's extras under Work/rollout_cache/<label>/extras/<case>/ (which no
scan of the cache and no loader of the records looks into), the modes of
make_submission (--overview, its alias --index, and the default mode that
redraws every complete run), a time grid that is not a fixed step, and a copy
of the real records redrawn into a temporary Results folder.

Every rollout is synthetic (evaluate_seed is stubbed), the checkpoint is a
small file of this suite's own, and nothing is written outside _scratch.
"""
import contextlib
import hashlib
import io as _io
import json
import os
import shutil
import sys
import time
import types
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.axes
import numpy as np

from _paths import REAL_RECORDS, ROOT, SCRATCH, project_state

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


BASE = os.path.join(SCRATCH, "flat")
shutil.rmtree(BASE, ignore_errors=True)
os.makedirs(BASE)
CKPT = os.path.join(BASE, "models", "train_t0500", "fno_ch.pt")
os.makedirs(os.path.dirname(CKPT))
with open(CKPT, "wb") as fh:
    fh.write(b"a checkpoint of this suite, not a network")
FP = fno_paths.fingerprint(CKPT)

# the sweep images have no title; the old panel tags "(a)  $l = 128$" ... and
# "(a)  $\bar\psi_0 = -0.4$" ... must not come back.  Other figures set
# titles "(a)", "(b)" too, so the whole title is kept and checked.
PANELS = []
_title = matplotlib.axes.Axes.set_title


def _spy_title(self, label, *a, **k):
    if str(label)[:3] in ("(a)", "(b)", "(c)", "(d)", "(e)"):
        PANELS.append(str(label))
    return _title(self, label, *a, **k)


matplotlib.axes.Axes.set_title = _spy_title

KEYS = ("EVAL_MODE", "SIZE_SWEEP", "EVAL_SEEDS", "SWEEP_SEEDS", "WRITE_VIDEO",
        "RESULTS_DIR", "RESULTS_TEMPLATE", "VIEWS", "WRITE_TIMING",
        "PER_SEED_FIGURES", "PER_SEED_OUTPUT", "WRITE_SUBMISSION", "WRITE_LOG",
        "START_TIME", "END_TIME", "SNAPSHOT_TIMES", "MODEL_PATH", "REDO",
        "KEEP_CACHE", "load_checkpoint", "evaluate_seed", "RESULT_CACHE",
        "GT_CACHE", "CACHE_RESULTS", "RECORDS", "ROOT", "NO_SUPERSEDE",
        "SEED_SET", "LATTICE", "LABEL", "AUTO_LABEL", "LOG_TIME",
        "SNAPSHOTS_EXPLICIT", "RECORDED_FINGERPRINT", "CACHE_END_TIME",
        "_RUN_SNAPSHOTS", "DIAG_EVERY", "VIDEO_EVERY", "VIDEO_SCHEDULE",
        "evaluate_seeds", "EVAL_BATCH", "VIDEO_DTYPE", "ENGINE",
        "DIAGNOSTICS", "EXTRAS_ROOT", "KEEP_FILM")
SAVED = {k: getattr(P, k) for k in KEYS}
SAVED.update(WRITE_LOG=False, MODEL_PATH=CKPT, GT_CACHE="",
             RESULTS_DIR=os.path.join(BASE, "results_default"),
             RESULT_CACHE=os.path.join(BASE, "cache_default"),
             ROOT=os.path.join(BASE, "root"))
LOG_DIR = fno_runlog.LOG_DIR
WORK, RESULTS = fno_paths.WORK, fno_paths.RESULTS
FLAGS = dict(hard_constraints=True, hard_mass_conservation=True,
             zero_dc=True, label="fno_ch", note="")
LAT = 24


class FakeModel:
    config = {"backbone": "physical_k", "hard_mass_conservation": True,
              "residual_prediction": True, "mode_scaling": 1.0,
              "zero_dc": True}

    def n_parameters(self):
        return 1234

    def parameters(self):
        yield types.SimpleNamespace(dtype="float32")


def rollout(seed, off=0.0, lat=LAT, die=None):
    """A synthetic rollout over the present window."""
    g = np.random.default_rng([int(seed), int(round(off * 10)) + 10, lat])
    t = np.arange(P.START_TIME + 1, P.END_TIME + 1)
    diag = t[::max(1, t.size // 19)]
    r2 = np.clip(1.0 - (t - t[0]) / 9000.0 - 0.01 * g.random(), -1, 1)
    mp = np.full(t.size, off) + 1e-7 * (t - t[0])
    if die is not None:
        gone = t > die
        r2 = np.where(gone, P.R2_FLOOR, r2)
        mp = np.where(gone, np.nan, mp)
        diag = diag[diag <= die]
    snaps = {}
    for tt in P.SNAPSHOT_TIMES:
        f = np.tanh(3 * g.standard_normal((lat, lat))) + off
        q = np.clip(f + 0.05 * g.standard_normal(f.shape), -1.05, 1.05)
        if tt == P.START_TIME:
            q = f.copy()
        if die is not None and tt > die:
            q = np.full_like(f, np.nan)
        snaps[tt] = (f, q)
    L = np.linspace(5, 14, diag.size)
    return {
        "times": t, "r2": r2, "mass_drift": np.full(t.size, 1e-9),
        "mean_true": np.full(t.size, off), "mean_pred": mp,
        "diag_times": diag,
        "energy_true": np.linspace(1, 0.5, diag.size),
        "energy_pred": np.linspace(1, 0.5, diag.size),
        "t_stop": float(t[-1]), "stopped_early": False,
        "diverged_at": None if die is None else float(die),
        "L_true": {k: L for k in ("L_zero", "L_peak", "L_moment", "L_grad",
                                  "L_half")},
        "L_pred": {k: L * 1.01 for k in ("L_zero", "L_peak", "L_moment",
                                         "L_grad", "L_half")},
        "snapshots": snaps,
        "gt_seconds": 1.0, "gt_cached": False, "gt_steps": 1,
        "fno_seconds": 0.5, "fno_steps": 1900,
        "video_t": np.asarray([]), "video_true": None, "video_pred": None}


def fake_evaluate_seed(model, seed, device, off=0.0, lattice=None):
    return rollout(seed, off, LAT if lattice is None else lattice,
                   die=2500.0 if (off > 0.3 and seed == P.EVAL_SEEDS[0])
                   else None)


def no_rollout(*a, **k):
    raise AssertionError("a rollout was asked for")


def setup():
    for k, v in SAVED.items():
        setattr(P, k, v)
    fno_runlog.LOG_DIR = LOG_DIR
    fno_paths.WORK, fno_paths.RESULTS = WORK, RESULTS
    P.MODEL_FLAGS.update(FLAGS)
    P.load_checkpoint = lambda path, device=None: (FakeModel(), {
        "best": {"epoch": 7, "score": 0.91, "survival": 2000.0}})
    P.evaluate_seed = fake_evaluate_seed
    P.evaluate_seeds = lambda model, seeds, device, off=0.0, lattice=None: [
        fake_evaluate_seed(model, s, device, off, lattice) for s in seeds]
    P.WRITE_TIMING = False
    P.PER_SEED_FIGURES = False
    P.LATTICE = LAT
    P.START_TIME, P.END_TIME = 100.0, 3000.0
    P.SNAPSHOT_TIMES = (100.0, 500.0, 1000.0, 2000.0, 3000.0)


def quiet(fn, *a, **k) -> tuple[Any, str]:
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            out = fn(*a, **k)
        except SystemExit as exc:
            out = exc.code
    return out, buf.getvalue()


def files_under(d):
    return sorted(os.path.relpath(os.path.join(b, f), d).replace(os.sep, "/")
                  for b, _, fs in os.walk(d) for f in fs)


def tree(d, skip=()):
    out = {}
    for f in files_under(d):
        if not f.startswith(tuple(skip)):
            with open(os.path.join(d, *f.split("/")), "rb") as fh:
                out[f] = fh.read()
    return out


def read(p):
    with open(p, "rb") as fh:
        return fh.read()


def text(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def make_submission_main(argv):
    """make_submission's main with its run log off."""
    real = S.fno_runlog.start_log
    S.fno_runlog.start_log = lambda *a, **k: None
    try:
        return quiet(S.main, argv)
    finally:
        S.fno_runlog.start_log = real


print("1. case tokens and file names")
check([S.case_token(o) for o in (-0.4, -0.2, 0.0, 0.2, 0.4)]
      == ["psi0m0.40", "psi0m0.20", "psi0p0.00", "psi0p0.20", "psi0p0.40"]
      and S.case_token(-0.0) == "psi0p0.00"
      and S.case_token(-1e-15) == "psi0p0.00"
      and [S.case_token(0.0, n) for n in (256, 384, 512)]
      == ["l256", "l384", "l512"]
      and [S.case_token(o, 256) for o in (-0.4, 0.2)]
      == ["psi0m0.40_l256", "psi0p0.20_l256"]
      and S.case_token(-1e-15, 512) == "l512",
      "psi0m0.40 ... psi0p0.40 for the compositions at the base box, l256 / "
      "l384 / l512 for psi0 = 0 at the other sizes, psi0p0.20_l256 "
      "for any other composition there, and a zero of either sign is "
      "psi0p0.00")
check([S.token_place(t) for t in ("psi0p0.20", "l384", "psi0m0.40_l256",
                                  "psi0p0.00_sweep_seeds", "x")]
      == [(f"l{P.LATTICE}", "psi0p0.20"), ("l384", "psi0p0.00"),
          ("l256", "psi0m0.40"), None, None]
      and S.case_file("snapshots", "psi0p0.20_l256")
      == "l256/snapshots_psi0p0.20.png"
      and S.size_folder(512) == "l512",
      "token_place: a token's size folder and composition, and the path of "
      "a case's figure from the run folder")
check(S.FIG_KINDS == ("snapshots", "phase_ordering_kinetics",
                      "domain_growth", "psi_distribution", "composition",
                      "free_energy")
      and S.fig_name("free_energy", "l256") == "free_energy_l256.png"
      and S.fig_name("phase_ordering_kinetics", "psi0m0.20")
      == "phase_ordering_kinetics_psi0m0.20.png"
      and S.RUN_FILES == ("composition.csv", "manifest.json", "report.txt")
      and S.SWEEP_DIRS == ("r2_sweep_per_size", "r2_sweep_per_case")
      and S.SIZE_FILES == ("composition_all.png", "r2_time.png",
                           "figure_notes.txt"),
      "<kind>_<composition>.png for the six kinds, the files at the top of "
      "a run and those of each size's folder")

print("\n2. a run: the flat set, report.txt, r2_sweep, manifest.json")
setup()
RES_ROOT = os.path.join(BASE, "root", "Results")
RUN1 = os.path.join(RES_ROOT, "train_t0500_pred_t03000")
CACHE1 = os.path.join(BASE, "root", "Work", "rollout_cache",
                      "train_t0500_pred_t03000")
code, log = quiet(P.main, [
    "--model", CKPT, "--results-dir", RUN1, "--result-cache", CACHE1,
    "--eval-mode", "all", "--sizes", f"{LAT},32", "--seeds", "3",
    "--no-gt-cache", "--no-diagnostics", "--no-video"])
check(code == 0, f"a run to 3000 over five compositions and a box size "
                 f"exits 0: {code}")
TOKENS = ["psi0m0.40", "psi0m0.20", "psi0p0.00", "psi0p0.20", "psi0p0.40",
          "psi0m0.40_l32", "psi0m0.20_l32", "l32", "psi0p0.20_l32",
          "psi0p0.40_l32"]
SIZES = (LAT, 32)
SWEEPF = ([f"r2_sweep_per_size/l{n}.png" for n in SIZES]
          + [f"r2_sweep_per_case/{c}.png" for c in (
              "psi0m0.40", "psi0m0.20", "psi0p0.00", "psi0p0.20", "psi0p0.40")]
          + [f"{d}/figure_notes.txt" for d in S.SWEEP_DIRS])
FLAT = sorted([S.case_file(k, t) for t in TOKENS for k in S.FIG_KINDS]
              + [f"l{n}/{f}" for n in SIZES for f in S.SIZE_FILES]
              + list(S.RUN_FILES) + SWEEPF)
got = files_under(RUN1)
check(got == FLAT and len(FLAT) == 10 * 6 + 2 * 3 + len(S.RUN_FILES) + len(SWEEPF),
      f"the run folder: the run's {len(S.RUN_FILES)} files at the top, the "
      f"{len(SWEEPF)} files of the two sweep folders, and "
      f"for each of the two sizes a folder with six figures for each of its "
      f"five compositions and its own {len(S.SIZE_FILES)} files "
      f"({len(got)} files; unexpected {sorted(set(got) ^ set(FLAT))[:4]})")
NOTES1 = S.read_notes(os.path.join(RUN1, S.NOTES_FILE))
NOTES_SIZE = {n: S.read_notes(os.path.join(RUN1, f"l{n}", S.NOTES_FILE))
              for n in SIZES}
NOTES_SW = {d: S.read_notes(os.path.join(RUN1, d, S.NOTES_FILE))
            for d in S.SWEEP_DIRS}
check(not NOTES1
      and set(NOTES_SW["r2_sweep_per_size"]) == {f"l{n}.png" for n in SIZES}
      and len(NOTES_SW["r2_sweep_per_case"]) == 5
      and all(NOTES_SIZE[n].keys() == {
          f.split("/")[1] for f in FLAT
          if f.startswith(f"l{n}/") and f.endswith(".png")}
          for n in SIZES)
      and all(NOTES_SIZE[n][f] for n in SIZES for f in NOTES_SIZE[n])
      and any("FNO monotone" in x
              for x in NOTES_SIZE[32]["free_energy_psi0p0.00.png"])
      and any("seed shown" in x
              for x in NOTES_SIZE[LAT]["snapshots_psi0p0.00.png"])
      and all(os.path.isfile(os.path.join(RUN1, n)) for n in NOTES1),
      "the run's figure_notes.txt has no sweep block (each sweep folder has its own); "
      "the one in each size's folder for each figure of that folder (the "
      "case figures, composition_all.png and r2_time.png) and for nothing "
      "else")
check(not [f for f in os.listdir(RUN1) if f.startswith(".")]
      and not os.path.exists(os.path.join(RUN1, "README.md")),
      "no scratch folder or README.md in it")
EX1 = os.path.join(CACHE1, "extras")
cases_x = sorted(d for d in os.listdir(EX1)
                 if os.path.isdir(os.path.join(EX1, d)))
check(cases_x == sorted([f"psi0{o}_l{n}" for o in
                         ("-0.20", "-0.40", "+0.00", "+0.20", "+0.40")
                         for n in SIZES])
      and all(os.path.isfile(os.path.join(EX1, c, f))
              for c in cases_x for f in ("mean_r2.csv", "psi_distribution.csv",
                                         "domain_growth.csv",
                                         "composition.csv"))
      and os.path.isfile(os.path.join(EX1, S.BOOKKEEPING))
      and os.path.isfile(os.path.join(EX1, S.CURVES_FILE)),
      f"FNO's extras are in Work/rollout_cache/<label>/extras/<case>/: "
      f"{cases_x}")

rep = text(os.path.join(RUN1, "report.txt"))
HEADS = ["1  TRAINED MODEL", "2  MIXTURE COMPOSITION AND ITS CONSERVATION",
         "3  SOLVE TIME", "4  SUMMARY", "5  STEP SCHEDULE", "6  DIAGNOSTICS",
         "7  LOW-k INCREMENT RESPONSE", "8  CONVENTIONS"]
pos = [rep.find(h) for h in HEADS]
check(all(p >= 0 for p in pos) and pos == sorted(pos)
      and all(len(ln) <= 110 for ln in rep.splitlines()
              if not ln.lstrip().startswith(("checkpoint", "variant"))),
      "report.txt has the flat layout's sections 1 to 7 and FNO's Conventions, in order, "
      "its tables within 110 columns (a path aside)")
conv = rep[pos[-1]:]
check(all(len(ln) <= 78 for ln in conv.splitlines()),
      "and the prose of the Conventions is wrapped to the flat layout's 78 columns")
conv = " ".join(conv.split())
check("|psi| exceeds 5" in conv and "R2 = -1" in conv
      and "(the all-seeds mean)" in conv
      and "the survivors-only mean, which leaves diverged seeds out, is used "
          "only in brackets in the R2 table of section 4" in conv
      and "extras/" in conv,
      "the Conventions note: diverged means |psi| > 5, a diverged seed "
      "stays in every mean at R2 = -1 (the all-seeds mean), the "
      "survivors-only mean is used only in brackets, and where the FNO-only "
      "outputs are")
check(fno_paths.OLD_STORED_NAME not in rep.lower(),
      "and report.txt names no other project anywhere")
check("fingerprint" in rep and FP in rep and "epoch" in rep
      and "no diagnostics.json for this run" in rep
      and all(f"  {t:<15}" in rep for t in TOKENS),
      "section 1 names the checkpoint and its selection, sections 5 and 6 "
      "say there is no diagnostics.json, section 4 has every case")

png = read(os.path.join(RUN1, "r2_sweep_per_size", f"l{LAT}.png"))
check(png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 20000
      and not [f for f in files_under(RUN1) if f.endswith(".pdf")],
      f"r2_sweep_per_size/l{LAT}.png ({len(png)} bytes); no PDF in the run")
check(not any(s.startswith("(") and "$l = " in s[:12] for s in PANELS)
      and not any(r"$\bar\psi_0 = " in s for s in PANELS),
      "the sweep images draw no title and no panel tag")

m = jload(os.path.join(RUN1, "manifest.json"))
TOP = ("label", "source", "cache", "checkpoint", "checkpoint_fingerprint",
       "train_t_end", "t_start", "t_end", "pdf_time", "lattice", "box_off",
       "threshold", "snapshot_times", "snapshot_times_main",
       "seeds", "n_seeds", "schedules",
       "best_seed_rule", "figures", "aliases", "cases", "files", "written")
CASE = ("kind", "lattice", "psi0", "n_seeds", "t_start", "t_end", "r2_at_end",
        "r2_trajectory_mean", "survival", "seeds", "growth", "composition",
        "finite_size", "composition_series", "snapshots", "psi_pdf",
        "schedule", "r2_curve", "r2_at_times")
check(all(k in m for k in TOP) and m["label"] == "train_t0500_pred_t03000"
      and m["checkpoint_fingerprint"] == FP and m["train_t_end"] == 500.0
      and m["t_start"] == 100.0 and m["t_end"] == 3000.0
      and m["lattice"] == LAT and m["box_off"] == 0.0
      and m["threshold"] == P.R2_THRESHOLD
      and set(m["figures"]) == {"best_seed", "seed_average", "data"}
      and m["aliases"] == {f"l{LAT}": "psi0p0.00"}
      and sorted(m["cases"]) == sorted(TOKENS)
      and set(m["files"]) == set(got),
      "manifest.json has the flat schema's top-level keys, the cases by token, the "
      "base box as an alias of the critical case, and the files it lists")
c = m["cases"]["psi0p0.20"]
check(all(k in c for k in CASE) and c["kind"] == "composition"
      and m["cases"]["l32"]["kind"] == "box"
      and set(c["r2_at_end"]) >= {"mean", "sd", "median", "best", "worst"}
      and set(c["seeds"]) == {"best", "median", "worst"}
      and set(c["survival"]) >= {"threshold", "mean", "sd",
                                 "n_survive_to_end"}
      and set(c["schedule"]) >= {"kind", "dt", "steps"}
      and c["snapshots"]["panels"] and "snapshots_long" not in c,
      "and each case has the flat schema's entry: R2 at the end and its spread, the "
      "best / median / worst seed, survival, the schedule and snapshots")
BK1 = jload(os.path.join(CACHE1, "extras", S.BOOKKEEPING))
check(sorted(BK1["compositions"]) == ["+0.00", "+0.20", "+0.40", "-0.20",
                                      "-0.40"]
      and sorted(BK1["sizes"]) == [str(LAT), "32"]
      and sorted(BK1["grid"]) == ["+0.20_l32", "+0.40_l32", "-0.20_l32",
                                  "-0.40_l32"]
      and m["cases"]["psi0m0.20_l32"]["kind"] == "box"
      and m["cases"]["psi0m0.20_l32"]["lattice"] == 32
      and all("r2_at_end" in m["cases"][t] for t in TOKENS)
      and m["cases"]["psi0m0.20_l32"]["token"] == "psi0m0.20_l32"
      and BK1["grid"]["-0.20_l32"]["files"][0]
      == "l32/snapshots_psi0m0.20.png"
      and all(f.startswith("l32/") and f.endswith("_psi0m0.20.png")
              for f in BK1["grid"]["-0.20_l32"]["files"]),
      "the keys: psi0m0.40 ... psi0p0.40 and l32 as before (a reader of "
      "man['cases']['psi0p0.00']['r2_at_end'] is unchanged), "
      "psi0m0.20_l32 for another composition at another size, and in the "
      "bookkeeping the kinds compositions, sizes and grid")
check(m["size_folders"] == [f"l{LAT}", "l32"]
      and m["pooled_r2_at_end"]["common_6_cases"] is None
      and m["pooled_r2_at_end"]["all_cases"]["cases"] == 10
      and m["pooled_r2_at_end"]["all_cases"]["seeds"] == 30
      and sorted(m["pooled_r2_at_end"]["per_size"]) == [f"l{LAT}", "l32"]
      and m["pooled_r2_at_end"]["per_size"]["l32"]["cases"] == 5,
      "manifest.json: the size folders, and the pooled R2 at the end over "
      "all the cases and per size (the common 6 cases only when a run has "
      "all of them)")
check(all(f"l{LAT}/" + n in m["files"] for n in S.SIZE_FILES)
      and sorted(set(m["files"])) == sorted(m["files"]),
      "and lists the files of each size folder")
head1 = text(os.path.join(RUN1, "composition.csv")).splitlines()
check(head1[0].split(",")[-1] == "l"
      and {ln.split(",")[-1] for ln in head1[1:]} == {str(LAT), "32"},
      "composition.csv has a column for the box size l")
check(m["snapshot_times_main"]["psi0p0.00"] == [100.0, 500.0, 1000.0, 2000.0,
                                                3000.0]
      and "snapshot_times_long" not in m,
      "the snapshots are the columns of the one figure, and there is no "
      "long list")

print("\n3. the same bytes from the records, in every mode of make_submission")
setup()
P.evaluate_seed = P.evaluate_seeds = no_rollout
RE1 = os.path.join(BASE, "redraw1")
res, _ = make_submission_main(["--records", CACHE1, "--out", RE1])
check(res is None and tree(RE1, ("manifest.json", "report.txt"))
      == tree(RUN1, ("manifest.json", "report.txt")),
      "a redraw from the records is the run's flat set, byte for byte "
      "(figures, CSV, the sweep images too)")
check(files_under(RE1) == FLAT,
      "and has no more or fewer files")

# a results root with two complete runs and a hand-written folder
OV = os.path.join(BASE, "overview", "Results")
for label in ("train_t0300_lap_pred_t03000", "train_t0500_lap_pred_t03000"):
    shutil.copytree(RUN1, os.path.join(OV, label))
os.makedirs(os.path.join(OV, "notes"))
with open(os.path.join(OV, "notes", "a.txt"), "w") as fh:
    fh.write("mine")
runs_before = {f: v for f, v in tree(OV).items()
               if f.startswith(("train_", "notes"))}
fno_paths.RESULTS = OV
res, log = make_submission_main(["--overview"])
top = sorted(f for f in files_under(OV)
             if not f.startswith(("train_", "notes")))
# the critical quench is comparison.png; no comparison file repeats it
CRIT = S.case_token(S.P.CRITICAL_OFF)


def place(token):
    """The (size folder, composition) of a token that has one."""
    got = S.token_place(token)
    assert got is not None, token
    return got


CMP = sorted(
    [f"comparisons/{place(t)[0]}/comparison_{place(t)[1]}.png"
     for t in TOKENS if t != CRIT]
    + [f"comparisons/l{n}/{S.NOTES_FILE}" for n in SIZES])
check(res is None and top == sorted(["README.md", "comparison.png",
                                     S.NOTES_FILE] + CMP)
      and {f: v for f, v in tree(OV).items()
           if f.startswith(("train_", "notes"))} == runs_before,
      f"--overview writes README.md, comparison.png and a "
      f"comparisons/l<size>/comparison_<composition>.png (with a "
      f"figure_notes.txt for each size) for each of the other 9 cases (these "
      f"runs hold more than the critical quench), and draws nothing in any "
      f"run folder: {top}")
ov_first = {f: read(os.path.join(OV, *f.split("/"))) for f in top}
for f in top:
    os.remove(os.path.join(OV, *f.split("/")))
res, log = make_submission_main(["--index"])
check(res is None
      and {f: read(os.path.join(OV, *f.split("/"))) for f in top}
      == ov_first,
      "--index is an alias of --overview: the same files, byte for byte")
idx = text(os.path.join(OV, "README.md"))
check("`train_t0300_lap_pred_t03000/`" in idx
      and "`train_t0500_lap_pred_t03000/`" in idx
      and "- `notes/`" in idx and "| psi0p0.20 |" in idx
      and "| psi0p0.20_l32 |" in idx,
      "the README lists both runs with their cases, and the other folder")

print("\n4. the default mode redraws every complete run")
setup()
P.evaluate_seed = P.evaluate_seeds = no_rollout
W4 = os.path.join(BASE, "default", "Work")
R4 = os.path.join(BASE, "default", "Results")
for label in ("train_t0500_pred_t03000", "train_t0300_pred_t03000"):
    shutil.copytree(CACHE1, os.path.join(W4, "rollout_cache", label))
bad = os.path.join(W4, "rollout_cache", "train_t0100_pred_t03000")
shutil.copytree(CACHE1, bad)
os.remove(os.path.join(bad, "records", f"psi0+0.20_l{LAT}",
                       sorted(os.listdir(os.path.join(
                           bad, "records", f"psi0+0.20_l{LAT}")))[0]))
os.makedirs(os.path.join(W4, "rollout_cache", "no_meta_here"))
fno_paths.WORK, fno_paths.RESULTS = W4, R4
found = S.discover()
check(sorted(found) == ["train_t0100_pred_t03000", "train_t0300_pred_t03000",
                        "train_t0500_pred_t03000"]
      and found["train_t0500_pred_t03000"][1]
      and not found["train_t0100_pred_t03000"][1]
      and "psi0+0.20" in found["train_t0100_pred_t03000"][2],
      "discover: the folders with a meta.json, complete or not (and why)")
res, log = make_submission_main([])
check(res is None and "[skip] train_t0100_pred_t03000" in log,
      "no argument: every complete run redrawn, an incomplete one skipped "
      "with its reason")
two = sorted(d for d in os.listdir(R4)
             if os.path.isdir(os.path.join(R4, d)) and d != "comparisons")
t4a, t4b = (tree(os.path.join(R4, two[1]), ("manifest.json", "report.txt")),
            tree(RUN1, ("manifest.json", "report.txt")))
check(two == ["train_t0300_pred_t03000", "train_t0500_pred_t03000"]
      and all(files_under(os.path.join(R4, d)) == FLAT for d in two)
      and t4a == t4b,
      "both complete runs have their run folder in Results/<label>/, the "
      "figures as the run drew them")
check(sorted(f for f in files_under(R4)
             if not f.startswith(("train_", "notes")))
      == sorted(["README.md", "comparison.png", S.NOTES_FILE] + CMP),
      "and then the overview: README.md, comparison.png, figure_notes.txt "
      "and the comparison "
      "of each case")
before_run = tree(os.path.join(R4, two[0]))
res, log = make_submission_main(["--run", "train_t0300_pred_t03000"])
check(res is None
      and tree(os.path.join(R4, two[0]), ("manifest.json", "report.txt"))
      == {f: v for f, v in before_run.items()
          if f not in ("manifest.json", "report.txt")}
      and "[redraw] train_t0300_pred_t03000" in log
      and "[redraw] train_t0500_pred_t03000" not in log,
      "--run LABEL redraws just that run (and the overview)")
res, log = make_submission_main(["--run", "train_t0100_pred_t03000"])
check(isinstance(res, str) and "not complete" in res,
      "and refuses an incomplete one")

print("\n5. extras/ is not the cache: no scan or loader looks into it")
setup()
PR = os.path.join(BASE, "probe")
shutil.copytree(CACHE1, PR)
meta = P.read_meta(PR)
status = P.records_status(PR, meta)
assert meta is not None
for stray in ("stray.npz", f"psi0+0.00_l{LAT}/seed99999.npz",
              f"records/psi0+0.00_l{LAT}/seed1.npz"):
    path = os.path.join(PR, "extras", *stray.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"not a rollout")
check(P.read_meta(PR) == meta and P.records_status(PR, meta) == status
      and all(status.values()),
      "meta.json and the status of the records are the same with strays in "
      "extras/, even under extras/records/")
P.RESULT_CACHE, P.EVAL_MODE, P.SIZE_SWEEP = PR, "critical", (LAT,)
check(P.cache_is_old(PR) is False,
      "cache_is_old does not see npz files in extras/")
old = os.path.join(PR, "fno_ch_l24_seed5_0123456789ab.npz")
with open(old, "wb") as fh:
    fh.write(b"another model's rollout")
check(P.cache_is_old(PR) is True,
      "while one at the top of the cache folder counts")
os.remove(old)
before = tree(PR)
quiet(P.drop_case_cache, 0.0, None, (5,))
quiet(P.remove_empty_cache)
check(tree(PR) == before and os.path.isdir(os.path.join(PR, "extras")),
      "neither drop_case_cache nor remove_empty_cache touches extras/, "
      "records or meta.json")
only_extras = os.path.join(BASE, "only_extras")
os.makedirs(os.path.join(only_extras, "extras", "psi0+0.00_l24"))
with open(os.path.join(only_extras, "extras", "x.json"), "w") as fh:
    fh.write("{}")
P.RESULT_CACHE = only_extras
buf = _io.StringIO()
with contextlib.redirect_stdout(buf):
    P.remove_empty_cache()
check(os.path.isfile(os.path.join(only_extras, "extras", "x.json"))
      and "removed" not in buf.getvalue(),
      "a cache folder that holds only extras/ is not removed")
rec = P.record_file(PR, f"psi0+0.00_l{LAT}", P.EVAL_SEEDS[0])
check(os.path.isfile(rec) and "extras" not in rec,
      "and the record files are under records/, not extras/")

print("\n6. a time grid that is not a fixed step")
setup()
GRID = np.unique(np.round(np.geomspace(100.0, 3000.0, 60), 1))
GRID = GRID[GRID > 100.0]


def sched_result(seed, off=0.0, with_start=True, die=None):
    """A rollout on a schedule: steps from t = 100 that grow, so t[1] - t[0]
    is nothing like the spacing elsewhere."""
    g = np.random.default_rng([seed, 5])
    t = GRID.copy()
    r2 = np.clip(1.0 - (t - 100.0) / 9000.0 - 0.01 * g.random(), -1, 1)
    mp = np.full(t.size, off) + 1e-7 * (t - 100.0)
    if die is not None:
        r2 = np.where(t > die, P.R2_FLOOR, r2)
        mp = np.where(t > die, np.nan, mp)
    snaps = {}
    for tt in (100.0, 500.0, 1000.0, 2000.0, 3000.0):
        f = np.tanh(3 * g.standard_normal((16, 16))) + off
        snaps[tt] = (f, np.clip(f + 0.05 * g.standard_normal(f.shape),
                                -1.05, 1.05))
    dt = t[::3]
    L = np.linspace(5, 14, dt.size)
    r = {"times": t, "r2": r2, "mass_drift": np.full(t.size, 1e-9),
         "mean_true": np.full(t.size, off), "mean_pred": mp,
         "diag_times": dt, "energy_true": np.linspace(1, 0.5, dt.size),
         "energy_pred": np.linspace(1, 0.5, dt.size),
         "t_stop": float(t[-1]), "stopped_early": False,
         "diverged_at": None if die is None else float(die),
         "L_true": {k: L for k in ("L_zero", "L_half")},
         "L_pred": {k: L * 1.01 for k in ("L_zero", "L_half")},
         "snapshots": snaps, "gt_seconds": 1.0, "gt_cached": False,
         "gt_steps": 1, "fno_seconds": 0.5, "fno_steps": 1900,
         "video_t": np.asarray([]), "video_true": None, "video_pred": None}
    if with_start:
        r["t_start"] = 100.0
    return r


res6 = [sched_result(s) for s in (1, 2, 3)]
res6[2] = sched_result(3, die=1500.0)
check(S.grid_index(GRID, float(GRID[7])) == 7
      and S.grid_index(GRID, 3000.0) == len(GRID) - 1
      and S.grid_index(GRID, 1500.0) in (None, int(np.argmin(
          np.abs(GRID - 1500.0))))
      and S.grid_index(GRID, 5000.0) is None,
      "grid_index finds a time on the schedule by value, and no time off it")
tiny = np.array([101.0, 102.0, 103.0])
check(S.grid_index(tiny, 102.0) == 1 and S.grid_index(tiny, 102.4) == 1
      and S.grid_index(tiny, 107.0) is None
      and abs(S.grid_slack(tiny) - 0.5) < 1e-12
      and abs(S.grid_slack(GRID) - 0.5 * (GRID[-1] - GRID[-2])) < 1e-12,
      "a fixed grid behaves as it did; the slack is half the last spacing, "
      "whatever the earlier ones were")
check(S.result_start(res6[0]) == 100.0
      and S.result_start(sched_result(1, with_start=False))
      == float(GRID[0] - (GRID[1] - GRID[0]))
      and S.result_start({"times": np.arange(101.0, 11.0 + 200.0)}) == 100.0,
      "result_start is the result's t_start when it has one, else one "
      "spacing before its first sample")
snaps_, t_end, t_start = S.rollout_times(res6)
check(snaps_ == [100.0, 500.0, 1000.0, 2000.0, 3000.0] and t_end == 3000.0
      and t_start == 100.0,
      f"rollout_times on a schedule: {snaps_}, {t_end}, {t_start}")
r2t = S.r2_at_times(res6, (500.0, 2000.0, 3000.0, 9999.0))
i500 = int(np.argmin(np.abs(GRID - 500.0)))
want500 = [res6[0]["r2"][i500], res6[1]["r2"][i500], res6[2]["r2"][i500]]
check(set(r2t) == {"500", "2000", "3000"}
      and abs(r2t["500"]["mean"] - float(np.mean(want500))) < 1e-12
      and r2t["500"]["n"] == 3 and r2t["500"]["diverged"] == 0
      and r2t["2000"]["diverged"] == 1
      and abs(r2t["2000"]["mean"] - float(np.mean(
          [res6[0]["r2"][int(np.argmin(np.abs(GRID - 2000.0)))],
           res6[1]["r2"][int(np.argmin(np.abs(GRID - 2000.0)))],
           P.R2_FLOOR]))) < 1e-12,
      "r2_at_times reads the schedule's samples by time; a diverged seed "
      "stays in the mean at R2 = -1")
tt, mean, sd, overall = S.mean_sd_curve(res6)
check(tt.size == GRID.size and np.allclose(tt, GRID)
      and abs(mean[-1] - float(np.mean([res6[0]["r2"][-1],
                                        res6[1]["r2"][-1],
                                        P.R2_FLOOR]))) < 1e-12
      and sd[-1] > 0 and np.isfinite(overall),
      "mean_sd_curve is positional on the common schedule: the diverged "
      "seed is at the floor at the end")
comp = S.composition_series(res6)
assert comp is not None
check(abs(comp["full"][0][0] - GRID[0]) < 1e-12
      and comp["full"][0].size == GRID.size,
      "composition_series keeps the schedule's own times")
w = S.data_window(res6)
check(w["start"] == 100.0 and w["end"] == 3000.0
      and w["snapshots"] == [100.0, 500.0, 1000.0, 2000.0, 3000.0],
      "data_window: the schedule's start and end")
EX6 = os.path.join(BASE, "x6")
OUT6 = os.path.join(BASE, "out6")
quiet(S.write_case, res6, (1, 2, 3), 0.0, LAT, OUT6, key="k", videos=False,
      root=OUT6, extras=EX6, token="psi0p0.00")
npz1 = os.path.join(BASE, "with_start.npz")
npz2 = os.path.join(BASE, "without_start.npz")
P._save_result(npz1, sched_result(1))
P._save_result(npz2, sched_result(1, with_start=False))
with np.load(npz1) as z1, np.load(npz2) as z2:
    names = (set(z1.files), set(z2.files))
check(names[0] - names[1] == {"t_start"} and names[1] <= names[0]
      and P._load_result(npz1)["t_start"] == 100.0
      and P._load_result(npz2)["t_start"] is None
      and sorted(f for f in files_under(OUT6) if f.endswith(".png")),
      "a result carries its t_start through the npz, a record without one "
      "stays readable, and a flat set is drawn from a schedule")

print("\n7. a copy of the real records, redrawn by the default mode")
REAL = REAL_RECORDS      # FNO_TEST_RECORDS, else the laptop's folder
if REAL is None:
    print("     (the real records are not here; skipped.  FNO_TEST_RECORDS "
          "names a rollout-cache folder to use)")
else:
    setup()
    P.evaluate_seed = P.evaluate_seeds = no_rollout
    W7 = os.path.join(BASE, "real", "Work")
    R7 = os.path.join(BASE, "real", "Results")
    COPY = os.path.join(W7, "rollout_cache", os.path.basename(REAL))
    shutil.copytree(REAL, COPY, ignore=shutil.ignore_patterns("extras"))
    fno_paths.WORK, fno_paths.RESULTS = W7, R7
    meta7 = jload(os.path.join(COPY, "meta.json"))
    found7 = S.discover()
    label7 = os.path.basename(REAL)
    have = sorted(d for d in os.listdir(os.path.join(COPY, "records"))
                  if os.path.isdir(os.path.join(COPY, "records", d)))
    if found7[label7][1]:
        complete = True
    else:
        # the real run planned two cases it never rolled out: the copy is
        # given the plan it did complete, so the default mode takes it
        meta7["planned_cases"] = have
        meta7["cases"] = {k: v for k, v in meta7["cases"].items()
                          if k in have}
        with open(os.path.join(COPY, "meta.json"), "w") as fh:
            json.dump(meta7, fh)
        complete = S.discover()[label7][1]
    check(complete and len(have) >= 1,
          f"the copy's records complete the cases they hold: {have}")
    t0 = time.time()
    res, log7 = make_submission_main([])
    print(f"     ({time.time() - t0:.0f}s)")
    out7 = os.path.join(R7, label7)
    got7 = files_under(out7) if os.path.isdir(out7) else []
    tok7 = [S.case_token(float(c[4:9]),
                         None if c.endswith("_l128") else int(c.split("_l")[1]))
            for c in have]
    kinds7 = [k for k in S.FIG_KINDS]
    def place7(t):
        """(size folder, composition) in the real run, whose lattice is the
        meta.json's, not this suite's."""
        if t.startswith("psi0") and "_l" not in t:
            return f"l{meta7['lattice']}", t
        return place(t)

    sizes7 = sorted({int(place7(t)[0][1:]) for t in tok7})
    comps7 = {n: {place7(t)[1] for t in tok7
                  if place7(t)[0] == f"l{n}"} for n in sizes7}
    want7 = sorted([f"{place7(t)[0]}/{k}_{place7(t)[1]}.png"
                    for t in tok7 for k in kinds7]
                   + [f"l{n}/{f}" for n in sizes7 for f in S.SIZE_FILES]
                   + [f for f in S.RUN_FILES
                      if ("per_size" not in f
                          or any(len(v) > 1 for v in comps7.values()))
                      and ("per_case" not in f
                           or any(sum(c in v for v in comps7.values()) > 1
                                  for c in set().union(*comps7.values())))])
    check(res is None and got7 == want7,
          f"the run folder appears for its {len(tok7)} cases {tok7}: "
          f"{len(got7)} files (unexpected "
          f"{sorted(set(got7) ^ set(want7))[:4]})")
    m7 = jload(os.path.join(out7, "manifest.json"))
    parsed7 = fno_paths.parse_label(label7)
    t_train7 = None if parsed7 is None else parsed7["t_train"]
    check(sorted(m7["cases"]) == sorted(tok7)
          and m7["n_seeds"] == len(meta7["eval_seeds"])
          and m7["train_t_end"] == t_train7 and m7["t_end"] == meta7["end"],
          f"manifest.json: {sorted(m7['cases'])}, {m7['n_seeds']} seeds, "
          f"trained to {m7['train_t_end']}, rolled out to {m7['t_end']}")
    check(sorted(f for f in os.listdir(R7) if os.path.isfile(
              os.path.join(R7, f))) == ["README.md"]
          and os.path.isdir(os.path.join(COPY, "extras"))
          and [f for f in files_under(COPY) if not f.startswith("extras/")]
          == [f for f in files_under(REAL) if not f.startswith("extras/")],
          "the overview was written, the extras went to the copy's cache "
          "folder, and the records are where they were")
    shutil.rmtree(os.path.join(BASE, "real"), ignore_errors=True)

print("\n8. the training curves: drawn with a run, and by the overview")
setup()
S.WRITE_TRAINING = True                # the suites keep it off (_paths.py)
R8 = os.path.join(BASE, "t8")          # a root of its own, as --smoke has
P.ROOT = R8
CK8 = os.path.join(R8, "models", "train_t0500", "fno_ch.pt")
HIST8 = os.path.join(os.path.dirname(CK8), "fno_ch_history.json")
os.makedirs(os.path.dirname(CK8))
with open(CK8, "wb") as fh:
    fh.write(b"a checkpoint of this suite, with a history beside it")


def score_of(epoch):
    return 0.9 - 0.8 / epoch


def fixture_history(n=24, diverged=None):
    """A training history as fno_train writes it: an entry for every epoch,
    the unroll depth stepping 1, 2, 4, the rollout validation (score,
    survival, in-range R2) at every second epoch, and ``diverged`` the
    epoch whose entry is the blow-up of an epoch rolled back."""
    rows = []
    for e in range(1, n + 1):
        if e == diverged:
            rows.append({"epoch": e, "diverged": True, "train": 2.6e4,
                         "skipped": 0, "stalled": False, "unphysical": False,
                         "delta_rel": 0.07, "rolled_back_to": e - 1})
            continue
        row = {"epoch": e, "unroll": 1 if e < 4 else 2 if e < 8 else 4,
               "train": 0.08 / e ** 0.5, "val1": 3e-5 + 1e-6 * (e % 5),
               "delta_rel": 0.06, "seconds": 1.0, "skipped": 0, "spiked": 0,
               "nonfinite_grad": 0}
        if e % 2 == 0:
            row.update(score=score_of(e), survival=100.0 * e,
                       per_seed=[100.0 * e, 120.0 * e, 90.0 * e],
                       survival_interp=100.0 * e, in_range=1.0 - 0.5 / e)
        rows.append(row)
    return rows


def put_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)


put_json(HIST8, fixture_history(diverged=9))
BEST8 = {"epoch": 20, "score": score_of(20), "survival": 9000.0,
         "accuracy": 0.99}
P.load_checkpoint = lambda path, device=None: (FakeModel(), {"best": BEST8})
RES8 = os.path.join(R8, "Results")
RUN8 = os.path.join(RES8, "train_t0500_pred_t03000")
code, log = quiet(P.main, [
    "--model", CK8, "--results-dir", RUN8, "--result-cache",
    os.path.join(R8, "cache"), "--eval-mode", "all", "--sizes", f"{LAT},32",
    "--seeds", "3", "--no-gt-cache", "--no-diagnostics", "--no-video"])
got8 = files_under(RUN8)
check(code == 0 and got8 == sorted(FLAT + [S.TRAINING_FIG, S.NOTES_FILE])
      and f"[saved] {os.path.join(RUN8, S.TRAINING_FIG)}" in log,
      f"a run whose model has a history writes the flat set and "
      f"{S.TRAINING_FIG}: {code}, {len(got8)} files (unexpected "
      f"{sorted(set(got8) ^ set(FLAT + [S.TRAINING_FIG]))[:4]})")
png8 = read(os.path.join(RUN8, S.TRAINING_FIG))
m8 = jload(os.path.join(RUN8, "manifest.json"))
check(png8[:8] == b"\x89PNG\r\n\x1a\n" and len(png8) > 20000
      and set(m8["files"]) == set(got8) and S.TRAINING_FIG in m8["files"],
      f"it is a PNG ({len(png8)} bytes), and the manifest lists it with the "
      f"other files")
check(m8["checkpoint_best"] == BEST8
      and S.selected_checkpoint(RUN8, m8) == {
          "epoch": 20, "score": score_of(20), "survival": 9000.0},
      "manifest.json holds the checkpoint selection (epoch, score, "
      "survival, accuracy), which is what marks the epoch in the figure")
check(sorted(f for f in os.listdir(RES8)
             if os.path.isfile(os.path.join(RES8, f))) == ["README.md"],
      "one run: the overview at its end has no comparison yet")

print("\n   a model with no history, or a history of another training")
bare = os.path.join(R8, "bare")
os.makedirs(bare)
no_hist, log = quiet(S.draw_training_curves, bare, "train_t0900_pred_t03000",
                     os.path.join(R8, "models", "nowhere", "fno_ch.pt"),
                     BEST8, 1234, R8)
check(no_hist is False and not os.path.exists(
          os.path.join(bare, S.TRAINING_FIG))
      and "[training] train_t0900_pred_t03000: no history file" in log,
      "no history beside the checkpoint or under the root: nothing is drawn, "
      "the log says so, and nothing fails")
put_json(os.path.join(R8, "models", "train_t0900", "fno_ch_history.json"),
         fixture_history(8))
other, log = quiet(S.draw_training_curves, bare, "train_t0900_pred_t03000",
                   os.path.join(R8, "models", "nowhere", "fno_ch.pt"),
                   BEST8, 1234, R8)
check(other is True and os.path.isfile(os.path.join(bare, S.TRAINING_FIG))
      and "does not agree with the selected epoch 20" in log,
      "a checkpoint that is not there falls back to the history of the "
      "train label; its 8 epochs do not hold the selected epoch 20, so the "
      "figure is drawn with no epoch marked, and the log says why")
os.remove(os.path.join(bare, S.TRAINING_FIG))
put_json(os.path.join(R8, "models", "train_t0900", "fno_ch_history.json"),
         {"not": "a history"})
none, log = quiet(S.draw_training_curves, bare, "train_t0900_pred_t03000",
                  os.path.join(R8, "models", "nowhere", "fno_ch.pt"),
                  BEST8, 1234, R8)
check(none is False and "holds no epochs" in log and not os.path.exists(
          os.path.join(bare, S.TRAINING_FIG)),
      "a history that is not a list of epochs is reported, not drawn")
S.WRITE_TRAINING = False
off, log = quiet(S.draw_training_curves, bare, "train_t0500_pred_t03000", CK8,
                 BEST8, 1234, R8)
S.WRITE_TRAINING = True
check(off is False and log == "" and not os.path.exists(
          os.path.join(bare, S.TRAINING_FIG)),
      "with WRITE_TRAINING off it draws nothing and says nothing")

print("\n   the overview draws it for the runs that lack it")
# an older run: no checkpoint_best in its manifest (report.txt has the
# selection), the figure not drawn yet, and its model found by its label
OLD8 = os.path.join(RES8, "train_t0300_pred_t03000")
shutil.copytree(RUN8, OLD8)
os.remove(os.path.join(OLD8, S.TRAINING_FIG))
old_m = {k: v for k, v in m8.items() if k != "checkpoint_best"}
old_m.update(checkpoint="gone/fno_ch.pt",
             files=[f for f in m8["files"] if f != S.TRAINING_FIG])
put_json(os.path.join(OLD8, "manifest.json"), old_m)
put_json(os.path.join(R8, "models", "train_t0300", "fno_ch_history.json"),
         fixture_history(24))
# a run whose model has no history, an unreadable manifest, a hand-written
# folder, and a second run of the first run's model (it has the figure)
NOH8 = os.path.join(RES8, "train_t0100_pred_t03000")
shutil.copytree(RUN8, NOH8)
os.remove(os.path.join(NOH8, S.TRAINING_FIG))
put_json(os.path.join(NOH8, "manifest.json"), dict(
    old_m, checkpoint=os.path.join(R8, "models", "nowhere", "fno_ch.pt")))
BROKEN8 = os.path.join(RES8, "broken_run")
os.makedirs(BROKEN8)
with open(os.path.join(BROKEN8, "manifest.json"), "w") as fh:
    fh.write("{")
os.makedirs(os.path.join(RES8, "notes"))
with open(os.path.join(RES8, "notes", "a.txt"), "w") as fh:
    fh.write("mine")
TWIN8 = os.path.join(RES8, "train_t0500_lap_pred_t03000")
shutil.copytree(RUN8, TWIN8)
os.remove(os.path.join(TWIN8, S.TRAINING_FIG))
twin_m = dict(m8, files=[f for f in m8["files"] if f != S.TRAINING_FIG])
put_json(os.path.join(TWIN8, "manifest.json"), twin_m)

before = {p: read(os.path.join(p, "manifest.json"))
          for p in (NOH8, BROKEN8)}
check(S.selected_checkpoint(OLD8, old_m) == {
          "epoch": 20, "score": score_of(20), "survival": 9000.0}
      and S.report_selection(OLD8).get("epoch") == "20",
      "a manifest without checkpoint_best: the selection is read from the "
      "'checkpoint selection' block of report.txt")
os.remove(os.path.join(OLD8, "report.txt"))
check(S.selected_checkpoint(OLD8, old_m) is None
      and S.report_selection(OLD8) == {},
      "and with neither, no epoch is known (None), not a guess")
shutil.copy(os.path.join(RUN8, "report.txt"), OLD8)
found, log = quiet(S.write_results_index, RES8, R8)
check(found["training"] == ["train_t0300_pred_t03000",
                            "train_t0500_lap_pred_t03000",
                            "train_t0500_pred_t03000"]
      and all(os.path.isfile(os.path.join(RES8, d, S.TRAINING_FIG))
              for d in found["training"])
      and not os.path.exists(os.path.join(NOH8, S.TRAINING_FIG))
      and not os.path.exists(os.path.join(BROKEN8, S.TRAINING_FIG)),
      f"the overview drew the figure of the three runs whose model has a "
      f"history (beside the checkpoint, or by the train label), and of no "
      f"other: {found['training']}")
check(read(os.path.join(RUN8, S.TRAINING_FIG)) == png8
      and read(os.path.join(TWIN8, S.TRAINING_FIG)) == png8
      and "training curves of train_t0500_lap" in " ".join(
          S.read_notes(os.path.join(TWIN8, S.NOTES_FILE))[S.TRAINING_FIG])
      and S.read_notes(os.path.join(TWIN8, S.NOTES_FILE))
      != S.read_notes(os.path.join(OLD8, S.NOTES_FILE)),
      "the same history and selection give the same bytes (the overview's "
      "redraw of the run's own figure is the figure the run drew, and the "
      "figure has no title); the twin's notes name its own train label")
now_old, now_twin = (jload(os.path.join(p, "manifest.json"))
                     for p in (OLD8, TWIN8))
check(now_old["files"] == sorted(old_m["files"] + [S.TRAINING_FIG])
      and now_twin["files"] == sorted(twin_m["files"] + [S.TRAINING_FIG])
      and {k: v for k, v in now_old.items() if k != "files"}
      == {k: v for k, v in old_m.items() if k != "files"}
      and jload(os.path.join(RUN8, "manifest.json")) == m8,
      "the manifest of a folder it drew into now lists the figure, and "
      "nothing else of it changed; one that had it is as it was")
check({p: read(os.path.join(p, "manifest.json")) for p in (NOH8, BROKEN8)}
      == before
      and "[training] train_t0100_pred_t03000: no history file" in log
      and os.listdir(BROKEN8) == ["manifest.json"]
      and os.listdir(os.path.join(RES8, "notes")) == ["a.txt"],
      "a run with no history, an unreadable manifest and a hand-written "
      "folder are left as they were (bytes), the first with a line in the log")
top8 = sorted(f for f in files_under(RES8)
              if not f.startswith(("train_", "notes", "broken_run")))
check(top8 == sorted(["README.md", "comparison.png", S.NOTES_FILE,
                      S.LAPLACIAN_FIG] + CMP)
      and sorted(found["cases"]) == sorted(t for t in TOKENS if t != CRIT)
      and found["comparison"] is True
      and not [f for f in os.listdir(RES8) if ".partial" in f],
      f"and the comparison of every case: {len(top8)} files, "
      f"no temporary left (laplacian_on_off.png too: the runs trained to "
      f"t = 500 are SIZE_END)")
idx8 = text(os.path.join(RES8, "README.md"))
check("`training_curves.png`" in idx8
      and "the same for each other case in "
          "`comparisons/l<size>/comparison_<composition>.png` "
          f"(`{found['cases'][0]}` ... `{found['cases'][-1]}`)" in idx8,
      "README.md says what the new files are")

print("\n   switched off, the overview draws no training figure")
S.WRITE_TRAINING = False
os.remove(os.path.join(TWIN8, S.TRAINING_FIG))
twin_before = read(os.path.join(TWIN8, "manifest.json"))
found, log = quiet(S.write_results_index, RES8, R8)
check(found["training"] == [] and not os.path.exists(
          os.path.join(TWIN8, S.TRAINING_FIG))
      and read(os.path.join(TWIN8, "manifest.json")) == twin_before
      and "[training]" not in log,
      "WRITE_TRAINING = False: the overview leaves the run folders alone")
S.WRITE_TRAINING = True

print("\n9. a run folder in the old flat layout is not written into")
setup()
OLD9 = os.path.join(BASE, "old_flat", "Results", "train_t0500_pred_t03000")
CACHE9 = os.path.join(BASE, "old_flat", "Work", "rollout_cache",
                      "train_t0500_pred_t03000")
os.makedirs(OLD9)
for f in ("snapshots_psi0p0.20.png", "snapshots_l256.png",
          "domain_growth_psi0m0.40.png", "manifest.json"):
    with open(os.path.join(OLD9, f), "wb") as fh:
        fh.write(b"an old file")
before9 = tree(OLD9)
check(S.flat_case_files(OLD9) == ["domain_growth_psi0m0.40.png",
                                  "snapshots_l256.png",
                                  "snapshots_psi0p0.20.png"]
      and S.flat_case_files(RUN1) == []
      and S.flat_case_files(os.path.join(BASE, "none_here")) == [],
      "flat_case_files finds the per-case figures at the top of a folder "
      "of the old layout, and none in a folder of the new one")
code9, log9 = quiet(P.main, [
    "--model", CKPT, "--results-dir", OLD9, "--result-cache", CACHE9,
    "--eval-mode", "critical", "--sizes", f"{LAT}", "--seeds", "2",
    "--no-gt-cache", "--no-diagnostics", "--no-video"])
check(isinstance(code9, str) and "old flat layout" in code9
      and "Archive/" in code9 and "Nothing was written" in code9
      and tree(OLD9) == before9
      and not os.path.exists(os.path.join(CACHE9, "meta.json"))
      and not os.path.exists(os.path.join(CACHE9, "extras")),
      "a prediction into such a folder stops before it writes anything, "
      "and says to move the folder to Archive/ first; the folder, the "
      "records and the extras are as they were")
exit9, _ = quiet(S.write_run_outputs, OLD9, os.path.join(BASE, "x9"), {})
check(isinstance(exit9, str) and "old flat layout" in exit9
      and tree(OLD9) == before9 and not os.path.exists(os.path.join(
          BASE, "x9")),
      "so does writing the run-wide files into it")
code9, _ = quiet(make_submission_main, ["--records", CACHE1, "--out", OLD9])
check(tree(OLD9) == before9
      and sorted(os.listdir(OLD9)) == sorted(before9),
      "and a redraw into it: nothing was added, changed or deleted")

setup()
S.WRITE_TRAINING = False
matplotlib.axes.Axes.set_title = _title
check(project_state() == PROJECT,
      "nothing was written into the project, outside _scratch")
print("\nALL PASSED" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
