"""make_submission regenerates a results folder from a result cache, and
--all does so for several runs (sections 17, 18 and 24 of the former one-file
t_submission): the folder, one bad case, the exit status, same_model, a sweep
over mixed lattices, and a broken run among good ones.

Shares tests/_submission_base.py with t_submission (sections 1 to 16) and
t_submission_publish (19 to 23, 25 and 10).  Section 24 builds from the cache
that section 17 makes, as it did in the one file; section 18 also draws into
`es`, which only has to exist (made below)."""
import contextlib
import io as _io
import json
import os
import shutil

import matplotlib.axes
import matplotlib.figure
import numpy as np

from _paths import SCRATCH

import fno_paths as F
from _submission_base import (P, PLAIN, S, SAVED, SUB_BEFORE, TITLES,
                              _set_title, _suptitle, _text, bookkeeping, calls,
                              check, clean, fake_results, files_under, finish,
                              flat, flat_figs, leftovers_tree, manifest,
                              old_names, panel_titles, project_untouched, read,
                              real_growth, restore, run_files, run_main, setup,
                              submission_dirs, tree)

# --- state taken over from t_submission ---------------------------------------
# Section 18 draws mixed.png into `es`, the case folder that section 14 of
# t_submission publishes.  It reads nothing from it; the folder has to exist.
es = os.path.join(SCRATCH, "case_es")
clean(es)
os.makedirs(es)

print("\n17. a regeneration from cache: the folder, one bad case, exit status")
cache = os.path.join(SCRATCH, "result_cache_fake")
bld = os.path.join(SCRATCH, "results_bld")
clean(cache, bld)
restore()
for off, lat in ((-0.4, 128), (0.0, 128), (0.0, 512)):
    tag = "" if off == 0.0 else f"_off{off:+.2f}"
    for sd, r in zip((1, 2, 3), fake_results(off, 3)):
        P._save_result(os.path.join(
            cache, f"fno_ch{tag}_l{lat}_seed{sd}_{'a' * 12}.npz"), r)


def stat_tree(d):
    return {f: (os.path.getsize(os.path.join(d, *f.split("/"))),
                os.path.getmtime(os.path.join(d, *f.split("/"))))
            for f in files_under(d)}


cache_before = (tree(cache), stat_tree(cache))
P.WRITE_VIDEO = False
real_start_log = S.fno_runlog.start_log
S.fno_runlog.start_log = lambda *a, **k: None
BUILT = ("psi0+0.00_l128", "psi0+0.00_l512", "psi0-0.40_l128")
BTOK = ("psi0p0.00", "l512", "psi0m0.40")
XB = os.path.join(bld, "extras")        # a rebuilt folder keeps its extras
try:                                    # beside the figures, in extras/
    with contextlib.redirect_stdout(_io.StringIO()):
        mb = S.build(cache, bld)
    with np.load(os.path.join(XB, S.CURVES_FILE)) as z:
        groups = {tuple(nm.split("__")[:2]) for nm in z.files}
    check(groups == {("compositions", "-0.40"), ("compositions", "+0.00"),
                     ("sizes", "128"), ("sizes", "512")}
          and mb["failed"] == [],
          "build() writes the curve store a later top-up needs")
    check(sorted(d for d in os.listdir(XB)
                 if os.path.isdir(os.path.join(XB, d))) == sorted(BUILT)
          and [f for f in files_under(bld) if not f.startswith("extras/")]
          == sorted(flat_figs(BTOK)
                             + run_files((128, 512), (128,), ("psi0p0.00",)))
          and all(os.path.exists(os.path.join(XB, c, f))
                  for c in BUILT for f in PLAIN)
          and submission_dirs() == SUB_BEFORE,
          "it writes one merged results folder, a folder for each box size, "
          "with the figures of every cached case, and their data in "
          "extras/<case>/; no Submission folder")
    sums, seed_dirs = {}, {}
    for c, sfx in zip(BUILT, ("", "_l512", "_off-0.40")):
        with open(os.path.join(XB, c, f"fno_ch{sfx}_summary.json")) as fh:
            sums[c] = json.load(fh)
        seed_dirs[c] = sorted(d for d in os.listdir(os.path.join(XB, c))
                              if d.startswith("best_seed"))
    bs = sums["psi0+0.00_l128"]["best_seed"]
    check(all(seed_dirs[c] == [f"best_seed{sums[c]['best_seed']}"]
              for c in BUILT)
          and all(sums[c]["record"]["key"] == "a" * 12
                  and sums[c]["record"]["seeds"] == 3 for c in BUILT)
          and all(os.path.exists(os.path.join(XB, c, P.MEAN_R2_FILE))
                  for c in BUILT)
          and sorted(os.listdir(os.path.join(XB, "psi0+0.00_l128",
                                             f"best_seed{bs}")))
          == sorted(f"fno_ch_seed{bs}_{x}" for x in (
              "r2.csv", "summary.json", "r2_and_snapshots.png",
              "kinetics.png", "conservation.png")),
          "each case gets its summary (recording the cache key), its mean R2 "
          "and one best-seed folder with its own kinetics -- no film in it")
    mfile = bookkeeping(XB)
    flat_man = manifest(bld)
    check(mfile["source"] == f"make_submission, regenerated from {cache}"
          and mfile["cache"] == cache
          and flat_man["source"] == mfile["source"] and flat_man["cache"] == cache
          and mfile["sizes"]["128"] == mfile["compositions"]["+0.00"]
          and mfile["sizes"]["128"]["folder"] == "psi0p0.00"
          and flat_man["aliases"] == {"l128": "psi0p0.00"}
          and not old_names(bld),
          "its manifests name its source and cache, and l = 128 is the "
          "critical case's entry and token, with no copy")
    # a cache written before 2026-11-11 stores the old names; it reads the same
    old_x, new_r, old_r = (os.path.join(SCRATCH, n) for n in
                           ("extras_old", "run_new", "run_old"))
    clean(old_x, new_r, old_r)
    shutil.copytree(XB, old_x)
    with open(os.path.join(old_x, S.BOOKKEEPING), encoding="utf-8") as fh:
        book_old = json.load(fh)
    for kind in S.KINDS:
        for ent in book_old.get(kind, {}).values():
            ent[F.OLD_STORED_NAME] = ent.pop("flat")
    with open(os.path.join(old_x, S.BOOKKEEPING), "w", encoding="utf-8") as fh:
        json.dump(book_old, fh)
    with contextlib.redirect_stdout(_io.StringIO()):
        S.write_run_outputs(new_r, XB, {"seed_set": "group"})
        S.write_run_outputs(old_r, old_x, {"seed_set": F.OLD_STORED_NAME})

    def seed_line(d):
        with open(os.path.join(d, "report.txt"), encoding="utf-8") as fh:
            return [ln for ln in fh.read().splitlines() if "test seeds" in ln]

    check(manifest(old_r)["cases"] == manifest(new_r)["cases"]
          and manifest(old_r)["cases"] and seed_line(old_r) == seed_line(new_r)
          and "fixed list" in seed_line(old_r)[0],
          "an old cache (old seed-set value, old per-case key) gives the "
          "same manifest cases and report wording as the new names")
    check(sorted(mb["missing"]) == sorted([
              "psi0=-0.20 l=128", "psi0=+0.20 l=128", "psi0=+0.40 l=128",
              "psi0=+0.00 l=256"])
          and mb["failed"] == [] and mfile["failed_last_run"] == []
          and not any(os.path.exists(os.path.join(XB, f"psi0{x}"))
                      for x in ("-0.20_l128", "+0.00_l256")),
          f"cases the cache does not hold are listed as missing, not failed, "
          f"and get no folder: {mb['missing']}")
    check("missing" not in mfile and "failed" not in mfile
          and mfile.get("last_run", {}).get("model_path") == ""
          and not any(e["run"].get("model_path")
                      for e in mfile["compositions"].values()),
          "the missing and failed lists come back from build() rather than "
          "being written into the manifest, and with no --model no "
          "checkpoint is named")
    with open(os.path.join(XB, "fno_ch_timing.json")) as fh:
        timing = json.load(fh)
    with open(os.path.join(bld, "report.txt"), encoding="utf-8") as fh:
        report = fh.read()
    check(timing["meta"]["device"] == "unknown"
          and timing["meta"]["model_parameters"] is None
          and f"regenerated from {cache}" in report
          and all(f"  {t:<15}" in report for t in BTOK),
          "with no model the timing report says so, and report.txt names "
          "the cache and every case")
    check(P.RESULTS_DIR == SAVED["RESULTS_DIR"]
          and P.MODEL_PATH == SAVED["MODEL_PATH"]
          and P.START_TIME == SAVED["START_TIME"],
          "fno_predict's settings are put back afterwards")
    partial = os.path.join(SCRATCH, "result_cache_partial")
    clean(partial)
    shutil.copytree(cache, partial,
                    ignore=shutil.ignore_patterns("*_off-0.40_*"))
    kept_m04 = {f: v for f, v in tree(bld).items()
                if f.startswith("extras/psi0-0.40_l128/")
                or f.endswith("_psi0m0.40.png")}
    with contextlib.redirect_stdout(_io.StringIO()):
        mb = S.build(partial, bld)
    check("psi0=-0.40 l=128" in mb["missing"] and mb["failed"] == []
          and "-0.40" in mb["compositions"]
          and "last_attempt_failed" not in mb["compositions"]["-0.40"]
          and {f: v for f, v in tree(bld).items()
               if f.startswith("extras/psi0-0.40_l128/")
               or f.endswith("_psi0m0.40.png")} == kept_m04,
          "merging a cache that lacks a case into a folder that has it keeps "
          "that case's entry and files, listed as missing, not marked failed")
    first = tree(bld)

    def growth_fails(results, off, path, lattice=128, data=None, notes=None):
        if off < -0.3:
            raise RuntimeError("deliberate")
        return real_growth(results, off, path, lattice, data=data,
                           notes=notes)

    S.figure_growth = growth_fails
    code = None
    try:
        with contextlib.redirect_stdout(_io.StringIO()):
            S.main(["--cache", cache, "--out", bld])
    except SystemExit as exc:
        code = exc.code
    finally:
        S.figure_growth = real_growth
    mb = bookkeeping(XB)
    now = tree(bld)
    check(mb["failed_last_run"]
          and mb["failed_last_run"][0].startswith("submission for psi0=-0.40")
          and sorted(mb["compositions"]) == ["+0.00", "-0.40"]
          and "last_attempt_failed" in mb["compositions"]["-0.40"]
          and "last_attempt_failed" not in mb["compositions"]["+0.00"]
          and sorted(mb["sizes"]) == ["128", "512"],
          f"one failing case is recorded, keeps its entry marked, and the "
          f"rest are rebuilt: {mb['failed_last_run']}")
    check(all(now[f"extras/psi0-0.40_l128/{f}"]
              == first[f"extras/psi0-0.40_l128/{f}"] for f in PLAIN)
          and all(now[f] == first[f] for f in flat_figs(["psi0m0.40"])),
          "the failed case's previous submission files are untouched")
    check(isinstance(code, str) and "FAILED" in code,
          "and the exit status says a part failed")

    def growth_fails_192(results, off, path, lattice=128, data=None,
                         notes=None):
        if lattice == 512:
            raise RuntimeError("deliberate")
        return real_growth(results, off, path, lattice, data=data,
                           notes=notes)

    S.figure_growth = growth_fails_192
    try:
        with contextlib.redirect_stdout(_io.StringIO()):
            mb = S.build(cache, bld)
    finally:
        S.figure_growth = real_growth
    check(mb["failed"] and mb["failed"][0].startswith(
              "submission for psi0=+0.00 l=512")
          and sorted(mb["compositions"]) == ["+0.00", "-0.40"]
          and sorted(mb["sizes"]) == ["128", "512"]
          and "last_attempt_failed" in mb["sizes"]["512"]
          and "last_attempt_failed" not in mb["compositions"]["-0.40"],
          f"a failing lattice is recorded and marked, and the rest are "
          f"rebuilt: {mb['failed']}")
    check(mb["sizes"]["128"] == mb["compositions"]["+0.00"]
          and not old_names(bld),
          "the rebuild writes l = 128 with the critical case, as one entry "
          "and one folder")

    bad = os.path.join(SCRATCH, "result_cache_bad")
    bad_out = os.path.join(SCRATCH, "results_bad")
    clean(bad, bad_out)
    shutil.copytree(cache, bad)
    victim = sorted(f for f in os.listdir(bad) if "_l512_" in f)[0]
    with open(os.path.join(bad, victim), "wb") as fh:
        fh.write(b"PK\x03\x04 truncated")
    bad_before = tree(bad)
    with contextlib.redirect_stdout(_io.StringIO()):
        mb = S.build(bad, bad_out)
    check(mb["failed"] and mb["failed"][0].startswith("lattice 512")
          and sorted(mb["compositions"]) == ["+0.00", "-0.40"]
          and "512" not in mb["sizes"],
          f"an unreadable cache file costs its own case, not the rebuild: "
          f"{mb['failed']}")
    check(tree(bad) == bad_before,
          "and the damaged file is left in the cache as it was, not moved "
          "aside: the cache is only read")
    bad0 = os.path.join(SCRATCH, "result_cache_bad0")
    bad0_out = os.path.join(SCRATCH, "results_bad0")
    clean(bad0, bad0_out)
    shutil.copytree(cache, bad0)
    with open(os.path.join(bad0, f"fno_ch_l128_seed1_{'a' * 12}.npz"),
              "wb") as fh:
        fh.write(b"PK truncated")
    raised0 = None
    try:
        with contextlib.redirect_stdout(_io.StringIO()):
            mb = S.build(bad0, bad0_out)
    except Exception as exc:
        raised0, mb = exc, {"failed": [], "compositions": {}, "sizes": {}}
    check(raised0 is None
          and mb["failed"] == ["composition +0.00"]
          and sorted(mb["compositions"]) == ["-0.40"]
          and sorted(mb["sizes"]) == ["512"],
          f"a damaged first seed of the critical case, the one the window is "
          f"read from, also costs only its case: "
          f"{raised0!r} {mb['failed']}")

    real_write_case = S.write_case
    n_calls = []

    def dies_second(*a, **k):
        n_calls.append(1)
        if len(n_calls) == 2:
            raise KeyboardInterrupt
        return real_write_case(*a, **k)

    before_kill = tree(bld)
    S.write_case = dies_second
    killed = False
    try:
        with contextlib.redirect_stdout(_io.StringIO()):
            S.build(cache, bld)
    except KeyboardInterrupt:
        killed = True
    finally:
        S.write_case = real_write_case
    now = tree(bld)
    check(killed and P.RESULTS_DIR == SAVED["RESULTS_DIR"]
          and P.MODEL_PATH == SAVED["MODEL_PATH"]
          and all(now[f"extras/psi0+0.00_l128/{f}"]
                  == before_kill[f"extras/psi0+0.00_l128/{f}"] for f in PLAIN)
          and all(now[f] == before_kill[f] for f in flat_figs(["psi0p0.00"]))
          and "extras/psi0+0.00_l128/fno_ch_summary.json" not in now
          and not leftovers_tree(bld),
          "a rebuild killed part-way puts fno_predict's settings back; the "
          "case it was writing keeps its previous figures and has no "
          "summary, so it does not read as finished")
    with contextlib.redirect_stdout(_io.StringIO()):
        mb = S.build(cache, bld)
    mfile = bookkeeping(XB)
    check(mb["failed"] == [] and "extras/psi0+0.00_l128/fno_ch_summary.json"
          in tree(bld)
          and mfile["source"].startswith("make_submission")
          and mfile["cache"] == cache
          and not any("last_attempt_failed" in e
                      for kind in ("compositions", "sizes")
                      for e in mfile[kind].values()),
          "a finished one leaves every case whole and the folder stamped "
          "with its source")

    old_layout = os.path.join(SCRATCH, "results_oldlayout")
    clean(old_layout)
    os.makedirs(old_layout)
    with open(os.path.join(old_layout, "snapshots_psi0p0.00.png"), "wb") as fh:
        fh.write(b"old layout")
    refused = ""
    try:
        with contextlib.redirect_stdout(_io.StringIO()):
            S.main(["--cache", cache, "--out", old_layout])
    except SystemExit as exc:
        refused = str(exc.code)
    check("choose a new --out" in refused
          and files_under(old_layout) == ["snapshots_psi0p0.00.png"],
          "it refuses to write into a folder that holds files but no "
          "manifest (the old layout), and touches nothing there")
    check(tree(cache) == cache_before[0]
          and stat_tree(cache) == cache_before[1],
          "every rebuild left the cache exactly as it was")
finally:
    S.fno_runlog.start_log = real_start_log

setup(bld)
P.EXTRAS_ROOT = XB          # the prediction run writes into the rebuilt folder
log = run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "3",
                "--no-video"])
mt = bookkeeping(XB)
check(calls == [(0.0, 512)],
      "a rebuilt case (another key, other seeds) is not taken for finished "
      "by a prediction run")
check(mt["sweep"]["status"].startswith("r2_sweep drawn")
      and mt["sweep"]["missing_curves"] == [],
      "fno_predict topping up a rebuilt folder redraws the whole sweep")
check("cache" not in mt and "failed" not in mt
      and mt.get("last_build", {}).get("cache") == cache
      and mt["source"] == "fno_predict",
      "the rebuild's own fields move under last_build instead of reading "
      "as the folder's current state")
check(mt["checkpoints"]["mixed"] and mt["checkpoints"]["unknown"]
      and "unknown origin" in log,
      "entries rebuilt from a cache count as another checkpoint, and it warns")

print("\n18. same_model and a sweep over mixed lattices")
restore()
twin = os.path.join(SCRATCH, "result_cache_twin")
clean(twin)
shutil.copytree(cache, twin)
with contextlib.redirect_stdout(_io.StringIO()):
    same = S.same_model(cache, twin)
check(same == (0.0, 2), f"identical caches compare as the same model: {same}")
for f in os.listdir(twin):
    if f.startswith("fno_ch_l128_seed1_"):
        r = P._load_result(os.path.join(twin, f))
        r["r2"] = r["r2"][:-5]
        P._save_result(os.path.join(twin, f), r)
with contextlib.redirect_stdout(_io.StringIO()):
    same = S.same_model(cache, twin)
check(same is not None and same[0] == float("inf"),
      "a seed of different length is not taken as proof of the same model")
check(S.same_model(os.path.join(SCRATCH, "nope1"),
                   os.path.join(SCRATCH, "nope2")) is None,
      "nothing shared gives None rather than an error")

TITLES.clear()
with contextlib.redirect_stdout(_io.StringIO()):
    S.figure_sweeps({-0.4: flat(128), 0.0: flat(512)}, {},
                    os.path.join(es, "mixed.png"))
check(any("lattices differ" in t for t in panel_titles()),
      "panel (a) says its lattices differ instead of naming one")

TITLES.clear()
late = flat(128)
late["times"] = np.arange(201.0, 2001.0)
late["mean_curve"] = np.linspace(1, 0.9, 1800)
few = flat(128)
few["stats"] = {"n_seeds": 3, "n_end": 2}
with contextlib.redirect_stdout(_io.StringIO()):
    S.figure_sweeps({-0.4: flat(128), 0.0: late}, {128: flat(128), 512: few},
                    os.path.join(es, "mixed.png"))
check(any("(the curves differ)" in t for kind, t in TITLES if kind == "sup"),
      "a sweep over curves seeded at different times says so")
check(any("still running at the end: 2 of 3 at $l = 512$" in t
          for t in panel_titles()),
      "a curve averaged over fewer seeds at the end says so")
ft = S.folder_times([
    {"snapshots": {"times": [100.0, 2000.0], "t_start": 100.0,
                   "t_end": 2000.0}},
    {"snapshots": {"times": [200.0, 2000.0], "t_start": 200.0,
                   "t_end": 2000.0}}])
check(ft["rollout_start"] == [100.0, 200.0] and ft["pdf_time"] == 2000.0
      and not ft["times_agree"],
      "folder-level times list what the entries disagree on")

print("\n24. --all over several runs, one of them broken")
empty_dir = os.path.join(SCRATCH, "results_empty")
clean(empty_dir)
real_load_case = S.load_case


def load_with_empty(cache_, off, lat):
    if off < -0.3:
        rs = [dict(r, times=np.asarray([]), r2=np.asarray([]))
              for r in fake_results(off, 2)]
        return (1, 2), rs, "e" * 12
    return real_load_case(cache_, off, lat)


S.load_case = load_with_empty
P.WRITE_VIDEO = False
try:
    with contextlib.redirect_stdout(_io.StringIO()):
        me = S.build(cache, empty_dir)
    built_ok = True
except Exception:
    built_ok, me = False, {"failed": [], "compositions": {}}
finally:
    S.load_case = real_load_case
check(built_ok and me["failed"]
      and me["failed"][0].startswith("composition -0.40")
      and "+0.00" in me["compositions"],
      "a case whose rollouts have no times fails alone; the rebuild goes on")
runs_dir = os.path.join(SCRATCH, "results_runs_fake")
ca = os.path.join(SCRATCH, "result_cache_run_a")
cc = os.path.join(SCRATCH, "result_cache_run_c")
clean(runs_dir, ca, cc)
for cdir in (ca, cc):
    for sd, r in zip((1, 2, 3), fake_results(0.0, 3)):
        P._save_result(os.path.join(
            cdir, f"fno_ch_l128_seed{sd}_{'a' * 12}.npz"), r)
good_a = read(os.path.join(ca, f"fno_ch_l128_seed1_{'a' * 12}.npz"))
with open(os.path.join(ca, f"fno_ch_l128_seed1_{'a' * 12}.npz"), "wb") as fh:
    fh.write(b"PK\x03\x04 truncated")
saved_all = (S.RUNS, S.COMPOSITIONS, S.fno_runlog.start_log, S.build)
NO_MODEL = os.path.join(SCRATCH, "fno_ch_not_here.pt")
S.RUNS = ({"key": "A_x", "cache": ca, "model": None, "title": "broken",
           "note": "test"},
          {"key": "C_x", "cache": cc, "model": NO_MODEL, "title": "fine",
           "note": "test"})
S.COMPOSITIONS = (0.0,)
S.fno_runlog.start_log = lambda *a, **k: None
codes = []


def build_raising(cache_, out_, model_path=None):
    if cache_ == ca:
        raise RuntimeError("deliberate")
    return saved_all[3](cache_, out_, model_path)


try:
    for label, build_fn in (("a broken cache file", saved_all[3]),
                            ("build itself raising", build_raising)):
        S.build = build_fn
        clean(runs_dir)
        try:
            with contextlib.redirect_stdout(_io.StringIO()):
                S.main(["--all", "--out", runs_dir])
            codes.append(None)
        except SystemExit as exc:
            codes.append(exc.code)
        check(os.path.exists(os.path.join(runs_dir, "C_x", "manifest.json"))
              and isinstance(codes[-1], str) and "FAILED" in codes[-1]
              and "A_x" in codes[-1],
              f"with {label}, the other run is still built and the exit "
              f"status names the broken one")
        with open(os.path.join(runs_dir, "comparison.json")) as fh:
            cj = json.load(fh)
        check([r_["key"] for r_ in cj["runs"]]
              == (["A_x", "C_x"] if build_fn is saved_all[3] else ["C_x"])
              and os.path.exists(os.path.join(runs_dir, "comparison.png"))
              and os.path.exists(os.path.join(runs_dir, "README.md"))
              and os.path.exists(os.path.join(runs_dir, "C_x", "l128",
                                              "snapshots_psi0p0.00.png")),
              f"with {label}, every run built gets its folder under --out, "
              f"comparison.png, the README and comparison.json")

    with open(os.path.join(ca, f"fno_ch_l128_seed1_{'a' * 12}.npz"),
              "wb") as fh:
        fh.write(good_a)
    S.build = saved_all[3]
    clean(runs_dir)
    caches_before = (tree(ca), tree(cc))
    with contextlib.redirect_stdout(_io.StringIO()):
        S.main(["--all", "--out", runs_dir])
    with open(os.path.join(runs_dir, "comparison.json")) as fh:
        cj = json.load(fh)
    with open(os.path.join(runs_dir, "README.md"), encoding="utf-8") as fh:
        index = fh.read()
    mc_ = bookkeeping(os.path.join(runs_dir, "C_x", "extras"))
    check(sorted(d for d in os.listdir(runs_dir)
                 if os.path.isdir(os.path.join(runs_dir, d)))
          == ["A_x", "C_x"]
          and all(manifest(os.path.join(runs_dir, k))["source"]
                  == f"make_submission, regenerated from {c_}"
                  for k, c_ in (("A_x", ca), ("C_x", cc)))
          and mc_["failed_last_run"] == []
          and "`A_x/`" in index and "`C_x/`" in index
          and "- not in its cache: psi0=-0.40 l=128" in index,
          "main --all over good caches exits cleanly: one merged folder per "
          "run, each naming its cache, and missing cases listed in the "
          "README rather than failed")
    check([r_["key"] for r_ in cj["runs"]] == ["A_x", "C_x"]
          and cj["runs"][1]["cache"] == cc
          and cj["runs"][1]["critical_mean_r2"]
          == mc_["compositions"]["+0.00"]["mean_r2"]
          and cj["runs"][1]["sizes_mean_r2"]
          == {"128": mc_["sizes"]["128"]["mean_r2"]}
          and cj["paired"]["base"] == "A" and len(cj["paired"]["rows"]) == 1
          and cj["paired"]["rows"][0]["n"] == 3,
          "comparison.json holds each run's numbers as its manifest has them, "
          "and the paired table")
    check((tree(ca), tree(cc)) == caches_before,
          "and the caches are left as they were")
    index_before = (read(os.path.join(runs_dir, "README.md")),
                    read(os.path.join(runs_dir, "comparison.json")),
                    read(os.path.join(runs_dir, "comparison.png")))
    gone_a = ca + "_gone"
    os.replace(ca, gone_a)
    try:
        buf_ = _io.StringIO()
        with contextlib.redirect_stdout(buf_):
            S.main(["--all", "--out", runs_dir])
    finally:
        os.replace(gone_a, ca)
    check("kept the existing comparison.png" in buf_.getvalue()
          and (read(os.path.join(runs_dir, "README.md")),
               read(os.path.join(runs_dir, "comparison.json")),
               read(os.path.join(runs_dir, "comparison.png")))
          == index_before
          and os.path.exists(os.path.join(runs_dir, "A_x", "manifest.json")),
          "run again after one run's cache was deleted, --all leaves the index "
          "and comparison as they were, rather than redraw them without it")
    with open(os.path.join(runs_dir, "C_x", "report.txt"),
              encoding="utf-8") as fh:
        c_readme = fh.read()
    check(mc_["last_run"]["model_path"] == NO_MODEL
          and NO_MODEL in c_readme
          and bookkeeping(os.path.join(runs_dir, "A_x", "extras"))[
              "last_run"]["model_path"] == ""
          and all(r_.get("model") for r_ in saved_all[0]),
          "each run's bookkeeping and report name the checkpoint its RUNS entry "
          "gives, as provenance only, and every real RUNS entry names one")
    usage = None
    try:
        with contextlib.redirect_stdout(_io.StringIO()), \
                contextlib.redirect_stderr(_io.StringIO()):
            S.main(["--cache", cache])
    except SystemExit as exc:
        usage = exc.code
    check(usage == 2, "--cache without --out is refused (no argument at "
                      "all is the default mode now: see t_flat)")
    default_out = os.path.join(SCRATCH, "results_runs_default")
    clean(default_out)
    saved_all_out = S.ALL_OUT
    S.ALL_OUT = default_out
    try:
        with contextlib.redirect_stdout(_io.StringIO()):
            S.main(["--all"])
    finally:
        S.ALL_OUT = saved_all_out
    check(saved_all_out == os.path.join("Results", "results_runs_2026-09-30")
          and not hasattr(S, "OUT")
          and os.path.exists(os.path.join(default_out, "C_x", "l128",
                                          "snapshots_psi0p0.00.png"))
          and os.path.exists(os.path.join(default_out, "comparison.json")),
          "--all without --out writes into ALL_OUT (Results/results_runs_"
          "2026-09-30); there is no single-folder default any more")
    cache_dir = os.path.join("Work", "rollout_cache")
    t300 = os.path.join("Work", "models", "train_t0300", "fno_ch.pt")
    t500 = os.path.join("Work", "models", "train_t0500", "fno_ch.pt")
    check([(r_["key"], r_["cache"], r_["model"]) for r_ in saved_all[0]]
          == [("A_train_t300_eval_10seed",
               os.path.join(cache_dir, "result_cache_10_seed"), t300),
              ("B_train_t300_eval_50seed",
               os.path.join(cache_dir, "result_cache_50_seed_300"), t300),
              ("C_train_t500_eval_50seed",
               os.path.join(cache_dir, "result_cache_50_seed_500"), t500)],
          "the runs --all regenerates read their caches from Work/"
          "rollout_cache/ and name the models in Work/models/<train label>/")
finally:
    (S.RUNS, S.COMPOSITIONS, S.fno_runlog.start_log, S.build) = saved_all

c200 = os.path.join(SCRATCH, "result_cache_t200")
b200 = os.path.join(SCRATCH, "results_b200")
clean(c200, b200)
restore()
P.START_TIME, P.SNAPSHOT_TIMES = 200.0, (200.0, 500.0, 1000.0, 2000.0)
for sd, r in zip((1, 2, 3), fake_results(0.0, 3)):
    P._save_result(os.path.join(c200, f"fno_ch_l128_seed{sd}_{'a' * 12}.npz"),
                   r)
restore()
P.WRITE_VIDEO = False
with contextlib.redirect_stdout(_io.StringIO()):
    mb = S.build(c200, b200)
check(mb["rollout_start"] == 200.0
      and mb["snapshot_times"] == [500.0, 1000.0, 2000.0],
      "a rebuild takes its folder-level times from the cache, not the "
      "defaults (the columns of the snapshot figure, from t = 500)")
check(mb["compositions"]["+0.00"]["run"]["window"]
      == {"start": 200.0, "end": 2000.0,
          "snapshots": [200.0, 500.0, 1000.0, 2000.0]}
      and P.START_TIME == SAVED["START_TIME"],
      "and records each entry's window from the rollouts, then puts "
      "fno_predict's own back")
restore()

print("\n10b. nothing in sections 17, 18 and 24 reached the project")
project_untouched()

restore()
matplotlib.figure.Figure.suptitle = _suptitle
matplotlib.axes.Axes.set_title = _set_title
matplotlib.axes.Axes.text = _text
clean(es, cache, partial, twin, bld, bad, bad_out, bad0, bad0_out, old_layout,
      empty_dir, runs_dir, ca, cc, c200, b200, default_out,
      SAVED["RESULTS_DIR"], SAVED["RESULT_CACHE"], SAVED["GT_CACHE"])
finish()
