"""The submission figures go into each case's folder of the run's one results
folder, correctly and safely -- there is no Submission folder -- and
make_submission regenerates such folders from result caches.

Sections 1 to 16 of the three suites that share tests/_submission_base.py (the
one file took about 16 min): where the folders go, a full run, a top-up,
--no-submission, a failure halfway, an ablation, the run's own times, a
diverged or early-stopped seed, a locked or killed publish, a damaged manifest
and the paired verdict.  t_submission_regen has sections 17, 18 and 24,
t_submission_publish sections 19 to 23, 25 and 10."""
import contextlib
import io as _io
import json
import os
import time

import matplotlib.axes
import matplotlib.figure
import numpy as np

from _paths import SCRATCH
from _submission_base import (EXIT, FLATF, HOLD, P, PLAIN, REAL_MODEL, RES, S,
                              SAVED, STG, SUB_BEFORE, TITLES, TOKENS, XSTG,
                              _set_title, _suptitle, _text, bookkeeping, calls,
                              check, clean, contents, fake_model, fake_results,
                              files_under, finish, flat_figs, kill_mid_publish,
                              leftovers, leftovers_tree, manifest, old_names,
                              panel_titles, parent, sweep_files, project_untouched, read,
                              real_growth, real_replace, replace_failing,
                              restore, run_files, run_main, setup, snapshot_titles,
                              stopped, submission_dirs, tree, wcase, xr)

print("1. where it goes")
setup(RES)
check(P.submission_dir() == P.extras_dir() == xr()
      and xr() == os.path.join(P.RESULT_CACHE, "extras"),
      f"the bookkeeping is in extras/ of the rollout cache, the figures in "
      f"the results folder itself, not a Submission folder beside it: "
      f"{P.submission_dir()}")
P.MODEL_FLAGS["hard_constraints"] = False
check(P.submission_dir() == xr(),
      "the model flags do not move it")
P.EXTRAS_ROOT = os.path.join(SCRATCH, "extras_nohard_probe")
check(P.submission_dir() == P.EXTRAS_ROOT,
      "EXTRAS_ROOT does, which is what evaluate_all sets for an ablation "
      "(section 6)")
P.EXTRAS_ROOT = None
P.MODEL_FLAGS["hard_constraints"] = True
P.RESULTS_DIR = RES

print("\n2. a full run writes one flat folder, and a folder per case in extras/")
log = run_main(["--eval-mode", "both", "--sizes", "128,512", "--seeds", "3",
                "--no-video"])
check(os.path.isdir(RES) and submission_dirs() == SUB_BEFORE,
      "results_sub/ is written, and no Submission folder anywhere")
check(calls == [(-0.4, None), (0.0, None), (0.4, None), (-0.4, 512),
                (0.0, 512), (0.4, 512)],
      f"every composition of the mode runs once at the base lattice and "
      f"once at 512: {calls}")
check(EXIT[-1] == 0, "a clean run exits 0")
CASES = ("psi0-0.40_l128", "psi0+0.00_l128", "psi0+0.40_l128",
         "psi0-0.40_l512", "psi0+0.00_l512", "psi0+0.40_l512")
SFX = ("_off-0.40", "", "_off+0.40", "_off-0.40_l512", "_l512",
       "_off+0.40_l512")
missing = [f"{c}/{f}" for c in CASES for f in PLAIN
           if not os.path.exists(os.path.join(xr(), c, f))]
missing += [f for f in flat_figs(TOKENS)
            if not os.path.exists(os.path.join(RES, f))]
check(not missing, f"every case has its figures in the run folder under "
                   f"the flat layout's names, and its data in extras/<case>/ "
                   f"(missing: {missing})")
tops = files_under(RES)
check(tops == sorted(flat_figs(TOKENS) + run_files((128, 512))),
      f"the run folder: run-wide files at the top, and in each size folder "
      f"six figures for each composition and the size's own files: {tops}")
check(sorted(d for d in os.listdir(xr())
             if os.path.isdir(os.path.join(xr(), d))) == sorted(CASES)
      and sorted(f for f in os.listdir(xr())
                 if os.path.isfile(os.path.join(xr(), f)))
      == sorted([S.BOOKKEEPING, S.CURVES_FILE]),
      "extras/ has a folder per case and the run's bookkeeping and curves")
check(not leftovers_tree(RES) and not leftovers_tree(xr()),
      "no half-written scratch folder left behind, at the top or in a case")
check(not old_names(RES)
      and sum(f.split("/")[-1].startswith("snapshots_")
              for f in files_under(RES)) == 6
      and len(snapshot_titles()) == 6,
      "no l128 or other alias copy anywhere: the critical case is drawn "
      "once, as l128/..._psi0p0.00")
titles = snapshot_titles()
check(titles and all("model: Cahn-Hilliard FNO" in t for t in titles),
      f"snapshot figures' notes name the model "
      f"({titles[0][:40] if titles else 'none'}...)")
check(all(os.path.isfile(os.path.join(RES, f)) for f in sweep_files(
    (128, 512), ("psi0m0.40", "psi0p0.00", "psi0p0.40")))
      and not [f for f in os.listdir(RES) if f.endswith(".pdf")],
      "the sweeps are single images, one for each size and composition")
m = manifest(RES)
bk = bookkeeping()
check(sorted(m["cases"]) == sorted(TOKENS)
      and m["aliases"] == {"l128": "psi0p0.00"}
      and sorted(bk["compositions"]) == ["+0.00", "+0.40", "-0.40"]
      and sorted(bk["sizes"]) == ["128", "512"]
      and sorted(bk["grid"]) == ["+0.40_l512", "-0.40_l512"],
      f"manifest.json lists every case by token: {sorted(m['cases'])}, "
      f"{m['aliases']}; the bookkeeping has the other compositions at 512 "
      f"as grid entries")
check(bk["sizes"]["128"] == bk["compositions"]["+0.00"]
      and bk["sizes"]["128"]["folder"] == "psi0p0.00",
      "the bookkeeping's l=128 entry is the critical case's entry, naming "
      "the same token")
ents = [e for kind in S.KINDS for e in bk[kind].values()]
check(all(e["files"] == flat_figs([e["folder"]])
          and all(os.path.isfile(os.path.join(RES, f)) for f in e["files"])
          and all(os.path.isfile(os.path.join(xr(), e["extras"]["folder"],
                                              f))
                  for f in e["extras"]["files"]) for e in ents)
      and bk["sizes"]["512"]["folder"] == "l512"
      and bk["compositions"]["-0.40"]["folder"] == "psi0m0.40",
      "every entry names its token and its files in the run folder, and "
      "its extras, and they are there")
check(all(e["run"]["epoch"] == 7 for e in bk["compositions"].values()),
      "every entry is stamped with the checkpoint's epoch")
check(bk["source"] == "fno_predict" and bk["failed_last_run"] == []
      and m["source"] == "fno_predict",
      "the bookkeeping and manifest.json record their producer and a clean "
      "run")
with np.load(os.path.join(xr(), S.CURVES_FILE)) as z:
    groups = {tuple(n.split("__")[:2]) for n in z.files}
check(len(groups) == 7, f"one stored curve per bookkeeping entry: "
                        f"{len(groups)}")
check(all(os.path.exists(os.path.join(xr(), c, f"fno_ch{sfx}_summary.json"))
          and os.path.exists(os.path.join(xr(), c, P.MEAN_R2_FILE))
          for c, sfx in zip(CASES, SFX)),
      "each case's extras folder keeps its own summary and mean R2")
sums2, listing2 = {}, {}
for c, sfx in zip(CASES, SFX):
    with open(os.path.join(xr(), c, f"fno_ch{sfx}_summary.json")) as fh:
        sums2[c] = json.load(fh)
    listing2[c] = sorted(
        list(PLAIN) + [P.MEAN_R2_FILE, f"best_seed{sums2[c]['best_seed']}"]
        + [f"fno_ch{sfx}_{x}" for x in ("r2_and_snapshots.png",
                                        "conservation.png", "summary.json")])
check(all(sorted(d for d in os.listdir(os.path.join(xr(), c))
                 if os.path.isdir(os.path.join(xr(), c, d)))
          == [f"best_seed{sums2[c]['best_seed']}"] for c in CASES),
      "each case folder holds one seed folder, its best seed's, and no other")
check(all(sorted(os.listdir(os.path.join(xr(), c))) == listing2[c]
          for c in CASES),
      "and besides it exactly the case's own files: the data, the fno_ch "
      "figures, mean R2 and summary")
rec2 = sums2["psi0+0.00_l128"].get("record") or {}
check(set(rec2) == {"key", "seeds", "seed_digest", "fingerprint",
                    "numerics", "window", "submission", "video", "entry",
                    "curve", "figures", "complete", "written"}
      and rec2["numerics"] == P.numerics_of()
      and rec2["entry"] == bookkeeping()["compositions"]["+0.00"]
      and rec2["curve"] == {"n": 3, "n_end": 3, "n_div": 0, "lattice": 128}
      and rec2["figures"] == {"log_time": False, "per_seed_output": True,
                              "per_seed_figures": False}
      and rec2["key"] == P._cache_key(0.0, 128) and rec2["seeds"] == 3
      and rec2["seed_digest"] == P._seed_digest(P.make_eval_seeds(3))
      and rec2["window"] == [P.START_TIME, P.END_TIME]
      and rec2["submission"] is True and rec2["video"] is False
      and rec2["complete"] is True
      and "seeds" not in sums2["psi0+0.00_l128"]
      and sums2["psi0+0.00_l128"]["timing"]["n"] == 3,
      f"the case summary keeps no per-seed rows, and records what produced "
      f"it: the cache key, the seeds, the numerics, the window, the "
      f"submission in and no film asked for, the figure settings, and the "
      f"entry and curve counts it re-records when resumed ({rec2})")
with open(os.path.join(RES, "report.txt"), encoding="utf-8") as fh:
    report = fh.read()
check(all(f"  {t:<11}" in report for t in TOKENS[:3])
      and REAL_MODEL in report and "epoch" in report
      and "README.md" not in os.listdir(RES),
      "report.txt at the top names every case and the model; there is no "
      "README.md in the run folder")
first_192 = bk["sizes"]["512"]["run"]["written"]

print("\n3. a top-up skips what is finished, merges, and redraws the sweep")
time.sleep(1.2)
before_192 = tree(os.path.join(xr(), "psi0+0.00_l512"))
figs_192 = {f: read(os.path.join(RES, f)) for f in flat_figs(["l512"])}
setup(RES)
log = run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "3",
                "--no-video"])
check(calls == [] and "already finished" in log and EXIT[-1] == 0
      and bookkeeping()["sizes"]["512"]["run"]["written"] == first_192
      and tree(os.path.join(xr(), "psi0+0.00_l512")) == before_192
      and {f: read(os.path.join(RES, f)) for f in figs_192} == figs_192,
      "without --redo, a case an earlier job finished with this key and "
      "these seeds is not rolled out again, and its files are left alone")
setup(RES)
log = run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "2",
                "--no-video"])
check(calls == [(0.0, 512)] and "already finished" not in log,
      f"with other seeds the same case is not taken for finished: it is "
      f"rolled out again without --redo ({calls})")
setup(RES)
run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "3",
          "--no-video", "--redo"])
m2 = bookkeeping()
check(calls == [(0.0, 512)],
      f"with --redo only the requested lattice ran again: {calls}")
check(sorted(m2["compositions"]) == ["+0.00", "+0.40", "-0.40"]
      and "128" in m2["sizes"],
      "every case from the first run is still listed")
check(sorted(manifest(RES)["cases"]) == sorted(TOKENS)
      and m2["sizes"]["512"]["seeds"] == 3,
      "and the manifest.json the top-up rewrites still lists every case "
      "the folder holds")
check(m2["sizes"]["512"]["run"]["written"] > first_192,
      "l=512 was rewritten with a newer stamp")
check(all(os.path.isfile(os.path.join(RES, f)) for f in sweep_files(
    (128, 512), ("psi0m0.40", "psi0p0.00", "psi0p0.40"))),
      "the top-up's sweep still has all its images, from the stored curves")

print("\n4. --no-submission")
other = os.path.join(SCRATCH, "results_nosub")
clean(other, os.path.join(SCRATCH, "rc_results_nosub"))
setup(other)
run_main(["--eval-mode", "critical", "--sizes", "128", "--seeds", "2",
          "--no-video", "--no-submission"])
oc = os.path.join(xr(), "psi0+0.00_l128")
check(os.path.exists(os.path.join(oc, "fno_ch_summary.json"))
      and os.path.exists(os.path.join(oc, P.MEAN_R2_FILE)),
      "the case's own outputs are still written")
check(os.listdir(other) == []
      and not any(os.path.exists(os.path.join(oc, f))
                  for f in S.CASE_FILES + S.CASE_DATA)
      and not os.path.exists(os.path.join(xr(), S.BOOKKEEPING))
      and submission_dirs() == SUB_BEFORE,
      "no submission figures, film, data, manifest or report, and no "
      "submission folder")
with open(os.path.join(oc, "fno_ch_summary.json")) as fh:
    rec4 = json.load(fh)["record"]
check(rec4["complete"] is True and rec4["submission"] is False
      and rec4["video"] is False,
      "its record says the case is finished, without the submission")
setup(other)
log = run_main(["--eval-mode", "critical", "--sizes", "128", "--seeds", "2",
                "--no-video", "--no-submission"])
check(calls == [] and "already finished" in log,
      "another --no-submission run takes it for finished")
setup(other)
log = run_main(["--eval-mode", "critical", "--sizes", "128", "--seeds", "2",
                "--no-video"])
check(calls == [(0.0, None)] and "already finished" not in log
      and all(os.path.exists(os.path.join(oc, f)) for f in PLAIN)
      and all(os.path.exists(os.path.join(other, f))
              for f in flat_figs(["psi0p0.00"]) + ["manifest.json"]),
      "but a run that wants the submission rolls it out again and draws it")

print("\n5. a failure halfway through a case leaves the old files whole")
setup(RES)
crit = os.path.join(xr(), "psi0+0.00_l128")
FLAT_CRIT = flat_figs(["psi0p0.00"])
before = {f: read(os.path.join(crit, f)) for f in PLAIN}
before_flat = {f: read(os.path.join(RES, f)) for f in FLAT_CRIT}


def boom(*a, **k):
    raise RuntimeError("deliberate")


S.figure_growth = boom
try:
    log = run_main(["--eval-mode", "critical", "--sizes", "128",
                    "--seeds", "3", "--no-video", "--redo"])
finally:
    S.figure_growth = real_growth
after = {f: read(os.path.join(crit, f)) for f in PLAIN}
after_flat = {f: read(os.path.join(RES, f)) for f in FLAT_CRIT}
check(calls == [(0.0, None)] and before == after
      and before_flat == after_flat,
      "snapshots and P(psi) drawn before the failure did not replace the "
      "old ones, and neither did their data")
check(not leftovers_tree(RES) and not leftovers_tree(xr()),
      "the scratch folders are cleaned up after the failure")
m3 = bookkeeping()
check(any("submission for psi0=+0.00" in f for f in m3["failed_last_run"]),
      f"manifest.failed_last_run names the case: {m3['failed_last_run']}")
check("last_attempt_failed" in m3["compositions"]["+0.00"]
      and "last_attempt_failed" in m3["sizes"]["128"],
      "the kept entries are marked as having failed their last attempt")
with open(os.path.join(crit, "fno_ch_summary.json")) as fh:
    rec5 = json.load(fh)["record"]
check(not rec5["complete"] and not rec5["submission"],
      "the case summary was still written, and its record says the case "
      "is not finished")
check("submission figures for critical FAILED" in log
      and "submission for psi0=+0.00" in log,
      "the failure is reported where it happened and in the final banner")
check(EXIT[-1] == 1, "and the run exits 1, so SLURM does not call it COMPLETED")
setup(RES)
run_main(["--eval-mode", "critical", "--sizes", "128", "--seeds", "3",
          "--no-video"])
check(calls == [(0.0, None)] and EXIT[-1] == 0
      and "last_attempt_failed" not in bookkeeping()["compositions"]["+0.00"],
      "so the next job does that case again without --redo, and clears the "
      "mark")

print("\n6. an ablation model is named as one, in its own folder")
ab = os.path.join(SCRATCH, "results_ab")
clean(ab, ab + "_nohard", os.path.join(SCRATCH, "rc_results_ab"))
setup(ab, hard=False)
run_main(["--eval-mode", "critical", "--sizes", "128", "--seeds", "2",
          "--no-video"])
abn = os.path.join(P.RESULT_CACHE, P.EXTRAS_NAME + "_nohard",
                   "psi0+0.00_l128")
check(os.path.exists(os.path.join(ab + "_nohard", "manifest.json"))
      and os.path.exists(os.path.join(ab + "_nohard", "l128",
                                      "snapshots_psi0p0.00.png"))
      and os.path.exists(os.path.join(abn, "fno_ch_nohard_summary.json"))
      and os.path.exists(os.path.join(os.path.dirname(abn), S.BOOKKEEPING))
      and not os.path.exists(os.path.join(P.RESULT_CACHE, P.EXTRAS_NAME))
      and not os.path.exists(ab),
      "written to results_ab_nohard/, not results_ab/, with its extras in "
      "extras_nohard/ of the cache and its files labelled _nohard")
check(P.EXTRAS_ROOT is None, "and EXTRAS_ROOT is put back too")
check(P.RESULTS_DIR == ab, f"and P.RESULTS_DIR is put back: {P.RESULTS_DIR}")
titles = snapshot_titles()
check(titles and all("model: Cahn-Hilliard FNO [hard constraints off]" in t
                     for t in titles),
      "the notes of its snapshot figures say the hard constraints were off")
seen = []


def spy_case(model, device, off, lattice=None, seeds=None):
    seen.append((P.RESULTS_DIR, P.RESULTS_TEMPLATE))
    raise KeyboardInterrupt


setup(ab)
P.RESULTS_TEMPLATE = ab + "_P{H}"
P.EVAL_MODE, P.SIZE_SWEEP = "critical", ()
interrupted = False
try:
    with contextlib.redirect_stdout(_io.StringIO()):
        P.evaluate_all(fake_model(False), {}, "cpu", case_fn=spy_case,
                       resume=False, clean=False)
except KeyboardInterrupt:
    interrupted = True
check(interrupted and seen == [(ab + "_nohard", ab + "_P{H}_nohard")]
      and P.RESULTS_DIR == ab and P.RESULTS_TEMPLATE == ab + "_P{H}",
      f"the template gets the suffix too, and both are put back even when "
      f"the run is interrupted: {seen}")

print("\n7. the figures follow the run's own times")
t200 = os.path.join(SCRATCH, "results_t200")
clean(t200, os.path.join(SCRATCH, "rc_results_t200"))
setup(t200)
P.START_TIME, P.SNAPSHOT_TIMES = 200.0, (200.0, 500.0, 1000.0, 2000.0)
run_main(["--eval-mode", "critical", "--sizes", "128", "--seeds", "2",
          "--no-video"])
e = bookkeeping()["compositions"]["+0.00"]
check(e["snapshots"]["times"] == [500.0, 1000.0, 2000.0],
      f"the columns drawn from t = 200 are those of SNAPSHOT_COLUMNS the "
      f"run has: {e['snapshots']['times']}")
check(all("seeded at t = 200" in t for t in snapshot_titles()),
      "the notes say the rollout was seeded at t = 200")
setup(t200)
log = run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "2",
                "--no-video"])
m200 = bookkeeping()
check(m200["rollout_start"] == [100.0, 200.0] and not m200["times_agree"]
      and "rolled out over different times" in log,
      "a top-up seeded at another time makes the folder-level start a list, "
      "and the run warns")
setup(t200)
log = run_main(["--eval-mode", "critical", "--sizes", "128", "--seeds", "2",
                "--no-video"])
m200 = bookkeeping()
check(calls == [(0.0, None)] and "already finished" not in log
      and m200["rollout_start"] == 100.0 and m200["times_agree"],
      "the case finished from t = 200 has another cache key at t = 100: it is "
      "rolled out again without --redo, and the folder agrees again")

t1500 = os.path.join(SCRATCH, "results_t1500")
clean(t1500, os.path.join(SCRATCH, "rc_results_t1500"))
setup(t1500)
P.END_TIME, P.SNAPSHOT_TIMES = 1500.0, (100.0, 500.0, 1000.0, 1500.0)
log = run_main(["--eval-mode", "critical", "--sizes", "128", "--seeds", "2",
                "--no-video"])
check(os.path.exists(os.path.join(t1500, "l128",
                                  "psi_distribution_psi0p0.00.png"))
      and "FAILED" not in log,
      "END_TIME = 1500 no longer fails every case")
check(any("at t = 1500" in t for kind, t in TITLES if kind == "ax"),
      "P(psi) is drawn at the run's final time, 1500")
restore()
P.END_TIME = 1500.0
w = P.run_stamp({})["window"]
check(w == {"start": 100.0, "end": 1500.0,
            "snapshots": [100.0, 500.0, 1000.0]},
      f"the run stamp's window stops at END_TIME: {w}")
restore()
P.run_stamp({})
P.RESULTS_DIR = os.path.join(SCRATCH, "results_stamp_a")
first_dir = P.run_stamp()["results_dir"]
P.RESULTS_DIR = os.path.join(SCRATCH, "results_stamp_b")
check(first_dir == os.path.abspath(os.path.join(SCRATCH, "results_stamp_a"))
      and P.run_stamp()["results_dir"]
      == os.path.abspath(os.path.join(SCRATCH, "results_stamp_b")),
      "a new results folder gets a new run stamp")
restore()
P.run_stamp({})

print("\n8. one diverged seed does not cost the case its figures")
DIVROOT = os.path.join(SCRATCH, "results_div")
div = DIVROOT                              # the flat run folder
divx = div + "_x"                          # the case's extras
clean(DIVROOT, divx)
restore()
TITLES.clear()
res = fake_results(0.0, 3, diverge={2})
entry, _ = wcase(res, (1, 2, 3), 0.0, 128, div, videos=False)
check(entry["files"] == list(FLATF) and entry["folder"] == "psi0p0.00"
      and entry["token"] == "psi0p0.00"
      and entry["extras"]["files"] == list(PLAIN),
      f"the figures under the flat layout's names in the run folder, the CSVs of their "
      f"data in the extras: {entry['files']}")
check(entry["psi_pdf"]["pred_seeds_excluded"] == 1,
      "the diverged seed is left out of P(psi) and counted")
check(entry["growth"]["seeds_ending_early"] == 1,
      "the growth curve averages over the seeds still running")
check(any("1 diverged seed ends early" in t for _, t in TITLES),
      "and its title says so, in the singular")
check(any(": 1 (scored at R2 = -1)" in t for t in snapshot_titles())
      and entry["snapshots"]["r2_at_end_diverged"] == 1,
      "the snapshot caption says the worst is a diverged seed at the floor")
with open(os.path.join(divx, "psi_distribution.csv")) as fh:
    pdf_head = fh.readline().strip()
pdf_csv = np.loadtxt(os.path.join(divx, "psi_distribution.csv"),
                     delimiter=",", skiprows=1)
with open(os.path.join(divx, "domain_growth.csv")) as fh:
    growth_head = fh.readline().strip()
growth_csv = np.loadtxt(os.path.join(divx, "domain_growth.csv"),
                        delimiter=",", skiprows=1)
check(pdf_head == "psi,density_true,density_pred,lever_A_pct,"
                  "A_rich_true_pct,A_rich_pred_pct"
      and pdf_csv.shape == (S.PDF_BINS, 6)
      and growth_head == "t,L_true_mean,L_pred_mean,n_seeds_true,"
                         "n_seeds_pred,L_half_true_mean,L_half_pred_mean,"
                         "train_horizon,n_train_true,n_train_pred,"
                         "n_extrap_true,n_extrap_pred"
      and growth_csv.shape[1] == 12
      and np.all(np.isnan(growth_csv[:, 5:7]))
      and growth_csv[0, 3] == 3 and growth_csv[-1, 3] == 2
      and growth_csv[-1, 4] == 2
      and np.allclose(growth_csv[:, 1], np.linspace(5, 14, 19)),
      "the CSVs hold what the figures show: the pooled histograms, and the "
      "mean L(t) with how many seeds each time averages")

print("\n9. a film this run did not make is not left behind")
for f in ("rollout.mp4", "rollout.gif"):
    with open(os.path.join(divx, f), "wb") as fh:
        fh.write(b"old film")
wcase(res, (1, 2, 3), 0.0, 128, div, videos=False)
check(os.path.exists(os.path.join(divx, "rollout.mp4"))
      and os.path.exists(os.path.join(divx, "rollout.gif")),
      "a film the call did not make is left alone (a redraw from records "
      "makes none, and must not delete the earlier run's)")
wcase(res, (1, 2, 3), 0.0, 128, div, videos=False, keep_film=False)
check(not os.path.exists(os.path.join(divx, "rollout.mp4"))
      and not os.path.exists(os.path.join(divx, "rollout.gif")),
      "a live run (keep_film=False) removes a stale film, in either "
      "container")
check(files_under(div) == sorted(FLATF)
      and sorted(os.listdir(divx)) == sorted(PLAIN),
      "and the figures and their CSVs are published together, nothing else")
filmed = fake_results(0.0, 3)
for r in filmed:
    r.update(video_t=np.array([101.0, 111.0, 121.0]),
             video_true=np.zeros((3, 24, 24), dtype=np.float32),
             video_pred=np.zeros((3, 24, 24), dtype=np.float32))
film_dir = os.path.join(SCRATCH, "case_film")
clean(film_dir, film_dir + "_x")
with contextlib.redirect_stdout(_io.StringIO()):
    fe, _ = wcase(filmed, (1, 2, 3), 0.0, 128, film_dir, videos=True,
                         root=SCRATCH)
films = [f for f in os.listdir(film_dir + "_x") if f.startswith("rollout.")]
check(len(films) == 1 and films[0] in S.CASE_FILES
      and fe["film"] == {"t_first": 101.0, "t_last": 121.0}
      and fe["extras"]["files"] == [f for f in S.CASE_FILES + S.CASE_DATA
                                    if f in PLAIN + tuple(films)]
      and sorted(os.listdir(film_dir + "_x")) == sorted(PLAIN + tuple(films))
      and files_under(film_dir) == sorted(FLATF),
      f"a case with frames gets one film under its plain name beside its "
      f"data in the extras, and its entry lists it with the rest: {films}")

print("\n11. a locked file at publish time changes nothing")
old = contents(div)
oldx = contents(divx)
new_res = fake_results(0.0, 3)
for _r in new_res:
    # the figures have no text now: make the curves of the new run differ
    _r["energy_pred"] = 0.9 * np.asarray(_r["energy_pred"])
    _r["L_pred"] = {k: 1.1 * np.asarray(v) for k, v in _r["L_pred"].items()}
for phase, when in (
        ("placing the new figures",
         lambda s_, d_: parent(s_) == STG
         and os.path.basename(d_) == "domain_growth_psi0p0.00.png"),
        ("holding the old figures",
         lambda s_, d_: parent(d_) == HOLD
         and os.path.basename(d_) == "domain_growth_psi0p0.00.png")):
    os.replace = replace_failing(when)
    raised = False
    try:
        wcase(new_res, (1, 2, 3), 0.0, 128, div, videos=False)
    except PermissionError:
        raised = True
    finally:
        os.replace = real_replace
    check(raised and contents(div) == old and not leftovers(div)
          and not leftovers(divx),
          f"failing while {phase}: every figure of the run folder is the "
          f"previous run's, nothing half-made is left")
    oldx = contents(divx)    # the extras go first, and stay (a second step)
for phase, when in (
        ("placing the new data",
         lambda s_, d_: parent(s_) == XSTG
         and os.path.basename(d_) == "domain_growth.csv"),
        ("holding the old data",
         lambda s_, d_: parent(d_) == XSTG + ".old"
         and os.path.basename(d_) == "domain_growth.csv")):
    os.replace = replace_failing(when)
    raised = False
    try:
        wcase(new_res, (1, 2, 3), 0.0, 128, div, videos=False)
    except PermissionError:
        raised = True
    finally:
        os.replace = real_replace
    check(raised and contents(divx) == oldx and contents(div) == old
          and not leftovers(div) and not leftovers(divx),
          f"failing while {phase} (the extras are published first): the "
          f"data are the previous run's and the run folder was not touched")


def publish_failing(folder, when, exc: type[BaseException] = PermissionError,
                    **kw):
    os.replace = replace_failing(when, exc)
    try:
        wcase(fake_results(0.0, 3), (1, 2, 3), 0.0, 128, folder,
                     videos=False, **kw)
    except BaseException as e:
        return type(e)
    finally:
        os.replace = real_replace
    return None


m1 = os.path.join(SCRATCH, "case_m1")
clean(m1, m1 + "_x")
wcase(fake_results(0.0, 3), (1, 2, 3), 0.0, 128, m1, videos=False)
for f in ("psi_distribution_psi0p0.00.png", "composition_psi0p0.00.png"):
    os.remove(os.path.join(m1, f))
old_m1 = contents(m1)
got_exc = publish_failing(
    m1, lambda s_, d_: parent(s_) == STG
    and os.path.basename(d_) == "domain_growth_psi0p0.00.png")
check(got_exc is PermissionError and contents(m1) == old_m1
      and not leftovers(m1),
      "a failed publish that was adding new names (two figures, in a "
      "folder written before them) takes the ones it had placed back out")
with open(os.path.join(m1 + "_x", "rollout.mp4"), "wb") as fh:
    fh.write(b"old film")
old_m1x = contents(m1 + "_x")
got_exc = publish_failing(
    m1, lambda s_, d_: parent(s_) == XSTG
    and os.path.basename(d_) == "domain_growth.csv", keep_film=False)
check(got_exc is PermissionError and contents(m1 + "_x") == old_m1x
      and not leftovers(m1 + "_x"),
      "a film the failed publish was dropping (a live run, keep_film=False) "
      "is put back too")
got_exc = publish_failing(
    div, lambda s_, d_: parent(s_) == STG
    and os.path.basename(d_) == "psi_distribution_psi0p0.00.png",
    KeyboardInterrupt)
check(got_exc is KeyboardInterrupt and contents(div) == old
      and not leftovers(div),
      "Ctrl-C mid-publish is reverted as well, not only an ordinary error")

wcase(new_res, (1, 2, 3), 0.0, 128, div, videos=False)
now = contents(div)
check(set(now) == set(old) and all(now[f] != old[f] for f in old)
      and not leftovers(div),
      f"unlocked, every figure and CSV is replaced together (same: "
      f"{sorted(f for f in old if now.get(f) == old[f])}, "
      f"set diff {sorted(set(now) ^ set(old))})")

print("\n12. a run killed mid-publish is undone by the next one")
old = contents(div)
kill_mid_publish(div)
mixed = contents(div)
check(os.path.isdir(os.path.join(div, HOLD))
      and "snapshots_psi0p0.00.png" in mixed
      and "composition_psi0p0.00.png" not in mixed
      and mixed["snapshots_psi0p0.00.png"] != old["snapshots_psi0p0.00.png"],
      "the simulated kill left a half-published case and its journal, "
      "inside the run folder")
buf = _io.StringIO()
with contextlib.redirect_stdout(buf):
    S.recover_interrupted(div)
check(contents(div) == old and not leftovers(div)
      and "undoing an interrupted publish" in buf.getvalue(),
      "recover_interrupted puts every original back and says so")

kill_mid_publish(div)
buf = _io.StringIO()
with contextlib.redirect_stdout(buf):
    wcase(fake_results(0.0, 3), (1, 2, 3), 0.0, 128, div, videos=False)
now = contents(div)
check("undoing an interrupted publish" in buf.getvalue()
      and not leftovers(div) and set(now) == set(old)
      and all(now[f] != old[f] for f in old
              if not f.startswith("domain_growth")),
      "the next write_case of the same case settles it by itself and then "
      "publishes cleanly")
old = contents(div)
kill_mid_publish(div)
with contextlib.redirect_stdout(_io.StringIO()):
    wcase(fake_results(0.4, 3), (1, 2, 3), 0.4, 128, div, videos=False)
now = contents(div)
check({f: now[f] for f in old} == old and not leftovers(div)
      and os.path.exists(os.path.join(div, "snapshots_psi0p0.40.png")),
      "a different case publishes its own files in the same flat folder, "
      "and settles the killed one on its way in")
kill_mid_publish(div)
with contextlib.redirect_stdout(_io.StringIO()):
    S.recover_run(DIVROOT, divx)
check({f: contents(div)[f] for f in old} == old and not leftovers(div),
      "recover_run over the results folder settles a killed one as well")
for f in [f for f in contents(div) if f.endswith("_psi0p0.40.png")]:
    os.remove(os.path.join(div, f))

old = contents(div)
kill_mid_publish(div)
os.replace = replace_failing(lambda s_, d_: parent(s_).endswith(".old"))
raised = False
try:
    with contextlib.redirect_stdout(_io.StringIO()):
        S.recover_interrupted(div)
except RuntimeError:
    raised = True
finally:
    os.replace = real_replace
check(raised and os.path.isdir(os.path.join(div, HOLD)),
      "an undo that cannot put the originals back raises and keeps the hold")
with contextlib.redirect_stdout(_io.StringIO()):
    S.recover_interrupted(div)
check(contents(div) == old and not leftovers(div),
      "and once unlocked, the next recovery restores every original")

print("\n13. a seed that diverged at its first step")
k0dir = os.path.join(SCRATCH, "results_k0")
clean(k0dir, os.path.join(SCRATCH, "rc_results_k0"))
setup(k0dir)
P.WRITE_VIDEO = False
res = fake_results(0.0, 3, diverge={0})
d0 = dict(res[2])
f0 = {tt: v[0] for tt, v in d0["snapshots"].items()}
d0.update(r2=np.full(d0["times"].size, P.R2_FLOOR),
          diag_times=d0["diag_times"][:0], energy_true=np.array([]),
          energy_pred=np.array([]), L_true={}, L_pred={},
          diverged_at=float(d0["times"][0]), t_stop=float(d0["times"][0]),
          stopped_early=True, mean_pred=np.full(d0["times"].size, np.nan),
          snapshots={tt: (f, f.copy() if tt == P.START_TIME
                          else np.full_like(f, np.nan))
                     for tt, f in f0.items()})
res[2] = d0
buf = _io.StringIO()
with contextlib.redirect_stdout(buf):
    case = P.write_case_outputs(res, (11, 12, 13), 0.0, None)
e = case.get("submission", {}).get(("compositions", "+0.00"))
check("submission_error" not in case and e is not None,
      f"the case outputs and the submission are both written "
      f"({case.get('submission_error')})")
if e:
    check(e["growth"]["seeds_ending_early"] == 2,
          "the growth figure averages over the seeds still running, "
          "including one with no measurement at all")
    check(e["snapshots"]["seed"] == 12,
          f"the late-diverging best-mean seed is passed over for the "
          f"snapshots: shows {e['snapshots']['seed']}")
    check(e["snapshots"]["r2_at_end_worst"] == P.R2_FLOOR,
          "the NaN final R2 counts at the floor, not as a hole")
k0case = os.path.join(xr(), "psi0+0.00_l128")
bests = sorted(d for d in os.listdir(k0case) if d.startswith("best_seed"))
check(case["stats"]["best_seed"] == 12 and case["stats"]["best_seed_rank"] == 2
      and bests == ["best_seed12"],
      f"the summary's best seed and the one best-seed folder follow the same "
      f"rule as the snapshots, and its rank says a higher mean was passed "
      f"over: {case['stats']['best_seed']}, {bests}")
check(np.isnan(P.seed_summary(d0, 13)["growth_exponent_truth"]),
      "seed_summary gives no exponent rather than a KeyError")

print("\n14. a seed that stopped early (EARLY_STOP)")
TITLES.clear()
res = fake_results(0.0, 3)
res[1] = stopped(res[1])
es = os.path.join(SCRATCH, "case_es")
clean(es)
entry, curve14 = wcase(res, (1, 2, 3), 0.0, 128, es, videos=False)
check(len(entry["snapshots"]["times"]) == 4,
      f"the one short seed does not cost the case its later columns: "
      f"{entry['snapshots']['times']}")
check(entry["psi_pdf"]["pred_seeds_excluded"] == 0
      and entry["psi_pdf"]["seeds_not_reaching_t_end"] == 1
      and any("2 of 3 seeds pooled" in t and "1 stopped earlier" in t
              for _, t in TITLES)
      and not any("histogram excludes" in t for _, t in TITLES),
      "P(psi) pools the two that reached the end, and does not claim the "
      "prediction left one out that the truth kept")
check(entry["growth"]["seeds_ending_early"] == 1
      and not any("diverged seed" in t for _, t in TITLES),
      "the growth title does not call an early stop a divergence")
check(any("over the 2 seeds that reach t" in t for t in snapshot_titles()),
      "the snapshot caption says its median is over the two that reached")
check(curve14["stats"]["n_end"] == 2,
      "the sweep curve knows only two seeds are left at the end")

print("\n15. a damaged manifest or curve store")
cor = os.path.join(SCRATCH, "results_cor")
clean(cor)
setup(cor)
run_main(["--eval-mode", "critical", "--sizes", "128,512", "--seeds", "2",
          "--no-video"])
with open(os.path.join(xr(), S.BOOKKEEPING), "w") as fh:
    fh.write('{"compositions": {"+0.00": ')
setup(cor)
log = run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "2",
                "--no-video", "--redo"])
mc = bookkeeping()
with np.load(os.path.join(xr(), S.CURVES_FILE)) as z:
    groups = {tuple(nm.split("__")[:2]) for nm in z.files}
check(os.path.exists(os.path.join(xr(), S.BOOKKEEPING + ".corrupt"))
      and "restarted" in mc and list(mc["sizes"]) == ["512"],
      "a truncated manifest is kept as .corrupt and a fresh one started")
check(len(groups) == 3 and mc["sweep"]["status"].startswith("r2_sweep drawn")
      and mc["sweep"]["undocumented_curves"]
      == ["compositions +0.00", "sizes 128"],
      f"no curve is thrown away, and the sweep names what it cannot "
      f"document: {len(groups)} {mc['sweep']}")

with open(os.path.join(xr(), S.CURVES_FILE), "wb") as fh:
    fh.write(b"PK\x03\x04 truncated")
setup(cor)
log = run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "2",
                "--no-video", "--redo"])
mc = bookkeeping()
check(os.path.exists(os.path.join(xr(), S.CURVES_FILE + ".corrupt"))
      and "(BadZipFile)" in log,
      "a truncated curve store (BadZipFile) is recognised and really moved "
      "aside -- np.load's leaked handle no longer blocks it on Windows")
check(mc["failed_last_run"] == [] and "FAILED" not in log,
      f"and it no longer stops the manifest: {mc['failed_last_run']} "
      f"{[ln for ln in log.splitlines() if 'FAILED' in ln][:3]}")
check(mc["sweep"]["missing_curves"] == [] and
      mc["sweep"]["status"].startswith("r2_sweep drawn"),
      f"with the restarted manifest documenting only l=512, the sweep is "
      f"drawn: {mc['sweep']['status']}")
open(os.path.join(xr(), S.CURVES_FILE), "wb").close()
check(S.load_curves(xr()) == {},
      "an empty curve store (EOFError) is set aside too")
check(not leftovers(cor) and not leftovers(xr()),
      "no temporary files left in the folder")

setup(cor)
run_main(["--eval-mode", "critical", "--sizes", "128,384,512", "--seeds", "2",
          "--no-video", "--redo"])
os.remove(os.path.join(xr(), S.CURVES_FILE))
sweep_before = read(os.path.join(cor, "l128", "r2_time.png"))
setup(cor)
log = run_main(["--eval-mode", "none", "--sizes", "384,512", "--seeds", "2",
                "--no-video", "--redo"])
mc = bookkeeping()
check(mc["sweep"]["status"].startswith("not redrawn")
      and mc["sweep"]["missing_curves"] == ["compositions +0.00",
                                            "sizes 128"]
      and read(os.path.join(cor, "l128", "r2_time.png")) == sweep_before
      and "NOT redrawn" in log,
      "cases with no stored curve keep the old r2_time.png of their size "
      "rather than one that drops a line, and the manifest says which")

print("\n16. the paired verdict reads its own numbers")


def row(off, delta, t):
    return {"off": off, "n": 50, "a": 0.9, "b": 0.9 + delta, "delta": delta,
            "se": abs(delta / t) if t else 0.0, "t": t}


cases_v = {
    "quiet, all down": [row(-0.4, -0.001, -1.0), row(0.0, -0.002, -1.5)],
    "quiet, mixed": [row(-0.4, -0.001, -1.0), row(0.0, 0.002, 1.5)],
    "one up": [row(-0.4, 0.01, 2.5), row(0.0, -0.001, -0.5)],
    "one down": [row(-0.4, -0.01, -2.5), row(0.0, 0.001, 0.5)],
    "both ways": [row(-0.4, -0.01, -2.5), row(0.0, 0.01, 2.5)]}
got = {k: S.paired_verdict(v) for k, v in cases_v.items()}
check(got["quiet, all down"][0] == "No, not measurably."
      and "Every change is negative" in " ".join(got["quiet, all down"][2]),
      "no significant change, one sign: says the sign is consistent")
check("not consistent" in " ".join(got["quiet, mixed"][2]),
      "no significant change, mixed signs: says so")
check(got["one up"][0].startswith("Possibly")
      and "Nowhere" not in " ".join(got["one up"][2]),
      "a significant gain is not followed by 'Nowhere ... helps'")
check(got["one down"][0].startswith("No --")
      and got["one down"][2][-1].startswith("Nowhere does"),
      "a significant loss reads as a complete sentence")
check(got["both ways"][0] == "Mixed.", "gains and losses: Mixed.")
check(all(b.endswith(".") and not b[0].islower()
          for _, _, body in got.values() for b in body),
      "every sentence of every branch is whole")

print("\n10a. nothing in sections 1 to 16 reached the project")
project_untouched()

restore()
matplotlib.figure.Figure.suptitle = _suptitle
matplotlib.axes.Axes.set_title = _set_title
matplotlib.axes.Axes.text = _text
clean(RES, RES + "_nohard", other, ab, ab + "_nohard", t200, t1500, DIVROOT,
      k0dir, es, cor, m1, film_dir, SAVED["RESULTS_DIR"],
      SAVED["RESULT_CACHE"], SAVED["GT_CACHE"],
      os.path.join(SCRATCH, "results_stamp_a"),
      os.path.join(SCRATCH, "results_stamp_b"))
finish()
