"""A publish that fails or is interrupted leaves the folders whole, and the
odd cases still draw (sections 19 to 23, 25 and 10 of the former one-file
t_submission): every seed diverged or stopped, temporaries and the stores,
bookkeeping that fails after the figures are published, a damaged result
cache, an interrupted publish settled by a run with no cases, the real
evaluate_seed on synthetic frames, and make_submission sharing fno_predict
when that runs as a script.

Shares tests/_submission_base.py with t_submission (sections 1 to 16) and
t_submission_regen (17, 18 and 24)."""
import contextlib
import io as _io
import json
import os
import shutil
import subprocess
import sys
import time

import matplotlib.axes
import matplotlib.figure
import numpy as np

from _paths import ROOT, SCRATCH
from _submission_base import (EXIT, HOLD, P, PLAIN, S, SAVED, TITLES,
                              _set_title, _suptitle, _text, bookkeeping, check,
                              clean, contents, fake_results, fake_run_case,
                              finish, flat, kill_mid_publish, leftovers,
                              leftovers_tree, panel_titles, project_untouched,
                              read, real_replace, replace_failing, restore,
                              rng, run_main, setup, snapshot_titles, stopped,
                              wcase, xr)

# --- state taken over from t_submission ---------------------------------------
# The scratch folder starts empty.  What sections 19, 20, 23 and 25 find there
# in the one-file suite is made again with the same calls (no checks):
#   div  the flat run folder with the case section 8 publishes, one seed of it
#        diverged; section 23 kills a publish in it.
#   es   the folder of the case section 14 publishes, one seed stopped early;
#        sections 19, 20 and 25 publish into it, and section 20 hands
#        curve14, the curve record of that publish, to record_cases.
DIVROOT = os.path.join(SCRATCH, "results_div")
div = DIVROOT                              # the flat run folder
divx = div + "_x"                          # the case's extras
clean(DIVROOT, divx)
restore()
es = os.path.join(SCRATCH, "case_es")
clean(es)
with contextlib.redirect_stdout(_io.StringIO()):
    res = fake_results(0.0, 3, diverge={2})
    entry, _ = wcase(res, (1, 2, 3), 0.0, 128, div, videos=False)
    res = fake_results(0.0, 3)
    res[1] = stopped(res[1])
    entry, curve14 = wcase(res, (1, 2, 3), 0.0, 128, es, videos=False)
TITLES.clear()

print("\n19. every seed diverged, or diverged and stopped")
TITLES.clear()
res = fake_results(0.0, 3, diverge={0, 1, 2})
with contextlib.redirect_stdout(_io.StringIO()):
    entry, _ = wcase(res, (1, 2, 3), 0.0, 128, es, videos=False)
check(entry["growth"]["seeds_ending_early"] == 3
      and any("3 diverged seeds end early" in t for _, t in TITLES),
      "seeds that all diverged at the same step are counted, not taken as "
      "the full length")
check(entry["snapshots"]["t_end"] == 2000.0
      and entry["snapshots"]["r2_at_end_median"] == P.R2_FLOOR
      and entry["snapshots"]["r2_at_end_diverged"] == 3
      and entry["psi_pdf"]["pred_seeds_excluded"] == 3
      and any("; diverged at t = " in t for t in snapshot_titles())
      and any("2000: diverged" in t for t in snapshot_titles()),
      "when every seed diverged the case is judged at the end, at the "
      "floor, and says so -- not at the last frame before the divergence")
all_div = entry
res = fake_results(0.0, 3, diverge={0, 1, 2})
for r in res:
    r["snapshots"] = {tt: (f, f.copy() if tt == P.START_TIME
                           else np.full_like(f, np.nan))
                      for tt, (f, _) in r["snapshots"].items()}
with contextlib.redirect_stdout(_io.StringIO()):
    entry, _ = wcase(res, (1, 2, 3), 0.0, 128, es, videos=False)
check(entry["snapshots"]["t_end"] == 2000.0
      and entry["snapshots"]["r2_at_end_median"] == P.R2_FLOOR
      and entry["snapshots"]["r2_at_end_worst"] == P.R2_FLOOR,
      "every seed diverging before the second snapshot scores the floor, "
      "not R2 = 1 at the input frame")
healthy, _ = wcase(fake_results(0.0, 3), (1, 2, 3), 0.0, 128, es,
                          videos=False)
check(S.folder_times([all_div, healthy])["times_agree"],
      "and a failed case does not make the folder's times disagree")
res = fake_results(0.0, 3, diverge={0})
res[1], res[2] = stopped(res[1]), stopped(res[2])
with contextlib.redirect_stdout(_io.StringIO()):
    entry, _ = wcase(res, (1, 2, 3), 0.0, 128, es, videos=False)
check(entry["snapshots"]["t_end"] == 1000.0
      and entry["snapshots"]["times"] == [100.0, 500.0, 1000.0]
      and entry["psi_pdf"]["t"] == 1000.0
      and np.isfinite(entry["psi_pdf"]["frac_pos_pred"]),
      "a diverged seed's NaN frames do not set the final time when the "
      "others stopped early")


full_t = np.arange(101.0, 2001.0)


def bare(times, snaps, diverged=None, nan_after=None, stop=False):
    """A minimal rollout: its times, which snapshots it has, whether (and
    after when) its prediction is NaN, and whether EARLY_STOP ended it."""
    t = np.asarray(times, dtype=float)
    f = np.tanh(3 * rng.standard_normal((16, 16)))
    return {"times": t, "r2": np.linspace(1.0, 0.9, t.size),
            "diverged_at": diverged, "stopped_early": bool(stop),
            "t_stop": float(t[-1]) if stop else float(full_t[-1]),
            "snapshots": {float(x): (f, np.full_like(f, np.nan)
                                     if nan_after is not None and x > nan_after
                                     else f + 0.01)
                          for x in snaps}}


mix = [bare(np.arange(101.0, 401.0), [100.0]),
       bare(full_t, [100.0, 500.0, 1000.0, 2000.0], 1400.0, 1400.0),
       bare(np.arange(101.0, 451.0), [100.0])]
check(S.rollout_times(mix)[1] == 1000.0 and S.best_index(mix) == 1,
      "EARLY_STOP with every survivor stopped before t = 500: judged where "
      "the seed that diverged later still had a prediction, not at the "
      "input frame")
old500 = [bare(np.arange(101.0, 501.0), [100.0, 500.0, 1000.0, 2000.0],
               300.0, 300.0) for _ in range(3)]
check(S.rollout_times(old500)[:2] == ([100.0, 500.0], 500.0),
      "every seed diverged with END_TIME = 500: judged at 500, and a cache "
      "that filled snapshots past the end does not label them later")
TITLES.clear()
early = [bare(np.arange(101.0, 401.0), [100.0]) for _ in range(3)]
with contextlib.redirect_stdout(_io.StringIO()):
    S.figure_snapshots(early, (1, 2, 3), 0.0, os.path.join(es, "early.png"))
check(any("no seed reached a snapshot after the input frame" in t
          for t in snapshot_titles()),
      "a case where nothing got past the input frame says so rather than "
      "just showing R2 = 1")

TITLES.clear()
res = fake_results(0.0, 3, diverge={2})
res[1] = stopped(res[1])
with contextlib.redirect_stdout(_io.StringIO()):
    entry, curve19 = wcase(res, (1, 2, 3), 0.0, 128, es, videos=False)
check(any("over the 2 seeds that reach t = 2000" in t
          and ": 1 (scored at R2 = -1)" in t
          and "1 seed stopped before it" in t for t in snapshot_titles()),
      "with one seed stopped and one diverged, the caption counts them apart")
check(curve19["stats"]["n_div"] == 1 and curve19["stats"]["n_end"] == 2,
      "the sweep curve counts the diverged seed among those at the end")
rec19 = S.curve_record(curve19)
check(S.curve_digest(rec19) != S.curve_digest(dict(rec19, n_end=3))
      and S.curve_digest(rec19) != S.curve_digest(dict(rec19, n_div=0)),
      "the digest changes with the end count and the divergence count")
TITLES.clear()
div_curve = flat(128)
div_curve["stats"] = {"n_seeds": 3, "n_end": 2, "n_div": 1}
with contextlib.redirect_stdout(_io.StringIO()):
    S.figure_sweeps({}, {128: flat(128), 512: div_curve},
                    os.path.join(es, "mixed.png"))
check(any("2 of 3 at $l = 512$, 1 of them diverged" in t
          for t in panel_titles()),
      "the sweep note says when a seed left at the end is a diverged one")

win = {"start": 100.0, "end": 2000.0, "snapshots": [100.0, 500.0, 1000.0,
                                                     2000.0]}
a_e = {"snapshots": {"times": [100.0, 500.0, 1000.0], "t_start": 100.0,
                     "t_end": 1000.0}, "run": {"window": win}}
b_e = {"snapshots": {"times": [100.0, 400.0, 1000.0, 2000.0],
                     "t_start": 100.0, "t_end": 2000.0}, "run": {"window": win}}
check(S.folder_times([a_e, b_e])["times_agree"]
      and not S.folder_times([{"snapshots": a_e["snapshots"]},
                              {"snapshots": b_e["snapshots"]}])["times_agree"],
      "agreement is judged from the configured window where there is one")
live1500 = {"run": {"window": {"start": 100.0, "end": 1500.0,
                               "snapshots": [100.0, 500.0, 1000.0]}}}
rebuilt = [bare(np.arange(101.0, 1501.0), [100.0, 500.0, 1000.0])
           for _ in range(2)]
check(S.data_window(rebuilt) == {"start": 100.0, "end": 1500.0,
                                 "snapshots": [100.0, 500.0, 1000.0],
                                 "truncated": False}
      and S.folder_times([live1500,
                          {"run": {"window": S.data_window(rebuilt)}}])[
          "times_agree"],
      "a rebuilt entry and a live one from the same config agree, even when "
      "END_TIME is not a snapshot time")


def rebuilt_entry(window):
    return {"run": {"source": "make_submission", "window": window}}


w1200 = S.data_window([bare(np.arange(101.0, 1201.0), [100.0, 500.0, 1000.0])])
w1300 = S.data_window([bare(np.arange(101.0, 1301.0), [100.0, 500.0, 1000.0])])
w_full = S.data_window([bare(full_t, [100.0, 500.0, 1000.0, 2000.0])])
w_early = S.data_window([bare(np.arange(101.0, 457.0), [100.0],
                              stop=True)])
live2000 = {"run": {"window": {"start": 100.0, "end": 2000.0,
                               "snapshots": [100.0, 500.0, 1000.0, 2000.0]}}}
check(S.folder_times([rebuilt_entry(w1200), rebuilt_entry(w1300)])[
          "times_agree"],
      "two rebuilt cases stopped at different times still agree")
check(S.folder_times([rebuilt_entry(w_full), rebuilt_entry(w_early),
                      live2000])["times_agree"],
      "a rebuilt case whose seeds all stopped early agrees with full ones, "
      "rebuilt or live, from the same config")
check(not S.folder_times([live1500, rebuilt_entry(w_full)])["times_agree"]
      and not S.folder_times([live1500, live2000])["times_agree"],
      "a configured END_TIME = 1500 does not agree with a 2000 window")
w_1500 = S.data_window([bare(np.arange(101.0, 1501.0),
                             [100.0, 500.0, 1000.0])])
check(not w_1500["truncated"] and w_early["truncated"]
      and not S.folder_times([live2000, rebuilt_entry(w_1500)])["times_agree"],
      "a rebuild from an END_TIME = 1500 cache is not mistaken for one that "
      "EARLY_STOP cut short")
w_late_stop = S.data_window([bare(full_t, [100.0, 500.0, 1000.0], stop=True)])
check(not S.folder_times([rebuilt_entry(w_late_stop),
                          rebuilt_entry(w_full)])["times_agree"],
      "a cut-short window that ran past the longer one's next snapshot "
      "does not agree")
check(not S.folder_times([live2000, {"run": {"window": {
          **live2000["run"]["window"], "start": 200.0}}}])["times_agree"],
      "windows seeded at different times do not agree")
check(not S.folder_times([
          {"snapshots": {"times": [100.0, 500.0], "t_start": 100.0,
                         "t_end": 500.0}},
          {"snapshots": {"times": [100.0, 500.0, 1000.0, 2000.0],
                         "t_start": 100.0, "t_end": 2000.0}}])["times_agree"],
      "older entries without a window are compared exactly")
check(S.data_window([bare(np.arange(101.0, 1201.0), [100.0, 500.0, 1000.0]),
                     bare(full_t, [100.0, 500.0, 1000.0, 2000.0])])
      == {"start": 100.0, "end": 2000.0,
          "snapshots": [100.0, 500.0, 1000.0, 2000.0], "truncated": False}
      and S.data_window(old500) == {"start": 100.0, "end": 500.0,
                                    "snapshots": [100.0, 500.0],
                                    "truncated": False},
      "data_window: seeds of unequal length, and snapshots past the data")

mix2 = [bare(full_t, [100.0, 500.0, 1000.0, 2000.0], 600.0, 600.0),
        bare(full_t, [100.0, 500.0, 1000.0, 2000.0], 1400.0, 1400.0),
        bare(np.arange(101.0, 401.0), [100.0])]
mix2[0]["r2"] = np.full(full_t.size, 0.99)
mix2[1]["r2"] = np.full(full_t.size, 0.5)
check(S.rollout_times(mix2)[1] == 1000.0 and S.best_index(mix2) == 1,
      "best_index skips a seed whose frame at the final time is NaN, even "
      "with the best mean")
check(P.pick_result(mix2)[:2] == (1, 3),
      "and fno_predict's best seed is the same one, ranked third by mean")
pair = [bare(np.arange(101.0, 1201.0), [100.0, 500.0, 1000.0]),
        bare(full_t, [100.0, 500.0, 1000.0, 2000.0], 1400.0, 1400.0)]
pair[0]["r2"] = np.full(1100, 0.5)
pair[1]["r2"] = np.full(full_t.size, 0.99)
check(S.best_index(pair) == 0,
      "and prefers a seed that never diverged to one that diverged later")
nanmix = [bare(full_t, [100.0, 500.0, 1000.0, 2000.0], 300.0, 300.0),
          bare(np.arange(101.0, 401.0), [100.0]),
          bare(np.arange(101.0, 451.0), [100.0])]
TITLES.clear()
with contextlib.redirect_stdout(_io.StringIO()):
    nm = S.figure_snapshots(nanmix, (1, 2, 3), 0.0,
                            os.path.join(es, "nanmix.png"))
    pdf_early = S.figure_psi_pdf(early, 0.0, os.path.join(es, "early_pdf.png"))
check(S.rollout_times(nanmix)[1] == 100.0 and nm["at_input_frame"]
      and any("no seed reached a snapshot after the input frame" in t
              for t in snapshot_titles()),
      "a divergence before t = 500 plus early stops: judged at the input "
      "frame, and the caption says why")
check(pdf_early["at_input_frame"]
      and any("prediction = truth by construction" in t
              for kind, t in TITLES if kind == "ax"),
      "P(psi) at the input frame says so too")
raised = False
try:
    S.rollout_times([bare(np.arange(101.0, 1501.0), [2000.0], 1000.0,
                          1000.0)])
except Exception:
    raised = True
check(not raised, "rollout_times survives every snapshot lying past the data")

def report_for(**man_extra):
    """report.txt for a manifest with these run-level fields and no cases."""
    man = {"label": "r", "written": "now", "cases": {}, "diverge_abs": 5.0,
           "threshold": 0.8, "checkpoint": "m.pt",
           "checkpoint_fingerprint": "abc", **man_extra}
    path = os.path.join(SCRATCH, "report_times.txt")
    with contextlib.redirect_stdout(_io.StringIO()):
        S.write_report(path, man, {})
    with open(path, encoding="utf-8") as fh:
        return fh.read()


rep_ = report_for(t_start=100.0, t_end=1500.0,
                  snapshot_times=[100.0, 500.0, 1000.0, 1500.0])
check("t = 100.0 to 1500.0, snapshots [100.0, 500.0, 1000.0, 1500.0]" in rep_
      and "2000" not in [ln for ln in rep_.splitlines()
                         if ln.strip().startswith("rolled out")][0],
      f"report.txt states the run's own times, not a hardcoded t = 2000 "
      f"{[ln for ln in rep_.splitlines() if '2000' in ln]}")
rep_ = report_for(t_start=200.0, t_end=1400.0,
                  snapshot_times=[500.0, 1000.0, 2000.0])
check("t = 200.0 to 1400.0" in rep_ and "t = 100" not in rep_,
      "and the rollout's own start, when it was seeded at t = 200")
rep_ = report_for(t_start=None, t_end=None, snapshot_times=None)
check("t = None to None" in rep_ and "Traceback" not in rep_,
      "a manifest with no cases still gets a report")

TITLES.clear()
at_end = [bare(full_t, [100.0, 500.0, 1000.0, 2000.0], 2000.0, 1999.0),
          bare(full_t, [100.0, 500.0, 1000.0, 2000.0]),
          bare(full_t, [100.0, 500.0, 1000.0, 2000.0])]
with contextlib.redirect_stdout(_io.StringIO()):
    sn_end = S.figure_snapshots(at_end, (1, 2, 3), 0.0,
                                os.path.join(es, "at_end.png"))
    pdf_end = S.figure_psi_pdf(at_end, 0.0, os.path.join(es, "at_end_pdf.png"))
check(sn_end["r2_at_end_diverged"] == 1
      and sn_end["r2_at_end_diverged_later"] == 0
      and pdf_end["pred_seeds_diverged_later"] == 0
      and all("diverged later: 0" in t for t in snapshot_titles())
      and not any("seeds diverged later" in t for _, t in TITLES),
      "a seed that diverges on the very last step (the default config) is "
      "counted once, at the floor, not also as diverging later")

print("\n20. temporaries and the stores")
jpath = os.path.join(es, "m14.json")
S.write_json(jpath, {"ok": 1})
try:
    S.write_json(jpath, {"ok": 2, "bad": object()})
except TypeError:
    pass
with open(jpath) as fh:
    kept = json.load(fh)
check(kept == {"ok": 1} and not os.path.exists(jpath + ".tmp"),
      "a manifest write that fails leaves the old file and no temporary")
with open(os.path.join(es, S.BOOKKEEPING), "w") as fh:
    fh.write("[]")
with contextlib.redirect_stdout(_io.StringIO()):
    mm = S.load_manifest(es)
check("restarted" in mm and os.path.exists(os.path.join(es,
                                                        S.BOOKKEEPING
                                                        + ".corrupt")),
      "a manifest that is valid JSON but not an object is set aside too")
np.savez(os.path.join(es, S.CURVES_FILE),
         sizes__512__times=np.arange(3.0), sizes__512__curve=np.ones(3),
         sizes__512__lattice=np.asarray(512))
check(S.load_curves(es) == {}, "a stored curve missing a part is dropped")

rec = os.path.join(SCRATCH, "rec_store")
clean(rec)
e14 = dict(entry)
S.record_cases(rec, {("sizes", "512"): e14}, curve14)
os.replace = replace_failing(
    lambda s_, d_: os.path.basename(d_) == S.CURVES_FILE)
try:
    S.record_cases(rec, {("sizes", "256"): e14}, curve14)
except PermissionError:
    pass
finally:
    os.replace = real_replace
with contextlib.redirect_stdout(_io.StringIO()):
    documented = {(k_, x) for k_ in ("compositions", "sizes")
                  for x in S.load_manifest(rec)[k_]}
check(documented <= set(S.load_curves(rec)) and not leftovers(rec),
      "record_cases stores the curve before the manifest mentions the case")
check(S.curves_as_cases(S.load_curves(rec))[(0.0, 512)]["stats"]["n_end"] == 2,
      "the end count survives the curve store")

for f in (S.STAGE, HOLD + ".stale1_2"):
    os.makedirs(os.path.join(es, f, "sub"), exist_ok=True)
    open(os.path.join(es, f, "half.png"), "wb").close()
for f in S.LEFTOVERS:
    open(os.path.join(es, f), "wb").close()
with contextlib.redirect_stdout(_io.StringIO()):
    S.recover_interrupted(es)
check(not leftovers(es) and not any(".stale" in f for f in os.listdir(es)),
      "a hard-killed run's scratch folders and temporaries are cleared")

hold = os.path.join(es, HOLD)
os.makedirs(hold)
open(os.path.join(hold, S.PUBLISH_DONE), "w").close()
open(os.path.join(hold, "snapshots.png"), "wb").close()
real_rmtree = shutil.rmtree


def rmtree_stuck(path, *a, **k):
    if os.fspath(path) == hold:
        return None
    return real_rmtree(path, *a, **k)


shutil.rmtree = rmtree_stuck
buf = _io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        S.recover_interrupted(es)
finally:
    shutil.rmtree = real_rmtree
check(not os.path.exists(hold) and "undoing" not in buf.getvalue()
      and any(".stale" in f for f in os.listdir(es)),
      "a committed hold that will not delete is moved aside, not undone")
with contextlib.redirect_stdout(_io.StringIO()):
    S.recover_interrupted(es)
check(not any(".stale" in f for f in os.listdir(es)),
      "and a later recovery removes it")

es_b = os.path.join(SCRATCH, "case_es_b")
clean(es_b)
os.makedirs(hold)
open(os.path.join(hold, S.PUBLISH_DONE), "w").close()
shutil.rmtree = rmtree_stuck
os.replace = replace_failing(lambda s_, d_: os.fspath(s_) == hold)
buf = _io.StringIO()
blocked = other_ok = False
try:
    with contextlib.redirect_stdout(buf):
        S.recover_interrupted(es)
        try:
            wcase(fake_results(0.0, 3), (1, 2, 3), 0.0, 128, es,
                         videos=False)
        except RuntimeError as exc:
            blocked = "still in use" in str(exc)
        wcase(fake_results(0.4, 3), (1, 2, 3), 0.4, 128, es_b,
                     videos=False)
        other_ok = True
finally:
    shutil.rmtree = real_rmtree
    os.replace = real_replace
check("WARNING" in buf.getvalue() and blocked and other_ok,
      "a hold that cannot be removed at all blocks only its own case folder, "
      "with a clear message; every other case still publishes")
with contextlib.redirect_stdout(_io.StringIO()):
    S.recover_interrupted(es)
check(not os.path.exists(hold), "and it goes once released")

st = os.path.join(SCRATCH, "case_st")
clean(st, st + "_x")
with contextlib.redirect_stdout(_io.StringIO()):
    wcase(fake_results(0.0, 3), (1, 2, 3), 0.0, 128, st, videos=False)
for f in ("psi_distribution_psi0p0.00.png", "composition_psi0p0.00.png"):
    os.remove(os.path.join(st, f))
kill_mid_publish(st, at="domain_growth_psi0p0.00.png")
st_hold = os.path.join(st, HOLD)


def rmtree_st(path, *a, **k):
    if os.fspath(path) == st_hold:
        return None
    return real_rmtree(path, *a, **k)


shutil.rmtree = rmtree_st
os.replace = replace_failing(lambda s_, d_: os.fspath(s_) == st_hold)
undone = False
try:
    with contextlib.redirect_stdout(_io.StringIO()):
        S.recover_interrupted(st)
        undone = not os.path.exists(os.path.join(st, "psi_distribution_psi0p0.00.png"))
        # a file of a name the undone publish had added, put there since
        with open(os.path.join(st, "psi_distribution_psi0p0.00.png"), "w") as fh:
            fh.write("published since")
        S.recover_interrupted(st)
    settled = sorted(os.listdir(st_hold))
finally:
    shutil.rmtree = real_rmtree
    os.replace = real_replace
check(undone and S.PUBLISH_SETTLED in settled
      and S.PUBLISH_JOURNAL not in settled
      and os.path.exists(os.path.join(st, "psi_distribution_psi0p0.00.png")),
      "an undone publish is marked settled, so a stuck hold is never undone "
      "twice -- a file of a name it had added, put there since, survives")
with contextlib.redirect_stdout(_io.StringIO()):
    S.recover_interrupted(st)
check(not os.path.exists(st_hold), "and the settled hold goes once released")

HOLD04 = S.stage_name("psi0p0.40") + ".old"
q4_hold = os.path.join(es_b, HOLD04)


def rmtree_q4(path, *a, **k):
    if os.fspath(path) == q4_hold:
        return None
    return real_rmtree(path, *a, **k)


before_q4 = read(os.path.join(es_b, "snapshots_psi0p0.40.png"))
shutil.rmtree = rmtree_q4
os.replace = replace_failing(lambda s_, d_: os.fspath(s_) == q4_hold)
buf = _io.StringIO()
returned = False
try:
    with contextlib.redirect_stdout(buf):
        wcase(fake_results(0.4, 3), (1, 2, 3), 0.4, 128, es_b,
                     videos=False)
    returned = True
finally:
    shutil.rmtree = real_rmtree
    os.replace = real_replace
check(returned and "WARNING" in buf.getvalue()
      and HOLD04 in buf.getvalue()
      and read(os.path.join(es_b, "snapshots_psi0p0.40.png")) != before_q4,
      "a publish that committed is not reported as failed when its hold "
      "cannot be removed; it warns and the new figures stay")
with contextlib.redirect_stdout(_io.StringIO()):
    S.recover_interrupted(es_b)
check(not os.path.exists(q4_hold), "and that hold goes once released")

TITLES.clear()
sweep_png = os.path.join(es, "sweeps.png")
with contextlib.redirect_stdout(_io.StringIO()):
    S.figure_sweeps({-0.4: flat(128), 0.0: flat(128)}, {}, sweep_png)
sweep_old = read(sweep_png)
real_savefig = matplotlib.figure.Figure.savefig


def savefig_fails(self, fname, *a, **k):
    real_savefig(self, fname, *a, **k)
    raise OSError("disk full (test)")


matplotlib.figure.Figure.savefig = savefig_fails
try:
    with contextlib.redirect_stdout(_io.StringIO()):
        S.figure_sweeps({-0.4: flat(512), 0.0: flat(512)}, {}, sweep_png)
except OSError:
    pass
finally:
    matplotlib.figure.Figure.savefig = real_savefig
check(read(sweep_png) == sweep_old and not leftovers(es),
      "a sweep that fails to save keeps the old sweeps.png and no temporary")

print("\n21. bookkeeping that fails after the figures are published")
rbk = os.path.join(SCRATCH, "results_rbk")
clean(rbk)
setup(rbk)
run_main(["--eval-mode", "critical", "--sizes", "128,512", "--seeds", "2",
          "--no-video"])
time.sleep(1.2)
once = []


def manifest_locked_once(s_, d_):
    if os.path.basename(d_) == S.BOOKKEEPING and not once:
        once.append(1)
        return True
    return False


setup(rbk)
os.replace = replace_failing(manifest_locked_once)
try:
    log = run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "2",
                    "--no-video", "--redo"])
finally:
    os.replace = real_replace
mr = bookkeeping()
check(once and "recording them failed" in log
      and mr["sizes"]["512"]["run"]["written"] == mr["last_run"]["written"]
      and "last_attempt_failed" not in mr["sizes"]["512"]
      and mr["failed_last_run"] == [] and EXIT[-1] == 0
      and mr["sweep"]["status"].startswith("r2_sweep drawn")
      and not leftovers_tree(rbk),
      "a manifest write that fails after the publish is recorded at the end "
      "of the run; the case is not called a failure")

once.clear()
setup(rbk)
os.replace = replace_failing(manifest_locked_once)
try:
    log = run_main(["--eval-mode", "none", "--sizes", "384", "--seeds", "2",
                    "--no-video"])
finally:
    os.replace = real_replace
mr = bookkeeping()
check("384" in mr["sizes"] and mr["sweep"]["undocumented_curves"] == [],
      "a case new to the folder whose own record failed is documented before "
      "the sweep is judged")

calls_npz = []


def backstop_only(s_, d_):
    if os.path.basename(d_) == S.CURVES_FILE:
        calls_npz.append(1)
        return len(calls_npz) > 1
    return False


setup(rbk)
os.replace = replace_failing(backstop_only)
try:
    log = run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "2",
                    "--no-video", "--redo"])
finally:
    os.replace = real_replace
mr = bookkeeping()
setup(rbk)
run_main(["--eval-mode", "critical", "--sizes", "128", "--seeds", "2",
          "--no-video", "--redo"])
mr2 = bookkeeping()
check(len(calls_npz) == 2 and "not_stored" not in mr["sweep"]
      and "could not store" not in log
      and mr2["sweep"]["status"].startswith("r2_sweep drawn"),
      "a failed end-of-run save of curves the case had already stored is "
      "not treated as a missing curve later")

setup(rbk)
os.replace = replace_failing(
    lambda s_, d_: os.path.basename(d_) == S.CURVES_FILE)
try:
    log = run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "3",
                    "--no-video", "--redo"])
finally:
    os.replace = real_replace
mr = bookkeeping()
check(mr["sweep"]["not_stored"] == ["sizes 512"]
      and mr["sizes"]["512"].get("curve_digest")
      and mr["sweep"]["status"].startswith("r2_sweep drawn")
      and not leftovers_tree(rbk),
      "a new curve that cannot be stored is drawn from memory and reported")
setup(rbk)
run_main(["--eval-mode", "critical", "--sizes", "128", "--seeds", "2",
          "--no-video", "--redo"])
mr = bookkeeping()
check(mr["sweep"]["missing_curves"] == ["sizes 512"]
      and mr["sweep"]["status"].startswith("not redrawn"),
      "a later run does not redraw from that entry's older stored curve")
setup(rbk)
run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "2",
          "--no-video", "--redo"])
mr = bookkeeping()
check(mr["sweep"]["status"].startswith("r2_sweep drawn")
      and mr["sweep"]["missing_curves"] == [],
      "and redoing that case clears it")

with open(os.path.join(xr(), S.BOOKKEEPING)) as fh:
    legacy = json.load(fh)
for kind in ("compositions", "sizes"):
    for e_ in legacy[kind].values():
        e_.pop("curve_digest", None)
with open(os.path.join(xr(), S.BOOKKEEPING), "w") as fh:
    json.dump(legacy, fh)
setup(rbk)
run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "2",
          "--no-video", "--redo"])
mr = bookkeeping()
check(mr["sweep"]["status"].startswith("r2_sweep drawn")
      and mr["sweep"]["missing_curves"] == [],
      "entries written before digests existed still count as having their "
      "stored curve")
with open(os.path.join(xr(), S.BOOKKEEPING)) as fh:
    legacy = json.load(fh)
legacy["sizes"]["384"].pop("curve_digest", None)
legacy["sizes"]["384"]["curve_not_stored"] = "2026-09-01T00:00:00"
with open(os.path.join(xr(), S.BOOKKEEPING), "w") as fh:
    json.dump(legacy, fh)
setup(rbk)
run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "2",
          "--no-video", "--redo"])
mr = bookkeeping()
check("sizes 384" in mr["sweep"]["missing_curves"]
      and mr["sweep"]["status"].startswith("not redrawn"),
      "an entry flagged curve_not_stored by the round-4 code still counts as "
      "missing its curve")
setup(rbk)
run_main(["--eval-mode", "none", "--sizes", "384", "--seeds", "2",
          "--no-video", "--redo"])

once.clear()
setup(rbk)
os.replace = replace_failing(manifest_locked_once)
real_wco2 = P.write_case_outputs


def publish_then_fail(results, seeds, off, lattice=None, key=None):
    real_wco2(results, seeds, off, lattice, key=key)
    raise OSError(28, "No space left on device (test)")


P.write_case_outputs = publish_then_fail
try:
    log = run_main(["--eval-mode", "none", "--sizes", "256", "--seeds", "2",
                    "--no-video"])
finally:
    os.replace = real_replace
    P.write_case_outputs = real_wco2
mr = bookkeeping()
check("256" in mr["sizes"]
      and mr["sizes"]["256"]["run"]["written"] == mr["last_run"]["written"]
      and "last_attempt_failed" not in mr["sizes"]["256"]
      and "sizes 256" not in mr["sweep"]["undocumented_curves"],
      "a case whose record failed and whose run then failed is still "
      "documented, from what it published")

SWEEP_IMG = sorted(f for f in os.listdir(os.path.join(rbk, "r2_sweep_per_case"))
                   if f.endswith(".png"))[0]
sweep_old = read(os.path.join(rbk, "r2_sweep_per_case", SWEEP_IMG))
real_sweeps = S.figure_r2_sweeps


def sweeps_fail(*a, **k):
    raise RuntimeError("deliberate")


S.figure_r2_sweeps = sweeps_fail
setup(rbk)
try:
    log = run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "2",
                    "--no-video", "--redo"])
finally:
    S.figure_r2_sweeps = real_sweeps
mr = bookkeeping()
check(mr["sweep"]["status"].startswith("FAILED")
      and "submission manifest and sweep" in (
          log.split("!" * 72)[-2] if log.count("!" * 72) >= 2 else "")
      and read(os.path.join(rbk, "r2_sweep_per_case", SWEEP_IMG)) == sweep_old
      and EXIT[-1] == 1,
      "a sweep that fails is written into the manifest, named in the "
      "banner, keeps the old figure, and fails the exit status")


def fails_192(model, device, off, lattice=None, seeds=None):
    if lattice == 512:
        raise RuntimeError("deliberate rollout failure")
    return fake_run_case(model, device, off, lattice, seeds)


setup(rbk)
P.run_case = fails_192
log = run_main(["--eval-mode", "critical", "--sizes", "128,512", "--seeds",
                "2", "--no-video", "--redo"])
mr = bookkeeping()
check("last_attempt_failed" in mr["sizes"]["512"]
      and "last_attempt_failed" not in mr["sizes"]["128"]
      and "last_attempt_failed" not in mr["compositions"]["+0.00"],
      "a case whose rollouts failed has its kept entry marked, and only it")


def fails_critical(model, device, off, lattice=None, seeds=None):
    if lattice is None and abs(off) < 1e-12:
        raise RuntimeError("deliberate rollout failure")
    return fake_run_case(model, device, off, lattice, seeds)


setup(rbk)
P.run_case = fails_critical
run_main(["--eval-mode", "critical", "--sizes", "128", "--seeds", "2",
          "--no-video", "--redo"])
mr = bookkeeping()
check("last_attempt_failed" in mr["compositions"]["+0.00"]
      and "last_attempt_failed" in mr["sizes"]["128"]
      and mr["sizes"]["128"]["run"]["written"] != mr["last_run"]["written"],
      "the failed critical run is marked, and so is its l = 128 alias: the "
      "grid has no case of its own for that size, so nothing wrote it")

setup(rbk)
real_wco = P.write_case_outputs


def wco_then_fail(results, seeds, off, lattice=None, key=None):
    real_wco(results, seeds, off, lattice, key=key)
    raise OSError(28, "No space left on device (test)")


P.write_case_outputs = wco_then_fail
try:
    run_main(["--eval-mode", "none", "--sizes", "512", "--seeds", "2",
              "--no-video", "--redo"])
finally:
    P.write_case_outputs = real_wco
mr = bookkeeping()
check("last_attempt_failed" not in mr["sizes"]["512"]
      and mr["sizes"]["512"]["run"]["written"] == mr["last_run"]["written"],
      "a case that fails after publishing its figures keeps the entry it "
      "wrote, unmarked")

mk = os.path.join(SCRATCH, "results_mk")
clean(mk)
setup(mk)
run_main(["--eval-mode", "both", "--sizes", "128", "--seeds", "2",
          "--no-video"])


def fails_m04(model, device, off, lattice=None, seeds=None):
    if lattice is None and off < -0.3:
        raise RuntimeError("deliberate rollout failure")
    return fake_run_case(model, device, off, lattice, seeds)


setup(mk)
P.run_case = fails_m04
run_main(["--eval-mode", "both", "--sizes", "128", "--seeds", "2",
          "--no-video", "--redo"])
check("last_attempt_failed" in bookkeeping()["compositions"]["-0.40"],
      "a failed off-critical composition has its kept entry marked")

setup(mk)
P.SWEEP_SEEDS = (901, 902)
run_main(["--eval-mode", "critical", "--sizes", "512", "--seeds", "2",
          "--no-video"])
mm = bookkeeping()
had_128 = "128" in mm["sizes"]
sw = os.path.join(xr(), "psi0+0.00_l128_sweep_seeds")
check(had_128 and mm["sizes"]["128"]["folder"] == "l128_sweep_seeds"
      and mm["sizes"]["128"]["seed_list"] == [901, 902]
      and mm["compositions"]["+0.00"]["folder"] == "psi0p0.00"
      and all(os.path.exists(os.path.join(sw, f)) for f in PLAIN)
      and os.path.exists(os.path.join(sw, "snapshots_psi0p0.00_sweep_seeds.png"))
      and not any("sweep_seeds" in f for f in os.listdir(mk))
      and os.path.exists(os.path.join(sw, "fno_ch_l128_summary.json")),
      "with sweep seeds that differ, the critical case on them gets its own "
      "folder in extras/, psi0+0.00_l128_sweep_seeds, with its figures (the flat "
      "layout has no such case, so the run folder gets none), and the l = 128 "
      "entry points there")


def fails_sweep_critical(model, device, off, lattice=None, seeds=None):
    if lattice == 128:
        raise RuntimeError("deliberate rollout failure")
    return fake_run_case(model, device, off, lattice, seeds)


setup(mk)
P.SWEEP_SEEDS = (901, 902)
P.run_case = fails_sweep_critical
run_main(["--eval-mode", "critical", "--sizes", "512", "--seeds", "2",
          "--no-video", "--redo"])
check(had_128 and "last_attempt_failed" in bookkeeping()["sizes"]["128"],
      "a failed rerun of the critical case on the sweep seeds is marked too")

print("\n22. the result cache survives a damaged file")
check(P.ENGINE == "seed" and not P.CONDITIONED,
      f"the seed engine runs here, as the stubbed evaluate_seed needs "
      f"(engine {P.ENGINE!r}, conditioned {P.CONDITIONED})")
rc = os.path.join(SCRATCH, "result_cache_rc")
rres = os.path.join(SCRATCH, "results_rc")
clean(rc, rres)
setup(rres)
P.WRITE_SUBMISSION = False
P.WRITE_VIDEO = False
P.RESULT_CACHE, P.CACHE_RESULTS = rc, True
rolled = []


def fake_evaluate(model, seed, device, off=0.0, lattice=None):
    rolled.append(seed)
    return fake_results(off, 1)[0]


P.evaluate_seed = fake_evaluate
path = P._result_path(0.0, 5, 128)
os.makedirs(rc, exist_ok=True)
with open(path, "wb") as fh:
    fh.write(b"PK\x03\x04 truncated by a SLURM kill")
with contextlib.redirect_stdout(_io.StringIO()):
    SAVED["run_case"](None, "cpu", 0.0, seeds=(5,))
check(rolled == [5] and os.path.exists(path + ".corrupt")
      and P._load_result(path)["r2"].size > 0
      and not [f for f in os.listdir(rc) if f.endswith(".tmp")],
      "an unreadable cached rollout is set aside and rolled out again, and "
      "the new one is written without a temporary left over")
with contextlib.redirect_stdout(_io.StringIO()):
    SAVED["run_case"](None, "cpu", 0.0, seeds=(5,))
check(rolled == [5], "and the rewritten file is reused next time")

good = read(path)
real_savez = np.savez


def savez_dies(file, *a, **k):
    file.write(b"PK\x03\x04 half")
    raise KeyboardInterrupt


def savez_full(file, *a, **k):
    file.write(b"PK")
    raise OSError(28, "No space left on device (test)")


for exc_name, dies in (("a kill", savez_dies), ("a full disk", savez_full)):
    setattr(np, "savez", dies)
    try:
        P._save_result(path, fake_results(0.0, 1)[0])
    except (KeyboardInterrupt, OSError):
        pass
    finally:
        setattr(np, "savez", real_savez)
    check(read(path) == good
          and not [f for f in os.listdir(rc) if f.endswith(".tmp")],
          f"{exc_name} during a cache save leaves the old file whole and no "
          f"temporary behind")

real_load = P._load_result


def load_oom(p_):
    raise MemoryError


P._load_result = load_oom
raised = False
try:
    with contextlib.redirect_stdout(_io.StringIO()):
        SAVED["run_case"](None, "cpu", 0.0, seeds=(5,))
except MemoryError:
    raised = True
finally:
    P._load_result = real_load
check(raised and read(path) == good,
      "running out of memory while loading a good cache file does not throw "
      "the file away")


def load_locked(p_):
    raise PermissionError(13, "locked (test)", p_)


def load_bad_zip(p_):
    raise NotImplementedError("That compression method is not supported")


rolled.clear()
P._load_result = load_locked
raised = False
buf = _io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        SAVED["run_case"](None, "cpu", 0.0, seeds=(5,))
except PermissionError:
    raised = True
finally:
    P._load_result = real_load
corrupts = sorted(f for f in os.listdir(rc) if ".corrupt" in f)
check(raised and not rolled and read(path) == good
      and len(corrupts) == 1
      and f"could not read {path}" in buf.getvalue()
      and "left in place" in buf.getvalue(),
      "a cache file that is locked, not damaged, is left alone, the error "
      "raised, and the file named")
first_corrupt = read(os.path.join(rc, corrupts[0]))
P._load_result = load_bad_zip
try:
    with contextlib.redirect_stdout(_io.StringIO()):
        SAVED["run_case"](None, "cpu", 0.0, seeds=(5,))
finally:
    P._load_result = real_load
corrupts = sorted(f for f in os.listdir(rc) if ".corrupt" in f)
check(rolled == [5] and len(corrupts) == 2
      and read(path + ".corrupt") == first_corrupt,
      "damaged zip metadata is a miss too, and a second damaged file does "
      "not overwrite the first one set aside")


def load_einval(p_):
    raise OSError(22, "Invalid argument (a seek before the start)")


P._load_result = load_einval
try:
    with contextlib.redirect_stdout(_io.StringIO()):
        SAVED["run_case"](None, "cpu", 0.0, seeds=(5,))
except Exception as exc:
    print(f"   (run_case raised {type(exc).__name__}: {exc})")
finally:
    P._load_result = real_load
check(rolled == [5, 5]
      and len([f for f in os.listdir(rc) if ".corrupt" in f]) == 3,
      "an EINVAL from a damaged zip offset is a miss, not an I/O error")
restore()

print("\n23. an interrupted publish is settled even by a run with no cases")
setup(DIVROOT)
before_kill = contents(div)
kill_mid_publish(div)
os.makedirs(os.path.join(div, S.STAGE, "sub"), exist_ok=True)
for f in S.LEFTOVERS:
    open(os.path.join(DIVROOT, f), "wb").close()
with contextlib.redirect_stdout(_io.StringIO()):
    try:
        P.write_submission([], [])
    except (Exception, SystemExit):
        # the folder holds plain per-case files, which a run folder does not
        # (the old flat layout): the settling comes first, the refusal after
        pass
check(not os.path.exists(os.path.join(div, HOLD))
      and contents(div) == before_kill and not leftovers(div)
      and not leftovers(DIVROOT),
      "write_submission settles an interrupted publish in the run folder "
      "and clears its own temporaries (recover_run)")

res_a = os.path.join(SCRATCH, "results_keya")
res_b = os.path.join(SCRATCH, "results_keyb")
clean(res_a, res_b)
setup(res_a)
P._SUBMITTED.clear()
with contextlib.redirect_stdout(_io.StringIO()):
    P.write_submission_case(fake_results(0.0, 2), (1, 2), 0.0, 512)
    key_a = os.path.abspath(P.submission_dir())
    P.RESULTS_DIR = res_b
    P.RESULT_CACHE = os.path.join(SCRATCH, "rc_results_keyb")
    shutil.rmtree(P.RESULT_CACHE, ignore_errors=True)
    P.write_submission_case(fake_results(0.4, 2), (1, 2), 0.4, None)
    P.write_submission([], [])
mb_ = bookkeeping()
check("512" not in mb_["sizes"] and "+0.40" in mb_["compositions"]
      and mb_["compositions"]["+0.40"]["folder"] == "psi0p0.40"
      and key_a in P._SUBMITTED
      and os.path.abspath(P.submission_dir()) not in P._SUBMITTED
      and os.path.exists(os.path.join(res_a, "l512", "snapshots_psi0p0.00.png"))
      and not os.path.exists(os.path.join(res_b, "l512", "snapshots_psi0p0.00.png")),
      "write_submission records only its own folder's cases (its extras "
      "folder's) and leaves another folder's pending")
P._SUBMITTED.clear()
restore()

print("\n25. the real evaluate_seed, on synthetic frames")


def real_rollout(end, diverge_at, seed):
    """evaluate_seed itself, with the solver and the model replaced."""
    restore()
    P.END_TIME, P.WRITE_VIDEO = float(end), False
    gt_t = np.arange(0.0, end + 1.0, P.DT_STEP)
    g = np.random.default_rng(seed)
    frames = np.tanh(3 * g.standard_normal((gt_t.size, 16, 16)))

    def fake_truth(lattice, seed_, off, t_end, dt, cache_dir=None,
                   device=None):
        return gt_t, frames

    def fake_frames(model, gt_times, gt_frames, t0, t1, device, amp_ctx=None):
        for tt in np.arange(t0 + P.DT_STEP, t1 + 0.5 * P.DT_STEP, P.DT_STEP):
            true = gt_frames[int(round(tt / P.DT_STEP))]
            pred = (np.full_like(true, np.nan) if tt >= diverge_at
                    else true + 0.02 * g.standard_normal(true.shape))
            yield float(tt), true, pred

    P.ch_ground_truth, P.rollout_frames = fake_truth, fake_frames
    try:
        with contextlib.redirect_stdout(_io.StringIO()):
            return SAVED["evaluate_seed"](None, seed, "cpu", 0.0, 16)
    finally:
        restore()


r500 = real_rollout(500, 300, 1)
check(r500["diverged_at"] is not None
      and sorted(r500["snapshots"]) == [100.0, 500.0],
      f"evaluate_seed no longer fills snapshots past END_TIME after a "
      f"divergence: {sorted(r500['snapshots'])}")
late4 = [real_rollout(1500, d, i) for i, d in
         enumerate((1100, 1200, 1300, 1400))]
TITLES.clear()
with contextlib.redirect_stdout(_io.StringIO()):
    e25, _ = wcase(late4, (1, 2, 3, 4), 0.0, 16, es, videos=False)
check(e25["snapshots"]["t_end"] == 1000.0
      and e25["snapshots"]["r2_at_end_diverged_later"] == 4
      and e25["psi_pdf"]["pred_seeds_diverged_later"] == 4
      and any("diverged later: 4 (by t = 1400)" in t
              for t in snapshot_titles())
      and any("4 seeds diverged later, by t = 1400" in t
              for kind, t in TITLES if kind == "ax"),
      "with END_TIME = 1500, seeds that diverged after the last snapshot are "
      "reported, not shown as a clean result")
one_late = late4[:1] + [real_rollout(1500, 1e9, i) for i in (7, 8)]
TITLES.clear()
with contextlib.redirect_stdout(_io.StringIO()):
    wcase(one_late, (1, 2, 3), 0.0, 16, es, videos=False)
check(any("1 seed diverged later, by t = 1100" in t for kind, t in TITLES
          if kind == "ax")
      and not any("1 seeds" in t for _, t in TITLES),
      "and one such seed is named in the singular, with its own time")

print("\n10. run as a script, make_submission shares this module")
src_path = os.path.join(ROOT, "fno_predict.py")
code = f"""
import sys, types
sys.path.insert(0, {ROOT!r})
src = open({src_path!r}, encoding="utf-8").read().rstrip()
tail = 'if __name__ == "__main__":\\n    sys.exit(main())'
assert src.endswith(tail), "the main block moved"
mod = types.ModuleType("__main__")
mod.__file__ = {src_path!r}
sys.modules["__main__"] = mod
exec(compile(src[:-len(tail)], mod.__file__, "exec"), mod.__dict__)
S = mod._submission()
print("SAME" if S.P is mod else "DIFFERENT")
"""
r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                   capture_output=True, text=True, timeout=300)
out = (r.stdout or "").strip().splitlines()
check(r.returncode == 0 and out and out[-1] == "SAME",
      f"no second copy of fno_predict is loaded "
      f"({out[-1] if out else r.stderr[-200:]})")
check(P._submission().P is P, "and imported normally it is the same too")

project_untouched()

restore()
matplotlib.figure.Figure.suptitle = _suptitle
matplotlib.axes.Axes.set_title = _set_title
matplotlib.axes.Axes.text = _text
clean(DIVROOT, divx, es, es_b, st, rec, rbk, mk, rc, rres, res_a, res_b,
      SAVED["RESULTS_DIR"], SAVED["RESULT_CACHE"], SAVED["GT_CACHE"])
finish()
