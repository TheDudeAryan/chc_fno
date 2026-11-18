"""fno_paths: where the project keeps its files, and how a run is named."""
import contextlib
import io as _io
import itertools
import os
import re
import shutil
import subprocess
import sys
import time

from _paths import ROOT, SCRATCH

import fno_paths as F

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


def raises(fn, *a, **k):
    """The ValueError message fn raises, or None if it did not."""
    try:
        fn(*a, **k)
    except ValueError as exc:
        return str(exc)
    return None


J = os.path.join

print("1. the roots, and the builders")
check((F.RESULTS, F.WORK, F.ARCHIVE) == ("Results", "Work", "Archive"),
      f"the three roots: {(F.RESULTS, F.WORK, F.ARCHIVE)}")
check(F.data_dir() == J("Work", "data")
      and F.gt_dir() == J("Work", "gt_cache")
      and F.models_dir() == J("Work", "models")
      and F.log_dir() == J("Work", "logs")
      and F.slurm_dir() == J("Work", "slurm")
      and F.smoke_root() == J("Work", "smoke"),
      "data, gt_cache, models, logs, slurm and smoke sit under Work/")
check(F.model_dir("train_t1000") == J("Work", "models", "train_t1000")
      and F.model_path("train_t1000")
      == J("Work", "models", "train_t1000", "fno_ch.pt")
      and F.rollout_dir("train_t1000_pred_t20000")
      == J("Work", "rollout_cache", "train_t1000_pred_t20000")
      and F.results_dir("train_t1000_pred_t20000")
      == J("Results", "train_t1000_pred_t20000"),
      "a model has a folder with fno_ch.pt in it, a run a rollout cache and "
      "a results folder")
check(F.STATE_FILE == "fno_ch_last.pt" and F.MODEL_FILE == "fno_ch.pt",
      "the checkpoint file names")
sm = F.smoke_root()
check(F.data_dir(sm) == J("Work", "smoke", "data")
      and F.model_path("x", sm) == J("Work", "smoke", "models", "x",
                                     "fno_ch.pt")
      and F.results_dir("x", sm) == J("Work", "smoke", "Results", "x")
      and F.rollout_dir("x", "r") == J("r", "rollout_cache", "x")
      and F.log_dir("r") == J("r", "logs"),
      "every builder can be pointed at another root, as --smoke does")

BUILDERS = (lambda: F.data_dir(), lambda: F.gt_dir(), lambda: F.models_dir(),
            lambda: F.model_dir("x"), lambda: F.model_path("x"),
            lambda: F.rollout_dir("x"), lambda: F.log_dir(),
            lambda: F.slurm_dir(), lambda: F.smoke_root())
real_work, real_results = F.WORK, F.RESULTS
before = [b() for b in BUILDERS]
try:
    F.WORK, F.RESULTS = "elsewhere", "Reports"
    after = [b() for b in BUILDERS]
    moved_results = F.results_dir("x")
finally:
    F.WORK, F.RESULTS = real_work, real_results
check(all(a == b.replace("Work", "elsewhere", 1)
          for a, b in zip(after, before)) and len(set(after)) == len(after),
      f"rebinding WORK moves every builder: {after}")
check(moved_results == J("Reports", "x"),
      "and rebinding RESULTS moves the results folders")
check([b() for b in BUILDERS] == before, "and put back, they are as before")

out = subprocess.run([sys.executable, "-c",
                      "import sys; sys.path.insert(0, %r); import fno_paths; "
                      "print('torch' in sys.modules, 'numpy' in sys.modules)"
                      % ROOT], capture_output=True, text=True, timeout=120)
check(out.stdout.strip() == "False False",
      f"importing it needs neither torch nor numpy ({out.stdout.strip()!r})")

print("\n2. time tokens")
for t, width, want in ((1000, 4, "1000"), (300, 4, "0300"), (0, 4, "0000"),
                       (100, 5, "00100"), (2000, 5, "02000"),
                       (20000, 5, "20000"), (20000, 4, "20000"),
                       (250.5, 4, "0250p5"), (0.25, 4, "0000p25"),
                       (300.00000000000006, 4, "0300"),
                       (1000.0000004, 4, "1000"), (999.9999996, 4, "1000"),
                       (0.0000004, 4, "0000"), (12.3456789, 4, "0012p345679"),
                       (1000.0, 4, "1000")):
    got = F.time_token(t, width)
    check(got == want, f"time_token({t!r}, {width}) -> {got}")
check("negative" in (raises(F.time_token, -1, 4) or ""),
      "a negative time cannot be in a label")
check(all(F._time(F.time_token(t, 4)) == t
          for t in (0, 1, 100, 250.5, 1000, 12345.25, 20000)),
      "and a token reads back as the time it came from")
check(all(F.parse_label("train_t" + F.time_token(t, 4)) is not None
          for t in (1000.0000004, 0.0000004, 2e-9, 999.9999996, 0.1 + 0.2)),
      "whatever the time, the token is one parse_label can read")

print("\n3. train and run labels")
for got, want in ((F.train_label(1000), "train_t1000"),
                  (F.train_label(300), "train_t0300"),
                  (F.train_label(20000), "train_t20000"),
                  (F.train_label(250.5), "train_t0250p5"),
                  (F.train_label(500, True), "train_t0500_lap"),
                  (F.train_label(1000, False, True), "train_t1000_nohard"),
                  (F.train_label(1000, False, False, "mine"),
                   "train_t1000_mine"),
                  (F.train_label(1000, True, True, "mine"),
                   "train_t1000_lap_nohard_mine"),
                  (F.train_label(1000.0, False, False, "T2000"),
                   "train_t1000_T2000")):
    check(got == want, f"train_label gives {want}: {got}")
check(F.train_label(1000, nohard=True, tag="a.b")
      == F.train_label(1000, False, True, "a.b") == "train_t1000_nohard_a.b",
      "by keyword or by position")

T1 = "train_t1000"
for args, want in (((T1, 20000), "train_t1000_pred_t20000"),
                   ((T1, 2000), "train_t1000_pred_t02000"),
                   ((T1, 2000, 100), "train_t1000_pred_t02000"),
                   ((T1, 2000, 100.0), "train_t1000_pred_t02000"),
                   ((T1, 2000, None), "train_t1000_pred_t02000"),
                   ((T1, 4000, 200), "train_t1000_pred_t00200-04000"),
                   ((T1, 10000, 50.5), "train_t1000_pred_t00050p5-10000"),
                   ((T1 + "_lap_mine", 10000),
                    "train_t1000_lap_mine_pred_t10000")):
    got = F.run_label(*args)
    check(got == want, f"run_label{args} -> {got}")
check(F.run_template(T1) == "train_t1000_pred_t{H:0>5}"
      and F.run_template(T1, 100.0) == "train_t1000_pred_t{H:0>5}"
      and F.run_template(T1, 200) == "train_t1000_pred_t00200-{H:0>5}",
      "a template leaves the horizon as {H:0>5}, after the start if it is "
      "not t = 100")
tpl = F.run_template(T1)
check(tpl.format(H="2000") == F.run_label(T1, 2000)
      and tpl.format(H="10000") == F.run_label(T1, 10000)
      and tpl.format(H="20000") == F.run_label(T1, 20000)
      and tpl.format(H="1") != tpl.format(H="2"),
      "filled with a view's horizon as view_dir writes it (f'{h:g}'), a "
      "template names the same folder run_label does")
tpl2 = F.run_template(T1, 200)
check(tpl2.format(H="4000") == F.run_label(T1, 4000, 200),
      "also from a start other than 100")

print("\n4. parse_label reads back what the builders wrote")
bad = []
n = 0
for t, lap, nohard, tag in itertools.product(
        (100, 300, 500, 1000, 20000, 250.5), (False, True), (False, True),
        ("", "mine", "T1000", "a.b-c+d", "9lives", "pred", "lapx")):
    train = F.train_label(t, lap, nohard, tag)
    p = F.parse_label(train)
    want = {"train": train, "t_train": float(t), "lap": lap,
            "nohard": nohard, "tag": tag, "start": None, "end": None}
    n += 1
    if p != want:
        bad.append((train, p))
    for end, start in itertools.product((2000, 20000, 4000.5),
                                        (None, 100, 200, 50.5)):
        run = F.run_label(train, end, start)
        q = F.parse_label(run)
        want = dict(want, start=100.0 if start is None else float(start),
                    end=float(end))
        n += 1
        if (q is None or q != want
                or F.run_label(q["train"], q["end"], q["start"]) != run):
            bad.append((run, q))
check(not bad, f"{n} labels parse to the parts they were built from, and "
               f"rebuild to themselves: {bad[:2]}")
for text in ("results", "fno_ch", "fno_ch_T1000", "train_1000", "train_t",
             "train_t1000_", "train_t10x0", "", "Train_t1000",
             "train_t1000_pred_t", "train_t1000_pred_t20000_x",
             "train_t1000_lap_lap", "train_t1000_nohard_lap",
             "train_t1000_nohard_nohard", "train_t1000_a_b",
             "train_t1000_pred_t0200-"):
    check(F.parse_label(text) is None, f"{text!r} is not a label")
check(F.parse_label("train_t1000_pred_t00200-04000")
      == {"train": "train_t1000", "t_train": 1000.0, "lap": False,
          "nohard": False, "tag": "", "start": 200.0, "end": 4000.0},
      "a run from another start gives both times")
check(F.parse_label("train_t1000_pred")
      == {"train": "train_t1000_pred", "t_train": 1000.0, "lap": False,
          "nohard": False, "tag": "pred", "start": None, "end": None},
      "a tag may be 'pred' (only 'pred_t' is the run marker)")

print("\n5. check_tag")
for tag in ("mine", "T1000", "a.b", "x-1", "9lives", "a+b", "A", "lapx",
            "nohardish"):
    check(F.check_tag(tag) == tag, f"{tag!r} is a fine tag")
for tag in ("", "lap", "nohard", "a_b", "_a", "-x", ".x", "+x", "a b",
            "a/b", "a\\b", "a:b", "a*", "ä", "pred_t5", "a\n"):
    msg = raises(F.check_tag, tag)
    check(msg is not None and repr(tag) in msg and "--tag" in msg,
          f"{tag!r} is refused, naming the tag ({msg})")
check(raises(F.train_label, 1000, tag="a_b") is not None
      and raises(F.train_label, 1000, tag="lap") is not None,
      "train_label refuses them too, rather than write an ambiguous label")
check(F.train_label(1000, tag="") == "train_t1000",
      "no tag is no tag")

print("\n6. model_label")
for path, want in (
        (J("Work", "models", "train_t1000", "fno_ch.pt"), "train_t1000"),
        (J("Work", "models", "train_t0500", "fno_ch_last.pt"), "train_t0500"),
        (J("Work", "models", "train_t1000_lap_nohard_mine", "fno_ch.pt"),
         "train_t1000_lap_nohard_mine"),
        (J("x", "train_t0250p5", "fno_ch.pt"), "train_t0250p5"),
        ("fno_ch_T1000.pt", "train_t1000"),
        (J("old", "fno_ch_T1000.pt"), "train_t1000"),
        ("fno_ch_T300_nohard.pt", "train_t0300_nohard"),
        ("fno_ch_T2000.5.pt", "train_t2000p5"),
        ("fno_ch.pt", "fno_ch"),
        ("other.pt", "other"),
        (J("Work", "models", "other.pt"), "other"),
        ("E:/HPC/fno_ch.pt", "fno_ch"),
        (J("Results", "train_t1000_pred_t20000", "fno_ch.pt"), "fno_ch"),
        (J("train_t1000", "x.pt"), "train_t1000"),
        ("fno_ch_500.pt", "train_t0500"),
        ("fno_ch_300.pt", "train_t0300"),
        (J("old", "fno_ch_1000.pt"), "train_t1000"),
        ("fno_ch_500_nohard.pt", "train_t0500_nohard"),
        ("fno_ch_2000.5.pt", "train_t2000p5"),
        ("fno_ch_mine.pt", "fno_ch_mine"),
        ("fno_ch_500_last.pt", "fno_ch_500_last"),
        ("fno_ch_T1000_last.pt", "fno_ch_T1000_last")):
    got = F.model_label(path)
    check(got == want, f"model_label({path}) -> {got}")
check(F.model_label(J(os.getcwd(), "Work", "models", "train_t1000",
                      "fno_ch.pt")) == "train_t1000",
      "an absolute path is read the same way")

print("\n7. nohard_name")
for path, want in (
        (J("Results", "train_t1000_pred_t20000"),
         J("Results", "train_t1000_nohard_pred_t20000")),
        (J("Results", "train_t1000_pred_t{H:0>5}"),
         J("Results", "train_t1000_nohard_pred_t{H:0>5}")),
        (J("Results", "train_t1000_lap_mine_pred_t02000"),
         J("Results", "train_t1000_lap_nohard_mine_pred_t02000")),
        (J("Results", "train_t1000_pred_t00200-04000"),
         J("Results", "train_t1000_nohard_pred_t00200-04000")),
        (J("Results", "train_t1000_pred_t00200-{H:0>5}"),
         J("Results", "train_t1000_nohard_pred_t00200-{H:0>5}")),
        (J("Results", "train_t1000_nohard_pred_t02000"),
         J("Results", "train_t1000_nohard_pred_t02000")),
        (J("Results", "train_t1000_nohard_pred_t{H:0>5}"),
         J("Results", "train_t1000_nohard_pred_t{H:0>5}")),
        (J("Work", "models", "train_t0500"),
         J("Work", "models", "train_t0500_nohard")),
        ("train_t1000_pred_t02000", "train_t1000_nohard_pred_t02000"),
        (J("out", "results"), J("out", "results_nohard")),
        (J("out", "results_P{H}"), J("out", "results_P{H}_nohard")),
        (J("out", "mine_nohard"), J("out", "mine_nohard")),
        (J("out", "results_P{H}_nohard"), J("out", "results_P{H}_nohard"))):
    got = F.nohard_name(path)
    check(got == want, f"nohard_name({path}) -> {got}")
check(F.nohard_name("Results\\train_t1000_pred_t{H:0>5}")
      == "Results\\train_t1000_nohard_pred_t{H:0>5}"
      and F.nohard_name("Results/train_t1000_pred_t02000")
      == "Results/train_t1000_nohard_pred_t02000"
      and F.nohard_name("/abs/dir/Results/train_t1000_pred_t02000")
      == "/abs/dir/Results/train_t1000_nohard_pred_t02000",
      "whichever separators the path uses are the ones it keeps")
tpl = J("Results", F.run_template("train_t1000"))
check(F.nohard_name(tpl).format(H="2000")
      == J("Results", F.run_label("train_t1000_nohard", 2000)),
      "a template named for the ablation still fills in to the ablation's "
      "run label")

print("\n8. legacy: an input that has not moved yet")
box = J(SCRATCH, "paths_legacy")
shutil.rmtree(box, ignore_errors=True)
os.makedirs(box)
here = os.getcwd()
os.chdir(box)


def quiet(fn, *a):
    log = _io.StringIO()
    with contextlib.redirect_stdout(log):
        got = fn(*a)
    return got, log.getvalue()


try:
    new, old = J("Work", "data", "test.npz"), "test.npz"
    got, log = quiet(F.legacy, new, old, "the dataset")
    check(got == new and log == "",
          f"neither there: the new place, and nothing said ({got!r})")
    with open(old, "w") as fh:
        fh.write("x")
    got, log = quiet(F.legacy, new, old, "the dataset")
    check(got == old and log.count("\n") == 1 and "the dataset" in log
          and old in log and new in log and "migrate_layout.py" in log,
          f"only the old one there: it is used, in one line that names both "
          f"places and the script that moves it ({log.strip()!r})")
    got, log = quiet(F.legacy, new, old)
    check(got == old and log == "",
          "and without a name for it nothing is printed")
    os.makedirs(os.path.dirname(new))
    with open(new, "w") as fh:
        fh.write("y")
    got, log = quiet(F.legacy, new, old, "the dataset")
    check(got == new and log == "",
          "both there: the new one, silently")
    os.remove(old)
    got, log = quiet(F.legacy, new, old, "the dataset")
    check(got == new and log == "", "only the new one there: the new one")
    os.makedirs("gt_cache")
    got, _ = quiet(F.legacy, J("Work", "gt_cache"), "gt_cache", "refs")
    check(got == "gt_cache", "a folder is found the same way as a file")
finally:
    os.chdir(here)
    shutil.rmtree(box, ignore_errors=True)

print("\n9. latest_model")
box = J(SCRATCH, "paths_latest")
shutil.rmtree(box, ignore_errors=True)
check(F.latest_model(box) is None, "no models folder: None")
now = time.time()
for i, name in enumerate(("train_t0300", "train_t0500", "train_t1000")):
    os.makedirs(F.model_dir(name, box))
    with open(F.model_path(name, box), "w") as fh:
        fh.write("m")
    stamp = now - 100 + 10 * i
    os.utime(F.model_path(name, box), (stamp, stamp))
with open(J(F.model_dir("train_t0500", box), "fno_ch_last.pt"), "w") as fh:
    fh.write("state")                      # the newest file of all
check(F.latest_model(box) == F.model_path("train_t1000", box),
      "the newest best checkpoint, whatever else is in the folders")
shutil.rmtree(box, ignore_errors=True)

print("\n10. ftp.sh names its model from END, VARIANT, CHANNELS, MODES and"
      " SEED: tested in t_protocol, section 5")

print("\n11. case_clash: results/ and Results/ where case is ignored")
box = J(SCRATCH, "paths_case")
shutil.rmtree(box, ignore_errors=True)
os.makedirs(J(box, "results", "old_run"))
os.makedirs(J(box, "Work", "models"))
folds = os.path.isdir(J(box, "RESULTS"))     # this filesystem ignores case
print(f"      ({'case is ignored' if folds else 'case counts'} on this "
      f"filesystem)")
check(F.case_clash("Results", box) == ("results" if folds else None),
      f"Results/ next to a lower-case results/: "
      f"{F.case_clash('Results', box)!r}")
check(F.case_clash(J("Results", "new_run"), box)
      == ("results" if folds else None),
      "the first part is the one that clashes, whatever follows it")
check(F.case_clash(J("Results", "old_run"), box)
      == ("results" if folds else None), "also when the rest is there too")
check(F.case_clash("results", box) is None,
      "a part spelt as it is on disk is no clash")
check(F.case_clash(J("results", "OLD_RUN"), box)
      == ("old_run" if folds else None), "a later part is checked as well")
check(F.case_clash(J("work", "Models"), box)
      == ("Work" if folds else None),
      "and the name returned is the one on disk")
check(F.case_clash("Archive", box) is None
      and F.case_clash(J("nothing", "Here"), box) is None,
      "a folder that is not there is no clash")
if not folds:
    os.makedirs(J(box, "Results"))
    check(F.case_clash("Results", box) is None
          and F.case_clash("results", box) is None,
          "where case counts, both folders existing is no clash")
shutil.rmtree(box, ignore_errors=True)

print("\n12. the history and config files of a model")
check(F.HISTORY_FILE == "fno_ch_history.json"
      and F.CONFIG_FILE == "fno_ch_config.json",
      "the names training writes beside the best checkpoint")
check(F.history_path("train_t1000_lap_c24m8")
      == J("Work", "models", "train_t1000_lap_c24m8", "fno_ch_history.json")
      and F.config_path("train_t1000")
      == J("Work", "models", "train_t1000", "fno_ch_config.json"),
      "history_path and config_path: in the model's folder")
check(os.path.dirname(F.history_path("train_t0500"))
      == os.path.dirname(F.model_path("train_t0500"))
      == os.path.dirname(F.config_path("train_t0500")),
      "the same folder as the best checkpoint")
check(F.history_path("train_t1000", F.smoke_root())
      == J("Work", "smoke", "models", "train_t1000", "fno_ch_history.json")
      and F.config_path("train_t1000", "elsewhere")
      == J("elsewhere", "models", "train_t1000", "fno_ch_config.json"),
      "under the root given, as the other builders: the smoke root's models")
real_work = F.WORK
F.WORK = J(SCRATCH, "paths_history", "W")      # a name only: nothing is made
try:
    moved = F.history_path("train_t0300")
finally:
    F.WORK = real_work
check(moved == J(SCRATCH, "paths_history", "W", "models", "train_t0300",
                 "fno_ch_history.json")
      and F.history_path("train_t0300")
      == J("Work", "models", "train_t0300", "fno_ch_history.json"),
      "read when called, so a test that rebinds WORK moves them too")

print("\n13. no live text file names the old seed set")
SKIP_DIRS = {"Archive", "Work", "Results", "env", ".venv", "__pycache__",
             "node_modules"}
TEXT_SUFFIX = (".py", ".sh", ".md", ".txt", ".json", ".toml", ".cfg", ".ini",
               ".yml", ".yaml", ".csv", ".tex", ".html")
scratch_top = J(ROOT, "tests", "_scratch")
read_only = "/iitjhome/m25ph1005/chc_" + F.OLD_STORED_NAME
name_re = re.compile(r"(?<![A-Za-z])" + re.escape(F.OLD_STORED_NAME)
                     + r"(?![A-Za-z])", re.I)
hits = []
for top, dirs, names in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")
               and J(top, d) != scratch_top]
    for nm in names:
        if nm.endswith((".out", ".err", ".log")) or not nm.endswith(
                TEXT_SUFFIX):
            continue
        path = J(top, nm)
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for no, ln in enumerate(lines, 1):
            if name_re.search(ln.replace(read_only, "")):
                hits.append((os.path.relpath(path, ROOT), no, ln))
helper = [h for h in hits if h[0] == "fno_paths.py"
          and h[2].startswith("OLD_STORED_NAME = ")]
extra = [h for h in hits if h not in helper]
for rel, no, _ in extra:
    print(f"     {rel}:{no}")
check(len(helper) == 1 and not extra,
      "no live text file names the old seed set, apart from the helper line "
      "in fno_paths.py")
check(F.seed_set_name(F.OLD_STORED_NAME) == "group"
      and F.seed_set_name("group") == "group"
      and F.seed_set_name("fno") == "fno"
      and F.flat_entry({"flat": {"a": 1}}) == {"a": 1}
      and F.flat_entry({F.OLD_STORED_NAME: {"a": 2}}) == {"a": 2}
      and F.flat_entry({"flat": {"a": 1}, F.OLD_STORED_NAME: {"a": 2}})
      == {"a": 1}
      and F.flat_entry({}) is None and F.flat_entry(None) is None
      and F.flat_entry("x") is None,
      "the two readers of old stored values: seed_set_name and flat_entry")

# FNO_FAST_FIGURES: off, nothing changes; on, half the DPI and light PNGs
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import matplotlib.image  # noqa: E402

saved_env = os.environ.get("FNO_FAST_FIGURES")
fig = matplotlib.figure.Figure(figsize=(3, 2))
fig.add_subplot().plot([0, 1], [0, 1])
os.makedirs(J(SCRATCH, "figdpi"), exist_ok=True)
sizes = {}
try:
    for mode in ("", "1"):
        if mode:
            os.environ["FNO_FAST_FIGURES"] = mode
        else:
            os.environ.pop("FNO_FAST_FIGURES", None)
        out = {}
        for ext in ("png", "pdf"):
            p = J(SCRATCH, "figdpi", f"f{mode or 0}.{ext}")
            F.save_figure(fig, p, 100, bbox_inches="tight")
            out[ext] = p
        sizes[mode] = (F.fast_figures(), F.figure_dpi(100),
                       matplotlib.image.imread(out["png"]).shape,
                       os.path.getsize(out["png"]), os.path.exists(out["pdf"]))
    os.environ["FNO_FAST_FIGURES"] = "0"
    zero = F.fast_figures()
finally:
    if saved_env is None:
        os.environ.pop("FNO_FAST_FIGURES", None)
    else:
        os.environ["FNO_FAST_FIGURES"] = saved_env
ref_p = J(SCRATCH, "figdpi", "plain.png")
fig.savefig(ref_p, dpi=100, bbox_inches="tight")
off, on = sizes[""], sizes["1"]
check(off[:2] == (False, 100) and on[:2] == (True, 50) and not zero
      and off[2] == matplotlib.image.imread(ref_p).shape
      and abs(on[2][0] * 2 - off[2][0]) <= 2 and on[4] and off[4],
      f"FNO_FAST_FIGURES off: plain savefig at 100 dpi {off[2]}; on: 50 dpi "
      f"{on[2]}; '0' is off; a PDF still saves")
check(open(J(SCRATCH, "figdpi", "f0.png"), "rb").read()
      == open(ref_p, "rb").read(),
      "off: the PNG is byte for byte what fig.savefig writes")

print("\n" + ("PATHS PASS" if ok else "PATHS FAIL"))
sys.exit(0 if ok else 1)
