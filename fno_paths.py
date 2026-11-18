"""Where the project keeps its files, and how runs are named.

Three folders sit beside the code:

  Results/  what goes in a report: one folder per run label, and the index
  Work/     what runs need and make on the way: data/, gt_cache/,
            models/<train label>/, rollout_cache/<run label>/, logs/, slurm/
  Archive/  older outputs and code snapshots, moved there and never deleted;
            superseded/<label>_<fingerprint>_<time>/ holds the results of a
            run a different checkpoint has taken the place of

A label names one experiment everywhere, spelt as in the group's runs
so the architectures' folders line up:

  train_t1000                     a model trained on t = 0..1000
  train_t1000_lap                 ... with the Laplacian-output head
  train_t1000_nohard              ... the no-hard-constraints ablation
  train_t1000_mine                ... named with --tag mine
  train_t1000_lap_s1              ... trained with --seed 1
  train_t1000_lap_c24m8           ... with --channels 24 --modes 8
  train_t1000_lap_c24m8-s1-mine   ... and both, with --tag mine
  train_t1000_pred_t20000         that model rolled out from t = 100 to 20000
  train_t1000_pred_t00200-04000   ... from t = 200 to 4000

Paths are relative to the working directory, as they always were, and
nothing here changes directory.  The roots are read when a path is built, so
a test that rebinds WORK moves everything after it.  --smoke builds the whole
tree under Work/smoke/ instead.  Only the standard library is imported:
make_submission and migrate_layout use this without torch.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import shutil

RESULTS = "Results"
WORK = "Work"
ARCHIVE = "Archive"

DEFAULT_START = 100.0
MODEL_FILE = "fno_ch.pt"
STATE_FILE = "fno_ch_last.pt"
# What training writes beside the best checkpoint, in the model's folder: one
# entry per epoch, and the setup the model was trained with.
HISTORY_FILE = "fno_ch_history.json"
CONFIG_FILE = "fno_ch_config.json"

_NUM = r"\d+(?:p\d+)?"
_TAG = r"[A-Za-z0-9][A-Za-z0-9.+-]*"
_RESERVED = ("lap", "nohard")

# Rollout caches written before 2026-11-11 store two values under an older
# name: "seed_set" in meta.json (now "group") and the per-case key of
# extras/submission_manifest.json (now "flat").  They are read through the two
# functions below and never rewritten.  This is the only copy of that name in
# the live tree: tests/t_paths.py checks that no other line has it.
OLD_STORED_NAME = "uno"


def fast_figures():
    """True when FNO_FAST_FIGURES is set (the test suites set it; a real run
    does not): figures are then saved at half the DPI with light PNG
    compression, which is quicker to draw and to write."""
    return os.environ.get("FNO_FAST_FIGURES", "") not in ("", "0")


def figure_dpi(dpi):
    """``dpi`` as it will be saved: halved under FNO_FAST_FIGURES."""
    return max(1, round(dpi / 2)) if fast_figures() else dpi


def save_figure(fig, path, dpi, **kwargs):
    """fig.savefig(path, dpi=dpi, ...), the one place figures are written.

    Without FNO_FAST_FIGURES this is exactly fig.savefig.  With it the DPI is
    halved and a PNG is compressed at level 1.
    """
    if fast_figures():
        dpi = figure_dpi(dpi)
        if str(path).lower().endswith(".png"):
            kwargs["pil_kwargs"] = {"compress_level": 1}
    fig.savefig(path, dpi=dpi, **kwargs)


def seed_set_name(value):
    """A stored seed-set value as it is named now: "group" for the old name."""
    return "group" if value == OLD_STORED_NAME else value


def flat_entry(entry):
    """The flat-layout part of one case of extras/submission_manifest.json:
    its "flat" key, or the old key in a cache written before 2026-11-11.  None
    when ``entry`` is not a dict or has neither key."""
    if not isinstance(entry, dict):
        return None
    got = entry.get("flat")
    return got if got is not None else entry.get(OLD_STORED_NAME)
_LABEL = re.compile(
    rf"train_t(?P<t>{_NUM})(?P<lap>_lap)?(?P<nohard>_nohard)?"
    rf"(?:_(?P<tag>(?!pred_t){_TAG}))?"
    rf"(?:_pred_t(?:(?P<start>{_NUM})-)?(?P<end>{_NUM}))?")
_LEGACY = re.compile(r"fno_ch_T?(\d+(?:\.\d+)?)(_nohard)?")


def smoke_root():
    """Where --smoke puts everything it writes."""
    return os.path.join(WORK, "smoke")


def _work(name, root):
    return os.path.join(WORK if root is None else root, name)


def data_dir(root=None):
    return _work("data", root)


def gt_dir(root=None):
    return _work("gt_cache", root)


def ref_cache_dir(root=None):
    """Where the 4.0-spaced references (fno_gt, the group format) are kept: the
    environment variable FNO_GT_CACHE when it is set, else the gt_cache
    folder.  The 15-case set is about 860 GB in 800 files, so a cluster run
    points the variable at scratch.  ``root`` (--smoke) always wins: a smoke run
    writes only under its own folder."""
    env = os.environ.get("FNO_GT_CACHE")
    if root is None and env:
        return env
    return gt_dir(root)


def free_bytes(path):
    """Free bytes on the disk that holds ``path`` (its nearest existing
    parent when it does not exist yet).  Tests patch this.  It does not see
    a quota: the 200 GB GPFS quota on /iitjhome is not in disk_usage."""
    path = os.path.abspath(path)
    while not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    return shutil.disk_usage(path).free


def models_dir(root=None):
    return _work("models", root)


def model_dir(label, root=None):
    return os.path.join(models_dir(root), label)


def model_path(label, root=None):
    """The best checkpoint of the model ``label``."""
    return os.path.join(model_dir(label, root), MODEL_FILE)


def history_path(label, root=None):
    """The training history of the model ``label``: the entries of every
    epoch, beside its best checkpoint."""
    return os.path.join(model_dir(label, root), HISTORY_FILE)


def config_path(label, root=None):
    """The setup the model ``label`` was trained with (its size, seed and
    schedule), beside its best checkpoint."""
    return os.path.join(model_dir(label, root), CONFIG_FILE)


def rollout_root(root=None):
    return _work("rollout_cache", root)


def rollout_dir(label, root=None):
    return os.path.join(rollout_root(root), label)


def archive_dir(root=None):
    """Archive/, or the one inside the smoke root, so --smoke never moves
    anything into the real one."""
    return ARCHIVE if root is None else os.path.join(root, ARCHIVE)


def log_dir(root=None):
    return _work("logs", root)


def slurm_dir():
    return os.path.join(WORK, "slurm")


def results_root(root=None):
    return RESULTS if root is None else os.path.join(root, RESULTS)


def results_dir(label, root=None):
    return os.path.join(results_root(root), label)


def time_token(t, width):
    """A time for a label: 1000 -> '1000' (zero-padded to ``width``), and a
    fractional one with p for the point, 250.5 -> '0250p5'.  Times are
    rounded to six decimals, so 1000.0000004 is still '1000'."""
    t = round(float(t), 6)
    if t < 0:
        raise ValueError(f"a label cannot hold the negative time {t:g}")
    head, _, tail = f"{t:.6f}".rstrip("0").partition(".")
    return f"{int(head):0{width}d}" + (f"p{tail}" if tail else "")


def _time(token):
    return float(token.replace("p", "."))


def check_tag(tag):
    """Refuse a tag that would make a label ambiguous or an awkward
    folder name; returns the tag."""
    if (not re.fullmatch(_TAG, tag) or tag in _RESERVED
            or tag.startswith("pred_t")):
        raise ValueError(f"--tag {tag!r}: use letters, digits, '.', '+' and "
                         f"'-' only (no '_'), starting with a letter or "
                         f"digit, and not 'lap' or 'nohard'")
    return tag


# The tag slot of a train label holds, in this order and joined by '-':
#   c<C>m<M>   the model size, when --channels or --modes gave another
#   s<N>       the training seed, when --seed is not 0
#   <name>     the user's own --tag
# so a plain run keeps its old name and the three never collide.
_TAG_PARTS = re.compile(
    r"(?:c(?P<c>\d+)m(?P<m>\d+)(?:-|$))?(?:s(?P<s>\d+)(?:-|$))?(?P<name>.*)",
    re.DOTALL)


def split_tag(tag):
    """The parts of a label's tag slot: {"size": (channels, modes) or None,
    "seed": int (0 when absent), "name": the user's own tag or ''}."""
    m = _TAG_PARTS.fullmatch(tag or "")
    assert m is not None
    return {"size": None if m["c"] is None else (int(m["c"]), int(m["m"])),
            "seed": int(m["s"] or 0), "name": m["name"]}


def check_user_tag(name):
    """check_tag for a --tag typed by the user: it may not look like the
    size or seed token that --channels, --modes and --seed add."""
    check_tag(name)
    if split_tag(name)["name"] != name:
        raise ValueError(f"--tag {name!r}: c<C>m<M> and s<N> are what "
                         f"--channels, --modes and --seed put in the label; "
                         f"pick another name")
    return name


def compose_tag(size=None, seed=0, name=""):
    """The tag slot for a model of another ``size`` (channels, modes) than
    the file's, a training ``seed`` other than 0, and the user's own
    ``name``: c<C>m<M>[-s<N>][-<name>], '' for none of them."""
    if seed < 0:
        raise ValueError(f"the training seed cannot be negative ({seed})")
    parts = []
    if size is not None:
        parts.append(f"c{int(size[0])}m{int(size[1])}")
    if seed:
        parts.append(f"s{int(seed)}")
    if name:
        parts.append(check_user_tag(name))
    return "-".join(parts)


def train_label(t_train, lap=False, nohard=False, tag=""):
    """train_t<T:04d>[_lap][_nohard][_<tag>]: the name of a trained model.
    The tag is one token; compose_tag builds it from size, seed and name."""
    return (f"train_t{time_token(t_train, 4)}" + ("_lap" if lap else "")
            + ("_nohard" if nohard else "")
            + (f"_{check_tag(tag)}" if tag else ""))


def _start(start):
    if start is None or abs(float(start) - DEFAULT_START) < 1e-9:
        return ""
    return f"{time_token(start, 5)}-"


def run_label(train, end, start=None):
    """<train label>_pred_t<H:05d>, with the start before the end when it is
    not the usual t = 100: ..._pred_t00200-04000."""
    return f"{train}_pred_t{_start(start)}{time_token(end, 5)}"


def run_template(train, start=None):
    """run_label with the horizon left as {H} (a view's horizon, written as
    view_dir writes it, then zero-padded to five digits)."""
    return f"{train}_pred_t{_start(start)}{{H:0>5}}"


def parse_label(label):
    """The parts of a train or run label, or None if it is not one (a tag of
    'lap' or 'nohard' is not: check_tag refuses it, so it is a label out of
    order).

    Returns {"train", "t_train", "lap", "nohard", "tag", "start", "end"};
    start and end are None for a train label, and start is DEFAULT_START
    when the label does not give it.
    """
    m = _LABEL.fullmatch(label)
    if m is None or m["tag"] in _RESERVED:
        return None
    end = m["end"]
    train = label[:label.rfind("_pred_t")] if end else label
    return {"train": train, "t_train": _time(m["t"]),
            "lap": bool(m["lap"]), "nohard": bool(m["nohard"]),
            "tag": m["tag"] or "",
            "start": (_time(m["start"]) if m["start"] else DEFAULT_START)
            if end else None,
            "end": _time(end) if end else None}


def model_label(path):
    """The train label of a checkpoint, for naming its results.

    Its folder's name when that is a train label (Work/models/train_t1000/
    fno_ch.pt -> train_t1000); for a loose file from before the layout,
    fno_ch_T1000.pt or fno_ch_1000.pt -> train_t1000; otherwise the file's
    stem, which names the run without claiming a training horizon.
    """
    folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    p = parse_label(folder)
    if p is not None and p["end"] is None:
        return folder
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _LEGACY.fullmatch(stem)
    if m:
        return train_label(float(m[1]), nohard=bool(m[2]))
    return stem


def nohard_name(path):
    """``path`` named for the no-hard-constraints ablation: the token goes
    into the train label when the folder is named by one (also inside a
    {H} template), and is appended otherwise.  Unchanged if already there.
    """
    base = os.path.basename(path)
    head = path[:len(path) - len(base)]
    train, sep, rest = base.partition("_pred_t")
    p = parse_label(train)
    if p is not None and p["end"] is None:
        if p["nohard"]:
            return path
        train = train_label(p["t_train"], lap=p["lap"], nohard=True,
                            tag=p["tag"])
        return head + train + sep + rest
    if "_nohard" in base:
        return path
    if sep:
        return f"{head}{train}_nohard{sep}{rest}"
    return path + "_nohard"


def legacy(new, old, what=None):
    """``old`` if ``new`` is not there but ``old`` is, else ``new``.

    For an input the folder layout moved (the dataset, the references, the
    model), while it may still sit where it was; what a run writes always
    goes to the new place.  Given ``what``, a fallback says so in one line.
    """
    if os.path.exists(new) or not os.path.exists(old):
        return new
    if what:
        print(f"[layout] {what}: using {old}, as {new} is not there yet; "
              f"python migrate_layout.py moves it", flush=True)
    return old


def case_clash(path, root="."):
    """The on-disk name that stands in for a part of ``path``, or None.

    ``path`` is relative to ``root``.  Where the filesystem ignores case
    (Windows, macOS) the old results/ and the new Results/ are one folder,
    and a run writing into Results/ writes into the old one.  A part clashes
    when the folder holds an entry spelt differently that is the very same
    folder (os.path.samefile); where case counts, results and Results are
    two folders, or only one exists, and nothing clashes.
    """
    here = root
    for part in os.path.normpath(path).split(os.sep):
        try:
            names = os.listdir(here)
        except OSError:
            return None
        for name in names:
            if name == part or name.lower() != part.lower():
                continue
            try:
                same = os.path.samefile(os.path.join(here, name),
                                        os.path.join(here, part))
            except OSError:
                same = False
            if same:
                return name
        here = os.path.join(here, part)
    return None


_PRINTS: dict = {}


def fingerprint(path):
    """The first 12 hex digits of the checkpoint file's sha256: its identity,
    whatever it is called or when it was written.

    Hashed once and kept: asked again for a file of the same size and time
    it answers from memory; a file that changed is hashed afresh.  A run
    starts by forgetting what an earlier one in the same process hashed.
    """
    path = os.path.abspath(path)
    st = os.stat(path)
    seen = (st.st_size, st.st_mtime_ns)
    got = _PRINTS.get(path)
    if got is None or got[0] != seen:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        got = _PRINTS[path] = (seen, h.hexdigest()[:12])
    return got[1]


def forget_fingerprints():
    _PRINTS.clear()


def latest_model(root=None):
    """The most recently written best checkpoint under ``root``'s models
    folder, or None."""
    found = glob.glob(os.path.join(models_dir(root), "*", MODEL_FILE))
    return max(found, key=os.path.getmtime) if found else None


def latest_rollouts(root=None):
    """The most recently written folder of cached rollouts under ``root``,
    or None."""
    found = [p for p in glob.glob(os.path.join(_work("rollout_cache", root),
                                               "*")) if os.path.isdir(p)]
    return max(found, key=os.path.getmtime) if found else None


def under(path, root):
    """Whether ``path`` is ``root`` or lies inside it, both made absolute
    first, so a '..' or a folder that only shares the prefix does not count."""
    path, root = os.path.abspath(path), os.path.abspath(root)
    try:
        common = os.path.commonpath([path, root])
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(root)
