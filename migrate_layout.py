#!/usr/bin/env python
"""Move the loose outputs of the old layout into Results/, Work/ and Archive/.

    python migrate_layout.py [--root DIR]            print the plan only
    python migrate_layout.py [--root DIR] --apply    carry it out

DIR is the project root (default: the current folder).  The plan is a table
of steps, each todo, done, clash, refused, absent or leave (failed appears
only while applying):

    fno_ch*.pt, fno_ch*_history.json   -> Work/models/<train label>/
    test.npz, test_numpy.npz, data/*   -> Work/data/
    gt_cache/, logs/                   -> Work/gt_cache/, Work/logs/
    result_cache*/                     -> Work/rollout_cache/
    results, results_*/                -> Results/<run label>/
    results_runs/                      -> Results/results_runs_2026-09-30/
    *_<job>.out, *_<job>.err           -> Work/slurm/
    Old/, archive/, Submission*, ...   -> Archive/
    copies of a file already moved     -> Archive/duplicates/

A checkpoint is filed by the training horizon T it was made with: from its
name, else run_meta.t_train, else a byte-identical twin, else the best
checkpoint it shares a best epoch with, else the state whose history it is.
A name that contradicts the file, a _nohard that disagrees with the config,
or a horizon nobody can vouch for makes the step refused.  Checkpoint
metadata is read without torch: the pickle in the zip is loaded by an
unpickler that turns every torch object into a stub.

Files are only moved (os.replace; shutil.move across filesystems), so bytes
and times are kept and a rollout-cache key made from a checkpoint's size and
mtime stays valid.  Nothing is deleted and nothing is overwritten: a clash
leaves the source where it is, a source folder left empty is reported, and
running again reports what is done.  --apply writes
Work/logs/migrate_layout_<time>.json with every move and how to undo it.
No SLURM job may be queued or running while it applies.
"""
from __future__ import annotations

import argparse
import builtins
import collections
import errno
import json
import os
import pickle
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from numpy.lib import format as npformat

sys.dont_write_bytecode = True  # a dry run leaves the project as it was
import fno_paths as FP

HISTORY_FILE = "fno_ch_history.json"
RUNS_DIR = "results_runs"
RUNS_DEST = f"{FP.RESULTS}/results_runs_2026-09-30"
DUPES = f"{FP.ARCHIVE}/duplicates"
FLAT_DIR = "results_flat_pre_folders"
BIG = 64 << 20
CHUNK = 8 << 20
NUM = r"\d+(?:\.\d+)?"
SAFE = ("dict", "list", "tuple", "set", "frozenset", "int", "float", "str",
        "bool", "bytes", "bytearray", "complex", "slice", "range", "object")
ROLE_FILE = {"best": FP.MODEL_FILE, "last": FP.STATE_FILE,
             "history": HISTORY_FILE}
ROLE_RANK = {"best": 0, "last": 1, "history": 2}

_LAST_N = re.compile(rf"fno_ch_last_({NUM})\.pt")
_HIST_N = re.compile(rf"fno_ch_history_({NUM})\.json")
_CKPT = re.compile(r"fno_ch(.*?)(_last\.pt|_history\.json|\.pt)")
_ROLE = {"_last.pt": "last", "_history.json": "history", ".pt": "best"}
_RES_T = re.compile(rf"results_T?(?P<t>{NUM})(?:_S(?P<s>{NUM}))?"
                    rf"(?:_P(?P<h>{NUM}))?(?P<nohard>_nohard)?")
_RES_DEFAULT = re.compile(rf"results(?P<n1>_nohard)?(?:_S(?P<s>{NUM}))?"
                          rf"(?:_P(?P<h>{NUM}))?(?P<n2>_nohard)?")
_CACHE_T = re.compile(rf"result_cache_T(?P<t>{NUM})(?P<nohard>_nohard)?"
                      rf"_P(?P<h>{NUM})")
_SLURM = re.compile(r".+_\d+\.(?:out|err)")


def log(*parts: object) -> None:
    print(*parts, flush=True)


def rel(path: str) -> str:
    """A path from fno_paths with '/' separators, as plan steps spell it."""
    return path.replace(os.sep, "/")


def at(root: str, name: str) -> str:
    return os.path.join(root, *name.split("/"))


def human(n: float) -> str:
    for unit, k in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= k:
            return f"{n / k:.2f} {unit}"
    return f"{int(n)} B"


def tree_size(path: str) -> int:
    """Bytes in a file, or in every file under a folder."""
    if not os.path.lexists(path):
        return 0
    if os.path.islink(path) or not os.path.isdir(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for here, _, files in os.walk(path):
        for name in files:
            p = os.path.join(here, name)
            if not os.path.islink(p):
                try:
                    total += os.path.getsize(p)
                except OSError:
                    pass
    return total


def near(a: object, b: object) -> bool:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return abs(float(a) - float(b)) < 1e-9


def same_bytes(a: str, b: str) -> bool:
    """True if two files hold the same bytes; stops at the first difference."""
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(CHUNK), fb.read(CHUNK)
            if x != y:
                return False
            if not x:
                return True


def same_folder(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def canon(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=lambda _: None)


class _Stub:
    """Stands in for any torch object while a checkpoint is unpickled."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __setstate__(self, state: object) -> None:
        pass


class _Reader(pickle.Unpickler):
    """Loads a checkpoint's dict; tensors and other torch objects are stubs."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins" and name in SAFE:
            return getattr(builtins, name)
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        return _Stub

    def persistent_load(self, pid: Any) -> Any:
        return None


def read_checkpoint(path: str) -> dict | None:
    """What a torch checkpoint holds (config, run_meta, best, epoch, history),
    with the tensors gone; None for a file that is not one."""
    try:
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.endswith("/data.pkl")]
            if not members:
                return None
            with zf.open(min(members, key=len)) as fh:
                found = _Reader(fh).load()
    except Exception:  # a damaged pickle can fail in any way
        return None
    return found if isinstance(found, dict) else None


def npz_gate(path: str) -> tuple[bool, str]:
    """Whether path is an npz holding train_X, judged by its header alone."""
    try:
        with zipfile.ZipFile(path) as zf, zf.open("train_X.npy") as fh:
            version = npformat.read_magic(fh)
            read = (npformat.read_array_header_1_0 if version == (1, 0)
                    else npformat.read_array_header_2_0)
            shape = tuple(read(fh)[0])
    except KeyError:
        return False, "no train_X.npy inside, so not a training dataset"
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as e:
        return False, f"not a readable npz: {e}"
    if len(shape) < 2:
        return False, (f"train_X has shape {shape}, expected "
                       f"(trajectories, frames, ...)")
    frames = shape[1]
    return True, (f"train_X {shape}: {frames} frames = "
                  f"horizon t = {(frames + 10) / 10:g}")


@dataclass
class Step:
    kind: str                 # mkdir, move, recase or leave
    src: str
    dst: str = ""
    status: str = "todo"
    why: str = ""
    size: int = 0
    group: str = ""           # steps that stand or fall together
    deferred: bool = False    # a mkdir that waits for a whole-folder move
    mtime: float = 0.0
    when: str = ""
    error: str = ""


class Plan:
    """The ordered steps, and what they claim so far."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.steps: list[Step] = []
        self.taken: set[str] = set()
        self.whole: dict[str, str] = {}
        self.emptied: list[str] = []
        self._same: dict[tuple[str, str], bool] = {}

    def path(self, name: str) -> str:
        return at(self.root, name)

    def has(self, name: str) -> bool:
        return os.path.lexists(self.path(name))

    def same(self, a: str, b: str) -> bool:
        """Cached same_bytes; says so first when the files are large."""
        key = (a, b) if a <= b else (b, a)
        if key not in self._same:
            if os.path.getsize(a) >= BIG and os.path.getsize(a) == \
                    os.path.getsize(b):
                log(f"[check] comparing {os.path.basename(a)} with "
                    f"{os.path.basename(b)} ({human(os.path.getsize(a))})")
            self._same[key] = same_bytes(a, b)
        return self._same[key]

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    def move(self, src: str, dst: str, why: str = "", refuse: str = "",
             group: str = "") -> Step:
        """Plan src -> dst (names from the root) and decide its status."""
        there, landed = self.has(src), self.has(dst)
        key = os.path.normcase(dst)
        size = tree_size(self.path(src if there else dst))
        if not there:
            status = "done" if landed else "absent"
        elif refuse:
            status, why = "refused", refuse
        elif landed:
            status, why = "clash", f"{dst} already exists; {src} stays"
        elif key in self.taken:
            status, why = "clash", (f"another step already moves something "
                                    f"to {dst}; {src} stays")
        else:
            status = "todo"
            self.taken.add(key)
        return self.add(Step("move", src, dst, status, why, size, group))

    def leave(self, src: str, why: str) -> Step:
        return self.add(Step("leave", src, status="leave", why=why))

    def emptying(self, folder: str, steps: list[Step]) -> None:
        if steps and all(s.status == "todo" for s in steps):
            self.emptied.append(folder)


@dataclass
class Ckpt:
    """A loose checkpoint, state or history file at the root."""
    name: str
    path: str
    role: str                 # best, last or history
    stem: str
    name_t: float | None
    nohard: bool              # as the name spells it
    size: int
    mtime: float
    meta: dict | None = None
    history: list | None = None
    t: float | None = None
    how: str = ""
    via: Ckpt | None = None
    hard_off: bool = False    # the model's own verdict
    tag: str = ""
    label: str = ""
    dst: str = ""
    refuse: str = ""


def parse_name(name: str) -> tuple[str, str, float | None, bool] | None:
    """(role, stem, T, nohard) from a loose checkpoint's name; T is None
    when the name carries none (fno_ch.pt, fno_ch_mine.pt)."""
    m = _LAST_N.fullmatch(name)
    if m:
        return "last", f"fno_ch_{m[1]}", float(m[1]), False
    m = _HIST_N.fullmatch(name)
    if m:
        return "history", f"fno_ch_{m[1]}", float(m[1]), False
    m = _CKPT.fullmatch(name)
    if m is None:
        return None
    mid, role = m[1], _ROLE[m[2]]
    nohard = mid.endswith("_nohard")
    core = mid[:-len("_nohard")] if nohard else mid
    stem = "fno_ch" + mid
    if core == "":
        return role, stem, None, nohard
    t = re.fullmatch(rf"_T?({NUM})", core)
    if t:
        return role, stem, float(t[1]), nohard
    if re.fullmatch(r"_[^_]+", core):
        return role, stem, None, nohard
    return None


def run_meta(c: Ckpt) -> dict:
    found = (c.meta or {}).get("run_meta")
    return found if isinstance(found, dict) else {}


def meta_t(c: Ckpt) -> float | None:
    t = run_meta(c).get("t_train")
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return float(t)
    return None


def meta_nohard(c: Ckpt) -> bool | None:
    """True if the config says the hard constraints are off; None if the file
    has no config to ask."""
    cfg = (c.meta or {}).get("config")
    if not isinstance(cfg, dict):
        return None
    return not (cfg.get("hard_mass_conservation", True)
                and cfg.get("zero_dc", True))


def best_of(c: Ckpt) -> dict | None:
    best = (c.meta or {}).get("best")
    if (isinstance(best, dict) and isinstance(best.get("epoch"), (int, float))
            and isinstance(best.get("score"), (int, float))):
        return best
    return None


def share_best(a: Ckpt, b: Ckpt) -> bool:
    x, y = best_of(a), best_of(b)
    return (x is not None and y is not None
            and near(x["epoch"], y["epoch"]) and near(x["score"], y["score"]))


def one_horizon(found: list[Ckpt]) -> float | None:
    """The horizon all of ``found`` agree on, or None."""
    ts = {k.t for k in found}
    return found[0].t if len(ts) == 1 else None


def load_ckpts(plan: Plan, entries: list[str]) -> list[Ckpt]:
    cks = []
    for n in entries:
        parsed = parse_name(n)
        full = plan.path(n)
        if parsed is None or not os.path.isfile(full):
            continue
        role, stem, name_t, nohard = parsed
        st = os.stat(full)
        c = Ckpt(n, full, role, stem, name_t, nohard, st.st_size, st.st_mtime)
        if role == "history":
            try:
                with open(full, encoding="utf-8") as fh:
                    found = json.load(fh)
                c.history = found if isinstance(found, list) else None
            except (OSError, ValueError):
                c.history = None
        else:
            c.meta = read_checkpoint(full)
            hist = (c.meta or {}).get("history")
            c.history = hist if isinstance(hist, list) else None
        cks.append(c)
    return cks


def placed_models(plan: Plan) -> list[Ckpt]:
    """Checkpoints already filed under Work/models, whose label vouches for
    their horizon: what lets a re-run place a state whose best moved first."""
    out = []
    base = plan.path(rel(FP.models_dir()))
    if not os.path.isdir(base):
        return out
    for label in sorted(os.listdir(base)):
        parsed = FP.parse_label(label)
        if parsed is None or parsed["end"] is not None:
            continue
        for role, f in (("best", FP.MODEL_FILE), ("last", FP.STATE_FILE)):
            path = os.path.join(base, label, f)
            if not os.path.isfile(path):
                continue
            st = os.stat(path)
            c = Ckpt(f"{rel(FP.model_dir(label))}/{f}", path, role, "", None,
                     parsed["nohard"], st.st_size, st.st_mtime)
            c.meta = read_checkpoint(path)
            hist = (c.meta or {}).get("history")
            c.history = hist if isinstance(hist, list) else None
            c.t, c.label = parsed["t_train"], label
            c.hard_off, c.tag = parsed["nohard"], parsed["tag"]
            out.append(c)
    return out


def find_horizons(plan: Plan, cks: list[Ckpt], placed: list[Ckpt]) -> None:
    """Fill in c.t for every checkpoint that can be vouched for."""
    for c in cks:
        mt = meta_t(c)
        if c.name_t is not None and mt is not None and not near(c.name_t, mt):
            c.refuse = (f"its name says t = {c.name_t:g} but its run_meta "
                        f"says t_train = {mt:g}")
        elif c.name_t is not None:
            c.t, c.how = c.name_t, "from its name"
        elif mt is not None:
            c.t, c.how = mt, "from run_meta.t_train"
    live = [c for c in cks if not c.refuse]
    pool = live + placed

    known = [c for c in pool if c.t is not None]
    for c in live:
        if c.t is None:
            twins = [k for k in known if k.size == c.size
                     and plan.same(c.path, k.path)]
            t = one_horizon(twins)
            if t is not None:
                c.t, c.how = t, f"byte-identical to {twins[0].name}"

    bests = [c for c in pool if c.role == "best" and c.t is not None]
    for c in live:
        if c.t is None and c.role == "last":
            twins = [k for k in bests if share_best(c, k)]
            t = one_horizon(twins)
            if t is not None:
                c.t, c.how = t, f"same best epoch and score as {twins[0].name}"

    states = [c for c in pool if c.role == "last" and c.t is not None
              and c.history is not None]
    for c in live:
        if c.t is not None or c.role != "history":
            continue
        twins = [k for k in states
                 if c.history is not None
                 and canon(k.history) == canon(c.history)]
        twins = twins or [k for k in live if k.role == "best"
                          and k.stem == c.stem and k.t is not None]
        t = one_horizon(twins)
        if t is not None:
            c.t, c.via = t, twins[0]
            c.how = (f"its history is that of {twins[0].name}"
                     if twins[0].role == "last"
                     else f"the history of {twins[0].name}")


def kind_of(nohard: bool) -> str:
    return "no-hard (_nohard)" if nohard else "hard-constraint"


def settle(c: Ckpt) -> None:
    """Apply the identity gates to c and give it its label."""
    if c.refuse:
        return
    if c.t is None:
        c.refuse = "cannot tell its training horizon; move it by hand"
        return
    if c.via is not None:
        if c.via.refuse:
            c.refuse = f"belongs with {c.via.name}, which is refused"
            return
        c.hard_off, c.tag = c.via.hard_off, c.via.tag
    else:
        verdict = meta_nohard(c)
        c.hard_off = c.nohard if verdict is None else verdict
        tag = run_meta(c).get("tag")
        auto = ("", f"T{c.t:g}")
        c.tag = tag if isinstance(tag, str) and tag not in auto else ""
    if c.hard_off != c.nohard:
        c.refuse = (f"its name says the {kind_of(c.nohard)} model but "
                    f"{'its config' if c.via is None else c.via.name} says "
                    f"the {kind_of(c.hard_off)} model")
        return
    try:
        if c.tag:
            FP.check_tag(c.tag)
        c.label = FP.train_label(c.t, nohard=c.hard_off, tag=c.tag)
    except ValueError as e:
        c.refuse = str(e)
        return
    c.dst = f"{rel(FP.model_dir(c.label))}/{ROLE_FILE[c.role]}"
    c.how = (f"t = {c.t:g} ({c.how})" + (f", tag {c.tag}" if c.tag else "")
             + (", no-hard" if c.hard_off else ""))


@dataclass
class Known:
    """A checkpoint whose model is known, for naming the results it made."""
    name: str                 # "" for a file already under Work/models
    size: int
    mtime: float
    label: str


def plan_checkpoints(plan: Plan, entries: list[str]) -> list[Known]:
    cks = load_ckpts(plan, entries)
    placed = placed_models(plan)
    find_horizons(plan, cks, placed)
    for c in sorted(cks, key=lambda c: c.role == "history"):
        settle(c)
    ready = sorted((c for c in cks if not c.refuse),
                   key=lambda c: (c.label, ROLE_RANK[c.role], c.name))
    seen = set()
    for c in ready:
        if c.dst in seen:
            continue
        seen.add(c.dst)
        group = sorted((k for k in ready if k.dst == c.dst),
                       key=lambda k: (k.mtime, k.name))
        landed = plan.has(c.dst)
        ref = c.dst if landed else group[0].name
        ref_path = plan.path(ref)
        if not landed:
            plan.move(group[0].name, c.dst, group[0].how)
            group = group[1:]
        for k in group:
            if os.path.isfile(ref_path) and plan.same(ref_path, k.path):
                plan.move(k.name, f"{DUPES}/{k.name}",
                          f"byte-identical to {ref}, which "
                          + ("is already in place" if landed
                             else "has the earliest mtime"))
            else:
                plan.add(Step("move", k.name, c.dst, "clash",
                              f"{ref} also maps to {c.dst} and differs; "
                              f"{k.name} stays", k.size))
    for c in sorted((c for c in cks if c.refuse), key=lambda c: c.name):
        plan.move(c.name, f"{rel(FP.models_dir())}/?/{ROLE_FILE[c.role]}",
                  refuse=c.refuse)
    return ([Known(c.name, c.size, c.mtime, c.label)
             for c in cks if not c.refuse and c.role != "history"]
            + [Known("", c.size, c.mtime, c.label) for c in placed])


@dataclass
class Data:
    src: str
    kind: str                 # npz, npy or other
    size: int
    mtime: float
    side: str = ""
    refuse: str = ""
    why: str = ""
    dup_of: str = ""


def plan_data(plan: Plan, entries: list[str]) -> None:
    dst_dir = rel(FP.data_dir())
    found: list[Data] = []

    def add(src: str, kind: str, side: str = "") -> Data:
        full = plan.path(src)
        d = Data(src, kind, tree_size(full), os.lstat(full).st_mtime, side)
        if kind == "npz":
            ok, d.why = npz_gate(full)
            d.refuse = "" if ok else d.why
        found.append(d)
        return d

    for n in ("test.npz", "test_numpy.npz"):
        if n in entries and os.path.isfile(plan.path(n)):
            add(n, "npz")
    folder = [n for n in entries if n == "data" and
              os.path.isdir(plan.path(n))]
    kids = sorted(os.listdir(plan.path("data"))) if folder else []
    stems = {os.path.splitext(k)[0] for k in kids if k.endswith(".npy")}
    for k in kids:
        src = f"data/{k}"
        stem, ext = os.path.splitext(k)
        if ext == ".json" and stem in stems:
            continue
        if ext == ".npz" and os.path.isfile(plan.path(src)):
            add(src, "npz")
        elif ext == ".npy" and os.path.isfile(plan.path(src)):
            side = f"data/{stem}.json"
            if plan.has(side) or plan.has(f"{dst_dir}/{stem}.json"):
                add(src, "npy", side if plan.has(side) else "")
            else:
                add(src, "npy").refuse = (
                    "an .npy without its .json sidecar (the file ch_gpu.py "
                    "--lean writes beside it); copy the sidecar here first")
        else:
            add(src, "other")

    firsts: list[Data] = []
    older: dict[str, str] = {}
    live = sorted((d for d in found if d.kind != "other" and not d.refuse),
                  key=lambda d: (d.mtime, d.src))
    for d in live:
        dest = f"{dst_dir}/{os.path.basename(d.src)}"
        twin = next((f for f in firsts if f.size == d.size
                     and plan.same(plan.path(f.src), plan.path(d.src))), None)
        if twin is not None:
            d.dup_of = twin.src
            older[d.src] = "which is older"
        elif os.path.isfile(plan.path(dest)) and plan.same(
                plan.path(dest), plan.path(d.src)):
            d.dup_of = dest
            older[d.src] = "which is already in place"
        else:
            firsts.append(d)

    steps: list[Step] = []
    for d in sorted(found, key=lambda d: (d.mtime, d.src)):
        name = os.path.basename(d.src)
        target = DUPES if d.dup_of else dst_dir
        why = d.why
        if d.dup_of:
            why = f"byte-identical to {d.dup_of}, {older[d.src]}"
        group = d.src if d.side else ""
        steps.append(plan.move(d.src, f"{target}/{name}", why,
                               refuse=d.refuse, group=group))
        if d.side:
            steps.append(plan.move(d.side,
                                   f"{target}/{os.path.basename(d.side)}",
                                   "the sidecar of the .npy above",
                                   group=group))
    if folder:
        if kids:
            plan.emptying("data", steps)
        else:
            plan.leave("data", "empty; remove by hand if you like")


def plan_folder(plan: Plan, entries: list[str], name: str, dest: str) -> None:
    """A folder that moves whole, or entry by entry if dest is there."""
    if name not in entries:
        plan.move(name, dest)
        return
    if not os.path.isdir(plan.path(name)):
        plan.leave(name, "is not a folder; left alone")
        return
    if not plan.has(dest):
        step = plan.move(name, dest, "the whole folder")
        if step.status == "todo":
            plan.whole[dest] = name
        return
    kids = sorted(os.listdir(plan.path(name)))
    if not kids:
        plan.leave(name, "empty; remove by hand if you like")
        return
    plan.emptying(name, [plan.move(f"{name}/{k}", f"{dest}/{k}")
                         for k in kids])


@dataclass
class Facts:
    error: str = ""
    name: str = ""
    size: int | None = None
    mtime: float | None = None
    start: float | None = None
    end: float | None = None


def number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def manifest_facts(folder: str) -> Facts:
    """Which checkpoint made a results folder, and its time window, from the
    manifest.json that make_submission.py keeps there."""
    try:
        with open(os.path.join(folder, "manifest.json"),
                  encoding="utf-8") as fh:
            m = json.load(fh)
    except (OSError, ValueError):
        return Facts("it has no usable manifest.json")
    if not isinstance(m, dict):
        return Facts("its manifest.json is not a JSON object")
    runs = []
    if isinstance(m.get("last_run"), dict):
        runs.append(m["last_run"])
    for kind in ("compositions", "sizes"):
        group = m.get(kind)
        if isinstance(group, dict):
            runs += [e["run"] for e in group.values()
                     if isinstance(e, dict) and isinstance(e.get("run"), dict)]
    names = {os.path.basename(str(r["model_path"]).replace("\\", "/"))
             for r in runs if r.get("model_path")}
    stats = {(r["model_bytes"], r["model_mtime"]) for r in runs
             if number(r.get("model_bytes")) is not None
             and isinstance(r.get("model_mtime"), str)}
    ck = m.get("checkpoints")
    if isinstance(ck, dict) and ck.get("mixed"):
        return Facts("its manifest says more than one checkpoint made it")
    if not names:
        return Facts("its manifest names no checkpoint")
    if len(names) > 1 or len(stats) > 1:
        return Facts("its manifest names more than one checkpoint")
    f = Facts(name=names.pop())
    if stats:
        size, stamp = stats.pop()
        f.size = int(size)
        try:
            f.mtime = datetime.fromisoformat(stamp).timestamp()
        except ValueError:
            f.size = None
    last = m.get("last_run")
    win = last.get("window") if isinstance(last, dict) else None
    win = win if isinstance(win, dict) else {}
    times = m.get("snapshot_times")
    snaps = ([t for t in map(number, times) if t is not None]
             if isinstance(times, list) else [])
    f.start = number(win.get("start"))
    if f.start is None:
        f.start = number(m.get("rollout_start"))
    f.end = number(win.get("end"))
    if f.end is None:
        f.end = max(snaps) if snaps else number(m.get("pdf_time"))
    return f


def model_of(known: list[Known], f: Facts) -> tuple[str, str]:
    """(train label, why) of the model a manifest names, or ("", why not)."""
    stat = f.size is not None and f.mtime is not None

    def same_file(k: Known) -> bool:
        return (stat and k.size == f.size and f.mtime is not None
                and abs(k.mtime - f.mtime) <= 2.0)

    named = [k for k in known if k.name == f.name]
    pick = [k for k in named if same_file(k)] if stat else named
    if not pick and stat:
        pick = [k for k in known if same_file(k)]
    labels = sorted({k.label for k in pick})
    if len(labels) == 1:
        return labels[0], f"its manifest names {f.name}: model {labels[0]}"
    if labels:
        return "", (f"its manifest names {f.name}, which fits "
                    f"{', '.join(labels)}")
    if named:
        return "", (f"its manifest names {f.name}, but the file of that "
                    f"name is not the one that made it (size or time)")
    return "", f"its manifest names {f.name}, which is not found here"


def results_label(plan: Plan, name: str, known: list[Known]
                  ) -> tuple[str, str]:
    """(run label, why) for a results folder, or ("", why it is refused)."""
    t_m = _RES_T.fullmatch(name)
    d_m = None if t_m else _RES_DEFAULT.fullmatch(name)
    m = t_m or d_m
    if m is None:
        return "", ("not a results folder name this script knows; move it "
                    "by hand")
    f = manifest_facts(plan.path(name))
    if t_m is not None:
        train = FP.train_label(float(t_m["t"]), nohard=bool(t_m["nohard"]))
        why = "named by its folder"
    else:
        if f.error:
            return "", f"cannot tell which model made it: {f.error}"
        train, why = model_of(known, f)
        if not train:
            return "", f"cannot tell which model made it: {why}"
        says = bool(m["n1"] or m["n2"])
        parsed = FP.parse_label(train)
        if says and parsed is not None and not parsed["nohard"]:
            return "", (f"its name says _nohard but the model it names "
                        f"({train}) has the hard constraints")
    start = float(m["s"]) if m["s"] else f.start
    end = float(m["h"]) if m["h"] else f.end
    if end is None:
        return "", ("cannot tell its horizon: no _P<H> in the name and "
                    "none in a manifest")
    return FP.run_label(train, end, start=start), why


def plan_results(plan: Plan, entries: list[str], known: list[Known]) -> None:
    skip = (RUNS_DIR, FLAT_DIR)
    names = [n for n in entries
             if (n == "results" or n.startswith("results_"))
             and n not in skip and os.path.isdir(plan.path(n))]
    first = len(plan.steps)
    for n in names:
        label, why = results_label(plan, n, known)
        if label:
            plan.move(n, f"{FP.RESULTS}/{label}", why)
        else:
            plan.move(n, f"{FP.RESULTS}/?", refuse=why)
    plan.move(RUNS_DIR, RUNS_DEST,
              "the whole folder; its README.md arrives unchanged")
    alias = next((s for s in plan.steps[first:] if s.src == "results"
                  and s.kind == "move"), None)
    if (alias is not None and alias.status != "todo"
            and same_folder(plan.path("results"), plan.path(FP.RESULTS))):
        for s in plan.steps[first:]:
            if s is not alias and s.status == "todo":
                s.status = "refused"
                s.why = (f"results/ is the same folder as {FP.RESULTS}/ on "
                         f"this filesystem and it is not being moved "
                         f"({alias.status}); sort it out first")


def plan_archive(plan: Plan, entries: list[str]) -> None:
    plan.move("Old", f"{FP.ARCHIVE}/Old")
    lower = next((n for n in entries
                  if n.lower() == FP.ARCHIVE.lower() and n != FP.ARCHIVE
                  and os.path.isdir(plan.path(n))), "")
    if lower:
        if plan.has(FP.ARCHIVE) and same_folder(plan.path(lower),
                                                plan.path(FP.ARCHIVE)):
            plan.add(Step("recase", lower, FP.ARCHIVE, "todo",
                          "the same folder on this filesystem: only the "
                          "capital changes, nothing inside moves",
                          tree_size(plan.path(lower))))
        else:
            kids = sorted(os.listdir(plan.path(lower)))
            steps = [plan.move(f"{lower}/{k}", f"{FP.ARCHIVE}/{k}")
                     for k in kids]
            if kids:
                plan.emptying(lower, steps)
            else:
                plan.leave(lower, "empty; remove by hand if you like")
    for n in entries:
        if n.startswith("Submission") and os.path.isdir(plan.path(n)):
            plan.move(n, f"{FP.ARCHIVE}/{n}")
    plan.move("Submission.zip", f"{FP.ARCHIVE}/Submission.zip")
    plan.move(FLAT_DIR, f"{FP.ARCHIVE}/{FLAT_DIR}")


KEEP_FILES = ("README.md", "progress.md", "pyrightconfig.json")
KEEP_DIRS = ("env", "tests", "__pycache__")
NEW_DIRS = (FP.WORK, FP.RESULTS, FP.ARCHIVE)


def plan_leave(plan: Plan, entries: list[str]) -> None:
    claimed = {s.src.split("/")[0] for s in plan.steps}
    rest = [n for n in entries if n not in claimed]
    code = [n for n in rest if n.endswith((".py", ".sh"))
            and os.path.isfile(plan.path(n))]
    groups = (
        ("*.py and *.sh", code, f"{len(code)} code files stay beside the "
                                f"code"),
        (", ".join(n for n in KEEP_FILES if n in rest),
         [n for n in KEEP_FILES if n in rest], "notes and settings stay"),
        (", ".join(f"{n}/" for n in KEEP_DIRS if n in rest),
         [n for n in KEEP_DIRS if n in rest], "stay where they are"),
        (", ".join(f"{n}/" for n in NEW_DIRS if n in rest),
         [n for n in NEW_DIRS if n in rest], "the new layout"),
    )
    shown = set()
    for label, members, why in groups:
        if members:
            plan.leave(label, why)
            shown.update(members)
    for n in rest:
        if n in shown:
            continue
        plan.leave(n, "hidden; left alone" if n.startswith(".")
                   else "not recognised; left alone")


def folder_steps(plan: Plan) -> list[Step]:
    names = [rel(FP.data_dir()), rel(FP.gt_dir()), rel(FP.models_dir()),
             rel(os.path.dirname(FP.rollout_dir("x"))), rel(FP.log_dir()),
             rel(FP.slurm_dir()), rel(FP.results_root()), FP.ARCHIVE]
    out = []
    for n in names:
        if plan.has(n):
            out.append(Step("mkdir", n, status="done"))
        elif n in plan.whole:
            out.append(Step("mkdir", n, why=f"made by the move of "
                                            f"{plan.whole[n]}/ below",
                            deferred=True))
        else:
            out.append(Step("mkdir", n))
    return out


def cache_label(name: str) -> str:
    """The rollout-cache folder a result cache moves to: fp_long.sh's
    result_cache_T1000_P20000 becomes the run label fno_predict now derives
    (train_t1000_pred_t20000), so a stopped job finds its cached seeds;
    any other name is kept."""
    m = _CACHE_T.fullmatch(name)
    if m is None:
        return name
    train = FP.train_label(float(m["t"]), nohard=bool(m["nohard"]))
    return FP.run_label(train, float(m["h"]))


def build_plan(root: str) -> Plan:
    plan = Plan(root)
    entries = sorted(os.listdir(root))
    known = plan_checkpoints(plan, entries)
    plan_data(plan, entries)
    plan_folder(plan, entries, "gt_cache", rel(FP.gt_dir()))
    for n in entries:
        if n.startswith("result_cache") and os.path.isdir(plan.path(n)):
            plan.move(n, rel(FP.rollout_dir(cache_label(n))))
    plan_results(plan, entries, known)
    plan_folder(plan, entries, "logs", rel(FP.log_dir()))
    for n in entries:
        if _SLURM.fullmatch(n) and os.path.isfile(plan.path(n)):
            plan.move(n, f"{rel(FP.slurm_dir())}/{n}")
    plan_archive(plan, entries)
    plan_leave(plan, entries)
    plan.steps[0:0] = folder_steps(plan)
    return plan


def show(plan: Plan) -> None:
    log(f"  {'status':8s} {'size':>10s}  source -> destination")
    absent = []
    for s in plan.steps:
        if s.status == "absent":
            absent.append(s.src)
            continue
        what = {"mkdir": f"make folder {s.src}/",
                "leave": s.src}.get(s.kind, f"{s.src} -> {s.dst}")
        log(f"  {s.status:8s} {human(s.size) if s.size else '':>10s}  {what}")
        if s.why:
            log(f"  {'':8s} {'':>10s}    {s.why}")
    if absent:
        log(f"  (not here, nothing to do: {', '.join(absent)})")
    for folder in plan.emptied:
        log(f"  {folder}/ will be empty afterwards; remove by hand if you "
            f"like")
    count: dict[str, int] = {}
    for s in plan.steps:
        count[s.status] = count.get(s.status, 0) + 1
    todo = [s for s in plan.steps if s.status == "todo" and s.kind != "mkdir"]
    totals = ", ".join(f"{v} {k}" for k, v in sorted(count.items()))
    log(f"  totals: {totals}; {len(todo)} item(s) to move, "
        f"{human(sum(s.size for s in todo))}")


class Clash(Exception):
    """A destination appeared after the plan was made."""


def swallows(plan: Plan, src: str, dst: str) -> bool:
    """True if src is a folder that dst sits inside on this filesystem, as
    results/ and Results/ are on Windows."""
    if not os.path.isdir(plan.path(src)):
        return False
    parts = dst.split("/")[:-1]
    return any(same_folder(plan.path("/".join(parts[:i])), plan.path(src))
               for i in range(1, len(parts) + 1)
               if os.path.exists(plan.path("/".join(parts[:i]))))


def mend_link(src: str, dst: str) -> None:
    """Re-point a relative symlink that has just moved, so it still lands on
    the same target."""
    if not os.path.islink(dst):
        return
    target = os.readlink(dst)
    if os.path.isabs(target):
        return
    there = os.path.join(os.path.dirname(src), target)
    fixed = os.path.relpath(there, os.path.dirname(dst))
    if fixed != target:
        tmp = dst + ".link-partial"
        os.symlink(fixed, tmp, target_is_directory=os.path.isdir(there))
        os.replace(tmp, dst)


def carry_out(plan: Plan, s: Step) -> None:
    """Do one move or rename; raises OSError, or Clash if dst has appeared."""
    src, dst = plan.path(s.src), plan.path(s.dst)
    if not os.path.lexists(src):
        raise FileNotFoundError(errno.ENOENT, "it is no longer there", src)
    s.mtime = os.lstat(src).st_mtime
    if s.kind == "recase":
        tmp = f"{dst}.case-{os.getpid()}"
        os.replace(src, tmp)
        try:
            os.replace(tmp, dst)
        except OSError:
            os.replace(tmp, src)
            raise
        return
    if os.path.lexists(dst):
        raise Clash(f"{s.dst} appeared; {s.src} stays")
    if swallows(plan, s.src, s.dst):
        hold = f"{src}.moving-{os.getpid()}"
        os.replace(src, hold)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(hold, dst)
        except OSError as e:
            try:
                os.replace(hold, src)
            except OSError:
                raise OSError(f"{e}; the folder was left at {hold}") from e
            raise
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        shutil.move(src, dst)
    mend_link(src, dst)


def apply_plan(plan: Plan, ran: list[Step]) -> None:
    """Run the todo steps in order, adding each attempted one to ``ran``."""
    todo = [s for s in plan.steps if s.status == "todo"]
    todo.sort(key=lambda s: s.deferred)
    dead: set[str] = set()
    for s in todo:
        s.when = datetime.now().isoformat(timespec="seconds")
        ran.append(s)
        try:
            if s.group and s.group in dead:
                raise OSError("skipped because the other half of its pair "
                              "failed")
            if s.kind == "mkdir":
                os.makedirs(plan.path(s.src), exist_ok=True)
            else:
                carry_out(plan, s)
            s.status = "done"
            if s.kind != "mkdir":
                log(f"  moved   {s.src} -> {s.dst}")
        except Clash as e:
            s.status, s.error = "clash", str(e)
            log(f"  clash   {e}")
        except OSError as e:
            s.status, s.error = "failed", f"{type(e).__name__}: {e}"
            if s.group:
                dead.add(s.group)
            log(f"  FAILED  {s.src} -> {s.dst}: {s.error}")


def write_log(plan: Plan, ran: list[Step], started: datetime) -> str:
    """Write the JSON log through a .partial file; returns its path."""
    folder = plan.path(rel(FP.log_dir()))
    os.makedirs(folder, exist_ok=True)
    stamp = f"{started:%Y%m%d-%H%M%S}"
    path = os.path.join(folder, f"migrate_layout_{stamp}.json")
    n = 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(folder, f"migrate_layout_{stamp}-{n}.json")
    moves = []
    for s in ran:
        if s.kind == "mkdir":
            continue
        ok = s.status == "done"
        moves.append({
            "src": s.src, "dst": s.dst, "bytes": s.size, "mtime": s.mtime,
            "mtime_text": (datetime.fromtimestamp(s.mtime).isoformat(
                timespec="seconds") if s.mtime else ""),
            "status": s.status, "time": s.when, "error": s.error,
            "undo": {"src": s.dst, "dst": s.src} if ok else None})
    doc = {"script": "migrate_layout.py", "root": plan.root,
           "started": started.isoformat(timespec="seconds"),
           "finished": datetime.now().isoformat(timespec="seconds"),
           "folders_made": [s.src for s in ran if s.kind == "mkdir"
                            and s.status == "done"],
           "moves": moves,
           "undo": "to undo, run each move's undo (src -> dst) from the "
                   "last entry back to the first"}
    partial = path + ".partial"
    with open(partial, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    os.replace(partial, path)
    return path


def save_log(plan: Plan, ran: list[Step], started: datetime) -> str:
    """Write the log if any move was tried; says so if it cannot."""
    if not any(s.kind != "mkdir" for s in ran):
        return ""
    try:
        return write_log(plan, ran, started)
    except OSError as e:
        log(f"WARNING: the log could not be written: {e}")
        return ""


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    ap = argparse.ArgumentParser(
        description="Move the old layout's loose outputs into Results/, "
                    "Work/ and Archive/ (a dry run unless --apply).")
    ap.add_argument("--root", default=".",
                    help="the project root to migrate (default: here)")
    ap.add_argument("--apply", action="store_true",
                    help="carry the plan out (default: only print it)")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        ap.error(f"--root {args.root!r} is not a folder")
    started = datetime.now()
    log(f"migrate_layout: {root}  "
        f"({'APPLY' if args.apply else 'dry run, nothing is changed'})")
    log("WARNING: no SLURM job may be queued or running while this applies "
        "(check with squeue -u $USER).")
    log("")
    plan = build_plan(root)
    log("")
    log("plan:")
    show(plan)
    log("")
    if not args.apply:
        log("nothing was changed; run with --apply to carry this out")
        return 0
    if not any(s.status == "todo" for s in plan.steps):
        log("nothing to do: the layout is already migrated.")
        return 0
    log("applying:")
    ran: list[Step] = []
    try:
        apply_plan(plan, ran)
    finally:
        wrote = save_log(plan, ran, started)
    for folder in plan.emptied:
        here = plan.path(folder)
        if os.path.isdir(here) and not os.listdir(here):
            log(f"  {folder}/ is now empty; remove by hand if you like")
    failed = [s for s in ran if s.status == "failed"]
    moved = [s for s in ran if s.status == "done" and s.kind != "mkdir"]
    log("")
    if wrote:
        log(f"log: {wrote}")
    left = [s for s in plan.steps if s.status in ("clash", "refused")]
    log(f"moved {len(moved)}, failed {len(failed)}, "
        f"left in place {len(left)} (clash or refused)")
    lost_log = not wrote and any(s.kind != "mkdir" for s in ran)
    return 1 if failed or lost_log else 0


if __name__ == "__main__":
    sys.exit(main())
