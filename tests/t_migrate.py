"""migrate_layout.py has to move the old layout without losing a byte.

Every tree here is synthetic and lives under SCRATCH: tiny checkpoints written
with torch.save, an npz with 2990 frames, a lean .npy and its sidecar, caches,
logs and result folders.  A dry run must change nothing; --apply must move
each item to its planned place with its bytes and mtime kept, delete nothing,
overwrite nothing, and do nothing the second time.  Each refusal gate gets a
tree of its own, with a control that must still move, so a gate that refuses
everything cannot pass.
"""
import collections
import contextlib
import errno
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime

import numpy as np
import torch
from numpy.lib import format as npformat

from _paths import ROOT, SCRATCH

import migrate_layout as M

SCRIPT = os.path.join(ROOT, "migrate_layout.py")
BASE = os.path.join(SCRATCH, "migrate")
shutil.rmtree(BASE, ignore_errors=True)
os.makedirs(BASE)
T0 = 1_760_000_000

ok = True
count = 0


def check(c, m):
    global ok, count
    count += 1
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


def P(root, rel):
    return os.path.join(root, *rel.split("/"))


def tree(name):
    root = os.path.join(BASE, name)
    os.makedirs(root)
    return root


def stamp(path, when):
    if when:
        os.utime(path, (T0 + when, T0 + when))


def put(root, rel, data: bytes | str = b"x", when=0):
    path = P(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data if isinstance(data, bytes) else data.encode("utf-8"))
    stamp(path, when)
    return path


def clone(root, src, dst, when=0):
    path = P(root, dst)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.copyfile(P(root, src), path)
    stamp(path, when)


def hist(n, base=0.0):
    return [{"epoch": i + 1, "train": base + i / 10, "val1": 0.5}
            for i in range(n)]


def put_json(root, rel, obj, when=0):
    put(root, rel, json.dumps(obj, indent=1), when)


def RM(t, tag=""):
    return {"dataset": "test.npz", "t_train": t, "val_t_start": 100.0,
            "val_t_end": 2000.0, "monitor_t_end": None, "tag": tag}


def best_of(epoch, score):
    return {"score": score, "survival": 1900.0, "epoch": epoch}


def ckpt(root, rel, kind="best", hard=True, zero_dc=True, meta=None,
         best=None, seed=0, history=None, when=0):
    """A tiny torch checkpoint shaped like the ones fno_train.py writes."""
    d = {"config": {"channels": 2, "modes": 2, "zero_dc": zero_dc,
                    "hard_mass_conservation": hard},
         "state_dict": collections.OrderedDict(w=torch.full((3,),
                                                            float(seed)))}
    if best is not None:
        d["best"], d["epoch"] = best, best["epoch"]
    if meta is not None:
        d["run_meta"] = meta
    if kind == "best":
        d["precision"] = "fp32"
    else:
        d.update(optimizer={"state": {}, "param_groups": [
            {"lr": 0.1, "params": [0]}]}, scheduler={"last_epoch": 4},
            history=history or [])
    path = P(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(d, path)
    stamp(path, when)


def npz(root, rel, frames=2990, seed=0, when=0):
    path = P(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    np.savez(path, train_X=rng.random((2, frames, 2, 2), dtype=np.float32),
             val_X=rng.random((1, 5, 2, 2), dtype=np.float32))
    stamp(path, when)


def lean(root, stem, seed=0, when=0):
    path = P(root, stem + ".npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.random.default_rng(seed).random((3, 4)))
    stamp(path, when)
    put_json(root, stem + ".json", {"format": "ch_lean", "frames": 4}, when)


def manifest(root, folder, model, ref=None, start=100.0, end=2000.0,
             mixed=False):
    run = {"model_path": model,
           "window": {"start": start, "end": end, "snapshots": [start, end]}}
    if ref is not None:
        st = os.stat(P(root, ref))
        run["model_bytes"] = st.st_size
        run["model_mtime"] = datetime.fromtimestamp(st.st_mtime).isoformat(
            timespec="seconds")
    put_json(root, f"{folder}/manifest.json",
             {"last_run": run, "compositions": {"+0.00": {"run": run}},
              "checkpoints": {"mixed": mixed}})


def listing(root):
    """Every path with its size and mtime, folders included."""
    out = {}
    for here, dirs, files in os.walk(root):
        for n in dirs + files:
            p = os.path.join(here, n)
            st = os.lstat(p)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = (
                os.path.isdir(p), 0 if os.path.isdir(p) else st.st_size,
                st.st_mtime_ns)
    return out


def digests(root, skip_logs=True):
    """{path: (sha256, mtime_ns, bytes)} of every file."""
    out = {}
    for here, _, files in os.walk(root):
        for n in files:
            p = os.path.join(here, n)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            if skip_logs and re.fullmatch(
                    r"Work/logs/migrate_layout_.*\.json", rel):
                continue
            with open(p, "rb") as fh:
                data = fh.read()
            out[rel] = (hashlib.sha256(data).hexdigest(),
                        os.stat(p).st_mtime_ns, len(data))
    return out


def logs_of(root):
    found = []
    for here, _, files in os.walk(root):
        found += [os.path.join(here, n) for n in files
                  if n.startswith("migrate_layout_")]
    return sorted(found)


def run(root, *flags):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = M.main(["--root", root, *flags])
    return rc, buf.getvalue()


def steps_of(root):
    return {s.src: s for s in M.build_plan(root).steps if s.kind != "mkdir"}


def case_insensitive(folder):
    probe = os.path.join(folder, "CaseProbe.tmp")
    open(probe, "w").close()
    try:
        return os.path.exists(os.path.join(folder, "caseprobe.tmp"))
    finally:
        os.remove(probe)


CI = case_insensitive(BASE)


# ---------------------------------------------------------------- reader
print("1. reading a checkpoint without torch")
r1 = tree("reader")
best = best_of(35, 0.91)
ckpt(r1, "full.pt", kind="last", meta=RM(500.0, "x"), best=best, seed=1,
     history=hist(3))
got = M.read_checkpoint(P(r1, "full.pt"))
check(got is not None and got["config"]["zero_dc"] is True
      and got["config"]["hard_mass_conservation"] is True,
      "the config comes back")
check(got is not None and got["run_meta"] == RM(500.0, "x"),
      "run_meta comes back")
check(got is not None and got["best"] == best and got["epoch"] == 35,
      "best and epoch come back")
check(got is not None and got["history"] == hist(3),
      "the history comes back")
check(got is not None and not any(isinstance(v, torch.Tensor)
                                  for v in got["state_dict"].values()),
      "tensors are stubs, not tensors")
put(r1, "text.pt", "not a zip at all")
put(r1, "empty.pt", b"")
with open(P(r1, "full.pt"), "rb") as fh:
    cut = fh.read()[:200]
put(r1, "cut.pt", cut)
with zipfile.ZipFile(P(r1, "nopkl.pt"), "w") as zf:
    zf.writestr("a/b.txt", "hello")
check(all(M.read_checkpoint(P(r1, n)) is None
          for n in ("text.pt", "empty.pt", "cut.pt", "nopkl.pt")),
      "a non-zip, empty, cut-off or pickle-less file means no metadata")
check(M.read_checkpoint(P(r1, "missing.pt")) is None,
      "a missing file means no metadata")

marker, marker2 = P(r1, "pwned.txt"), P(r1, "pwned_dir")


class Evil:
    def __reduce__(self):
        return exec, (f"open({marker!r}, 'w').close()",)


class Evil2:
    def __reduce__(self):
        return os.makedirs, (marker2,)


with zipfile.ZipFile(P(r1, "evil.pt"), "w") as zf:
    zf.writestr("evil/data.pkl", pickle.dumps((Evil(), Evil2())))
check(M.read_checkpoint(P(r1, "evil.pt")) is None
      and not os.path.exists(marker) and not os.path.exists(marker2),
      "a pickle that calls exec() or os.makedirs() is not run")
src = open(SCRIPT, encoding="utf-8").read()
check(not re.search(r"^\s*(import|from)\s+torch", src, re.M),
      "the script never imports torch")
code = (f"import sys; sys.path.insert(0, {ROOT!r})\n"
        "import migrate_layout as M\n"
        f"d = M.read_checkpoint({P(r1, 'full.pt')!r})\n"
        f"M.build_plan({r1!r})\n"
        "print('keys', sorted(d or {}))\n"
        "print('torch loaded:', 'torch' in sys.modules)\n")
p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                   text=True, timeout=120)
check("'config'" in p.stdout and "'run_meta'" in p.stdout,
      "a fresh process read the metadata")
check(p.returncode == 0 and "torch loaded: False" in p.stdout,
      f"and torch is not in sys.modules ({p.stdout.strip()[-24:]!r})")
check(not re.search(r"os\.(remove|unlink|rmdir)\(|rmtree", src),
      "the script has no call that deletes anything")

print("\n2. the npz gate reads a header, never the data")
d2 = tree("npz")
npz(d2, "good.npz", 2990, seed=1)
npz(d2, "short.npz", 1990, seed=2)
okay, why = M.npz_gate(P(d2, "good.npz"))
check(okay and "2990 frames" in why and "t = 300" in why,
      f"2990 frames read as t = 300 ({why})")
okay, why = M.npz_gate(P(d2, "short.npz"))
check(okay and "t = 200" in why, f"1990 frames read as t = 200 ({why})")
buf = io.BytesIO()
npformat.write_array_header_1_0(buf, {"descr": "<f4", "fortran_order": False,
                                      "shape": (2, 2990, 2, 2)})
with zipfile.ZipFile(P(d2, "cut.npz"), "w") as zf:
    zf.writestr("train_X.npy", buf.getvalue() + b"\0\0\0\0")
okay, why = M.npz_gate(P(d2, "cut.npz"))
check(okay and "2990 frames" in why,
      "a member whose data is cut off still passes: only the header is read")
with zipfile.ZipFile(P(d2, "nox.npz"), "w") as zf:
    zf.writestr("val_X.npy", b"x")
put(d2, "junk.npz", "this is not a zip")
check(not M.npz_gate(P(d2, "nox.npz"))[0]
      and not M.npz_gate(P(d2, "junk.npz"))[0]
      and not M.npz_gate(P(d2, "absent.npz"))[0],
      "no train_X, not a zip and missing all fail the gate")


# ------------------------------------------------------------ main tree
FILES = {
    "fno_ch.pt": "Work/models/train_t0500/fno_ch.pt",
    "fno_ch_500.pt": "Archive/duplicates/fno_ch_500.pt",
    "fno_ch_last.pt": "Work/models/train_t0500/fno_ch_last.pt",
    "fno_ch_history.json": "Work/models/train_t0500/fno_ch_history.json",
    "fno_ch_300.pt": "Work/models/train_t0300/fno_ch.pt",
    "fno_ch_last_300.pt": "Work/models/train_t0300/fno_ch_last.pt",
    "fno_ch_T1000.pt": "Work/models/train_t1000/fno_ch.pt",
    "fno_ch_T1000_last.pt": "Work/models/train_t1000/fno_ch_last.pt",
    "fno_ch_T1000_history.json":
        "Work/models/train_t1000/fno_ch_history.json",
    "fno_ch_T1000_nohard.pt": "Work/models/train_t1000_nohard/fno_ch.pt",
    "fno_ch_mine.pt": "Work/models/train_t0250_mine/fno_ch.pt",
    "test.npz": "Work/data/test.npz",
    "test_numpy.npz": "Work/data/test_numpy.npz",
    "data/lean_a.npy": "Work/data/lean_a.npy",
    "data/lean_a.json": "Work/data/lean_a.json",
    "data/extra.npz": "Work/data/extra.npz",
    "data/copy.npz": "Archive/duplicates/copy.npz",
    "data/notes.txt": "Work/data/notes.txt",
    "ch_gpu_123.out": "Work/slurm/ch_gpu_123.out",
    "ch_gpu_123.err": "Work/slurm/ch_gpu_123.err",
    "fno_train_77.out": "Work/slurm/fno_train_77.out",
    "Submission.zip": "Archive/Submission.zip",
}
FOLDERS = {
    "gt_cache": "Work/gt_cache",
    "logs": "Work/logs",
    "Old": "Archive/Old",
    "archive": "Archive",
    "results_runs": "Results/results_runs_2026-09-30",
    "results_T1000_P20000": "Results/train_t1000_pred_t20000",
    "results": "Results/train_t0500_pred_t02000",
    "results_S200_P4000": "Results/train_t0300_pred_t00200-04000",
    "results_T500_P20000_nohard":
        "Results/train_t0500_nohard_pred_t20000",
    "results_nohard": "Results/train_t1000_nohard_pred_t02000",
    "result_cache_50": "Work/rollout_cache/result_cache_50",
    "result_cache_T1000_P20000":
        "Work/rollout_cache/train_t1000_pred_t20000",
    "result_cache_T1000_nohard_P20000":
        "Work/rollout_cache/train_t1000_nohard_pred_t20000",
    "Submission_final": "Archive/Submission_final",
    "results_flat_pre_folders": "Archive/results_flat_pre_folders",
}
STAY = ("fno_train.py", "fno_predict.py", "ch_gpu.sh", "README.md",
        "progress.md", "pyrightconfig.json", "env/pyvenv.cfg",
        "tests/t_x.py", "__pycache__/x.pyc",
        "notes.txt", ".hidden")
README = "# Analysis\r\nR\u00b2 = 0.94 \u2014 written by hand\r\n".encode()


def build_main(root):
    h500, h300, h1000 = hist(3), hist(4, 1.0), hist(5, 2.0)
    b500, b300 = best_of(35, 0.91), best_of(55, 0.93)
    b1000 = best_of(7, 0.88)
    ckpt(root, "fno_ch.pt", best=b500, seed=1, when=10)
    clone(root, "fno_ch.pt", "fno_ch_500.pt", when=500)
    ckpt(root, "fno_ch_last.pt", "last", best=b500, history=h500, seed=1,
         when=20)
    put_json(root, "fno_ch_history.json", h500, when=21)
    ckpt(root, "fno_ch_300.pt", best=b300, seed=2, when=30)
    ckpt(root, "fno_ch_last_300.pt", "last", best=b300, history=h300,
         seed=2, when=31)
    ckpt(root, "fno_ch_T1000.pt", best=b1000, meta=RM(1000.0, "T1000"),
         seed=3, when=40)
    ckpt(root, "fno_ch_T1000_last.pt", "last", best=b1000, history=h1000,
         meta=RM(1000.0, "T1000"), seed=3, when=41)
    put_json(root, "fno_ch_T1000_history.json", h1000, when=42)
    ckpt(root, "fno_ch_T1000_nohard.pt", hard=False, best=best_of(9, 0.77),
         meta=RM(1000.0, "T1000"), seed=4, when=43)
    ckpt(root, "fno_ch_mine.pt", best=best_of(3, 0.5),
         meta=RM(250.0, "mine"), seed=5, when=44)
    npz(root, "test.npz", 2990, seed=1, when=100)
    npz(root, "test_numpy.npz", 2990, seed=2, when=101)
    lean(root, "data/lean_a", seed=3, when=102)
    npz(root, "data/extra.npz", 1990, seed=4, when=103)
    clone(root, "test.npz", "data/copy.npz", when=900)
    put(root, "data/notes.txt", "d", when=104)
    put(root, "gt_cache/gt_a.npz", b"g1", when=110)
    put(root, "gt_cache/gt_b.npz", b"g22", when=111)
    put(root, "logs/train.log", "log\n", when=120)
    put(root, "ch_gpu_123.out", "out", when=130)
    put(root, "ch_gpu_123.err", "err", when=131)
    put(root, "fno_train_77.out", "o", when=132)
    put(root, "Old/old_notes.txt", "old", when=140)
    put(root, "Old/sub/old.pt", b"\1\2\3", when=141)
    put(root, "archive/a.py", "print(1)\n", when=142)
    put(root, "archive/chfno/x.py", "print(2)\n", when=143)
    put(root, "results_runs/README.md", README, when=150)
    put(root, "results_runs/A/manifest.json", "{}", when=151)
    put(root, "results_runs/comparison.png", bytes(range(256)), when=152)
    put(root, "results_T1000_P20000/fig.png", b"f1", when=160)
    put(root, "results/fig.png", b"f2", when=161)
    manifest(root, "results", "fno_ch.pt", ref="fno_ch.pt")
    put(root, "results_S200_P4000/fig.png", b"f3", when=162)
    manifest(root, "results_S200_P4000", "E:\\work\\fno_ch_300.pt")
    put(root, "results_T500_P20000_nohard/fig.png", b"f4", when=163)
    put(root, "results_nohard/fig.png", b"f5", when=164)
    manifest(root, "results_nohard", "fno_ch_T1000_nohard.pt")
    put(root, "result_cache_50/s1.npz", b"c1", when=170)
    put(root, "result_cache_T1000_P20000/s2.npz", b"c2", when=171)
    put(root, "result_cache_T1000_nohard_P20000/s3.npz", b"c3", when=172)
    put(root, "Submission_final/fig.png", b"s", when=180)
    put(root, "Submission.zip", b"PK", when=181)
    put(root, "results_flat_pre_folders/old.png", b"o", when=182)
    for n in STAY:
        put(root, n, n, when=190)


def expected(rel):
    if rel in FILES:
        return FILES[rel]
    top = rel.split("/")[0]
    return FOLDERS[top] + rel[len(top):] if top in FOLDERS else rel


print("\n3. a dry run of the whole project changes nothing")
main = tree("main")
build_main(main)
before_list, before_dig = listing(main), digests(main)
rc, out = run(main)
check(rc == 0, "the dry run exits 0")
check(listing(main) == before_list,
      "every path, size and mtime is as it was")
check(not any(os.path.exists(P(main, n))
              for n in ("Work", "Archive/duplicates", "Archive/Old")),
      "no Work/, duplicates/ or Old/ folder was made")
check(not logs_of(main), "no log was written")
lines = out.splitlines()
check(any("SLURM" in ln for ln in lines[:3]),
      "a warning about SLURM jobs is at the top")
check(lines[-1] == "nothing was changed; run with --apply to carry this out",
      "it ends by saying nothing was changed")
check(all(f"{s} -> {d}" in out for s, d in FILES.items()),
      "the table shows every file's source -> destination")
check("status" in out and "size" in out and "totals:" in out,
      "the table has a header and totals")
check("2990 frames = horizon t = 300" in out,
      "the dataset's frame count and horizon are in the reason")
check("byte-identical to fno_ch.pt, which has the earliest mtime" in out,
      "the duplicate's reason names the file that stays")

print("\n4. the plan")
S = steps_of(main)
bad = [s for s, d in FILES.items()
       if S[s].status != "todo" or S[s].dst != d]
check(not bad, f"every file is todo to its planned place {bad[:2]}")
bad = [s for s, d in FOLDERS.items() if s != "archive"
       and (S[s].status != "todo" or S[s].dst != d)]
check(not bad, f"every folder is todo to its planned place {bad[:2]}")
check(S["fno_ch.pt"].dst == "Work/models/train_t0500/fno_ch.pt"
      and S["fno_ch_500.pt"].dst == "Archive/duplicates/fno_ch_500.pt",
      "fno_ch.pt, byte-identical to fno_ch_500.pt, is the one that stays")
check(S["fno_ch_history.json"].dst.startswith("Work/models/train_t0500/")
      and S["fno_ch_last.pt"].dst.startswith("Work/models/train_t0500/"),
      "the state and the history follow their run to train_t0500")
check(S["data/copy.npz"].dst == "Archive/duplicates/copy.npz"
      and S["test.npz"].dst == "Work/data/test.npz",
      "of two identical datasets the later one is the duplicate")
check(not [n for n in STAY if n in S and S[n].status == "todo"],
      "code, notes and the new folders are not in the plan as moves")
unknown = [s for s in M.build_plan(main).steps
           if s.kind == "leave" and s.src == "notes.txt"]
check(len(unknown) == 1 and "not recognised" in unknown[0].why,
      "an unknown file at the root is leave (not recognised)")
mk = [s for s in M.build_plan(main).steps if s.kind == "mkdir"]
check(len(mk) == 8 and mk[0].src == "Work/data"
      and {s.src for s in mk} >= {"Work/models", "Results", "Archive",
                                  "Work/slurm", "Work/rollout_cache"},
      "step 1 lists the eight folders first")

print("\n5. --apply moves everything and keeps bytes and times")
plan_before = [s for s in M.build_plan(main).steps
               if s.status == "todo" and s.kind in ("move", "recase")]
rc, out = run(main, "--apply")
check(rc == 0, "apply exits 0")
after_dig = digests(main)
lost = [r for r, v in before_dig.items()
        if expected(r) not in after_dig or after_dig[expected(r)] != v]
check(not lost, f"every file arrived with its sha256 and mtime {lost[:2]}")
left = [r for r in before_dig if expected(r) != r
        and os.path.normcase(expected(r)) != os.path.normcase(r)
        and os.path.exists(P(main, r))]
check(not left, f"and no source is left behind {left[:2]}")
check(len(after_dig) == len(before_dig)
      and sum(v[2] for v in after_dig.values())
      == sum(v[2] for v in before_dig.values()),
      "no file was deleted: same count and total bytes")
check(all(after_dig[n][0] == before_dig[n][0] for n in STAY),
      "code, notes and the unknown file stayed")
check(after_dig["Results/results_runs_2026-09-30/README.md"][0]
      == hashlib.sha256(README).hexdigest(),
      "results_runs/README.md arrived byte-identical")
check(after_dig["Work/models/train_t0500/fno_ch.pt"][1]
      == before_dig["fno_ch.pt"][1]
      and after_dig["Archive/duplicates/fno_ch_500.pt"][1]
      == before_dig["fno_ch_500.pt"][1],
      "the copy with the earliest mtime kept its place and its time")
names = os.listdir(main)
check("Archive" in names and ("archive" not in names or not CI),
      "archive/ became Archive/ "
      + ("(the same folder here: only its capital changed)" if CI
         else "(its entries moved)"))
check("Results" in names and "results" not in names
      and all(os.path.isdir(P(main, "Results/" + os.path.basename(d)))
              for d in FOLDERS.values() if d.startswith("Results/")),
      "results/ made way for Results/ and every run folder is inside it")
if CI:
    check(os.path.isfile(P(main, "Archive/chfno/x.py")),
          "on this filesystem archive/ kept its contents in place")
else:
    check(os.path.isdir(P(main, "archive"))
          and not os.listdir(P(main, "archive")),
          "archive/ is left empty")
check(os.path.isdir(P(main, "data")) and not os.listdir(P(main, "data"))
      and "data/ is now empty; remove by hand if you like" in out,
      "the emptied data/ is left and reported")
check(os.path.isfile(P(main, "Work/logs/train.log"))
      and not os.path.exists(P(main, "logs")),
      "logs/ became Work/logs/ (made by the move, not beforehand)")

print("\n6. the JSON log")
files = logs_of(main)
check(len(files) == 1 and re.fullmatch(
    r"migrate_layout_\d{8}-\d{6}\.json", os.path.basename(files[0]))
    and os.path.dirname(files[0]) == P(main, "Work/logs"),
    f"one log named migrate_layout_<time>.json in Work/logs {files}")
doc = json.load(open(files[0], encoding="utf-8"))
logged = {(e["src"], e["dst"]) for e in doc["moves"]}
planned = {(s.src, s.dst) for s in plan_before}
check(logged == planned and len(doc["moves"]) == len(planned),
      f"it lists every executed move ({len(planned)})")
check(all(e["status"] == "done" and e["undo"] == {"src": e["dst"],
                                                  "dst": e["src"]}
          for e in doc["moves"]),
      "each entry is done and carries the reverse move")
e = next(e for e in doc["moves"] if e["src"] == "fno_ch.pt")
check(e["bytes"] == before_dig["fno_ch.pt"][2]
      and e["mtime"] == T0 + 10 and e["time"],
      "an entry records bytes, mtime and the time it ran")
check(not [f for f in os.listdir(P(main, "Work/logs"))
           if f.endswith(".partial")],
      "no .partial file is left behind")

print("\n7. running --apply again does nothing")
mid = listing(main)
rc, out = run(main, "--apply")
check(rc == 0 and "nothing to do" in out, "the second apply says so")
check(listing(main) == mid, "and the tree is exactly as it was")
check(len(logs_of(main)) == 1, "and writes no second log")
statuses = {s.status for s in M.build_plan(main).steps}
check(statuses <= {"done", "absent", "leave"},
      f"every step is now done, absent or leave {sorted(statuses)}")
S2 = steps_of(main)
check(all(S2[n].status == "done" for n in
          ("Old", "gt_cache", "logs", "results_runs", "Submission.zip",
           "results_flat_pre_folders")),
      "and what moved is reported done (destination there, source gone)")


# ----------------------------------------------------------- the undo
print("\n8. the undo hints put everything back")
und = tree("undo")
ckpt(und, "fno_ch_T100.pt", best=best_of(1, 0.5), seed=1, when=10)
npz(und, "test.npz", 2990, seed=1, when=20)
put(und, "gt_cache/gt_a.npz", b"g", when=30)
put(und, "Old/o.txt", "o", when=40)
put(und, "archive/a.py", "a", when=50)
put(und, "results_runs/README.md", README, when=60)
put(und, "results_T1000_P20000/f.png", b"f", when=70)
orig = digests(und)
rc, out = run(und, "--apply")
entries = json.load(open(logs_of(und)[0], encoding="utf-8"))["moves"]
for e in reversed(entries):
    # archive/a.py -> Archive/a.py is one file on Windows, where there is
    # nothing to undo, and a real move where case matters (Linux)
    if os.path.normcase(e["src"]) == os.path.normcase(e["dst"]):
        continue
    os.makedirs(os.path.dirname(P(und, e["undo"]["dst"])), exist_ok=True)
    os.replace(P(und, e["undo"]["src"]), P(und, e["undo"]["dst"]))
back = {os.path.normcase(k): v for k, v in digests(und).items()}
check(rc == 0 and all(back.get(os.path.normcase(r)) == v
                      for r, v in orig.items()),
      "undoing each logged move, last first, restores every file and time")


# ------------------------------------------------------------ subprocess
print("\n9. as a command, with the default --root")
sub = tree("sub")
ckpt(sub, "fno_ch_T100.pt", best=best_of(1, 0.5), seed=1, when=10)
put(sub, "Old/o.txt", "o", when=20)
sub_list = listing(sub)
p = subprocess.run([sys.executable, SCRIPT], cwd=sub, capture_output=True,
                   text=True, timeout=120)
check(p.returncode == 0 and listing(sub) == sub_list
      and p.stdout.strip().endswith("run with --apply to carry this out"),
      "python migrate_layout.py with no flags only prints the plan")
p = subprocess.run([sys.executable, SCRIPT, "--apply"], cwd=sub,
                   capture_output=True, text=True, timeout=120)
check(p.returncode == 0
      and os.path.isfile(P(sub, "Work/models/train_t0100/fno_ch.pt"))
      and os.path.isfile(P(sub, "Archive/Old/o.txt"))
      and not os.path.exists(P(sub, "fno_ch_T100.pt")),
      "python migrate_layout.py --apply moves them, from the current folder")
p = subprocess.run([sys.executable, SCRIPT, "--root",
                    os.path.join(BASE, "nowhere")], capture_output=True,
                   text=True, timeout=120)
check(p.returncode == 2 and not os.path.exists(os.path.join(BASE, "nowhere")),
      "a --root that is not a folder is an error and makes nothing")


# ------------------------------------------------------------ resolution
print("\n10. how the training horizon is found")
rl = tree("rules")
bA, bB, bC = best_of(9, 0.66), best_of(4, 0.41), best_of(12, 0.7)
ckpt(rl, "fno_ch_T1500.pt", best=best_of(2, 0.2), seed=1, when=10)
clone(rl, "fno_ch_T1500.pt", "fno_ch_copy.pt", when=20)
ckpt(rl, "fno_ch_alpha.pt", best=bB, meta=RM(420.5, "alpha"), seed=2,
     when=30)
ckpt(rl, "fno_ch_T2000.pt", best=bA, seed=3, when=40)
ckpt(rl, "fno_ch_beta_last.pt", "last", best=bA, history=hist(4), seed=3,
     when=41)
put_json(rl, "fno_ch_beta_history.json", hist(4), when=42)
ckpt(rl, "fno_ch_gamma.pt", best=bC, meta=RM(2600.0, "gamma"), seed=4,
     when=50)
put_json(rl, "fno_ch_gamma_history.json", hist(2, 5.0), when=51)
ckpt(rl, "fno_ch_nohard.pt", hard=False, best=best_of(6, 0.3),
     meta=RM(900.0), seed=5, when=60)
ckpt(rl, "fno_ch_nohard_last.pt", "last", hard=False, best=best_of(6, 0.3),
     history=hist(2, 9.0), seed=5, when=61)
put_json(rl, "fno_ch_nohard_history.json", hist(2, 9.0), when=62)
ckpt(rl, "fno_ch_last_450.pt", "last", best=bA, seed=6, when=70)
put_json(rl, "fno_ch_history_450.json", hist(1), when=71)
S = steps_of(rl)
want = {
    "fno_ch_T1500.pt": "Work/models/train_t1500/fno_ch.pt",
    "fno_ch_copy.pt": "Archive/duplicates/fno_ch_copy.pt",
    "fno_ch_alpha.pt": "Work/models/train_t0420p5_alpha/fno_ch.pt",
    "fno_ch_T2000.pt": "Work/models/train_t2000/fno_ch.pt",
    "fno_ch_beta_last.pt": "Work/models/train_t2000/fno_ch_last.pt",
    "fno_ch_beta_history.json": "Work/models/train_t2000/fno_ch_history.json",
    "fno_ch_gamma.pt": "Work/models/train_t2600_gamma/fno_ch.pt",
    "fno_ch_gamma_history.json":
        "Work/models/train_t2600_gamma/fno_ch_history.json",
    "fno_ch_nohard.pt": "Work/models/train_t0900_nohard/fno_ch.pt",
    "fno_ch_nohard_last.pt": "Work/models/train_t0900_nohard/fno_ch_last.pt",
    "fno_ch_nohard_history.json":
        "Work/models/train_t0900_nohard/fno_ch_history.json",
    "fno_ch_last_450.pt": "Work/models/train_t0450/fno_ch_last.pt",
    "fno_ch_history_450.json": "Work/models/train_t0450/fno_ch_history.json",
}
for src_name, dst_name in want.items():
    check(S[src_name].status == "todo" and S[src_name].dst == dst_name,
          f"{src_name} -> {dst_name}")
hows = {s.src: s.why for s in M.build_plan(rl).steps if s.kind == "move"}
check("run_meta" in hows["fno_ch_alpha.pt"]
      and "from its name" in hows["fno_ch_T2000.pt"]
      and "same best epoch" in hows["fno_ch_beta_last.pt"]
      and "history is that of fno_ch_beta_last.pt"
      in hows["fno_ch_beta_history.json"]
      and "history of fno_ch_gamma.pt" in hows["fno_ch_gamma_history.json"],
      "each reason says how its horizon was found: name, run_meta, best "
      "epoch, history, same-stem best")
rc, out = run(rl, "--apply")
check(rc == 0 and all(os.path.isfile(P(rl, d)) and
                      not os.path.exists(P(rl, s))
                      for s, d in want.items()),
      "applied, they all arrive")


# -------------------------------------------------------------- refusals
print("\n11. a name that contradicts run_meta is refused")
rf = tree("refuse_t")
ckpt(rf, "fno_ch_T700.pt", meta=RM(1000.0, "T700"), best=best_of(1, .1),
     seed=1, when=10)
ckpt(rf, "fno_ch_T800.pt", meta=RM(800.0, "T800"), best=best_of(2, .2),
     seed=2, when=11)
S = steps_of(rf)
check(S["fno_ch_T700.pt"].status == "refused"
      and "700" in S["fno_ch_T700.pt"].why
      and "1000" in S["fno_ch_T700.pt"].why,
      f"T700 vs run_meta 1000: {S['fno_ch_T700.pt'].why}")
check(S["fno_ch_T800.pt"].status == "todo",
      "its consistent neighbour still moves")
rc, out = run(rf, "--apply")
check(rc == 0 and os.path.isfile(P(rf, "fno_ch_T700.pt"))
      and not os.path.exists(P(rf, "Work/models/train_t0700"))
      and not os.path.exists(P(rf, "Work/models/train_t1000"))
      and os.path.isfile(P(rf, "Work/models/train_t0800/fno_ch.pt")),
      "applied, the refused file stays and exits 0")

print("\n12. a _nohard that disagrees with the config is refused")
rn = tree("refuse_nohard")
ckpt(rn, "fno_ch_T300_nohard.pt", hard=True, best=best_of(1, .1), seed=1,
     when=10)
ckpt(rn, "fno_ch_T301.pt", zero_dc=False, best=best_of(2, .2), seed=2,
     when=11)
ckpt(rn, "fno_ch_T302_nohard.pt", hard=False, best=best_of(3, .3), seed=3,
     when=12)
S = steps_of(rn)
check(S["fno_ch_T300_nohard.pt"].status == "refused"
      and "_nohard" in S["fno_ch_T300_nohard.pt"].why,
      "_nohard in the name, hard config: refused")
check(S["fno_ch_T301.pt"].status == "refused",
      "no _nohard in the name, zero_dc False in the config: refused")
check(S["fno_ch_T302_nohard.pt"].status == "todo"
      and S["fno_ch_T302_nohard.pt"].dst
      == "Work/models/train_t0302_nohard/fno_ch.pt",
      "the agreeing _nohard file goes to a _nohard folder")

print("\n13. a horizon nobody can vouch for is refused")
ru = tree("refuse_unknown")
ckpt(ru, "fno_ch_mystery.pt", best=best_of(1, .1), seed=1, when=10)
ckpt(ru, "fno_ch_last.pt", "last", best=best_of(8, .8), history=hist(2),
     seed=2, when=11)
put_json(ru, "fno_ch_history.json", hist(2, 3.0), when=12)
ckpt(ru, "fno_ch_T50.pt", best=best_of(5, .5), seed=3, when=13)
S = steps_of(ru)
msg = "cannot tell its training horizon; move it by hand"
check(all(S[n].status == "refused" and S[n].why == msg
          for n in ("fno_ch_mystery.pt", "fno_ch_last.pt",
                    "fno_ch_history.json")),
      "a tagless best, a state and a history with no T all say so")
check(S["fno_ch_T50.pt"].status == "todo", "a file with a name T moves")
rc, out = run(ru, "--apply")
check(rc == 0 and all(os.path.isfile(P(ru, n)) for n in
                      ("fno_ch_mystery.pt", "fno_ch_last.pt",
                       "fno_ch_history.json"))
      and msg in out, "applied, they stay and the message is printed")

print("\n14. a bad tag is refused")
rt = tree("refuse_tag")
ckpt(rt, "fno_ch_bad.pt", meta=RM(100.0, "bad_tag"), best=best_of(1, .1),
     seed=1, when=10)
ckpt(rt, "fno_ch_ok.pt", meta=RM(100.0, "good.tag-1"), best=best_of(2, .2),
     seed=2, when=11)
S = steps_of(rt)
check(S["fno_ch_bad.pt"].status == "refused" and "--tag" in S[
    "fno_ch_bad.pt"].why, "run_meta.tag 'bad_tag' fails check_tag")
check(S["fno_ch_ok.pt"].dst == "Work/models/train_t0100_good.tag-1/fno_ch.pt",
      "a valid run_meta tag becomes the label's tag")

print("\n15. datasets: an .npy without its sidecar, and the npz gate")
rd = tree("refuse_data")
lean(rd, "data/paired", seed=1, when=10)
put(rd, "data/lonely.npy", b"\x93NUMPY", when=11)
put(rd, "test.npz", "not a zip", when=12)
with zipfile.ZipFile(P(rd, "data/nox.npz"), "w") as zf:
    zf.writestr("val_X.npy", b"x")
npz(rd, "test_numpy.npz", 2990, seed=2, when=13)
S = steps_of(rd)
check(S["data/lonely.npy"].status == "refused"
      and "sidecar" in S["data/lonely.npy"].why,
      "an .npy without its .json sidecar is refused")
check(S["data/paired.npy"].status == "todo"
      and S["data/paired.json"].status == "todo"
      and S["data/paired.json"].dst == "Work/data/paired.json",
      "an .npy with its sidecar moves as a pair")
check(S["test.npz"].status == "refused" and S["data/nox.npz"].status
      == "refused", "an npz that is no zip, or has no train_X, is refused")
check(S["test_numpy.npz"].status == "todo", "a good npz moves")
rc, out = run(rd, "--apply")
check(rc == 0 and os.path.isfile(P(rd, "data/lonely.npy"))
      and os.path.isfile(P(rd, "Work/data/paired.npy"))
      and os.path.isfile(P(rd, "test.npz")),
      "applied, the refused datasets stay where they were")

print("\n16. results folders whose model cannot be told are refused")
rr = tree("refuse_results")
ckpt(rr, "fno_ch.pt", best=best_of(1, .1), seed=1, when=10)
clone(rr, "fno_ch.pt", "fno_ch_100.pt", when=500)
put(rr, "results/fig.png", b"a", when=20)
put(rr, "results_P100/fig.png", b"b", when=21)
manifest(rr, "results_P100", "ghost.pt")
put(rr, "results_P200/fig.png", b"c", when=22)
manifest(rr, "results_P200", "fno_ch.pt", ref="fno_ch.pt")
os.utime(P(rr, "fno_ch.pt"), (T0 + 999, T0 + 999))
put(rr, "results_P300/fig.png", b"d", when=23)
manifest(rr, "results_P300", "fno_ch_100.pt", mixed=True)
put(rr, "results_P400/fig.png", b"e", when=24)
put(rr, "results_P400/manifest.json", "{ not json", when=24)
put(rr, "results_weird_name/fig.png", b"f", when=25)
S = steps_of(rr)
check(S["results"].status == "refused"
      and "cannot tell which model made it" in S["results"].why,
      "results/ with no manifest: cannot tell which model made it")
check(S["results_P100"].status == "refused"
      and "ghost.pt" in S["results_P100"].why,
      "a manifest naming a checkpoint that is not here is refused")
check(S["results_P200"].status == "refused"
      and "not the one that made it" in S["results_P200"].why,
      "a manifest whose size/time no longer fit the file is refused")
check(S["results_P300"].status == "refused"
      and S["results_P400"].status == "refused",
      "a mixed manifest and an unreadable one are refused")
check(S["results_weird_name"].status == "refused",
      "a name that is no known results pattern is refused")
rc, out = run(rr, "--apply")
check(rc == 0 and all(os.path.isfile(P(rr, f"{n}/fig.png")) for n in
                      ("results", "results_P100", "results_P200",
                       "results_P300", "results_P400",
                       "results_weird_name")),
      "applied, none of them moved")

print("\n17. a results/ folder is the same folder as Results/ on Windows")
ra = tree("alias")
put(ra, "results/fig.png", b"a", when=10)
put(ra, "results_runs/README.md", README, when=11)
S = steps_of(ra)
if CI:
    check(S["results"].status == "refused"
          and S["results_runs"].status == "refused"
          and "same folder" in S["results_runs"].why,
          "an unmoved results/ blocks the move into Results/")
    rc, out = run(ra, "--apply")
    check(os.path.isfile(P(ra, "results_runs/README.md"))
          and not os.path.exists(P(ra, "results/results_runs_2026-09-30")),
          "so results_runs is not buried inside it")
else:
    check(S["results"].status == "refused"
          and S["results_runs"].status == "todo",
          "on a case-sensitive filesystem results_runs is not blocked")
    print("SKIP  the Results/results alias needs a case-insensitive disk")

# --------------------------------------------------------------- clashes
print("\n18. a clash leaves both files alone")
cl = tree("clash")
ckpt(cl, "fno_ch_T700.pt", meta=RM(700.0), best=best_of(1, .1), seed=5,
     when=10)
ckpt(cl, "Work/models/train_t0700/fno_ch.pt", best=best_of(1, .1), seed=99,
     when=11)
ckpt(cl, "fno_ch_T500.pt", best=best_of(2, .2), seed=7, when=50)
ckpt(cl, "fno_ch_500.pt", best=best_of(2, .2), seed=8, when=60)
ckpt(cl, "fno_ch_T600.pt", best=best_of(3, .3), seed=9, when=70)
clone(cl, "fno_ch_T600.pt", "Work/models/train_t0600/fno_ch.pt", when=71)
put(cl, "gt_cache/gt_a.npz", b"mine", when=80)
put(cl, "Work/gt_cache/gt_a.npz", b"theirs", when=81)
put(cl, "gt_cache/gt_new.npz", b"new", when=82)
put(cl, "results_T1000_P20000/fig.png", b"new", when=90)
put(cl, "Results/train_t1000_pred_t20000/fig.png", b"old", when=91)
before = digests(cl)
S = steps_of(cl)
check(S["fno_ch_T700.pt"].status == "clash"
      and S["gt_cache/gt_a.npz"].status == "clash"
      and S["results_T1000_P20000"].status == "clash",
      "a checkpoint, a gt_cache file and a results folder are clashes")
check(S["gt_cache/gt_new.npz"].status == "todo",
      "gt_cache is merged file by file when Work/gt_cache exists")
check(S["fno_ch_T500.pt"].status == "todo"
      and S["fno_ch_500.pt"].status == "clash"
      and "differs" in S["fno_ch_500.pt"].why,
      "two different files for one destination: the earlier moves, the "
      "later is a clash")
rc, out = run(cl, "--apply")
after = digests(cl)
check(rc == 0, "a clash is not a failure")
check(all(after.get(p) == before[p] for p in
          ("fno_ch_T700.pt", "Work/models/train_t0700/fno_ch.pt",
           "gt_cache/gt_a.npz", "Work/gt_cache/gt_a.npz",
           "results_T1000_P20000/fig.png",
           "Results/train_t1000_pred_t20000/fig.png",
           "fno_ch_500.pt")),
      "both sides of every clash are untouched, bytes and time")
check(after["Work/models/train_t0500/fno_ch.pt"]
      == before["fno_ch_T500.pt"]
      and "Work/gt_cache/gt_new.npz" in after
      and not os.path.exists(P(cl, "gt_cache/gt_new.npz")),
      "and the rest of the plan still ran")
check("Archive/duplicates/fno_ch_T600.pt" in after
      and after["Work/models/train_t0600/fno_ch.pt"]
      == before["Work/models/train_t0600/fno_ch.pt"],
      "a source identical to what is already in place goes to duplicates")
log_pairs = {(e["src"], e["dst"]) for e in json.load(
    open(logs_of(cl)[0], encoding="utf-8"))["moves"]}
check(("fno_ch_T700.pt", "Work/models/train_t0700/fno_ch.pt")
      not in log_pairs, "the log does not claim the clash was moved")

print("\n19. duplicates keep the earliest mtime, not the first name")
dp = tree("dupes")
ckpt(dp, "fno_ch_500.pt", best=best_of(2, .2), seed=1, when=5)
clone(dp, "fno_ch_500.pt", "fno_ch.pt", when=50)
npz(dp, "test.npz", 2990, seed=3, when=60)
clone(dp, "test.npz", "test_numpy.npz", when=6)
S = steps_of(dp)
check(S["fno_ch_500.pt"].dst == "Work/models/train_t0500/fno_ch.pt"
      and S["fno_ch.pt"].dst == "Archive/duplicates/fno_ch.pt",
      "the older fno_ch_500.pt stays, the newer fno_ch.pt is the duplicate")
check(S["test_numpy.npz"].dst == "Work/data/test_numpy.npz"
      and S["test.npz"].dst == "Archive/duplicates/test.npz",
      "the older dataset stays, the later-mtime one is the duplicate")
orig = digests(dp)
rc, out = run(dp, "--apply")
check(digests(dp)["Work/models/train_t0500/fno_ch.pt"] == orig[
    "fno_ch_500.pt"], "and the kept file is the older one, mtime intact")

# -------------------------------------------------------------- failures
print("\n20. a failed move is reported and the rest goes on")
fl = tree("fail")
ckpt(fl, "fno_ch_T10.pt", best=best_of(1, .1), seed=1, when=10)
ckpt(fl, "fno_ch_T20.pt", best=best_of(2, .2), seed=2, when=11)
ckpt(fl, "fno_ch_T40.pt", best=best_of(3, .3), seed=3, when=12)
lean(fl, "data/p", seed=1, when=13)
put(fl, "ch_gpu_9.out", "o", when=14)
orig = digests(fl)
real_replace, real_move = os.replace, shutil.move
moved_by_shutil = []


def flaky(src, dst, *a, **k):
    if str(src).endswith(("fno_ch_T10.pt", "p.npy")):
        raise PermissionError(errno.EACCES, "locked by another program",
                              str(src))
    if str(src).endswith("fno_ch_T40.pt"):
        raise OSError(errno.EXDEV, "cross-device link", str(src))
    return real_replace(src, dst, *a, **k)


def spy(src, dst, *a, **k):
    moved_by_shutil.append(str(dst))
    return real_move(src, dst, *a, **k)


os.replace, shutil.move = flaky, spy
try:
    rc, out = run(fl, "--apply")
finally:
    os.replace, shutil.move = real_replace, real_move
now = digests(fl)
check(rc == 1, "the exit code is 1 when a step failed")
check("FAILED" in out and "locked by another program" in out,
      "the failure and its reason are printed")
check(now["fno_ch_T10.pt"] == orig["fno_ch_T10.pt"]
      and "Work/models/train_t0010/fno_ch.pt" not in now,
      "the locked file stays")
check(now["Work/models/train_t0020/fno_ch.pt"] == orig["fno_ch_T20.pt"]
      and now["Work/slurm/ch_gpu_9.out"] == orig["ch_gpu_9.out"],
      "the other moves went on")
check("data/p.npy" in now and "data/p.json" in now
      and "Work/data/p.json" not in now,
      "a pair whose .npy failed is not split: the sidecar stays too")
check(len(moved_by_shutil) == 1
      and now["Work/models/train_t0040/fno_ch.pt"] == orig["fno_ch_T40.pt"],
      "EXDEV falls back to shutil.move and keeps bytes and mtime")
entries = {e["src"]: e for e in json.load(
    open(logs_of(fl)[0], encoding="utf-8"))["moves"]}
check(entries["fno_ch_T10.pt"]["status"] == "failed"
      and entries["fno_ch_T10.pt"]["undo"] is None
      and entries["fno_ch_T20.pt"]["status"] == "done",
      "the log records the failure and the successes")

print("\n21. a re-run places what a failed run left behind")
rr2 = tree("rerun")
b5 = best_of(35, 0.91)
ckpt(rr2, "fno_ch.pt", best=b5, seed=1, when=10)
clone(rr2, "fno_ch.pt", "fno_ch_500.pt", when=500)
ckpt(rr2, "fno_ch_last.pt", "last", best=b5, history=hist(3), seed=1,
     when=20)
put(rr2, "results_P2000/fig.png", b"r", when=30)
manifest(rr2, "results_P2000", "fno_ch.pt", ref="fno_ch.pt")
orig = digests(rr2)


def locked(src, dst, *a, **k):
    if str(src).endswith(("fno_ch_last.pt", "results_P2000")):
        raise PermissionError(errno.EACCES, "in use", str(src))
    return real_replace(src, dst, *a, **k)


os.replace = locked
try:
    rc1, out1 = run(rr2, "--apply")
finally:
    os.replace = real_replace
check(rc1 == 1 and os.path.isfile(P(rr2, "fno_ch_last.pt"))
      and os.path.isfile(P(rr2, "results_P2000/fig.png"))
      and os.path.isfile(P(rr2, "Work/models/train_t0500/fno_ch.pt")),
      "the first run fails on a locked state and a locked results folder")
S = steps_of(rr2)
check(S["fno_ch_last.pt"].status == "todo"
      and S["fno_ch_last.pt"].dst == "Work/models/train_t0500/fno_ch_last.pt",
      "the second plan finds its horizon from the best already filed")
check(S["results_P2000"].dst == "Results/train_t0500_pred_t02000",
      "and names the results folder from the manifest's size and time, "
      "though fno_ch.pt has left the root")
rc2, out2 = run(rr2, "--apply")
now = digests(rr2)
check(rc2 == 0 and now["Work/models/train_t0500/fno_ch_last.pt"]
      == orig["fno_ch_last.pt"]
      and now["Results/train_t0500_pred_t02000/fig.png"]
      == orig["results_P2000/fig.png"], "and the second run moves both")
check(len(logs_of(rr2)) == 2, "each run wrote its own log")

print("\n22. a symlinked dataset keeps pointing at its target")
sl = tree("link")
npz(sl, "elsewhere/real.npz", 2990, seed=1, when=10)
try:
    os.symlink(os.path.join("elsewhere", "real.npz"), P(sl, "test.npz"))
    linked = True
except (OSError, NotImplementedError):
    linked = False
if linked:
    rc, out = run(sl, "--apply")
    new = P(sl, "Work/data/test.npz")
    check(rc == 0 and os.path.islink(new)
          and os.path.samefile(new, P(sl, "elsewhere/real.npz")),
          "the moved link still resolves to the same file")
else:
    print("SKIP  symlinks need a privilege this account lacks")

if ok:
    shutil.rmtree(BASE, ignore_errors=True)
print("\n" + ("MIGRATE PASS" if ok else "MIGRATE FAIL") + f" ({count} checks)")
sys.exit(0 if ok else 1)
