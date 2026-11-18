"""Run the suites side by side, then pyright over the shipped files.

    python tests/run_all.py                    every suite, 4 at a time, then pyright
    python tests/run_all.py cli health         the suites whose name contains one of these
    python tests/run_all.py --changed fno_gt.py make_submission.py
                                               only the suites that import (directly
                                               or through another project module) one
                                               of these files; pyright on them only
    python tests/run_all.py --failed           rerun what failed or crashed last time
    python tests/run_all.py --sections         where the time went in the slowest suites
    python tests/run_all.py -j 6 --threads 3   six at a time, three threads each
    python tests/run_all.py --changed fno_gt.py --dry-run   which suites, and why, without running
    --no-lint     skip pyright              --tail N    log lines shown for a failure (15)
    --timeout S   limit per suite (7200 s)  --keep      keep the scratch of passing suites

Put suite names before --changed, which takes every word after it.

Each suite is a standalone script that exits non-zero on failure.  Run here,
every suite is its own process with

* a scratch folder of its own (FNO_TEST_SCRATCH=tests/_scratch/<suite>, see
  _paths.py), so suites that list or clear their scratch never meet;
* the thread libraries held to --threads (default min(4, 24 // jobs)), so
  four suites do not become four times every core of a shared login node;
* its matplotlib and torch caches in tests/_scratch/_env/<suite>, outside the
  suite's own scratch (a suite may list that), seeded with the font list so
  matplotlib does not rebuild it each time.

The reference cache (FNO_GT_CACHE) and the checkpoint (FNO_TEST_MODEL) are
shared, and are read through from the environment unchanged.  Both are safe to
share: the references are written under a host-and-pid name and renamed into
place (fno_gt, fno_train), so two suites solving the same one cannot corrupt it.

Output goes to tests/_scratch/logs/<suite>.log, each line prefixed with the
seconds since that suite started, so the slow part of a suite shows without
rerunning it (--sections reads them).  The console gets one line per suite and,
for a failure, its failing checks and the end of its log.  The latest result of
every suite is kept in tests/_scratch/last_run.json (status, seconds, counts,
log path) for --failed and for the slowest-first order of the next run.

tests/_scratch/run_all.lock holds the pid: a second runner refuses to start
while the first lives, and takes the lock over when that pid is gone.
"""
from __future__ import annotations

import argparse
import ast
import atexit
import collections
import concurrent.futures as cf
import datetime
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

from _paths import ROOT, SCRATCH

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(SCRATCH, "logs")
LAST = os.path.join(SCRATCH, "last_run.json")
LOCK = os.path.join(SCRATCH, "run_all.lock")
ENVS = os.path.join(SCRATCH, "_env")

SUITES = ("t_paths", "t_trainval", "t_conserve", "t_timing", "t_layout",
          "t_health", "t_runlog", "t_budget", "t_seeds", "t_cli",
          "t_gtcache", "t_submission", "t_submission_regen",
          "t_submission_publish", "t_records", "t_figures",
          "t_report", "t_lean_data",
          "t_lockstep", "t_migrate", "t_gtbatch", "t_smoke", "t_ckpt",
          "t_modelsel", "t_provenance", "t_diag", "t_tools", "t_protocol",
          "t_vdt", "t_flat", "t_schedule", "t_sweeps", "t_drift", "t_ablate")

SHIPPED = ("ch_gpu.py", "fno_paths.py", "fno_runlog.py",
           "fno_train.py", "fno_gt.py", "fno_predict.py", "make_submission.py",
           "migrate_layout.py", "hpc_storage.py", "make_diagram.py")

DEFAULT_JOBS = 4
MAX_THREADS = 4          # per suite
THREAD_BUDGET = 24       # all suites together; the node is shared
DEFAULT_TIMEOUT = 7200   # seconds per suite
BAD = ("FAIL", "CRASH", "TIMEOUT", "INTERRUPTED")

# Seconds the slowest suites took (2026-10-29, -j 4 x 4 threads, the login
# node); the order of a run is slowest first, from last_run.json where it has
# the suite and from these where it does not, so the long suite never starts
# last.  The rest are under 30 s.
TYPICAL: dict[str, float] = {
    "t_submission_regen": 350.0, "t_records": 330.0,
    "t_submission": 320.0, "t_submission_publish": 320.0, "t_flat": 300.0,
    "t_lockstep": 260.0, "t_schedule": 110.0, "t_report": 110.0,
    "t_smoke": 100.0, "t_layout": 80.0, "t_timing": 70.0, "t_figures": 60.0,
    "t_provenance": 60.0, "t_cli": 60.0, "t_protocol": 50.0,
    "t_conserve": 45.0, "t_vdt": 30.0}

CHECK = re.compile(r"\s*(PASS|FAIL|SKIP)(?=\s|$)")
STAMP = re.compile(r"^\[\s*(\d+(?:\.\d+)?)s\] ?(.*)$")
HEADING = re.compile(r"\d+[a-z]?\. \S")

_running: dict[str, subprocess.Popen] = {}
_state = threading.Lock()


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


# --- the lock: one runner at a time -----------------------------------------

def pid_alive(pid):
    """Whether the process exists (and, where /proc says, is a runner)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    cmd = f"/proc/{pid}/cmdline"
    if os.path.exists(cmd):
        try:
            with open(cmd, "rb") as fh:
                return b"run_all" in fh.read()
        except OSError:
            return True
    return True


def read_lock():
    try:
        with open(LOCK, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def drop_lock():
    held = read_lock()
    if held and held.get("pid") == os.getpid():
        try:
            os.remove(LOCK)
        except OSError:
            pass


def take_lock():
    """None once the lock is ours; else what holds it."""
    host = socket.gethostname()
    for _ in range(3):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            held = read_lock()
            if held is None:     # being written just now, or left empty
                try:
                    young = time.time() - os.path.getmtime(LOCK) < 10
                except OSError:
                    continue
                if young:
                    time.sleep(0.5)
                    continue
                held = {"pid": -1, "host": host}
            # another host's pid cannot be looked at from here
            if held.get("host") != host or pid_alive(int(held.get("pid", -1))):
                return held
            try:
                os.remove(LOCK)      # its runner is gone
            except OSError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid(), "host": host, "started": now(),
                       "argv": sys.argv[1:]}, fh)
        atexit.register(drop_lock)
        return None
    return {"pid": -1, "host": host, "started": "?"}


# --- which suites: names, --changed, --failed -------------------------------

def project_modules():
    """name -> path of the project's own modules: the root's and the helpers
    in tests/ (_paths), but not the suites."""
    mods = {}
    for folder in (ROOT, HERE):
        for f in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(f)
            if ext == ".py" and not stem.startswith("t_") and stem != "run_all":
                mods.setdefault(stem, os.path.join(folder, f))
    return mods


_imports: dict[str, dict[str, bool]] = {}


def imported_names(path):
    """{top-level module name: True when only imported inside a function}
    for every import statement in the file (and import_module("x") calls)."""
    if path in _imports:
        return _imports[path]
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), path)
    found: dict[str, bool] = {}

    def note(name, inner):
        top = name.split(".")[0]
        found[top] = found.get(top, True) and inner

    def visit(node, inner):
        for child in ast.iter_child_nodes(node):
            here = inner or isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, ast.Import):
                for a in child.names:
                    note(a.name, here)
            elif isinstance(child, ast.ImportFrom):
                if child.level == 0 and child.module:
                    note(child.module, here)
            elif isinstance(child, ast.Call) and child.args:
                fn = child.func
                name = (fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute) else "")
                arg = child.args[0]
                if (name in ("import_module", "__import__")
                        and isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)):
                    note(arg.value, here)
            visit(child, here)

    visit(tree, False)
    _imports[path] = found
    return found


def reach(path, mods):
    """{project module: (chain from the file's own import, lazy)}: every
    project module the file imports, directly (an import inside a function
    counts: lazy) or through the others, by the shortest chain.  Past the
    file's own imports only top-level imports are followed: a module that
    imports another inside a function (fno_predict reaching make_submission,
    _paths building a model) would otherwise tie every suite to everything;
    the suites that exercise such a module name it, which pick_changed sees."""
    out: dict[str, tuple[list[str], bool]] = {}
    queue: collections.deque[str] = collections.deque()
    for name, inner in imported_names(path).items():
        if name in mods:
            out[name] = ([name], inner)
            queue.append(name)
    while queue:
        cur = queue.popleft()
        chain, lazy = out[cur]
        for name, inner in imported_names(mods[cur]).items():
            if name in mods and name not in out and not inner:
                out[name] = (chain + [name], lazy)
                queue.append(name)
    return out


def find_file(name):
    for base in (os.getcwd(), ROOT, HERE):
        p = os.path.abspath(os.path.join(base, name))
        if os.path.exists(p):
            return p
    return None


def pick_changed(files, suites):
    """(suites, report lines, python files to lint) for the changed files."""
    mods = project_modules()
    reaches = {s: reach(os.path.join(HERE, s + ".py"), mods) for s in suites}
    texts = {}
    for s in suites:
        with open(os.path.join(HERE, s + ".py"), encoding="utf-8") as fh:
            texts[s] = fh.read()
    picked: set[str] = set()
    lines, lint = [], []
    for f in files:
        path = find_file(f)
        if path is None:
            sys.exit(f"--changed {f}: no such file (looked in the current "
                     f"folder, {ROOT} and {HERE})")
        base = os.path.basename(path)
        stem, ext = os.path.splitext(base)
        if ext == ".py":
            lint.append(os.path.relpath(path, ROOT))
        groups: dict[str, list[str]] = {}
        for s in suites:
            if os.path.abspath(path) == os.path.join(HERE, s + ".py"):
                key = "is that suite"
            elif ext == ".py" and stem in reaches[s]:
                chain, lazy = reaches[s][stem]
                key = ("imports it" if len(chain) == 1 else
                       "via " + " > ".join(chain[:-1]))
                key += " (in a function)" if lazy else ""
            elif re.search(r"(?<![\w.])" + re.escape(stem if ext == ".py"
                                                     else base) + r"(?!\w)",
                           texts[s]):
                key = "names it (runs or reads it)"
            else:
                continue
            groups.setdefault(key, []).append(s)
            picked.add(s)
        if not groups:
            lines.append(f"    {base}: no suite imports or names it")
        for key, names in groups.items():
            lines.append(f"    {base}: {key}: {' '.join(names)}")
    return [s for s in suites if s in picked], lines, lint


def load_last():
    try:
        with open(LAST, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def save_last(data):
    tmp = LAST + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, LAST)


def expected(name, last):
    got = (last.get("suites") or {}).get(name) or {}
    return float(got.get("seconds") or TYPICAL.get(name, 0.0))


# --- one suite --------------------------------------------------------------

def seed_font_cache(dst):
    """Copy matplotlib's font list into dst, so a suite does not rebuild it."""
    src = os.environ.get("MPLCONFIGDIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
        "matplotlib")
    try:
        for f in os.listdir(src):
            if f.startswith("fontlist-") and f.endswith(".json"):
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    except OSError:
        pass


def kill_group(proc, sig=signal.SIGKILL):
    try:
        if sys.platform == "win32":
            proc.kill()
        else:
            os.killpg(proc.pid, sig)
    except OSError:
        pass


def run_suite(name, threads, timeout, tail):
    """Run one suite to its end; its log is written line by line."""
    scratch = os.path.join(SCRATCH, name)
    caches = os.path.join(ENVS, name)
    for d in (scratch, caches):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(scratch)
    os.makedirs(os.path.join(caches, "mpl"))
    seed_font_cache(os.path.join(caches, "mpl"))
    n = str(threads)
    env = dict(os.environ)
    env.update(FNO_TEST_SCRATCH=scratch, PYTHONUNBUFFERED="1",
               MPLCONFIGDIR=os.path.join(caches, "mpl"),
               TORCHINDUCTOR_CACHE_DIR=os.path.join(caches, "inductor"),
               TRITON_CACHE_DIR=os.path.join(caches, "triton"),
               OMP_NUM_THREADS=n, MKL_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n,
               NUMEXPR_NUM_THREADS=n, VECLIB_MAXIMUM_THREADS=n)
    log_path = os.path.join(LOGS, name + ".log")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    fails: list[str] = []
    ends: collections.deque[str] = collections.deque(maxlen=max(tail, 1))
    timed_out = []
    t0 = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"[    0.0s] # run_all.py: {n} thread(s), scratch {scratch}\n")
        proc = subprocess.Popen(
            [sys.executable, "-u", os.path.join(HERE, name + ".py")],
            cwd=HERE, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
            start_new_session=(sys.platform != "win32"))
        with _state:
            _running[name] = proc

        def too_long():
            timed_out.append(True)
            kill_group(proc)

        timer = threading.Timer(timeout, too_long)
        timer.daemon = True
        timer.start()
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            stamped = f"[{time.monotonic() - t0:7.1f}s] {line}"
            log.write(stamped + "\n")
            log.flush()
            ends.append(stamped)
            m = CHECK.match(line)
            if m:
                counts[m.group(1)] += 1
                if m.group(1) == "FAIL" and len(fails) < 10:
                    fails.append(stamped)
        code = proc.wait()
        timer.cancel()
    with _state:
        _running.pop(name, None)
    secs = time.monotonic() - t0
    if timed_out:
        status = "TIMEOUT"
    elif counts["FAIL"]:
        status = "FAIL"
    elif code == 0:
        status = "PASS"
    else:
        status = "CRASH"
    return {"suite": name, "status": status, "seconds": round(secs, 1),
            "passes": counts["PASS"], "fails": counts["FAIL"],
            "skips": counts["SKIP"], "returncode": code,
            "log": log_path, "finished": now(),
            "_fail_lines": fails, "_ends": list(ends)}


def describe(res):
    n = res["passes"] + res["fails"]
    bits = [f"{n} checks"] if n else []
    if res["fails"]:
        bits = [f"{res['fails']} of {n} checks failed"]
    if res["skips"]:
        bits.append(f"{res['skips']} SKIP")
    if res["status"] == "CRASH":
        bits.append(f"exit {res['returncode']}")
    elif res["status"] == "TIMEOUT":
        bits.append("killed at the time limit")
    elif res["status"] == "FAIL" and res["returncode"] == 0:
        bits.append("exit 0 but FAIL lines")
    return ", ".join(bits)


def short(line, width=170):
    return line if len(line) <= width else line[:width - 3] + "..."


def crash_view(log_path, tail):
    """What of a crashed suite's log is worth reading: its last traceback
    without the frames inside installed libraries (torch's call chain is most
    of it), or the last lines when there is none."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    start = next((i for i in range(len(lines) - 1, -1, -1)
                  if "Traceback (most recent call last):" in lines[i]), None)
    if start is None:
        return lines[-tail:]
    out, skipping = [], False
    for ln in lines[start:]:
        m = STAMP.match(ln)
        body = m.group(2) if m else ln
        if body.lstrip().startswith('File "'):
            skipping = "site-packages" in body or "/lib/python" in body
        elif not body.startswith(" "):
            skipping = False
        if not skipping:
            out.append(ln)
    return out if len(out) <= tail else out[:1] + out[-(tail - 1):]


# --- pyright ----------------------------------------------------------------

# pyright runs a bundled node, which needs libatomic: on the cluster it is in
# the gcc module's lib64, which the login shell does not put on the library
# path.  Added only where the folder exists, so other machines are unaffected.
GCC_LIB = "/opt/ohpc/pub/compiler/gcc/12.2.0/lib64"


def pyright_env():
    """The environment for pyright: this one, plus GCC_LIB on the library path
    where that folder exists."""
    env = dict(os.environ)
    if os.path.isdir(GCC_LIB):
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            x for x in (GCC_LIB, env.get("LD_LIBRARY_PATH", "")) if x)
    return env


def run_pyright(files):
    t0 = time.monotonic()
    # --pythonpath: resolve imports (torch above all) from this interpreter, as
    # a manual `pyright --pythonpath env/bin/python` run does; without it two
    # false errors appear in fno_train.py (log_gain, gain_exp).
    p = subprocess.run([sys.executable, "-m", "pyright", "--pythonpath",
                        sys.executable, *files], cwd=ROOT, env=pyright_env(),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    secs = time.monotonic() - t0
    out = (p.stdout or "").strip() or (p.stderr or "").strip()
    with open(os.path.join(LOGS, "pyright.log"), "w", encoding="utf-8") as fh:
        fh.write(out + "\n")
    tail = [ln for ln in out.splitlines() if ln.strip()]
    counts = [ln for ln in tail if re.match(r"\s*\d+ errors?,", ln)]
    summary = (counts[-1] if counts else
               tail[-1] if tail else "pyright produced no output")
    detail = [ln for ln in tail
              if ln is not summary and not ln.startswith("WARNING")
              and "PYRIGHT_PYTHON_FORCE_VERSION" not in ln]
    ok = summary.strip().startswith("0 errors") and " 0 warnings" in summary
    return {"status": "PASS" if ok else "FAIL", "summary": summary.strip(),
            "seconds": round(secs, 1), "files": len(files),
            "_detail": detail}


# --- where the time went ----------------------------------------------------

def log_sections(path):
    """([(start, heading)], end, [(silence, line, at)]) of a stamped log:
    the numbered headings the suite prints, and the longest gaps between
    lines."""
    heads: list[tuple[float, str]] = [(0.0, "(before the first heading)")]
    gaps: list[tuple[float, str, float]] = []
    prev, end = 0.0, 0.0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            m = STAMP.match(raw.rstrip("\n"))
            if not m:
                continue
            t, text = float(m.group(1)), m.group(2)
            if HEADING.match(text):
                heads.append((t, text.strip()))
            gaps.append((t - prev, text.strip(), t))
            prev = end = t
    return heads, end, gaps


def show_sections(names, rows=6):
    last = load_last()
    have = last.get("suites") or {}
    if not names:
        names = [s for s, _ in sorted(have.items(),
                                      key=lambda kv: -kv[1].get("seconds", 0))[:3]]
    if not names:
        print("no earlier run in", LAST)
        return 1
    for name in names:
        got = have.get(name)
        path = (got or {}).get("log") or os.path.join(LOGS, name + ".log")
        if not os.path.exists(path):
            print(f"{name}: no log at {path}")
            continue
        heads, end, gaps = log_sections(path)
        spans = [(heads[i + 1][0] if i + 1 < len(heads) else end) - heads[i][0]
                 for i in range(len(heads))]
        print(f"{name}  {end:.1f} s in the log"
              + (f"  (last run {got.get('status')}, {got.get('seconds')} s)"
                 if got else ""))
        for span, (at, title) in sorted(zip(spans, heads), reverse=True)[:rows]:
            if span >= 1.0:
                print(f"  {span:8.1f} s  from {at:7.1f} s  {short(title, 90)}")
        for gap, text, at in sorted(gaps, reverse=True)[:3]:
            if gap >= 5.0:
                print(f"  silence {gap:.1f} s before [{at:.1f} s] "
                      f"{short(text, 80)}")
    return 0


# --- the run ----------------------------------------------------------------

def interrupt_children():
    with _state:
        procs = list(_running.values())
    for p in procs:
        kill_group(p, signal.SIGTERM)
    time.sleep(1.0)
    for p in procs:
        kill_group(p, signal.SIGKILL)


def parse(argv):
    ap = argparse.ArgumentParser(
        prog="run_all.py", description="Run the test suites, then pyright.",
        epilog="Put suite names before --changed, which takes every word "
               "after it.")
    ap.add_argument("names", nargs="*", metavar="NAME",
                    help="only the suites whose name contains one of these")
    ap.add_argument("-j", "--jobs", type=int, default=DEFAULT_JOBS,
                    help=f"suites at once (default {DEFAULT_JOBS})")
    ap.add_argument("--threads", type=int, default=None,
                    help="threads per suite (default min(%d, %d // jobs))"
                         % (MAX_THREADS, THREAD_BUDGET))
    ap.add_argument("--changed", nargs="+", metavar="FILE",
                    help="the suites that import one of these files; pyright "
                         "on these files only")
    ap.add_argument("--failed", action="store_true",
                    help="only what failed or crashed in the last run")
    ap.add_argument("--sections", nargs="*", metavar="SUITE",
                    help="show where the time went in these suites' logs "
                         "(default: the 3 slowest of the last run) and exit")
    ap.add_argument("--no-lint", action="store_true", help="skip pyright")
    ap.add_argument("--dry-run", action="store_true",
                    help="only say which suites would run, and why")
    ap.add_argument("--tail", type=int, default=15,
                    help="log lines shown for a crash (default 15)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help=f"seconds a suite may run (default {DEFAULT_TIMEOUT})")
    ap.add_argument("--keep", action="store_true",
                    help="keep the scratch folder of suites that pass")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    if args.sections is not None:
        return show_sections(args.sections)
    if args.jobs < 1:
        sys.exit("-j: at least 1")
    last = load_last()
    unknown = sorted(f[:-3] for f in os.listdir(HERE)
                     if f.startswith("t_") and f.endswith(".py")
                     and f[:-3] not in SUITES)
    if unknown:
        print(f"  note: not in SUITES, so not run: {', '.join(unknown)}")

    picked = [s for s in SUITES
              if not args.names or any(a in s for a in args.names)]
    notes: list[str] = []
    lint_files = None
    if args.changed:
        picked, notes, py = pick_changed(args.changed, picked)
        lint_files = py
    lint_wanted = not args.no_lint
    if args.failed:
        old = last.get("suites") or {}
        picked = [s for s in picked if (old.get(s) or {}).get("status") in BAD]
        # a pyright that failed last time is run again, even with no suite
        lint_wanted = lint_wanted and bool(picked) or (
            lint_wanted and (last.get("pyright") or {}).get("status") == "FAIL")
    elif args.changed:
        lint_wanted = lint_wanted and bool(lint_files)
    if not picked and not lint_wanted:
        if args.failed:
            print(f"  nothing failed or crashed in the last run ({LAST})")
            return 0
        if args.changed:
            print("\n".join(notes))
            print("  no suite imports a changed file; nothing to run")
            return 0
        sys.exit(f"no suite matches {args.names}; have {', '.join(SUITES)}")

    if args.dry_run:
        if notes:
            print("\n".join(notes))
        print(f"  {len(picked)} suite(s) would run: {' '.join(picked)}"
              + ("" if lint_wanted else "  (no pyright)"))
        return 0

    held = take_lock()
    if held:
        print(f"  another run_all.py holds {LOCK}: pid {held.get('pid')} on "
              f"{held.get('host')}, started {held.get('started')} "
              f"({' '.join(held.get('argv') or [])}).\n"
              f"  Never two test runs at once.  If that run is gone, delete "
              f"the lock file.")
        return 2

    cores = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
             else os.cpu_count() or 1)
    jobs = max(1, min(args.jobs, len(picked) or 1, cores))
    threads = args.threads or max(
        1, min(MAX_THREADS, THREAD_BUDGET // jobs, cores // jobs))
    order = sorted(picked, key=lambda s: -expected(s, last))
    bar = "=" * 66
    print(f"{bar}\n  {len(picked)} suite(s), {jobs} at a time x {threads} "
          f"thread(s), against {ROOT}")
    if args.changed:
        print("  --changed " + " ".join(args.changed))
        print("\n".join(notes))
    print(f"  logs: {LOGS}\n{bar}", flush=True)

    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(ENVS, exist_ok=True)
    started = now()
    t_all = time.monotonic()
    data = load_last()
    data.setdefault("suites", {})
    results: dict[str, dict] = {}
    lint_box: dict[str, dict] = {}
    lint_thread = None
    if lint_wanted:
        files = (lint_files if lint_files is not None else
                 [*SHIPPED, *sorted(os.path.join("tests", f)
                                    for f in os.listdir(HERE)
                                    if f.endswith(".py"))])
        files = [f for f in files if os.path.exists(os.path.join(ROOT, f))]
        if files:
            lint_thread = threading.Thread(
                target=lambda: lint_box.update(r=run_pyright(files)),
                daemon=True)
            lint_thread.start()

    interrupted = False
    pool = cf.ThreadPoolExecutor(max_workers=jobs)
    try:
        futures = [pool.submit(run_suite, s, threads, args.timeout, args.tail)
                   for s in order]
        for fut in cf.as_completed(futures):
            res = fut.result()
            name = res["suite"]
            results[name] = res
            print(f"  {name:<14} {res['status']:<7}{res['seconds']:7.1f}s  "
                  f"{describe(res)}", flush=True)
            if res["status"] != "PASS":
                if res["status"] == "FAIL":
                    for ln in res["_fail_lines"]:
                        print("      " + short(ln))
                    if res["fails"] > len(res["_fail_lines"]):
                        print(f"      ... {res['fails'] - len(res['_fail_lines'])}"
                              f" more FAIL lines in the log")
                    for ln in res["_ends"][-4:]:
                        print("      " + short(ln))
                else:
                    for ln in crash_view(res["log"], args.tail):
                        print("      " + short(ln))
                print(f"      log: {res['log']}", flush=True)
            elif not args.keep:
                shutil.rmtree(os.path.join(SCRATCH, name), ignore_errors=True)
                shutil.rmtree(os.path.join(ENVS, name), ignore_errors=True)
            data["suites"][name] = {k: v for k, v in res.items()
                                    if not k.startswith("_")}
            save_last(data)
    except KeyboardInterrupt:
        interrupted = True
        print("\n  interrupted: stopping the suites", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        interrupt_children()
        for s in order:
            if s not in results:
                data["suites"][s] = {"suite": s, "status": "INTERRUPTED",
                                     "seconds": 0.0, "log": os.path.join(
                                         LOGS, s + ".log"), "finished": now()}
    finally:
        pool.shutdown(wait=True)
    print(bar)

    lint_bad = 0
    if lint_thread is not None and not interrupted:
        lint_thread.join()
        lint = lint_box["r"]
        print(f"  pyright        {lint['summary']}  ({lint['seconds']:.0f} s, "
              f"{lint['files']} files)")
        if lint["status"] != "PASS":
            lint_bad = 1
            for ln in lint["_detail"][:40]:
                print("      " + short(ln.strip()))
            print(f"      log: {os.path.join(LOGS, 'pyright.log')}")
        data["pyright"] = {k: v for k, v in lint.items()
                           if not k.startswith("_")}
        data["pyright"]["finished"] = now()

    failed = [s for s in order if (results.get(s) or {}).get("status") != "PASS"]
    wall = time.monotonic() - t_all
    work = sum(r["seconds"] for r in results.values())
    data["run"] = {"started": started, "finished": now(),
                   "wall_seconds": round(wall, 1), "jobs": jobs,
                   "threads": threads, "argv": sys.argv[1:],
                   "suites": order}
    save_last(data)
    slow = sorted(results.values(), key=lambda r: -r["seconds"])[:3]
    print(f"  {len(order) - len(failed)}/{len(order)} suites passed in "
          f"{wall / 60:.1f} min  (the suites alone add up to "
          f"{work / 60:.1f} min)")
    if slow:
        print("  slowest: " + ", ".join(f"{r['suite']} {r['seconds']:.0f}s"
                                        for r in slow)
              + "   (python tests/run_all.py --sections)")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        print("  rerun them with:  python tests/run_all.py --failed")
    print(bar, flush=True)

    if not failed and not lint_bad and not args.keep and not args.changed \
            and not args.names and not args.failed:
        # a clean full run leaves only the logs, last_run.json and the lock
        for entry in os.listdir(SCRATCH):
            if entry not in ("logs", os.path.basename(LAST),
                             os.path.basename(LOCK)):
                p = os.path.join(SCRATCH, entry)
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
    if interrupted:
        return 130
    return 1 if (failed or lint_bad) else 0


if __name__ == "__main__":
    sys.exit(main())
