"""The run log has to survive the ways these jobs actually end."""
import glob
import io
import os
import shutil
import subprocess
import sys
import time

from _paths import ROOT, SCRATCH

HERE = SCRATCH
os.chdir(HERE)
LOGS = os.path.join(HERE, "Work", "logs")
ELSEWHERE = os.path.join(HERE, "runlog_elsewhere")
for d in (LOGS, ELSEWHERE):
    shutil.rmtree(d, ignore_errors=True)

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


def child(body, kill_after=None):
    """Run a snippet in its own process so atexit and signals are real."""
    src = (f"import sys; sys.path.insert(0, {ROOT!r})\n"
           "import fno_runlog as runlog\n" + body)
    p = subprocess.Popen([sys.executable, "-c", src], cwd=HERE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    if kill_after is not None:
        time.sleep(kill_after)
        p.kill()
    out, _ = p.communicate(timeout=90)
    return out, p.returncode


def newest():
    fs = sorted(glob.glob(os.path.join(LOGS, "*.log")), key=os.path.getmtime)
    return fs[-1] if fs else None


def read(path):
    return io.open(path, encoding="utf-8").read()


print("1. a normal run")
out, rc = child("""
runlog.start_log("probe")
print("hello from stdout")
print("and from stderr", file=sys.stderr)
""")
log = newest()
check(log is not None, f"a log file was created ({os.path.basename(log or '')})")
check(rc == 0, f"a clean run still exits 0 (got {rc})")
body = read(log)
check("hello from stdout" in out, "terminal still sees stdout")
check("hello from stdout" in body, "log captured stdout")
check("and from stderr" in body, "log captured stderr")
check("host" in body and "python" in body and "command" in body,
      "header records host, python and command line")
check("elapsed" in body and "finished" in body, "footer written at exit")

print("\n2. an unhandled exception")
out, rc = child("""
runlog.start_log("probe")
print("before the crash")
raise RuntimeError("boom")
""")
body = read(newest())
check(rc != 0, "the process still exits non-zero")
check("before the crash" in body, "output before the crash is kept")
check("RuntimeError: boom" in body and "Traceback" in body,
      "the traceback lands in the log, not just the terminal")

print("\n3. killed mid-run, as SLURM does at the wall clock")
out, rc = child("""
import time
runlog.start_log("probe")
for i in range(200):
    print(f"epoch {i}", flush=True)
    time.sleep(0.05)
""", kill_after=1.5)
body = read(newest())
lines = [l for l in body.splitlines() if l.startswith("epoch ")]
check(len(lines) > 3, f"work up to the kill survived ({len(lines)} lines)")
check("elapsed" not in body, "no footer, which is how a killed run reads")

print("\n4. non-ASCII does not kill a run on a cp1252 console")
out, rc = child("""
runlog.start_log("probe")
print("psi-bar = 0.5 " + chr(0x03C8) + " " + chr(0x2014) + " done")
print("still alive")
""")
body = read(newest())
check(rc == 0, "the run survived the character")
check("still alive" in body, "and carried on afterwards")
check(chr(0x03C8) in body, "the log kept the real character (it is utf-8)")

print("\n5. calling start_log twice is harmless")
out, rc = child("""
a = runlog.start_log("probe")
b = runlog.start_log("probe")
print("same path:", a == b)
""")
check("same path: True" in out, "second call returns the first path")
check(len(glob.glob(os.path.join(LOGS, "*.log"))) == 5,
      f"one file per run, got {len(glob.glob(os.path.join(LOGS, '*.log')))}")

print("\n6. the real ch_gpu entry point")
r = subprocess.run([sys.executable,
                    os.path.join(ROOT, "ch_gpu.py"), "--benchmark"],
                   cwd=HERE, capture_output=True, text=True, timeout=600)
logs = sorted(glob.glob(os.path.join(LOGS, "ch_gpu_*.log")))
check(bool(logs), "ch_gpu.py wrote its own log")
if logs:
    body = read(logs[-1])
    check("projected for all 25 trajectories" in body,
          "the benchmark output is in the log")
    check("run        ch_gpu" in body, "tagged as ch_gpu")
    print("   |", [l for l in body.splitlines()
                   if "this," in l or "ch.py," in l])

print("\n7. the folder is LOG_DIR, Work/logs unless it is rebound")
before = set(glob.glob(os.path.join(LOGS, "*.log")))
out, rc = child(f"""
print("DEFAULT=" + runlog.LOG_DIR)
runlog.LOG_DIR = {ELSEWHERE!r}
print("PATH=" + runlog.start_log("probe"))
""")
said = dict(ln.split("=", 1) for ln in out.splitlines()
            if ln.startswith(("DEFAULT=", "PATH=")))
check(said.get("DEFAULT") == os.path.join("Work", "logs"),
      f"by default it is Work/logs ({said.get('DEFAULT')!r})")
check(os.path.dirname(said.get("PATH", "")) == ELSEWHERE
      and glob.glob(os.path.join(ELSEWHERE, "probe_*.log")),
      f"a run after LOG_DIR is rebound logs there ({said.get('PATH')!r})")
check(set(glob.glob(os.path.join(LOGS, "*.log"))) == before,
      "and nothing goes to Work/logs")

for d in (LOGS, ELSEWHERE):
    shutil.rmtree(d, ignore_errors=True)
print("\n" + ("RUNLOG PASS" if ok else "RUNLOG FAIL"))
sys.exit(0 if ok else 1)
