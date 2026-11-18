"""One log file per run, so a finished terminal session can still be read.

Both machines this project runs on lose their output otherwise: on the cluster
SLURM keeps only the job's own .out, and on the cloud PC a console window is
gone once it closes.  Every script calls start_log() and from then on writes to
the terminal and to Work/logs/ at the same time.
"""
from __future__ import annotations

import atexit
import datetime
import os
import platform
import socket
import sys
import time

import fno_paths

LOG_DIR = fno_paths.log_dir()


class _Tee:
    """Write to the terminal and to the log at once.

    The file is flushed on every write, so a job killed at its wall-clock
    limit still leaves everything it had printed.

    The log is opened as utf-8 and always gets the real text.  The terminal is
    the fragile one: a Windows console is cp1252, and an unencodable character
    is not raised on write but held in the wrapper's buffer and raised on
    flush -- at interpreter shutdown that becomes exit code 120 with the run's
    own output as the casualty.  start_log reconfigures the stream to replace
    such characters; the check here covers a stream that would not reconfigure.
    """

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh
        self._enc = getattr(stream, "encoding", None) or "utf-8"
        self._is_runlog_tee = True
        self._runlog_path = ""

    def write(self, text):
        try:
            self._fh.write(text)
            self._fh.flush()
        except (ValueError, OSError):
            pass
        try:
            self._stream.write(text if text.isascii() else self._fit(text))
        except (UnicodeEncodeError, ValueError, OSError):
            pass
        return len(text)

    def _fit(self, text):
        try:
            text.encode(self._enc)
            return text
        except UnicodeEncodeError:
            return text.encode(self._enc, "replace").decode(self._enc,
                                                            "replace")

    def flush(self):
        for stream in (self._stream, self._fh):
            try:
                stream.flush()
            except (ValueError, OSError):
                pass

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self):
        return self._stream.fileno()

    def writable(self):
        return True

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in str(name))


def _gpu_line():
    """Which card this ran on, without importing torch when it is not wanted."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return f"torch {torch.__version__}, no CUDA device"
        p = torch.cuda.get_device_properties(0)
        return (f"torch {torch.__version__}, {p.name}, "
                f"{p.total_memory / 1e9:.1f} GB, capability "
                f"{p.major}.{p.minor}")
    except Exception as exc:
        return f"torch {torch.__version__}, GPU query failed ({exc})"


def start_log(tag, log_dir=None):
    """Begin teeing stdout and stderr into Work/logs/<tag>_<when>_<host>.log
    (LOG_DIR, read now, unless ``log_dir`` is given).

    Returns the path, and prints it at both ends of the run so the file is easy
    to find afterwards.  Safe to call twice; the second call is ignored.
    """
    if getattr(sys.stdout, "_is_runlog_tee", False):
        return getattr(sys.stdout, "_runlog_path", None)

    log_dir = LOG_DIR if log_dir is None else log_dir
    os.makedirs(log_dir, exist_ok=True)
    started = time.time()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    host = _safe(socket.gethostname())[:24]
    job = os.environ.get("SLURM_JOB_ID")
    tail = f"j{job}" if job else f"p{os.getpid()}"
    path = os.path.join(log_dir, f"{tag}_{stamp}_{host}_{tail}.log")

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            pass

    fh = open(path, "w", encoding="utf-8", buffering=1)
    out = _Tee(sys.stdout, fh)
    err = _Tee(sys.stderr, fh)
    out._runlog_path = path
    sys.stdout, sys.stderr = out, err

    rows = [("run", tag),
            ("started", datetime.datetime.now().isoformat(timespec="seconds")),
            ("host", socket.gethostname()),
            ("platform", f"{platform.system()} {platform.release()} "
                         f"({platform.machine()})"),
            ("python", sys.version.split()[0]),
            ("cwd", os.getcwd()),
            ("command", " ".join(sys.argv))]
    if job:
        rows.append(("slurm", f"job {job} on "
                              f"{os.environ.get('SLURM_JOB_NODELIST', '?')}, "
                              f"partition "
                              f"{os.environ.get('SLURM_JOB_PARTITION', '?')}"))
    gpu = _gpu_line()
    if gpu:
        rows.append(("torch", gpu))

    bar = "=" * 72
    print(bar)
    for k, v in rows:
        print(f"  {k:<10} {v}")
    print(f"  {'log':<10} {path}")
    print(bar, flush=True)

    def finish():
        mins = (time.time() - started) / 60.0
        print(f"\n{bar}")
        print(f"  finished   "
              f"{datetime.datetime.now().isoformat(timespec='seconds')}")
        print(f"  elapsed    {mins:.1f} min ({mins / 60:.2f} h)")
        print(f"  log        {path}")
        print(bar, flush=True)
        sys.stdout, sys.stderr = out._stream, err._stream
        try:
            fh.close()
        except (ValueError, OSError):
            pass

    atexit.register(finish)
    return path
