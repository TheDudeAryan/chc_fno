"""References for the variable-time-step runs: the group's cache, ported as it is.

The training validation and the prediction compare a rollout with the exact
solver at the times of a schedule, so what they read is a trajectory saved
every 4.0 time units from t = 100 to 10000 (2476 frames), one file per
(lattice, quench, seed).  The file names and the bytes are the group
format's, so a cache made by either code serves the other:

    gt_l128_off+0.0_seed31337_t100-10000_ev4.npz

members ``times`` (float64), ``fields`` (float32, (n, l, l), stored
uncompressed so it can be memory-mapped), and ``seed, off, l, device,
solver_dtype, solver_dt, torch_version``.  A frame at time t is the state
after round(t / 0.01) float64 Euler steps from the random start, the same
update as ch.py, ch_gpu.py and fno_train.ch_step.

The folder is Work/gt_cache, or the environment variable FNO_GT_CACHE
(fno_paths.ref_cache_dir): the 15-case set of a prediction is about 860 GB
in 800 files.
The older 1.0-spaced float64 files of fno_train.ch_ground_truth have other
names and are not touched.

Also here: make_schedule, the times a variable-step rollout visits, which
fno_train (validation) and fno_predict share.
"""

from __future__ import annotations

import os
import socket
import struct
import time
import zipfile

import numpy as np
import torch
from numpy.lib import format as npformat

SOLVER_DT = 0.01
# the reference grid: from t = 100, every 4.0 time units
GT_T0 = 100.0
GT_EVERY = 4.0
# "auto": a solve inside a SLURM job must run on CUDA (a CPU solve would
# take hours); True / False force it
GT_REQUIRE_CUDA: str | bool = "auto"
# the lattices (cells) one batched solve holds on the device, and the most
# files it keeps open
GT_BATCH_CELLS = 1.6e7
GT_MAX_OPEN = 128

# The group protocol's validation seeds (also its tuning seeds)
VAL_SEEDS = (31337, 24601, 8675309, 20250, 42424,
             90210, 23571, 55555, 48151, 62831)


def make_ic(l, seed, off=0.0):
    """Reproduces PhaseOrdering.set_ic from ch.py."""
    np.random.seed(seed)
    ic = np.random.uniform(-0.1, 0.1, size=(l, l))
    return ic - ic.mean() + off


def _del_sq_t(psi):
    return (torch.roll(psi, 1, -2) + torch.roll(psi, -1, -2)
            + torch.roll(psi, 1, -1) + torch.roll(psi, -1, -1) - 4.0 * psi)


@torch.no_grad()
def integrate(ic, t_end, save_times, dt=SOLVER_DT,
              device: str | torch.device = "cuda", sink=None):
    """Explicit Euler in float64, operation for operation as in ch.py.

    Returns (times, frames) with the frames cast to float32.  With sink, each
    saved frame goes to sink(j, frame) as soon as it is computed instead, and
    frames is None, so host memory holds one frame rather than all of them.
    """
    dev = torch.device(device)
    if dev.type == "mps":
        dev = torch.device("cpu")
    psi = torch.as_tensor(np.ascontiguousarray(ic), dtype=torch.float64,
                          device=dev)
    save = {int(round(t / dt)): i for i, t in enumerate(save_times)}
    out = None
    if sink is None:
        out = np.empty((len(save_times),) + tuple(psi.shape), dtype=np.float32)
    if 0 in save:
        if out is not None:
            out[save[0]] = psi.cpu().numpy()
        elif sink is not None:
            sink(save[0], psi.cpu().numpy().astype(np.float32))
    for step in range(1, int(round(t_end / dt)) + 1):
        mu = psi**3 - psi - _del_sq_t(psi)
        psi = psi + _del_sq_t(mu) * dt
        j = save.get(step)
        if j is not None:
            frame = psi.cpu().numpy().astype(np.float32)
            if out is not None:
                out[j] = frame
            elif sink is not None:
                sink(j, frame)
    return np.asarray(save_times, dtype=np.float64), out


def make_schedule(t0, t1, alpha, dt_min, dt_max, quantum=1.0, must_hit=()):
    """Times visited by an autoregressive rollout, stride ~ alpha * t.

    The stride is alpha * t clipped to [dt_min, dt_max] and rounded to a
    multiple of ``quantum`` (the reference spacing, so every time is a
    reference frame); times in ``must_hit`` between t0 and t1 are added.
    """
    ts, t = [float(t0)], float(t0)
    while t < t1 - 1e-9:
        dt = min(max(alpha * t, dt_min), dt_max)
        dt = max(quantum, round(dt / quantum) * quantum)
        t = min(t1, t + dt)
        ts.append(round(t, 6))
    ts += [float(m) for m in must_hit if t0 < m < t1]
    return np.array(sorted(set(ts)))


def gt_path(cache_dir, l, off, seed, t0, t1, save_every):
    return os.path.join(cache_dir, f"gt_l{l}_off{off:+.1f}_seed{seed}"
                                   f"_t{t0:g}-{t1:g}_ev{save_every:g}.npz")


def gt_times(t0, t1, save_every):
    """The frame times of a ground-truth file, exactly as it stores them."""
    return np.arange(t0, t1 + 0.5 * save_every, save_every).round(6)


def gt_file_bytes(l, t0, t1, save_every):
    """Bytes of one cache file: the float32 fields and the float64 times,
    uncompressed, plus about 2 KB for the small members and the headers."""
    n = len(gt_times(t0, t1, save_every))
    return n * l * l * 4 + n * 8 + 2048


def find_gt(cache_dir, l, off, seed, t0, t1, save_every):
    """Path of a cached file that holds this trajectory, or None.

    The exact name wins.  Otherwise any file with the same (l, off, seed, t0,
    save_every) and a longer t1 serves, the shortest such: its first frames
    are the same solve, bit for bit.  load_gt(path, t1) slices it.
    """
    path = gt_path(cache_dir, l, off, seed, t0, t1, save_every)
    if os.path.exists(path):
        return path
    k = (t1 - t0) / save_every
    if abs(k - round(k)) > 1e-6 or not os.path.isdir(cache_dir):
        return None
    pre = f"gt_l{l}_off{off:+.1f}_seed{seed}_t{t0:g}-"
    suf = f"_ev{save_every:g}.npz"
    best = None
    for name in os.listdir(cache_dir):
        if not (name.startswith(pre) and name.endswith(suf)):
            continue
        try:
            t_long = float(name[len(pre):-len(suf)])
        except ValueError:
            continue
        if t_long >= t1 - 1e-9 and (best is None or t_long < best[0]):
            best = (t_long, name)
    return None if best is None else os.path.join(cache_dir, best[1])


def _stored_npy_offset(path, member):
    """Byte offset of an uncompressed .npz member's .npy data, else None."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            return None
        start = info.header_offset
    with open(path, "rb") as f:
        f.seek(start)
        head = f.read(30)
        if head[:4] != b"PK\x03\x04":
            return None
        n_name, n_extra = struct.unpack("<HH", head[26:30])
        f.seek(start + 30 + n_name + n_extra)
        major, _ = npformat.read_magic(f)
        if major == 1:
            shape, fortran, dtype = npformat.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = npformat.read_array_header_2_0(f)
        return f.tell(), tuple(shape), fortran, dtype


def load_gt(path, t1=None):
    """(times, fields) of a ground-truth file; fields is a read-only memmap.

    Nothing but the frames that get indexed is read from disk.  With t1 the
    arrays stop at times <= t1 (a longer cached solve serves a shorter one).
    A compressed member falls back to np.load.
    """
    with np.load(path) as z:
        times = np.asarray(z["times"])
    loc = _stored_npy_offset(path, "fields.npy")
    if loc is None or loc[2]:
        with np.load(path) as z:
            fields = z["fields"]
    else:
        start, shape, _, dtype = loc
        fields = np.memmap(path, dtype=dtype, mode="r", offset=start,
                           shape=shape)
    if t1 is not None:
        n = int(np.count_nonzero(times <= t1 + 1e-6))
        times, fields = times[:n], fields[:n]
    return times, fields


def gt_provenance(path):
    """Solver metadata a ground-truth file records; None where it predates it."""
    keys = ("device", "solver_dtype", "solver_dt", "torch_version")
    with np.load(path) as z:
        return {k: (z[k].item() if k in z.files else None) for k in keys}


_GT_CPU_NOTED = [False]


def gt_require(require=None):
    """Whether ground truth must be solved on CUDA; 'auto' = inside Slurm."""
    if require is None:
        require = GT_REQUIRE_CUDA
    if isinstance(require, str) and require.lower() == "auto":
        return bool(os.environ.get("SLURM_JOB_ID"))
    return bool(require)


def gt_device(device, require=None):
    """The device ground truth really runs on (MPS falls back to the CPU).

    When CUDA is required (see gt_require) and not available, stop instead of
    spending hours on a silent CPU solve.
    """
    dev = torch.device(device)
    if dev.type == "mps":
        dev = torch.device("cpu")
    if dev.type == "cuda" and torch.cuda.is_available():
        return dev
    if gt_require(require):
        raise SystemExit(
            f"[gt] ground truth must be solved on CUDA here "
            f"(GT_REQUIRE_CUDA={require or GT_REQUIRE_CUDA!r}, "
            f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')!r}) but the device "
            f"is {device} and torch.cuda.is_available() is "
            f"{torch.cuda.is_available()}. Request a GPU node, or set "
            f"fno_gt.GT_REQUIRE_CUDA = False to allow a CPU solve.")
    if not _GT_CPU_NOTED[0]:
        print("[gt] solving ground truth on the CPU (float64)", flush=True)
        _GT_CPU_NOTED[0] = True
    return torch.device("cpu")


def device_label(dev):
    """'cuda:0 NVIDIA A100-SXM4-40GB', or 'cpu'."""
    dev = torch.device(dev)
    if dev.type != "cuda":
        return dev.type
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    return f"cuda:{idx} {torch.cuda.get_device_name(idx)}"


class _GTWriter:
    """One ground-truth .npz written frame by frame, then renamed into place.

    The zip is uncompressed (ZIP_STORED) with the members np.savez writes, so
    np.load and load_gt read it like any other cached file.  The temporary
    name carries the host and pid: two jobs solving the same file write
    separate partials, and the last rename installs identical bytes.
    """

    def __init__(self, path, times, shape):
        tag = f"{socket.gethostname()}-{os.getpid()}"
        self.path, self.tmp = path, f"{path[:-4]}.{tag}.partial.npz"
        self.n, self.shape, self.done = len(times), tuple(shape), 0
        self.zf = zipfile.ZipFile(self.tmp, "w", zipfile.ZIP_STORED,
                                  allowZip64=True)
        self._put("times", np.asarray(times, dtype=np.float64))
        self.fh = self.zf.open("fields.npy", "w", force_zip64=True)
        npformat.write_array_header_1_0(self.fh, dict(
            descr=npformat.dtype_to_descr(np.dtype(np.float32)),
            fortran_order=False, shape=(self.n,) + self.shape))

    def _put(self, name, value):
        with self.zf.open(f"{name}.npy", "w", force_zip64=True) as f:
            npformat.write_array(f, np.asanyarray(value), allow_pickle=False)

    def frame(self, j, arr):
        assert j == self.done and arr.shape == self.shape, (j, self.done)
        self.fh.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        self.done += 1

    def close(self, **small):
        assert self.done == self.n, (self.path, self.done, self.n)
        self.fh.close()
        for k, v in small.items():
            self._put(k, v)
        self.zf.close()
        os.replace(self.tmp, self.path)

    def abort(self):
        try:
            self.fh.close()
            self.zf.close()
        except Exception:
            pass
        try:
            os.remove(self.tmp)
        except OSError:
            pass


def ground_truth(cache_dir, l, off, seed, t0, t1, save_every, dt, device,
                 quiet=False, require_cuda=None):
    """Exact trajectory, cached to .npz and reused on later runs.

    Returns the path of a file that holds it, possibly a longer solve of the
    same seed: read it with load_gt(path, t1).
    """
    path = find_gt(cache_dir, l, off, seed, t0, t1, save_every)
    if path is not None:
        if not quiet:
            print(f"[data] ground truth cached: {os.path.basename(path)}",
                  flush=True)
        return path
    ground_truth_batch(cache_dir, l, off, [seed], t0, t1, save_every, dt,
                       device, require_cuda=require_cuda)
    path = find_gt(cache_dir, l, off, seed, t0, t1, save_every)
    return path or gt_path(cache_dir, l, off, seed, t0, t1, save_every)


def ground_truth_batch(cache_dir, l, off, seeds, t0, t1, save_every, dt,
                       device, cells=GT_BATCH_CELLS, max_open=GT_MAX_OPEN,
                       require_cuda=None):
    """Solve every uncached seed together; each file matches ground_truth().

    The Laplacian acts on the last two axes, so a batch of trajectories is
    integrated elementwise exactly as each would be alone: the stored fields
    are bit-identical to one-at-a-time solves.  Frames stream straight into
    per-seed files, so host memory holds one frame per seed and a batch is
    capped only by cells (l*l per seed, the device-side state) and by
    max_open files.  A seed already held by a longer cached solve counts as
    present.  Returns the number of trajectories solved.
    """
    todo: list[int] = []
    for s in seeds:
        s = int(s)
        if s not in todo and find_gt(cache_dir, l, off, s, t0, t1,
                                     save_every) is None:
            todo.append(s)
    if not todo:
        return 0
    os.makedirs(cache_dir, exist_ok=True)
    dev = gt_device(device, require_cuda)
    label = device_label(dev)
    times = gt_times(t0, t1, save_every)
    steps = int(round(t1 / dt))
    per = max(1, min(int(cells // (l * l)), int(max_open)))
    n_batch = -(-len(todo) // per)
    print(f"[gt] solving {len(todo)} ground-truth trajectories at l={l} "
          f"off={off:+.1f} to t={t1:g} ({len(times)} frames, every "
          f"{save_every:g}) on {label}, {n_batch} batch(es) of <= {per}",
          flush=True)
    small = dict(off=off, l=l, device=label, solver_dtype="float64",
                 solver_dt=float(dt), torch_version=torch.__version__)
    for k in range(n_batch):
        chunk = [s for s in todo[k * per:(k + 1) * per]
                 if find_gt(cache_dir, l, off, s, t0, t1, save_every) is None]
        if not chunk:
            print(f"[gt]   batch {k + 1}/{n_batch}: solved meanwhile by "
                  f"another job, skipped", flush=True)
            continue
        t_wall = time.time()
        ws = [_GTWriter(gt_path(cache_dir, l, off, s, t0, t1, save_every),
                        times, (l, l)) for s in chunk]

        def sink(j, frame, ws=ws):
            for b, w in enumerate(ws):
                w.frame(j, frame[b])

        try:
            ic = np.stack([make_ic(l, s, off) for s in chunk])
            integrate(ic, t1, list(times), dt=dt, device=dev, sink=sink)
            for w, s in zip(ws, chunk):
                w.close(seed=s, **small)
        except BaseException:
            for w in ws:
                w.abort()
            raise
        print(f"[gt]   batch {k + 1}/{n_batch} on {label}: {len(chunk)} "
              f"seeds {chunk}, {steps:,} steps, {time.time() - t_wall:.0f}s",
              flush=True)
    return len(todo)
