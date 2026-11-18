"""The reference cache: atomic writes, damaged files, the batched solver."""
import contextlib
import errno
import io
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from _paths import ROOT, SCRATCH

os.chdir(SCRATCH)
import fno_predict as P
import fno_train as T
T.use_fixed_step()

ok = True


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


def talk(fn, *a, **k):
    """(what fn returned, what it printed)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


@contextlib.contextmanager
def patched(obj, **names):
    old = {k: getattr(obj, k) for k in names}
    for k, v in names.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def attempt(what, fn, *a, **k):
    """fn's result; None, after a FAIL line, if it raised."""
    try:
        return fn(*a, **k)
    except Exception as exc:
        check(False, f"{what}: raised {type(exc).__name__}: {exc}")
        return None


def fresh(name):
    d = os.path.join(SCRATCH, "gtb_" + name)
    shutil.rmtree(d, ignore_errors=True)
    return d


def same(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def read(path):
    with open(path, "rb") as fh:
        return fh.read()


real_integrate = T._integrate
real_savez = np.savez
solves = []


def spy_integrate(psi, *a, **k):
    solves.append(tuple(psi.shape))
    return real_integrate(psi, *a, **k)


@contextlib.contextmanager
def counting():
    """The shape of the initial condition of every solve inside the block."""
    solves.clear()
    with patched(T, _integrate=spy_integrate):
        yield solves


L, END = 16, 4.0
SEEDS = (11, 22, 33)


def single(seed, off, l=L, **kw):
    return T.ch_ground_truth(l, seed, off, END, T.DT_STEP, cache_dir="",
                             verbose=False, **kw)


def batch(seeds, off, cache_dir: str | None = "", l=L, **kw):
    return T.ch_ground_truth_batch(l, seeds, off, END, T.DT_STEP,
                                   cache_dir=cache_dir, verbose=False, **kw)


def cached(seed, off, d):
    return T.ch_ground_truth(L, seed, off, END, T.DT_STEP, cache_dir=d,
                             verbose=False)


def name_of(seed, off=0.0):
    return T.gt_cache_name(L, seed, off, END)


def old_integrate(psi, n_total, every, n_frames):
    """The solver as it was before it took a batch: axes 0 and 1."""
    def lap(q):
        return (np.roll(q, 1, 0) + np.roll(q, -1, 0)
                + np.roll(q, 1, 1) + np.roll(q, -1, 1) - 4.0 * q) / 1.0 ** 2
    frames = np.empty((n_frames,) + psi.shape, dtype=np.float64)
    frames[0] = psi
    j = 1
    for n in range(1, n_total + 1):
        psi = psi + lap(psi ** 3 - psi - lap(psi)) * T.DT
        if n % every == 0 and j < n_frames:
            frames[j] = psi
            j += 1
    return frames


print("1. a batch is each seed alone, bit for bit")
every, n_total, n_frames, _ = T._gt_grid(END, T.DT_STEP)
for off in (0.0, 0.2):
    alone = [single(s, off) for s in SEEDS]
    got = batch(SEEDS, off)
    check(len(got) == 3 and all(
        same(g[1], a[1]) and same(g[0], a[0]) for g, a in zip(got, alone)),
        f"l={L} off={off:+.1f}: frames and times of 3 seeds identical to "
        f"3 single solves")
    check(all(same(a[1], old_integrate(T.initial_condition(L, s, off),
                                       n_total, every, n_frames))
              for s, a in zip(SEEDS, alone)),
          f"l={L} off={off:+.1f}: a single seed is unchanged from the solver "
          f"before it took a batch")
    check(all(g[1].flags.c_contiguous and g[1].shape == (n_frames, L, L)
              for g in got), "each seed's frames are (n_frames, l, l), "
                             "contiguous")
for l in (15, 17):
    alone = [single(s, 0.2, l) for s in SEEDS]
    got = batch(SEEDS, 0.2, l=l)
    check(all(same(g[1], a[1]) for g, a in zip(got, alone)),
          f"odd l={l}: identical too (no vector remainder path differs)")
def old_device_integrate(psi, n_total, every, n_frames, device):
    """The device branch as it was: the whole frame buffer on the device."""
    lead = tuple(psi.shape[:-2])
    p = torch.as_tensor(psi, device=device, dtype=torch.float64)
    buf = torch.empty((*lead, n_frames, L16, L16), dtype=torch.float64,
                      device=device)
    buf[..., 0, :, :] = p
    j = 1
    for n in range(1, n_total + 1):
        p = T.ch_step(p)
        if n % every == 0 and j < n_frames:
            buf[..., j, :, :] = p
            j += 1
    return buf.cpu().numpy()


L16 = 16
devs = ["cpu:0"] + (["cuda"] if torch.cuda.is_available() else [])
for dev in devs:
    for ic in (T.initial_condition(L16, 3, 0.2),
               np.stack([T.initial_condition(L16, s, 0.2) for s in (3, 4)])):
        new = T._integrate(ic, 300, 100, 4, dev)
        old = old_device_integrate(ic, 300, 100, 4, dev)
        check(isinstance(new, np.ndarray) and new.dtype == old.dtype
              and new.shape == old.shape and np.array_equal(new, old),
              f"device={dev}, input {ic.shape}: frames equal the whole-buffer-"
              f"on-device solver, bit for bit (host float64 array)")

if torch.cuda.is_available():
    alone = [single(s, 0.2, device="cuda") for s in SEEDS]
    got = batch(SEEDS, 0.2, device="cuda")
    check(all(same(g[1], a[1]) and same(g[0], a[0])
              for g, a in zip(got, alone)),
          "on CUDA: identical to single solves on CUDA")
else:
    print("SKIP no CUDA device; the batch was checked on the CPU only")
alone = single(11, 0.0)
check(same(batch([11], 0.0)[0][1], alone[1]),
      "a batch of one is the single solve")

print("\n2. only the missing seeds are solved")
d = fresh("missing")
cached(22, 0.0, d)
with counting() as calls:
    got = batch([33, 22, 11], 0.0, d)
check(calls == [(2, L, L)],
      f"22 was on disk, so one solve of 2 seeds ran: {calls}")
check([same(g[1], single(s, 0.0)[1]) for g, s in zip(got, (33, 22, 11))]
      == [True] * 3, "and the list is in the order asked: 33, 22, 11")
with counting() as calls:
    batch([11, 22, 33], 0.0, d)
check(calls == [], "all three on disk: nothing is solved")
with counting() as calls:
    got = batch([5, 5, 6], 0.0, d)
check(calls == [(2, L, L)] and same(got[0][1], got[1][1]),
      f"a seed named twice is solved once: {calls}")
before = set(os.listdir(SCRATCH))
with counting() as calls:
    batch(SEEDS, 0.0, "")
    batch(SEEDS, 0.0, None)
check(calls == [(3, L, L)] * 2 and set(os.listdir(SCRATCH)) == before,
      "with the cache off every seed is solved and nothing is written")

print("\n3. the files")
d = fresh("files")
got = batch(SEEDS, 0.2, d)
want = sorted(name_of(s, 0.2) for s in SEEDS)
check(sorted(os.listdir(d)) == want,
      f"one file per seed and no partial left: {sorted(os.listdir(d))}")
check(name_of(11) == "gt_l16_seed11_off+0.00_t4_f64.npz"
      and T.gt_cache_name(256, 99999, -0.4, 2000.0)
      == "gt_l256_seed99999_off-0.40_t2000_f64.npz",
      "the file name pattern is the one the 159 GB cache already uses")
check(os.path.basename(P._gt_cache_path(L, 11, 0.2, END)) == name_of(11, 0.2),
      "fno_predict's mirror of the name agrees")
members = {"times", "frames", "device", "solver_dtype", "solver_dt",
           "torch_version", "created"}
good = True
for (times, frames), s in zip(got, SEEDS):
    with np.load(os.path.join(d, name_of(s, 0.2))) as z:
        good = (good and set(z.files) == members
                and same(z["times"], times) and same(z["frames"], frames)
                and same(frames, single(s, 0.2)[1])
                and str(z["device"]) == "cpu"
                and str(z["solver_dtype"]) == "float64"
                and float(z["solver_dt"]) == T.DT
                and str(z["torch_version"]) == torch.__version__
                and time.strptime(str(z["created"]), "%Y-%m-%dT%H:%M:%SZ")
                is not None)
check(good, "each loads back equal to what the batch returned and to a "
            "single solve, with exactly times, frames and the five "
            "provenance members")
t_one, f_one = cached(44, 0.0, d)
with np.load(os.path.join(d, name_of(44))) as z:
    check(set(z.files) == members and same(z["frames"], f_one),
          "a single solve writes the same members")
old_t, old_f = single(55, 0.0)
np.savez(os.path.join(d, name_of(55)), times=old_t, frames=old_f)
with counting() as calls:
    t_old, f_old = cached(55, 0.0, d)
check(calls == [] and same(f_old, old_f) and same(t_old, old_t),
      "a file in the old format (times and frames only) still hits")

print("\n4. a write that stops half way")


def dying(file, *a, **k):
    file.write(b"PK\x03\x04 half a zip")
    raise OSError(errno.ENOSPC, "No space left on device (test)")


d = fresh("interrupt")
seen_while_writing = []


def watch(file, *a, **k):
    seen_while_writing.append((os.path.basename(file.name),
                               sorted(os.listdir(d))))
    return real_savez(file, *a, **k)


with patched(np, savez=watch):
    cached(11, 0.0, d)
part, listing = seen_while_writing[0]
check(re.fullmatch(r"gt_l16_seed11_off\+0\.00_t4_f64\.[\w.-]+-\d+\.partial"
                   r"\.npz", part) is not None
      and part.endswith(f"-{os.getpid()}.partial.npz"),
      f"the write goes to <name>.<host>-<pid>.partial.npz: {part}")
check(name_of(11) not in listing and sorted(os.listdir(d)) == [name_of(11)],
      "which is renamed over the final name only when finished")

d = fresh("interrupt")
raised = None
with patched(np, savez=dying), counting():
    try:
        cached(11, 0.0, d)
    except OSError as exc:
        raised = exc
check(raised is not None and raised.errno == errno.ENOSPC,
      "a full disk during the write is raised, as before")
check(os.listdir(d) == [], "and leaves nothing: no final name, no partial")

# A killed job cannot clean up, so make the cleanup a no-op as well.
with patched(np, savez=dying), patched(os, remove=lambda p: None):
    try:
        cached(11, 0.0, d)
    except OSError:
        pass
left = sorted(os.listdir(d))
check(len(left) == 1 and left[0].endswith(".partial.npz")
      and name_of(11) not in left,
      f"killed mid-write: only the partial is left, {left}")
with patched(P, GT_CACHE=d):
    check(not os.path.exists(P._gt_cache_path(L, 11, 0.0, END)),
          "so the reference does not look cached")
# the partial of another job: this process would reuse its own name
partial = os.path.join(d, name_of(11)[:-4] + ".elsewhere-1.partial.npz")
os.replace(os.path.join(d, left[0]), partial)
left = [os.path.basename(partial)]
stale = read(partial)
with counting() as calls:
    t_new, f_new = cached(11, 0.0, d)
check(calls == [(L, L)] and same(f_new, single(11, 0.0)[1]),
      "the next run solves it rather than loading the partial")
check(sorted(os.listdir(d)) == sorted(left + [name_of(11)])
      and read(partial) == stale,
      "and leaves the partial as it was: not renamed, not deleted")
with counting() as calls:
    got = batch([11, 22], 0.0, d)
check(calls == [(1, L, L)] and same(got[1][1], single(22, 0.0)[1]),
      "the batch treats the partial alike")

d = fresh("interrupt_batch")
n_saved = [0]


def dies_second(file, *a, **k):
    n_saved[0] += 1
    if n_saved[0] == 2:
        return dying(file)
    return real_savez(file, *a, **k)


try:
    with patched(np, savez=dies_second):
        batch(SEEDS, 0.0, d)
except OSError:
    pass
check(sorted(os.listdir(d)) == [name_of(11)]
      and same(np.load(os.path.join(d, name_of(11)))["frames"],
               single(11, 0.0)[1]),
      "a batch that dies on its second file keeps the first, whole, and "
      "leaves no trace of the second or third")
with counting() as calls:
    batch(SEEDS, 0.0, d)
check(calls == [(2, L, L)] and len(os.listdir(d)) == 3,
      "and the rerun solves just those two")

d = fresh("rename_fails")


def no_rename(src, dst):
    raise PermissionError(errno.EACCES, "in use (test)")


raised = None
buf = io.StringIO()
with patched(os, replace=no_rename), contextlib.redirect_stdout(buf):
    try:
        cached(11, 0.0, d)
    except PermissionError as exc:
        raised = exc
left = os.listdir(d)
check(raised is not None and len(left) == 1 and name_of(11) not in left
      and "left as" in buf.getvalue(),
      "a rename that fails is raised, keeps the finished partial and says "
      "where it is")
with np.load(os.path.join(d, left[0])) as z:
    check(same(z["frames"], single(11, 0.0)[1]),
          "which holds the whole solve")

d = fresh("killed")
code = (
    "import os, sys\n"
    "sys.path.insert(0, sys.argv[2])\n"
    "import numpy as np\n"
    "import fno_train as T\n"
    "T.use_fixed_step()\n"
    "def die(file, *a, **k):\n"
    "    file.write(b'PK\\x03\\x04 half a zip')\n"
    "    file.flush()\n"
    "    os._exit(9)\n"
    "setattr(np, 'savez', die)\n"
    "T.ch_ground_truth(16, 11, 0.0, 4.0, T.DT_STEP, cache_dir=sys.argv[1], "
    "verbose=False)\n")
run = subprocess.run([sys.executable, "-c", code, d, ROOT], cwd=SCRATCH,
                     capture_output=True, text=True)
left = sorted(os.listdir(d)) if os.path.isdir(d) else []
check(run.returncode == 9 and len(left) == 1 and name_of(11) not in left
      and left[0].endswith(".partial.npz"),
      f"a process killed outright leaves no final name: {left}")
if run.returncode != 9:
    print(run.stderr[-400:])
with counting() as calls:
    t_new, f_new = cached(11, 0.0, d)
check(calls == [(L, L)] and same(f_new, single(11, 0.0)[1]),
      "and the next process solves it")

print("\n5. a damaged file is moved aside and solved again")
ref = single(11, 0.0)
d0 = fresh("good_source")
cached(11, 0.0, d0)
good_bytes = read(os.path.join(d0, name_of(11)))


def npz_bytes(**arrays):
    buf = io.BytesIO()
    real_savez(buf, **arrays)
    return buf.getvalue()


damage = {
    "truncated": good_bytes[:len(good_bytes) // 2],
    "empty": b"",
    "not a zip": b"this was never an npz file\n" * 20,
    "no frames": npz_bytes(times=ref[0]),
    "no times": npz_bytes(frames=ref[1]),
    "flipped bytes": (good_bytes[:200] + bytes(b ^ 0xFF for b in
                                               good_bytes[200:1200])
                      + good_bytes[1200:]),
}
for what, blob in damage.items():
    d = fresh("damaged")
    os.makedirs(d)
    path = os.path.join(d, name_of(11))
    with open(path, "wb") as fh:
        fh.write(blob)
    with counting() as calls:
        out = attempt(what, talk, cached, 11, 0.0, d)
    if out is None:
        continue
    (t_got, f_got), said = out
    bad = path + ".corrupt"
    check(same(f_got, ref[1]) and same(t_got, ref[0]) and len(calls) == 1
          and os.path.exists(bad) and read(bad) == blob,
          f"{what}: moved to .corrupt with its bytes, and solved again")
    check(len(said.strip().splitlines()) == 1 and ".corrupt" in said,
          f"{what}: one line printed")
    with np.load(path) as z:
        check(same(z["frames"], ref[1]) and "created" in z.files,
              f"{what}: the new file is whole")

d = fresh("damaged_twice")
os.makedirs(d)
path = os.path.join(d, name_of(11))
with open(path + ".corrupt", "wb") as fh:
    fh.write(b"an earlier casualty")
with open(path, "wb") as fh:
    fh.write(b"damaged again")
attempt("a second casualty", talk, cached, 11, 0.0, d)
extra = [f for f in os.listdir(d) if f.startswith(name_of(11) + ".corrupt.")]
check(read(path + ".corrupt") == b"an earlier casualty" and len(extra) == 1
      and re.fullmatch(r"\.corrupt\.\d+", extra[0][len(name_of(11)):])
      and read(os.path.join(d, extra[0])) == b"damaged again",
      "when .corrupt is taken the new casualty gets .corrupt.<time_ns>, and "
      "the old one is untouched")

d = fresh("damaged_batch")
cached(11, 0.0, d)
with open(os.path.join(d, name_of(22)), "wb") as fh:
    fh.write(good_bytes[:300])
with counting() as calls:
    out = attempt("a batch with a damaged file", talk,
                  T.ch_ground_truth_batch, L, [11, 22, 33], 0.0, END,
                  T.DT_STEP, cache_dir=d, verbose=True)
got = out[0] if out else []
check(calls == [(2, L, L)] and os.path.exists(
    os.path.join(d, name_of(22) + ".corrupt")),
    "a batch moves the damaged one aside and solves it with the missing one")
check(len(got) == 3 and all(same(g[1], single(s, 0.0)[1])
                            for g, s in zip(got, (11, 22, 33))),
      "and all three come back right")


def boom(exc):
    def load(*a, **k):
        raise exc
    return load


d = fresh("machine")
cached(11, 0.0, d)
path = os.path.join(d, name_of(11))
for exc, is_damage in ((MemoryError(), False),
                       (OSError(errno.EIO, "I/O error"), False),
                       (OSError(errno.EINVAL, "invalid argument"), True),
                       (EOFError("No data left in file"), True),
                       (ValueError("cannot reshape"), True)):
    blob = read(path)
    raised = False
    with patched(np, load=boom(exc)):
        try:
            talk(cached, 11, 0.0, d)
        except (MemoryError, OSError) as e:
            raised = e is exc
        except Exception:
            pass
    moved = os.path.exists(path + ".corrupt")
    kind = type(exc).__name__ + (f" errno {exc.errno}"
                                 if isinstance(exc, OSError) else "")
    if is_damage:
        check(not raised and moved, f"{kind} while reading: file moved aside")
        os.replace(path + ".corrupt", path)
    else:
        check(raised and not moved and read(path) == blob,
              f"{kind} while reading: raised, file left in place")

print("\n6. the off-critical training cache")
d = fresh("offcrit")
off_name = "offcrit_l8_n3_x2_+0.20.npz"
with patched(T, GT_CACHE=d, OFF_CRITICAL_OFFS=(0.2,), OFF_CRITICAL_TRAJ=2):
    with counting() as calls:
        first, _ = talk(T.off_critical_trajectories, 8, 3, "cpu", False)
    check(calls == [(8, 8)] * 2 and os.listdir(d) == [off_name]
          and first.shape == (2, 3, 8, 8),
          "solved once and written under the usual name, no partial left")
    with np.load(os.path.join(d, off_name)) as z:
        check(set(z.files) == {"frames", "device", "solver_dtype",
                               "solver_dt", "torch_version", "created"}
              and z["frames"].dtype == np.float32,
              "with its frames and the provenance members")
    with counting() as calls:
        again, _ = talk(T.off_critical_trajectories, 8, 3, "cpu", False)
    check(calls == [] and same(again, first), "a second call reuses it")

    path = os.path.join(d, off_name)
    blob = read(path)[:500]
    with open(path, "wb") as fh:
        fh.write(blob)
    with counting() as calls:
        redo, said = talk(T.off_critical_trajectories, 8, 3, "cpu", True)
    check(same(redo, first) and len(calls) == 2
          and read(path + ".corrupt") == blob and "damaged" in said,
          "a damaged one is moved aside to .corrupt and solved again")

    d2 = fresh("offcrit_interrupt")
    T.GT_CACHE = d2
    try:
        with patched(np, savez=dying):
            talk(T.off_critical_trajectories, 8, 3, "cpu", False)
    except OSError:
        pass
    check(os.listdir(d2) == [], "a full disk mid-write leaves nothing")
    try:
        with patched(np, savez=dying), patched(os, remove=lambda p: None):
            talk(T.off_critical_trajectories, 8, 3, "cpu", False)
    except OSError:
        pass
    left = os.listdir(d2)
    check(len(left) == 1 and off_name not in left
          and left[0].endswith(".partial.npz"),
          f"a kill leaves only the partial: {left}")
    with counting() as calls:
        talk(T.off_critical_trajectories, 8, 3, "cpu", False)
    check(len(calls) == 2 and off_name in os.listdir(d2),
          "and the next call solves it")

print("\n7. the seed engine solves its references in batches")
check(P.GT_BATCH == 16, f"GT_BATCH defaults to 16, got {P.GT_BATCH}")
saved_batch = P.GT_BATCH
args = P.parse_args([])
check(args.gt_batch is None, "--gt-batch is off unless given")
moved, said = talk(P.apply_overrides, P.parse_args(["--gt-batch", "4"]))
check(P.GT_BATCH == 4 and moved == ["GT_BATCH 16 -> 4"]
      and "GT_BATCH 16 -> 4" in said, "--gt-batch 4 sets GT_BATCH and says so")
for bad in ("0", "-3"):
    try:
        talk(P.apply_overrides, P.parse_args([f"--gt-batch={bad}"]))
        died = False
    except SystemExit as exc:
        died = "at least 1" in str(exc)
    check(died and P.GT_BATCH == 4,
          f"--gt-batch={bad} is refused before anything is solved")
P.GT_BATCH = saved_batch
with patched(P, END_TIME=2000.0):
    sizes = [P.gt_batch_size(lat) for lat in (16, 128, 256)]
    with patched(P, GT_BATCH_GB=1e-6):
        tiny = P.gt_batch_size(16)
check(sizes == [16, 15, 3] and tiny == 1,
      f"a batch is cut to fit GT_BATCH_GB: {sizes} at l = 16, 128, 256 to "
      f"t = 2000, and never below one seed ({tiny})")

GT = fresh("engine_gt")
RC = fresh("engine_rc")
MODEL = os.path.join(SCRATCH, "gtb_model.pt")
with open(MODEL, "wb") as fh:
    fh.write(b"not a model; only its size and date are read")
batches, rolled, wrote = [], [], []
real_batch = P.ch_ground_truth_batch


def spy_batch(l, seeds, *a, **k):
    batches.append({"l": l, "seeds": list(seeds), "t_end": a[1],
                    "cache_dir": k.get("cache_dir")})
    return real_batch(l, seeds, *a, **k)


def fake_eval(model, seed, device, off=0.0, lattice=None):
    lat = P.LATTICE if lattice is None else lattice
    present = os.path.exists(P._gt_cache_path(lat, seed, off, P.END_TIME))
    T.ch_ground_truth(lat, seed, off, P.END_TIME, T.DT_STEP,
                      cache_dir=P.GT_CACHE, device=device, verbose=False)
    rolled.append((seed, present))
    return {"r2": np.ones(2), "t_stop": P.END_TIME, "stopped_early": False,
            "diverged_at": None, "gt_cached": present, "gt_seconds": 0.0}


def fake_cache_rollout(cache, res):
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as fh:
        fh.write(b"x")


def fake_load(cache):
    if P.CACHE_RESULTS and os.path.exists(cache):
        return {"r2": np.ones(2), "t_stop": P.END_TIME,
                "stopped_early": False, "diverged_at": None,
                "gt_cached": True, "gt_seconds": 0.0}
    return None


def fake_outputs(results, seeds, off, lattice=None, key=None):
    wrote.append(list(results))
    return {}


def cli(*argv):
    """The flags applied as a run would, at the file's own horizon."""
    with patched(P, END_TIME=2000.0, START_TIME=100.0):
        return talk(P.apply_overrides, P.parse_args(list(argv)))[0]


def case(seeds, off=0.0):
    batches.clear()
    rolled.clear()
    wrote.clear()
    with counting():
        talk(P.run_case, None, "cpu", off, lattice=L, seeds=seeds)
    return list(solves)


P.MODEL_FLAGS.update(hard_constraints=True, hard_mass_conservation=True,
                     zero_dc=True, label="fno_ch", note="")
with patched(P, GT_CACHE=GT, RESULT_CACHE=RC, MODEL_PATH=MODEL, ENGINE="seed",
             END_TIME=END, START_TIME=1.0, CACHE_RESULTS=True, GT_BATCH=2,
             evaluate_seed=fake_eval, write_case_outputs=fake_outputs,
             _load_cached=fake_load, _cache_rollout=fake_cache_rollout,
             ch_ground_truth_batch=spy_batch):
    seeds = (101, 102, 103, 104, 105)
    ran = case(seeds)
    check([b["seeds"] for b in batches] == [[101, 102], [103, 104], [105]],
          f"5 missing references, GT_BATCH = 2: one batch call per chunk, "
          f"{[b['seeds'] for b in batches]}")
    check(all(b["l"] == L and b["t_end"] == END and b["cache_dir"] == GT
              for b in batches),
          "each with the case's lattice, END_TIME and GT_CACHE passed "
          "explicitly")
    check(ran == [(2, L, L), (2, L, L), (1, L, L)],
          f"and those were the only solves: {ran}")
    check(rolled == [(s, True) for s in seeds],
          "every seed found its reference cached when it was rolled out")
    check(len(wrote) == 1 and [r["gt_cached"] for r in wrote[0]] == [False] * 5
          and all(r["gt_seconds"] > 0 for r in wrote[0]),
          "and their timing says solved, not read from disk")

    shutil.rmtree(RC)
    ran = case(seeds)
    check(batches == [] and ran == [] and len(rolled) == 5,
          "references all there: no batch, no solve")

    shutil.rmtree(RC)
    for s in (101, 102):
        os.remove(os.path.join(GT, name_of(s)))
    os.makedirs(RC)
    fake_cache_rollout(P._result_path(0.0, 101, L), None)
    ran = case(seeds)
    check([b["seeds"] for b in batches] == [[102]]
          and [s for s, _ in rolled] == [102, 103, 104, 105],
          "a seed whose rollout is cached needs no reference: only 102 was "
          "solved")

    batches.clear()
    with patched(P, ENGINE="lockstep"):
        check(P._solve_missing_references("cpu", 0.0, L, seeds) == {}
              and batches == [],
              "the lock-step engine solves nothing in advance")

    shutil.rmtree(GT)
    shutil.rmtree(RC)
    cli("--gt-batch", "3")
    seeds = tuple(range(201, 208))
    ran = case(seeds)
    check([len(b["seeds"]) for b in batches] == [3, 3, 1],
          f"--gt-batch 3 is honoured: chunks of "
          f"{[len(b['seeds']) for b in batches]} for 7 seeds")

    P.GT_BATCH = 100
    shutil.rmtree(GT)
    shutil.rmtree(RC)
    case(seeds)
    check([len(b["seeds"]) for b in batches] == [7],
          "a GT_BATCH larger than the case is one call")

    gt_off = fresh("engine_gt_off")
    P.GT_CACHE = gt_off
    P.CACHE_RESULTS = False
    cli("--no-gt-cache")
    before = sorted(os.listdir(SCRATCH))
    ran = case(seeds[:3])
    check(P.GT_CACHE == "" and batches == [] and ran == [(L, L)] * 3,
          f"--no-gt-cache: no batch, each reference solved when its seed "
          f"comes up: {ran}")
    check(sorted(os.listdir(SCRATCH)) == before and not os.path.exists(gt_off)
          and rolled == [(s, False) for s in seeds[:3]],
          "and nothing is written")
P.GT_BATCH = saved_batch

for d in [f for f in os.listdir(SCRATCH) if f.startswith("gtb_")]:
    shutil.rmtree(os.path.join(SCRATCH, d), ignore_errors=True)
print("\n" + ("GTBATCH PASS" if ok else "GTBATCH FAIL"))
sys.exit(0 if ok else 1)
