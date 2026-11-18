from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import fno_paths
import fno_runlog

L = 64
DT = 0.01
START = 0
END = 500
N_AHEAD = 100
GAP = 10
NENS = 25
OFF_VALUES = [-0.4, -0.2, 0.0, 0.2, 0.4]
FNAME = os.path.join(fno_paths.data_dir(), "test.npz")


def npz_path(end):
    """Work/data/test_t<END:04d>.npz, the name the group protocol gives the
    dataset to ``end``: test_t0500.npz for END = 500."""
    return os.path.join(fno_paths.data_dir(),
                        f"test_t{fno_paths.time_token(end, 4)}.npz")
WRITE_LOG = True

BATCH = 25
DX = 1

LEAN_FORMAT = "ch_lean_v1"


def set_ic(l, seed, off, dtype=np.float64):
    """ch.py's PhaseOrdering.set_ic, verbatim, so the RNG stream is identical."""
    np.random.seed(seed)
    ic = np.random.uniform(-0.1, 0.1, size=(l, l)).astype(dtype)
    return ic - ic.mean() + off


def plan(nens=NENS, off_values=OFF_VALUES):
    """ch.py's seed assignment: which (seed, off) pairs, in which order."""
    np.random.seed(42)
    train_ens = int(0.8 * nens)
    train_seeds = np.random.randint(0, 10000, train_ens)
    val_seeds = np.random.randint(10000, 20000, nens - train_ens)

    def pairs(seeds):
        out = []
        for off, group in zip(off_values,
                              np.array_split(seeds, len(off_values))):
            out.extend((int(s), float(off)) for s in group)
        return out

    return pairs(train_seeds), pairs(val_seeds)


def _del_sq(psi):
    """The 5-point periodic Laplacian, on the last two axes so it batches."""
    return (torch.roll(psi, 1, dims=-2) + torch.roll(psi, -1, dims=-2)
            + torch.roll(psi, 1, dims=-1) + torch.roll(psi, -1, dims=-1)
            - 4 * psi) / DX ** 2


def integrate(psi, n_steps, gap, progress=None, sink=None):
    """Explicit-Euler Cahn-Hilliard, keeping every ``gap``-th frame.

    ch.py allocates every one of the ``n_steps`` frames and then throws away
    all but one in ``gap``.  prepare_features_labels only ever reads indices
    that are multiples of ``gap``, so only those are kept here -- at END = 500
    that is 5,000 frames of 164 MB rather than 50,000 of 1.64 GB, per
    trajectory.  Frame j of the result is ch.py's data[j * gap].

    With ``sink`` each kept frame is handed to sink(j, frame) instead of being
    collected, and nothing is returned -- write_lean streams a dataset to disk
    that way, so a long horizon needs no more memory than a short one.  The
    update itself is the same line either way.
    """
    n_keep = (n_steps + gap - 1) // gap
    kept = (None if sink is not None else
            np.empty((psi.shape[0], n_keep) + tuple(psi.shape[1:]),
                     dtype=np.float64))
    j = 0
    for i in range(n_steps):
        psi += _del_sq(psi ** 3 - psi - _del_sq(psi)) * DT
        if i % gap == 0:
            if sink is not None:
                sink(j, psi.cpu().numpy())
            elif kept is not None:
                kept[:, j] = psi.cpu().numpy()
            j += 1
        if progress and (i + 1) % progress == 0:
            print(f"    step {i + 1:,}/{n_steps:,}", flush=True)
    return kept


def run_batches(specs, l, n_steps, gap, device, batch=BATCH, tag=""):
    """Roll every (seed, off) in ``specs``, in order, ``batch`` at a time."""
    out = []
    for s in range(0, len(specs), batch):
        chunk = specs[s:s + batch]
        ic = np.stack([set_ic(l, seed, off) for seed, off in chunk])
        psi = torch.from_numpy(ic).to(device=device, dtype=torch.float64)
        t0 = time.time()
        out.append(integrate(psi, n_steps, gap))
        print(f"  [{tag}] {s + len(chunk)}/{len(specs)} trajectories "
              f"({time.time() - t0:.1f}s)", flush=True)
        del psi
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def features_labels(kept, n_ahead, gap):
    """ch.py's prepare_features_labels, expressed on the kept frames.

    ch.py takes X = data[:-n_ahead][::gap] and y = data[n_ahead:][::gap].  With
    start = 0 and gap dividing n_ahead those are exactly kept[:-shift] and
    kept[shift:] for shift = n_ahead // gap -- the same frames, not a
    re-derivation of them.
    """
    shift = n_ahead // gap
    return kept[:, :-shift], kept[:, shift:]


def generate(fname=FNAME, device=None):
    if START != 0 or N_AHEAD % GAP != 0:
        raise ValueError(
            f"the frame-skipping path assumes START == 0 (got {START}) and "
            f"GAP dividing N_AHEAD (got {N_AHEAD} % {GAP}). Use ch.py, or keep "
            f"every frame, if that changes.")

    os.makedirs(os.path.dirname(os.path.abspath(fname)), exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    n_steps = int(END / DT)
    train_specs, val_specs = plan()
    print(f"[gen] device={device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda"
             else "")
          + f", l={L}, {n_steps:,} steps, dt={DT}", flush=True)
    print(f"[gen] {len(train_specs)} train + {len(val_specs)} val "
          f"trajectories over off={OFF_VALUES}", flush=True)

    t0 = time.time()
    train = run_batches(train_specs, L, n_steps, GAP, device, tag="train")
    val = run_batches(val_specs, L, n_steps, GAP, device, tag="val")
    train_X, train_y = features_labels(train, N_AHEAD, GAP)
    val_X, val_y = features_labels(val, N_AHEAD, GAP)

    total = (train_X.nbytes + train_y.nbytes + val_X.nbytes + val_y.nbytes)
    print(f"[gen] train_X {train_X.shape}  val_X {val_X.shape}  "
          f"{train_X.dtype}", flush=True)
    print(f"[gen] writing {fname} ({total / 1e9:.2f} GB)", flush=True)
    np.savez(fname, train_X=train_X, train_y=train_y,
             val_X=val_X, val_y=val_y)
    print(f"[gen] done in {(time.time() - t0) / 60:.1f} min", flush=True)


class _RowWriter:
    """integrate()'s sink for write_lean: frame j of a batch into its rows.

    A small object rather than a closure, so the memory-mapped array has one
    owner that write_lean can release -- on Windows a file that is still
    mapped cannot be renamed into place.
    """

    def __init__(self, arr, rows):
        self.arr: "np.ndarray | None" = arr
        self.rows = rows

    def __call__(self, j, frame):
        if self.arr is None:
            raise RuntimeError("write_lean has already released the file")
        self.arr[self.rows, j] = frame.astype(np.float32)


def lean_sidecar(path):
    """The metadata file that goes with a lean dataset."""
    return os.path.splitext(path)[0] + ".json"


def write_lean(path, t_end, device=None, batch=BATCH):
    """Every trajectory ch.py would make, to ``t_end``, streamed to one file.

    The same seeds, initial conditions, update and frames as generate(), but
    written as it goes into a float32 array of shape (25, F, L, L) on disk --
    20 train rows then 5 val rows, the order fno_train's TRAIN_ON_VAL puts
    them in -- with F = 10 * t_end, frame j being the state after 10 j + 1
    Euler steps (ch.py's data[10 j]).  Host memory stays at one batch of
    frames however long the run, so t_end = 2000 fits a 20 GB machine.

    Only X is stored: ch.py's y is X shifted by N_AHEAD // GAP frames, and
    fno_train rebuilds its targets from that shift anyway.  float32 is what
    fno_train casts ch.py's float64 to on load, so nothing it trains on
    changes.  A longer file is a prefix extension of a shorter one -- the
    same operations on the same batch -- so one t_end = 2000 file serves
    training to 300, 500, 1000 or 2000 by slicing (fno_train --t-train).
    """
    if START != 0 or N_AHEAD % GAP != 0:
        raise ValueError(f"the lean path assumes START == 0 and GAP dividing "
                         f"N_AHEAD (got {START}, {N_AHEAD} % {GAP})")
    if not str(path).lower().endswith(".npy"):
        raise ValueError(f"{path}: a lean dataset must be named *.npy, so "
                         f"fno_train recognises it and finds its .json")
    if os.path.exists(path) or os.path.exists(lean_sidecar(path)):
        raise FileExistsError(f"{path} (or its .json) already exists; delete "
                              f"it or choose another name")
    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    n_steps = int(round(t_end / DT))
    n_keep = (n_steps + GAP - 1) // GAP
    train_specs, val_specs = plan()
    specs = [(s, o, "train") for s, o in train_specs] + \
            [(s, o, "val") for s, o in val_specs]
    size = len(specs) * n_keep * L * L * 4
    print(f"[lean] {len(specs)} trajectories to t = {t_end:g} ({n_steps:,} "
          f"steps, {n_keep:,} frames each) -> {path} ({size / 1e9:.2f} GB, "
          f"float32), device={device}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".partial.npy"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                    shape=(len(specs), n_keep, L, L))
    t0 = time.time()
    done = False
    writer = None
    try:
        for s in range(0, len(specs), batch):
            chunk = specs[s:s + batch]
            ic = np.stack([set_ic(L, seed, off) for seed, off, _ in chunk])
            psi = torch.from_numpy(ic).to(device=device, dtype=torch.float64)
            writer = _RowWriter(out, slice(s, s + len(chunk)))
            integrate(psi, n_steps, GAP, progress=max(n_steps // 10, 1),
                      sink=writer)
            print(f"  [lean] {s + len(chunk)}/{len(specs)} trajectories "
                  f"({time.time() - t0:.0f}s)", flush=True)
        out.flush()
        done = True
    finally:
        # A traceback still holds integrate's frame, and through it the
        # writer; cut its link to the map so the file can go (Windows).
        if writer is not None:
            writer.arr = None
        del out
        if not done:
            try:
                os.remove(tmp)
            except OSError as exc:
                print(f"[lean] could not remove {tmp}: {exc}", flush=True)
    os.replace(tmp, path)
    meta = {"format": LEAN_FORMAT, "l": L, "dt": DT, "gap": GAP,
            "n_ahead": N_AHEAD, "shift": N_AHEAD // GAP, "t_end": float(t_end),
            "n_frames": n_keep, "dtype": "float32",
            "frame_rule": "frame j = state after 10 j + 1 Euler steps "
                          "(ch.py data[10 j]); ch.py's X for horizon T is "
                          "frames [0, 10 T - 10)",
            "n_train": len(train_specs), "n_val": len(val_specs),
            "rows": [{"seed": seed, "off": off, "split": split}
                     for seed, off, split in specs],
            "device": str(device), "torch": torch.__version__,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    side = lean_sidecar(path)
    with open(side + ".tmp", "w") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(side + ".tmp", side)
    print(f"[lean] done in {(time.time() - t0) / 60:.1f} min; metadata in "
          f"{side}", flush=True)
    return meta


def _del_sq_np(psi):
    return (np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0)
            + np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1)
            - 4 * psi) / DX ** 2


def _chc_np(psi, n_steps, gap):
    """ch.py's chc, keeping the same frames integrate() keeps."""
    n_keep = (n_steps + gap - 1) // gap
    kept = np.empty((n_keep,) + psi.shape, dtype=np.float64)
    psi, j = psi.copy(), 0
    for i in range(n_steps):
        phi = psi ** 3 - psi - _del_sq_np(psi)
        psi += _del_sq_np(phi) * DT
        if i % gap == 0:
            kept[j] = psi
            j += 1
    return kept


def verify(n_traj, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    n_steps = int(END / DT)
    specs = plan()[0][:n_traj]
    print(f"[verify] {n_traj} trajectories, {n_steps:,} steps, "
          f"ch.py numpy vs {device}\n", flush=True)
    gpu = run_batches(specs, L, n_steps, GAP, device, tag="gpu")
    print()
    worst, ok = 0.0, True
    for i, (seed, off) in enumerate(specs):
        t0 = time.time()
        ref = _chc_np(set_ic(L, seed, off), n_steps, GAP)
        final_ref, final_gpu = ref[-1], gpu[i, -1]
        d = float(np.abs(ref - gpu[i]).max())
        flips = int(np.sum(np.sign(final_ref) != np.sign(final_gpu)))
        worst = max(worst, d)
        ok = ok and flips == 0
        print(f"  seed {seed:>5} off={off:+.1f}: max|diff| = {d:.3e}, "
              f"cells in the wrong phase at t={END} = {flips}, "
              f"mass drift = {abs(final_gpu.mean() - final_ref.mean()):.2e}  "
              f"({time.time() - t0:.0f}s for the numpy reference)", flush=True)
    print(f"\n[verify] worst difference {worst:.3e} over {n_traj} "
          f"trajectories; phase agreement "
          f"{'EXACT' if ok else 'BROKEN'}", flush=True)
    return ok


def benchmark(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    n_steps, probe = int(END / DT), 2000
    seed, off = plan()[0][0]
    t0 = time.time()
    _chc_np(set_ic(L, seed, off), probe, GAP)
    t_np = (time.time() - t0) * n_steps / probe * NENS

    ic = np.stack([set_ic(L, s, o) for s, o in plan()[0][:BATCH]])
    psi = torch.from_numpy(ic).to(device=device, dtype=torch.float64)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    integrate(psi, probe, GAP)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_gpu = (time.time() - t0) * n_steps / probe

    print(f"projected for all {NENS} trajectories, {n_steps:,} steps, l={L}")
    print(f"  ch.py, numpy, one at a time   {t_np / 60:6.1f} min")
    print(f"  this, {device.type}, {BATCH} batched      "
          f"{t_gpu / 60:6.1f} min   ({t_np / t_gpu:.1f}x)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(__doc__ or "GPU Cahn-Hilliard simulation").splitlines()[0])
    ap.add_argument("--verify", type=int, nargs="?", const=3, metavar="N",
                    help="compare N trajectories against ch.py's numpy path")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force the CPU backend")
    ap.add_argument("--out", default="",
                    help="output .npz (default Work/data/test_t<END>.npz, "
                         "e.g. test_t0500.npz for --end 500)")
    ap.add_argument("--end", type=float, default=END,
                    help=f"integrate to this time instead of {END:g}")
    ap.add_argument("--lean", metavar="PATH",
                    help="write a streamed float32 X-only dataset (.npy plus "
                         "a .json sidecar) instead of ch.py's test.npz; one "
                         "file to --end serves every shorter training horizon")
    a = ap.parse_args()
    if WRITE_LOG:
        fno_runlog.start_log("ch_gpu")
    dev = torch.device("cpu") if a.cpu else None
    END = a.end

    if a.verify:
        raise SystemExit(0 if verify(a.verify, dev) else 1)
    if a.lean:
        write_lean(a.lean, END, dev)
    elif a.benchmark:
        benchmark(dev)
    else:
        out = a.out or npz_path(END)
        if os.path.exists(out):
            print(f"{out} already exists; delete it or pass --out to "
                  f"regenerate")
        else:
            generate(out, dev)
