"""hpc_storage and make_diagram: the two small tools."""
import ast
import contextlib
import io as _io
import os
import shutil
import struct
import subprocess
import sys
import types

import matplotlib.image as mpimg
import torch

from _paths import ROOT, SCRATCH, project_state

os.chdir(ROOT)
PROJECT = project_state()

import fno_paths
import fno_predict
import fno_train as T
T.use_fixed_step()
import hpc_storage as H
import make_diagram as D
import run_all

ok = True
J = os.path.join
BOX = J(SCRATCH, "tools")


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


@contextlib.contextmanager
def patched(obj, name, value):
    """obj.name is value inside the block, and as it was after, even if it
    was not there."""
    had = hasattr(obj, name)
    old = getattr(obj, name, None)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if had:
            setattr(obj, name, old)
        else:
            delattr(obj, name)


def fresh(*parts):
    path = J(BOX, *parts)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def png_size(path):
    data = read_bytes(path)
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", data[16:24])


def param_count():
    """Parameters of a model built from fno_train's constants as they stand,
    counted here and not by make_diagram."""
    m = T.build_model(n_ref=64, device="cpu")
    return sum(p.numel() for p in m.parameters())


def first_conv_in():
    m = T.build_model(n_ref=16, device="cpu")
    conv = next(c for c in m.modules() if isinstance(c, torch.nn.Conv2d))
    return conv.in_channels


FOUR = ["fno_architecture.png", "fno_architecture.pdf",
        "project_pipeline.png", "project_pipeline.pdf"]

print("1. make_diagram writes both figures, as PNG and PDF")
out = fresh("figs")
paths = D.write(out)
check([os.path.basename(p) for p in paths] == FOUR
      and list(D.FILES) == FOUR and sorted(os.listdir(out)) == sorted(FOUR),
      "fno_architecture and project_pipeline, .png and .pdf, and nothing "
      "else")
for p in paths:
    if p.endswith(".png"):
        w, h = png_size(p) or (0, 0)
        pic = mpimg.imread(p)
        check(w >= 2000 and h >= 800 and float(pic.std()) > 0.05,
              f"{os.path.basename(p)} is a PNG {w}x{h} with something drawn")
    else:
        data = read_bytes(p)
        check(data[:5] == b"%PDF-" and b"%%EOF" in data[-1024:]
              and len(data) > 5000,
              f"{os.path.basename(p)} is a PDF of {len(data)} bytes")
again = fresh("figs_again")
paths2 = D.write(again)
check(all(read_bytes(a) == read_bytes(b) for a, b in zip(paths, paths2)),
      "drawn twice, the four files are the same bytes (the PDF has no date)")
via_cli = fresh("figs_cli")
check(D.main(["--out", via_cli]) == 0
      and sorted(os.listdir(via_cli)) == sorted(FOUR),
      "--out DIR puts them in DIR")
check(D.default_out() == fno_paths.results_root()
      and D.default_out() == J("Results"),
      "by default the figures go to the Results root, as in the flat layout")
box = fresh("project_like")
here = os.getcwd()
os.chdir(box)
try:
    D.main([])
finally:
    os.chdir(here)
check(sorted(os.listdir(J(box, "Results"))) == sorted(FOUR),
      "run with no --out from a project root, they land in Results/ "
      "(no diagrams/ folder)")

print("\n2. it needs no GPU and disturbs nothing")
calls = []
real_build = T.build_model


def spy(*a, **k):
    calls.append(k)
    return real_build(*a, **k)


with patched(T, "build_model", spy):
    D.facts()
check(len(calls) == 1 and calls[0].get("device") == "cpu"
      and calls[0].get("n_ref", 10 ** 6) <= 32,
      f"the model is built on the CPU at a small n_ref: {calls}")
check(not torch.cuda.is_initialized(), "CUDA was never started")
torch.manual_seed(5)
before = torch.rand(3)
torch.manual_seed(5)
D.facts()
after = torch.rand(3)
check(bool((before == after).all()),
      "drawing leaves the random state as it found it")
m16, m64 = T.build_model(n_ref=16), T.build_model(n_ref=64)
check(m16.n_parameters() == m64.n_parameters()
      == D.facts()["params"],
      "the parameter count does not depend on the n_ref it is built at")

print("\n3. the figures say what fno_train says")
L = D.labels()
n_par = param_count()
fourier = L["fourier"].split("\n")
check(fourier[0] == f"{T.LAYERS} Fourier layers"
      and f"{T.CHANNELS} channels" in fourier
      and f"{T.MODES} modes" in fourier
      and f"{T.LOCAL_KERNEL}x{T.LOCAL_KERNEL} local conv" in fourier,
      f"the Fourier layers box: {fourier}")
check(L["arch_title"] == f"FNO architecture: {n_par:,} parameters",
      f"the parameter count is the built model's: {L['arch_title']}")
n_in = first_conv_in()
check(L["input"].split("\n")[0] == f"Input: {n_in} channels"
      and L["lift"].split("\n")[1] == f"1x1 conv {n_in} -> {T.CHANNELS}",
      "the input channels are the lift's")
check(f"{2 * T.MODES + 1}x{T.MODES + 1} wavenumbers" in L["spectral"]
      and f"keep {T.MODES} modes" in L["spectral"]
      and f"{T.CHANNELS}x{T.CHANNELS} complex" in L["spectral"],
      "the spectral conv's weights are as wide as the constants say")
check(L["project"].split("\n")[1:4:2]
      == [f"1x1 conv {T.CHANNELS} -> {2 * T.CHANNELS}",
          f"1x1 conv {2 * T.CHANNELS} -> 1"],
      "the projection and head widths")
check(L["laplacian"].startswith("Laplacian output: ON")
      and L["mass"].startswith("Hard mass constraint: ON")
      and L["zero_dc"].startswith("Zero-DC constraint: ON"),
      "the Laplacian option and the two hard constraints, as set")
check(not T.DT_CONDITIONED and L["output"] == f"Output\npsi(t + {T.DT_STEP:g})"
      and L["train"].split("\n")[1]
      == f"{T.EPOCHS} epochs of {T.STEPS_PER_EPOCH} steps"
      and "dt_in" not in L,
      "the step size and the training length (this suite trains the "
      "fixed-step model: no dt boxes)")
with patched(T, "DT_CONDITIONED", True):
    LC = D.labels()
    nc = param_count()
emb = 2 * (T.COND_DIM // 2) + 1
check(LC["output"] == "Output\npsi(t + dt)"
      and LC["arch_note"].startswith("One call maps psi(t) to psi(t + dt)")
      and f"{T.DT_MIN:g} to {T.DT_MAX:g} time units" in LC["arch_note"]
      and LC["arch_title"] == f"FNO architecture: {nc:,} parameters"
      and nc > n_par,
      "a dt-conditioned model maps psi(t) to psi(t + dt), dt from DT_MIN to "
      "DT_MAX, and has the parameters of its conditioning")
check(LC["dt_in"].split("\n")[1] == f"{T.DT_MIN:g} to {T.DT_MAX:g} time units"
      and f"sin, cos at {T.COND_DIM // 2} log-spaced" in LC["dt_embed"]
      and f"{emb} numbers" in LC["dt_embed"]
      and f"linear {emb} -> {T.COND_DIM}, GELU" in LC["dt_mlp"]
      and "starts at dt" in LC["dt_gain"] and LC["film"].startswith("FiLM")
      and "h (1 + g) + b" in LC["film"] and "scale and shift" in LC["film_arrow"]
      and f"{T.COND_DIM} numbers" in LC["film_in"],
      "the dt conditioning: log dt embedding, small MLP, FiLM scale and "
      "shift, and the output gain that starts at dt")
with patched(T, "MU_CHANNEL", True):
    L_mu = D.labels()
with patched(T, "MU_CHANNEL", True), patched(T, "MU_CLIP", None):
    L_mu_free = D.labels()
check(f"(mu of psi clipped to |psi| <= {T.MU_CLIP:g})" in L_mu["input"]
      and "clipped" not in L_mu_free["input"] and "clipped" not in L["input"],
      "the input box says the mu channel is computed from psi clipped to "
      "MU_CLIP, and nothing when there is no channel or no clip")
check(L["lowk_fill"].startswith("Low-k fill: ON")
      and "k = 0 weight is filled" in L["lowk_fill"]
      and "zeroed again" in L["lowk_fill"],
      "the low-k fill on the spectral conv (LOWK_FILL is on by default)")

with patched(T, "CHANNELS", 17):
    L2 = D.labels()
    n2 = param_count()
check("17 channels" in L2["fourier"] and "38 channels" not in L2["fourier"]
      and L2["lift"].split("\n")[1] == f"1x1 conv {n_in} -> 17"
      and "17x17 complex" in L2["spectral"]
      and L2["arch_title"] == f"FNO architecture: {n2:,} parameters"
      and n2 != n_par,
      "CHANNELS = 17 changes the channels and the parameter count")
with patched(T, "MODES", 7):
    L2 = D.labels()
check("7 modes" in L2["fourier"] and "keep 7 modes" in L2["spectral"]
      and "15x8 wavenumbers" in L2["spectral"],
      "MODES = 7 changes the modes and the weight grid")
with patched(T, "LAYERS", 3):
    L2 = D.labels()
check(L2["fourier"].startswith("3 Fourier layers")
      and "one of the 3 Fourier" in L2["layer_title"],
      "LAYERS = 3 changes the number of layers")
with patched(T, "LOCAL_KERNEL", 5):
    L2 = D.labels()
check("5x5 local conv" in L2["fourier"] and "5x5, circular" in L2["local"],
      "LOCAL_KERNEL = 5 changes the local conv")
with patched(T, "LAPLACIAN_OUTPUT", False):
    L2 = D.labels()
check(L2["laplacian"].startswith("Laplacian output: off"),
      "LAPLACIAN_OUTPUT off is drawn as off")
check("Laplacian stencil" in L["laplacian"],
      "and on, the default, is drawn with its stencil")
with patched(T, "HARD_MASS_CONSERVATION", False):
    L2 = D.labels()
check(L2["mass"].startswith("Hard mass constraint: off")
      and L2["zero_dc"].startswith("Zero-DC constraint: ON"),
      "the mass constraint off, the zero-DC one still on")
with patched(T, "ZERO_DC", False):
    L2 = D.labels()
check(L2["zero_dc"].startswith("Zero-DC constraint: off"),
      "the zero-DC constraint off is drawn as off")
with patched(T, "EPOCHS", 7), patched(T, "DT_STEP", 2.5):
    L2 = D.labels()
check(L2["train"].split("\n")[1].startswith("7 epochs of")
      and L2["output"].endswith("psi(t + 2.5)")
      and L2["arch_note"].startswith("One call maps psi(t) to psi(t + 2.5)"),
      "the training length and the step size are read, not typed")
with patched(T, "DT_CONDITIONED", True), patched(T, "DT_MAX", 30.0):
    L2 = D.labels()
check("1 to 30 time units" in L2["dt_in"],
      "the dt range is read from DT_MIN and DT_MAX")
with patched(T, "LOWK_FILL", False):
    L2 = D.labels()
check(L2["lowk_fill"].startswith("Low-k fill: off"),
      "the low-k fill off is drawn as off")
with patched(fno_paths, "WORK", "Elsewhere"):
    L2 = D.labels()
check(L2["data"].startswith("Elsewhere/data")
      and L2["models"].startswith("Elsewhere/models/<label>")
      and L2["rollouts"].startswith("Elsewhere/rollout_cache"),
      "the Work folders come from fno_paths")
with patched(fno_predict, "RECORDS_DIR", "recs"):
    L2 = D.labels()
check("recs/" in L2["rollouts"], "and the records folder from fno_predict")
check(D.pretty("1x1 conv 2 -> 45, psi") == "1\u00d71 conv 2 \u2192 45, \u03c8",
      "pretty turns ASCII into the symbols drawn")

print("\n4. what is drawn follows the labels")
changed = fresh("figs_changed")
with patched(T, "CHANNELS", 17):
    paths3 = D.write(changed)
check(read_bytes(paths3[0]) != read_bytes(paths[0])
      and read_bytes(paths3[1]) != read_bytes(paths[1])
      and read_bytes(paths3[2]) == read_bytes(paths[2])
      and read_bytes(paths3[3]) == read_bytes(paths[3]),
      "CHANNELS = 17 changes fno_architecture (png and pdf) and leaves "
      "project_pipeline")
with patched(T, "DT_CONDITIONED", True):
    paths4 = D.write(fresh("figs_cond"))
check(read_bytes(paths4[0]) != read_bytes(paths[0])
      and read_bytes(paths4[2]) == read_bytes(paths[2])
      and (png_size(paths4[0]) or (0, 0))[1]
      > (png_size(paths[0]) or (0, 0))[1],
      "a dt-conditioned model is drawn with the dt rows (a taller figure)")
want = ("ch_gpu.py", "fno_train.py", "fno_predict.py", "make_submission.py")
text = "\n".join(L[k] for k in ("gen", "train", "predict", "submit", "index"))
check(all(n in text and os.path.isfile(J(ROOT, n)) for n in want),
      "the pipeline names four scripts, and each one is in the project")

print("\n5. hpc_storage refuses off Linux")
for plat in ("win32", "darwin"):
    check("Linux" in (H.unsupported(plat) or "")
          and plat in (H.unsupported(plat) or ""),
          f"it refuses {plat} and says why")
check(H.unsupported("linux") is None, "and runs on linux")
asked = []


def never(cmd, timeout=30):
    asked.append(cmd)
    return ""


for plat in ("win32", "darwin"):
    buf, asked[:] = _io.StringIO(), []
    why = None
    with patched(sys, "platform", plat), patched(H, "run", never), \
            contextlib.redirect_stdout(buf):
        try:
            H.main(["--du"])
        except SystemExit as exc:
            why = str(exc.code)
    check(why is not None and "needs Linux" in why and plat in why
          and not asked and not buf.getvalue(),
          f"{plat}: main stops with a message, runs nothing, prints nothing")
if sys.platform == "win32":
    p = subprocess.run([sys.executable, "hpc_storage.py", "--du"], cwd=ROOT,
                       capture_output=True, text=True, errors="replace")
    check(p.returncode != 0 and "needs Linux" in p.stderr
          and "Traceback" not in p.stderr and not p.stdout,
          "run for real on Windows: a clear message and no traceback")
else:
    print("     (the real Windows run is skipped on this platform)")
p = subprocess.run([sys.executable, "-c",
                    "import sys; sys.modules['grp'] = None; "
                    "import hpc_storage"],
                   cwd=ROOT, capture_output=True, text=True, errors="replace")
check(p.returncode == 0 and not p.stderr,
      "importing it needs no grp, so it loads where there is none")

print("\n6. hpc_storage formatting")
h = H.human
check([h(0), h(1023), h(1024), h(1536), h(1024 ** 2), h(5 * 1024 ** 3)]
      == ["0 B", "1023 B", "1.0 KB", "1.5 KB", "1.0 MB", "5.0 GB"],
      "human sizes")
check(h(1024 ** 2 - 1) == "1.0 MB" and h(3 * 1024 ** 5) == "3.0 PB"
      and h(1024 ** 6) == "1024.0 PB",
      "human sizes at the rounding edge and past the largest unit")
gb = 1024 ** 3
check(H.fs_lines(["cwd", "$HOME"], "/home/u", "gpfs",
                 (200 * gb, 150 * gb, 50 * gb, 1000000, 250000))
      == ["cwd $HOME  ->  /home/u  [gpfs]",
          "  size 200.0 GB | used 150.0 GB (75%) | available 50.0 GB",
          "  inodes: 250,000 used of 1,000,000"],
      "a filesystem's three lines")
check(H.fs_lines(["x"], "/x", "?", (0, 0, 0, 0, 0))[1:]
      == ["  size 0 B | used 0 B (0%) | available 0 B"],
      "an empty filesystem: 0%, and no inode line")
mounts = H.parse_mounts([
    "/dev/sda1 / ext4 rw 0 0", "tmpfs /tmp tmpfs rw 0 0",
    "gpfs0 /gpfs gpfs rw 0 0", "scratch /gpfs/home gpfs rw 0 0",
    "srv:/x /mnt/my\\040disk nfs4 rw 0 0", "short line"])
check(len(mounts) == 5 and ("/mnt/my disk", "nfs4", "srv:/x") in mounts,
      "/proc/mounts lines, with \\040 undone")
check(H.mount_info("/gpfs/home/u", mounts) == ("gpfs", "scratch")
      and H.mount_info("/gpfs/homework/x", mounts) == ("gpfs", "gpfs0")
      and H.mount_info("/mnt/my disk/a", mounts) == ("nfs4", "srv:/x")
      and H.mount_info("/etc", mounts) == ("ext4", "/dev/sda1")
      and H.mount_info("/gpfs", mounts) == ("gpfs", "gpfs0")
      and H.mount_info("relative", []) == ("?", ""),
      "the longest mount that holds the path, not one that only shares "
      "its prefix")

print("\n7. hpc_storage picks the quota tool by filesystem")


def found(name):
    return "/bin/" + name


def lacking(name):
    return None


plan = H.quota_plan("gpfs", "/dev/gpfs0", "/p", "me", ["g1", "g2"], "fs1",
                    found)
check(plan == ([["/bin/mmlsquota", "-u", "me", "--block-size", "auto",
                 "gpfs0"],
                ["/bin/mmlsquota", "-g", "g1", "--block-size", "auto",
                 "gpfs0"],
                ["/bin/mmlsquota", "-g", "g2", "--block-size", "auto",
                 "gpfs0"],
                ["/bin/mmlsquota", "-j", "fs1", "--block-size", "auto",
                 "gpfs0"]], None),
      "GPFS: the user, each group and the fileset")
check(H.quota_plan("mmfs", "gpfs0", "/p", "me", [], None, found)[0]
      == [["/bin/mmlsquota", "-u", "me", "--block-size", "auto", "gpfs0"]],
      "mmfs is GPFS; no group and no fileset, one command")
check(H.quota_plan("lustre", "d", "/scratch/x", "me", [], None, found)
      == ([["/bin/lfs", "quota", "-h", "-u", "me", "/scratch/x"]], None),
      "Lustre: lfs quota")
check(H.quota_plan("beegfs", "d", "/p", "me", [], None, found)
      == ([["/bin/beegfs-ctl", "--getquota", "--uid", "me"]], None),
      "BeeGFS: beegfs-ctl")
check(H.quota_plan("nfs4", "d", "/p", "me", [], None, found)
      == ([["/bin/quota", "-s"]], None)
      and H.quota_plan("nfs", "d", "/p", "me", [], None, found)[0]
      == [["/bin/quota", "-s"]],
      "NFS: quota -s")
check(H.quota_plan("ext4", "d", "/p", "me", [], None, found) == ([], None),
      "a filesystem with no quota tool of ours: nothing to run")
check([H.quota_plan(k, "d", "/p", "me", [], None, lacking)
       for k in ("gpfs", "lustre", "beegfs", "nfs4")]
      == [([], "mmlsquota"), ([], "lfs"), ([], "beegfs-ctl"),
          ([], "quota")],
      "a missing tool is named, not run")

print("\n8. hpc_storage targets and the whole report")
a, b = fresh("fs", "a"), fresh("fs", "b")
with open(J(b, "file.txt"), "w") as fh:
    fh.write("x")
got = H.targets([b, J(b, "file.txt"), J(BOX, "fs", "missing")],
                env={"HOME": a, "SCRATCH": "", "WORK": J(BOX, "fs", "no")},
                cwd=a, tmp=J(BOX, "fs", "nothing"))
check(len(got) == 1 and got[0][0] == ["cwd", "$HOME", b]
      and got[0][1] == os.path.realpath(a),
      "one filesystem is reported once, with every name for it")

work, res = fresh("proj", "Work"), fresh("proj", "Results")
proj = J(BOX, "proj")
other = fresh("other")
fake_fs = types.SimpleNamespace(f_blocks=1000, f_frsize=1024 ** 2,
                                f_bfree=250, f_bavail=200, f_files=1000,
                                f_ffree=400)
runs = []


def fake_run(cmd, timeout=30):
    runs.append(list(cmd))
    return "OUT " + " ".join(os.path.basename(c) for c in cmd[:1])


def report(argv):
    """What main prints, and the commands it ran, on a made-up cluster."""
    runs.clear()
    buf = _io.StringIO()
    code = 1
    here = os.getcwd()
    os.chdir(proj)
    try:
        with patched(sys, "platform", "linux"), \
                patched(os, "statvfs", lambda path: fake_fs), \
                patched(os, "environ", {"PATH": "", "HOME": proj}), \
                patched(H, "TMP_DIR", J(BOX, "nothing")), \
                patched(H, "run", fake_run), \
                patched(H, "find_tool", lambda n: "/usr/bin/" + n), \
                patched(H, "mount_info",
                        lambda path, mounts: ("gpfs", "/dev/gpfs0")), \
                patched(H, "group_names", lambda path: ["grp1"]), \
                patched(H, "gpfs_fileset", lambda path: "fs1"), \
                patched(H.getpass, "getuser", lambda: "tester"), \
                contextlib.redirect_stdout(buf):
            code = H.main(argv)
    finally:
        os.chdir(here)
    return code, buf.getvalue()


code, text = report(["--du"])
lines = text.splitlines()
real = os.path.realpath(proj)
check(code == 0 and lines[0] == f"cwd $HOME Work Results  ->  {real}  [gpfs]"
      and lines[1] == "  size 1000.0 MB | used 750.0 MB (75%) | "
                      "available 200.0 MB"
      and lines[2] == "  inodes: 600 used of 1,000",
      "the filesystem block, from the numbers statvfs gave")
check("$ /usr/bin/mmlsquota -u tester --block-size auto gpfs0" in text
      and "$ /usr/bin/mmlsquota -g grp1 --block-size auto gpfs0" in text
      and "$ /usr/bin/mmlsquota -j fs1 --block-size auto gpfs0" in text
      and "(user tester)" in text
      and "Quota-related commands visible here:" in text,
      "then the quota: the user's, the group's and the fileset's")
du_runs = [r for r in runs if r[0] == "du"]
check(du_runs == [["du", "-sh", os.path.realpath(work)],
                  ["du", "-sh", os.path.realpath(res)]]
      and "Archive: not there, nothing to size" in text,
      "--du alone sizes Work and Results, and says Archive is not there")
check(H.du_default() == ["Work", "Results", "Archive"]
      and H.du_default() == [fno_paths.WORK, fno_paths.RESULTS,
                             fno_paths.ARCHIVE],
      "the default folders are fno_paths' three roots")
code, text = report(["--du", other])
check([r for r in runs if r[0] == "du"]
      == [["du", "-sh", os.path.realpath(other)]],
      "--du PATH sizes that path and not the defaults")
code, text = report([other])
check(not [r for r in runs if r[0] == "du"] and "$ du" not in text
      and other in text.splitlines()[0],
      "a path with no --du is reported on but not sized")
allowed = {"mmlsquota", "mmlsattr", "lfs", "beegfs-ctl", "quota", "du"}
report(["--du"])
check(runs and all(os.path.basename(r[0]) in allowed for r in runs),
      f"only reading commands are run: {sorted({r[0] for r in runs})}")

print("\n9. hpc_storage only reads")
with open(J(ROOT, "hpc_storage.py"), encoding="utf-8") as fh:
    tree = ast.parse(fh.read())
names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
risky = {"remove", "unlink", "rmtree", "rmdir", "removedirs", "rename",
         "replace", "move", "copy", "copyfile", "chmod", "chown", "kill",
         "system", "truncate", "mkdir", "makedirs", "symlink", "write"}
opens = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
         and isinstance(n.func, ast.Name) and n.func.id == "open"]
modes = [n.args[1] for n in opens if len(n.args) > 1]
modes += [k for n in opens for k in n.keywords if k.arg == "mode"]
check(not (names & risky) and opens and not modes,
      f"no delete, move, write or open-for-writing in it: "
      f"{sorted(names & risky)}")
raw = read_bytes(J(ROOT, "hpc_storage.sh"))
sh = raw.decode()
check(b"\r" not in raw and sh.startswith("#!/bin/bash\n")
      and "#SBATCH --output=hs_%j.out\n" in sh
      and "#SBATCH --error=hs_%j.err\n" in sh
      and "#SBATCH --partition=test\n" in sh
      and 'python3 hpc_storage.py "$@"' in sh
      and "change it if this account needs another" in sh,
      "hpc_storage.sh: LF, logs next to it, the group partition and a "
      "note that it may need changing, arguments passed on")

print("\n10. they are registered")
check("hpc_storage.py" in run_all.SHIPPED
      and "make_diagram.py" in run_all.SHIPPED
      and all(os.path.isfile(J(ROOT, n)) for n in run_all.SHIPPED),
      "both are in SHIPPED, so pyright reads them")
check("t_tools" in run_all.SUITES
      and os.path.isfile(J(ROOT, "tests", "t_tools.py")),
      "this suite is in SUITES")
check(project_state() == PROJECT, "nothing was written into the project")

shutil.rmtree(BOX, ignore_errors=True)
print("\n" + ("TOOLS PASS" if ok else "TOOLS FAIL"))
sys.exit(0 if ok else 1)
