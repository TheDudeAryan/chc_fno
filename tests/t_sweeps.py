"""The single R2 sweep images: r2_sweep_per_size/ and r2_sweep_per_case/.

The images and lines for a fake run of 15 cases (three box sizes, five
compositions), the same with stored curves and manifest entries of the dropped
sizes 160 and 192 left in (ignored), a fake older run of 6 cases and a run of
one case; one image each, with no title, no panel letter and no PDF; the notes
of each folder; the missing-curve rule; the note on stale files.
"""
import contextlib
import io
import os
import sys

import matplotlib
import matplotlib.figure
import matplotlib.image
import numpy as np

from _paths import ROOT, SCRATCH, project_state

os.chdir(ROOT)
PROJECT = project_state()

import fno_paths
import make_submission as S

ok = True
J = os.path.join
BOX = J(SCRATCH, "sweeps")


def check(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL "), m)


def curve(off, lat, rng):
    t = np.arange(50, 10001, 50.0)
    c = 1 - 0.02 * (t / 10000) ** 0.5 * (1 + abs(off)) - 0.01 * rng.random()
    return {"times": t, "curve": c, "sd": 0.01 + 0.05 * (t / 10000),
            "n": np.array(50), "lattice": np.array(lat)}


def fake(sizes, comps, crit_sizes=()):
    """sweep_curves entries: the compositions at 128 under "compositions",
    the critical case at other sizes under "sizes", the others under "grid"."""
    rng = np.random.default_rng(1)
    cur = {}
    for lat in sizes:
        for off in comps:
            v = curve(off, lat, rng)
            if lat == 128:
                cur[("compositions", f"{off:+.2f}")] = v
            else:
                cur[("grid", f"{off:+.2f}_l{lat}")] = v
    for lat in crit_sizes:
        cur[("sizes", str(lat))] = curve(0.0, lat, rng)
    return cur


COMPS = (-0.4, -0.2, 0.0, 0.2, 0.4)
drawn_log = []
real_draw = S._draw_sweep


def spy(drawn, horizon, folder):
    drawn_log.append((os.path.basename(folder), drawn))
    return real_draw(drawn, horizon, folder)


S._draw_sweep = spy


def run(name, curves):
    folder = J(BOX, name)
    os.makedirs(folder, exist_ok=True)
    drawn_log.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        names = S.figure_r2_sweeps(curves, folder, 1000.0)
    return folder, names, {n: d for n, d in drawn_log}


def shape(d):
    """[(image name, [line labels])] of one set."""
    return [(name, [ln[0] for ln in lines]) for name, _, lines in d]


# 15 cases: 3 sizes x 5 compositions, no older cases
f15, n15, d15 = run("c15", fake((128, 256, 512), COMPS))
want15 = (["r2_sweep_per_size/" + f"l{n}.png" for n in (128, 256, 512)]
          + ["r2_sweep_per_case/" + c + ".png" for c in
             ("psi0m0.40", "psi0m0.20", "psi0p0.00", "psi0p0.20",
              "psi0p0.40")])
check(n15 == want15
      and all(os.path.getsize(J(f15, n)) > 5000 for n in n15),
      f"15 cases: {len(n15)} single images written")
check(sorted(os.listdir(f15)) == ["r2_sweep_per_case", "r2_sweep_per_size"]
      and sorted(os.listdir(J(f15, "r2_sweep_per_size")))
      == ["figure_notes.txt", "l128.png", "l256.png", "l512.png"]
      and sorted(os.listdir(J(f15, "r2_sweep_per_case")))
      == sorted(["figure_notes.txt"] + [c.split("/")[1] for c in want15[3:]])
      and not [f for _, _, fs in os.walk(f15) for f in fs
               if f.endswith(".pdf") or ".partial" in f],
      "each folder holds its images and a figure_notes.txt; no PDF, no "
      "temporary")
for n in n15[:1] + n15[3:4]:
    im = matplotlib.image.imread(J(f15, n))
    scale = fno_paths.figure_dpi(300) / 300   # FNO_FAST_FIGURES halves it
    check(im.shape[1] > 1100 * scale and im.shape[0] > 800 * scale and abs(
        im.shape[1] / im.shape[0] - 4.4 / 3.2) < 0.25,
        f"{n} is a single panel at {fno_paths.figure_dpi(300)} dpi "
        f"{im.shape}")
ps = shape(d15["r2_sweep_per_size"])
check([n for n, _ in ps] == ["l128.png", "l256.png", "l512.png"]
      and all(len(ls) == 5 for _, ls in ps),
      f"per size: three images in order of size with five lines each")
pc = shape(d15["r2_sweep_per_case"])
check([n for n, _ in pc] == [c.split("/")[1] for c in want15[3:]]
      and all(n == S.comp_token(c) + ".png" for (n, _), c in zip(pc, COMPS))
      and all(len(ls) == 3 for _, ls in pc),
      f"per case: five images named by the composition token, three lines "
      f"each (128, 256, 512): {[n for n, _ in pc]}")
lines = d15["r2_sweep_per_size"][2][2]
check(lines[2][7] and not any(l[7] for l in lines[:2] + lines[3:]),
      "psi0 = 0 is the heaviest line")
col128 = {ln[5] for _, _, ls in d15["r2_sweep_per_case"] for ln in ls
          if ln[0] == "$l = 128$"}
col512 = {ln[5] for _, _, ls in d15["r2_sweep_per_case"] for ln in ls
          if ln[0] == "$l = 512$"}
check(len(col128) == 1 and col128 == {S.BOX_COLORS[128]}
      and col512 == {S.BOX_COLORS[512]} and col128 != col512,
      "a box size has the same colour in every image")
check(len(set(S.BOX_COLORS.values())) == 3 and 160 not in S.BOX_COLORS
      and 192 not in S.BOX_COLORS and 160 not in S.BOX_MARKS
      and 192 not in S.BOX_MARKS,
      "three box sizes with three colours; none for 160 and 192")

# no title, no panel letter, the horizon in the legend, one range for the set
seen: list = []
real_save = matplotlib.figure.Figure.savefig


def spy_save(self, *a, **k):
    ax = self.axes[0]
    seen.append((ax.get_title(), self.texts, ax.get_xlabel(),
                 ax.get_ylabel(), ax.get_xlim(), ax.get_ylim(),
                 [t.get_text() for t in ax.get_legend().get_texts()],
                 a[0], k.get("dpi")))
    return real_save(self, *a, **k)


matplotlib.figure.Figure.savefig = spy_save
try:
    run("c15spy", fake((128, 256, 512), COMPS))
finally:
    matplotlib.figure.Figure.savefig = real_save
check(len(seen) == 8 and all(x[0] == "" and not x[1] for x in seen)
      and all(x[2] == "$t$" and x[3] == "$R^2$" for x in seen)
      and all("training horizon" in x[6] for x in seen)
      and all(x[8] == fno_paths.figure_dpi(S.SWEEP_DPI) and x[7].endswith(".partial.png")
              for x in seen),
      "no title and no panel letter; $t$ and $R^2$ labels; the training "
      "horizon in every legend; 300 dpi through a temporary PNG")
check(len({x[4] for x in seen[:3]}) == 1 and len({x[5] for x in seen[:3]}) == 1
      and len({x[4] for x in seen[3:]}) == 1
      and len({x[5] for x in seen[3:]}) == 1,
      "the images of one set share one t range and one R2 range")

# the notes: one block per image, in each folder's own figure_notes.txt
nts = S.read_notes(J(f15, "r2_sweep_per_size", "figure_notes.txt"))
ntc = S.read_notes(J(f15, "r2_sweep_per_case", "figure_notes.txt"))
check(sorted(nts) == ["l128.png", "l256.png", "l512.png"]
      and sorted(ntc) == sorted(c.split("/")[1] for c in want15[3:])
      and any("(50 seeds)" in x and "t = 1000:" in x and "t = 10000:" in x
              and "+-" in x for x in nts["l128.png"])
      and fno_paths.OLD_STORED_NAME not in " ".join(nts["l128.png"]).lower()
      and not os.path.exists(J(f15, "figure_notes.txt")),
      f"notes by image, with seeds and mean +- sd at 1000 to 10000: "
      f"{nts['l128.png'][2][:90]}")

# a cache that still holds curves of the dropped sizes: ignored
left = fake((128, 256, 512), COMPS, (160, 192))
left.update(fake((160, 192), (-0.4, 0.4)))
EXD = J(BOX, "left", "extras")
os.makedirs(EXD, exist_ok=True)
S.save_curves(EXD, left)
man = {"compositions": {"+0.00": {}}, "sizes": {"160": {"folder": "l160"},
       "192": {}, "256": {}}, "grid": {"-0.40_l192": {}, "-0.40_l256": {}},
       "sweep": {"cases": ["sizes 160", "grid -0.40_l192", "sizes 256"],
                 "missing_curves": ["sizes 192"], "undocumented_curves": [],
                 "status": "x"}}
S.write_json(J(EXD, S.BOOKKEEPING), man)
got = S.load_curves(EXD)
mm = S.load_manifest(EXD)
check(not any(S.dropped_key(*k) for k in got) and len(got) == len(left) - 6
      and sorted(mm["sizes"]) == ["256"] and sorted(mm["grid"]) == ["-0.40_l256"]
      and mm["sweep"]["cases"] == ["sizes 256"]
      and mm["sweep"]["missing_curves"] == [],
      "stored curves and the manifest's entries of l = 160 and 192 are "
      "ignored")
fl, nl, dl = run("c15left", got)
check(nl == want15 and sorted(os.listdir(J(fl, "r2_sweep_per_size")))
      == ["figure_notes.txt", "l128.png", "l256.png", "l512.png"]
      and not any("160" in x or "192" in x for _, ds in dl.items()
                  for _, _, ls in ds for ln in ls for x in (ln[0],))
      and not any("160" in x or "192" in x for f in (
          J(fl, "r2_sweep_per_size", "figure_notes.txt"),
          J(fl, "r2_sweep_per_case", "figure_notes.txt"))
          for x in open(f).read().split()),
      "drawn from stored curves that include 160 and 192: the same eight "
      "images, no line and no note for them")

# older run: the five compositions at 128 and the critical quench at 256
f8, n8, d8 = run("c6", fake((128,), COMPS, ()) | {
    ("sizes", str(l)): curve(0.0, l, np.random.default_rng(2))
    for l in (256,)})
p1, p2 = shape(d8["r2_sweep_per_size"]), shape(d8["r2_sweep_per_case"])
check(n8 == ["r2_sweep_per_size/l128.png", "r2_sweep_per_case/psi0p0.00.png"]
      and len(p1) == 1 and len(p1[0][1]) == 5
      and len(p2) == 1 and len(p2[0][1]) == 2,
      f"6 cases: one l = 128 image with five lines, one psi0 = 0 image with "
      f"l = 128 and 256 {n8}")

# one case: no image, no folder
f1, n1, d1 = run("c1", fake((128,), (0.0,)))
check(n1 == [] and os.listdir(f1) == [] and not d1,
      "one case: no image, nothing written")

# the old end-of-run files are not written
check(not any(os.path.exists(J(f, n)) for f in (f15, f8)
              for n in S.STALE_SWEEP_FILES)
      and not any(n in S.RUN_FILES for n in S.STALE_SWEEP_FILES)
      and {"r2_sweep_per_size.png", "r2_sweep_per_size.pdf",
           "r2_sweep_per_case.png", "r2_sweep_per_case.pdf"}
      <= set(S.STALE_SWEEP_FILES)
      and not hasattr(S, "figure_r2_sweep"),
      "the sweep figures of the earlier forms are stale and not in RUN_FILES")

check(S.sweep_panels(fake((128, 256), COMPS))[1].__len__() == 5,
      "five per-case images for two sizes")

# the missing-curve rule
internal = {"compositions": {"+0.00": {}, "-0.20": {}},
            "sizes": {"128": {}, "256": {}}, "grid": {"-0.20_l256": {}}}
cur = {("compositions", "+0.00"): 1, ("compositions", "-0.20"): 1,
       ("sizes", "256"): 1, ("grid", "-0.20_l256"): 1}
check(S.lost_curves(internal, cur) == []
      and S.lost_curves(internal, {k: v for k, v in cur.items()
                                   if k != ("grid", "-0.20_l256")})
      == ["grid -0.20_l256"]
      and S.lost_curves(internal, {k: v for k, v in cur.items()
                                   if k != ("sizes", "256")},
                        256) == ["sizes 256"],
      "a documented case with no stored curve is named (the base-size "
      "'sizes' entry is not counted)")

# the stale-file note, through write_run_outputs on an empty run
root, extras = J(BOX, "stale", "run"), J(BOX, "stale", "x")
os.makedirs(root, exist_ok=True)
os.makedirs(extras, exist_ok=True)
for n in S.STALE_SWEEP_FILES[:2] + ("r2_boxes.png",):
    with open(J(root, n), "wb") as fh:
        fh.write(b"old")
os.makedirs(J(root, "l160"), exist_ok=True)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try:
        S.write_run_outputs(root, extras, {})
    except Exception as exc:
        print("raised", exc)
out = buf.getvalue()
note = [x for x in out.splitlines() if "no longer drawn" in x]
check(len(note) == 1 and "r2_sweep.png" in note[0] and "r2_boxes.png" in note[0]
      and "l160/" in note[0] and "r2_compositions.png" not in note[0] and "Archive/" in note[0]
      and all(os.path.isfile(J(root, n)) for n in
              S.STALE_SWEEP_FILES[:2] + ("r2_boxes.png",)),
      f"one note names the stale files and nothing is moved: {note}")
check(not S.stale_sweep_files(J(BOX, "c15"), n15), "a fresh folder has none")
# an image of an earlier draw that is no longer made is reported, not deleted
open(J(f15, "r2_sweep_per_size", "l384.png"), "wb").close()
check(S.stale_sweep_files(f15, n15) == ["r2_sweep_per_size/l384.png"]
      and os.path.isfile(J(f15, "r2_sweep_per_size", "l384.png")),
      "an image left from an earlier draw is named as stale and kept")

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
