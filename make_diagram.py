"""Draw the FNO and the path a run takes through the project.

    python make_diagram.py              into Results/
    python make_diagram.py --out DIR    into DIR instead

Two figures, each as a PNG and a PDF (the flat layout's names, in the Results root):
    fno_architecture   the network: input channels, lift, the Fourier layers,
                       the projection and head, the Laplacian-output option,
                       the hard mass and zero-DC constraints, the low-k fill,
                       and for a dt-conditioned model how dt reaches it (log
                       dt embedding, small MLP, FiLM scale and shift in every
                       Fourier layer, the gain on the head's output)
    project_pipeline   the scripts and folders from the dataset to the index

Every number is read from fno_train: its constants, and the model that
build_model makes on the CPU for the parameter counts.  Folder and file names
come from fno_paths, fno_predict and make_submission.  No GPU is needed, and
the same code draws the same bytes.
"""
from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

import fno_paths
import fno_predict
import fno_train as T
import make_submission

# The parameter count does not depend on this, so the model built to count
# them is a small one.
N_REF = 16
STEMS = ("fno_architecture", "project_pipeline")
EXTS = ("png", "pdf")
FILES = tuple(f"{stem}.{ext}" for stem in STEMS for ext in EXTS)
DPI = 150
WIDTH = 200.0

PAL = {
    "data": ("#e4eef9", "#2b6cb0"),
    "program": ("#e3f2e1", "#2f7d4f"),
    "model": ("#ece5f8", "#5b3fa8"),
    "key": ("#fde6e6", "#b42323"),
    "option": ("#f3f4f6", "#6b7280"),
    "out": ("#fff1dc", "#b4651b"),
}
INK = "#1f2937"
GREY = "#6b7280"


def posix(path):
    return path.replace(os.sep, "/")


def pretty(text):
    """ASCII in the labels, typographic symbols on the page."""
    for a, b in ((" -> ", " \u2192 "), ("psi", "\u03c8"),
                 (">=", "\u2265"), ("<=", "\u2264")):
        text = text.replace(a, b)
    return re.sub(r"(?<=\d)x(?=\d)", "\u00d7", text)


def count(module):
    return sum(p.numel() for p in module.parameters())


def convs(module):
    return [m for m in module.modules() if isinstance(m, nn.Conv2d)]


def facts():
    """What the figures say, read from fno_train.  The model is built on the
    CPU and the random state is put back, so drawing a figure does not
    change a run that follows."""
    with torch.random.fork_rng(devices=[]):
        model = T.build_model(n_ref=N_REF, device="cpu")
    body = model.backbone
    lift, head = convs(body.lift), convs(body.project)
    blocks = [b for b in body.blocks if isinstance(b, T.FNOBlock)]
    cfg = model.config
    film = blocks[0].film if blocks else None
    return {
        "params": model.n_parameters(),
        "params_lift": count(body.lift),
        "params_layer": count(blocks[0]) if blocks else 0,
        "params_head": count(body.project),
        "params_cond": count(model.cond) if model.cond is not None else 0,
        "params_film": count(film) if film is not None else 0,
        "in_ch": lift[0].in_channels,
        "channels": lift[-1].out_channels,
        "head_hidden": head[0].out_channels,
        "head_out": head[-1].out_channels,
        "layers": len(blocks),
        "modes": cfg["modes"],
        "kernel": cfg["local_kernel"],
        "mode_scaling": cfg["mode_scaling"],
        "mean_channel": cfg["use_mean_channel"],
        "mu_channel": cfg["mu_channel"],
        "zero_dc": cfg["zero_dc"],
        "hard_mass": cfg["hard_mass_conservation"],
        "residual": cfg["residual_prediction"],
        "laplacian": cfg["laplacian_output"],
        "lowk_fill": cfg["lowk_fill"],
        "mu_clip": cfg.get("mu_clip"),
        "dt_conditioned": cfg["dt_conditioned"],
        "cond_dim": cfg["cond_dim"],
        "emb_dim": getattr(model, "cond_in", 0),
        "dt_min": T.DT_MIN,
        "dt_max": T.DT_MAX,
        "dt_step": T.DT_STEP,
        "frame_gap": T.DT * T.GAP,
        "dataset": os.path.basename(T.DATASET),
        "epochs": T.EPOCHS,
        "steps": T.STEPS_PER_EPOCH,
        "batch": T.BATCH_SIZE,
        "unroll": T.MAX_UNROLL,
        "precision": T.PRECISION,
        "select": T.SELECT,
        "start": fno_paths.DEFAULT_START,
    }


def state(flag):
    return "ON" if flag else "off"


def architecture_labels(f):
    """The text of the architecture figure.  The first line of each is the
    box's title."""
    c, m, k = f["channels"], f["modes"], f["kernel"]
    n_in = f["in_ch"]
    scaling = ("The modes follow the physical wavenumbers, so the same "
               "weights run on any box size."
               if f["mode_scaling"] == "physical" else
               "The same number of modes is used at every box size.")
    cond = f["dt_conditioned"]
    inputs = (["psi(t)"]
              + (["mean of psi over the box"] if f["mean_channel"] else [])
              + (["mu = psi^3 - psi - lap(psi)"] if f["mu_channel"] else [])
              + ([f"(mu of psi clipped to |psi| <= {f['mu_clip']:g})"]
                 if f["mu_channel"] and f["mu_clip"] is not None else []))
    fill = bool(f["lowk_fill"] and f["zero_dc"])
    dt_labels = {}
    if cond:
        half = (f["emb_dim"] - 1) // 2
        dt_labels = {
            "dt_in": "\n".join([
                "Step size dt",
                f"{f['dt_min']:g} to {f['dt_max']:g} time units",
                "given at every call"]),
            "dt_embed": "\n".join([
                "Embedding of log dt",
                f"sin, cos at {half} log-spaced",
                "frequencies, and log dt",
                f"{f['emb_dim']} numbers"]),
            "dt_mlp": "\n".join([
                "Small MLP",
                f"linear {f['emb_dim']} -> {f['cond_dim']}, GELU",
                f"linear {f['cond_dim']} -> {f['cond_dim']}, GELU",
                f"{f['params_cond']:,} parameters"]),
            "dt_gain": "\n".join([
                "Output gain",
                "exp(log_gain + gain_exp * log dt)",
                "starts at dt: log_gain 0, gain_exp 1",
                "2 parameters"]),
            "film_arrow": "FiLM: scale and shift per layer",
            "gain_arrow": "multiplies the head output",
            "log_dt_arrow": "log dt",
            "film": "\n".join([
                "FiLM", "h (1 + g) + b", "g, b per channel", "start at zero",
                f"{f['params_film']:,} parameters"]),
            "film_in": f"dt features ({f['cond_dim']} numbers)",
        }
    return {
        **dt_labels,
        "arch_title": f"FNO architecture: {f['params']:,} parameters",
        "arch_note": (
            (f"One call maps psi(t) to psi(t + dt): dt is given at each call,"
             f" from {f['dt_min']:g} to {f['dt_max']:g} time units.\n"
             if cond else
             f"One call maps psi(t) to psi(t + {f['dt_step']:g}).  ")
            + scaling),
        "input": "\n".join(
            [f"Input: {n_in} channel{'s' * (n_in != 1)}"] + inputs),
        "lift": "\n".join([
            "Lift", f"1x1 conv {n_in} -> {c}", "GELU",
            f"1x1 conv {c} -> {c}", f"{f['params_lift']:,} parameters"]),
        "fourier": "\n".join([
            f"{f['layers']} Fourier layers", f"{c} channels",
            f"{m} modes", f"{k}x{k} local conv",
            f"{f['params_layer']:,} parameters each"]),
        "project": "\n".join([
            "Projection and head",
            f"1x1 conv {c} -> {f['head_hidden']}", "GELU",
            f"1x1 conv {f['head_hidden']} -> {f['head_out']}",
            "starts at zero", f"{f['params_head']:,} parameters"]),
        "laplacian": "\n".join([
            f"Laplacian output: {state(f['laplacian'])}",
            *(["the head's output goes through",
               "the solver's Laplacian stencil"] if f["laplacian"] else
              ["the head's output is used", "as the increment, as it is"])]),
        "mass": "\n".join([
            f"Hard mass constraint: {state(f['hard_mass'])}",
            "the mean of the increment is removed,",
            "so the box mean of psi cannot change" if f["hard_mass"] else
            "so the box mean of psi may drift"]),
        "residual": "\n".join([
            f"Residual step: {state(f['residual'])}",
            "psi(t) + increment" if f["residual"] else
            "the output is the new psi itself"]),
        "output": ("Output\npsi(t + dt)" if cond else
                   f"Output\npsi(t + {f['dt_step']:g})"),
        "layer_title": f"Inside one of the {f['layers']} Fourier layers",
        "spectral": "\n".join([
            "Spectral conv", f"rFFT2, keep {m} modes",
            f"{c}x{c} complex weights", f"on {2 * m + 1}x{m + 1} "
            "wavenumbers", "irFFT2"]),
        "zero_dc": "\n".join([
            f"Zero-DC constraint: {state(f['zero_dc'])}",
            "the k = 0 weight of every",
            "spectral conv is set to zero" if f["zero_dc"] else
            "spectral conv is left free"]),
        "lowk_fill": "\n".join([
            f"Low-k fill: {state(fill)}",
            *(["on a larger box, the k = 0 weight is filled",
               "with the mean of its neighbours before the",
               "weights are resampled, then zeroed again"] if fill else
              ["the weights are resampled as trained,",
               "with the k = 0 weight left at zero"])]),
        "local": f"Local conv\n{k}x{k}, circular padding",
        "mix": f"1x1 conv\n{c} -> {c}",
        "gelu": "GELU",
    }


def pipeline_labels(f):
    """The text of the pipeline figure, by stage."""
    return {
        "pipe_title": "From the dataset to the index",
        "pipe_note": "Paths are relative to the project root.",
        "gen": "\n".join([
            "1  ch_gpu.py", "Cahn-Hilliard solver",
            "on the GPU", f"frames every {f['frame_gap']:g}"]),
        "data": "\n".join([
            posix(fno_paths.data_dir()), f["dataset"],
            "train and validation", "trajectories"]),
        "train": "\n".join([
            f"2  {os.path.basename(T.__file__)}",
            f"{f['epochs']} epochs of {f['steps']} steps",
            f"batch {f['batch']}, unroll up to {f['unroll']}",
            f"{f['precision']}, best by {f['select']}"]),
        "models": "\n".join([
            posix(fno_paths.model_dir("<label>")),
            f"{fno_paths.MODEL_FILE}  (best)",
            f"{fno_paths.STATE_FILE}  (to resume)"]),
        "predict": "\n".join([
            f"3  {os.path.basename(fno_predict.__file__)}",
            f"rolls out from t = {f['start']:g}",
            "against the solver's truth",
            f"truth kept in {posix(fno_paths.gt_dir())}"]),
        "rollouts": "\n".join([
            posix(fno_paths.rollout_root()),
            "rollouts while a run goes",
            f"{fno_predict.RECORDS_DIR}/ keeps a record",
            "of every seed"]),
        "submit": "\n".join([
            f"4  {os.path.basename(make_submission.__file__)}",
            "figures, tables, report",
            "from the cache or the records"]),
        "results": "\n".join([
            posix(fno_paths.results_dir("<label>")),
            make_submission.MANIFEST, "figures and tables"]),
        "index": "\n".join([
            f"5  {os.path.basename(make_submission.__file__)}",
            "--overview", "one page over all the runs"]),
        "page": "\n".join([
            posix(fno_paths.results_root()),
            make_submission.INDEX_FILE, make_submission.COMPARISON_FILE]),
    }


def labels(f=None):
    """Every piece of text on both figures, by name."""
    f = facts() if f is None else f
    return {**architecture_labels(f), **pipeline_labels(f)}


def box(ax, x, y, w, h, text, kind, dashed=False, fs=10.4):
    face, edge = PAL[kind]
    title, *lines = [pretty(t) for t in text.split("\n")]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=face,
        ec=edge, lw=1.4, ls="--" if dashed else "-", zorder=2))
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top",
            fontsize=fs + 1.2, fontweight="bold", color=edge, zorder=3)
    if lines:
        ax.text(x + w / 2, y + h - 4.8, "\n".join(lines), ha="center",
                va="top", fontsize=fs, color=INK, linespacing=1.4, zorder=3)


def arrow(ax, p, q, color=INK, lw=1.4, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=12, lw=lw, color=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=1.0,
        shrinkB=1.0, zorder=4))


def route(ax, points, color=INK, lw=1.4, ls="-"):
    """A line through ``points`` that ends in an arrowhead."""
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=4,
            solid_capstyle="round")
    arrow(ax, points[-2], points[-1], color=color, lw=lw, ls=ls)


def plus(ax, x, y):
    ax.add_patch(Circle((x, y), 1.8, fc="white", ec=INK, lw=1.3, zorder=5))
    ax.text(x, y - 0.1, "+", ha="center", va="center", fontsize=11,
            fontweight="bold", color=INK, zorder=6)


def note(ax, x, y, text, fs=10.4, color=GREY, ha="center", bold=False):
    ax.text(x, y, pretty(text), ha=ha, va="center", fontsize=fs,
            color=color, fontweight="bold" if bold else "normal", zorder=5)


def canvas(height):
    fig = plt.figure(figsize=(18.0, 18.0 * height / WIDTH))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def on_or_dashed(flag):
    return "key" if flag else "option"


def draw_architecture(L, f):
    cond = f["dt_conditioned"]
    up = 28.0 if cond else 0.0          # the rows of the dt conditioning
    fig, ax = canvas(130.0 + up)
    note(ax, 100, 125.5 + up, L["arch_title"], fs=16, color=INK, bold=True)
    note(ax, 100, 119.5 + up + (1.0 if cond else 0.0), L["arch_note"], fs=11)

    ya, ha = 92.0, 19.0
    box(ax, 2, ya, 34, ha, L["input"], "data")
    box(ax, 46, ya, 34, ha, L["lift"], "model")
    box(ax, 90, ya, 46, ha, L["fourier"], "model")
    box(ax, 146, ya, 52, ha, L["project"], "model")
    for x0, x1 in ((36, 46), (80, 90), (136, 146)):
        arrow(ax, (x0, ya + ha / 2), (x1, ya + ha / 2))

    if cond:
        # dt: embedded, through a small MLP, into every Fourier layer as a
        # scale and shift; and, as log dt, into the gain on the head
        y0, h0 = ya + ha + 9.0, 16.0
        box(ax, 2, y0, 34, h0, L["dt_in"], "data")
        box(ax, 46, y0, 34, h0, L["dt_embed"], "model")
        box(ax, 90, y0, 46, h0, L["dt_mlp"], "model")
        box(ax, 146, y0, 52, h0, L["dt_gain"], "model")
        for x0, x1 in ((36, 46), (80, 90)):
            arrow(ax, (x0, y0 + h0 / 2), (x1, y0 + h0 / 2))
        arrow(ax, (113, y0), (113, ya + ha))
        note(ax, 115.5, y0 - 4.5, L["film_arrow"], fs=10, ha="left")
        arrow(ax, (172, y0), (172, ya + ha))
        note(ax, 174.5, y0 - 4.5, L["gain_arrow"], fs=10, ha="left")
        yr = y0 + h0 + 3.5
        route(ax, [(19, y0 + h0), (19, yr), (172, yr), (172, y0 + h0)],
              color=PAL["data"][1])
        note(ax, 95, yr + 2.0, L["log_dt_arrow"], fs=10,
             color=PAL["data"][1])

    yb, hb = 65.0, 15.0
    box(ax, 146, yb, 52, hb, L["laplacian"], "model" if f["laplacian"]
        else "option", dashed=not f["laplacian"])
    box(ax, 88, yb, 52, hb, L["mass"], on_or_dashed(f["hard_mass"]),
        dashed=not f["hard_mass"])
    box(ax, 44, yb, 36, hb, L["residual"], "model" if f["residual"]
        else "option", dashed=not f["residual"])
    box(ax, 2, yb, 34, hb, L["output"], "out")
    arrow(ax, (172, ya), (172, yb + hb))
    for x0, x1 in ((146, 140), (88, 80), (44, 36)):
        arrow(ax, (x0, yb + hb / 2), (x1, yb + hb / 2))
    route(ax, [(19, ya), (19, ya - 6.0), (62, ya - 6.0), (62, yb + hb)],
          color=GREY, ls="--")
    note(ax, 41, ya - 3.8, "psi(t) is kept for the residual step", fs=10)

    ax.add_patch(FancyBboxPatch(
        (1, 2), 198, 58, boxstyle="round,pad=0,rounding_size=1.5",
        fc="#fafafa", ec="#c3c8d0", lw=1.0, ls="--", zorder=1))
    note(ax, 4, 56.5, L["layer_title"], fs=11.5, color=INK, ha="left",
         bold=True)
    note(ax, 62, 56.5, "dashed grey: a skip, the input is added back at the "
         "end", fs=10, ha="left")
    note(ax, 118, 56.5, "red: a hard constraint, on.  dashed box: an option "
         "that is off", fs=10, ha="left", color=PAL["key"][1])
    yc = 21.0
    note(ax, 5, yc, "x", fs=13, color=INK)
    box(ax, 14, 25, 48, 17, L["spectral"], "model")
    box(ax, 14, 5, 48, 11, L["local"], "model")
    box(ax, 68, 33, 40, 14, L["zero_dc"], on_or_dashed(f["zero_dc"]),
        dashed=not f["zero_dc"])
    ax.plot([62, 68], [39, 39], color=PAL["key"][1] if f["zero_dc"]
            else GREY, lw=1.2, ls="--", zorder=3)
    fill = bool(f["lowk_fill"] and f["zero_dc"])
    box(ax, 112, 32, 52, 15, L["lowk_fill"], "model" if fill else "option",
        dashed=not fill)
    ax.plot([108, 112], [39, 39], color=GREY, lw=1.2, ls="--", zorder=3)
    plus(ax, 70, yc)
    arrow(ax, (62, 33.5), (68.7, yc + 1.3))
    arrow(ax, (62, 10.5), (68.7, yc - 1.3))
    arrow(ax, (7, yc), (14, 33.5))
    arrow(ax, (7, yc), (14, 10.5))
    if cond:
        box(ax, 76, yc - 8.5, 30, 17, L["film"], "model")
        arrow(ax, (71.8, yc), (76, yc))
        arrow(ax, (91, 9.5), (91, yc - 8.5), color=PAL["data"][1])
        note(ax, 91, 6.5, L["film_in"], fs=10, color=PAL["data"][1])
        arrow(ax, (106, yc), (112, yc))
    else:
        arrow(ax, (71.8, yc), (112, yc))
    box(ax, 112, 16, 18, 10, L["gelu"], "model")
    box(ax, 136, 14, 24, 14, L["mix"], "model")
    plus(ax, 168, yc)
    note(ax, 176, yc, "next layer", fs=10.4, ha="left")
    arrow(ax, (130, yc), (136, yc))
    arrow(ax, (160, yc), (166.2, yc))
    arrow(ax, (169.8, yc), (174, yc))
    route(ax, [(5, yc + 2), (5, 51.5), (168, 51.5), (168, yc + 1.8)],
          color=GREY, ls="--")
    return fig


def draw_pipeline(L):
    fig, ax = canvas(72.0)
    note(ax, 100, 68, L["pipe_title"], fs=16, color=INK, bold=True)
    note(ax, 100, 62.5, L["pipe_note"], fs=11)
    w, gap = 32.0, 9.0
    xs = [2 + i * (w + gap) for i in range(5)]
    yp, hp = 38.0, 17.0
    ys, hs = 10.0, 15.0
    stages = (("gen", "data"), ("train", "models"), ("predict", "rollouts"),
              ("submit", "results"), ("index", "page"))
    for x, (prog, store) in zip(xs, stages):
        box(ax, x, yp, w, hp, L[prog], "program")
        box(ax, x, ys, w, hs, L[store], "out" if store in ("results", "page")
            else "data")
        arrow(ax, (x + w / 2, yp), (x + w / 2, ys + hs))
    for x0, x1 in zip(xs[:-1], xs[1:]):
        arrow(ax, (x0 + w, ys + hs / 2), (x1, yp + hp / 2), rad=-0.15)
    note(ax, 33, 4, "green: a script", color=PAL["program"][1])
    note(ax, 100, 4, "blue: a folder under Work", color=PAL["data"][1])
    note(ax, 167, 4, "orange: under Results", color=PAL["out"][1])
    return fig


def save(fig, out, stem):
    """The figure as <stem>.png and <stem>.pdf in ``out``; returns both
    paths.  The PDF carries no date, so a figure drawn twice is the same
    bytes."""
    paths = []
    for ext in EXTS:
        path = os.path.join(out, f"{stem}.{ext}")
        if ext == "pdf":
            fig.savefig(path, facecolor="white",
                        metadata={"CreationDate": None, "ModDate": None})
        else:
            fig.savefig(path, dpi=DPI, facecolor="white")
        print(f"[saved] {path}", flush=True)
        paths.append(path)
    plt.close(fig)
    return paths


def default_out():
    return fno_paths.results_root()


def write(out=None, f=None):
    """Draw both figures into ``out`` (the Results root by default); returns
    the four paths: fno_architecture.png, .pdf, project_pipeline.png, .pdf
    (FILES)."""
    out = default_out() if out is None else out
    f = facts() if f is None else f
    L = labels(f)
    os.makedirs(out, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        paths = save(draw_architecture(L, f), out, STEMS[0])
        paths += save(draw_pipeline(L), out, STEMS[1])
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Draw the FNO architecture and the project's pipeline.")
    ap.add_argument("--out", metavar="DIR", default=None,
                    help=f"where to write the figures (default "
                         f"{default_out()})")
    write(ap.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
