"""Draw the three README diagrams (pipeline, architecture, training stages) into results/."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RESULTS = Path("results")
INK, MUTED, SURFACE = "#1f1f1f", "#5f5e5a", "#ffffff"
# (fill, edge) pairs, paper style: soft fill, darker border
GREY = ("#eeeeec", "#8a8985")
BLUE = ("#dbe8fb", "#2a78d6")
ORANGE = ("#fde3d6", "#eb6834")
GREEN = ("#dcf1e3", "#2f9e5b")
PURPLE = ("#ebe2f8", "#7c4dcc")
YELLOW = ("#fdf3cf", "#c9a227")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "text.color": INK})


def canvas(w: float, h: float):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, color, sub=None, size=11, bold=True, lw=1.6, dashed=False):
    """A rounded box centred on (x, y)."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
        fc=color[0], ec=color[1], lw=lw, ls="--" if dashed else "-", zorder=2))
    dy = 0.13 if sub else 0
    ax.text(x, y + dy, text, ha="center", va="center", fontsize=size,
            weight="bold" if bold else "normal", zorder=3)
    if sub:
        ax.text(x, y - 0.17, sub, ha="center", va="center", fontsize=size - 2.5, color=MUTED, zorder=3)


def arrow(ax, a, b, color=INK, rad=0.0, lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=1))


def plus(ax, x, y):
    ax.add_patch(plt.Circle((x, y), 0.13, fc=SURFACE, ec=INK, lw=1.4, zorder=3))
    ax.text(x, y - 0.01, "+", ha="center", va="center", fontsize=13, zorder=4)


def save(fig, name):
    fig.savefig(RESULTS / f"{name}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def pipeline():
    """Text -> tokens -> transformer -> probabilities -> sampled token, in a loop."""
    fig, ax = canvas(12, 3.2)
    y, w, h = 1.85, 1.9, 1.0
    xs = [1.25, 3.55, 5.95, 8.35, 10.75]
    boxes = [("Text", GREY, '"Once upon a"'),
             ("Tokenizer", YELLOW, "text → token ids"),
             ("Transformer", BLUE, "6 layers, 5.9M params"),
             ("Probabilities", PURPLE, "one per token (4,096)"),
             ("Sampling", GREEN, "pick the next token")]
    for x, (t, c, s) in zip(xs, boxes):
        box(ax, x, y, w, h, t, c, s)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + w / 2 + 0.05, y), (b - w / 2 - 0.05, y))
    # loop back from sampling to the transformer
    ax.plot([xs[4], xs[4]], [y - h / 2, 0.55], color=GREEN[1], lw=1.4)
    ax.plot([xs[4], xs[2]], [0.55, 0.55], color=GREEN[1], lw=1.4)
    arrow(ax, (xs[2], 0.55), (xs[2], y - h / 2 - 0.03), color=GREEN[1])
    ax.text((xs[2] + xs[4]) / 2, 0.33, 'add the new token ("time") and repeat',
            ha="center", va="center", fontsize=9.5, color=GREEN[1], style="italic")
    save(fig, "diagram_pipeline")


def architecture():
    """Paper-style decoder: embedding, N x (attention + MLP) with residuals, output head."""
    fig, ax = canvas(6.4, 9.6)
    cx, w = 2.6, 2.6
    # bottom to top
    box(ax, cx, 0.55, w, 0.55, "Input tokens", GREY, size=10.5)
    box(ax, cx, 1.45, w, 0.6, "Token embedding", YELLOW, "4,096 × 256", size=10.5)
    arrow(ax, (cx, 0.83), (cx, 1.15))
    # repeated block frame
    ax.add_patch(FancyBboxPatch((cx - w / 2 - 0.45, 2.2), w + 0.9, 5.15,
                                boxstyle="round,pad=0,rounding_size=0.2",
                                fc="#f7f7f5", ec=MUTED, lw=1.2, ls="--", zorder=0))
    ax.text(0.42, 4.65, "× 6", fontsize=16, weight="bold", va="center", ha="center")
    ax.plot([cx, cx], [1.75, 2.4], color=INK, lw=1.4)

    def sub_block(y0, name, color, sub):
        # residual stream enters at y0; norm, layer, then add
        ax.plot([cx, cx], [y0 - 0.2, y0], color=INK, lw=1.4)
        box(ax, cx, y0 + 0.3, w, 0.42, "RMSNorm", GREY, size=10)
        arrow(ax, (cx, y0 + 0.51), (cx, y0 + 0.8))
        box(ax, cx, y0 + 1.15, w, 0.7, name, color, sub, size=10.5)
        arrow(ax, (cx, y0 + 1.5), (cx, y0 + 1.82))
        plus(ax, cx, y0 + 1.95)
        # residual skip on the left
        xl = cx - w / 2 - 0.25
        ax.plot([cx, xl, xl], [y0 - 0.05, y0 - 0.05, y0 + 1.95], color=INK, lw=1.4)
        arrow(ax, (xl, y0 + 1.95), (cx - 0.13, y0 + 1.95))
        return y0 + 2.08

    y = sub_block(2.6, "Causal self-attention", BLUE, "8 heads, RoPE")
    ax.plot([cx, cx], [y, y + 0.3], color=INK, lw=1.4)
    y = sub_block(y + 0.5, "SwiGLU MLP", ORANGE, "256 → 704 → 256")
    arrow(ax, (cx, y), (cx, 7.6))
    box(ax, cx, 7.85, w, 0.45, "RMSNorm", GREY, size=10)
    arrow(ax, (cx, 8.08), (cx, 8.35))
    box(ax, cx, 8.65, w, 0.55, "Linear (tied)", YELLOW, size=10.5)
    arrow(ax, (cx, 8.93), (cx, 9.2))
    ax.text(cx, 9.38, "next-token probabilities", ha="center", fontsize=10.5, weight="bold")
    # tied weights link
    xr = cx + w / 2 + 0.75
    ax.plot([cx + w / 2, xr, xr, cx + w / 2], [8.65, 8.65, 1.45, 1.45],
            color=YELLOW[1], lw=1.5, ls=(0, (2, 2)), zorder=1)
    ax.text(xr + 0.12, 5.05, "same weights", fontsize=9.5, color=YELLOW[1],
            rotation=90, ha="left", va="center")
    save(fig, "diagram_architecture")


def training():
    """The three training stages, with their data and the test score after each."""
    fig, ax = canvas(12, 3.3)
    y, w, h = 2.05, 3.0, 1.15
    stages = [("1. Pre-training", BLUE, "read 328M tokens of stories", "0.05%"),
              ("2. SFT", ORANGE, "copy good answers to orders", "30%"),
              ("3. DPO", GREEN, "prefer good answers over bad", "73%")]
    xs = [2.0, 6.0, 10.0]
    for x, (t, c, s, score) in zip(xs, stages):
        box(ax, x, y, w, h, t, c, s, size=12.5)
        ax.text(x, 0.95, score, ha="center", va="center", fontsize=17, weight="bold", color=c[1])
        ax.text(x, 0.52, "orders followed", ha="center", va="center", fontsize=9.5, color=MUTED)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + w / 2 + 0.08, y), (b - w / 2 - 0.08, y), lw=1.8)
    save(fig, "diagram_training")


if __name__ == "__main__":
    pipeline()
    architecture()
    training()
