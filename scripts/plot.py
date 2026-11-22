"""Build every figure and the summary table from the JSON files in results/."""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
BLUE, ORANGE = "#2a78d6", "#eb6834"  # checked for colour-vision deficiency
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "axes.titlelocation": "left",
    "legend.frameon": False, "font.size": 11,
})


def load(name: str):
    return json.loads((RESULTS / f"{name}.json").read_text())


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}.png", dpi=160)
    plt.close(fig)


def short(text: str, n: int = 150) -> str:
    text = " ".join("".join(c if 32 <= ord(c) < 127 else " " for c in text).split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def learning() -> None:
    """The main chart: validation loss, with what the model writes at that point."""
    evals = load("train_base")["evals"]
    stories = {d["step"]: d["text"] for d in load("demo") if d["stage"] == "pretrain"}
    steps = [s for s in (0, 1000, 20000) if s in stories]
    fig = plt.figure(figsize=(12, 5.2))
    ax, side = fig.subplots(1, 2, width_ratios=(1, 1.1))
    x = [e["tokens"] / 1e6 for e in evals]
    ax.plot(x, [e["train"] for e in evals], color=ORANGE, label="training")
    ax.plot(x, [e["val"] for e in evals], color=BLUE, label="validation")
    for n, step in enumerate(steps, 1):
        e = next(e for e in evals if e["step"] == step)
        ax.plot(e["tokens"] / 1e6, e["val"], "o", ms=8, color=BLUE, mec=SURFACE, mew=2)
        ax.annotate(f"{n}", (e["tokens"] / 1e6, e["val"]), (9, -4), textcoords="offset points")
    ax.set(title="The model learns to write", xlabel="tokens seen (millions)",
           ylabel="loss (lower is better)", ylim=(0, 9.2))
    ax.legend(loc="upper right")
    side.axis("off")
    for n, step in enumerate(steps, 1):
        y = 1 - (n - 1) / len(steps)
        side.text(0, y, f"{n}. After {step:,} steps", va="top", weight="bold", transform=side.transAxes)
        lines = textwrap.wrap(short(stories[step], 190), 62)[:4]
        body = "\n".join(lines).replace("$", r"\$")
        side.text(0, y - 0.075, body, va="top", color=MUTED, fontsize=10, transform=side.transAxes)
    save(fig, "learning")


def tokenizer() -> None:
    rows = load("tokenizer")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([r["vocab"] for r in rows], [r["bytes_per_token"] for r in rows], "o-", ms=8, color=BLUE)
    used = next(r for r in rows if r["vocab"] == 4096)
    ax.annotate(f"{used['bytes_per_token']:.2f} (used)", (used["vocab"], used["bytes_per_token"]),
                (-70, 10), textcoords="offset points")
    ax.set_xscale("log", base=2)
    ax.set_xticks([r["vocab"] for r in rows], [str(r["vocab"]) for r in rows])
    ax.set(title="A bigger vocabulary compresses better", xlabel="vocabulary size",
           ylabel="bytes per token", ylim=(0, 4.8))
    save(fig, "tokenizer")


def speed() -> None:
    devices = [d for d in ("mps", "cpu") if (RESULTS / f"speed_{d}.json").exists()]
    fig, axes = plt.subplots(1, len(devices), figsize=(5.5 * len(devices), 4), squeeze=False)
    for ax, device in zip(axes[0], devices):
        rows = load(f"speed_{device}")["rows"]
        lengths = [r["new_tokens"] for r in rows]
        for key, label, color in (("cache", "with KV-cache", BLUE), ("no_cache", "without", ORANGE)):
            ax.plot(lengths, [r[key] for r in rows], "o-", ms=8, color=color, label=label)
        ax.set(title=f"Generation on {device.upper()}", xlabel="tokens generated",
               ylabel="tokens per second", ylim=0)
        ax.legend()
    save(fig, "speed")


def ablations() -> dict:
    """One dot per seed, one tick for the mean."""
    runs = {key: [load(f"bpb_abl_{key}_s{seed}")["bpb"] for seed in range(3)]
            for key in ("rope", "learned", "v1024")}
    panels = {"Positions": [("RoPE", "rope"), ("learned", "learned")],
              "Vocabulary": [("4096", "rope"), ("1024", "v1024")]}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (title, variants) in zip(axes, panels.items()):
        for x, ((_, key), color) in enumerate(zip(variants, (BLUE, ORANGE))):
            mean = np.mean(runs[key])
            ax.plot([x] * 3, runs[key], "o", ms=9, color=color, mec=SURFACE, mew=1.5)
            ax.hlines(mean, x - 0.15, x + 0.15, color=color)
            ax.annotate(f"{mean:.3f}", (x + 0.18, mean), va="center")
        ax.set_xticks(range(2), [label for label, _ in variants])
        ax.set(title=title, xlim=(-0.5, 1.6))
    axes[0].set_ylabel("test bits per byte (lower is better)")
    save(fig, "ablations")
    return {k: dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)), runs=v) for k, v in runs.items()}


def summary() -> dict:
    """Success rates with 95% intervals, and paired differences between stages."""
    out, wins = {}, {}
    for stage in ("base", "sft", "dpo"):
        result = load(f"follow_{stage}")
        wins[stage] = np.array(result["successes"], dtype=float)
        half = 1.96 * wins[stage].std(ddof=1) / len(wins[stage]) ** 0.5
        out[stage] = dict(n=len(wins[stage]), ci95=float(half), mean_tokens=result["mean_tokens"],
                          **result["rates"], **load(f"bpb_{stage}"))
    for a, b in (("base", "sft"), ("sft", "dpo")):
        diff = wins[b] - wins[a]  # same instructions, so a paired interval
        out[f"{b}_minus_{a}"] = dict(diff=float(diff.mean()),
                                     ci95=float(1.96 * diff.std(ddof=1) / len(diff) ** 0.5))
    return out


if __name__ == "__main__":
    learning()
    tokenizer()
    speed()
    report = dict(stages=summary(), ablations=ablations())
    (RESULTS / "summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
