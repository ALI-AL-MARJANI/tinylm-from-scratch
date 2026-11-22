"""The demo: the same prompt and seed at every stage, as JSON, a GIF, or typed in the terminal."""

import argparse
import json
import textwrap
import time
from pathlib import Path

import torch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont  # installed with matplotlib

from tinylm.eval import read_records, verify
from tinylm.generate import generate
from tinylm.train import load_checkpoint

START = "Once upon a time"
SEED = 0  # fixed in advance: the stories are not cherry-picked
DEMO = Path("results/demo.json")
SIZE = (960, 540)


def build() -> list[dict]:
    """Generate one story per checkpoint."""
    words, scenes = read_records("test", 1)[0]["words"], []
    steps = (0, 100, 1000, 3000, 10000, 20000)
    stages = [("pretrain", step, f"checkpoints/base/step_{step}.pt") for step in steps]
    stages += [("sft", None, "checkpoints/sft/final.pt"), ("dpo", None, "checkpoints/dpo/final.pt")]
    for stage, step, path in stages:
        model, tok, _ = load_checkpoint(path, "cpu")
        free = stage == "pretrain"
        prompt = [tok.eot, *tok.encode(START)] if free else tok.encode_prompt(words)
        out = generate(model, torch.tensor([prompt]), 120 if free else None, tok.eot, seed=SEED)[0]
        ended = out[-1] == tok.eot
        text = tok.decode(out[:-1] if ended else out)
        scene = dict(stage=stage, step=step, text=START + text if free else text)
        if free:
            scene["title"] = f"Pre-training: {step:,} steps"
        else:
            result = verify(words, text, ended)
            scene.update(words=words, checks=result)
            scene["title"] = f"After {stage.upper()}: use the words {', '.join(words)}"
            outcome = "pass" if result["success"] else "fail"
            scene["verdict"] = f"verifier: {result['words_found']}/3 words, {outcome}"
        scenes.append(scene)
    return scenes


def lines_of(text: str) -> list[str]:
    clean = "".join(c if c.isprintable() or c == "\n" else "." for c in text.replace("\r", "\n"))
    return [line for part in clean.split("\n") for line in textwrap.wrap(part, 86)][:17]


def render_gif(scenes: list[dict], path: Path) -> None:
    """Type each story out, then hold it long enough to be read."""
    regular = font_manager.findfont("DejaVu Sans")
    bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    body, head = ImageFont.truetype(regular, 17), ImageFont.truetype(bold, 19)
    frames, durations = [], []
    for scene in scenes:
        text = "\n".join(lines_of(scene["text"]))
        for i in range(1, 11):
            image = Image.new("RGB", SIZE, "#1a1a19")
            draw = ImageDraw.Draw(image)
            draw.text((28, 22), scene["title"], font=head, fill="#3987e5")
            draw.multiline_text((28, 68), text[: len(text) * i // 10], font=body, fill="#f0efec", spacing=6)
            if i == 10 and "verdict" in scene:
                draw.text((28, SIZE[1] - 36), scene["verdict"], font=body, fill="#c3c2b7")
            frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=16))
            durations.append(6000 if i == 10 else 350)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


def play(scenes: list[dict]) -> None:
    """Type the stories in the terminal, for a screen recording."""
    for scene in scenes:
        print(f"\033[2J\033[H\033[1;34m{scene['title']}\033[0m\n")
        for line in lines_of(scene["text"]):
            for char in line:
                print(char, end="", flush=True)
                time.sleep(0.006)
            print()
        print(f"\n{scene.get('verdict', '')}")
        time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--play", action="store_true", help="replay results/demo.json in the terminal")
    if parser.parse_args().play:
        play(json.loads(DEMO.read_text()))
    else:
        demo = build()
        DEMO.write_text(json.dumps(demo, indent=1, ensure_ascii=False))
        render_gif(demo, DEMO.with_suffix(".gif"))
