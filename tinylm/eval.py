"""Metrics: verifier, bits-per-byte, instruction following, generation speed."""

import argparse
import json
import math
import re
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tinylm.generate import complete, generate
from tinylm.model import TinyLM
from tinylm.tokenizer import Tokenizer
from tinylm.train import get_device, load_checkpoint


def verify(words: list[str], story: str, ended: bool) -> dict[str, bool | int]:
    """Score one story. Only `success` counts; the other fields explain a failure."""
    text = story.lower().strip()
    tokens = re.findall(r"[a-z']+", text)
    found = sum(bool(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", text)) for w in words)
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    checks = {
        "all_words": found == len(words),
        "proper_end": ended and bool(re.search(r"[.!?][\"'”’]?$", text)),
        "long_enough": len(tokens) >= 50,
        "not_repetitive": bool(trigrams) and len(set(trigrams)) >= 0.8 * len(trigrams),
        "not_a_list": not any(set(words) <= set(tokens[i : i + 6]) for i in range(len(tokens))),
    }
    return {**checks, "words_found": found, "success": all(checks.values())}


def read_records(split: str, n: int, skip: int = 0, max_bytes: int | None = None) -> list[dict]:
    """Read n records with distinct instructions, optionally only short stories."""
    out = {}
    with open(f"data/{split}.jsonl", encoding="utf-8") as file:
        for line in islice(file, skip, None):
            if len(out) == n:
                break
            item = json.loads(line)
            if max_bytes is None or len(item["story"].encode()) <= max_bytes:
                out.setdefault(",".join(sorted(item["words"])), item)
    return list(out.values())


@torch.inference_mode()
def bits_per_byte(model: TinyLM, tok: Tokenizer, stories: list[str]) -> dict[str, float]:
    """Cost of compressing whole documents: comparable across tokenizers, unlike perplexity."""
    device, T = model.embed.weight.device, model.cfg.max_len
    nll, n_tokens = 0.0, 0
    for i in range(0, len(stories), 64):
        x = torch.full((len(stories[i : i + 64]), T), tok.eot)
        y = torch.full_like(x, -100)
        for j, story in enumerate(stories[i : i + 64]):
            ids = [tok.eot, *tok.encode(story), tok.eot]
            assert len(ids) <= T + 1, "document longer than the context"
            x[j, : len(ids) - 1], y[j, : len(ids) - 1] = torch.tensor(ids[:-1]), torch.tensor(ids[1:])
            n_tokens += len(ids) - 1
        logits = model(x.to(device)).flatten(0, 1)
        nll += F.cross_entropy(logits, y.to(device).flatten(), ignore_index=-100, reduction="sum").item()
    n_bytes = sum(len(story.encode()) for story in stories)
    return dict(bpb=nll / math.log(2) / n_bytes, perplexity=math.exp(nll / n_tokens))


def follow_rate(model: TinyLM, tok: Tokenizer, records: list[dict], seed: int = 0) -> dict:
    """Generate one story per instruction and verify it."""
    outputs = complete(model, tok, [tok.encode_prompt(r["words"]) for r in records], seed=seed)
    ended = [out[-1] == tok.eot for out in outputs]
    texts = [tok.decode(out[:-1] if end else out) for out, end in zip(outputs, ended)]
    scores = [verify(r["words"], text, e) for r, text, e in zip(records, texts, ended)]
    rates = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    samples = [dict(words=r["words"], story=text) for r, text in zip(records[:5], texts)]
    return dict(rates=rates, successes=[int(s["success"]) for s in scores],
                mean_tokens=float(np.mean([len(o) for o in outputs])), samples=samples)


def speed(model: TinyLM, tok: Tokenizer) -> list[dict]:
    """Greedy tokens/s with and without cache, for growing lengths."""
    idx, rows = torch.tensor([[tok.eot]], device=model.embed.weight.device), []
    c = model.cfg
    for n in (16, 32, 64, 128, 256):
        row = dict(new_tokens=n, cache_bytes=2 * c.n_layers * c.d_model * n * 4)  # heads * head_dim = d
        for name, use_cache in (("cache", True), ("no_cache", False)):
            times = []
            for _ in range(5):
                start = time.perf_counter()
                generate(model, idx, n, temperature=0, use_cache=use_cache)
                times.append(time.perf_counter() - start)
            row[name] = n / float(np.median(times))  # median: ignores the first (compiling) call
        rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bpb", "follow", "speed"))
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default=get_device())
    args = parser.parse_args()
    model, tok, _ = load_checkpoint(args.ckpt, args.device)
    if args.mode == "bpb":
        stories = read_records(args.split, 2000, max_bytes=600)
        result = bits_per_byte(model, tok, [r["story"] for r in stories])
    elif args.mode == "follow":
        result = follow_rate(model, tok, read_records(args.split, 2000))
    else:
        result = dict(device=args.device, rows=speed(model, tok))
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(args.out, result.get("rates", result))
