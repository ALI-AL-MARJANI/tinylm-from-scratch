"""Sampling with temperature / top-k / top-p, and a KV-cache."""

import argparse
from collections import defaultdict

import torch

from tinylm.model import TinyLM
from tinylm.tokenizer import Tokenizer
from tinylm.train import get_device, load_checkpoint


def next_token(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float, rng: torch.Generator
) -> torch.Tensor:
    """Pick one token per row of (B, vocab) logits."""
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        kth = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(-1)
    if top_p < 1:
        ranked, order = probs.sort(descending=True)
        ranked[ranked.cumsum(-1) - ranked >= top_p] = 0  # keep the token that crosses top_p
        probs = torch.zeros_like(probs).scatter(-1, order, ranked)
    return torch.multinomial(probs, 1, generator=rng)


@torch.inference_mode()
def generate(
    model: TinyLM, idx: torch.Tensor, max_new: int | None = None, eot: int | None = None,
    temperature: float = 0.8, top_k: int = 50, top_p: float = 1.0,
    seed: int = 0, use_cache: bool = True,
) -> list[list[int]]:
    """Continue a batch of same-length prompts. Each result stops right after `eot`."""
    room = model.cfg.max_len + 1 - idx.shape[1]  # the last token is never fed back
    max_new = room if max_new is None else min(max_new, room)
    rng = torch.Generator().manual_seed(seed)  # on CPU: same seed, same draws on every device
    cache = [None] * len(model.blocks) if use_cache else None
    done = torch.zeros(len(idx), dtype=torch.bool)
    new, feed = [], idx
    for _ in range(max_new):
        logits = model(feed, cache)[:, -1].float().cpu()
        token = next_token(logits, temperature, top_k, top_p, rng)
        new.append(token)
        if eot is not None:
            done |= token[:, 0] == eot
            if done.all():
                break
        token = token.to(idx.device)
        feed = token if use_cache else torch.cat((feed, token), dim=1)  # cache: only the new token
    rows = torch.cat(new, dim=1).tolist()
    return [row[: row.index(eot) + 1] if eot in row else row for row in rows]


def complete(
    model: TinyLM, tok: Tokenizer, prompts: list[list[int]], batch: int = 128, seed: int = 0, **kw
) -> list[list[int]]:
    """Complete prompts of any length by batching equal lengths together (so no padding)."""
    groups, out = defaultdict(list), [[] for _ in prompts]
    for i, prompt in enumerate(prompts):
        groups[len(prompt)].append(i)
    chunks = [g[i : i + batch] for g in groups.values() for i in range(0, len(g), batch)]
    for n, chunk in enumerate(chunks):
        idx = torch.tensor([prompts[i] for i in chunk], device=model.embed.weight.device)
        for i, row in zip(chunk, generate(model, idx, eot=tok.eot, seed=seed * 100_003 + n, **kw)):
            out[i] = row
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--words", default="", help="comma-separated, empty for a free story")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    model, tok, _ = load_checkpoint(args.ckpt, get_device())
    words = [w.strip() for w in args.words.split(",") if w.strip()]
    prompt = tok.encode_prompt(words) if words else [tok.eot]
    print(tok.decode(complete(model, tok, [prompt], seed=args.seed)[0]))
