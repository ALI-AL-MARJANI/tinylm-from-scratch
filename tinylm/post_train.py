"""DPO: prefer sampled stories that pass the verifier over sampled stories that fail it."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tinylm.eval import bits_per_byte, follow_rate, read_records, verify
from tinylm.generate import complete
from tinylm.model import TinyLM
from tinylm.tokenizer import Tokenizer
from tinylm.train import get_device, load_checkpoint, save_checkpoint


def make_pairs(model: TinyLM, tok: Tokenizer, records: list[dict], samples: int) -> tuple[list, dict]:
    """Sample each prompt several times, then keep one success and one failure."""
    prompts = [tok.encode_prompt(r["words"]) for r in records for _ in range(samples)]
    outputs = complete(model, tok, prompts, seed=1)
    pairs, n_success = [], 0
    for i, record in enumerate(records):
        wins, fails = [], []
        for out in outputs[i * samples : (i + 1) * samples]:
            ended = out[-1] == tok.eot
            ok = verify(record["words"], tok.decode(out[:-1] if ended else out), ended)["success"]
            (wins if ok else fails).append(out)
        n_success += len(wins)
        if wins and fails:  # log-probs are sums over tokens, so match the lengths
            chosen = wins[0]
            rejected = min(fails, key=lambda out: abs(len(out) - len(chosen)))
            pairs.append(dict(prompt=prompts[i * samples], chosen=chosen, rejected=rejected))
    stats = dict(prompts=len(records), samples=samples, pairs=len(pairs),
                 sample_success=n_success / len(prompts))
    return pairs, stats


def to_tensors(pairs: list[dict], key: str, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad prompt + answer. Targets are -100 everywhere except on the answer."""
    x = torch.zeros(len(pairs), T, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, pair in enumerate(pairs):
        ids = torch.tensor(pair["prompt"] + pair[key])
        start = len(pair["prompt"]) - 1  # the last prompt token predicts the first answer token
        x[i, : len(ids) - 1] = ids[:-1]
        y[i, start : len(ids) - 1] = ids[start + 1 :]
    return x, y


def answer_logp(model: TinyLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """log p(answer | prompt) for each row: a sum over the answer tokens."""
    logits = model(x).flatten(0, 1)
    nll = F.cross_entropy(logits, y.flatten(), ignore_index=-100, reduction="none")
    return -nll.view(y.shape).sum(1)


def dpo_loss(
    chosen: torch.Tensor, rejected: torch.Tensor,
    ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """-log sigmoid(beta * (how much more we moved towards chosen than towards rejected))."""
    margin = (chosen - ref_chosen) - (rejected - ref_rejected)
    return -F.logsigmoid(beta * margin).mean(), margin


def train_dpo(args: argparse.Namespace) -> None:
    device, out = get_device(), Path("checkpoints") / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model, tok, _ = load_checkpoint(args.init, device)
    if not Path(args.pairs).exists():
        records = read_records("train", args.prompts, skip=400_000)  # lines never used as SFT targets
        pairs, stats = make_pairs(model, tok, records, args.samples)
        Path(args.pairs).write_text(json.dumps(dict(stats=stats, pairs=pairs)))
    saved = json.loads(Path(args.pairs).read_text())
    pairs, T = saved["pairs"], model.cfg.max_len
    random.Random(args.seed).shuffle(pairs)
    data = [t.to(device) for key in ("chosen", "rejected") for t in to_tensors(pairs, key, T)]
    with torch.no_grad():  # the reference never changes: compute it once, keep no second model
        refs = [torch.cat([answer_logp(model, x[i : i + 64], y[i : i + 64])
                           for i in range(0, len(pairs), 64)])
                for x, y in (data[:2], data[2:])]
    val = read_records("val", 1000)
    short = [r["story"] for r in read_records("val", 1000, max_bytes=600)]
    log = dict(config=vars(args), pair_stats=saved["stats"], epochs=[], steps=[])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    rng, best, step = np.random.default_rng(args.seed), -1.0, 0
    for epoch in range(args.epochs + 1):  # epoch 0 is the untouched SFT model
        if epoch:
            order = torch.from_numpy(rng.permutation(len(pairs))).to(device)
            for i in range(0, len(pairs) - args.batch_size + 1, args.batch_size):
                rows = order[i : i + args.batch_size]
                xc, yc, xr, yr = (t[rows] for t in data)
                logp = answer_logp(model, torch.cat((xc, xr)), torch.cat((yc, yr))).chunk(2)
                loss, margin = dpo_loss(*logp, refs[0][rows], refs[1][rows], args.beta)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step % 10 == 0:
                    log["steps"].append(dict(step=step, loss=loss.item(), margin=margin.mean().item()))
                step += 1
        result = follow_rate(model, tok, val)
        rates = result["rates"]
        log["epochs"].append(dict(epoch=epoch, step=step, mean_tokens=result["mean_tokens"],
                                  **rates, **bits_per_byte(model, tok, short)))
        print(log["epochs"][-1], flush=True)
        if rates["success"] > best:
            best, log["best_epoch"] = rates["success"], epoch
            save_checkpoint(out / "best.pt", model, tok)
        Path(f"results/{args.name}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--name", default="dpo")
    parser.add_argument("--pairs", default="data/dpo_pairs.json")
    for flag, default in dict(prompts=5000, samples=6, epochs=4, batch_size=16, seed=0).items():
        parser.add_argument(f"--{flag}", type=int, default=default)
    for flag, default in dict(beta=0.1, lr=1e-5).items():
        parser.add_argument(f"--{flag}", type=float, default=default)
    train_dpo(parser.parse_args())
