"""One training loop for pre-training and SFT: only the batches and the targets differ."""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tinylm.model import ModelConfig, TinyLM
from tinylm.tokenizer import Tokenizer


@dataclass
class TrainConfig:
    """Every hyperparameter. Each field is also a command-line flag."""

    name: str = "base"
    stage: str = "pretrain"  # or "sft"
    data: str = "data/v4096"
    init: str = ""  # checkpoint to start from (SFT)
    steps: int = 20_000
    batch_size: int = 64
    lr: float = 1e-3
    warmup: int = 500
    weight_decay: float = 0.01  # on matrices only, never on norm gains
    grad_clip: float = 1.0
    eval_every: int = 1000
    eval_batches: int = 20
    save_at: str = "0,100,1000,3000,10000"  # kept for the demo
    seed: int = 0
    d_model: int = ModelConfig.d_model
    n_layers: int = ModelConfig.n_layers
    n_heads: int = ModelConfig.n_heads
    d_ff: int = ModelConfig.d_ff
    pos: str = ModelConfig.pos


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to 5% of the peak."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))


def load_split(cfg: TrainConfig, split: str) -> tuple[np.ndarray, np.ndarray | None]:
    root = Path(cfg.data)
    if cfg.stage == "pretrain":
        return np.memmap(root / f"{split}.bin", dtype=np.uint16, mode="r"), None
    return np.load(root / f"sft_{split}.npy", "r"), np.load(root / f"sft_{split}_spans.npy")


def get_batch(data: tuple, size: int, T: int, seed: tuple, device: str) -> tuple:
    """Return (inputs, targets). SFT targets are -100 on the prompt and on the padding."""
    tokens, spans = data
    rng = np.random.default_rng(seed)  # depends on the step, so resuming never replays batches
    if spans is None:
        starts = rng.integers(0, len(tokens) - T, size)
        batch = np.stack([tokens[i : i + T + 1] for i in starts]).astype(np.int64)
        x, y = batch[:, :-1], batch[:, 1:]
    else:
        rows = rng.integers(0, len(tokens), size)
        batch = np.asarray(tokens[rows]).astype(np.int64)
        position = np.arange(1, T + 1)  # position of each target in the sequence
        active = (position >= spans[rows, :1]) & (position < spans[rows, 1:])
        x, y = batch[:, :-1], np.where(active, batch[:, 1:], -100)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def compute_loss(model: TinyLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)


@torch.no_grad()
def evaluate(model: TinyLM, data: tuple, cfg: TrainConfig, device: str) -> float:
    total = 0.0
    for i in range(cfg.eval_batches):  # always the same batches
        x, y = get_batch(data, cfg.batch_size, model.cfg.max_len, (12345, i), device)
        total += compute_loss(model, x, y).item()
    return total / cfg.eval_batches


def load_checkpoint(path: str | Path, device: str) -> tuple[TinyLM, Tokenizer, dict]:
    """A checkpoint is self-contained: weights, architecture and tokenizer."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = TinyLM(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, Tokenizer([tuple(pair) for pair in ckpt["merges"]]), ckpt


def save_checkpoint(path: Path, model: TinyLM, tok: Tokenizer, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(model=model.state_dict(), config=asdict(model.cfg), merges=tok.merges, **extra)
    torch.save(state, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)  # atomic: a killed session cannot corrupt it


def train(cfg: TrainConfig) -> None:
    device, out = get_device(), Path("checkpoints") / cfg.name
    torch.manual_seed(cfg.seed)
    tok = Tokenizer.load(Path(cfg.data) / "tokenizer.json")
    shape = {k: getattr(cfg, k) for k in ("d_model", "n_layers", "n_heads", "d_ff", "pos")}
    model = TinyLM(ModelConfig(vocab_size=tok.vocab_size, **shape)).to(device)
    if cfg.init:
        model, tok, _ = load_checkpoint(cfg.init, device)
    decay = [p for p in model.parameters() if p.dim() > 1]
    no_decay = [p for p in model.parameters() if p.dim() == 1]
    groups = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, cfg.lr, (0.9, 0.95), weight_decay=cfg.weight_decay)
    step, log = 0, {"config": asdict(cfg), "params": model.num_params(), "seconds": 0.0}
    log.update(device=device, evals=[], steps=[])
    if (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, log = ckpt["step"], ckpt["log"]
        if step >= cfg.steps:
            return print(f"{cfg.name} is already trained")
    first, start, previous = step, time.time(), log["seconds"]
    train_data, val_data = load_split(cfg, "train"), load_split(cfg, "val")
    T, milestones = model.cfg.max_len, {int(s) for s in cfg.save_at.split(",") if s}
    while True:
        if step in milestones or step == cfg.steps:
            save_checkpoint(out / f"step_{step}.pt", model, tok)
        if step == cfg.steps or (step % cfg.eval_every == 0 and (step > first or step == 0)):
            losses = {s: evaluate(model, d, cfg, device)
                      for s, d in (("train", train_data), ("val", val_data))}
            log["seconds"] = previous + time.time() - start
            tokens = step * cfg.batch_size * T
            log["evals"].append(dict(step=step, tokens=tokens, seconds=log["seconds"], **losses))
            print("eval", log["evals"][-1], flush=True)
            Path("results").mkdir(exist_ok=True)
            Path(f"results/train_{cfg.name}.json").write_text(json.dumps(log, indent=1))
            state = dict(optimizer=optimizer.state_dict(), step=step, log=log)
            save_checkpoint(out / "last.pt", model, tok, **state)
        if step >= cfg.steps:
            return
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        x, y = get_batch(train_data, cfg.batch_size, T, (cfg.seed, step), device)
        loss = compute_loss(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if step % 100 == 0:
            entry = dict(step=step, loss=loss.item(), lr=lr_at(step, cfg), grad_norm=norm.item())
            assert math.isfinite(entry["loss"]), f"loss diverged: {entry}"
            log["steps"].append(entry)
            print(entry, flush=True)
        step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in fields(TrainConfig):
        parser.add_argument(f"--{field.name}", type=field.type, default=field.default)
    train(TrainConfig(**vars(parser.parse_args())))
