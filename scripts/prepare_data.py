"""Data pipeline: download, split by instruction, train the tokenizer, save uint16 shards."""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np

from tinylm.eval import verify
from tinylm.model import ModelConfig
from tinylm.tokenizer import Tokenizer

REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"  # pinned dataset commit
URL = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}/TinyStories_all_data.tar.gz"
DATA = Path("data")
SPLITS = ("train", "val", "test")


def split_of(words: list[str]) -> str:
    """Map a word triplet to a split. Sorting first keeps permutations on the same side."""
    digest = hashlib.sha256(",".join(sorted(words)).encode()).hexdigest()
    bucket = int(digest, 16) % 100
    return "val" if bucket < 2 else "test" if bucket < 4 else "train"


def download(n_shards: int) -> None:
    """Stream the archive and keep GPT-4 stories with their words, without storing the archive."""
    DATA.mkdir(exist_ok=True)
    files = {s: open(DATA / f"{s}.jsonl.tmp", "w", encoding="utf-8") for s in SPLITS}
    counts = Counter()
    with urllib.request.urlopen(URL, timeout=120) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if not member.name.endswith(".json"):
                    continue
                for item in json.load(archive.extractfile(member)):
                    story = item["story"].strip()
                    if item["source"] != "GPT-4" or not story:
                        continue
                    words = [w.lower() for w in item["instruction"]["words"]]
                    split = split_of(words)
                    files[split].write(json.dumps({"words": words, "story": story}) + "\n")
                    counts[split] += 1
                counts["shards"] += 1
                print(member.name, dict(counts), flush=True)
                if counts["shards"] >= n_shards:
                    break
    for split, file in files.items():
        file.close()
        (DATA / f"{split}.jsonl.tmp").replace(DATA / f"{split}.jsonl")


def read(split: str, limit: int | None = None):
    """Yield the records of one split."""
    with open(DATA / f"{split}.jsonl", encoding="utf-8") as file:
        for line in islice(file, limit):
            yield json.loads(line)


def encode_stream(tok: Tokenizer, split: str, out: Path) -> None:
    """Save EOT story EOT story ... EOT as one uint16 stream."""
    with open(out / f"{split}.bin", "wb") as file:
        np.array([tok.eot], dtype=np.uint16).tofile(file)
        for item in read(split):
            np.array([*tok.encode(item["story"]), tok.eot], dtype=np.uint16).tofile(file)


def encode_sft(tok: Tokenizer, split: str, out: Path, limit: int) -> None:
    """Keep stories that pass the verifier and fit, so SFT teaches exactly what is measured."""
    width = ModelConfig.max_len + 1  # +1 because the inputs and targets are shifted
    rows, spans, seen = [], [], 0
    for item in read(split):
        seen += 1
        if not verify(item["words"], item["story"], ended=True)["success"]:
            continue
        prompt = tok.encode_prompt(item["words"])
        ids = [*prompt, *tok.encode(item["story"]), tok.eot]
        if len(ids) <= width:
            rows.append(np.array(ids + [tok.eot] * (width - len(ids)), dtype=np.uint16))
            spans.append((len(prompt), len(ids)))
            if len(rows) == limit:
                break
    np.save(out / f"sft_{split}.npy", np.stack(rows))
    np.save(out / f"sft_{split}_spans.npy", np.array(spans, dtype=np.uint16))
    print(f"sft {split}: kept {len(rows)} of {seen}", flush=True)


def compression(max_vocab: int = 8192) -> None:
    """Bytes per token against vocabulary size. Greedy BPE has a prefix property, so train once."""
    merges = Tokenizer.train((r["story"] for r in read("train", 100_000)), max_vocab).merges
    stories = [r["story"] for r in read("val", 5000)]
    n_bytes = sum(len(story.encode()) for story in stories)
    rows = []
    for vocab in (259, 512, 1024, 2048, 4096, 8192):
        tok = Tokenizer(merges[: vocab - 259])
        assert all(tok.decode(tok.encode(story)) == story for story in stories[:500])
        n_tokens = sum(len(tok.encode(story)) for story in stories)
        rows.append(dict(vocab=vocab, bytes_per_token=n_bytes / n_tokens))
        print(rows[-1], flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/tokenizer.json").write_text(json.dumps(rows, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=20)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--compression", action="store_true", help="only measure compression")
    parser.add_argument("--sft-limit", type=int, default=200_000)
    args = parser.parse_args()
    if not all((DATA / f"{s}.jsonl").exists() for s in SPLITS):
        download(args.shards)
    if args.compression:
        return compression()
    out = DATA / f"v{args.vocab}"
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.train((r["story"] for r in read("train", 100_000)), args.vocab)
    tok.save(out / "tokenizer.json")
    print(f"tokenizer: {tok.vocab_size} tokens", flush=True)
    for split in ("train", "val"):
        encode_stream(tok, split, out)
        encode_sft(tok, split, out, args.sft_limit)
        print(f"{split}: {(out / f'{split}.bin').stat().st_size // 2} tokens", flush=True)


if __name__ == "__main__":
    main()
