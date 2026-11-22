"""Download TinyStories metadata and split it by instruction, so no test prompt is seen in training."""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=20)
    download(parser.parse_args().shards)
