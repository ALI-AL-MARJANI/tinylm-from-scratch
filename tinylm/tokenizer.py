"""Byte-level BPE: learn the most frequent merges, then replay them to encode."""

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

# GPT-2 style pieces: a word carries its leading space; letters, digits and symbols never mix.
PATTERN = r" ?[^\W\d_]+| ?\d+| ?(?:[^\w\s]|_)+|\s+(?!\S)|\s+"
Pair = tuple[int, int]


def merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace each occurrence of `pair`, from left to right, by `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    """Ids are: 256 raw bytes, then merges in learned order, then 3 special tokens."""

    def __init__(self, merges: list[Pair]):
        self.merges = merges
        self.ranks = {pair: 256 + i for i, pair in enumerate(merges)}
        self.vocab = [bytes([i]) for i in range(256)]
        for a, b in merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.eot = len(self.vocab)
        self.vocab += [b"<|endoftext|>", b"<|user|>", b"<|assistant|>"]
        self.user, self.assistant = self.eot + 1, self.eot + 2
        self.cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "Tokenizer":
        """Greedily merge the most frequent pair, updating only the pieces that contain it."""
        piece_counts = Counter(piece for text in texts for piece in re.findall(PATTERN, text))
        words = [list(piece.encode()) for piece in piece_counts]
        freqs = list(piece_counts.values())
        pair_counts: Counter[Pair] = Counter()
        where: dict[Pair, set[int]] = defaultdict(set)  # pair -> pieces containing it
        for i, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freqs[i]
                where[pair].add(i)
        merges: list[Pair] = []
        while pair_counts and len(merges) < vocab_size - 256 - 3:
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))  # ties must be deterministic
            new_id = 256 + len(merges)
            merges.append(best)
            for i in list(where[best]):
                for pair in zip(words[i], words[i][1:]):
                    pair_counts[pair] -= freqs[i]
                    where[pair].discard(i)
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]
                words[i] = merge(words[i], best, new_id)
                for pair in zip(words[i], words[i][1:]):
                    pair_counts[pair] += freqs[i]
                    where[pair].add(i)
        return cls(merges)

    def encode_piece(self, piece: str) -> list[int]:
        """Apply the earliest learned merge until none applies."""
        if piece not in self.cache:
            ids = list(piece.encode())
            while len(ids) > 1:
                best = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
                if best not in self.ranks:
                    break
                ids = merge(ids, best, self.ranks[best])
            self.cache[piece] = ids
        return self.cache[piece]

    def encode(self, text: str) -> list[int]:
        """Encode plain text. Special tokens are never parsed from text (prompt injection)."""
        return [i for piece in re.findall(PATTERN, text) for i in self.encode_piece(piece)]

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def encode_prompt(self, words: list[str]) -> list[int]:
        """The instruction format shared by SFT, DPO and evaluation."""
        text = f"Write a story that uses the words: {', '.join(words)}"
        return [self.eot, self.user, *self.encode(text), self.assistant]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.merges))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls([tuple(pair) for pair in json.loads(Path(path).read_text())])
