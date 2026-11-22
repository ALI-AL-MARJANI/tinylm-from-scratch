import random
import re

import pytest
import tiktoken

from tinylm.tokenizer import PATTERN, Tokenizer

CORPUS = [
    "Once upon a time, there was a little girl named Lily. She loved to play.",
    "Tom said, \"Let's play with the red ball!\" They played and played all day.",
    "The lowest branch was lower than the low wall. Lily was happy at the end.",
]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.train(CORPUS, 400)


def test_round_trip_on_random_unicode(tok):
    # Why: any lost byte silently corrupts the training data.
    rng = random.Random(0)
    alphabet = "ab Z09_.,!\"'\n\t\r\xa0²½é中🙂​́\x00"
    for _ in range(2000):
        text = "".join(rng.choices(alphabet, k=rng.randint(0, 40)))
        assert tok.decode(tok.encode(text)) == text
    for _ in range(2000):
        text = "".join(chr(rng.randrange(0x20000)) for _ in range(10))
        text = re.sub("[\ud800-\udfff]", "", text)  # lone surrogates are not valid UTF-8
        assert tok.decode(tok.encode(text)) == text


def test_toy_merges():
    # Why: checks the greedy rule on an example that can be done by hand.
    tok = Tokenizer.train(["low"] * 5 + ["lower"] * 2 + ["lot"] * 3, 256 + 2 + 3)
    assert tok.merges == [(ord("l"), ord("o")), (256, ord("w"))]
    assert tok.encode("lowest") == [257, ord("e"), ord("s"), ord("t")]


def test_merges_compress(tok):
    text = CORPUS[0]
    assert len(tok.encode(text)) < len(text.encode()) / 2


def test_special_tokens_cannot_be_injected(tok):
    # Why: user text must not be able to end the document or fake a role.
    ids = tok.encode("<|endoftext|><|user|><|assistant|>")
    assert not {tok.eot, tok.user, tok.assistant} & set(ids)


def test_save_and_load(tok, tmp_path):
    tok.save(tmp_path / "tok.json")
    assert Tokenizer.load(tmp_path / "tok.json").encode(CORPUS[1]) == tok.encode(CORPUS[1])


def test_matches_tiktoken_with_my_merges(tok):
    # Why: an independent implementation must agree on the same ranks and pattern.
    reference = tiktoken.Encoding(
        "tinylm",
        pat_str=PATTERN,
        mergeable_ranks={token: i for i, token in enumerate(tok.vocab[: tok.eot])},
        special_tokens={},
    )
    for text in CORPUS + ["a lower wall, played 123 times!", "Lily's café 🙂\n\nThe end."]:
        assert tok.encode(text) == reference.encode(text)
