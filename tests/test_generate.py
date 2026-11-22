import pytest
import torch

from tinylm.generate import complete, generate, next_token
from tinylm.model import ModelConfig, TinyLM
from tinylm.tokenizer import Tokenizer


def make(pos: str = "rope", vocab: int = 97) -> TinyLM:
    torch.manual_seed(1)
    cfg = ModelConfig(vocab_size=vocab, d_model=48, n_layers=2, n_heads=4, d_ff=96, max_len=48, pos=pos)
    return TinyLM(cfg)


@pytest.mark.parametrize("pos", ["rope", "learned"])
def test_cache_gives_the_same_tokens(pos):
    # Why: a wrong position offset or mask in the cache changes the output without crashing.
    model, idx = make(pos), torch.randint(0, 97, (3, 7))
    fast = generate(model, idx, temperature=0, use_cache=True)
    slow = generate(model, idx, temperature=0, use_cache=False)
    assert fast == slow and len(fast[0]) == 48 + 1 - 7


@pytest.mark.parametrize("pos", ["rope", "learned"])
def test_cache_gives_the_same_logits(pos):
    # Why: equal greedy tokens can hide a small error in the logits.
    model, idx = make(pos), torch.randint(0, 97, (2, 20))
    cache = [None, None]
    pieces = [model(idx[:, :8], cache)] + [model(idx[:, i : i + 1], cache) for i in range(8, 20)]
    assert torch.allclose(torch.cat(pieces, 1), model(idx), atol=1e-5)
    keys = cache[0][0]
    assert keys.shape == (2, 4, 20, 12)  # the memory formula: B x heads x T x head_dim


def test_top_k_and_top_p():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]]).repeat(4000, 1)
    rng = torch.Generator().manual_seed(0)
    assert set(next_token(probs.log(), 1, 2, 1.0, rng).flatten().tolist()) == {0, 1}
    assert set(next_token(probs.log(), 1, 0, 0.75, rng).flatten().tolist()) == {0, 1}
    assert set(next_token(probs.log(), 0, 0, 1.0, rng).flatten().tolist()) == {0}


def test_temperature_changes_frequencies():
    logits, rng = torch.tensor([[2.0, 1.0, 0.0]]).repeat(20000, 1), torch.Generator().manual_seed(0)
    share = (next_token(logits, 0.5, 0, 1.0, rng) == 0).float().mean().item()
    assert abs(share - 0.867) < 0.02


def test_stops_at_eot():
    model, idx = make(), torch.randint(0, 97, (1, 5))
    free = generate(model, idx, temperature=0)[0]
    stopped = generate(model, idx, temperature=0, eot=free[3])[0]
    assert stopped == free[: free.index(free[3]) + 1]


def test_complete_keeps_the_input_order():
    tok = Tokenizer.train(["aaa bbb"], 259)
    model = make(vocab=tok.vocab_size)
    prompts = [[5, 6, 7], [8, 9], [1, 2, 3], [4]]
    together = complete(model, tok, prompts, temperature=0)
    assert together == [complete(model, tok, [p], temperature=0)[0] for p in prompts]
