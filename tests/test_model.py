import math

import pytest
import torch
import torch.nn.functional as F

from tinylm.model import ModelConfig, TinyLM, rotate

SMALL = dict(vocab_size=97, d_model=48, n_layers=2, n_heads=4, d_ff=96, max_len=32)


def make(pos: str = "rope") -> TinyLM:
    torch.manual_seed(0)
    return TinyLM(ModelConfig(**SMALL, pos=pos))


def test_shapes():
    assert make()(torch.randint(0, 97, (3, 10))).shape == (3, 10, 97)


def test_parameter_count_matches_the_formula():
    # Why: an extra bias or an untied head would show up here.
    c = ModelConfig()
    d, f = c.d_model, c.d_ff
    expected = c.vocab_size * d + c.n_layers * (4 * d * d + 3 * d * f + 2 * d) + d
    assert TinyLM(c).num_params() == expected == 5_868_800


def test_initial_loss_is_uniform():
    # Why: a model that knows nothing must score ln(vocab).
    idx = torch.randint(0, 97, (16, 32))
    targets = torch.randint(0, 97, (16, 32))
    loss = F.cross_entropy(make()(idx).reshape(-1, 97), targets.reshape(-1))
    assert abs(loss.item() - math.log(97)) < 0.3


@pytest.mark.parametrize("pos", ["rope", "learned"])
def test_causal_mask(pos):
    # Why: seeing the future gives a wonderful loss and a useless model.
    model = make(pos)
    a = torch.randint(0, 97, (2, 20))
    b = a.clone()
    b[:, 10:] = (b[:, 10:] + 1) % 97
    before, after = model(a), model(b)
    assert torch.allclose(before[:, :10], after[:, :10], atol=1e-6)
    assert not torch.allclose(before[:, 10:], after[:, 10:], atol=1e-3)


def test_rope_depends_only_on_distance():
    q, k = torch.randn(8), torch.randn(8)
    theta = 10000.0 ** (-torch.arange(0, 8, 2) / 8)

    def score(m: int, n: int) -> torch.Tensor:
        rq = rotate(q, (m * theta).cos(), (m * theta).sin())
        rk = rotate(k, (n * theta).cos(), (n * theta).sin())
        return rq @ rk

    assert torch.allclose(score(3, 1), score(23, 21), atol=1e-5)


@pytest.mark.parametrize("pos", ["rope", "learned"])
def test_overfits_one_batch(pos):
    # Why: proves that the graph, the targets and the gradients are wired correctly.
    model = make(pos)
    batch = torch.randint(0, 97, (4, 17))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0)
    for _ in range(250):
        loss = F.cross_entropy(model(batch[:, :-1]).reshape(-1, 97), batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple Silicon")
def test_mps_matches_cpu():
    # Why: MPS bugs are silent, so compare logits AND gradients with the CPU.
    idx = torch.randint(0, 97, (2, 16))
    cpu, mps = make(), make().to("mps")
    out_cpu, out_mps = cpu(idx), mps(idx.to("mps"))
    assert torch.allclose(out_cpu, out_mps.cpu(), atol=1e-4, rtol=1e-3)
    out_cpu.logsumexp(-1).mean().backward()
    out_mps.logsumexp(-1).mean().backward()
    for p, q in zip(cpu.parameters(), mps.parameters()):
        assert torch.allclose(p.grad, q.grad.cpu(), atol=1e-4, rtol=1e-2)


def test_sft_masks_prompt_and_padding():
    # Why: a shifted mask trains on the instruction or never teaches the model to stop.
    import numpy as np

    from tinylm.train import get_batch

    tokens = np.array([[9, 1, 2, 3, 4, 5, 9, 9]], dtype=np.uint16)  # prompt 0..2, answer 3..6
    x, y = get_batch((tokens, np.array([[3, 7]], dtype=np.uint16)), 1, 7, (0, 0), "cpu")
    assert x.tolist() == [[9, 1, 2, 3, 4, 5, 9]]
    assert y.tolist() == [[-100, -100, 3, 4, 5, 9, -100]]


def test_dpo_targets_and_loss():
    # Why: before any update the loss must be ln 2, and only the answer may count.
    from tinylm.post_train import dpo_loss, to_tensors

    x, y = to_tensors([dict(prompt=[9, 1, 2], chosen=[3, 4, 9])], "chosen", 7)
    assert x.tolist() == [[9, 1, 2, 3, 4, 0, 0]]
    assert y.tolist() == [[-100, -100, 3, 4, 9, -100, -100]]
    logp = torch.tensor([-100.0, -120.0])
    loss, margin = dpo_loss(logp, logp, logp, logp, 0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6 and margin.abs().sum() == 0
    better, _ = dpo_loss(logp + 5, logp - 5, logp, logp, 0.1)
    assert abs(better.item() - math.log(1 + math.exp(-1))) < 1e-5


def test_verifier():
    # Why: the reward is the specification, and a model will exploit any hole in it.
    from tinylm.eval import verify

    words = ["dive", "job", "sorry"]
    body = " ".join(f"word{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(60))
    good = f"He can dive. {body} It is his job. {body.replace('word', 'term')} He was sorry."
    assert verify(words, good, True)["success"]
    assert not verify(words, good, False)["proper_end"]
    assert not verify(words, good.replace("dive", "diver"), True)["all_words"]
    assert not verify(words, good[:-1], True)["proper_end"]
    assert not verify(words, f"dive, job, sorry. {good}", True)["not_a_list"]
    assert not verify(words, "He can dive. " + "fun fun fun. " * 40, True)["not_repetitive"]
