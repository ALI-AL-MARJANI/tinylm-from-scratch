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
