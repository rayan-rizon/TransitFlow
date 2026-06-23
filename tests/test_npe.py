import torch

from transitflow.models.npe import NPEHead, _HAS_ZUKO


def test_realnvp_fallback_logprob_and_sample():
    head = NPEHead(param_dim=7, embed_dim=16, hidden=32, backend="realnvp")
    e = torch.randn(8, 16)
    theta = torch.randn(8, 7)
    lp = head.log_prob(theta, e)
    assert lp.shape == (8,) and torch.isfinite(lp).all()
    s = head.sample(e, 50)
    assert s.shape == (8, 50, 7) and torch.isfinite(s).all()
    nll = head.nll(theta, e)
    nll.backward()


def test_realnvp_density_normalizes_roughly():
    """A trained RealNVP should assign higher density to in-distribution points."""
    torch.manual_seed(0)
    head = NPEHead(param_dim=2, embed_dim=2, hidden=64, backend="realnvp")
    opt = torch.optim.Adam(head.parameters(), lr=5e-3)
    # target: theta ~ N(context, 0.1 I)
    for _ in range(300):
        ctx = torch.randn(256, 2)
        theta = ctx + 0.1 * torch.randn(256, 2)
        opt.zero_grad()
        loss = -head.log_prob(theta, ctx).mean()
        loss.backward()
        opt.step()
    ctx = torch.zeros(1, 2)
    near = head.log_prob(torch.zeros(1, 2), ctx)
    far = head.log_prob(5 * torch.ones(1, 2), ctx)
    assert near.item() > far.item()


def test_zuko_backend_if_available():
    if not _HAS_ZUKO:
        return
    head = NPEHead(param_dim=7, embed_dim=16, backend="auto")
    assert head.backend == "zuko"
    e = torch.randn(8, 16)
    theta = torch.randn(8, 7)
    lp = head.log_prob(theta, e)
    assert lp.shape == (8,) and torch.isfinite(lp).all()
    s = head.sample(e, 20)
    assert s.shape == (8, 20, 7)
