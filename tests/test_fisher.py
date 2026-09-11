"""Fisher's identity: is the EM surrogate a valid ascent direction on the marginal?

The beat-only path never sees a class label. It trains on line 52's surrogate

    cls(theta; 1) = - sum_h pi_h * s_h(theta),     pi frozen at theta_old

instead of on the quantity we actually care about, the marginal likelihood of the
observed beats with (omega, L) summed out,

    nll(theta) = - log sum_h exp(s_h(theta)).

EM is only justified because these two have the SAME GRADIENT at theta = theta_old:

    d/dtheta [-log sum_h exp s_h] = - sum_h softmax(s)_h ds_h/dtheta
                                  = - sum_h pi_h ds_h/dtheta   when pi = softmax(s_old)

If that holds numerically, every M-step descends the true marginal and the beat-only
path is learning the right thing as efficiently as the model permits. If it fails, the
surrogate is optimising something else and no amount of tuning will fix it.

These are exact identities, not approximations: they must hold to float precision.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alignbeat.classes import BEAT, DOWNBEAT
from alignbeat.criterion import SubsetCriterion

DATA_PRIOR = {"downbeat": 0.28532, "beat": 0.71468}     # fold 0, as measured
METER_PRIOR = {2: 0.09928, 3: 0.05, 4: 0.7943, 5: 0.01, 6: 0.03, 8: 0.01642}


def criterion():
    return SubsetCriterion(data_prior=DATA_PRIOR, meter_prior=METER_PRIOR).double()


def logits(M, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(M, 3, generator=g, dtype=torch.double, requires_grad=True)


def marginal_nll(crit, log_p):
    """-log sum_h exp(s_h), plus the event-mass term the surrogate now carries.

    The event term does not depend on (omega, L), so it is additive on both sides and
    the identity is unchanged by it. Including it here is what makes these tests confirm
    that the added term left the EM structure alone rather than perturbing it."""
    scores = crit._log_scores(crit._class_log_posterior(log_p))
    return (-torch.logsumexp(torch.cat([scores[L] for L in scores]), dim=0)
            - torch.logsumexp(log_p[:, [DOWNBEAT, BEAT]], dim=-1).sum())


def surrogate(crit, log_p, pi):
    return crit._beat_only_term(log_p, pi)


def frozen_pi(crit, log_p):
    """pi_{omega,L} exactly as the E-step forms it (Algorithm 1 lines 39-41)."""
    with torch.no_grad():
        scores = crit._log_scores(crit._class_log_posterior(log_p))
        return torch.softmax(torch.cat([scores[L] for L in scores]), dim=0)


@pytest.mark.parametrize("M,seed", [(16, 0), (24, 1), (32, 2), (12, 3), (64, 4)])
def test_surrogate_gradient_equals_marginal_gradient(M, seed):
    """The identity itself, at theta = theta_old."""
    crit = criterion()

    z = logits(M, seed)
    log_p = torch.log_softmax(z, dim=-1)
    pi = frozen_pi(crit, log_p)
    g_surr, = torch.autograd.grad(surrogate(crit, log_p, pi), z)

    z2 = z.detach().clone().requires_grad_(True)
    g_marg, = torch.autograd.grad(marginal_nll(crit, torch.log_softmax(z2, dim=-1)), z2)

    assert torch.allclose(g_surr, g_marg, atol=1e-10), \
        f"max |diff| = {(g_surr - g_marg).abs().max():.3e}"


def test_pi_is_a_distribution():
    """pi must be a normalised posterior over hypotheses, or the dot product in line 52
    is not an expectation and the identity above cannot hold."""
    crit = criterion()
    log_p = torch.log_softmax(logits(24, 7), dim=-1)
    pi = frozen_pi(crit, log_p)
    assert pi.min() >= 0.0
    assert abs(float(pi.sum()) - 1.0) < 1e-12


def test_surrogate_upper_bounds_the_marginal():
    """Jensen: -sum pi s_h >= -log sum exp(s_h), with equality only at theta_old. This is
    what makes the M-step's descent on the surrogate imply descent on the marginal."""
    crit = criterion()
    log_p = torch.log_softmax(logits(24, 11), dim=-1)
    pi = frozen_pi(crit, log_p)

    # At theta_old the two agree up to the constant entropy of pi.
    gap0 = float(surrogate(crit, log_p, pi)) - float(marginal_nll(crit, log_p))
    entropy = float(-(pi * torch.log(pi.clamp_min(1e-300))).sum())
    assert abs(gap0 - entropy) < 1e-9, f"gap {gap0} != H[pi] {entropy}"

    # Away from theta_old the bound must not be violated.
    for seed in range(5):
        pert = torch.log_softmax(logits(24, 11) + 0.3 * logits(24, 100 + seed), dim=-1)
        assert float(surrogate(crit, pert, pi)) + 1e-9 >= float(marginal_nll(crit, pert))


def test_em_step_decreases_the_marginal():
    """The end-to-end property: E-step, then a gradient step on the surrogate, must not
    increase the true marginal NLL. Run to convergence it must decrease it a lot."""
    crit = criterion()
    z = logits(32, 5)

    before = float(marginal_nll(crit, torch.log_softmax(z, dim=-1)))
    opt = torch.optim.SGD([z], lr=0.05)
    for _ in range(50):
        log_p = torch.log_softmax(z, dim=-1)
        pi = frozen_pi(crit, log_p)               # E-step at the current theta
        opt.zero_grad()
        surrogate(crit, log_p, pi).backward()     # M-step on the frozen pi
        opt.step()
    after = float(marginal_nll(crit, torch.log_softmax(z, dim=-1)))

    assert after < before, f"marginal NLL rose: {before:.6f} -> {after:.6f}"


def test_single_em_step_is_monotone():
    """Monotonicity at every individual step, not just end to end."""
    crit = criterion()
    z = logits(24, 9)
    opt = torch.optim.SGD([z], lr=0.02)
    prev = float(marginal_nll(crit, torch.log_softmax(z, dim=-1)))
    for step in range(30):
        log_p = torch.log_softmax(z, dim=-1)
        pi = frozen_pi(crit, log_p)
        opt.zero_grad()
        surrogate(crit, log_p, pi).backward()
        opt.step()
        now = float(marginal_nll(crit, torch.log_softmax(z, dim=-1)))
        assert now <= prev + 1e-9, f"step {step}: {prev:.8f} -> {now:.8f}"
        prev = now


# ---------------------------------------------------------------------------
# End to end, through forward(): the helpers being right does not prove the
# plumbing is. These run the real call path -- E-step, sigma, M-step, aggregate --
# and check the identity survives it.
# ---------------------------------------------------------------------------

from alignbeat.classes import CLASS_UNKNOWN


def beat_only_batch(M=20, N=188, seed=0):
    g = torch.Generator().manual_seed(seed)
    class_logits = torch.randn(1, N, 3, generator=g, dtype=torch.double,
                               requires_grad=True)
    t_hat = torch.linspace(0.0, 30.0, N, dtype=torch.double)[None, :]
    # Events on a regular 4/4 grid inside the window, all labels hidden.
    times = torch.linspace(2.0, 26.0, M, dtype=torch.double)
    targets = [{"classes": torch.full((M,), CLASS_UNKNOWN), "times": times}]
    return class_logits, t_hat, targets


def test_forward_beat_only_gradient_matches_the_marginal():
    """The gradient forward() actually backpropagates on a beat-only fragment must be
    the marginal's gradient, at the sigma the E-step chose. This is the identity as the
    training loop experiences it -- any stale pi, wrong indexing or misplaced detach
    between _e_step and _beat_only_term shows up here and nowhere else."""
    crit = criterion()
    crit.eval()
    class_logits, t_hat, targets = beat_only_batch()

    losses, _ = crit(class_logits, t_hat, targets)
    g_forward, = torch.autograd.grad(losses["class"], class_logits, retain_graph=True)

    # Recompute the marginal directly at the E-step's own sigma.
    with torch.no_grad():
        log_p = torch.log_softmax(class_logits, dim=-1)[0]
        match = crit._e_step(log_p, t_hat[0], targets[0]["classes"], targets[0]["times"])
    z2 = class_logits.detach().clone().requires_grad_(True)
    matched = torch.log_softmax(z2, dim=-1)[0][torch.from_numpy(match.sigma)]
    g_marg, = torch.autograd.grad(marginal_nll(crit, matched), z2)

    # forward() may scale the class term when aggregating; compare direction exactly
    # and scale separately so a normalisation constant is not mistaken for a bug.
    s = float((g_forward * g_marg).sum() / (g_marg * g_marg).sum())
    assert torch.allclose(g_forward, s * g_marg, atol=1e-10), \
        f"direction differs; max |diff| = {(g_forward - s * g_marg).abs().max():.3e}"
    assert abs(s - 1.0) < 1e-9, f"class term is scaled by {s:.6f}, not 1"


def test_forward_gradient_is_nonzero_and_finite():
    """A surrogate that is provably correct but delivers no gradient teaches nothing."""
    crit = criterion()
    class_logits, t_hat, targets = beat_only_batch(seed=3)
    losses, _ = crit(class_logits, t_hat, targets)
    g, = torch.autograd.grad(sum(v for v in losses.values() if v.requires_grad),
                             class_logits)
    assert torch.isfinite(g).all()
    assert float(g.abs().max()) > 1e-6


def test_pi_carries_no_gradient():
    """pi is the frozen posterior at theta_old (Algorithm 2 line 46). If gradient leaked
    into it the objective would be the marginal's entropy-free variant and Fisher would
    silently not apply."""
    crit = criterion()
    class_logits, t_hat, targets = beat_only_batch(seed=4)
    log_p = torch.log_softmax(class_logits, dim=-1)[0]
    match = crit._e_step(log_p, t_hat[0], targets[0]["classes"], targets[0]["times"])
    assert match.pi is not None
    assert not match.pi.requires_grad
