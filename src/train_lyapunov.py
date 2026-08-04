"""Fit the Lyapunov candidate V to a FROZEN policy.

This is the step that sets up the question the audit asks. We train V until random
sampling is satisfied with it, i.e. until the empirical violation rate on a large
sample is at or near zero. Only then is "a directed search finds a violation anyway"
a statement about the gap between sampling and proof, rather than a statement about
an undertrained certificate.

What is NOT happening here: the policy is never updated. v1 audits a policy as
trained. Only V moves.

Sampling design. Training draws from a mixture of
  (a) uniform over the certification box, so V is shaped everywhere it must be certified, and
  (b) the empirical on-policy support, so capacity goes where the policy actually lives.
Training only on (a) leaves V bad exactly where it matters; only on (b) leaves the
certification box unconstrained and makes violations trivial to find, which would
manufacture the very gap we are trying to measure honestly.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from .dynamics import ClosedLoop
from .lyapunov import LyapunovPD, DecreaseCondition

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _in_hole(s, hole_lo, hole_hi):
    return torch.all((s > torch.as_tensor(hole_lo)) & (s < torch.as_tensor(hole_hi)), dim=1)


def sample_states(n, lo, hi, support, hole_lo, hole_hi, frac_on_policy=0.5, rng=None):
    """Mixture of uniform-over-box and on-policy states, with the hole removed."""
    rng = rng or np.random.default_rng(0)
    n_pol = int(n * frac_on_policy)
    n_uni = n - n_pol

    uni = rng.uniform(lo, hi, size=(n_uni, len(lo)))
    idx = rng.integers(0, len(support.states), size=n_pol)
    pol = support.states[idx]
    # clip on-policy draws into the box so every training point is one we certify over
    pol = np.clip(pol, lo, hi)

    s = torch.tensor(np.concatenate([uni, pol], axis=0), dtype=torch.float32)
    return s[~_in_hole(s, hole_lo, hole_hi)]


def train_V(policy, support, lo, hi, hole_lo, hole_hi, s_star, *, c=0.5, hidden=32,
            steps=4000, batch=4096, lr=1e-3, rel_margin=0.01, seed=0, verbose=True):
    """Fit V so that V(f(s)) <= (1 - rel_margin) * V(s) on the region.

    The margin is RELATIVE to V, not absolute, and that distinction is load-bearing.
    An absolute hinge (`cond >= 1e-3`) is unachievable near the equilibrium, where
    cond ~ (1 - rho) V -> 0 by construction, so training spends its capacity chasing an
    impossible target there and distorts V across the whole region. Measured: an
    absolute margin plateaued at ~20% sampled violations regardless of budget, from
    500 up to 15000 steps.

    Relative margin is the same thing as a geometric decrease rate, which is the
    scale-free notion of stability and the one that stays meaningful at every distance
    from x*.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    V = LyapunovPD(s_star, c=c, hidden=hidden)
    loop = ClosedLoop(policy)
    cond = DecreaseCondition(V, loop)

    opt = torch.optim.Adam(V.net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    for i in range(steps):
        s = sample_states(batch, lo, hi, support, hole_lo, hole_hi, rng=rng)
        # hinge on a margin PROPORTIONAL to V(s): demand V(f) <= (1 - rel_margin) V,
        # which is achievable at every scale, unlike a fixed absolute decrease.
        loss = torch.relu(rel_margin * V(s) - cond(s)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (i + 1) % 1000 == 0:
            with torch.no_grad():
                viol = float((cond(s) < 0).float().mean())
            print(f"  step {i+1:5d}  hinge={float(loss):.3e}  batch_viol={viol:.4%}")

    V.eval()
    for p in V.parameters():
        p.requires_grad_(False)
    return V


@torch.no_grad()
def sampled_violation_rate(cond, lo, hi, hole_lo, hole_hi, support,
                           n=500_000, seed=123, chunk=50_000, frac_on_policy=0.5):
    """Empirical violation rate: the number a sampling-only audit would report.

    This is the baseline the gap is measured against, so it is deliberately generous:
    half a million states, drawn from the same mixture V was NOT specifically fit to,
    covering the whole certification region.
    """
    rng = np.random.default_rng(seed)
    chunk = min(chunk, n)          # else a small n still draws one full chunk
    n_viol, n_seen = 0, 0
    worst = float("inf")
    worst_s = None
    for _ in range(0, n, chunk):
        s = sample_states(chunk, lo, hi, support, hole_lo, hole_hi,
                          frac_on_policy=frac_on_policy, rng=rng)
        v = cond(s).squeeze(-1)
        n_viol += int((v < 0).sum())
        n_seen += len(s)
        m = int(torch.argmin(v))
        if float(v[m]) < worst:
            worst = float(v[m])
            worst_s = s[m].numpy().tolist()
    return dict(n_sampled=n_seen, n_violations=n_viol,
                violation_rate=n_viol / max(n_seen, 1),
                worst_cond=worst, worst_state=worst_s)
