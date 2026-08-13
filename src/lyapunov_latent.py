"""Lyapunov candidate on the DreamerV3 latent space, mirroring cnl-work E10.

    V(z) = || h(z) - h(z*) ||^2 + c || z - z* ||^2,     z = (deter, stoch_flat)

Same positive-definite-by-construction form as src/lyapunov.py, on the latent
state instead of the pendulum observation. h is a small boundable MLP (ELU); the
quadratic term uses the identity on the latent (g = id). V(z*) = 0 exactly and
V > 0 away from z*, so condition (4a) is structural here too.

The certified map is the FULL one-step closed loop on the latent itself,

    T: z -> z' = (deter', stoch'),   deter' = core(z, pi(z)),
                                     stoch'  = mean of the prior categorical,

a SQUARE map z -> z' with the joint fixed point z* = (deter*, stoch*). V is
defined on the full latent and cond(z) = V(z) - V(T(z)) is the same function the
sampling audit and branch-and-bound consume.

WHY NOT V ON DETER ALONE -- a documented trap. The deter-component map z -> deter'
is not square, and V_d(deter*) = 0 is its global minimum, so on the 128-dim slice
{deter = deter*, stoch != stoch*} the condition reads cond = 0 - V_d(deter') < 0
BY CONSTRUCTION, no matter how stable the model's dynamics are. Training cannot
fix it (the frozen model fixes deter'), uniform sampling never hits the
measure-zero slice, but branch-and-bound finds those points as 'violations' and
the reachability gate rates them near-support (the on-policy trajectory passes
through deter ~ deter* on its way to z*). Such a result would be manufactured --
the A0 trap on the latent. The full map has no such slice: the only structural
zero is z* itself, exactly like the pendulum.

The centring requirement is identical to the pendulum and is NOT optional: z*
must be an ACTUAL fixed point of the full one-step closed loop T(z, pi(z)). If it
is not, cond(z*) = V(z*) - V(T(z*)) = -V(T(z*)) < 0 by construction, and the
audit measures the centring error instead of the dynamics. The D1 run must settle
and verify z* before V is fit (see d1_sampling_gap.find_latent_fixed_point).

The stoch component is the mean of the prior categorical -- the same deterministic
surrogate everywhere, stated in every artifact. Its softmax is what makes
branch-and-bound return unknown on boxes (the guardrail-predicted categorical
wall); the SAMPLING audit evaluates it exactly in float precision, so the
empirical baseline is unaffected. Category 3, never a finding.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from .dreamer import OneStepClosedLoop
from .lyapunov import BoundableELU, DecreaseCondition


class LatentLyapunovNet(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=16):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, hidden)
        self.act = BoundableELU()
        self.dense2 = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, z):
        return self.dense2(self.act(self.dense1(z)))


class LatentLyapunovPD(nn.Module):
    """V on the full latent, positive definite about z_star by construction.

    z_star is REQUIRED and must be the full one-step closed loop's joint fixed
    point. Same failure mode as the pendulum's: centred anywhere else,
    cond(z*) < 0 by construction, independent of training and of the verifier.
    """

    def __init__(self, z_star, c=0.5, hidden=64, out_dim=16):
        super().__init__()
        z_star = torch.as_tensor(z_star, dtype=torch.float32).reshape(1, -1)
        self.in_dim = z_star.shape[-1]
        self.net = LatentLyapunovNet(self.in_dim, hidden=hidden, out_dim=out_dim)
        self.c = float(c)
        self.register_buffer("z_star", z_star.clone(), persistent=False)

    def _h_star(self):
        return self.net(self.z_star)

    def forward(self, z):
        d = z - self.z_star
        h = self.net(z) - self._h_star()
        return (h * h).sum(-1, keepdim=True) + self.c * (d * d).sum(-1, keepdim=True)


def _in_hole(z, hole_lo, hole_hi):
    return torch.all((z > torch.as_tensor(hole_lo)) & (z < torch.as_tensor(hole_hi)),
                     dim=1)


def sample_latent_states(n, lo, hi, support, hole_lo, hole_hi,
                         frac_on_policy=0.5, rng=None):
    """Mixture of uniform-over-box and on-policy latent states, hole removed.

    Identical design to train_lyapunov.sample_states: uniform over the box so V is
    shaped everywhere it must certify, plus on-policy draws so capacity goes where
    the model actually lives. Support is (N, dim) numpy over the full latent.
    """
    rng = rng or np.random.default_rng(0)
    dim = len(lo)
    n_pol = int(n * frac_on_policy)
    n_uni = n - n_pol

    uni = rng.uniform(lo, hi, size=(n_uni, dim))
    idx = rng.integers(0, len(support), size=n_pol)
    pol = np.clip(support[idx], lo, hi)
    z = torch.tensor(np.concatenate([uni, pol], axis=0), dtype=torch.float32)
    return z[~_in_hole(z, hole_lo, hole_hi)]


def train_latent_V(closed_loop, support, lo, hi, hole_lo, hole_hi, z_star, *,
                   c=0.5, hidden=64, steps=4000, batch=4096, lr=1e-3,
                   rel_margin=0.01, seed=0, verbose=True):
    """Fit V so that V(T(z, pi(z))) <= (1 - rel_margin) * V(z) on the region.

    Same relative-margin hinge as train_lyapunov.train_V: an absolute hinge is
    unachievable near z* where cond ~ (1 - rho) V -> 0, and the budget would be
    spent chasing it. `support` is (N, dim) numpy over the FULL latent z.
    `closed_loop` is the full map z -> z' (OneStepClosedLoop). Returns V; the
    caller builds cond = DecreaseCondition(V, closed_loop), the exact certified
    function the sampling audit and branch-and-bound consume.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    V = LatentLyapunovPD(z_star, c=c, hidden=hidden)
    cond = DecreaseCondition(V, closed_loop)

    opt = torch.optim.Adam(V.net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    for i in range(steps):
        z = sample_latent_states(batch, lo, hi, support, hole_lo, hole_hi, rng=rng)
        loss = torch.relu(rel_margin * V(z) - cond(z)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (i + 1) % 1000 == 0:
            with torch.no_grad():
                viol = float((cond(z) < 0).float().mean())
            print(f"  step {i+1:5d}  hinge={float(loss):.3e}  batch_viol={viol:.4%}")

    V.eval()
    for p in V.parameters():
        p.requires_grad_(False)
    return V


@torch.no_grad()
def sampled_latent_violation_rate(cond, lo, hi, hole_lo, hole_hi, support,
                                  n=500_000, seed=123, chunk=50_000,
                                  frac_on_policy=0.5, return_conds=False):
    """Empirical violation rate over the latent region; the sampling baseline.
    With return_conds=True the dict also carries the raw condition values
    (as a float32 array) for plotting; callers that persist JSON must pop it.
    """
    rng = np.random.default_rng(seed)
    chunk = min(chunk, n)
    n_viol, n_seen = 0, 0
    worst = float("inf")
    worst_z = None
    conds = [] if return_conds else None
    for _ in range(0, n, chunk):
        z = sample_latent_states(chunk, lo, hi, support, hole_lo, hole_hi,
                                 frac_on_policy=frac_on_policy, rng=rng)
        v = cond(z).squeeze(-1)
        n_viol += int((v < 0).sum())
        n_seen += len(z)
        m = int(torch.argmin(v))
        if float(v[m]) < worst:
            worst = float(v[m])
            worst_z = z[m].cpu().numpy().tolist()
        if return_conds:
            conds.append(v.detach().cpu().numpy())
    out = dict(n_sampled=n_seen, n_violations=n_viol,
               violation_rate=n_viol / max(n_seen, 1),
               worst_cond=worst, worst_state=worst_z)
    if return_conds:
        out["conds"] = np.concatenate(conds)
    return out
