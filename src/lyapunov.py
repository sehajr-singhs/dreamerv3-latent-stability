"""Positive-definite-by-construction Lyapunov candidate, ported from cnl-work E10.

    V(x) = || g(o) - g(o*) ||^2 + c || o - o* ||^2,     o = (cos th, sin th, thdot)

Two adaptations from the swing-model original, both load-bearing.

1. V is a function of the OBSERVATION o, not of raw theta. Pendulum's state space is a
   cylinder: theta and theta + 2pi are the same physical state. A V built on raw theta
   would give V(0) = 0 and V(2pi) != 0 for one and the same pendulum, which is not a
   function on the state space at all, let alone a certificate. Building V on
   (cos th, sin th, thdot) makes it periodic automatically and by construction.

2. The reference is o* = (1, 0, 0), the upright equilibrium, so no reduced-coordinate
   projection R is needed. The swing model needed R to quotient out a rotational
   symmetry; the pendulum has no such symmetry, and its equilibrium IS isolated on the
   cylinder.

Positive definiteness survives both changes, and is worth checking explicitly rather
than inheriting on faith. With o* = (cos th*, sin th*, thdot*) the quadratic term is

    || o - o* ||^2 = 2 - 2 cos(th - th*) + (thdot - thdot*)^2

which is >= 0, and is 0 exactly when th = th* (mod 2pi) and thdot = thdot*. So
V >= c || o - o* ||^2 > 0 away from s* and V(s*) = 0 exactly. Condition (4a) is
therefore structural here too, not something the loss must learn.

Note s* is the CLOSED LOOP's fixed point, not upright. See LyapunovPD for why that
distinction is fatal rather than cosmetic.

g(o*) is recomputed inside forward, never cached, so V(x*) stays exactly 0 as the
weights move; a stale cached g(o*) would silently break the guarantee. That bug is
specifically what E10 documents.
"""

import torch
import torch.nn as nn

from .dynamics import PendulumObs


class BoundableELU(nn.Module):
    """ELU written so auto_LiRPA bounds it as exp/relu rather than a fused op.

    Mirrors cnl-work's BoundableELU: elu(a) = relu(a) - relu(-(exp(-relu(-a)) - 1)).
    Kept because the same expression must both train and verify; a torch.nn.ELU that
    bounds differently than it trains would make the certificate describe another net.
    """

    def forward(self, a):
        return torch.relu(a) - (1.0 - torch.exp(-torch.relu(-a)))


class LyapunovNet(nn.Module):
    def __init__(self, in_dim=3, hidden=32, out_dim=16):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, hidden)
        self.act = BoundableELU()
        self.dense2 = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, o):
        return self.dense2(self.act(self.dense1(o)))


class LyapunovPD(nn.Module):
    """V on the pendulum cylinder, positive definite about s_star by construction.

    s_star is REQUIRED and is the closed loop's actual fixed point, which for a trained
    SAC policy is NOT upright. Measured for the seed-0 policy: the attracting fixed
    point is at (theta, thetadot) = (0.142586, 0), an 8.17 degree steady-state offset,
    because the policy commands u = +0.4999 at upright where holding needs exactly 0.

    Centring on the wrong point is not a small error, it is fatal by construction. If
    f_cl(s_star) != s_star then V(s_star) = 0 while V(f_cl(s_star)) > 0, so

        cond(s_star) = -V(f_cl(s_star)) < 0

    is a guaranteed violation at the centre that no training and no verifier can remove.
    That is precisely what an earlier version of this class produced: persistent ~1e-3
    violations in every narrow region. cnl-work's LyapunovPDOnState had this right, and
    documents the same requirement for its own off-origin equilibrium.
    """

    def __init__(self, s_star, c=0.5, hidden=32, out_dim=16):
        super().__init__()
        self.obs = PendulumObs()
        self.net = LyapunovNet(in_dim=3, hidden=hidden, out_dim=out_dim)
        self.c = float(c)
        s_star = torch.as_tensor(s_star, dtype=torch.float32).reshape(1, 2)
        self.register_buffer("s_star", s_star.clone(), persistent=False)
        # o* = (cos th*, sin th*, thdot*). Stored WITHOUT a batch dimension: auto_LiRPA
        # warns that constant operands should not carry one, and a (1, 3) constant breaks
        # the JacobianOP gradient expansion. Shape (3,) broadcasts identically against
        # (B, 3), so the function is unchanged.
        with torch.no_grad():
            o_star = self.obs(s_star).reshape(-1)
        self.register_buffer("o_star", o_star.clone(), persistent=False)

    def _g_star(self):
        return self.net(self.o_star)

    def forward(self, s):
        o = self.obs(s)
        d = o - self.o_star
        h = self.net(o) - self._g_star()
        return (h * h).sum(-1, keepdim=True) + self.c * (d * d).sum(-1, keepdim=True)


class DecreaseCondition(nn.Module):
    """cond(s) = V(s) - V(f_cl(s)), the amount V decreases over one closed-loop step.

    Sign convention matches cnl-work's `certify_box`, which certifies `cond >= -eps`.
    So cond >= 0 means "V decreased", and a NEGATIVE value is a violation.

    Note the structural equality at the equilibrium: V(x*) = 0 and f_cl(x*) = x*, so
    cond(x*) = 0 exactly. Any box containing x* has true infimum 0, and cannot certify
    at a strictly positive margin no matter how good the verifier is. That is a property
    of the problem, not a verifier failure, and it is why certification is done on an
    ANNULUS excluding a small ball about upright. E10 records the same three-way
    distinction: genuine violation, structural equality, and verifier incompleteness all
    read as "did not certify", and only the first is a defect in V.
    """

    def __init__(self, V, closed_loop):
        super().__init__()
        self.V = V
        self.f = closed_loop

    def forward(self, s):
        return self.V(s) - self.V(self.f(s))


class ExponentialDecreaseCondition(nn.Module):
    """cond(s) = beta * V(s) - V(f_cl(s)), certifying a geometric rate.

    beta in (0, 1) demands V shrink by at least a factor beta per step, which is
    strictly stronger than plain decrease. Unlike the plain condition this does NOT
    vanish at x* only because V(x*) = 0; it is still 0 there, so the annulus is
    required for the same reason.
    """

    def __init__(self, V, closed_loop, beta=0.99):
        super().__init__()
        self.V = V
        self.f = closed_loop
        self.beta = float(beta)

    def forward(self, s):
        return self.beta * self.V(s) - self.V(self.f(s))
