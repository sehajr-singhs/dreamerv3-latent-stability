"""Locate the closed loop's fixed point. Everything downstream is centred on it.

A trained policy does not generally hold the nominal equilibrium. The seed-0 SAC policy
commands u = +0.4999 at upright, where holding requires exactly 0, and instead settles
at theta = 0.142586 rad (8.17 degrees). Building V about upright therefore guarantees a
violation at the centre, since V(s*) = 0 while V(f_cl(s*)) > 0. Finding s* is a
precondition for the audit meaning anything, not a refinement of it.

`verify_fixed_point` is the gate: it is not enough for Newton to converge, the residual
must actually be at machine precision, and the point must be ATTRACTING if we intend to
certify a neighbourhood of it. This closed loop has at least two fixed points (0.142586
attracting, 0.411218 at the torque-saturation boundary), so "Newton converged" alone
does not identify the right one.
"""

import copy

import numpy as np
import torch


def _f64(loop):
    """A float64 COPY of the closed loop.

    torch's `.double()` mutates in place, so calling it on the caller's module would
    silently leave the shared policy in float64 and break every float32 consumer
    downstream. Copying is the only safe option here, and it is cheap: these networks
    are tiny.
    """
    return copy.deepcopy(loop).double()


def newton_fixed_point(loop, s0, iters=100, tol=1e-13):
    """Solve f_cl(s) - s = 0 by Newton. Returns (s, residual, converged)."""
    loop = _f64(loop)
    s = torch.as_tensor([list(s0)], dtype=torch.float64)
    for _ in range(iters):
        s = s.detach().requires_grad_(True)
        r = loop(s) - s
        J = torch.zeros(2, 2, dtype=torch.float64)
        for i in range(2):
            g, = torch.autograd.grad(r[0, i], s, retain_graph=True)
            J[i] = g[0]
        res = float(torch.abs(r.detach()).max())
        if res < tol:
            return s.detach().numpy().ravel(), res, True
        try:
            step = torch.linalg.solve(J, r.detach().reshape(2, 1))
        except Exception:
            return s.detach().numpy().ravel(), res, False
        s = s.detach() - step.reshape(1, 2)
    with torch.no_grad():
        res = float(torch.abs(loop(s.detach()) - s.detach()).max())
    return s.detach().numpy().ravel(), res, False


def settle(loop, starts, steps=4000):
    """Iterate the closed loop to expose the attractor."""
    loop = _f64(loop)
    s = torch.as_tensor(starts, dtype=torch.float64)
    with torch.no_grad():
        for _ in range(steps):
            s = loop(s)
    return s.numpy()


def find_attracting_fixed_point(loop, seed=0, n_starts=8, spread=0.3, steps=4000):
    """Settle first, then polish with Newton, so we land on the ATTRACTING point.

    Newton from upright converges to the saturation-boundary fixed point instead, which
    is not the one a neighbourhood certificate should be built around. Settling first
    picks the right basin; Newton then gives machine precision.
    """
    rng = np.random.default_rng(seed)
    starts = np.stack([rng.uniform(-spread, spread, n_starts),
                       rng.uniform(-spread, spread, n_starts)], axis=1)
    tail = settle(loop, starts, steps=steps)
    spread_seen = float(np.abs(tail - tail[0:1]).max())
    s_hat = tail.mean(axis=0)
    s_star, res, ok = newton_fixed_point(loop, tuple(s_hat))
    return dict(s_star=s_star, residual=res, converged=bool(ok),
                start_spread=spread_seen, settled_estimate=s_hat.tolist())


def verify_fixed_point(loop, s_star, tol=1e-10):
    """Assert f_cl(s*) == s* to `tol`. Raise otherwise: a wrong centre is fatal."""
    loop = _f64(loop)
    s = torch.as_tensor([list(s_star)], dtype=torch.float64)
    with torch.no_grad():
        drift = float(torch.abs(loop(s) - s).max())
    if drift > tol:
        raise AssertionError(
            f"s* = {np.asarray(s_star).tolist()} is not a fixed point: drift {drift:.3e} "
            f"> {tol:.1e}. Centring V here would guarantee cond(s*) < 0 regardless of "
            "training, so the audit would be measuring the centring error, not the policy.")
    return drift


def closed_loop_jacobian(loop, s_star):
    """d f_cl / ds at s*, by autograd, in float64."""
    loop = _f64(loop)
    s = torch.tensor([list(s_star)], dtype=torch.float64, requires_grad=True)
    out = loop(s)
    A = torch.zeros(2, 2, dtype=torch.float64)
    for i in range(2):
        g, = torch.autograd.grad(out[0, i], s, retain_graph=True)
        A[i] = g[0]
    return A.detach().numpy()
