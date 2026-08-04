"""A0d: where is the closed loop's ACTUAL attractor?

A0c found that upright is not a fixed point: the frozen policy commands u = +0.4999
at upright, where holding requires exactly 0, and the state drifts 7.5e-02 per step.

This matters more than it might look. V is built so V(x*) = 0 and V > 0 elsewhere, so
if f_cl(x*) != x* then V(f_cl(x*)) > 0 and

    cond(x*) = V(x*) - V(f_cl(x*)) = -V(f_cl(x*)) < 0

is a violation AT THE CENTRE, guaranteed by construction and independent of training.
That is the source of the persistent ~1e-3 violations A0 found in the narrow regions,
and no amount of training or verifier effort can remove it. The certificate was
centred on the wrong point.

cnl-work already had this right: LyapunovPDOnState centres on a z* that is explicitly
NOT the origin, for exactly this reason. This script finds the pendulum's analogue.

Three possibilities, and they lead to different projects:
  fixed point  -> re-centre V there and continue as planned
  limit cycle  -> no V has strict decrease anywhere on the cycle; the honest object is
                  a decrease condition relative to the CYCLE, not to a point
  neither      -> the policy does not settle, and the audit premise fails
"""

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.dynamics import ClosedLoop, wrap_angle
from src.policy import extract_sac_actor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def newton_fixed_point(loop, s0, iters=100, tol=1e-13):
    """Solve f_cl(s) - s = 0 by Newton. Returns (s, residual, converged)."""
    s = torch.tensor([list(s0)], dtype=torch.float64)
    loop = loop.double()
    for _ in range(iters):
        s = s.detach().requires_grad_(True)
        r = loop(s) - s
        J = torch.zeros(2, 2, dtype=torch.float64)
        for i in range(2):
            g, = torch.autograd.grad(r[0, i], s, retain_graph=True)
            J[i] = g[0]
        res = float(torch.abs(r).max())
        if res < tol:
            return s.detach().numpy().ravel(), res, True
        try:
            step = torch.linalg.solve(J, r.detach().reshape(2, 1))
        except Exception:
            return s.detach().numpy().ravel(), res, False
        s = (s.detach() - step.reshape(1, 2))
    r = loop(s.detach()) - s.detach()
    return s.detach().numpy().ravel(), float(torch.abs(r).max()), False


def main(seed=0, burn=4000, keep=400):
    from stable_baselines3 import SAC

    model = SAC.load(os.path.join(ROOT, "models", "sac_pendulum"), device="cpu")
    net = extract_sac_actor(model)
    loop = ClosedLoop(net).eval()

    # ---- long run from several starts, to see what the policy actually settles onto
    rng = np.random.default_rng(seed)
    starts = np.stack([rng.uniform(-0.3, 0.3, 8), rng.uniform(-0.3, 0.3, 8)], axis=1)
    s = torch.tensor(starts, dtype=torch.float64)
    loop_d = loop.double()
    with torch.no_grad():
        for _ in range(burn):
            s = loop_d(s)
        tail = []
        for _ in range(keep):
            s = loop_d(s)
            tail.append(s.numpy().copy())
    tail = np.stack(tail)                       # (keep, 8, 2)
    tail[..., 0] = np.remainder(tail[..., 0] + np.pi, 2 * np.pi) - np.pi

    spread = tail.max(axis=0) - tail.min(axis=0)      # per-start, per-coord range
    worst = spread.max()
    print(f"after {burn} steps, per-start spread over the last {keep} steps:")
    print(f"  max theta range   {spread[:,0].max():.3e}")
    print(f"  max thdot range   {spread[:,1].max():.3e}")

    attractor_mean = tail.reshape(-1, 2).mean(axis=0)
    print(f"  mean tail state   theta={attractor_mean[0]:+.6f}  "
          f"thdot={attractor_mean[1]:+.6f}")
    print()

    out = dict(spread_theta=float(spread[:, 0].max()),
               spread_thdot=float(spread[:, 1].max()),
               tail_mean=attractor_mean.tolist(), burn=burn, keep=keep)

    if worst < 1e-8:
        print("The tail is a POINT: the closed loop has an attracting fixed point.")
        out["attractor"] = "fixed_point"
    else:
        print(f"The tail is NOT a point (spread {worst:.3e}): the closed loop settles")
        print("onto a LIMIT CYCLE or a small invariant set, not an equilibrium.")
        out["attractor"] = "limit_cycle_or_set"

    # ---- Newton, from upright and from the observed tail mean
    for name, s0 in [("upright", (0.0, 0.0)),
                     ("tail mean", tuple(attractor_mean))]:
        sf, res, ok = newton_fixed_point(loop, s0)
        sf_w = float(wrap_angle(torch.tensor(sf[0])))
        status = "converged" if ok else "did NOT converge"
        print(f"\nNewton from {name}: {status}, residual {res:.3e}")
        print(f"  s* = (theta={sf_w:+.8f}, thdot={sf[1]:+.8f})")
        if ok:
            with torch.no_grad():
                # net was converted in place by loop.double(); match it or torch raises
                dt = next(net.parameters()).dtype
                o = torch.tensor([[np.cos(sf_w), np.sin(sf_w), sf[1]]], dtype=dt)
                u = float(net(o))
            need = 10.0 * np.sin(sf_w)       # m*g*l*sin(th), torque to hold
            print(f"  policy torque there u = {u:+.6f}; "
                  f"torque needed to hold = {need:+.6f}; residual {u-need:+.3e}")
        out[f"newton_from_{name.replace(' ','_')}"] = dict(
            converged=bool(ok), residual=float(res),
            theta=float(sf_w), thetadot=float(sf[1]))

    path = os.path.join(ROOT, "results", f"a0d_equilibrium_seed{seed}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
