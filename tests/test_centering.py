"""Regression tests for the centring bug, which cost the most time of anything here.

The failure was silent: V trained, sampling ran, the verifier ran, and every number
looked like a modest modelling shortfall rather than a construction error. Nothing
crashed. These tests make it loud.

The invariant: V must be centred on a point that is ACTUALLY a fixed point of the
closed loop. If it is not, cond(centre) < 0 by construction, independent of training
and of the verifier, and the audit measures the centring error instead of the policy.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.dynamics import ClosedLoop
from src.equilibrium import (find_attracting_fixed_point, verify_fixed_point,
                             newton_fixed_point, closed_loop_jacobian)
from src.lyapunov import LyapunovPD, DecreaseCondition
from src.policy import extract_sac_actor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    from stable_baselines3 import SAC
    model = SAC.load(os.path.join(ROOT, "models", "sac_pendulum"), device="cpu")
    net = extract_sac_actor(model)
    return net, ClosedLoop(net).eval()


def test_upright_is_not_a_fixed_point():
    """Documents the fact that motivated all of this. If this ever fails, the policy
    changed and the recorded s* must be recomputed."""
    net, loop = _load()
    with torch.no_grad():
        s0 = torch.zeros(1, 2)
        drift = float(torch.abs(loop(s0) - s0).max())
    assert drift > 1e-3, (
        f"upright drift is only {drift:.3e}; the policy now nearly holds upright, so "
        "the recorded s* and every region built on it must be recomputed")
    print(f"[PASS] upright is not a fixed point (drift {drift:.3e})")


def test_found_fixed_point_is_actually_fixed():
    net, loop = _load()
    eq = find_attracting_fixed_point(loop, seed=0)
    drift = verify_fixed_point(loop, eq["s_star"], tol=1e-10)
    assert eq["converged"], "Newton did not converge"
    print(f"[PASS] s* = ({eq['s_star'][0]:.8f}, {eq['s_star'][1]:.8f}) "
          f"is fixed to {drift:.2e}")


def test_verify_fixed_point_rejects_a_wrong_centre():
    """The gate must actually fire. A gate that never rejects is not a gate."""
    net, loop = _load()
    try:
        verify_fixed_point(loop, np.array([0.0, 0.0]), tol=1e-10)
    except AssertionError:
        print("[PASS] verify_fixed_point rejects upright as a centre")
        return
    raise AssertionError("verify_fixed_point accepted upright, which is NOT a fixed point")


def test_cond_at_correct_centre_is_machine_zero():
    """Structural equality (category 2), not a genuine violation (category 1)."""
    net, loop = _load()
    s_star = find_attracting_fixed_point(loop, seed=0)["s_star"]
    V = LyapunovPD(s_star, c=0.5, hidden=32)
    cond = DecreaseCondition(V, loop).eval()
    with torch.no_grad():
        v = float(V(torch.tensor([s_star], dtype=torch.float32)))
        cv = float(cond(torch.tensor([s_star], dtype=torch.float32)))
    assert v == 0.0, f"V(s*) = {v}, must be exactly 0 by construction"
    assert abs(cv) < 1e-10, f"cond(s*) = {cv:.3e}, expected machine zero"
    print(f"[PASS] V(s*) = {v}, cond(s*) = {cv:.2e} (structural, not a violation)")


def test_wrong_centre_guarantees_a_violation():
    """The bug itself, pinned. Untrained V, so this is construction, not training."""
    net, loop = _load()
    V = LyapunovPD(np.array([0.0, 0.0]), c=0.5, hidden=32)
    cond = DecreaseCondition(V, loop).eval()
    with torch.no_grad():
        cv = float(cond(torch.zeros(1, 2)))
    assert cv < -1e-4, (
        f"expected a guaranteed violation at a wrong centre, got cond = {cv:.3e}")
    print(f"[PASS] wrong centre gives cond = {cv:.3e} < 0 by construction alone")


def test_closed_loop_is_schur_stable_at_s_star():
    """If this fails the audit's premise fails: no V has strict decrease anywhere."""
    net, loop = _load()
    s_star = find_attracting_fixed_point(loop, seed=0)["s_star"]
    A = closed_loop_jacobian(loop, s_star)
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    assert rho < 1.0, f"rho(A) = {rho:.6f} >= 1: not locally stable at s*"
    print(f"[PASS] Schur stable at s*, rho = {rho:.6f} "
          f"(achievable rel margin {1-rho**2:.3f})")


def test_equilibrium_finder_does_not_mutate_the_policy():
    """torch's .double() mutates in place; this caught a real downstream breakage."""
    net, loop = _load()
    before = next(net.parameters()).dtype
    find_attracting_fixed_point(loop, seed=0)
    closed_loop_jacobian(loop, np.array([0.1, 0.0]))
    newton_fixed_point(loop, (0.1, 0.0))
    after = next(net.parameters()).dtype
    assert before == after == torch.float32, (
        f"policy dtype changed {before} -> {after}: the equilibrium helpers mutated "
        "the caller's module")
    with torch.no_grad():                    # and it must still run in float32
        loop(torch.zeros(1, 2))
    print(f"[PASS] policy dtype preserved ({after}) and still callable")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
            except AssertionError as e:
                fails += 1
                print(f"[FAIL] {name}: {e}")
    print(f"\n{'all centring tests passed' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)
