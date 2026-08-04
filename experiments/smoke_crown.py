"""De-risk smoke test. Run before building anything on top of the verifier.

Four gates, each of which would silently invalidate every downstream number:

  1. Our dynamics constants match gymnasium's PendulumEnv.
  2. Our PendulumStep reproduces env.step() bit-for-bit on random states.
  3. The extracted actor reproduces SB3's own deterministic action.
  4. auto_LiRPA can trace and bound the closed loop (sin/cos + clamp + tanh),
     and the bounds are sound and non-vacuous.

Gate 4 is the real unknown: if CROWN cannot bound sin/cos and clamp tightly, the
whole approach needs rethinking, and it is much cheaper to learn that now.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from src import dynamics as dyn
from src.dynamics import ClosedLoop, PendulumStep, PendulumObs
from src.policy import extract_sac_actor, check_matches_sb3

OK, FAIL = "PASS", "FAIL"
results = []


def gate(name, fn):
    try:
        detail = fn()
        results.append((OK, name, detail))
        print(f"[{OK}] {name}: {detail}")
        return True
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"[{FAIL}] {name}: {e}")
        return False


# ---------------------------------------------------------------- gate 1
def g1():
    env = gym.make("Pendulum-v1").unwrapped
    pairs = {
        "g": (env.g, dyn.G), "m": (env.m, dyn.M), "l": (env.l, dyn.L),
        "dt": (env.dt, dyn.DT),
        "max_speed": (env.max_speed, dyn.MAX_SPEED),
        "max_torque": (env.max_torque, dyn.MAX_TORQUE),
    }
    bad = {k: v for k, v in pairs.items() if float(v[0]) != float(v[1])}
    if bad:
        raise AssertionError(f"constant drift vs gymnasium: {bad}")
    return f"all 6 constants match gymnasium {gym.__version__}"


# ---------------------------------------------------------------- gate 2
def g2():
    env = gym.make("Pendulum-v1").unwrapped
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    step = PendulumStep()
    worst = 0.0
    for _ in range(500):
        th = rng.uniform(-np.pi, np.pi)
        thdot = rng.uniform(-8.0, 8.0)
        # Pass a float64 action. With a float32 one, NEP 50 keeps gymnasium's
        # `3.0/(m*l**2) * u` term in float32 while the sin(th) term stays float64, so
        # the env mixes precisions internally and we would be measuring float32 eps
        # (~1e-8 here) rather than whether the dynamics agree. float64 on both sides
        # isolates the modelling question, which is the one this gate is asking.
        u = rng.uniform(-3.0, 3.0)               # outside [-2,2] to exercise the clip
        env.state = np.array([th, thdot], dtype=np.float64)
        env.step(np.array([u], dtype=np.float64))
        ref = np.asarray(env.state, dtype=np.float64)
        ours = step(torch.tensor([[th, thdot]], dtype=torch.float64),
                    torch.tensor([[u]], dtype=torch.float64)).numpy().ravel()
        worst = max(worst, float(np.abs(ours - ref).max()))
    if worst > 1e-9:
        raise AssertionError(f"dynamics disagree with env.step by {worst:.3e}")
    return f"500 random states, max |ours - env.step| = {worst:.2e}"


# ---------------------------------------------------------------- gate 3
def g3():
    model = SAC("MlpPolicy", "Pendulum-v1", seed=0, device="cpu")
    net = extract_sac_actor(model)
    rng = np.random.default_rng(1)
    th = rng.uniform(-np.pi, np.pi, size=256)
    thdot = rng.uniform(-8, 8, size=256)
    obs = np.stack([np.cos(th), np.sin(th), thdot], axis=1).astype(np.float32)
    err = check_matches_sb3(net, model, obs)
    return f"untrained SAC actor, 256 obs, max |ours - sb3.predict| = {err:.2e}"


# ---------------------------------------------------------------- gate 4
def g4():
    model = SAC("MlpPolicy", "Pendulum-v1", seed=0, device="cpu")
    net = extract_sac_actor(model)
    loop = ClosedLoop(net).eval()

    center = torch.tensor([[0.3, 0.5]])
    bm = BoundedModule(loop, center, verbose=False)

    out = []
    for r in (0.05, 0.2, 0.5):
        ptb = PerturbationLpNorm(norm=np.inf, eps=r)
        bt = BoundedTensor(center, ptb)
        lb, ub = bm.compute_bounds(x=(bt,), method="CROWN")
        lb, ub = lb.detach(), ub.detach()

        if not (torch.isfinite(lb).all() and torch.isfinite(ub).all()):
            raise AssertionError(f"non-finite bounds at r={r}")

        # soundness: sampled forward values must lie inside [lb, ub]
        rng = np.random.default_rng(2)
        s = torch.tensor(
            rng.uniform(center.numpy() - r, center.numpy() + r, size=(4000, 2)),
            dtype=torch.float32)
        with torch.no_grad():
            f = loop(s)
        viol = float(torch.maximum(lb - f, f - ub).max())
        if viol > 1e-4:
            raise AssertionError(f"UNSOUND at r={r}: sample outside bounds by {viol:.3e}")

        width = (ub - lb).squeeze(0)
        span = (f.max(0).values - f.min(0).values)
        looseness = (width / span.clamp(min=1e-9)).max().item()
        out.append(f"r={r}: width={width.tolist()}, looseness={looseness:.1f}x")

    return "sound at all radii; " + "; ".join(out)


if __name__ == "__main__":
    gate("dynamics constants match gymnasium", g1)
    gate("PendulumStep == env.step", g2)
    gate("extracted actor == SB3 predict", g3)
    gate("CROWN bounds the closed loop", g4)

    n_fail = sum(1 for r in results if r[0] == FAIL)
    print("\n" + "=" * 60)
    print(f"{len(results) - n_fail}/{len(results)} gates passed")
    sys.exit(1 if n_fail else 0)
