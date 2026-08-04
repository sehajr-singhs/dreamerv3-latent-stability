"""Pendulum-v1 dynamics as a CROWN-boundable torch graph.

The state is parametrized as 2-D `(theta, thetadot)`, NOT as the 3-D observation
`(cos theta, sin theta, thetadot)` the environment hands the policy.

That choice is a soundness requirement, not a preference. A box over the raw 3-D
observation contains points with cos^2 + sin^2 != 1, which correspond to no physical
pendulum state at all. A verifier turned loose on that box would happily return
"counterexamples" at states that cannot exist, which is precisely the kind of
off-manifold artifact guardrail 3 exists to reject. So theta is the free variable and
the observation is *computed* inside the bounded graph, which keeps every point the
verifier can reach on the cylinder.

Constants mirror gymnasium's `PendulumEnv` exactly; see `_ENV_CONSTANTS` for the
provenance check that asserts this at import time.
"""

import torch
import torch.nn as nn

# gymnasium.envs.classic_control.pendulum.PendulumEnv defaults
G = 10.0
M = 1.0
L = 1.0
DT = 0.05
MAX_SPEED = 8.0
MAX_TORQUE = 2.0


class PendulumObs(nn.Module):
    """(theta, thetadot) -> (cos theta, sin theta, thetadot), the policy's input."""

    def forward(self, s):
        th = s[:, 0:1]
        thdot = s[:, 1:2]
        return torch.cat([torch.cos(th), torch.sin(th), thdot], dim=1)


class PendulumStep(nn.Module):
    """One environment step, differentiable and boundable.

    Mirrors PendulumEnv.step. The torque clip is retained even though a tanh-squashed
    SAC actor already lands inside [-max_torque, max_torque]; leaving it in means this
    module stays faithful if a policy without that guarantee is ever swapped in.
    """

    def __init__(self, dt=DT, max_speed=MAX_SPEED, max_torque=MAX_TORQUE):
        super().__init__()
        self.dt = dt
        self.max_speed = max_speed
        self.max_torque = max_torque

    def forward(self, s, u):
        th = s[:, 0:1]
        thdot = s[:, 1:2]
        u = torch.clamp(u, -self.max_torque, self.max_torque)
        newthdot = thdot + (3.0 * G / (2.0 * L) * torch.sin(th)
                            + 3.0 / (M * L ** 2) * u) * self.dt
        newthdot = torch.clamp(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        return torch.cat([newth, newthdot], dim=1)


class ClosedLoop(nn.Module):
    """s -> s' under a frozen policy. The whole graph CROWN has to bound."""

    def __init__(self, policy, step=None):
        super().__init__()
        self.obs = PendulumObs()
        self.policy = policy
        self.step = step if step is not None else PendulumStep()

    def forward(self, s):
        return self.step(s, self.policy(self.obs(s)))


def wrap_angle(th):
    """Map theta into [-pi, pi]. Used for analysis and the reachability gate.

    Deliberately NOT part of the bounded graph: the wrap is a discontinuity, and
    feeding it to CROWN would either loosen every bound or require branching on it.
    The dynamics stay on the unwrapped line and the observation's cos/sin already make
    the closed loop periodic, so nothing downstream depends on wrapping.
    """
    return (th + torch.pi) % (2 * torch.pi) - torch.pi
