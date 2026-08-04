"""The reachability gate. Guardrail 3, and the thing that separates a finding from a
party trick.

Any unconstrained network violates almost any property somewhere, so "CROWN found a
counterexample in a box" is worth nothing on its own; widen the box and you can always
manufacture one. The only violations that matter are ones at states the policy can
actually put itself in, starting from initial conditions the environment actually
hands it. That is what this module decides, and it decides it empirically rather than
by assertion.

Method. Roll the FROZEN policy out from the environment's own reset distribution
(Pendulum-v1: theta ~ U(-pi, pi), thetadot ~ U(-1, 1)) and record every state visited.
That set IS the on-policy support, by construction. A candidate counterexample is then
scored by its distance to the nearest visited state.

Metric. theta is periodic and thetadot is not, and they carry different units, so a
raw Euclidean distance would be meaningless. Distances are computed on the wrapped
angular difference and are normalized by each coordinate's physical range (2*pi for
theta, 2*max_speed for thetadot), making the score dimensionless and comparable
across coordinates. A score of 0.01 means "within 1% of the state space's extent of a
state the policy actually visited."

Honesty notes baked in:
  - The verdict is three-way, mirroring the verifier: IN_SUPPORT / NEAR / OFF_DISTRIBUTION.
    OFF_DISTRIBUTION is not a weaker finding, it is NOT a finding, and callers must
    drop those counterexamples rather than reporting them with a caveat.
  - Empirical support is a lower bound on true reachability: absence from the sample
    is evidence of rarity, not proof of unreachability. `coverage_report` exists so
    the sample size behind any such claim is always stated next to it.
"""

import numpy as np
import torch

from .dynamics import PendulumStep, PendulumObs, MAX_SPEED

# Verdicts
IN_SUPPORT = "IN_SUPPORT"
NEAR = "NEAR"
OFF_DISTRIBUTION = "OFF_DISTRIBUTION"

# Normalization ranges, stated once so every reported distance is interpretable.
THETA_RANGE = 2.0 * np.pi
THDOT_RANGE = 2.0 * MAX_SPEED


def wrap(th):
    return (th + np.pi) % (2 * np.pi) - np.pi


class OnPolicySupport:
    """Empirical on-policy state distribution of a frozen policy.

    Parameters
    ----------
    in_support_tol : float
        Normalized distance within which a state counts as visited. Defaults to 0.01,
        i.e. 1% of the state space extent. Report it with every verdict.
    near_tol : float
        Outer band. Beyond this, OFF_DISTRIBUTION.
    """

    def __init__(self, states, in_support_tol=0.01, near_tol=0.05, by_step=None):
        # by_step, when present, has shape (T+1, n_episodes, 2) and lets `tail` carve
        # out the post-transient (steady-state) portion of the same rollouts.
        self.by_step = by_step
        self.states = np.asarray(states, dtype=np.float64)
        self.states[:, 0] = wrap(self.states[:, 0])
        self.in_support_tol = float(in_support_tol)
        self.near_tol = float(near_tol)
        self._tree = None
        # Duplicate the angular coordinate at +/-2pi so a KD-tree (which knows nothing
        # about periodicity) still measures the true wrapped distance near +/-pi.
        if len(self.states):
            try:
                from scipy.spatial import cKDTree
                pts = self._embed(self.states)
                shifted = [pts]
                for d in (-1.0, 1.0):
                    q = pts.copy()
                    q[:, 0] += d          # one full period == 1.0 in normalized units
                    shifted.append(q)
                self._tree = cKDTree(np.vstack(shifted))
            except ImportError:
                self._tree = None

    @staticmethod
    def _embed(s):
        """Normalized coordinates: theta/(2pi), thetadot/(2*max_speed)."""
        out = np.empty_like(np.asarray(s, dtype=np.float64))
        out[:, 0] = wrap(np.asarray(s)[:, 0]) / THETA_RANGE
        out[:, 1] = np.asarray(s)[:, 1] / THDOT_RANGE
        return out

    def distance(self, query):
        """Normalized distance from each query state to the nearest visited state."""
        q = np.atleast_2d(np.asarray(query, dtype=np.float64))
        qe = self._embed(q)
        if self._tree is not None:
            d, _ = self._tree.query(qe, k=1)
            return np.asarray(d).ravel()
        # Brute-force fallback, chunked so a large support set does not blow up memory.
        pts = self._embed(self.states)
        out = np.empty(len(qe))
        for i in range(0, len(qe), 256):
            blk = qe[i:i + 256]
            dth = blk[:, None, 0] - pts[None, :, 0]
            dth -= np.round(dth)                       # wrapped, in normalized units
            dv = blk[:, None, 1] - pts[None, :, 1]
            out[i:i + 256] = np.sqrt(dth ** 2 + dv ** 2).min(axis=1)
        return out

    def verdict(self, query):
        d = self.distance(query)
        return [
            IN_SUPPORT if x <= self.in_support_tol
            else NEAR if x <= self.near_tol
            else OFF_DISTRIBUTION
            for x in d
        ], d

    def gate(self, query):
        """True only for states the policy demonstrably reaches. NEAR does not pass.

        Deliberately strict: a counterexample that only *nearly* sits on the support is
        not something we will put in front of the DreamerV3 authors as a finding.
        """
        v, _ = self.verdict(query)
        return np.array([x == IN_SUPPORT for x in v])

    def tail(self, burn_in):
        """The same rollouts restricted to steps >= burn_in: the steady-state region.

        Why this exists. Pendulum's policy is a SWING-UP controller: from a hanging
        start it must pump energy in before it can stabilize, so V genuinely INCREASES
        during the transient and no monotone Lyapunov function can exist over the full
        on-policy support. Certifying decrease there is not a hard problem, it is an
        impossible one, and a violation found there says nothing about the certificate.

        So the certification REGION is defined from the steady-state portion, while the
        reachability gate keeps using the full visited set (a transient state is still
        a state the policy reaches). Keeping those two roles separate is the point.
        """
        if self.by_step is None:
            raise ValueError("no per-step data retained; call collect_support(keep_by_step=True)")
        sub = self.by_step[burn_in:].reshape(-1, self.by_step.shape[-1])
        return OnPolicySupport(sub, self.in_support_tol, self.near_tol,
                               by_step=self.by_step[burn_in:])

    def quantile_box(self, q=0.99):
        """Central box holding fraction q of visited states, per coordinate.

        This is what guardrail 3 means by "report every box relative to the on-policy
        state distribution": a verification box should be quoted against this, so a
        reader can see whether it covers where the policy lives or somewhere it never goes.
        """
        lo = np.quantile(self.states, (1 - q) / 2, axis=0)
        hi = np.quantile(self.states, 1 - (1 - q) / 2, axis=0)
        return lo, hi

    def coverage_report(self, n_episodes, horizon):
        lo, hi = self.quantile_box(0.99)
        return dict(
            n_states=int(len(self.states)),
            n_episodes=int(n_episodes),
            horizon=int(horizon),
            in_support_tol=self.in_support_tol,
            near_tol=self.near_tol,
            theta_p99=[float(lo[0]), float(hi[0])],
            thetadot_p99=[float(lo[1]), float(hi[1])],
            theta_minmax=[float(self.states[:, 0].min()), float(self.states[:, 0].max())],
            thetadot_minmax=[float(self.states[:, 1].min()), float(self.states[:, 1].max())],
            note=("Empirical support is a LOWER BOUND on reachability: a state absent "
                  "from this sample is rare under the reset distribution, not proven "
                  "unreachable."),
        )


def collect_support(policy, n_episodes=2000, horizon=200, seed=0,
                    in_support_tol=0.01, near_tol=0.05):
    """Roll the frozen policy out from Pendulum's reset distribution.

    Uses our own PendulumStep rather than the gym env, which is exactly the dynamics
    the verifier bounds. That is intentional: the gate must describe reachability
    under the SAME transition function being certified, or it is answering a different
    question than the one asked. Gate 2 of the smoke test is what licenses this, having
    shown the two agree to 1e-16.
    """
    rng = np.random.default_rng(seed)
    # Pendulum-v1 reset: theta ~ U(-pi, pi), thetadot ~ U(-1, 1).
    th = rng.uniform(-np.pi, np.pi, size=n_episodes)
    thdot = rng.uniform(-1.0, 1.0, size=n_episodes)
    s = torch.tensor(np.stack([th, thdot], axis=1), dtype=torch.float32)

    obs, step = PendulumObs(), PendulumStep()
    visited = [s.numpy().copy()]
    with torch.no_grad():
        for _ in range(horizon):
            s = step(s, policy(obs(s)))
            visited.append(s.numpy().copy())

    by_step = np.stack(visited, axis=0)          # (horizon+1, n_episodes, 2)
    return OnPolicySupport(by_step.reshape(-1, by_step.shape[-1]),
                           in_support_tol=in_support_tol, near_tol=near_tol,
                           by_step=by_step)
