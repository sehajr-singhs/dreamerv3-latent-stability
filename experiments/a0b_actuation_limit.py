"""A0b: where COULD a monotone Lyapunov function exist, on physics alone?

This bounds the feasible region before any learning is involved, which matters for
reading A0's sweep honestly. If V cannot be fit on a region, there are two very
different reasons available:

  (i)  the region exceeds what the actuator can hold, so no controller whatsoever
       admits a monotone V there, or
  (ii) V or its training is inadequate.

Only (ii) is a shortcoming of our method. Conflating them would let us report a
verifier limitation as a physical law, or worse, a physical law as a finding.

The static bound. Holding the pendulum at angle th requires torque balancing gravity,
u = m*g*l*sin(th). With |u| <= max_torque this is possible only where

    |sin(th)| <= max_torque / (m*g*l)

Outside that band the pendulum cannot be held at all: it must fall and be swung back
up, so V necessarily increases somewhere along the way. This is a NECESSARY condition
on the region, not a sufficient one; the true basin is smaller still, because it must
also account for kinetic energy.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.dynamics import G, M, L, MAX_TORQUE, MAX_SPEED


def main():
    ratio = MAX_TORQUE / (M * G * L)
    th_max = float(np.arcsin(ratio))

    print(f"m*g*l              = {M*G*L:.3f}   (gravity torque at th = 90 deg)")
    print(f"max_torque         = {MAX_TORQUE:.3f}")
    print(f"ratio              = {ratio:.4f}")
    print()
    print(f"STATIC HOLD LIMIT  |theta| <= {th_max:.4f} rad = {np.degrees(th_max):.2f} deg")
    print()
    print("Beyond that band the actuator cannot balance gravity, so the pendulum must")
    print("fall and be swung back up. No controller admits a monotone Lyapunov function")
    print("on a region extending past it. This bounds the region NECESSARILY, not")
    print("sufficiently: the true certifiable basin is smaller once velocity is included.")
    print()

    # Where does the swing-up have to happen, in energy terms? Upright energy is the
    # maximum of the potential, so any state with too little total energy simply cannot
    # reach upright without the controller adding energy first.
    print("For reference, the regions A0 sweeps:")
    for name, half in [("burn_in=10  (approach)", (2.65, 6.75)),
                       ("burn_in=100 (settled)", (0.143, 0.103))]:
        verdict = "EXCEEDS static hold limit" if half[0] > th_max else "within static hold limit"
        print(f"  {name:24s} theta half-width {half[0]:.3f} rad -> {verdict}")
    print()
    print(f"So only regions with |theta| <~ {th_max:.3f} rad can possibly certify, and a")
    print("failure to fit V outside that band is physics, not a defect in the method.")


if __name__ == "__main__":
    main()
