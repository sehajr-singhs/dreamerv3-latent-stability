"""A0b: where COULD a monotone Lyapunov function exist, on physics alone?

This bounds the feasible region before any learning is involved, which matters for
reading A0's sweep honestly. If V cannot be fit on a region, there are two very
different reasons available:

  (i)  the region exceeds what the actuator can hold, so no controller whatsoever
       admits a monotone V there, or
  (ii) V or its training is inadequate.

Only (ii) is a shortcoming of our method. Conflating them would let us report a
verifier limitation as a physical law, or worse, a physical law as a finding.

The static bound, derived from the env's ACTUAL equation of motion rather than from a
textbook point-mass pendulum. gymnasium integrates

    thddot = (3g / 2l) sin(th) + (3 / m l^2) u

which is a uniform ROD pivoted at one end (I = m l^2 / 3, gravity torque m g (l/2) sin th),
not a point mass at radius l. Setting thddot = 0 gives the holding torque

    u = -(m g l / 2) sin(th)

so with |u| <= max_torque the pendulum can be held only where

    |sin(th)| <= max_torque / (m g l / 2)

An earlier version of this file used the point-mass balance u = m g l sin(th) and so
reported a limit of 0.2014 rad, a factor of two too tight. The corrected bound is
checked below against the closed loop's second fixed point, which A0d locates by Newton
at exactly the saturation boundary; that agreement is what makes this the right formula
rather than another plausible one.

This is a NECESSARY condition on the region, not a sufficient one; the true basin is
smaller still, because it must also account for kinetic energy.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.dynamics import G, M, L, MAX_TORQUE, MAX_SPEED


def main():
    hold_gain = M * G * L / 2.0            # |u| needed per unit sin(th), rod about end
    ratio = MAX_TORQUE / hold_gain
    th_max = float(np.arcsin(ratio))

    print(f"env EOM            thddot = (3g/2l) sin(th) + (3/m l^2) u   [rod about end]")
    print(f"holding torque     |u| = (m g l / 2) |sin th| = {hold_gain:.3f} |sin th|")
    print(f"max_torque         = {MAX_TORQUE:.3f}")
    print(f"ratio              = {ratio:.4f}")
    print()
    print(f"STATIC HOLD LIMIT  |theta| <= {th_max:.6f} rad = {np.degrees(th_max):.2f} deg")
    print()
    print("Beyond that band the actuator cannot balance gravity, so the pendulum must")
    print("fall and be swung back up. No controller admits a monotone Lyapunov function")
    print("on a region extending past it. This bounds the region NECESSARILY, not")
    print("sufficiently: the true certifiable basin is smaller once velocity is included.")
    print()

    # Cross-check against the measured second fixed point, if A0d has been run. The
    # closed loop's non-attracting fixed point should sit essentially ON this boundary,
    # since that is where the policy saturates trying to hold.
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "a0d_equilibrium_seed0.json")
    if os.path.isfile(path):
        import json
        d = json.load(open(path))
        far = d.get("newton_from_upright", {})
        if far.get("converged"):
            th_far = far["theta"]
            print(f"cross-check: A0d's second fixed point sits at theta = {th_far:.6f} rad,")
            print(f"             against the predicted saturation boundary {th_max:.6f} rad")
            print(f"             (difference {abs(th_far - th_max):.2e} rad). Agreement here")
            print("             is what validates the torque balance used above.")
            print()

    print("For reference, the regions A0 swept (boxes centred on ZERO, which A0d showed")
    print("is NOT the closed loop's equilibrium):")
    for name, half in [("burn_in=10  (approach)", 2.65),
                       ("burn_in=25", 1.95),
                       ("burn_in=50", 0.172),
                       ("burn_in=100 (settled)", 0.143)]:
        verdict = "EXCEEDS hold limit" if half > th_max else "within hold limit"
        print(f"  {name:24s} theta half-width {half:.3f} rad -> {verdict}")
    print()
    print(f"So only regions with |theta| <~ {th_max:.3f} rad can possibly certify, and a")
    print("failure to fit V outside that band is physics, not a defect in the method.")


if __name__ == "__main__":
    main()
