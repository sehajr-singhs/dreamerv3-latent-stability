"""A0c: what decay rate is even ACHIEVABLE near upright?

A0 found the narrow regions failing with violations of order 1e-3 to 1e-4, i.e. V
nearly flat rather than wrong. One specific hypothesis explains that: if the
closed-loop linearization at upright has spectral radius very close to 1, then V can
decrease only very slowly per step, and a training hinge demanding 1% decrease per step
is asking for something no Lyapunov function can deliver. Violations would then be an
artifact of the target, not of V.

This measures it rather than assuming it:

  1. A = d f_cl / ds at the closed loop's ACTUAL fixed point, by autograd. Not at
     upright: A0d showed upright is not a fixed point, so a Jacobian there is a valid
     derivative but says nothing about stability, since there is no equilibrium to be
     stable about.
  2. Its eigenvalues and spectral radius rho(A). Schur stable iff rho(A) < 1.
  3. The best achievable per-step decay of a QUADRATIC V along the dominant mode is
     rho(A)^2, so the largest feasible relative margin is about 1 - rho(A)^2.
  4. If stable, solve the discrete Lyapunov equation A' P A - P = -Q for P. That yields
     an ANALYTIC quadratic certificate for the linearization, with no training at all,
     and hence a clean upper bound on how well any V could do locally.

Step 4 matters beyond diagnostics. A certificate derived from the linearization needs no
fitting, so it removes training pathologies from the picture entirely and turns the audit
into a sharper question: over what region does a certificate that provably holds for the
LINEARIZED closed loop still hold for the TRUE nonlinear one? That is a genuine
sampling-versus-proof question with no learned-V confound.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import torch

from src.dynamics import ClosedLoop
from src.policy import extract_sac_actor
from src.equilibrium import find_attracting_fixed_point, verify_fixed_point, closed_loop_jacobian

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(seed=0):
    from stable_baselines3 import SAC

    model = SAC.load(os.path.join(ROOT, "models", "sac_pendulum"), device="cpu")
    net = extract_sac_actor(model)
    loop = ClosedLoop(net).eval()

    with torch.no_grad():
        s0 = torch.zeros(1, 2, dtype=torch.float32)
        drift_upright = float(torch.abs(loop(s0) - s0).max())
    u_upright = float(net(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)))
    print(f"at UPRIGHT: policy commands u = {u_upright:+.6f} (holding needs 0), "
          f"drift {drift_upright:.3e}")
    print("so upright is NOT a fixed point; linearizing there would describe nothing.\n")

    eq = find_attracting_fixed_point(loop, seed=seed)
    s_star = eq["s_star"]
    drift = verify_fixed_point(loop, s_star)
    u_star = float(net(torch.tensor([[np.cos(s_star[0]), np.sin(s_star[0]), s_star[1]]],
                                    dtype=next(net.parameters()).dtype)))
    print(f"fixed point s* = ({s_star[0]:+.8f}, {s_star[1]:+.8f}), drift {drift:.2e}")

    A = closed_loop_jacobian(loop, s_star)
    eig = np.linalg.eigvals(A)
    rho = float(np.max(np.abs(eig)))

    print(f"policy action at s*        u = {u_star:+.6f}   "
          f"(holding needs -(m g l/2) sin th* = {-5.0*np.sin(s_star[0]):+.6f})")
    print()
    print("closed-loop Jacobian A at s*:")
    print(np.array2string(A, precision=6))
    print(f"eigenvalues       {np.array2string(eig, precision=6)}")
    print(f"spectral radius   rho(A) = {rho:.6f}")
    print()

    out = dict(s_star=s_star.tolist(), u_at_s_star=u_star,
               u_at_upright=u_upright, drift_at_upright=drift_upright,
               drift_at_equilibrium=drift,
               A=A.tolist(), eigenvalues_real=eig.real.tolist(),
               eigenvalues_imag=eig.imag.tolist(), spectral_radius=rho)

    if rho >= 1.0:
        print("rho(A) >= 1: the closed loop is NOT locally asymptotically stable at")
        print("s*. No Lyapunov function with strict decrease exists in ANY")
        print("neighbourhood. That is a property of the trained policy, and it would")
        print("mean the audit's premise fails rather than its verifier.")
        out["verdict"] = "not_locally_stable"
    else:
        max_margin = 1.0 - rho ** 2
        print(f"Schur stable. Best per-step decay of a quadratic V along the dominant")
        print(f"mode is rho^2 = {rho**2:.6f}, so the largest feasible relative margin is")
        print(f"about 1 - rho^2 = {max_margin:.6f}  ({max_margin*100:.3f}% per step).")
        print()
        used = 0.01
        if used > max_margin:
            print(f"*** The A0 sweep trained with rel_margin = {used} = {used*100:.1f}%,")
            print(f"*** which EXCEEDS the achievable {max_margin*100:.3f}%. The hinge was")
            print("*** demanding a decay rate no Lyapunov function can deliver, so the")
            print("*** residual violations near the equilibrium are an artifact of the")
            print("*** target, not evidence that V is wrong.")
        else:
            print(f"rel_margin = {used} is within the achievable {max_margin:.6f}, so it")
            print("does not by itself explain the residual violations.")
        out["max_feasible_rel_margin"] = max_margin
        out["rel_margin_used_in_a0"] = used
        out["verdict"] = ("rel_margin_infeasible" if used > max_margin
                          else "rel_margin_feasible")

        # Analytic quadratic certificate for the linearization.
        try:
            from scipy.linalg import solve_discrete_lyapunov
            Q = np.eye(2)
            P = solve_discrete_lyapunov(A.T, Q)
            resid = A.T @ P @ A - P + Q
            print()
            print("discrete Lyapunov equation A' P A - P = -I:")
            print(np.array2string(P, precision=6))
            print(f"residual {np.abs(resid).max():.3e}, "
                  f"P eigenvalues {np.linalg.eigvalsh(P)}")
            print("This is a training-free quadratic certificate for the LINEARIZED loop.")
            out["P"] = P.tolist()
            out["P_residual"] = float(np.abs(resid).max())
        except Exception as e:
            print(f"(scipy unavailable for the Lyapunov solve: {e})")

    path = os.path.join(ROOT, "results", f"a0c_linearization_seed{seed}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
