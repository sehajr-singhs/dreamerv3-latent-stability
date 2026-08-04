"""A0: for WHICH regions does this frozen policy admit a monotone Lyapunov function?

Run before A1. A sampling-to-proof gap is only measurable where sampling is CLEAN
(empirical violation rate at or near zero). Where it is not, the honest conclusion is
"no such certificate exists on this region", which is a real result but a different
claim, and presenting a leftover violation rate as a gap would be dishonest.

The first version of this probe swept the TRAINING BUDGET and found the violation rate
plateauing near 20% regardless of budget. That was the wrong knob. The region is the
knob, and it is squeezed from both ends:

  burn_in small  ->  region includes the swing-up transient. The policy must PUMP ENERGY
                     IN before it can stabilize, so V necessarily increases and no
                     monotone V exists. Impossible, not hard.
  burn_in large  ->  region collapses onto the equilibrium, where cond -> 0 by
                     construction (cond(x*) = 0 exactly). Violations there are ~1e-3 and
                     the condition is degenerate rather than informative.

So this sweeps burn_in, which traces a family of regions from "approach" to "settled",
each one justified by construction as "states the policy visits at step >= burn_in".
That keeps guardrail 3 satisfied automatically: every region IS an on-policy region.

Also fixed here: the hole is now a FRACTION of the region half-width. A fixed +/-0.05
hole against a +/-0.14 region removed most of the volume being certified.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src import region as R
from src.dynamics import ClosedLoop
from src.lyapunov import DecreaseCondition, ExponentialDecreaseCondition
from src.policy import extract_sac_actor
from src.reachability import collect_support
from src.train_lyapunov import train_V, sampled_violation_rate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(seed=0, quantile=0.99, hole_frac=0.10, v_steps=2000,
         burn_ins=(10, 25, 50, 100), cs=(0.05, 0.5), hidden=32,
         n_samples=100_000, n_episodes=1000, horizon=200, beta=None):
    from stable_baselines3 import SAC

    model = SAC.load(os.path.join(ROOT, "models", "sac_pendulum"), device="cpu")
    net = extract_sac_actor(model)
    loop = ClosedLoop(net)

    support = collect_support(net, n_episodes=n_episodes, horizon=horizon, seed=seed)
    print(f"full support: {len(support.states)} states, "
          f"theta p99 {np.round(support.quantile_box(0.99)[0][0],3)} .. "
          f"{np.round(support.quantile_box(0.99)[1][0],3)}", flush=True)

    rows = []
    for b in burn_ins:
        steady = support.tail(b)
        lo, hi = steady.quantile_box(quantile)
        half = np.maximum(np.abs(lo), np.abs(hi))
        lo, hi = -half, half
        # hole scales with the region, so the annulus keeps its shape as the box moves
        hole = hole_frac * half
        hole_lo, hole_hi = -hole, hole

        cov_steady = R.describe(lo, hi, steady)["frac_on_policy_states_inside"]
        cov_full = R.describe(lo, hi, support)["frac_on_policy_states_inside"]
        print(f"\nburn_in={b:4d}  region +/-{np.round(half,4).tolist()}  "
              f"hole +/-{np.round(hole,4).tolist()}  "
              f"covers {cov_steady:.1%} steady / {cov_full:.1%} full", flush=True)

        for c in cs:
            t0 = time.time()
            V = train_V(net, steady, lo, hi, hole_lo, hole_hi,
                        steps=v_steps, hidden=hidden, c=c, seed=seed, verbose=False)
            cond = (DecreaseCondition(V, loop) if beta is None
                    else ExponentialDecreaseCondition(V, loop, beta=beta)).eval()
            s = sampled_violation_rate(cond, lo, hi, hole_lo, hole_hi, steady,
                                       n=n_samples, seed=seed + 999)
            row = dict(burn_in=b, c=c, v_steps=v_steps, beta=beta,
                       half_width=half.tolist(), hole=hole.tolist(),
                       cov_steady=cov_steady, cov_full=cov_full,
                       violation_rate=s["violation_rate"],
                       n_violations=s["n_violations"], n_sampled=s["n_sampled"],
                       worst_cond=s["worst_cond"], seconds=round(time.time() - t0, 1))
            rows.append(row)
            print(f"    c={c:<5} viol={s['violation_rate']:9.5%} "
                  f"({s['n_violations']}/{s['n_sampled']})  "
                  f"worst={s['worst_cond']:.3e}  [{row['seconds']}s]", flush=True)

    out = os.path.join(ROOT, "results", f"a0_feasibility_seed{seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(config=dict(seed=seed, quantile=quantile, hole_frac=hole_frac,
                                   v_steps=v_steps, hidden=hidden, beta=beta,
                                   n_samples=n_samples, n_episodes=n_episodes,
                                   horizon=horizon),
                       rows=rows), f, indent=2)
    print(f"\nwrote {out}")

    best = min(rows, key=lambda r: r["violation_rate"])
    print(f"\nbest: {best['violation_rate']:.5%} at burn_in={best['burn_in']}, "
          f"c={best['c']}, region +/-{np.round(best['half_width'],4).tolist()}")
    if best["violation_rate"] > 1e-4:
        print("VERDICT: no region here yields clean sampling. A gap CANNOT be claimed. "
              "Report the obstruction (swing-up vs degeneracy) as the result instead.")
    else:
        print("VERDICT: clean sampling reached; a gap is measurable on that region.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--v-steps", type=int, default=2000)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--hole-frac", type=float, default=0.10)
    p.add_argument("--burn-ins", type=int, nargs="+", default=[10, 25, 50, 100])
    p.add_argument("--cs", type=float, nargs="+", default=[0.05, 0.5])
    p.add_argument("--beta", type=float, default=None,
                   help="if set, certify V(f) <= beta*V instead of plain decrease")
    a = p.parse_args()
    main(seed=a.seed, v_steps=a.v_steps, hidden=a.hidden, hole_frac=a.hole_frac,
         burn_ins=tuple(a.burn_ins), cs=tuple(a.cs), beta=a.beta)
