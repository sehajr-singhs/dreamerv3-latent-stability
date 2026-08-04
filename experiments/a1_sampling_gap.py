"""A1: the sampling-to-proof gap on a frozen SB3 SAC policy for Pendulum-v1.

The question, stated precisely so the answer cannot be quietly inflated:

    Fix a frozen policy pi and a Lyapunov candidate V fit to it. Over a region
    defined FROM pi's own on-policy state distribution, minus a small ball about the
    upright equilibrium, how many states does a 500k-sample audit find where
    V(s) - V(f_cl(s)) < 0, and how many does CROWN branch-and-bound find that
    sampling missed AND that pi demonstrably reaches?

The last clause is the whole point. A counterexample that survives the reachability
gate is a real defect in the certificate. One that does not is an artifact of a box
drawn too wide, and is discarded here rather than reported with a caveat.

Nothing in this file hardcodes an expected gap. The number is whatever this run
measures, reported next to the environment, region, and condition that produced it.
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
from src.lyapunov import DecreaseCondition
from src.policy import extract_sac_actor, check_matches_sb3
from src.reachability import collect_support, IN_SUPPORT
from src.train_lyapunov import train_V, sampled_violation_rate
from src.verifier import certify_box, provenance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate_policy_return(model, n_episodes=30):
    import gymnasium as gym
    from stable_baselines3.common.evaluation import evaluate_policy
    m, s = evaluate_policy(model, gym.make("Pendulum-v1"),
                           n_eval_episodes=n_episodes, deterministic=True)
    return float(m), float(s)


def run(seed=0, n_samples=500_000, v_steps=4000, quantile=0.99,
        hole=0.05, time_budget=120.0, method="CROWN", min_width=1e-3,
        n_episodes=2000, horizon=200, burn_in=100, out=None):
    t_start = time.time()
    from stable_baselines3 import SAC

    log = {}
    log["config"] = dict(seed=seed, n_samples=n_samples, v_steps=v_steps,
                         quantile=quantile, hole=hole, time_budget=time_budget,
                         method=method, min_width=min_width,
                         n_episodes=n_episodes, horizon=horizon, burn_in=burn_in,
                         env="Pendulum-v1")
    log["verifier"] = provenance()

    # ---------------------------------------------------------------- policy
    model = SAC.load(os.path.join(ROOT, "models", "sac_pendulum"), device="cpu")
    net = extract_sac_actor(model)

    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi, np.pi, size=512)
    tv = rng.uniform(-8, 8, size=512)
    obs = np.stack([np.cos(th), np.sin(th), tv], axis=1).astype(np.float32)
    extraction_err = check_matches_sb3(net, model, obs)

    mean_r, std_r = evaluate_policy_return(model)
    log["policy"] = dict(mean_return=mean_r, std_return=std_r,
                         extraction_max_err=extraction_err,
                         net_arch=[64, 64], frozen=True)
    print(f"policy: return {mean_r:.1f} +/- {std_r:.1f}, "
          f"extraction err {extraction_err:.2e}")
    if mean_r < -300:
        print("WARNING: policy looks undertrained; audit results describe a bad policy.")

    # ------------------------------------------------------ on-policy support
    support = collect_support(net, n_episodes=n_episodes, horizon=horizon, seed=seed)
    log["support"] = support.coverage_report(n_episodes, horizon)
    print(f"support (full): {log['support']['n_states']} states, "
          f"theta p99 {log['support']['theta_p99']}, "
          f"thetadot p99 {log['support']['thetadot_p99']}")

    # The certification region comes from the STEADY-STATE portion of the rollouts.
    # Pendulum's policy is a swing-up controller: from a hanging start it must pump
    # energy in before it can stabilize, so V necessarily increases during the
    # transient and NO monotone Lyapunov function exists over the full support.
    # Certifying decrease there is impossible rather than hard, and a violation found
    # there would say nothing about the certificate. The reachability gate below still
    # uses the FULL support, because a transient state is still a state pi reaches.
    steady = support.tail(burn_in)
    log["steady_state"] = steady.coverage_report(n_episodes, horizon - burn_in)
    log["steady_state"]["burn_in"] = burn_in
    log["steady_state"]["why"] = (
        "Region defined from steps >= burn_in only. Swing-up necessarily increases any "
        "Lyapunov function, so the full on-policy support admits no monotone V. The "
        "reachability gate still uses the full support.")
    print(f"support (steady, burn_in={burn_in}): {log['steady_state']['n_states']} states, "
          f"theta p99 {log['steady_state']['theta_p99']}, "
          f"thetadot p99 {log['steady_state']['thetadot_p99']}")

    # ---------------------------------------------------------------- region
    lo, hi = steady.quantile_box(quantile)
    # keep the box symmetric about upright so the annulus is not lopsided
    half = np.maximum(np.abs(lo), np.abs(hi))
    lo, hi = -half, half
    hole_lo, hole_hi = np.array([-hole, -hole]), np.array([hole, hole])
    boxes = R.annulus_boxes(lo, hi, hole_lo, hole_hi)

    log["region"] = dict(
        box_vs_steady_state=R.describe(lo, hi, steady),
        box_vs_full_support=R.describe(lo, hi, support),
        hole=dict(lo=hole_lo.tolist(), hi=hole_hi.tolist(),
                  why=("cond(x*) = 0 exactly by construction, so any box containing x* "
                       "has true infimum 0 and cannot certify at a positive margin. "
                       "Structural, not a verifier failure.")),
        n_annulus_boxes=len(boxes),
        annulus_steady_state_coverage=R.coverage_of(boxes, steady),
        annulus_full_support_coverage=R.coverage_of(boxes, support),
        quantile=quantile,
    )
    print(f"region: box {np.round(lo,4).tolist()} .. {np.round(hi,4).tolist()}; "
          f"holds {log['region']['box_vs_steady_state']['frac_on_policy_states_inside']:.3%} "
          f"of steady-state states, "
          f"{log['region']['box_vs_full_support']['frac_on_policy_states_inside']:.3%} "
          f"of all visited states")

    # ------------------------------------------------------------- fit V
    print("training V ...")
    V = train_V(net, steady, lo, hi, hole_lo, hole_hi, steps=v_steps, seed=seed)
    loop = ClosedLoop(net)
    cond = DecreaseCondition(V, loop).eval()
    log["lyapunov"] = dict(form="||g(o)-g(o*)||^2 + c||o-o*||^2", c=0.5, hidden=32,
                           input="observation (cos th, sin th, thdot), cylinder-correct",
                           steps=v_steps)

    # --------------------------------------------------- sampling-only audit
    print(f"sampling audit ({n_samples} states) ...")
    samp = sampled_violation_rate(cond, lo, hi, hole_lo, hole_hi, steady,
                                  n=n_samples, seed=seed + 999)
    log["sampling"] = samp
    print(f"sampling: {samp['n_violations']}/{samp['n_sampled']} violations "
          f"({samp['violation_rate']:.3%}), worst cond {samp['worst_cond']:.3e}")

    # ----------------------------------------------------------- BaB audit
    print(f"branch-and-bound over {len(boxes)} annulus boxes ...")
    box_results, ces = [], []
    for i, (blo, bhi) in enumerate(boxes):
        xL = torch.tensor(blo, dtype=torch.float32).reshape(1, -1)
        xU = torch.tensor(bhi, dtype=torch.float32).reshape(1, -1)
        r = certify_box(cond, xL, xU, eps=0.0, method=method,
                        min_width=min_width, time_budget=time_budget, seed=seed)
        r["box"] = dict(lo=blo.tolist(), hi=bhi.tolist())
        box_results.append(r)
        print(f"  box {i}: {r['verdict']:>9}  "
              f"{r.get('subdomains','?')} subdomains  {r.get('seconds','?')}s")
        if r["verdict"] == "violation":
            ces.append(r["counterexample"])
    log["bab"] = dict(boxes=box_results,
                      n_violation=sum(1 for r in box_results if r["verdict"] == "violation"),
                      n_unknown=sum(1 for r in box_results if r["verdict"] == "unknown"),
                      n_certified=sum(1 for r in box_results if r.get("certified")))

    # ------------------------------------------------- reachability gate
    gated = []
    if ces:
        arr = np.array(ces, dtype=np.float64)
        verdicts, dists = support.verdict(arr)
        with torch.no_grad():
            cvals = cond(torch.tensor(arr, dtype=torch.float32)).squeeze(-1).numpy()
        for s, v, d, cv in zip(arr.tolist(), verdicts, dists.tolist(), cvals.tolist()):
            gated.append(dict(state=s, cond=float(cv), reach_verdict=v,
                              normalized_distance_to_support=float(d)))
            print(f"  CE {['%.4f'%x for x in s]} cond={cv:.3e} -> {v} (d={d:.4f})")
    log["counterexamples"] = gated

    n_real = sum(1 for g in gated if g["reach_verdict"] == IN_SUPPORT)
    log["result"] = dict(
        sampling_violations=samp["n_violations"],
        sampling_rate=samp["violation_rate"],
        bab_counterexamples=len(gated),
        reachable_counterexamples=n_real,
        gap_demonstrated=bool(samp["n_violations"] == 0 and n_real > 0),
        interpretation=(
            "gap_demonstrated is True only when a 500k-sample audit found NOTHING and "
            "branch-and-bound found at least one violation at a state the frozen policy "
            "demonstrably reaches. Counterexamples failing the reachability gate are "
            "reported above for transparency but are NOT findings."),
        seconds=round(time.time() - t_start, 1),
    )

    out = out or os.path.join(ROOT, "results", f"a1_seed{seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nwrote {out}")
    print(json.dumps(log["result"], indent=2))
    return log


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=500_000)
    p.add_argument("--v-steps", type=int, default=4000)
    p.add_argument("--time-budget", type=float, default=120.0)
    p.add_argument("--method", default="CROWN",
                   help="plain CROWN by default: E10 measured CROWN-Optimized as far "
                        "WORSE on sum-of-squares V (product nodes make alpha-optimization "
                        "expensive). Do not switch without re-measuring.")
    p.add_argument("--quick", action="store_true",
                   help="smoke-test sizes, for local runs only; never quote these numbers")
    a = p.parse_args()
    if a.quick:
        run(seed=a.seed, n_samples=20_000, v_steps=300, time_budget=20.0,
            n_episodes=200, horizon=100, burn_in=50,
            out=os.path.join(ROOT, "results", f"a1_quick_seed{a.seed}.json"))
    else:
        run(seed=a.seed, n_samples=a.n_samples, v_steps=a.v_steps,
            time_budget=a.time_budget, method=a.method)
