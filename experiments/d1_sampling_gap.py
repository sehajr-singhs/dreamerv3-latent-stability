"""D1: the sampling-to-proof gap on a frozen DreamerV3 one-step latent transition.

Mirrors A1 exactly, on the latent space, per the locked scope (one-step transition,
smallest official model size):

    Fix a frozen DreamerV3 (size1m) and a Lyapunov candidate V fit to its one-step
    latent dynamics. Over a latent region defined FROM the model's own on-policy
    latent distribution, minus an annulus around the latent fixed point z*, how
    many states fail V(z) - V(T(z, pi(z))) < 0, and how many does CROWN
    branch-and-bound find that sampling missed AND that the model reaches?

The last clause is the same gate as A1: a counterexample only counts if it lies on
the model's own latent support. The certified object is the DETERMINISTIC one-step
map under the deterministic actor: z' = T(z, pi(z)) with pi(z) = tanh(mean(actor)).
The stoch component is the mean of the prior categorical. Stated in every artifact.

Nothing here hardcodes an expected gap. The number is whatever this run measures,
reported next to the model, region, and condition that produced it.

Inputs (produced by the Colab notebook):
  --weights  npz of the JAX checkpoint params under upstream key names
             (world_model/dyn/..., world_model/pol/...), as exported by
             colab/export_dreamer.py
  --support  npz of on-policy latent states from encoder rollouts:
             deter (N, 512), stoch (N, 32, 4), n_episodes, horizon
  --synthetic  build a random-weight model and a synthetic closed-loop support.
               Smoke only; numbers from a synthetic run are NEVER quoted.
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src import region as R
from src.dreamer import (OneStepLatentTransition, PolicyMean, OneStepClosedLoop,
                         RSSMConfig, load_jax_arrays, UPSTREAM_DREAMERV3,
                         UPSTREAM_EMBODIED)
from src.lyapunov import DecreaseCondition
from src.lyapunov_latent import (LatentLyapunovPD, train_latent_V,
                                 sampled_latent_violation_rate)
from src.verifier import certify_box, provenance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_SUPPORT, NEAR, OFF_DISTRIBUTION = "IN_SUPPORT", "NEAR", "OFF_DISTRIBUTION"


def build_frozen(cfg, action_dim, weights_npz=None):
    """OneStepLatentTransition + PolicyMean + closed loop, FROZEN.

    Frozen means requires_grad_(False) after construction/loading: the audit
    certifies the model as trained; only V moves. Freezing also makes V training
    and the verifier cheaper and prevents accidental drift.
    """
    trans = OneStepLatentTransition(cfg, action_dim).eval()
    pol = PolicyMean(cfg.deter, cfg.stoch, cfg.classes, act_dim=action_dim).eval()
    if weights_npz is not None:
        arrays = np.load(weights_npz)
        load_jax_arrays(trans, {k: arrays[k] for k in arrays.files})
        load_jax_arrays(pol, {k: arrays[k] for k in arrays.files})
    loop = OneStepClosedLoop(trans, pol).eval()
    for p in loop.parameters():
        p.requires_grad_(False)
    return trans, pol, loop


@torch.no_grad()
def find_latent_fixed_point(loop, cfg, support_deter, support_stoch, *,
                            settle_iters=5000, polish_iters=500, tol=1e-6, seed=0):
    """Settle the one-step closed loop onto its attracting fixed point.

    z* must satisfy T(z*, pi(z*)) = z*, or cond(z*) < 0 by construction (the A1
    centring lesson, on the latent). Start from the mean of a few support states
    and iterate the deterministic closed loop; then damped polish. The residual is
    verified and the run REFUSES to proceed if it is not a fixed point: a V
    centred elsewhere is guaranteed a violation at its centre.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(support_deter), size=64)
    deter = torch.tensor(support_deter[idx], dtype=torch.float32).mean(0, keepdim=True)
    stoch = torch.tensor(support_stoch[idx], dtype=torch.float32).mean(0, keepdim=True)
    stoch = stoch / stoch.sum(-1, keepdim=True).clamp(min=1e-9)   # valid simplex point
    z = torch.cat([deter, torch.flatten(stoch, start_dim=-2)], -1)
    deter_dim = cfg.deter

    residual = float("inf")
    for _ in range(settle_iters):
        z2 = loop(z)
        residual = float((z2 - z).abs().amax())
        if residual < tol:
            break
        z = z2
    for _ in range(polish_iters):                    # damped polish, halves oscillation
        z = 0.5 * (z + loop(z))
    residual = float((loop(z) - z).abs().amax())
    if residual > 1e-4:
        raise RuntimeError(
            f"latent fixed point not reached (residual {residual:.2e}); a V centred "
            "here would be guaranteed cond(centre) < 0. Do not proceed.")
    z_star = z.squeeze(0)
    return dict(z_star=z_star.detach().cpu().numpy(), residual=residual,
                converged=residual <= 1e-6)


def latent_reach_verdict(ce, support_deter, support_stoch, scale, in_tol=0.01,
                         near_tol=0.05):
    """Normalized distance from a counterexample to the nearest support latent.

    Dimensionless: each coordinate is divided by its support range so the distance
    is comparable across the 512 deter and 128 stoch coordinates (the same idea as
    the pendulum gate, without a periodic coordinate to wrap).
    """
    z = np.concatenate([np.asarray(ce[:support_deter.shape[1]]),
                        np.asarray(ce[support_deter.shape[1]:]).reshape(1, -1)], 1)
    sup = np.concatenate([np.asarray(support_deter), np.asarray(support_stoch).reshape(
        len(support_deter), -1)], 1)
    dev = (z[None, :] - sup) / scale[None, :]
    dist = float(np.sqrt((dev * dev).sum(-1)).min())
    if dist <= in_tol:
        return IN_SUPPORT, dist
    if dist <= near_tol:
        return NEAR, dist
    return OFF_DISTRIBUTION, dist


def run(cfg=None, action_dim=6, weights_npz=None, support_npz=None, *,
        seed=0, n_samples=500_000, v_steps=4000, v_batch=4096,
        quantile=0.99, hole=0.10, hole_k=4, time_budget=120.0,
        method="CROWN", min_width=1e-3, n_episodes=2000, horizon=200,
        burn_in=100, synthetic=False, out=None):
    t_start = time.time()
    # The heavy run belongs on a GPU: everything the audit builds (model, V,
    # verifier boxes) follows torch's default device, so set it once here. Local
    # smoke runs on CPU are unaffected.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_default_device("cuda")
    cfg = cfg or RSSMConfig()

    log = {}
    log["config"] = dict(seed=seed, n_samples=n_samples, v_steps=v_steps,
                         v_batch=v_batch, quantile=quantile, hole=hole,
                         hole_k=hole_k, time_budget=time_budget, method=method,
                         min_width=min_width, n_episodes=n_episodes,
                         horizon=horizon, burn_in=burn_in, synthetic=synthetic,
                         device=device,
                         model="DreamerV3 one-step latent "
                         "transition, size1m (deter 512, stoch 32x4)",
                         scope=("one-step latent dynamics under the frozen actor; "
                                "NOT the imagined rollout"),
                         upstream_dreamerv3=UPSTREAM_DREAMERV3,
                         upstream_embodied=UPSTREAM_EMBODIED)
    log["verifier"] = provenance()

    # ------------------------------------------------------------- frozen model
    if synthetic:
        print("synthetic model + synthetic closed-loop support (smoke only; "
              "never quote these numbers)")
        trans, pol, loop = build_frozen(cfg, action_dim)
    else:
        assert weights_npz and support_npz, "real run needs --weights and --support"
        trans, pol, loop = build_frozen(cfg, action_dim, weights_npz)

    # ------------------------------------------------------------ latent support
    if synthetic:
        # The model's OWN closed-loop behaviour: iterate T(z, pi(z)) from random
        # starts. A legitimate stand-in for the on-policy latent distribution only
        # because the smoke run must exercise the same plumbing as the real one.
        rng = np.random.default_rng(seed)
        N = n_episodes * horizon
        deter = np.zeros((N, cfg.deter), np.float32)
        stoch = np.zeros((N, cfg.stoch, cfg.classes), np.float32)
        k = 0
        for ep in range(n_episodes):
            d = torch.randn(1, cfg.deter) * 0.05
            s = torch.softmax(torch.randn(1, cfg.stoch, cfg.classes) * 0.5, -1)
            with torch.no_grad():
                for t in range(horizon):
                    deter[k] = d.detach().cpu().numpy()
                    stoch[k] = s.detach().cpu().numpy()
                    z = loop(torch.cat([d, torch.flatten(s, start_dim=-2)], -1))
                    d = z[..., :cfg.deter]
                    s = z[..., cfg.deter:].reshape(1, cfg.stoch, cfg.classes)
                    k += 1
        support_deter, support_stoch = deter, stoch
    else:
        data = np.load(support_npz)
        support_deter = np.asarray(data["deter"], np.float32)
        support_stoch = np.asarray(data["stoch"], np.float32)
        log["support_meta"] = {k: (int(v) if np.isscalar(v) else v)
                               for k, v in data.items() if k not in ("deter", "stoch")}
        # The region's steady-state slice uses n_episodes; take it from the
        # support the run was actually given, not the defaults.
        if "n_episodes" in data:
            n_episodes = int(data["n_episodes"])
        if "horizon" in data:
            horizon = int(data["horizon"])
        log["config"]["n_episodes"] = n_episodes
        log["config"]["horizon"] = horizon

    # -------------------------------------------------- latent fixed point z*
    eq = find_latent_fixed_point(loop, cfg, support_deter, support_stoch, seed=seed)
    z_star = eq["z_star"]
    log["equilibrium"] = dict(z_star=z_star.tolist(), residual=eq["residual"],
                              converged=eq["converged"],
                              note=("z* is the one-step closed loop's fixed point. "
                                    "A V centred anywhere else has cond(centre) < 0 "
                                    "by construction."))
    print(f"equilibrium: residual {eq['residual']:.2e}, converged {eq['converged']}")

    # -------------------------------------------------------------- region
    zs = np.concatenate([support_deter, support_stoch.reshape(len(support_deter), -1)],
                        axis=1)
    z_steady = zs[burn_in * n_episodes:] if not synthetic else zs
    if synthetic:
        z_steady = zs
    dev = z_steady - z_star
    half = np.maximum(np.quantile(np.abs(dev), quantile, axis=0), 1e-6)
    lo, hi = z_star - half, z_star + half
    # The hole only needs to EXCLUDE z* from every certified box (cond(z*) = 0,
    # so a box containing it has true infimum 0 and cannot certify at a positive
    # margin). A full-dim hole would slab-decompose 640 dims into 2*640 boxes;
    # excluding the k dims of LARGEST halfwidth suffices and yields 2*k boxes.
    hole_dims = np.argsort(half)[-hole_k:] if hole_k < len(half) else np.arange(len(half))
    hole_half = hole * half
    hole_lo, hole_hi = lo.copy(), hi.copy()
    hole_lo[hole_dims] = z_star[hole_dims] - hole_half[hole_dims]
    hole_hi[hole_dims] = z_star[hole_dims] + hole_half[hole_dims]
    boxes = R.annulus_boxes(lo, hi, hole_lo, hole_hi)

    log["region"] = dict(
        box_vs_support=dict(n_states=len(z_steady),
                            frac_inside=float(np.mean(
                                np.all((z_steady >= lo) & (z_steady <= hi), axis=1))),
                            quantile=quantile,
                            why=("Region built FROM the model's own on-policy "
                                 "latent states, steady-state portion (burn_in).")),
        hole=dict(frac_of_halfwidth=hole, k_dims=hole_k,
                  dims=hole_dims.tolist(),
                  why=("cond(z*) = 0 by construction, so any box containing z* has "
                       "true infimum 0 and cannot certify at a positive margin. "
                       "The hole excludes only the k largest-halfwidth dims; that "
                       "suffices to remove z* from every certified box while keeping "
                       "the slab decomposition to 2*k boxes instead of 2*640.")),
        n_annulus_boxes=len(boxes),
    )
    print(f"region: {len(boxes)} annulus box(es), "
          f"holds {log['region']['box_vs_support']['frac_inside']:.3%} "
          f"of steady-state latents")

    # ------------------------------------------------------------- fit V
    print("training V on the latent ...")
    V = train_latent_V(loop, zs, lo, hi, hole_lo, hole_hi, z_star,
                       steps=v_steps, batch=v_batch, seed=seed)
    cond = DecreaseCondition(V, loop).eval()
    log["lyapunov"] = dict(
        form="||h(z)-h(z*)||^2 + c||z-z*||^2  (z = concat(deter, stoch_flat))",
        c=0.5, hidden=64, centre=z_star.tolist(), steps=v_steps,
        surrogate=("stoch is the mean of the prior categorical; the certified map "
                   "is the deterministic one-step dynamics under the deterministic "
                   "actor tanh(mean)."))

    # --------------------------------------------------- sampling-only audit
    print(f"sampling audit ({n_samples} latent states) ...")
    samp = sampled_latent_violation_rate(cond, lo, hi, hole_lo, hole_hi, zs,
                                         n=n_samples, seed=seed + 999)
    log["sampling"] = samp
    print(f"sampling: {samp['n_violations']}/{samp['n_sampled']} violations "
          f"({samp['violation_rate']:.3%}), worst cond {samp['worst_cond']:.3e}")

    # ----------------------------------------------------------- BaB audit
    print(f"branch-and-bound over {len(boxes)} annulus box(es) ...")
    box_results, ces = [], []
    # STRUCTURAL WALL PROBE first. The certified graph contains the stoch'
    # output, stoch_mean = exp(logits) / sum(exp): auto_LiRPA's interval pass
    # asserts in BoundReciprocal whenever the relaxed logits interval spans
    # exp -> 0 -- which the measured vacuity (d1_smoke) makes unavoidable on any
    # box. Per-box certify_box would each spend a ~4 min BoundedModule build to
    # fail identically. One probe on the first box establishes the wall; every
    # box then inherits unknown with the recorded reason. If the probe SURVIVES
    # (finite bounds), the normal per-box loop runs and per-box verdicts stand.
    wall = None
    if boxes:
        xL0 = torch.tensor(boxes[0][0], dtype=torch.float32).reshape(1, -1)
        xU0 = torch.tensor(boxes[0][1], dtype=torch.float32).reshape(1, -1)
        try:
            from auto_LiRPA import BoundedModule, BoundedTensor
            from auto_LiRPA.perturbations import PerturbationLpNorm
            bm = BoundedModule(cond, (xL0 + xU0) / 2, verbose=False)
            ptb = PerturbationLpNorm(norm=float("inf"), x_L=xL0, x_U=xU0)
            bx = BoundedTensor((xL0 + xU0) / 2, ptb)
            lb, _ = bm.compute_bounds(x=(bx,), method="ibp")
            lb = float(lb.min().item())
            if math.isnan(lb) or math.isinf(lb):
                wall = ("non-finite interval lower bound "
                        f"({lb}): categorical-output relaxation overflow, "
                        "category 3")
            else:
                lb, _ = bm.compute_bounds(x=(bx,), method=method)
                lb = float(lb.min().item())
                if math.isnan(lb) or math.isinf(lb):
                    wall = (f"non-finite {method} lower bound ({lb}): "
                            "categorical-output relaxation overflow, category 3")
        except Exception as e:                       # noqa: BLE001 - the wall
            wall = (f"verifier cannot process the certified graph: "
                    f"{type(e).__name__}: {e} (categorical-output wall, "
                    "category 3)")
    if wall:
        print(f"  WALL: {wall}")
        for i, (blo, bhi) in enumerate(boxes):
            box_results.append(dict(verdict="unknown", certified=False,
                                    reason=f"categorical-output wall (see bab.wall): "
                                           f"{wall}",
                                    subdomains=None, seconds=None,
                                    worst_lower_bound=None, method=method,
                                    box=dict(lo=blo.tolist(), hi=bhi.tolist())))
    else:
        for i, (blo, bhi) in enumerate(boxes):
            xL = torch.tensor(blo, dtype=torch.float32).reshape(1, -1)
            xU = torch.tensor(bhi, dtype=torch.float32).reshape(1, -1)
            try:
                r = certify_box(cond, xL, xU, eps=0.0, method=method,
                                min_width=min_width, time_budget=time_budget,
                                seed=seed)
                wlb = r.get("worst_lower_bound")
                if wlb is not None:
                    try:
                        wlb = float(wlb)
                    except (TypeError, ValueError):
                        wlb = None
                if wlb is not None and math.isnan(wlb):
                    r = dict(verdict="unknown", certified=False,
                             reason="relaxation_overflow: stoch' = softmax(prior) "
                                    "exp relaxation non-finite (category 3)",
                             subdomains=r.get("subdomains"),
                             seconds=r.get("seconds"), worst_lower_bound=wlb,
                             method=method)
            except Exception as e:              # noqa: BLE001 - record, never crash
                r = dict(verdict="unknown", certified=False,
                         reason=f"verifier_exception: {type(e).__name__}: {e}",
                         subdomains=None, seconds=None, worst_lower_bound=None,
                         method=method)
            r["box"] = dict(lo=blo.tolist(), hi=bhi.tolist())
            box_results.append(r)
            print(f"  box {i}: {r['verdict']:>9}  "
                  f"{r.get('subdomains', '?')} subdomains  {r.get('seconds', '?')}s"
                  + (f"  [{r['reason']}]" if r.get("reason") else ""))
            if r["verdict"] == "violation":
                ces.append(r["counterexample"])
    log["bab"] = dict(boxes=box_results,
                      wall=wall,
                      n_violation=sum(1 for r in box_results if r["verdict"] == "violation"),
                      n_unknown=sum(1 for r in box_results if r["verdict"] == "unknown"),
                      n_certified=sum(1 for r in box_results if r.get("certified")),
                      note=("certify_box is reused verbatim from cnl-work. The "
                            "certified graph contains stoch' = stoch_mean(prior) "
                            "whose exp/sum-div is not interval-bounded at the "
                            "measured vacuity; the structural probe records that "
                            "wall once (category 3) and every box is unknown -- "
                            "never certified, never a violation, never a finding. "
                            "If the probe survives, per-box branch-and-bound runs "
                            "with per-box verdicts."))

    # ------------------------------------------------- reachability gate (latent)
    scale = np.maximum(np.ptp(zs, axis=0), 1e-9)
    gated = []
    if ces:
        arr = np.array(ces, dtype=np.float64).reshape(len(ces), -1)
        with torch.no_grad():
            cvals = cond(torch.tensor(arr, dtype=torch.float32)).squeeze(-1).cpu().numpy()
        for s, cv in zip(arr.tolist(), cvals.tolist()):
            v, d = latent_reach_verdict(s, support_deter, support_stoch, scale)
            gated.append(dict(state=s, cond=float(cv), reach_verdict=v,
                              normalized_distance_to_support=float(d)))
            print(f"  CE cond={cv:.3e} -> {v} (d={d:.4f})")
    log["counterexamples"] = gated

    n_real = sum(1 for g in gated if g["reach_verdict"] == IN_SUPPORT)
    log["result"] = dict(
        sampling_violations=samp["n_violations"],
        sampling_rate=samp["violation_rate"],
        bab_counterexamples=len(gated),
        reachable_counterexamples=n_real,
        gap_demonstrated=bool(samp["n_violations"] == 0 and n_real > 0),
        interpretation=(
            "gap_demonstrated is True only when the sampling audit found NOTHING "
            "and branch-and-bound found at least one violation at a latent state "
            "the model demonstrably reaches. Unknown verdicts are verifier "
            "incompleteness (category 3), never a gap and never evidence of "
            "safety. The certified claim is about the one-step latent dynamics "
            "under the deterministic actor, NOT the imagined rollout."),
        seconds=round(time.time() - t_start, 1),
    )

    out = out or os.path.join(ROOT, "results", f"d1_seed{seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nwrote {out}")
    print(json.dumps(log["result"], indent=2))
    return log


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None, help="npz of JAX params under upstream key names")
    p.add_argument("--support", default=None, help="npz of on-policy latent states")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-dim", type=int, default=6,
                   help="action dim of the trained agent (dmc cartpole: 1)")
    p.add_argument("--n-samples", type=int, default=500_000)
    p.add_argument("--v-steps", type=int, default=4000)
    p.add_argument("--v-batch", type=int, default=4096)
    p.add_argument("--hole-k", type=int, default=4,
                   help="hole dims for the annulus (2*hole_k boxes; full-dim is 1280)")
    p.add_argument("--burn-in", type=int, default=100,
                   help="episodes to drop from the support before building the region")
    p.add_argument("--time-budget", type=float, default=120.0)
    p.add_argument("--method", default="CROWN",
                   help="plain CROWN by default; same reasoning as A1 (E10 measured "
                        "CROWN-Optimized as WORSE on product nodes).")
    p.add_argument("--synthetic", action="store_true",
                   help="random weights + synthetic closed-loop support; smoke only")
    p.add_argument("--quick", action="store_true",
                   help="smoke-test sizes, for local runs only; never quote these numbers")
    a = p.parse_args()
    if a.quick:
        # CPU-sized smoke: sizes chosen so the whole pipeline (incl. the ~4 min
        # BoundedModule build for the wall probe) fits a local run. The probe's
        # build dominates; the rest is deliberately small.
        run(seed=a.seed, n_samples=20_000, v_steps=20, v_batch=256,
            time_budget=10.0, hole_k=2, n_episodes=60, horizon=30,
            burn_in=15, synthetic=True,
            out=os.path.join(ROOT, "results", f"d1_quick_seed{a.seed}.json"))
    else:
        run(weights_npz=a.weights, support_npz=a.support, seed=a.seed,
            action_dim=a.action_dim, n_samples=a.n_samples, v_steps=a.v_steps,
            v_batch=a.v_batch, hole_k=a.hole_k, burn_in=a.burn_in,
            time_budget=a.time_budget, method=a.method, synthetic=a.synthetic)
