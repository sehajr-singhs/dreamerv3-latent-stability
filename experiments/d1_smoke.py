"""Local smoke gate for the DreamerV3 one-step latent transition port.

Two questions, mirroring smoke_crown.py's gate 4 for the Pendulum closed loop:

  1. Can auto_LiRPA trace and CROWN-bound the full one-step graph at all? The graph
     is silu (as x/(1+exp(-x))), RMSNorm (sqrt), block-linear, and the GRU's
     elementwise products — every piece probed boundable in d1_probe_boundable.py.
  2. Are the bounds SOUND and how loose are they? The GRU's reset*cand and
     update*cand products are where CROWN is known to go loose (the E10 note in
     CLAUDE.md). The gate FAILS on unsoundness or non-finiteness, and REPORTS
     looseness: a loose-but-sound bound is a scaling question for branch-and-bound,
     not a broken gate.

Local only, smoke sizes, never quoted in an artifact. The real D1 audit runs on
Colab against a trained model.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from src.dreamer import OneStepLatentTransition, RSSMConfig, stoch_mean

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


class _TransitionSlice(nn.Module):
    """One-step transition exposing a small fixed slice of the core outputs so the
    BoundedModule has a single tensor output and CROWN stays cheap.

    Deliberately bounds deter' and the prior logits, NOT the softmax-mean
    surrogate stoch'. The surrogate's exp is only defined for bounded logits; at
    pathological widths its relaxation overflows (the documented scaling
    boundary), and it is a deterministic function of logits anyway. The claim
    object of this leg is the transition (deter', logits'); that is what this gate
    proves boundable."""

    def __init__(self, trans, k=4):
        super().__init__()
        self.trans = trans
        self.k = k

    def forward(self, deter, stoch, action):
        d, l, s = self.trans(deter, stoch, action)
        lf = torch.flatten(l, start_dim=-2)      # (B, stoch*classes)
        return torch.cat([d[..., :self.k], lf[..., :self.k]], -1)


def g_boundable():
    torch.manual_seed(0)
    cfg = RSSMConfig()
    trans = OneStepLatentTransition(cfg, action_dim=6).eval()
    net = _TransitionSlice(trans).eval()

    B = 1
    deter = torch.randn(B, cfg.deter) * 0.1
    stoch = torch.full((B, cfg.stoch, cfg.classes), 1.0 / cfg.classes)
    action = torch.rand(B, 6) * 2 - 1
    center = (deter, stoch, action)

    bm = BoundedModule(net, center, verbose=False)

    out = []
    for r in (0.02, 0.1):
        ptbs = tuple(PerturbationLpNorm(norm=np.inf, eps=r) for _ in range(3))
        bt = tuple(BoundedTensor(c.clone(), p) for c, p in zip(center, ptbs))
        lb, ub = bm.compute_bounds(x=bt, method="CROWN")
        lb, ub = lb.detach(), ub.detach()
        if not (torch.isfinite(lb).all() and torch.isfinite(ub).all()):
            raise AssertionError(f"non-finite bounds at r={r}")

        rng = np.random.default_rng(0)
        sd = torch.tensor(rng.uniform(deter.numpy() - r, deter.numpy() + r,
                                      size=(2000, cfg.deter)), dtype=torch.float32)
        ss = torch.tensor(rng.uniform(stoch.numpy() - r, stoch.numpy() + r,
                                      size=(2000, cfg.stoch, cfg.classes)),
                          dtype=torch.float32).clamp(0, 1)
        sa = torch.tensor(rng.uniform(action.numpy() - r, action.numpy() + r,
                                      size=(2000, 6)), dtype=torch.float32).clamp(-1, 1)
        with torch.no_grad():
            f = net(sd, ss, sa)
        viol = float(torch.maximum(lb - f, f - ub).max())
        if viol > 1e-4:
            raise AssertionError(f"UNSOUND at r={r}: sample outside bounds by {viol:.3e}")
        width = float((ub - lb).max())
        span = float((f.max(0).values - f.min(0).values).max())
        out.append(f"r={r}: sound (viol {viol:.1e}), width/span={width / max(span, 1e-9):.1f}x")
    return ("sound at all radii; " + "; ".join(out) +
            " -- WIDE: single norm stages are ~1e2x loose and the full chain "
            "compounds to ~1e6x (RMSNorm's 1/sqrt(mean2) with mean2 lower bound 0 "
            "amplifies; GRU products compound it). Sound but vacuous at these radii: "
            "the predicted GRU/norm scaling boundary, and the reason the audit must "
            "branch-and-bound tiny boxes rather than bound one big region.")


def g_surrogate_is_distribution_mean():
    """The stoch' surrogate must be a probability vector over classes, exactly the
    unimix mixture the model samples from, and its output must stay in [0, 1]."""
    torch.manual_seed(1)
    cfg = RSSMConfig()
    trans = OneStepLatentTransition(cfg, action_dim=6).eval()
    deter = torch.randn(2, cfg.deter) * 0.1
    stoch = torch.full((2, cfg.stoch, cfg.classes), 1.0 / cfg.classes)
    action = torch.zeros(2, 6)
    with torch.no_grad():
        _, logits, s = trans(deter, stoch, action)
    assert torch.allclose(s.sum(-1), torch.ones_like(s.sum(-1)), atol=1e-5)
    assert float(s.min()) >= -1e-6 and float(s.max()) <= 1 + 1e-6
    assert torch.allclose(s, stoch_mean(logits, cfg.unimix), atol=1e-6)
    return "stoch' is a valid probability tensor equal to the unimix mean"


if __name__ == "__main__":
    gate("CROWN bounds the one-step transition", g_boundable)
    gate("stoch' surrogate is the distribution mean", g_surrogate_is_distribution_mean)

    n_fail = sum(1 for r in results if r[0] == FAIL)
    print("\n" + "=" * 60)
    print(f"{len(results) - n_fail}/{len(results)} gates passed")
    sys.exit(1 if n_fail else 0)
