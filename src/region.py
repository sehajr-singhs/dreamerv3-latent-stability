"""Verification regions, always quoted against the on-policy distribution.

Guardrail 3 requires every box to be reported relative to where the policy actually
lives, so regions are constructed FROM the empirical support rather than picked by
hand. `describe` returns the provenance that has to travel with any result.

The annulus exists because of the structural equality at the equilibrium documented in
`lyapunov.DecreaseCondition`: cond(x*) = 0 exactly, so a box containing x* has true
infimum 0 and cannot certify at a positive margin however good the verifier is. Cutting
a hole is standard practice for Lyapunov certification and is NOT a way of hiding
violations, provided the hole is small, stated, and reported. It is reported here.
"""

import numpy as np


def annulus_boxes(lo, hi, hole_lo, hole_hi):
    """Decompose the box [lo, hi] minus the open box (hole_lo, hole_hi) into disjoint boxes.

    Standard slab decomposition: peel the hole off one axis at a time, so the pieces
    are disjoint and their union is exactly the set difference. Returns [(lo, hi), ...].
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    hole_lo = np.maximum(np.asarray(hole_lo, dtype=np.float64), lo)
    hole_hi = np.minimum(np.asarray(hole_hi, dtype=np.float64), hi)

    if np.any(hole_lo >= hole_hi):
        return [(lo.copy(), hi.copy())]          # hole misses the box entirely

    boxes = []
    cur_lo, cur_hi = lo.copy(), hi.copy()
    for d in range(len(lo)):
        if hole_lo[d] > cur_lo[d]:               # slab below the hole on axis d
            b_hi = cur_hi.copy()
            b_hi[d] = hole_lo[d]
            boxes.append((cur_lo.copy(), b_hi))
        if hole_hi[d] < cur_hi[d]:               # slab above the hole on axis d
            b_lo = cur_lo.copy()
            b_lo[d] = hole_hi[d]
            boxes.append((b_lo, cur_hi.copy()))
        # remaining region is clamped to the hole's extent on this axis
        cur_lo[d] = max(cur_lo[d], hole_lo[d])
        cur_hi[d] = min(cur_hi[d], hole_hi[d])
    return boxes


def describe(lo, hi, support):
    """Provenance for a box: what fraction of on-policy states it actually contains.

    A box holding 99% of visited states is a claim about the policy's real behaviour.
    A box holding 0.1% is a claim about almost nothing, however impressive the verdict
    attached to it. Reporting this is what stops the second from masquerading as the first.
    """
    s = support.states
    inside = np.all((s >= np.asarray(lo)) & (s <= np.asarray(hi)), axis=1)
    return dict(
        lo=[float(v) for v in lo],
        hi=[float(v) for v in hi],
        frac_on_policy_states_inside=float(inside.mean()),
        n_on_policy_states_inside=int(inside.sum()),
    )


def coverage_of(boxes, support):
    """Fraction of on-policy states covered by a set of boxes (the annulus, typically)."""
    s = support.states
    hit = np.zeros(len(s), dtype=bool)
    for lo, hi in boxes:
        hit |= np.all((s >= np.asarray(lo)) & (s <= np.asarray(hi)), axis=1)
    return float(hit.mean())
