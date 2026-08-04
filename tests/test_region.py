"""Annulus decomposition must be exact: disjoint pieces whose union is box-minus-hole.

A gap would silently exclude states from certification (claiming coverage we do not
have); an overlap would double-count. Both are the kind of quiet error that makes a
headline number wrong, so this is checked by Monte Carlo against the set definition.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.region import annulus_boxes


def _check(lo, hi, hlo, hhi, n=200_000, seed=0):
    lo, hi = np.array(lo, float), np.array(hi, float)
    hlo, hhi = np.array(hlo, float), np.array(hhi, float)
    boxes = annulus_boxes(lo, hi, hlo, hhi)

    rng = np.random.default_rng(seed)
    p = rng.uniform(lo, hi, size=(n, len(lo)))

    in_hole = np.all((p > hlo) & (p < hhi), axis=1)
    counts = np.zeros(n, dtype=int)
    for blo, bhi in boxes:
        counts += np.all((p >= blo) & (p <= bhi), axis=1)

    # union == box minus hole
    assert not np.any(counts[in_hole] > 0), "hole leaked into the decomposition"
    assert np.all(counts[~in_hole] >= 1), "gap: annulus point covered by no box"
    # disjoint (boundary ties are measure zero but can register twice; allow a trickle)
    dup = float((counts > 1).mean())
    assert dup < 1e-3, f"boxes overlap on {dup:.4%} of samples"

    vol_boxes = sum(float(np.prod(bhi - blo)) for blo, bhi in boxes)
    vol_exact = float(np.prod(hi - lo) - np.prod(np.maximum(hhi - hlo, 0)))
    assert abs(vol_boxes - vol_exact) < 1e-9 * max(1.0, abs(vol_exact)), \
        f"volume mismatch: {vol_boxes} vs {vol_exact}"
    return len(boxes), vol_boxes, vol_exact


def test_centered_hole_2d():
    n, vb, ve = _check([-np.pi, -8], [np.pi, 8], [-0.1, -0.1], [0.1, 0.1])
    print(f"[PASS] centered 2-D hole: {n} boxes, volume {vb:.6f} == {ve:.6f}")


def test_offcenter_hole():
    n, vb, ve = _check([-1, -2], [3, 5], [0.5, 1.0], [1.5, 2.0])
    print(f"[PASS] off-centre hole: {n} boxes, volume {vb:.6f} == {ve:.6f}")


def test_hole_touching_edge():
    n, vb, ve = _check([-1, -1], [1, 1], [-1.0, -0.2], [-0.4, 0.2])
    print(f"[PASS] hole touching an edge: {n} boxes, volume {vb:.6f} == {ve:.6f}")


def test_hole_outside_is_noop():
    boxes = annulus_boxes([0, 0], [1, 1], [5, 5], [6, 6])
    assert len(boxes) == 1
    print("[PASS] hole disjoint from box leaves the box intact")


def test_3d():
    n, vb, ve = _check([-1, -1, -1], [1, 1, 1], [-0.3, -0.2, -0.1], [0.3, 0.2, 0.1])
    print(f"[PASS] 3-D hole: {n} boxes, volume {vb:.6f} == {ve:.6f}")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
            except AssertionError as e:
                fails += 1
                print(f"[FAIL] {name}: {e}")
    sys.exit(1 if fails else 0)
