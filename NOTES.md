# NOTES

Running record of what was measured and what it means. This file is the authority when
it disagrees with a summary elsewhere. Numbers here are JSON-backed; anything not backed
by a file in `results/` is marked as such.

---

## The distinctions that must not be blurred

Inherited from cnl-work E10. All of these read as "did not certify", and **only the
first is a defect in V**:

1. **genuine violation** — a real state where the condition fails
2. **structural equality** — `cond(s*) = 0` exactly, because `V(s*) = 0` and
   `f_cl(s*) = s*`. Every sound lower bound is `<= 0` on a box containing `s*`. Shows up
   as a "violation" verdict at machine-zero magnitude. Measured here: `-1.29e-14`.
3. **verifier incompleteness** — `unknown`: the bound stayed loose and no counterexample
   was found. Never evidence of safety.

This project adds two more, and both bit before any verifier ran:

4. **physical impossibility** — the region exceeds what the actuator can hold, so *no*
   controller admits a monotone Lyapunov function there.
5. **centring error** — V built about a point that is not the closed loop's fixed point.
   Guarantees `cond(centre) < 0` regardless of training or verifier. Not a finding.

---

## Correctness gates (`experiments/smoke_crown.py`), all passing

| gate | result |
|---|---|
| dynamics constants vs gymnasium 1.3.0 | all 6 match |
| `PendulumStep` vs `env.step`, 500 random states | max err **1.11e-16** |
| extracted actor vs `sb3.predict`, 256 obs | max err **1.19e-07** |
| CROWN bounds closed loop, r = 0.05 / 0.2 / 0.5 | sound; looseness **1.0 / 1.0 / 1.2x** |

The fourth gate was the project's main technical risk. `sin`/`cos` + `clamp` + `tanh`
bound essentially tightly, so branch-and-bound has room to work. No custom bound needed.

Float-precision trap worth not re-deriving: gymnasium's `env.step` takes a float32
action, and under NEP 50 numpy 2 keeps `3.0/(m*l^2) * u` in float32 while the `sin(th)`
term stays float64, so the env mixes precisions internally. Passing a float64 action
takes the agreement from 1.2e-08 to 1.1e-16.

---

## The frozen policy

SAC, `net_arch=[64, 64]`, `use_sde=False`, 20k steps, seed 0.
Return **-127.4 +/- 66.5** over 30 deterministic episodes (solved is conventionally near
-200). Small on purpose: narrow nets branch-and-bound far better, and gSDE's
deterministic action is not a plain `tanh(mu)`, so it would put an untraceable object in
the bounded graph. Never retrained; only V is fit.

---

## The policy does not hold upright (`a0d`, `a0c`)

**This was the root cause of every narrow-region failure.**

At upright the policy commands `u = +0.499954`, where holding requires exactly `0`, and
the state drifts `7.5e-02` per step. Upright is **not a fixed point of the closed loop**.

The closed loop's actual attracting fixed point is

    s* = (theta, thetadot) = (0.14258597, 0)      Newton residual 6.66e-17

an **8.17 degree steady-state offset**. There is a second fixed point at
`theta = 0.411218`, sitting at the torque-saturation boundary (`u = -1.9986`), which is
not the one to certify around. `find_attracting_fixed_point` therefore settles first and
polishes with Newton; Newton from upright lands on the wrong one.

Why centring is fatal rather than cosmetic. V is built so `V(s*) = 0` and `V > 0`
elsewhere. If `f_cl(centre) != centre` then `V(f_cl(centre)) > 0`, so

    cond(centre) = V(centre) - V(f_cl(centre)) = -V(f_cl(centre)) < 0

is a violation **at the centre**, guaranteed by construction, independent of training and
of the verifier. Measured directly: an upright-centred V evaluates to
`cond = -6.78e-03` at its own centre, which is exactly the band of the unexplained
violations the sweeps had been reporting (`-3.6e-03` to `-7.2e-03`). Correctly centred,
`cond(s*) = -1.29e-14`, i.e. machine zero, which is category 2 and not a defect.

cnl-work's `LyapunovPDOnState` already had this right and documents the same requirement
for its own off-origin equilibrium. This was a porting error, not a new phenomenon.

### Linearization at s* (`results/a0c_linearization_seed0.json`)

    A = [[ 0.787148,  0.009287],
         [-4.257041,  0.185730]]
    eigenvalues 0.712033, 0.260845      rho(A) = 0.712033

Comfortably Schur stable, so the largest feasible relative margin is
`1 - rho^2 = 0.493`, i.e. **49.3% decay per step**. The `rel_margin = 0.01` used in
training is far inside that, so the margin was never the binding constraint.

An earlier version of this file reported `rho = 0.9916` and a 1.68% ceiling. That was
computed at `(0,0)`, which is not a fixed point, so it was a valid Jacobian of nothing in
particular and its stability reading was meaningless. Superseded.

The discrete Lyapunov solve `A' P A - P = -I` gives a training-free quadratic certificate
for the linearization, `P = [[59.75, -0.431], [-0.431, 1.040]]`, residual 7.1e-15. Held
in reserve as a confound-free alternative to a learned V.

---

## Actuation limit (`a0b`) — corrected

gymnasium integrates `thddot = (3g/2l) sin(th) + (3/m l^2) u`, which is a uniform **rod
pivoted at one end** (`I = m l^2 / 3`, gravity torque `m g (l/2) sin th`), **not** a point
mass at radius `l`. Holding torque is therefore

    u = -(m g l / 2) sin(th) = -5 sin(th)        ->    |sin th| <= 0.4

    STATIC HOLD LIMIT   |theta| <= 0.411517 rad = 23.58 deg

An earlier version used the point-mass balance `u = m g l sin th` and reported
**0.2014 rad**, a factor of two too tight. Superseded. Two independent confirmations of
the corrected number:
- the policy's own action at `s*` is `-0.710516` against `-5 sin(0.142586) = -0.710517`
- the closed loop's second fixed point sits at `0.411218` against the predicted boundary
  `0.411517`, a difference of **3e-04 rad**

Past that band the actuator cannot balance gravity, so no controller admits a monotone V.
Necessary, not sufficient: the true basin is smaller once kinetic energy counts.

---

## Region: squeezed from both ends

**Swing-up.** The policy must pump energy in before it can stabilize, so V necessarily
*increases* during the transient and no monotone V exists over the full on-policy
support. The region is built from the steady-state portion of the rollouts (steps
`>= burn_in`); the reachability gate keeps using the **full** visited set, because a
transient state is still a state the policy reaches. Both coverages are reported.

**Degeneracy.** Push `burn_in` high and the region collapses onto `s*`, where `cond -> 0`
by construction and the condition stops being informative.

---

## Training: the margin must be relative

The first probe swept the **training budget** and found the sampled violation rate
plateauing near 20% regardless (41.9% @ 500 steps, 23.5% @ 2000, 26.1% @ 6000,
20.5% @ 15000). Budget was the wrong knob.

The hinge used an **absolute** margin, `cond >= 1e-3`. Near `s*`,
`cond ~ (1 - rho) V -> 0`, so that target is unachievable by construction. The margin is
now **relative**, `V(f) <= (1 - m) V`, a geometric decay rate: scale-free and meaningful
at every distance from `s*`. This measurably helped the wide regions (20%+ down to ~7%).

It did **not** fix the narrow regions, and the hypothesis that it would was wrong; that
was the centring error above. Both fixes are correct, but only the second was the cause.

Same class of bug: the annulus hole was a fixed `+/-0.05` against a `+/-0.143` region,
removing most of the volume being certified. Now a fraction of the region.

---

## Non-obvious gotchas

- `torch`'s `.double()` **mutates in place**. Calling it on a shared closed loop left the
  policy in float64 and broke every float32 consumer downstream with a confusing
  `mat1 and mat2 must have the same dtype`. `src/equilibrium.py` now deep-copies first.
- `torch.as_tensor` does not accept `requires_grad`; `torch.tensor` does.
- Piping `python -u` through `grep` re-buffers, so interim output still does not stream.
  Redirect to a file and grep the file instead.

---

## Status

- Gates **pass**. Policy **trained**. Pipeline **runs end to end**.
- Gap: **not claimed.** Pending the re-centred A0 sweep. Where sampling cannot be made
  clean, the honest result is the obstruction, not a gap.
- dReal confirmation harness **not written**. Nothing may be called dReal-confirmed.
- DreamerV3 **not started**. Scope stays the one-step latent transition, smallest model.
