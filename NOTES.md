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

## A0 re-centred sweep (`results/a0_feasibility_seed0.json`), run on Colab

The sweep the whole pipeline was waiting on. Run on Colab Linux, `v_steps=2000`,
`n_episodes=1000`, `n_samples=100000`, seed 0.

**Centring ported.** Colab found `s* = (0.14258597, 2.42e-18)`, matching the local value
to eight decimals, so the fix travels and is not an artifact of one machine.

### Six of the eight rows are physically inadmissible

Every region has to be checked against the `|theta| <= 0.411517` actuation limit *before*
its violation rate means anything. Most of them fail:

| burn_in | c | theta region | inside limit? | rate | n_viol / n_samp | worst cond |
|---|---|---|---|---|---|---|
| 10 | 0.05 | [-2.516, 2.801] | no, 6x over | 7.377% | 4172 / 56553 | -2.23e-01 |
| 10 | 0.5 | [-2.516, 2.801] | no | 6.723% | 3802 / 56553 | -2.09e+00 |
| 25 | 0.05 | [-1.722, 2.007] | no, 5x over | 5.540% | 3128 / 56458 | -1.80e-01 |
| 25 | 0.5 | [-1.722, 2.007] | no | 5.287% | 2985 / 56458 | -1.86e+00 |
| 50 | 0.05 | [-0.131, 0.4166] | no, edge over | 1.623% | 994 / 61261 | -2.21e-04 |
| 50 | 0.5 | [-0.131, 0.4166] | no | 1.882% | 1153 / 61261 | -1.71e-03 |
| **100** | **0.05** | **[0.0248, 0.2604]** | **yes** | **0.177%** | **97 / 54868** | **-3.93e-06** |
| **100** | **0.5** | **[0.0248, 0.2604]** | **yes** | **0.159%** | **87 / 54868** | **-3.69e-05** |

The `burn_in=50` rows are the ones worth not waving through. They exceed the limit by only
`5e-03` rad, but that is enough to put the **second fixed point at `0.411218` inside the
box**. A region containing the saturation equilibrium is category 4, not evidence about V.

Their rates are quoted above for transparency and are **not** results. Reading the 7.377%
row as "V is bad" would be reading physics as a training failure.

### The admissible setting is clean, above a training-budget threshold

At `v_steps=2000` the admissible row showed 87 violations in 54,868 samples (0.159%), and
this file previously concluded from that single budget that the policy admits no clean
certificate here, calling it an obstruction result. **That conclusion was wrong and is
retracted.** It was drawn from one point on an axis that had not been swept.

`experiments/a0_v_feasibility.py --burn-ins 100 --cs 0.05 0.5`, varying only `v_steps`
(`results/a0_vsteps{2000,4000,8000,16000,32000}_seed0.json`):

| v_steps | rate c=0.05 | rate c=0.5 | worst cond, c=0.5 | theta region |
|---|---|---|---|---|
| 2000 | 0.177% | 0.159% | `-3.685e-05` | [0.0248, 0.2604] |
| 4000 | 0.049% | 0.033% | `-7.891e-06` | [0.0248, 0.2604] |
| **8000** | **0** | **0** | **`+6.615e-06`** | [0.0248, 0.2604] |
| 16000 | 0 | 0 | `+6.729e-06` | [0.0248, 0.2604] |
| 32000 | 0 | 0 | `+6.725e-06` | [0.0248, 0.2604] |

**Control held.** One distinct theta region across all ten rows, `cov_full = 0.6834`
throughout, `s*` identical, `hidden=32` and `hole_frac=0.10` fixed. Only the optimizer
budget moved, so the comparison is valid and nothing was widened or reshaped.

**The signal is the sign flip, not the zero.** `worst_cond` crosses from `-3.685e-05` to
`-7.891e-06` to `+6.615e-06`, then sits stable at `+6.6e-06` through 16000 and 32000.
Positive means the *minimum* sampled `cond` clears zero with margin: every sampled point
strictly satisfies the decrease condition, rather than the violating set merely shrinking.
Stability across three budgets means it is not one lucky initialization.

So the earlier plateau reading was a **budget artifact measured at a single budget**. The
distinction that matters: this was found by sweeping the optimizer at a *fixed* region and
*fixed* V form, which is a budget question. Widening the region until violations vanish
would have been the forbidden move, and is not what happened.

### The resolution caveat, which is the whole basis of the claim

**0 violations in 54,868 samples does not mean the rate is zero.** By the rule of three it
bounds the true rate at roughly `< 5.5e-05`. Sampling cannot see below its own floor, and
that floor is exactly the room a directed search has to work in. This is what makes a
sampling-to-proof gap possible rather than a contradiction.

It is also the risk. **A1 samples 500,000, about 9x more.** If the true rate lies anywhere
in the `2e-06` to `5.5e-05` band, A0's audit sees nothing while A1's sees roughly 25. A1's
own 500k audit is therefore the real cleanliness test, and if it comes back dirty there is
no gap at that sample count. Read A1's sampling line before reading its BaB line.

**Use `c=0.5`, not `c=0.05`.** Both are sampling-clean, but c=0.05's margin is `+6.4e-07`,
close enough to the `1e-9` structural band that a small BaB counterexample would be hard to
separate from numerical noise. c=0.5 gives `+6.6e-06`, ten times the headroom.

`v_steps=8000` is the threshold; 16000 and 32000 buy nothing further. A1 uses **16000** to
sit clear of the knee rather than on it.

---

## A1, seed 0: no gap. The certificate holds.

`--seed 0 --n-samples 500000 --v-steps 16000`, `c=0.5`, `burn_in=100`, on the admissible
region. Run on Colab.

> **PROVISIONAL.** `results/a1_seed0.json` is **not yet committed**. The Colab session's
> local record was lost after the run, so the file could not be pulled; the numbers below
> are transcribed from the run output. No claim here is final until the JSON is landed and
> this notice is removed. Re-running seed 0 reproduces it deterministically.

**[1] Sampling, read first.** `0` violations in `275,674` samples, worst cond `+6.68e-06`.
The baseline is clean, so the gap question passes to branch-and-bound. Had sampling found
violations there would be no gap at this sample count and BaB would not have been read at
all.

**[2] Branch-and-bound.** Boxes 0, 1, 2 **verified**. Box 3 **unknown** (timeout, 180
subdomains). `bab_counterexamples = 0`, `reachable_counterexamples = 0`,
`gap_demonstrated = false`.

**The result: there is no sampling-to-proof gap on this policy and this region.** The
certificate holds where the verifier resolved it. Directed formal search found nothing that
sampling missed, because it found nothing at all.

**Box 3's `unknown` is verifier incompleteness and is not a finding.** It means the bound
stayed loose and no counterexample was found within the time budget. It is not a violation,
not a gap, not a partial gap, and not evidence of one. It is equally not evidence of safety
on that box: the box is simply unresolved. Anyone quoting this run must say "three boxes
verified, one unresolved," never "mostly verified" and never anything implying box 3 hides
something. Category 3, and nothing more.

**Resolution bound.** 0 in 275,674 samples bounds the true violation rate at roughly
`< 1.1e-05` by the rule of three. As always this is a bound, not a zero.

**One seed.** The standing rule for a positive was that a gap on seed 0 is a candidate
needing multi-seed confirmation. The same standard applies to a null: this is one
initialization of V, on one seed, with one box unresolved. See the strategy note below.

---

## Status

- Gates **pass**. Policy **trained**. Pipeline **runs end to end**.
- A0 re-centred sweep **done**, and the budget axis swept at the admissible setting.
- Clean sampling baseline **reached**: `burn_in=100`, `c=0.5`, `v_steps >= 8000` gives
  **0 violations in 54,868 samples** at `cov_full = 0.683`, worst cond `+6.6e-06`.
- The earlier "obstruction, no gap available" conclusion is **retracted**; it was drawn from
  `v_steps=2000` alone. See the retraction above.
- A1 seed 0 **run**: sampling clean (0 / 275,674), BaB found **0** counterexamples, three
  boxes verified and one unresolved. **No gap. The certificate holds.**
- Gap: **not found**, and therefore not claimed. The hypothesis did not hold here.
- `results/a1_seed0.json` **not yet committed** (session lost). The A1 section is marked
  provisional until it is.
- Multi-seed A1 **not run**. One seed is not an evidence base for a null any more than for
  a positive.
- Positive control **not run**. A null is only as strong as the demonstrated ability to
  detect a violation that is really there.
- dReal confirmation harness **not written**. Nothing may be called dReal-confirmed.
- DreamerV3 **not started**. Scope stays the one-step latent transition, smallest model.
- Repo is **local only**. No remote, nothing pushed, publishing undecided.
