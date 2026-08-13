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

## DreamerV3 leg (D1): one-step latent transition, ported and gated

**NOT a result.** No gap measured, no JSON, nothing to email, no artifact may quote
any number here as a finding. This section records the infrastructure and the first
measurement (boundability), per the locked scope: one-step latent transition, smallest
official model (`size1m`), nothing about the imagined rollout.

### The port (`src/dreamer.py`)

Faithful torch port of DreamerV3's one-step latent transition: `RSSM._core` +
`RSSM._prior`/`_logit` from danijar/dreamerv3 @ `e3f0224`, layers `Linear` /
`BlockLinear` / `Norm(rms)` from danijar/embodied @ `a6583c1`, at the `size1m` config
(deter=512, hidden=64, stoch=32, classes=4, blocks=8, silu, rms).

- Parameters keep upstream value names (`kernel`/`bias`/`scale`), so a JAX checkpoint
  loads by key via `load_jax_arrays`; only plain Linear kernels are transposed
  (in,out)->(out,in). BlockLinear expands to a block-diagonal weight so the verifier
  sees one BoundLinear.
- The certified object is the CLOSED one-step map on the latent itself,
  z -> z' = (deter', stoch'), under the frozen deterministic actor pi(z) = tanh(mean):
  deter' = core(z, pi(z)), stoch' = MEAN of the prior categorical
  (1-unimix)*softmax(logits) + unimix/classes (the transition module still returns
  (deter', logits', stoch') raw). The certificate describes the deterministic mean
  map; the model samples one-hot during imagination. Stated in every artifact.
- Upstream action normalization `a /= max(1, |a|)` is the identity for every action
  distribution DreamerV3 uses (tanh-bounded continuous, one-hot discrete). The port
  passes actions through, pins the identity by test, and documents the contract;
  feeding out-of-range actions makes the port describe a different function.

### Transcription cross-checks (`tests/test_dreamer_transition.py`), all passing

The port is checked against an INDEPENDENT re-implementation of the upstream formulas
(einsum BlockLinear, x*sigmoid silu, flat-tensor gates). Agreement ~1e-8. The gate
caught three real slips before they could become results:

1. the GRU candidate must be `tanh(reset * cand)`; dropping the sigmoided reset factor
   drifted the output by 1.4e-01,
2. `dyngru` consumes the DYNHID output (width deter), not the 2048-wide concat,
3. the gate split is on the GROUPED tensor (flat2group then split); my first reference
   (flat chunk) was the wrong one, not the port.

`load_jax_arrays` is verified end to end: a synthetic checkpoint with upstream key
names loads into a fresh port and reproduces the arrays' function to ~3e-8.

### Boundability gate (`experiments/d1_smoke.py`), passing -- and its measurement

auto_LiRPA 0.7.2 traces and CROWN-bounds the full one-step graph. Bounds are SOUND at
every radius tested. The measurement is the looseness, and it is the guardrail-predicted
scaling boundary:

| graph | width/span at r=0.02 |
|---|---|
| Linear -> RMSNorm -> SiLU (one stage) | 1.9e2x |
| BlockLinear -> RMSNorm(512) -> SiLU | 8.7e1x |
| full one-step transition | 4.3e6x |

Single stages are ~1e2x loose; the chain compounds to ~1e6x. The driver is RMSNorm's
1/sqrt(mean(x^2)+eps): interval propagation gives mean2 a lower bound of 0 on any box
that crosses zero, so rsqrt reaches 1/sqrt(eps)=100 and the norm amplifies; the GRU's
elementwise products compound it. Sound but vacuous at audit-relevant radii: the audit
must branch-and-bound tiny boxes, and the expected verdict on any realistic latent box
is "unknown" (category 3) -- a documented scaling boundary, not a gap, not a headline.

### auto_LiRPA 0.7.2 gotchas worth not re-deriving

- `BoundDiv.forward` has an ad-hoc LayerNorm special case that fires on ANY
  `x / sqrt(...)` pattern (div's second input being a BoundSqrt) and rebuilds a
  layer-norm-style deviation from `inputs[0].inputs[0]` -- wrong shape and value for
  RMSNorm. Write norms without div-by-sqrt: rsqrt(s) as exp(-0.5*log(s)) (exp and log
  both have proper relaxations).
- silu as x/(1+exp(-x)) is boundable but its exp relaxation overflows to inf slopes on
  wide intervals and poisons CROWN with NaN A matrices. Write silu as
  x*(0.5+0.5*tanh(x/2)): identically equal, finite everywhere.
- `copy_` from a grad-requiring tensor flips requires_grad on a buffer in torch >= 2.11;
  auto_LiRPA's jit tracer then rejects the module. Build derived buffers under no_grad.
- No BoundRsqrt and no BoundSiLU in 0.7.2; BoundPow with fractional exponents is not
  the safe path.

### Policy port and closed loop (`src/dreamer.py`)

The actor head is transcribed from dreamerv3/agent.py + embodied/jax/heads.py
(`MLPHead`, `bounded_normal`): feat = concat(deter, stoch_flat), 3x
(Linear(64) + RMSNorm + silu), Linear(64->act_dim, outscale 0.01 at init), a = tanh(mean).
`outs.Agg` aggregates only loss/entropy terms, never `pred`, so the deterministic
action is exactly per-component tanh(mean). Module names match upstream
(`mlp/linear{i}`, `mlp/norm{i}`, `head/mean`), so `load_jax_arrays` loads real
`world_model/pol/...` keys. The closed loop `OneStepClosedLoop` is the FULL square
map z -> z' under the policy (cross-checked by test to equal trans(policy(z))).

### Why the certified map must be FULL (deter, stoch) -> (deter', stoch')

A tempting design certifies only the deter component (V on deter, loop outputting
deter') because the stoch' output is the part CROWN cannot bound. That design is a
STRUCTURAL TRAP and was rejected: the deter-component map z -> deter' is not square,
and V_d(deter*) = 0 is its global minimum, so on the 128-dim slice
{deter = deter*, stoch != stoch*} the condition reads cond = -V_d(deter') < 0 BY
CONSTRUCTION -- manufactured violations the model never commits, and BaB would find
them (the on-policy trajectory passes through deter ~ deter* on its way to z*, so the
reachability gate rates them near-support). This is the A0 trap on the latent. The
full map has no such slice: the only structural zero is z* itself, exactly like the
pendulum. Recorded here so nobody re-derives it.

### The audit script (`experiments/d1_sampling_gap.py`), smoke-tested end to end

The D1 audit mirrors A1 on the latent: frozen model -> on-policy latent support from
the model's OWN closed loop -> latent fixed point z* (settle + damped polish, run
REFUSES to proceed if residual > 1e-4) -> region + k-dim annulus -> fit V (E10 form,
c=0.5) -> sampling audit -> branch-and-bound -> reachability gate -> gap_demonstrated.
The smoke run (--synthetic --quick, random weights, NEVER quoted) passes end to end:

    equilibrium residual 2.98e-08 (converged)      region: 4 annulus boxes, 83.8%
    sampling: 0/12866 violations, worst cond +2.23e-03
    BaB: WALL -- verifier cannot process the certified graph: AssertionError:
          Only positive values are supported in BoundReciprocal (category 3)
    gap_demonstrated: false                           total 130 s

Design points the smoke run validates:

- The annulus hole excludes z* along only the k largest-halfwidth dims (default k=4,
  smoke k=2): removing z* from every certified box is all that is required, and the
  slab decomposition then yields 2k boxes instead of 2*640. Same hole in training,
  sampling, and certification.
- BaB is WALL-GATED: the certified graph contains stoch' = exp/sum-div, and
auto_LiRPA's interval pass asserts in BoundReciprocal (exp interval lower bound 0 on
any box at the measured vacuity). One structural probe on the first box establishes
the wall; every box is then recorded unknown with the reason -- never certified,
never a violation. If the probe ever survived (finite bounds), the per-box
branch-and-bound loop runs with per-box verdicts.
- Per-call CPU timings on this laptop (heavily loaded): loop forward ~40 ms/call,
  BoundedModule build for the full cond graph ~4 min (torch.onnx export of the
  ~400-node graph), V step @512 ~3 s. The real run belongs on Colab (GPU), and the
  support export should batch the loop.
- Sampling clean on the RANDOM-weight synthetic model (worst cond +2.2e-03): the
  pipeline does not manufacture violations. Detection capability (cond < 0 flagged,
  reachability-gated) is exercised by construction; a positive-control model is still
  on the open list.

### Turnkey Colab notebook (`colab/d1_audit.ipynb`, built by `colab/build_d1_notebook.py`)

The heavy run is a notebook now, generated from the builder (edit the builder, not
the JSON). It: pins sources (rl-wm-audit zip upload; dreamerv3 @ e3f0224 with its
VENDORED embodied -- the exact nets.py the port transcribed, verified; cnl-work
with the verify.py SHA-256 pin), installs jax[cuda12]==0.4.33 + numpy<2 +
dm_control + auto_LiRPA on Colab GPU, runs the gates, trains size1m on
dmc_cartpole_balance (pixels), exports the JAX params under the upstream key
names (dyn/*, pol/* -- the loader's module boundaries), rolls out the agent for
the on-policy latent support (posterior latents; the certified map is the
prior-mean surrogate -- recorded, never conflated), and runs
`d1_sampling_gap.py --action-dim 1`.

**auto_LiRPA install (Colab-proven fix):** `pip install auto_LiRPA` FAILS on Colab.
The published PyPI releases carry legacy `torch<1.13` constraints; pip backtracks
and dies with `ResolutionImpossible` / `Could not find a version that satisfies
torch<1.13.0,>=1.8.0` against Colab's torch 2.11, and 0.7.2 is NOT on PyPI at all
(404). The validated verifier is auto_LiRPA 0.7.2 from GitHub at commit
`5a098e8f9fb5786a428a024981d833d303921f2d` (June 2026 release; requires
`torch>=2.0,<2.12`). The notebook installs it via
`git+https://github.com/Verified-Intelligence/auto_LiRPA.git@5a098e8f...` and
asserts `auto_LiRPA.__version__ == '0.7.2'` before anything runs. Order matters:
dreamerv3's `numpy<2` applies first, then auto_LiRPA's `numpy>=2` bump leaves
numpy 2.x -- the exact numpy/torch/auto_LiRPA combo validated locally. Install
lines fail loudly (no `| tail` to mask pip exit codes).

Second catch (Colab-proven): commit 5a098e8f's metadata declares
`python_requires='~=3.11.0'`, so pip refuses it on Colab's Python 3.12
(measured: `Package 'auto-lirpa' requires a different Python: 3.12.13 not in
'~=3.11.0'`). The constraint is advisory -- the identical commit runs on Python
3.13.9 on the validating machine -- so the notebook installs it with
`--ignore-requires-python` and the correctness gates re-validate the verifier on
Colab's runtime. The pip warnings about Colab-preinstalled flax/opencv/etc.
wanting newer jax/numpy are noise: the pinned dreamerv3 stack imports no flax
(verified in the vendored embodied: einops/jax/ninjax only).

Key upstream APIs pinned for the notebook (all from dreamerv3 @ e3f0224): the
checkpoint is `logdir/ckpt/<latest>/agent.pkl` = pickle of
`{'params': {flat ninjax keys}, 'counters'}`; `agent.policy(carry, obs)` returns
the visited latent in `carry[1] = {deter, stoch}`; `make_agent`/`make_env` from
`dreamerv3.main` rebuild the agent in-process; the task flag is
`dmc_cartpole_balance` (main.py splits on '_' and looks up the suite).

### Open items for the heavy run

- RUN the notebook on Colab. Training is ~1.5-3 h on a T4 at 1e6 steps (the
  notebook lowers to 3e5 for a quick run); the audit itself is ~10-20 min on GPU.
- Positive control (non-contractive latent map) to demonstrate the sampling audit can
  find violations; and multi-seed runs.
- Expected real-run outcome, stated plainly: sampling baseline (whatever it measures)
  + BaB unknown everywhere (the categorical wall, category 3). A NULL is the likely
  honest result; a sampling finding on a trained model would be the real result, and
  it must survive the reachability gate before it means anything.

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
- DreamerV3 leg **built end to end**: transition + actor ported (`src/dreamer.py`),
  transcription cross-checks pass (11), CROWN-boundability gate passes (sound but
  vacuous ~1e6x at r=0.02), and the full D1 audit script runs on smoke sizes
  (`experiments/d1_sampling_gap.py` --synthetic --quick: fixed point verified,
  sampling clean, BaB records the categorical wall, gap_demonstrated false). The
  certified map is the FULL one-step map z -> z' under the deterministic actor; the
  deter-only variant is a documented structural trap. Heavy run (trained model,
  Colab) **not done**; `d1_quick_seed0.json` is smoke-only, never quoted. Scope
  stays one-step, smallest model.
- Repo is **local only**. No remote, nothing pushed, publishing undecided.
