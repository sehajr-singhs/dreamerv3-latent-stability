# DreamerV3 Latent Stability

**Papers:** [Stable and Verifiable Latent Dynamics for World Models](docs/papers/DREAMER_LATENT_STABILITY_IEEE.pdf)
(IEEE format) · [same, NMI Letters format](docs/papers/DREAMER_LATENT_STABILITY_NMI.pdf) ·
[project website](https://sehajr-singhs.github.io/dreamerv3-latent-stability/).

The DreamerV3 leg of the RL world-model verification audit. Applying a
certified-Lyapunov verification toolchain to open-source RL, to measure the
**sampling-to-proof gap**: states where a large random-sampling audit of a safety or
decrease condition finds nothing, but directed formal search (CROWN branch-and-bound)
finds a genuine violation.

v1 audits a frozen Stable-Baselines3 SAC policy on `Pendulum-v1`. DreamerV3 comes second
and deliberately narrow (see *Scope* below).

## What a result here does and does not mean

This repo is built so a number cannot quietly become better than it is.

- **The reachability gate decides what counts.** Any unconstrained network violates almost
  any property over a wide enough box, so "BaB found a counterexample" is worth nothing on
  its own. A counterexample is a finding only if the frozen policy demonstrably reaches
  that state, measured against its own on-policy support. Off-distribution counterexamples
  are printed for transparency and explicitly do **not** count.
- **Three-way verdicts, never collapsed.** `certify_box` returns `violation`, `unknown`, or
  certified. `unknown` means the bound stayed loose and no counterexample was found: that is
  verifier incompleteness, and it is never evidence of safety.
- **Regions are reported against the on-policy distribution.** Every box records what
  fraction of visited states it contains, so a box covering almost nothing cannot pose as a
  box covering the policy's real behaviour.
- **No number is inherited.** Figures from the power-system work do not transfer; that was a
  different system. Every claim is backed by a committed JSON in `results/`.

## The two structural facts that shape the design

**Pendulum's state space is a cylinder.** A V built on raw `theta` would give `V(0) = 0` and
`V(2*pi) != 0` for the same physical state, so V is built on the observation
`(cos th, sin th, thdot)` about `o* = (1, 0, 0)`. Periodicity is then automatic, and the
quadratic term is `2 - 2 cos th + thdot^2`, positive definite about upright. For the same
reason the verifier's free variable is 2-D `(theta, thetadot)` with `cos`/`sin` computed
*inside* the bounded graph: a box over the raw 3-D observation would contain points with
`cos^2 + sin^2 != 1`, which are not physical states at all.

**The policy is a swing-up controller.** From a hanging start it must pump energy in before
it can stabilize, so V necessarily *increases* during the transient and **no monotone
Lyapunov function exists over the full on-policy support**. Certifying decrease there is
impossible rather than hard. The certification region is therefore built from the
*steady-state* portion of the rollouts, while the reachability gate keeps using the full
visited set, because a transient state is still a state the policy reaches.

A third, smaller fact: `cond(x*) = 0` exactly by construction, so any box containing the
equilibrium has true infimum 0 and cannot certify at a positive margin however good the
verifier is. Certification runs on an annulus; the hole is small, stated, and reported.

## Layout

```
src/dynamics.py       Pendulum dynamics as a boundable graph; matches gymnasium to 1e-16
src/policy.py         Frozen SB3 actor extracted into bare nn layers, checked vs sb3.predict
src/lyapunov.py       V, positive definite by construction on the cylinder; decrease conditions
src/reachability.py   The on-policy support and the reachability gate
src/region.py         Annulus decomposition, and box provenance vs the on-policy distribution
src/verifier.py       Loads certify_box from cnl-work; records its git commit
src/train_policy.py   Trains the frozen SAC policy (run once)
src/train_lyapunov.py Fits V to the frozen policy; sampled violation rate
src/dreamer.py        DreamerV3 one-step latent transition + deterministic actor:
                      faithful, boundable torch port of RSSM._core + _prior + policy
                      head at size1m; loads JAX checkpoints by key; closed loop z -> z'
src/lyapunov_latent.py Latent V (E10 form) on (deter, stoch_flat), sample/V-fit for D1

experiments/smoke_crown.py      4 correctness gates. Run first; stop if any fails.
experiments/a0_v_feasibility.py Can V be fit at all? Run before A1.
experiments/a1_sampling_gap.py  The audit.
experiments/d1_smoke.py         CROWN boundability gate for the Dreamer one-step transition.
experiments/d1_sampling_gap.py  The Dreamer audit (smoke-tested; heavy run on Colab).
tests/test_region.py            Monte-Carlo check that the annulus is exact.
tests/test_dreamer_transition.py Transcription cross-check: port vs independent reference.
colab/build_notebook.py         Generates colab/audit.ipynb, the Pendulum heavy run (edit this, not the JSON).
colab/build_d1_notebook.py      Generates colab/d1_audit.ipynb, the Dreamer heavy run:
                                train size1m on dmc_cartpole_balance, export weights +
                                on-policy latent support, run the D1 audit end to end.
                                auto_LiRPA is installed from the pinned GitHub commit
                                5a098e8f (0.7.2), NOT PyPI: PyPI releases carry torch<1.13
                                and cannot resolve against Colab's torch 2.11.
```

The verifier is **reused, not reimplemented**: `certify_box` is loaded by path from
`cnl-work/src/verify.py`, the version cross-checked against JacobianOP and dReal. Set
`CNL_WORK` to that repo root.

## Running

Heavy runs belong on Colab (`colab/audit.ipynb`). It is Linux, so the dReal wheel installs,
and it avoids the local OpenMP conflict described below.

```bash
python experiments/smoke_crown.py            # gates first
python experiments/a0_v_feasibility.py       # is a gap measurable?
python experiments/a1_sampling_gap.py        # the audit
python experiments/a1_sampling_gap.py --quick   # smoke sizes; never quote these
```

Locally, `KMP_DUPLICATE_LIB_OK=TRUE` is required or torch fails to import with
`OMP: Error #15`. Intel documents that workaround as able to *silently produce incorrect
results*, which is intolerable for a verification claim, so local runs are smoke tests only
and local numbers are never quoted in an artifact.

## Scope

The only positive claim made about DreamerV3 is about the **one-step latent transition** on
the smallest model size. This work does not claim to have verified DreamerV3. The full
imagined rollout compounds GRU product nodes and 32x32 straight-through categorical latents
and is expected to return `unknown`; that is reportable as a scaling boundary, not a headline.

Claims stay at: *directed formal search finds decrease-condition violations, in a defined and
on-policy-reachable region, that random sampling misses.* Nothing about alignment,
reward hacking, or guarantees.
