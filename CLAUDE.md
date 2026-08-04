# CLAUDE.md — RL World-Model Verification Audit

## What this project is

I (Sehaj) am adapting the certified-Lyapunov verification toolchain I built with
Prof. Wenqi Cui's group to **audit open-source RL world models** — first
Stable-Baselines3 (SB3) policies, later DreamerV3 latent dynamics. The goal is to
measure the **sampling-to-proof gap**: cases where hundreds of thousands of random
samples pass a safety/decrease condition, but a directed formal search (CROWN
branch-and-bound, confirmed with dReal) finds a real violation.

The eventual aim is a clean, defensible result I can email the DreamerV3 authors
about, to open a research conversation. That means **correctness and honesty matter
more than a big headline number.**

---

## Guardrails — read these before writing any code or claim

1. Do NOT assume or reproduce the 6.1% figure. It came from a specific power-system
   Lyapunov setup. On SB3 / DreamerV3 the gap is unknown until measured. Every artifact
   must report the gap THIS run found, with the environment, region, and decrease
   condition stated next to it. Never hardcode 6.1% anywhere.

2. State the Lyapunov / decrease condition and region explicitly, and justify V. A
   "violation" is only meaningful relative to a defined V and region. E10's
   positive-definite-by-construction V ports here — use it. An unjustified V is not a finding.

3. Reachability gate (hard requirement). A wide box makes the gap trivially winnable —
   any unconstrained NN violates almost anything. Gate every counterexample on whether the
   state is reachable under the policy's own closed-loop behavior from realistic initial
   states, and report every box relative to the on-policy state distribution. This is the
   E8 finding (certified region in the wrong place). Off-distribution violations are not findings.

4. SB3 first (Pendulum-v1, post-hoc Lyapunov certificate), DreamerV3 second and NARROW.
   The only positive Dreamer claim is the one-step latent transition over a latent box on
   the smallest model size. Not the RSSM rollout — GRU product nodes (E10) and 32x32
   straight-through categoricals compounding over imagined steps will return "unknown."
   A documented "unknown" is reportable as a scaling boundary; it is not the headline and
   not the email.

5. Preserve the three-way verdict. certify_box returns violation / unknown / certified and
   never launders "unknown" into "verified." Keep that. dReal (available on Colab's Linux)
   only CONFIRMS a specific CROWN counterexample; a dReal timeout is "unconfirmed," not "safe."

6. No overclaiming in any generated text. Never write "prevents deceptive alignment /
   reward-hacking / guarantees alignment." The honest claim: directed formal search finds
   decrease-condition violations in a defined, on-policy-reachable region that random
   sampling misses. Keep it to that.

7. Confer gates (hard stops). Before git push: show me results (JSON + figures) and the diff,
   wait for my OK. Before drafting or sending ANY author outreach: stop and confer — I write
   and approve all emails. Never push or send automatically.

---

## Decisions locked (2026-08-03)

These were conferred and approved. Do not relitigate them without asking.

- **v1 target = post-hoc Lyapunov certificate** on the *frozen* SB3 policy. Learn V for a
  policy we do not retrain, then measure the sampling-vs-proof gap on its decrease
  condition. Chosen for maximum reuse of E1/E4/E10 and because the claim shape matches the
  certified-neural-lyapunov paper, so both read together.
- **Environment = Pendulum-v1.** 3-D observation (cos, sin, thetadot), continuous torque,
  real equilibrium. CartPole / dimension-bump is deferred until Pendulum gives a clean,
  reachability-gated gap.
- **Dreamer scope = one-step latent transition only**, smallest model size. The email claims
  "sampling-to-proof gap in your one-step latent dynamics." It never claims "we verified
  DreamerV3." The full imagined rollout is out of scope as a positive claim; attempt it only
  to document the "unknown" scaling boundary, which is a footnote and not the headline.
- **Direct action-safety property (the rejected option 2)** may be added later as a cheap
  rung-1 gate, mirroring the two-bus gate in cnl-work. Not part of v1.

## Reuse, not rebuild

- `certify_box` from `cnl-work/src/verify.py:52` is already domain-agnostic: it takes any
  `nn.Module` condition plus a box and does CROWN + branch-and-bound + PGD counterexample
  search. Use it. Do not write a second verifier.
- Port from `cnl-work/experiments/e1_sampling_gap.py` rather than starting fresh.
- Domain coupling in cnl-work lives in `dynamics.py`, `controller.py`, `lyapunov.py`. Those
  are the files with analogues here; everything else should be imported.
- E10's positive-definite-by-construction V, `V = ||g(z) - g(z*)||^2 + c||z - z*||^2`, is the
  starting form. It exists specifically so condition (4a) is structural rather than learned.

## Execution split

- **Colab (Linux) runs the heavy loop.** dReal SMT confirmation is available there, which it
  is not on Windows (no wheel; cnl-work needed WSL2 + manually extracted libibex).
- **Local is for edits and smoke tests only.** Never run a long certification loop locally.

## Environment notes (measured 2026-08-03)

Local Windows has the dependencies split across two interpreters, and neither has both:

| interpreter | torch | auto_LiRPA | stable_baselines3 / gymnasium |
|---|---|---|---|
| `C:\Users\sehaj\anaconda3\python.exe` | 2.11.0 | 0.7.2 | missing |
| `AppData\Local\Programs\Python\Python312` | 2.10.0 | missing | 2.7.1 / 1.2.3 |

Do **not** pip install into anaconda base to fix this. That env is what `cnl-work` is
validated against, and `cnl-work` has a recommendation letter riding on it. Use the
project venv instead (see README), which inherits anaconda's torch + auto_LiRPA via
system-site-packages and layers SB3 on top.

**`KMP_DUPLICATE_LIB_OK=TRUE` is required locally.** Importing torch in the venv otherwise
dies with `OMP: Error #15` (two copies of libiomp5md.dll). Intel documents this workaround as
able to "silently produce incorrect results," which is intolerable for a verification result,
so it is another reason the heavy loop belongs on Colab's Linux where the conflict does not
exist. Local numbers are smoke-test only and are never quoted in an artifact.

Other inherited gotchas from cnl-work worth not re-deriving:
- Quote `limited_by` alongside any bisected rate; a timed-out step silently halves it.
- CROWN-Optimized is *not* automatically better. On sum-of-squares V its product nodes made
  it far worse than plain CROWN (unknown on a quarter box in 234 s vs full box in 55 s).
  Depth beat tightness. Expect the same on any V with products, and measure before assuming.
- "unknown" is a real third verdict. It is never evidence of safety.
