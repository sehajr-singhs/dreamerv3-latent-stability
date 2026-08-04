"""Generate colab/audit.ipynb. Editing this script is preferred over editing the
notebook JSON by hand, which is easy to corrupt and impossible to review in a diff."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/sehajr-singhs/rl-wm-audit"
CNL_URL = "https://github.com/sehajr-singhs/certified-neural-lyapunov"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {},
            "source": "\n".join(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": "\n".join(lines)}


cells = [
    md("# RL World-Model Verification Audit — Pendulum-v1 (A1)",
       "",
       "Measures the **sampling-to-proof gap** on a *frozen* Stable-Baselines3 SAC policy:",
       "states where a 500k-sample audit of a Lyapunov decrease condition finds nothing,",
       "but CROWN branch-and-bound finds a genuine violation **that the policy actually reaches**.",
       "",
       "### What this notebook will and will not claim",
       "",
       "- It reports the gap **this run** measures. No figure is carried over from the",
       "  power-system work; that setup was a different system and its numbers do not transfer.",
       "- A counterexample only counts if it passes the **reachability gate**: the state must lie",
       "  on the frozen policy's own on-policy support. Any unconstrained network violates almost",
       "  anything over a wide enough box, so an off-distribution violation is *not a finding* and",
       "  is discarded rather than reported with a caveat.",
       "- The verifier's verdict is **three-way**: `violation` / `unknown` / `certified`.",
       "  `unknown` means the bound stayed loose and no counterexample was found. That is verifier",
       "  incompleteness and is **never** evidence of safety.",
       "- Certification runs on an **annulus**. `cond(x*) = 0` exactly by construction, so any box",
       "  containing the equilibrium has true infimum 0 and cannot certify at a positive margin.",
       "  That is structural, not a verifier failure, and the hole is small and reported.",
       "",
       "Runtime: CPU is fine and is what this was measured on. No GPU needed."),

    md("## 1. Environment",
       "",
       "`dReal` has a Linux wheel, which is the main reason the heavy loop lives here rather",
       "than on Windows. It is used only to **confirm** a specific CROWN counterexample;",
       "a dReal timeout means *unconfirmed*, never *safe*."),

    code("!pip -q install 'auto_LiRPA' 'stable-baselines3>=2.3' 'gymnasium' 2>&1 | tail -2",
        "!pip -q install dreal 2>&1 | tail -2   # Linux-only; optional, used for confirmation",
        "",
        "import importlib",
        "for m in ['torch', 'auto_LiRPA', 'stable_baselines3', 'gymnasium']:",
        "    mod = importlib.import_module(m)",
        "    print(f'{m:20s}', getattr(mod, '__version__', 'ok'))",
        "try:",
        "    import dreal; print('dreal               available')",
        "except Exception as e:",
        "    print('dreal               NOT available ->', type(e).__name__,",
        "          '(CROWN results still stand; confirmation step will be skipped)')"),

    md("## 2. Source",
       "",
       "`certify_box` is **reused verbatim** from the cnl-work verifier, not reimplemented, so",
       "these results come from the same driver that was cross-checked against JacobianOP and",
       "dReal. Its git commit is recorded in every result JSON."),

    code(f"REPO_URL = '{REPO_URL}'",
        f"CNL_URL  = '{CNL_URL}'",
        "",
        "import os, subprocess",
        "",
        "def clone(url, dest):",
        "    if os.path.isdir(dest):",
        "        print('present:', dest); return True",
        "    r = subprocess.run(['git', 'clone', '--depth', '1', url, dest],",
        "                       capture_output=True, text=True)",
        "    print(('cloned: ' if r.returncode == 0 else 'FAILED: ') + dest)",
        "    if r.returncode: print(r.stderr.strip()[:400])",
        "    return r.returncode == 0",
        "",
        "ok_cnl  = clone(CNL_URL,  '/content/cnl-work')",
        "ok_repo = clone(REPO_URL, '/content/rl-wm-audit')",
        "",
        "if not (ok_cnl and ok_repo):",
        "    print()",
        "    print('If a repo is private or not yet pushed, upload a zip instead:')",
        "    print('  from google.colab import files; files.upload()')",
        "    print('  !unzip -q rl-wm-audit.zip -d /content/')",
        "",
        "os.environ['CNL_WORK'] = '/content/cnl-work'",
        "%cd /content/rl-wm-audit"),

    md("## 3. Correctness gates",
       "",
       "Four checks, each of which would silently invalidate every number downstream:",
       "our dynamics vs gymnasium's, our extracted actor vs `sb3.predict`, and whether CROWN",
       "can soundly and non-vacuously bound the closed loop (`sin`/`cos` + `clamp` + `tanh`).",
       "",
       "**If any gate fails, stop.** Do not interpret results past a failed gate."),

    code("!python experiments/smoke_crown.py"),

    md("## 4. The frozen policy",
       "",
       "Trained once and then never touched. v1 audits a policy *as trained*; only V is fit.",
       "",
       "The actor is deliberately small (`[64, 64]`) because narrow networks branch-and-bound",
       "far better, and `use_sde=False` so the deterministic action is a clean `tanh(mu)`",
       "rather than a state-dependent noise object the bounded graph cannot trace."),

    code("import os",
        "if not os.path.exists('models/sac_pendulum.zip'):",
        "    !python src/train_policy.py 0",
        "else:",
        "    print('using committed checkpoint models/sac_pendulum.zip')",
        "",
        "from stable_baselines3 import SAC",
        "from stable_baselines3.common.evaluation import evaluate_policy",
        "import gymnasium as gym",
        "m = SAC.load('models/sac_pendulum', device='cpu')",
        "r, s = evaluate_policy(m, gym.make('Pendulum-v1'), n_eval_episodes=30, deterministic=True)",
        "print(f'return {r:.1f} +/- {s:.1f}   (Pendulum-v1 is conventionally solved near -200)')"),

    md("## 5. A0 — is a gap even measurable?",
       "",
       "Run this **before** A1. A sampling-to-proof gap is only meaningful if sampling is *clean*,",
       "i.e. the empirical violation rate is at or near zero. If V cannot be driven there, the",
       "honest conclusion is *this policy admits no such certificate on this region* — a real",
       "result, but a different one. Presenting a leftover violation rate as a gap would be",
       "dishonest, so this probe decides which claim is available.",
       "",
       "Note the region is defined from the policy's **steady-state** behaviour. Pendulum's",
       "controller is a *swing-up* policy: from a hanging start it must pump energy in before it",
       "can stabilize, so V necessarily increases during the transient and **no monotone Lyapunov",
       "function exists over the full support**. Certifying decrease there is impossible rather",
       "than hard. The reachability gate still uses the full visited set."),

    code("!python experiments/a0_v_feasibility.py --seed 0"),

    md("## 6. A1 — the audit",
       "",
       "Sampling audit vs branch-and-bound over the annulus, then the reachability gate.",
       "`gap_demonstrated` goes true **only** when sampling found nothing and BaB found a",
       "violation at a state the frozen policy demonstrably reaches.",
       "",
       "Plain `CROWN`, not `CROWN-Optimized`: on a sum-of-squares V the product nodes make",
       "alpha-optimization expensive, and it was measured *far worse* (unknown on a quarter box",
       "after 234 s, versus the full box verified in 55 s with plain CROWN). Depth beats",
       "tightness here. Do not switch without re-measuring."),

    code("!python experiments/a1_sampling_gap.py --seed 0 --n-samples 500000 --v-steps 6000"),

    md("## 7. Read the result honestly",
       "",
       "The cell below prints the verdict and, importantly, the things that are *not* findings:",
       "off-distribution counterexamples and `unknown` boxes."),

    code("import json, glob",
        "path = sorted(glob.glob('results/a1_seed*.json'))[-1]",
        "log = json.load(open(path))",
        "",
        "print(path)",
        "print('verifier      ', log['verifier']['verifier_commit'][:12],",
        "      '(dirty)' if log['verifier'].get('verify_py_dirty') else '(clean)')",
        "print('policy return ', round(log['policy']['mean_return'], 1))",
        "print()",
        "r = log['result']",
        "print(f\"sampling      {r['sampling_violations']} violations in\",",
        "      f\"{log['sampling']['n_sampled']} samples ({r['sampling_rate']:.4%})\")",
        "print(f\"BaB           {r['bab_counterexamples']} counterexamples,\",",
        "      f\"{r['reachable_counterexamples']} pass the reachability gate\")",
        "print(f\"unknown boxes {log['bab']['n_unknown']}  <- verifier incompleteness, NOT safety\")",
        "print()",
        "print('GAP DEMONSTRATED:', r['gap_demonstrated'])",
        "print()",
        "for ce in log['counterexamples']:",
        "    tag = 'FINDING' if ce['reach_verdict'] == 'IN_SUPPORT' else 'not a finding'",
        "    print(f\"  {ce['reach_verdict']:16s} d={ce['normalized_distance_to_support']:.4f}\",",
        "          f\"cond={ce['cond']:.3e}  [{tag}]\")"),

    md("## 8. dReal confirmation (optional)",
       "",
       "dReal independently confirms a specific CROWN counterexample by SMT. It **confirms**;",
       "it does not certify the region. A timeout is `unconfirmed`, not `safe`.",
       "",
       "Skipped automatically if the dReal wheel did not install."),

    code("try:",
        "    import dreal",
        "except Exception:",
        "    print('dreal unavailable; skipping. CROWN counterexamples above are unaffected.')",
        "else:",
        "    print('dreal present. Confirmation harness is the next deliverable;')",
        "    print('until it is written, do not describe any counterexample as dReal-confirmed.')"),

    md("---",
       "",
       "### Scope note on DreamerV3",
       "",
       "The only positive claim this line of work makes about DreamerV3 is about the **one-step",
       "latent transition** on the smallest model size. It does not claim to have verified",
       "DreamerV3. The full imagined rollout compounds GRU product nodes and 32x32",
       "straight-through categorical latents and is expected to return `unknown`; that is",
       "reportable as a scaling boundary, and is a footnote rather than a headline."),
]

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

out = os.path.join(HERE, "audit.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out} ({len(cells)} cells)")
