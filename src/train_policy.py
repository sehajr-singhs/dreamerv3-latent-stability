"""Train the SAC policy that v1 audits. Run once; the checkpoint is the artifact.

Architecture choices here are verification-driven, and both cost a little reward:

  net_arch=[64, 64]  A narrower actor branch-and-bounds far better than the SB3
                     default [256, 256]. Pendulum does not need the capacity.
  use_sde=False      gSDE's deterministic action routes through a state-dependent
                     noise distribution instead of a plain tanh(mu), which would put
                     an untraceable object in the bounded graph.

Neither choice touches the audit's honesty: we audit whatever policy this produces,
as trained, and report its actual return alongside every certificate.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "sac_pendulum")


def train(seed=0, timesteps=20_000, save=True):
    env = gym.make("Pendulum-v1")
    model = SAC(
        "MlpPolicy", env, seed=seed, device="cpu", verbose=0,
        learning_rate=1e-3, buffer_size=200_000, batch_size=256,
        gamma=0.98, tau=0.02, train_freq=64, gradient_steps=64,
        learning_starts=1_000, use_sde=False,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)

    eval_env = gym.make("Pendulum-v1")
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=30,
                                    deterministic=True)
    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
    return model, float(mean_r), float(std_r)


def load(path=MODEL_PATH):
    return SAC.load(path, device="cpu")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model, mean_r, std_r = train(seed=seed)
    print(f"seed={seed}  return over 30 deterministic episodes: "
          f"{mean_r:.1f} +/- {std_r:.1f}")
    # Pendulum-v1 is conventionally treated as solved around -200; a policy that has
    # not learned would sit near -1200. Report, do not silently accept.
    print("status:", "trained" if mean_r > -300 else "UNDERTRAINED, do not audit this")
    print("saved:", MODEL_PATH + ".zip")
