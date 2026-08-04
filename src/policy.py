"""Extract a frozen SB3 policy into a plain, CROWN-boundable torch module.

We rebuild the deterministic actor out of bare `nn` layers rather than bounding
SB3's `Actor` object directly. SB3's forward path drags in feature extractors,
distribution objects and squashing helpers that auto_LiRPA has no business tracing,
and a graph it cannot trace is a graph we cannot certify. Copying the weights into a
Sequential keeps the bounded object honest and inspectable.

The policy is FROZEN throughout. v1 audits a policy as trained; nothing here trains
or fine-tunes it.
"""

import torch
import torch.nn as nn


class DeterministicActor(nn.Module):
    """obs -> deterministic action, matching SAC's `predict(deterministic=True)`.

    SAC's deterministic action is `tanh(mu(latent_pi(obs))) * scale + bias`; the
    Gaussian's log_std and sampling are irrelevant once we stop exploring.
    """

    def __init__(self, body: nn.Sequential, scale: float, bias: float):
        super().__init__()
        self.body = body
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("bias", torch.tensor(float(bias)))

    def forward(self, obs):
        return torch.tanh(self.body(obs)) * self.scale + self.bias


def extract_sac_actor(model) -> DeterministicActor:
    """Copy a trained SB3 SAC actor into a DeterministicActor.

    Raises rather than guessing if the architecture is not what we expect, since a
    silently-wrong policy would make every downstream number meaningless.
    """
    actor = model.policy.actor

    fe = actor.features_extractor
    if type(fe).__name__ != "FlattenExtractor":
        raise ValueError(
            f"expected FlattenExtractor (identity on a Box obs), got {type(fe).__name__}. "
            "A non-trivial feature extractor must be folded into the bounded graph "
            "explicitly before this policy can be certified."
        )

    layers = [l for l in actor.latent_pi] + [actor.mu]
    for l in layers:
        if not isinstance(l, (nn.Linear, nn.ReLU, nn.Tanh, nn.ELU)):
            raise ValueError(f"unboundable layer in actor: {type(l).__name__}")

    body = nn.Sequential(*[_clone(l) for l in layers])

    low = float(model.action_space.low[0])
    high = float(model.action_space.high[0])
    scale = (high - low) / 2.0
    bias = (high + low) / 2.0

    net = DeterministicActor(body, scale, bias)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _clone(layer):
    if isinstance(layer, nn.Linear):
        new = nn.Linear(layer.in_features, layer.out_features,
                        bias=layer.bias is not None)
        new.load_state_dict(layer.state_dict())
        return new
    return type(layer)()


def check_matches_sb3(net: DeterministicActor, model, obs, atol=1e-5):
    """Assert the extracted actor reproduces SB3's own deterministic action.

    This is the gate that makes the extraction trustworthy. If it does not fire, every
    certificate downstream is about a different network than the one that was trained.
    """
    import numpy as np
    ours = net(torch.as_tensor(obs, dtype=torch.float32)).detach().numpy()
    theirs, _ = model.predict(obs, deterministic=True)
    theirs = np.asarray(theirs, dtype=np.float32).reshape(ours.shape)
    err = float(np.abs(ours - theirs).max())
    if err > atol:
        raise AssertionError(
            f"extracted actor disagrees with SB3 by {err:.3e} (> {atol:.1e}). "
            "Do not certify this policy until the extraction is fixed."
        )
    return err
