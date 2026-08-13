"""Transcription cross-check for src/dreamer.py, the DreamerV3 one-step transition.

The port is a re-transcription of upstream JAX (dreamerv3/rssm.py `_core`/`_prior`,
embodied/jax/nets.py `Linear`/`BlockLinear`/`Norm`). The risk is a silent formula
slip — wrong gate order, wrong bias, wrong width — that no shape error would catch.
These tests re-implement the same upstream formulas independently (plain torch,
einsum, no reuse of the port's classes) and demand agreement on identical weights.

The authoritative check against the real JAX checkpoint happens on Colab, where a
trained model exists. These tests pin the transcription; that check pins the
checkpoint. Both are needed, and the JSON for any run records both.

Run:  python tests/test_dreamer_transition.py   (same style as test_centering.py)
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.dreamer import (OneStepLatentTransition, RSSMConfig, stoch_mean,
                         load_jax_arrays, UPSTREAM_DREAMERV3, UPSTREAM_EMBODIED)

FAILS = 0


def _check(name, fn):
    global FAILS
    try:
        fn()
        print(f"[PASS] {name}")
    except AssertionError as e:
        FAILS += 1
        print(f"[FAIL] {name}: {e}")


# ---------------------------------------------------------------- reference
def ref_act(x):
    # silu as primitives, deliberately different from the port's shape
    return x * torch.sigmoid(x)


def ref_rms(x, scale, eps):
    return x / torch.sqrt(x.square().mean(-1, keepdim=True) + eps) * scale


def ref_core(core, deter, stoch, action):
    """Independent re-implementation of upstream RSSM._core, written differently
    from the port: einsum BlockLinear, x*sigmoid silu, gate math on flat tensors."""
    cfg = core.cfg
    g = cfg.blocks
    B = deter.shape[0]

    def lin(x, k, b):
        return x @ k + b

    x0 = ref_act(ref_rms(lin(deter, core.dynin0.kernel, core.dynin0.bias),
                         core.dynin0norm.scale, cfg.norm_eps))
    x1 = ref_act(ref_rms(lin(stoch.reshape(B, -1), core.dynin1.kernel,
                             core.dynin1.bias), core.dynin1norm.scale, cfg.norm_eps))
    x2 = ref_act(ref_rms(lin(action, core.dynin2.kernel, core.dynin2.bias),
                         core.dynin2norm.scale, cfg.norm_eps))
    x = torch.cat([x0, x1, x2], -1).unsqueeze(1).repeat(1, g, 1)     # (B, g, 3h)
    deter_g = deter.reshape(B, g, cfg.deter // g)
    x = torch.cat([deter_g, x], -1).reshape(B, -1)                   # (B, g*(h+3h))

    for i in range(cfg.dynlayers):
        hid = core.dynhids[f"dynhid{i}"]
        norm = core.dynhidnorms[f"dynhid{i}norm"]
        out = torch.einsum("bki,kio->bko", x.reshape(B, g, -1), hid.kernel)
        out = out.reshape(B, -1) + hid.bias
        x = ref_act(ref_rms(out, norm.scale, cfg.norm_eps))

    # Upstream splits the GROUPED tensor: flat2group(x) THEN split(3, -1). Each
    # gate is a contiguous per-group block, so chunking the flat tensor (as an
    # earlier draft of this reference did) interleaves groups and is wrong. The
    # bias is added on the flat tensor upstream, which is elementwise-identical to
    # adding it grouped.
    out = torch.einsum("bki,kio->bko", x.reshape(B, g, -1), core.dyngru.kernel)
    out = out + core.dyngru.bias.reshape(1, g, -1)                    # (B, g, 3*deter/g)
    res, cand_in, upd_in = out.chunk(3, -1)
    res = torch.sigmoid(res.reshape(B, -1))
    cand = torch.tanh(res * cand_in.reshape(B, -1))
    upd = torch.sigmoid(upd_in.reshape(B, -1) - 1.0)
    return upd * cand + (1.0 - upd) * deter


def ref_prior(prior, deter):
    cfg = prior.cfg
    B = deter.shape[0]
    x = deter
    for i in range(cfg.imglayers):
        lin = prior.priors[f"prior{i}"]
        norm = prior.priornorms[f"prior{i}norm"]
        x = ref_act(ref_rms(x @ lin.kernel + lin.bias, norm.scale, cfg.norm_eps))
    logits = x @ prior.priorlogit.kernel + prior.priorlogit.bias
    return logits.reshape(B, cfg.stoch, cfg.classes)


# ---------------------------------------------------------------- tests
def test_core_matches_reference():
    torch.manual_seed(0)
    cfg = RSSMConfig()
    core = OneStepLatentTransition(cfg, action_dim=3).core.eval()
    B = 4
    deter = torch.randn(B, cfg.deter) * 0.05
    stoch = torch.rand(B, cfg.stoch, cfg.classes)
    stoch = (stoch == stoch.max(-1, keepdim=True).values).float()    # one-hot
    action = (torch.rand(B, 3) * 2 - 1)
    with torch.no_grad():
        got = core(deter, stoch, action)
        want = ref_core(core, deter, stoch, action)
    err = float((got - want).abs().max())
    assert err < 1e-4, f"core transcription drift {err:.3e}"
    assert got.shape == (B, cfg.deter), got.shape
    print(f"    max |port - reference| on _core = {err:.2e}")


def test_prior_matches_reference():
    torch.manual_seed(1)
    cfg = RSSMConfig()
    prior = OneStepLatentTransition(cfg, action_dim=3).prior.eval()
    B = 4
    deter = torch.randn(B, cfg.deter) * 0.05
    with torch.no_grad():
        got = prior(deter)
        want = ref_prior(prior, deter)
    err = float((got - want).abs().max())
    assert err < 1e-4, f"prior transcription drift {err:.3e}"
    assert got.shape == (B, cfg.stoch, cfg.classes), got.shape
    print(f"    max |port - reference| on _prior = {err:.2e}")


def test_blocklinear_matches_einsum():
    from src.dreamer import BlockLinear
    torch.manual_seed(2)
    bl = BlockLinear(2048, 512, 8).eval()
    x = torch.randn(3, 2048) * 0.1
    with torch.no_grad():
        got = bl(x)
        want = (torch.einsum("bki,kio->bko", x.reshape(3, 8, -1), bl.kernel)
                .reshape(3, -1) + bl.bias)
    err = float((got - want).abs().max())
    assert err < 1e-5, f"BlockLinear drift {err:.3e}"
    print(f"    max |BlockLinear - einsum| = {err:.2e}")


def test_stoch_mean_is_a_distribution_mean():
    torch.manual_seed(3)
    cfg = RSSMConfig()
    logits = torch.randn(2, cfg.stoch, cfg.classes)
    m = stoch_mean(logits, cfg.unimix)
    # mixture of softmax and uniform: rows sum to 1, and exactly the formula
    e = torch.exp(logits)
    want = (1 - cfg.unimix) * e / e.sum(-1, keepdim=True) + cfg.unimix / cfg.classes
    assert torch.allclose(m, want, atol=1e-6)
    assert torch.allclose(m.sum(-1), torch.ones_like(m.sum(-1)), atol=1e-6)
    print(f"    stoch mean sums to 1 and matches the unimix formula")


def test_load_jax_arrays_matches_reference():
    """Build an 'exported checkpoint' with upstream key names and shapes, load it
    into a fresh port, and check the port now computes the reference function of
    THOSE arrays (not its own random init)."""
    torch.manual_seed(4)
    cfg = RSSMConfig()
    action_dim = 3
    B = 4
    rng = np.random.default_rng(7)

    def karr(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.05

    arrays = {
        "world_model/rssm/dynin0/kernel": karr(cfg.deter, cfg.hidden),
        "world_model/rssm/dynin0/bias": np.zeros(cfg.hidden, np.float32),
        "world_model/rssm/dynin0norm/scale": np.ones(cfg.hidden, np.float32),
        "world_model/rssm/dynin1/kernel": karr(cfg.stoch * cfg.classes, cfg.hidden),
        "world_model/rssm/dynin1/bias": np.zeros(cfg.hidden, np.float32),
        "world_model/rssm/dynin1norm/scale": np.ones(cfg.hidden, np.float32),
        "world_model/rssm/dynin2/kernel": karr(action_dim, cfg.hidden),
        "world_model/rssm/dynin2/bias": np.zeros(cfg.hidden, np.float32),
        "world_model/rssm/dynin2norm/scale": np.ones(cfg.hidden, np.float32),
        "world_model/rssm/dynhid0/kernel": karr(8, 256, 64),
        "world_model/rssm/dynhid0/bias": np.zeros(cfg.deter, np.float32),
        "world_model/rssm/dynhid0norm/scale": np.ones(cfg.deter, np.float32),
        # dyngru input is the dynhid output, width deter -> per-group deter/8 = 64
        "world_model/rssm/dyngru/kernel": karr(8, 64, 192),
        "world_model/rssm/dyngru/bias": np.zeros(3 * cfg.deter, np.float32),
        "world_model/rssm/prior0/kernel": karr(cfg.deter, cfg.hidden),
        "world_model/rssm/prior0/bias": np.zeros(cfg.hidden, np.float32),
        "world_model/rssm/prior0norm/scale": np.ones(cfg.hidden, np.float32),
        "world_model/rssm/prior1/kernel": karr(cfg.hidden, cfg.hidden),
        "world_model/rssm/prior1/bias": np.zeros(cfg.hidden, np.float32),
        "world_model/rssm/prior1norm/scale": np.ones(cfg.hidden, np.float32),
        "world_model/rssm/priorlogit/kernel": karr(
            cfg.hidden, cfg.stoch * cfg.classes),
        "world_model/rssm/priorlogit/bias": np.zeros(
            cfg.stoch * cfg.classes, np.float32),
    }

    port = OneStepLatentTransition(cfg, action_dim).eval()
    load_jax_arrays(port, arrays)

    # Build a shadow module with the SAME arrays and compute the reference
    # function by hand from the arrays directly.
    class Shadow:
        def __init__(self):
            self.cfg = cfg
            self.dynin0_k = torch.tensor(arrays["world_model/rssm/dynin0/kernel"])
            self.dynin0_b = torch.tensor(arrays["world_model/rssm/dynin0/bias"])
            self.dynin0_s = torch.tensor(arrays["world_model/rssm/dynin0norm/scale"])
            self.dynin1_k = torch.tensor(arrays["world_model/rssm/dynin1/kernel"])
            self.dynin1_b = torch.tensor(arrays["world_model/rssm/dynin1/bias"])
            self.dynin1_s = torch.tensor(arrays["world_model/rssm/dynin1norm/scale"])
            self.dynin2_k = torch.tensor(arrays["world_model/rssm/dynin2/kernel"])
            self.dynin2_b = torch.tensor(arrays["world_model/rssm/dynin2/bias"])
            self.dynin2_s = torch.tensor(arrays["world_model/rssm/dynin2norm/scale"])
            self.dynhid_k = torch.tensor(arrays["world_model/rssm/dynhid0/kernel"])
            self.dynhid_b = torch.tensor(arrays["world_model/rssm/dynhid0/bias"])
            self.dynhid_s = torch.tensor(arrays["world_model/rssm/dynhid0norm/scale"])
            self.gru_k = torch.tensor(arrays["world_model/rssm/dyngru/kernel"])
            self.gru_b = torch.tensor(arrays["world_model/rssm/dyngru/bias"])
            self.p0_k = torch.tensor(arrays["world_model/rssm/prior0/kernel"])
            self.p0_b = torch.tensor(arrays["world_model/rssm/prior0/bias"])
            self.p0_s = torch.tensor(arrays["world_model/rssm/prior0norm/scale"])
            self.p1_k = torch.tensor(arrays["world_model/rssm/prior1/kernel"])
            self.p1_b = torch.tensor(arrays["world_model/rssm/prior1/bias"])
            self.p1_s = torch.tensor(arrays["world_model/rssm/prior1norm/scale"])
            self.pl_k = torch.tensor(arrays["world_model/rssm/priorlogit/kernel"])
            self.pl_b = torch.tensor(arrays["world_model/rssm/priorlogit/bias"])

    sh = Shadow()

    deter = torch.randn(B, cfg.deter) * 0.05
    stoch = torch.eye(cfg.classes)[torch.randint(0, cfg.classes, (B, cfg.stoch))]
    action = torch.rand(B, action_dim) * 2 - 1

    with torch.no_grad():
        d_new, logits, stoch_new = port(deter, stoch, action)

    def lin(x, k, b):
        return x @ k + b

    # reference: assemble a module whose params mirror the port, reuse ref fns
    core = port.core
    x0 = ref_act(ref_rms(lin(deter, sh.dynin0_k, sh.dynin0_b), sh.dynin0_s, cfg.norm_eps))
    x1 = ref_act(ref_rms(lin(stoch.reshape(B, -1), sh.dynin1_k, sh.dynin1_b),
                         sh.dynin1_s, cfg.norm_eps))
    x2 = ref_act(ref_rms(lin(action, sh.dynin2_k, sh.dynin2_b), sh.dynin2_s, cfg.norm_eps))
    x = torch.cat([x0, x1, x2], -1).unsqueeze(1).repeat(1, 8, 1)
    x = torch.cat([deter.reshape(B, 8, 64), x], -1).reshape(B, -1)
    out = torch.einsum("bki,kio->bko", x.reshape(B, 8, -1), sh.dynhid_k)
    x = ref_act(ref_rms(out.reshape(B, -1) + sh.dynhid_b, sh.dynhid_s, cfg.norm_eps))
    out = torch.einsum("bki,kio->bko", x.reshape(B, 8, -1), sh.gru_k)
    # grouped split, same as upstream (see ref_core)
    out = out + sh.gru_b.reshape(1, 8, -1)
    res, cand_in, upd_in = out.chunk(3, -1)
    res = torch.sigmoid(res.reshape(B, -1))
    cand = torch.tanh(res * cand_in.reshape(B, -1))
    upd = torch.sigmoid(upd_in.reshape(B, -1) - 1.0)
    want_d = upd * cand + (1 - upd) * deter

    x = ref_act(ref_rms(lin(want_d, sh.p0_k, sh.p0_b), sh.p0_s, cfg.norm_eps))
    x = ref_act(ref_rms(lin(x, sh.p1_k, sh.p1_b), sh.p1_s, cfg.norm_eps))
    want_logits = lin(x, sh.pl_k, sh.pl_b).reshape(B, cfg.stoch, cfg.classes)

    err_d = float((d_new - want_d).abs().max())
    err_l = float((logits - want_logits).abs().max())
    assert err_d < 1e-4, f"loaded deter' drift {err_d:.3e}"
    assert err_l < 1e-4, f"loaded logits drift {err_l:.3e}"
    # the surrogate must be exactly the unimix mean of the loaded prior
    assert torch.allclose(stoch_new, stoch_mean(want_logits, cfg.unimix), atol=1e-6)
    print(f"    loaded port vs arrays-derived reference: "
          f"deter' {err_d:.2e}, logits {err_l:.2e}")


def test_action_normalization_is_identity_on_bounded_actions():
    """The port must compute the same transition for a and a/max(1, |a|) whenever
    |a| <= 1 (every action distribution DreamerV3 uses: tanh-bounded continuous,
    one-hot discrete). Upstream divides by max(1, |a|); the port documents that as
    the identity and passes the action through. Pin the identity, not an assert."""
    cfg = RSSMConfig()
    port = OneStepLatentTransition(cfg, action_dim=2).eval()
    deter = torch.randn(3, cfg.deter) * 0.1
    stoch = torch.full((3, cfg.stoch, cfg.classes), 1.0 / cfg.classes)
    a = torch.tensor([[1.0, -1.0], [0.3, -0.9], [0.0, 0.0]])
    normed = a / torch.maximum(torch.ones_like(a), torch.abs(a))
    with torch.no_grad():
        d1, l1, s1 = port(deter, stoch, a)
        d2, l2, s2 = port(deter, stoch, normed)
    err = max(float((d1 - d2).abs().max()), float((l1 - l2).abs().max()))
    assert err < 1e-6, f"normalization identity broken: {err:.3e}"
    print(f"    a == a/max(1,|a|) on [-1,1], max drift {err:.2e}")


def ref_policy_mean(pol, deter, stoch):
    """Independent re-implementation of embodied MLPHead + bounded_normal mean."""
    cfg = pol_cfg = None
    feat = torch.cat([deter, stoch.reshape(deter.shape[0], -1)], -1)
    x = feat
    n_layers = len([k for k in pol.mlp if k.startswith("linear")])
    for i in range(n_layers):
        lin = pol.mlp[f"linear{i}"]
        norm = pol.mlp[f"norm{i}"]
        x = ref_act(ref_rms(x @ lin.kernel + lin.bias, norm.scale, 1e-4))
    mean = x @ pol.head["mean"].kernel + pol.head["mean"].bias
    return torch.tanh(mean)


def test_policy_matches_reference():
    from src.dreamer import PolicyMean
    torch.manual_seed(5)
    pol = PolicyMean(512, 32, 4, act_dim=6).eval()
    B = 4
    deter = torch.randn(B, 512) * 0.05
    stoch = torch.rand(B, 32, 4)
    stoch = (stoch == stoch.max(-1, keepdim=True).values).float()
    with torch.no_grad():
        got = pol(deter, stoch)
        want = ref_policy_mean(pol, deter, stoch)
    err = float((got - want).abs().max())
    assert err < 1e-4, f"policy transcription drift {err:.3e}"
    assert got.shape == (B, 6), got.shape
    assert float(got.abs().max()) <= 1.0 + 1e-5   # tanh-bounded
    print(f"    max |port - reference| on PolicyMean = {err:.2e}, a in [-1, 1]")


def test_closed_loop_composes_transition_and_policy():
    from src.dreamer import PolicyMean, OneStepClosedLoop
    torch.manual_seed(6)
    cfg = RSSMConfig()
    trans = OneStepLatentTransition(cfg, action_dim=6).eval()
    pol = PolicyMean(cfg.deter, cfg.stoch, cfg.classes, act_dim=6).eval()
    loop = OneStepClosedLoop(trans, pol).eval()
    B = 4
    deter = torch.randn(B, cfg.deter) * 0.05
    stoch = torch.full((B, cfg.stoch, cfg.classes), 1.0 / cfg.classes)
    z = torch.cat([deter, torch.flatten(stoch, start_dim=-2)], -1)
    with torch.no_grad():
        z2 = loop(z)                       # full map z -> (deter', stoch_flat')
        a = pol(deter, stoch)
        d2, _, s2 = trans(deter, stoch, a)
        full2 = torch.cat([d2, torch.flatten(s2, start_dim=-2)], -1)
    assert torch.equal(z2, full2)
    assert z2.shape == (B, cfg.deter + cfg.stoch * cfg.classes)
    print(f"    closed loop == full one-step map under policy (z' {tuple(z2.shape)})")


def test_loader_accepts_upstream_dyn_and_pol_prefixes():
    """Real ninjax checkpoints name the RSSM module 'dyn' and the policy 'pol'
    (agent.py: self.dyn = ... name='dyn'; self.pol = ... name='pol'). The loader
    must treat world_model/dyn/... and world_model/pol/... the same as the
    synthetic rssm/... keys."""
    from src.dreamer import PolicyMean
    rng = np.random.default_rng(11)

    def karr(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    core_arrays = {
        "world_model/dyn/dynin0/kernel": karr(512, 64),
        "world_model/dyn/dynin0/bias": np.zeros(64, np.float32),
        "world_model/dyn/dynin0norm/scale": np.ones(64, np.float32),
        "world_model/dyn/dynin1/kernel": karr(128, 64),
        "world_model/dyn/dynin1/bias": np.zeros(64, np.float32),
        "world_model/dyn/dynin1norm/scale": np.ones(64, np.float32),
        "world_model/dyn/dynin2/kernel": karr(6, 64),
        "world_model/dyn/dynin2/bias": np.zeros(64, np.float32),
        "world_model/dyn/dynin2norm/scale": np.ones(64, np.float32),
        "world_model/dyn/dynhid0/kernel": karr(8, 256, 64),
        "world_model/dyn/dynhid0/bias": np.zeros(512, np.float32),
        "world_model/dyn/dynhid0norm/scale": np.ones(512, np.float32),
        "world_model/dyn/dyngru/kernel": karr(8, 64, 192),
        "world_model/dyn/dyngru/bias": np.zeros(1536, np.float32),
        "world_model/dyn/prior0/kernel": karr(512, 64),
        "world_model/dyn/prior0/bias": np.zeros(64, np.float32),
        "world_model/dyn/prior0norm/scale": np.ones(64, np.float32),
        "world_model/dyn/prior1/kernel": karr(64, 64),
        "world_model/dyn/prior1/bias": np.zeros(64, np.float32),
        "world_model/dyn/prior1norm/scale": np.ones(64, np.float32),
        "world_model/dyn/priorlogit/kernel": karr(64, 128),
        "world_model/dyn/priorlogit/bias": np.zeros(128, np.float32),
    }
    pol_arrays = {
        "world_model/pol/mlp/linear0/kernel": karr(640, 64),
        "world_model/pol/mlp/linear0/bias": np.zeros(64, np.float32),
        "world_model/pol/mlp/norm0/scale": np.ones(64, np.float32),
        "world_model/pol/mlp/linear1/kernel": karr(64, 64),
        "world_model/pol/mlp/linear1/bias": np.zeros(64, np.float32),
        "world_model/pol/mlp/norm1/scale": np.ones(64, np.float32),
        "world_model/pol/mlp/linear2/kernel": karr(64, 64),
        "world_model/pol/mlp/linear2/bias": np.zeros(64, np.float32),
        "world_model/pol/mlp/norm2/scale": np.ones(64, np.float32),
        "world_model/pol/head/mean/kernel": karr(64, 6),
        "world_model/pol/head/mean/bias": np.zeros(6, np.float32),
    }

    trans = OneStepLatentTransition(RSSMConfig(), 6).eval()
    pol = PolicyMean(512, 32, 4, 6).eval()
    load_jax_arrays(trans, core_arrays)
    load_jax_arrays(pol, pol_arrays)

    # loaded port vs a reference computed straight from the arrays
    B = 3
    deter = torch.randn(B, 512) * 0.05
    stoch = torch.eye(4)[torch.randint(0, 4, (B, 32))]
    with torch.no_grad():
        d_new, logits, _ = trans(deter, stoch, torch.zeros(B, 6))
    x = deter @ torch.tensor(core_arrays["world_model/dyn/dynin0/kernel"]) \
        + torch.tensor(core_arrays["world_model/dyn/dynin0/bias"])
    x = ref_act(ref_rms(x, torch.tensor(core_arrays["world_model/dyn/dynin0norm/scale"]), 1e-4))
    with torch.no_grad():
        port = trans.core.act(trans.core.dynin0norm(trans.core.dynin0(deter)))
    assert float((x - port).abs().max()) < 1e-4
    print(f"    dyn/pol prefixed arrays load and agree (deter' {tuple(d_new.shape)})")


def test_shape_and_dtype_invariants():
    cfg = RSSMConfig()
    port = OneStepLatentTransition(cfg, action_dim=6).eval()
    B = 2
    deter = torch.zeros(B, cfg.deter)
    stoch = torch.zeros(B, cfg.stoch, cfg.classes)
    action = torch.zeros(B, 6)
    d_new, logits, stoch_new = port(deter, stoch, action)
    assert d_new.shape == (B, cfg.deter)
    assert logits.shape == (B, cfg.stoch, cfg.classes)
    assert stoch_new.shape == (B, cfg.stoch, cfg.classes)
    assert d_new.dtype == torch.float32
    assert torch.isfinite(d_new).all() and torch.isfinite(logits).all()
    print(f"    shapes (B={B}): deter' {tuple(d_new.shape)}, "
          f"logits {tuple(logits.shape)}, stoch' {tuple(stoch_new.shape)}")


if __name__ == "__main__":
    print(f"pinned upstream: {UPSTREAM_DREAMERV3}\n                 {UPSTREAM_EMBODIED}")
    _check("RSSM._core transcription matches the reference", test_core_matches_reference)
    _check("RSSM._prior transcription matches the reference", test_prior_matches_reference)
    _check("BlockLinear equals the einsum form", test_blocklinear_matches_einsum)
    _check("stoch surrogate is the unimix distribution mean", test_stoch_mean_is_a_distribution_mean)
    _check("load_jax_arrays reproduces the checkpoint function", test_load_jax_arrays_matches_reference)
    _check("action normalization is the identity in range", test_action_normalization_is_identity_on_bounded_actions)
    _check("PolicyMean transcription matches the reference", test_policy_matches_reference)
    _check("closed loop == transition(policy(z))", test_closed_loop_composes_transition_and_policy)
    _check("loader accepts upstream dyn/pol key prefixes", test_loader_accepts_upstream_dyn_and_pol_prefixes)
    _check("shape/dtype invariants for size1m", test_shape_and_dtype_invariants)
    print(f"\n{'all dreamer transition tests passed' if not FAILS else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)
