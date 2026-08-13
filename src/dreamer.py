"""Faithful torch port of DreamerV3's ONE-STEP latent transition (RSSM).

Scope, exactly as locked in CLAUDE.md: the one-step latent transition only, at the
smallest official model size. Not the encoder, not the decoder, not the imagined
rollout. The claim this leg can support is about the map

    (deter_t, stoch_t, a_t)  ->  deter_{t+1}  and  logits(prior at t+1),

i.e. `RSSM._core` followed by `RSSM._prior` / `RSSM._logit`. Everything else in
DreamerV3 (posterior, reconstruction, multi-step imagination) compounds GRU product
nodes and straight-through categoricals and is expected to return `unknown` under
CROWN; that is a documented scaling boundary, not a headline.

Transcription provenance (every formula below is quoted against these):
- github.com/danijar/dreamerv3, dreamerv3/rssm.py  @ e3f02248693a79dc8b0ebd62c93683888ddaccfe
- github.com/danijar/embodied, embodied/jax/nets.py @ a6583c14123ad310a3f20a6e78b5a0983d24a4fb
- model size: dreamerv3/configs.yaml `size1m` (the smallest real config; used by the
  paper for dmc_proprio), plus the `dyn.rssm` defaults it inherits.

This port is a re-implementation in torch for auto_LiRPA, not a line-for-line JAX
copy. Every deviation is deliberate and documented where it happens. The two that
matter:

1. auto_LiRPA 0.7.2 has no SiLU bound, so silu is written as the equivalent
   primitive composition x / (1 + exp(-x)). Same function, boundable.
2. `BlockLinear` is expanded to a block-diagonal linear so the verifier sees one
   BoundLinear instead of an einsum it cannot bound. Same function, faster and
   tighter bounds.

Parameter names match upstream value names (`kernel`, `bias`, `scale`), so a JAX
checkpoint exports to this port by key, with no renaming table. `load_jax_arrays`
below implements exactly that.

Actions: upstream `_core` does `action /= sg(jnp.maximum(1, |action|))`. For every
action distribution DreamerV3 actually uses this is the identity: continuous
actions are tanh-bounded to [-1, 1] and discrete actions are one-hot 0/1, so
`max(1, |a|) == 1` always. The port therefore asserts the bound and passes the
action through, and the certificate describes the model exactly because on the
reachable action range the upstream op IS the identity.

The deterministic latent used for certification is the MEAN of the prior
categorical: (1 - unimix) * softmax(logits) + unimix / classes. The real model
samples a one-hot from this distribution during imagination; the audit certifies
the deterministic mean map, stated explicitly in every artifact, exactly as the
Pendulum leg certifies the deterministic closed loop rather than the stochastic
environment.
"""

import dataclasses
import re
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pinned upstream. If you update the port, update these and re-run the cross-checks.
UPSTREAM_DREAMERV3 = (
    "github.com/danijar/dreamerv3 @ e3f02248693a79dc8b0ebd62c93683888ddaccfe "
    "(dreamerv3/rssm.py)"
)
UPSTREAM_EMBODIED = (
    "github.com/danijar/embodied @ a6583c14123ad310a3f20a6e78b5a0983d24a4fb "
    "(embodied/jax/nets.py)"
)


@dataclasses.dataclass(frozen=True)
class RSSMConfig:
    """The RSSM hyperparameters that change the forward function.

    Defaults are DreamerV3's `size1m` config (configs.yaml) — the smallest official
    model size — plus the `dyn.rssm` defaults it inherits. This is the locked scope:
    the smallest model, and nothing bigger is claimed.
    """
    deter: int = 512
    hidden: int = 64
    stoch: int = 32
    classes: int = 4
    blocks: int = 8
    dynlayers: int = 1
    imglayers: int = 2
    norm_eps: float = 1e-4
    unimix: float = 0.01
    act: str = "silu"
    norm: str = "rms"


class BoundableSiLU(nn.Module):
    """silu(x) = x * sigmoid(x), written with boundable primitives.

    auto_LiRPA 0.7.2 has no SiLU op. The naive rewrite x / (1 + exp(-x)) is
    mathematically exact, but its exp relaxation overflows to inf slopes once the
    pre-activation interval is wide (which the GRU's product nodes guarantee), and
    CROWN then poisons every downstream A matrix with NaN. The tanh form

        x * (0.5 + 0.5 * tanh(x / 2))

    is IDENTICALLY equal (sigmoid(x) = 0.5 + 0.5*tanh(x/2)) and stays finite
    everywhere: tanh's S-shaped relaxation is bounded on all of R. The remaining
    elementwise product is a product node — loose, but never NaN. Same philosophy
    as cnl-work's BoundableELU: the graph that trains is the graph that verifies.
    """

    def forward(self, x):
        return x * (0.5 + 0.5 * torch.tanh(x / 2.0))


class RMSNorm(nn.Module):
    """Embodied `Norm(impl='rms')`: x * scale / sqrt(mean(x^2) + eps).

    Upstream (embodied/jax/nets.py):

        mean2 = jnp.square(x).mean(axis, keepdims=True)   # axis = (-1,)
        x = x * (jax.lax.rsqrt(mean2 + self.eps) * scale) # eps default 1e-4

    `scale` is a learned per-feature vector named `scale` to match upstream. All
    pieces (square, mean, sqrt, div, mul) are boundable primitives.
    """

    def __init__(self, units: int, eps: float = 1e-4):
        super().__init__()
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(units))

    def forward(self, x):
        mean2 = x.square().mean(-1, keepdim=True)
        # rsqrt(s) written as exp(-0.5 * log(s)). Two reasons, both auto_LiRPA
        # 0.7.2 facts: there is no BoundRsqrt, and BoundDiv.forward contains an
        # ad-hoc LayerNorm special case that misfires on ANY `x / sqrt(...)`
        # pattern (it walks inputs[0].inputs[0] and rebuilds a layer-norm-style
        # deviation, so the div output gets the WRONG shape and value). exp and
        # log both have proper convex/concave relaxations. Same function as
        # x / sqrt(mean2 + eps); log's input is always >= eps > 1e-6, so the
        # clamp in BoundLog.forward never engages.
        rsqrt = torch.exp(-0.5 * torch.log(mean2 + self.eps))
        return x * rsqrt * self.scale


class Linear(nn.Module):
    """Embodied `Linear` with upstream layout: kernel (in, out), bias (out,).

    Parameter names match upstream (`kernel`, `bias`) so JAX checkpoints load with
    no renaming. torch stores weights as (out, in); the transpose happens in
    forward and auto_LiRPA folds it because the weight is a constant.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.trunc_normal_(self.kernel, std=0.02)

    def forward(self, x):
        out = F.linear(x, self.kernel.t(), self.bias)
        return out


class BlockLinear(nn.Module):
    """Embodied `BlockLinear` expanded to a block-diagonal linear.

    Upstream keeps a grouped kernel of shape (g, in_per_group, out_per_group) and
    computes einsum('...ki,kio->...ko'). A grouped affine map is exactly a
    block-diagonal matrix, so this port precomputes the (units, insize) block
    diagonal once and runs one F.linear. auto_LiRPA sees a single BoundLinear:
    faster and tighter than g separate matmuls.

    `kernel` keeps the upstream (g, in_per_group, out_per_group) layout so
    checkpoints load untouched; `sync()` rebuilds the block-diagonal weight buffer
    from it. Call `sync()` after loading weights (load_jax_arrays does).
    """

    def __init__(self, in_features: int, units: int, blocks: int, bias: bool = True):
        super().__init__()
        assert units % blocks == 0 and in_features % blocks == 0, (
            in_features, units, blocks)
        self.blocks = blocks
        self.in_per_group = in_features // blocks
        self.out_per_group = units // blocks
        self.kernel = nn.Parameter(
            torch.empty(blocks, self.in_per_group, self.out_per_group))
        self.bias = nn.Parameter(torch.zeros(units)) if bias else None
        self.register_buffer(
            "_w_blockdiag", torch.zeros(units, in_features), persistent=False)
        nn.init.trunc_normal_(self.kernel, std=0.02)
        self.sync()

    def sync(self):
        """Rebuild the block-diagonal weight buffer from `kernel`. Must be called
        after any direct assignment to `kernel` (e.g. loading JAX weights)."""
        # block_diag stacks the (in_per_group, out_per_group) blocks into
        # (in_total, out_total); F.linear wants (out, in), so transpose.
        # The copy MUST be under no_grad: in torch >= 2.11, copy_ from a tensor
        # that requires grad flips requires_grad on the buffer, and auto_LiRPA's
        # jit tracer then rejects it ("Cannot insert a Tensor that requires grad
        # as a constant").
        blocks = [self.kernel[i] for i in range(self.blocks)]
        with torch.no_grad():
            self._w_blockdiag.copy_(torch.block_diag(*blocks).t().contiguous())

    def forward(self, x):
        return F.linear(x, self._w_blockdiag, self.bias)


class RSSMCore(nn.Module):
    """DreamerV3 `RSSM._core`: one deterministic GRU step in latent space.

    Transcribed from upstream rssm.py (see module docstring for provenance):

        x0 = act(norm(Linear(deter)))                  # dynin0
        x1 = act(norm(Linear(stoch_flat)))             # dynin1
        x2 = act(norm(Linear(action)))                 # dynin2
        x  = concat([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
        x  = group2flat(concat([flat2group(deter), x], -1))
        for i in range(dynlayers):                     # dynhid{i}
            x = act(norm(BlockLinear(x, deter, g)))
        x  = BlockLinear(x, 3 * deter, g)              # dyngru
        reset, cand, update = split(x, 3, -1)
        reset  = sigmoid(reset)
        cand   = tanh(reset * cand)
        update = sigmoid(update - 1)                   # -1 is the update bias
        deter' = update * cand + (1 - update) * deter

    `stoch` arrives as (B, stoch, classes) and is flattened to (B, stoch*classes)
    before dynin1, exactly as upstream (`stoch.reshape((stoch.shape[0], -1))`).
    """

    def __init__(self, cfg: RSSMConfig, action_dim: int):
        super().__init__()
        self.cfg = cfg
        self.g = cfg.blocks
        g, h, d = cfg.blocks, cfg.hidden, cfg.deter
        assert d % g == 0
        # act(norm(Linear(...))) for the three dynin inputs
        self.dynin0 = Linear(d, h)
        self.dynin0norm = RMSNorm(h, cfg.norm_eps)
        self.dynin1 = Linear(cfg.stoch * cfg.classes, h)
        self.dynin1norm = RMSNorm(h, cfg.norm_eps)
        self.dynin2 = Linear(action_dim, h)
        self.dynin2norm = RMSNorm(h, cfg.norm_eps)
        # dynhid{i} input: per-group deter/g + 3*hidden (the three dynin features
        # concatenated), so the flat width is g*(deter/g + 3*hidden). Its OUTPUT is
        # width deter, and that output feeds dyngru (upstream: the dyngru call is
        # applied to the dynhid output), so dyngru's input width is deter, not din.
        # ModuleDict (not ModuleList) so parameter names are 'dynhid0.kernel',
        # matching upstream value names exactly and making checkpoint loading a
        # pure name match.
        din = g * (cfg.deter // g + 3 * h)
        self.dynhids = nn.ModuleDict(
            {f"dynhid{i}": BlockLinear(din, d, g) for i in range(cfg.dynlayers)})
        self.dynhidnorms = nn.ModuleDict(
            {f"dynhid{i}norm": RMSNorm(d, cfg.norm_eps)
             for i in range(cfg.dynlayers)})
        self.dyngru = BlockLinear(d, 3 * d, g)
        self.act = BoundableSiLU()

    def forward(self, deter, stoch, action):
        cfg, g = self.cfg, self.g
        # upstream: stoch.reshape((stoch.shape[0], -1))
        stoch_flat = stoch.reshape(*stoch.shape[:-2], -1)

        x0 = self.act(self.dynin0norm(self.dynin0(deter)))
        x1 = self.act(self.dynin1norm(self.dynin1(stoch_flat)))
        x2 = self.act(self.dynin2norm(self.dynin2(action)))
        # concat + expand to g groups: (..., 3h) -> (..., g, 3h)
        x = torch.cat([x0, x1, x2], -1).unsqueeze(-2).expand(*x0.shape[:-1], g, -1)
        # group2flat(concat([flat2group(deter), x], -1))
        per_group = cfg.deter // g
        deter_g = deter.reshape(*deter.shape[:-1], g, per_group)
        x = torch.cat([deter_g, x], -1).reshape(
            *deter.shape[:-1], g * (x.shape[-1] + per_group))
        for i in range(cfg.dynlayers):
            x = self.act(self.dynhidnorms[f"dynhid{i}norm"](
                self.dynhids[f"dynhid{i}"](x)))
        x = self.dyngru(x)
        # split into reset / cand / update gates, each (..., deter)
        res, cand, upd = x.reshape(*x.shape[:-1], g, -1).chunk(3, dim=-1)
        res = torch.sigmoid(res.reshape(*x.shape[:-1], cfg.deter))
        # upstream: cand = jnp.tanh(reset * cand) -- the sigmoided reset gates the
        # candidate. Dropping that factor is a real transcription error, caught by
        # tests/test_dreamer_transition.py; keep the multiplication.
        cand = torch.tanh(res * cand.reshape(*x.shape[:-1], cfg.deter))
        upd = torch.sigmoid(upd.reshape(*x.shape[:-1], cfg.deter) - 1.0)
        return upd * cand + (1.0 - upd) * deter


class Prior(nn.Module):
    """DreamerV3 `RSSM._prior` + `RSSM._logit`: prior logits over the stoch latent.

    Upstream:

        x = deter
        for i in range(imglayers):                     # prior{i}
            x = act(norm(Linear(x, hidden)))
        return reshape(Linear(x, stoch * classes), (stoch, classes))   # priorlogit

    Note the order in the loop: Linear first, then norm, then act — the same order
    as dynin. The final logit layer carries upstream's `outscale` only at INIT
    time; the forward is a plain linear, so no outscale appears here.
    """

    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg
        # ModuleDict again: upstream names are prior0/prior0norm/prior1/...
        self.priors = nn.ModuleDict(
            {f"prior{i}": Linear(cfg.deter if i == 0 else cfg.hidden, cfg.hidden)
             for i in range(cfg.imglayers)})
        self.priornorms = nn.ModuleDict(
            {f"prior{i}norm": RMSNorm(cfg.hidden, cfg.norm_eps)
             for i in range(cfg.imglayers)})
        self.priorlogit = Linear(cfg.hidden, cfg.stoch * cfg.classes)
        self.act = BoundableSiLU()

    def forward(self, deter):
        x = deter
        for i in range(self.cfg.imglayers):
            x = self.act(self.priornorms[f"prior{i}norm"](
                self.priors[f"prior{i}"](x)))
        logits = self.priorlogit(x)
        return logits.reshape(*logits.shape[:-1], self.cfg.stoch, self.cfg.classes)


def stoch_mean(logits, unimix: float):
    """Mean of DreamerV3's prior categorical distribution over stoch.

    Upstream `OneHot` mixes the softmax with a uniform by `unimix`, then `Agg`
    sums over the class axis:

        p = (1 - unimix) * softmax(logits) + unimix / classes

    Written with exp/sum/div primitives (no max-subtraction) so auto_LiRPA can
    bound it. Slightly less numerically stable than logsumexp form; irrelevant at
    the magnitudes CROWN reports, and exactness of the function is what matters.
    """
    e = torch.exp(logits)
    soft = e / e.sum(-1, keepdim=True)
    return (1.0 - unimix) * soft + unimix / logits.shape[-1]


class OneStepLatentTransition(nn.Module):
    """The locked claim object: one step of DreamerV3's latent dynamics.

        (deter, stoch, action)  ->  (deter', stoch')

    where deter' = RSSMCore(deter, stoch, action), logits' = Prior(deter'), and
    stoch' = mean of the prior categorical (see stoch_mean). Everything else about
    DreamerV3 is out of scope for a positive claim.

    `forward` returns (deter_new, logits, stoch_new) so an audit can bound either
    the raw logits or the deterministic surrogate.
    """

    def __init__(self, cfg: RSSMConfig, action_dim: int):
        super().__init__()
        self.cfg = cfg
        self.core = RSSMCore(cfg, action_dim)
        self.prior = Prior(cfg)

    def forward(self, deter, stoch, action):
        # Action normalization: upstream `action /= sg(max(1, |action|))` is the
        # identity for every distribution DreamerV3 uses (tanh-bounded continuous,
        # one-hot discrete), so the port passes the action through. The caller is
        # responsible for feeding actions in that range; feeding anything else
        # makes the port describe a different function than the model. (An assert
        # here would break auto_LiRPA's fx tracing, so it lives at the audit
        # level instead; the identity itself is pinned by a unit test.)
        deter_new = self.core(deter, stoch, action)
        logits = self.prior(deter_new)
        stoch_new = stoch_mean(logits, self.cfg.unimix)
        return deter_new, logits, stoch_new


class PolicyMean(nn.Module):
    """DreamerV3 actor, deterministic action only: a(z) = tanh(mean(mlp(feat))).

    Transcribed from dreamerv3/agent.py (`feat2tensor`, `self.pol = MLPHead(...)`)
    and embodied/jax/heads.py (`MLPHead`, `Head.bounded_normal`):

        feat = concat(deter, stoch_flat)                     # feat2tensor
        x    = MLP(feat)     # layers x (Linear(units) + RMSNorm + silu); the LAST
                             # layer also gets norm + act (embodied nets.MLP)
        mean = Linear(units -> act_dim)(x)   # outscale=0.01 affects INIT only
        a    = tanh(mean)                    # deterministic action of bounded_normal

    `outs.Agg` aggregates only loss/entropy terms, never `pred` or `sample`, so the
    deterministic action is exactly per-component tanh(mean) -- no summing over
    action dims. The deployed policy samples Normal(tanh(mean), stddev); the
    certified surrogate is the deterministic mean, stated in every artifact.

    Default widths match `size1m` (configs.yaml `.*.units: 64`, policy layers 3).
    """

    def __init__(self, deter_dim: int, stoch: int, classes: int, act_dim: int,
                 layers: int = 3, units: int = 64, norm_eps: float = 1e-4):
        super().__init__()
        feat_dim = deter_dim + stoch * classes
        # Module names match upstream so checkpoints load by key: the MLP submodule
        # is 'mlp' with linear{i}/norm{i}, the head is 'head' with sub 'mean'.
        self.mlp = nn.ModuleDict({
            **{f"linear{i}": Linear(feat_dim if i == 0 else units, units)
               for i in range(layers)},
            **{f"norm{i}": RMSNorm(units, norm_eps) for i in range(layers)},
        })
        self.head = nn.ModuleDict({"mean": Linear(units, act_dim)})
        self.act = BoundableSiLU()

    def forward(self, deter, stoch):
        feat = torch.cat([deter, torch.flatten(stoch, start_dim=-2)], -1)
        x = feat
        for i in range(len(self.mlp) // 2):
            x = self.act(self.mlp[f"norm{i}"](self.mlp[f"linear{i}"](x)))
        return torch.tanh(self.head["mean"](x))


class OneStepClosedLoop(nn.Module):
    """z' = (deter', stoch') = T(z, pi(z)): the FULL deterministic one-step latent
    transition under the frozen actor, as a map on the latent itself.

    Input is the single concatenated latent z = concat(deter, stoch_flat), because
    the audit's condition and certify_box pass one input/box (same as the pendulum
    ClosedLoop). The action comes from PolicyMean inside the graph.

    The output is the FULL next latent (deter', stoch_flat'), where stoch' is the
    mean of the prior categorical. The map is square (z -> z'), which the Lyapunov
    condition REQUIRES: a V on the full latent with the joint fixed point z* as
    its unique zero is the only structurally-sound object. A V on deter alone is a
    known trap -- on the slice {deter = deter*, stoch != stoch*} it has
    cond = -V(deter') < 0 by construction, manufacturing violations the model
    never commits (documented in lyapunov_latent).

    The stoch' OUTPUT is where branch-and-bound hits the guardrail-predicted wall:
    the softmax exp relaxation overflows once intermediate bounds are wide (the
    measured vacuity, d1_smoke). certify_box then returns unknown (or must be
    caught) -- that is category 3 verifier incompleteness, never a finding. The
    SAMPLING audit evaluates this same full map in float precision, where softmax
    is exact and cheap, so the empirical baseline is unaffected.
    """

    def __init__(self, trans: "OneStepLatentTransition", policy: PolicyMean):
        super().__init__()
        self.trans = trans
        self.policy = policy
        self.deter_dim = trans.cfg.deter

    def forward(self, z):
        deter = z[..., :self.deter_dim]
        stoch_flat = z[..., self.deter_dim:]
        stoch = stoch_flat.reshape(
            *stoch_flat.shape[:-1], self.trans.cfg.stoch, self.trans.cfg.classes)
        action = self.policy(deter, stoch)
        deter_new, _, stoch_new = self.trans(deter, stoch, action)
        return torch.cat([deter_new, torch.flatten(stoch_new, start_dim=-2)], -1)


def load_jax_arrays(module: nn.Module, arrays: Mapping[str, np.ndarray]) -> None:
    """Load upstream JAX parameter arrays into the port by key.

    `arrays` maps upstream parameter paths to arrays, e.g.
    "world_model/dyn/dynin0/kernel" -> (512, 64). Any prefix up to and including
    the module boundary ("/dyn/", "/rssm/", or "/pol/") is stripped; the
    remainder must equal the port's parameter name ("dynin0/kernel",
    "dynin0/bias", "dynin0norm/scale", "dynhid0/kernel", "mlp/linear0/kernel",
    "head/mean/kernel", ...).

    `kernel` arrays are (in, out) upstream; the port's `Linear` stores them
    transposed to (out, in) in the state_dict, so they are transposed here.
    `BlockLinear` kernels keep the upstream (g, in_per_group, out_per_group) layout
    and are copied as-is; `sync()` then rebuilds the block-diagonal buffers.

    After loading, the module is EXACTLY the frozen upstream function in fp32
    (upstream trains in bf16; the certificate describes the fp32 port, the same
    extraction-vs-library caveat the Pendulum leg already records).
    """
    named = dict(module.named_parameters())
    assigned = set()
    for key, arr in arrays.items():
        # 'world_model/dyn/dynin0/kernel' -> 'dynin0.kernel' (strip through the
        # module boundary), matched by SUFFIX so the loader works whether it is
        # handed the whole transition module ('core.'/'prior.' prefixes) or a bare
        # submodule. Boundary regex: the RSSM module is named 'dyn' upstream,
        # 'rssm' in the older test fixtures; the policy module is 'pol'.
        m = re.search(r"/(dyn|rssm|pol)/", key)
        if m:
            name = key[m.end():]
        else:
            name = key.split("/", 1)[-1] if "/" in key else key
        name = name.replace("/", ".")
        cands = [n for n in named if n.endswith(name)]
        if len(cands) != 1:
            raise KeyError(
                f"no unique parameter '{name}' in the port (from JAX key '{key}'; "
                f"matched {len(cands)})")
        full, target = cands[0], named[cands[0]]
        t = torch.as_tensor(np.asarray(arr, dtype=np.float32))
        if (target.dim() == 2 and name.endswith(".kernel")
                and t.shape != tuple(target.shape)):
            # upstream Linear kernel is (in, out); the port stores (out, in)
            t = t.t().contiguous()
        if tuple(t.shape) != tuple(target.shape):
            raise ValueError(
                f"shape mismatch for '{name}': JAX {tuple(t.shape)} vs port "
                f"{tuple(target.shape)}")
        with torch.no_grad():
            target.copy_(t)
        assigned.add(full)
    missing = set(named) - assigned
    if missing:
        raise ValueError(f"parameters never assigned from the checkpoint: "
                         f"{sorted(missing)}")
    for m in module.modules():
        if isinstance(m, BlockLinear):
            m.sync()
