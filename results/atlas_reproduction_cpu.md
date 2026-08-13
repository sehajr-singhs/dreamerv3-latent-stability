# Five-arm failure atlas — verified CPU reproduction

Reproduced 2026-08-13 on CPU (Intel i7-1280P, 4 OpenMP threads, fixed seed 0 per
arm, 250 training steps, horizon curriculum K = 1 -> 3 -> 6, 200-step eval,
Lorenz-63 sigma=10 rho=28 beta=8/3 at dt=0.01, nonlinear warp, latent dim 64,
batch 128). Reference: empirical true lambda1 = +0.884 (Benettin, textbook
0.9056); true divergence div f = -13.6667.

| Arm | finite | mratio | lam_model | lam_err | contract | \|d-13.7\| | tr(R)/n | chamfer |
|---|---|---|---|---|---|---|---|---|
| B/UNCONSTRAINED-MLP | yes | 538.671 | +26.570 | 25.686 | -18.370 | 4.704 | n/a | 8923.64 |
| C/RIGID-HAMILTONIAN | yes | 0.005 | +0.001 | 0.883 | +0.008 | 13.675 | n/a | 3.50 |
| D/METRIPLECTIC-NAIVE | yes | 0.814 | +0.711 | 0.173 | -0.514 | 13.152 | 0.0000 | 4.42 |
| E/METRIPLECTIC-FIXED | yes | 1.559 | +2.244 | 1.360 | -13.651 | 0.016 | 4.1825 | 22.34 |
| F/SPECTRAL-ALIGNED | yes | 1.928 | +2.831 | 1.947 | -13.557 | 0.110 | 4.4482 | 24.43 |

How each documented failure mode reproduces:

1. **Representation freeze (C).** mratio 0.005, lam +0.001: the rigid Hamiltonian
   does not move and carries no chaotic divergence, yet posts the *best* chamfer
   (3.50) — the overlap trap, confirmed: a frozen predictor scores near-perfect
   attractor overlap while dynamically dead.
2. **Scale-conditioning paradox (C, training).** raw kinetic proxy spread ~0.036
   vs integrator demand ~3.2 — magnitude-matching clamps the dynamics (the
   z-score anchor is the documented fix; not part of this run's C arm).
3. **Dead R saddle (D).** tr(R)/n = 0.0000 with R = LL^T zero-init; no
   dissipation engages (contract -0.514 vs -13.667).
4. **Sum-vs-lambda1 decoupling (E).** the divergence gate matches the volume law
   (|d-13.7| = 0.016) while lam1 runs +2.244 vs true +0.884 (2.5x high): a
   scalar volume constraint pins the sum, not the leading exponent.
5. **Finite-time spectral trap (F).** the differentiable QR proxy converged to
   lam1_hat ~ -0.16 during training while the asymptotic 200-step lam1 is +2.831
   (opposite sign, 2.5x high): shaping the short-horizon proxy does not move the
   real exponent. R engages (4.4482) and the volume law matches (-13.557), which
   is exactly what a sum constraint can do — and no more.

Honest framing (unchanged from the study): Lorenz is not a natural GENERIC
system; H, S, R here are learned proxies. We claim on-attractor stability +
invariant recovery, NOT recovery of true thermodynamics. Single seed per arm,
250 steps — the documented signature values (e.g. E contraction -13.667,
lambda1 +3.5) are from the study's own schedule; this CPU reproduction confirms
the qualitative atlas with its own numbers.
