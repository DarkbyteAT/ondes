# ondes — Design Decisions

A short, citable record of the design decisions that shape the public surface of `ondes`. Adopted 2026-05-16 after a five-voice design debate (see `debate/` for position documents and reasoning trails).

## The verdict

`ondes` ships only what `ondes` itself uses internally, plus the escape hatches needed for downstream composition.

**Library public surface:**

- `SIREN`, `HSIREN`, `WIRE` — basis-MLP trunks (`eqx.Module`, callable).
- `Encoding` family and factories — `Gaussian`, `Dyadic`, `NO_ENCODING`, `gaussian_fixed`, `gaussian_from_shape`, `gaussian_learn`, `dyadic`.
- `siren_init`, `nyquist_sigma` — init primitives `ondes` consumes internally.
- `inr.trunk(coord, **kw)` — public on every basis module; returns pre-readout features.
- `out_features` constructor kwarg on every basis class; `None` (default) ⇒ scalar return from `__call__`; integer `N > 1` ⇒ `(N,)` vector return. `out_features=1` is canonicalised to `None` at construction so the two scalar-return forms produce identical pytrees — relevant for serialisation, jit caching, and tree-equality checks.
- FiLM modulation via `inr(coord, film=...)`.

**Not in `ondes` (by design):**

- No `Head` value type.
- No `head=` constructor kwarg.
- No `ondes.heads` module — no prebuilt distribution wrappers, no Gram-Schmidt helpers, no `normal_params`, no `softplus_scale`, nothing of the kind.
- No bundled `distreqx` integration.

Composition between an `inr` and a head/distribution/parameterisation lives entirely in user code, typically as a small `Model(eqx.Module)` wrapper.

## Why no heads

Five independent rationales arrived at the same verdict:

1. **No neutral parameterisation exists** (data-scientist's literature evidence). NeRF density uses `softplus`; instant-NGP uses `exp(σ - 1)`; VolSDF uses Laplace CDF. VAE σ uses `exp(0.5·log_var)`; NeRF-W uses softplus with learnable floor. 3D rotations are parameterised four ways in active use (Gram-Schmidt, SVD, quaternion, axis-angle), none dominant. A library that ships any single default takes a methodological side without consulting its users.

2. **Heads are post-trunk** (staff-architect's structural observation). Anything with real coupling to `ondes`'s primitives (omega-aware readouts, spectral inits) is *already* in the trunk. Heads operate on post-readout features, which are basis-independent, which means user-owned by definition. The heads namespace is empty *by construction*, not by maintainer preference.

3. **Heads are port-fragile, trunks are port-survivable** (ml-engineer's MLOps argument). Every cross-language port of an INR-based system rewrites the head from scratch because heads encode distribution choices, loss conventions, and downstream-task postprocessing. The trunk — basis init, omega scheduling, encoding spectra — is the expensive-to-re-derive part. Shipping heads optimises the cheap-and-portable side at the expense of the discipline that keeps the expensive-and-fragile side correct.

4. **Cohort norm** (ml-engineer's empirical check). `samgria`, `rltrain`, `xptrack` export zero curated-convenience namespaces. `ondes.heads` would be the first such namespace in the cohort and would set the opposite gravity from the one explicitly chosen elsewhere.

5. **Reversibility asymmetry** (advocate's concession). Adding `head=` and `ondes.heads.X` later is a trivial additive change. Removing them later is a breaking change. Start at the smaller surface; grow only against demonstrated need.

## The three-AND-gate (for any future addition)

A function belongs in `ondes` only if **all three** hold:

1. **Spectral / init coupling** — its design depends on the basis or on input spectral properties. A function whose implementation has no dependence on `ondes`'s own internals is by definition user-side composition.
2. **Closed-form trap** — a competent naive implementation would plausibly be wrong (numerical stability, degenerate cases, non-obvious identity). One-line wrappers fail this gate.
3. **Multi-consumer with no plausible variant on the horizon** — the same shape is consumed by ≥2 of `{ondes, loom, samgria-jax, xptrack-jax-hooks}` *and* there's no actively-competing parameterisation in the literature that would compete for the same name. Present convergence alone is not enough; future stability is the second clause, because today's two consumers can diverge tomorrow. `gram_schmidt` fails this clause because the 4D/9D/3D variants are always on the horizon.

Applied to today's candidate set (`normal_params`, `unit_vector`, `complex_split`, `softplus_scale`, `gram_schmidt`, `mog_params`, `categorical_logits`, `lognormal_params`), zero survive. The surviving `ondes` primitives (`siren_init`, `nyquist_sigma`) already live in the library.

The gate is permanent policy. Promotion-on-evidence; no ship-just-in-case.

## Where head-shaped recipes live

A three-tier risk ranking, applied in reverse order (lowest risk first):

1. **`examples/*.py`** — self-contained CI-tested scripts demonstrating one paper or one task. Free to import `distreqx`, `equinox`, anything. Inline head helpers MUST name the parameterisation choice and cite at least one alternative in active use:
   ```python
   # We use softplus(σ) + ε here.
   # NeRF-W uses softplus with learnable floor.
   # Classical VAEs use exp(0.5 * log_var).
   # Pick the one your loss expects.
   ```
   CI is the anti-graduation enforcement: examples that don't run break the build, so they can't rot silently. Parameterisation changes change what the example demonstrates, so they don't drift either. Tooling beats convention for sustained discipline — a README appendix code block can lie, rot, or canonise silently; a CI-tested example can do none of those.

2. **README load-bearing snippet** — exactly one parameterisation-agnostic composition pattern showing the `(inr, head) Model(eqx.Module)` shape. Points at `examples/` for concrete recipes. The README is an index, not a parallel home.

3. **`ondes.heads.X`** — does not exist and should not exist. Any proposal to add it is gated by the three-AND-gate above.

## What this means for contributors

- A PR adding a "small convenience function" to the public namespace should expect to be asked which of the three gates it clears. If the answer is two-of-three, the PR lands as an `examples/` script, not as a library symbol.
- A PR adding inline content to the README that isn't the load-bearing snippet should expect to be asked why it isn't an `examples/` script instead.
- A PR adding a `distreqx` (or any distribution-library) dependency to `ondes` should expect to be asked which `ondes` internal requires it. If none does, the dependency stays in `examples/`.

## See also

- `debate/advocate-position.md`, `debate/pragmatist-position.md`, `debate/staff-architect-position.md`, `debate/data-scientist-position.md`, `debate/ml-engineer-position.md` — full reasoning trails for the verdict above.
- `CONTRIBUTING.md` — the three-AND-gate restated in PR-review language.
