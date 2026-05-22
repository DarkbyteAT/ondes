# ondes — Design Decisions

A short, citable record of the design decisions that shape the public surface of `ondes`. Adopted 2026-05-16 after a five-voice design debate (see `debate/` for position documents and reasoning trails).

## The verdict

`ondes` ships only what `ondes` itself uses internally, plus the escape hatches needed for downstream composition.

**Library public surface:**

- `Basis` — ABC for a single basis-MLP layer; exposes the shared linear + FiLM pre-activation and the abstract `_activate` hook subclasses override.
- `SIRENLayer`, `HSIRENLayer`, `WIRELayer` — concrete layer subclasses, one per basis family. Only `WIRELayer` carries the basis-specific learnable scalar `s`.
- `Body` — public base for basis-MLP trunks; exposes `trunk()`, `__call__`, the readout, FiLM dispatch, and `out_features` scalar/vector switch. Symmetric with `Basis` and `Encoding`. Not intended for external subclassing — new variants should subclass `SIREN`/`HSIREN`/`WIRE`, not `Body` directly.
- `SIREN`, `HSIREN`, `WIRE` — basis-MLP trunks (`eqx.Module`, callable, inherit `Body`). Each constructs the matching layer subclass. `WIRE.__init__` accepts an extra `s_init` kwarg; `SIREN` and `HSIREN` do not.
- `Encoding` — ABC for coord pre-encodings; exposes an `out_dim` property and `__call__(coord)` abstract method.
- `Identity`, `Gaussian`, `LearnedGaussian`, `Dyadic` — concrete encoding subclasses. Each is *operational* (materialises its own parameters at construction and acts as a callable `coord → embedded` map). The class itself is the constructor; there are no factory functions. `Gaussian` bakes `sigma` into the sampled `B` matrix at construction; `LearnedGaussian` carries `B_raw` + a learnable scalar `sigma` so optimisers update the spectral scale through gradients.
- `siren_init`, `nyquist_sigma` — init primitives `ondes` consumes internally.
- `inr.trunk(coord, **kw)` — public on every basis module; returns pre-readout features.
- `out_features` constructor kwarg on every basis class; `None` (default) ⇒ scalar return from `__call__`; integer `N > 1` ⇒ `(N,)` vector return. `out_features=1` is canonicalised to `None` at construction so the two scalar-return forms produce identical pytrees — relevant for serialisation, jit caching, and tree-equality checks.
- FiLM modulation via `inr(coord, film=...)`.

**Not in `ondes` (by design):**

- No `Head` value type.
- No `head=` constructor kwarg.
- No `ondes.heads` module — no prebuilt distribution wrappers, no Gram-Schmidt helpers, no `normal_params`, no `softplus_scale`, nothing of the kind.
- No bundled `distreqx` integration.
- No `kind: str` discriminators on any public type. Basis kind (SIREN/HSIREN/WIRE) and encoding kind (Identity/Gaussian/LearnedGaussian/Dyadic) are expressed as separate classes; consumers dispatch on `isinstance(...)` or, equivalently, on the type that comes out of the constructor. See §"Structural design — polymorphism over discriminators".
- No factory functions. The class IS the constructor — `Gaussian(rank=3, num_freqs=128, sigma=2.5, key=k)`, `LearnedGaussian(rank=3, num_freqs=128, key=k)`, `Dyadic(rank=3, num_bands=4)`, `Identity(in_dim=3)`.
- No `NO_ENCODING` singleton. Users construct `Identity(in_dim=...)` themselves; the in_dim is part of the encoding's interface (it's what `out_dim` reports).
- No `sigma_from_shape: Callable` field on `Gaussian`. The shape-rule pattern is a downstream concern (a renderer constructs one encoding per leaf with `sigma = nyquist_sigma(leaf.shape)`); ondes does not bake the rule into the encoding itself.

Composition between an `inr` and a head/distribution/parameterisation lives entirely in user code, typically as a small `Model(eqx.Module)` wrapper.

## Why no heads

Five independent rationales arrived at the same verdict:

1. **No neutral parameterisation exists** (data-scientist's literature evidence). NeRF density uses `softplus`; instant-NGP uses `exp(σ - 1)`; VolSDF uses Laplace CDF. VAE σ uses `exp(0.5·log_var)`; NeRF-W uses softplus with learnable floor. 3D rotations are parameterised four ways in active use (Gram-Schmidt, SVD, quaternion, axis-angle), none dominant. A library that ships any single default takes a methodological side without consulting its users.

2. **Heads are post-trunk** (staff-architect's structural observation). Anything with real coupling to `ondes`'s primitives (omega-aware readouts, spectral inits) is *already* in the trunk. Heads operate on post-readout features, which are basis-independent, which means user-owned by definition. The heads namespace is empty *by construction*, not by maintainer preference.

3. **Heads are port-fragile, trunks are port-survivable** (ml-engineer's MLOps argument). Every cross-language port of an INR-based system rewrites the head from scratch because heads encode distribution choices, loss conventions, and downstream-task postprocessing. The trunk — basis init, omega scheduling, encoding spectra — is the expensive-to-re-derive part. Shipping heads optimises the cheap-and-portable side at the expense of the discipline that keeps the expensive-and-fragile side correct.

4. **Cohort norm** (ml-engineer's empirical check). `samgria`, `rltrain`, `xptrack` export zero curated-convenience namespaces. `ondes.heads` would be the first such namespace in the cohort and would set the opposite gravity from the one explicitly chosen elsewhere.

5. **Reversibility asymmetry** (advocate's concession). Adding `head=` and `ondes.heads.X` later is a trivial additive change. Removing them later is a breaking change. Start at the smaller surface; grow only against demonstrated need.

## Structural design — polymorphism over discriminators

**No `kind: str` dispatch.** Activation choice (SIREN/HSIREN/WIRE), encoding choice (Identity/Gaussian/LearnedGaussian/Dyadic), and any future variant family is expressed via concrete subclasses inheriting a thin ABC. Each subclass carries only its own fields (e.g. `WIRELayer` has `s`; `SIRENLayer` doesn't; `LearnedGaussian` has a trainable `sigma`; `Gaussian` doesn't). String discriminators are forbidden in this codebase by convention — they make invalid states constructible, force XLA to compile dead branches, and propagate the anti-pattern into every downstream consumer that touches the type.

**The same prohibition applies at field granularity.** Type-discriminator dispatch via `callable(x)`, `isinstance(x, T)`, or any runtime check on a union field's inferred shape is the same anti-pattern as string discriminators and is also forbidden. The principle: a field is `T` (one type), not `T₁ | T₂` where the consumer dispatches on which arrived. If two semantics need different consumer behaviour, they need different classes. This applies across the library, regardless of whether the discriminator is at class granularity (`kind: str` → SIREN/HSIREN/WIRE) or field granularity (`sigma: float | Callable` → Gaussian/GaussianFromShape). The two cases share the same diagnostic — at the dispatch site the consumer is asking "which variant did I get?" — and the same fix: hoist the discrimination into the type system.

Three reasons spelled out:

1. **Pytree hygiene.** With a single shared class, every layer/encoding carries every variant's fields. A `BasisLayer(kind="siren")` would still pytree-leaf an unused `s` for WIRE's Gaussian-window scalar; JAX optimisers see those unused leaves, allocate optimiser state for them, and (silently) propagate gradients through them. The split puts `s` only on `WIRELayer` — non-WIRE pytrees genuinely don't contain it. Same logic kills the `sigma | sigma_from_shape | learn_sigma` triple-nullable on `Encoding`: each encoding now carries only the parameters its forward pass uses.

2. **Type-system support.** With a discriminator, callers can write `body.kind == "siren"` but the type checker has no opinion. With separate classes, `isinstance(body, SIREN)` (or pattern matching on type) gets static-analysis support, and downstream code (e.g. a renderer in `loom`) can use overload resolution rather than `if/elif` chains on string fields.

3. **Propagation cost.** The discriminator pattern looks cheap in one library and stays cheap in two; by the third consumer the `kind in {…}` checks have replicated into every downstream module that wants to dispatch on basis identity. Killing it before it propagates to `loom` (the immediate next downstream) is cheaper than killing it after.

The same logic applies to encoding factories: `gaussian_fixed(2.5)` and `gaussian_from_shape(rule)` and `gaussian_learn()` were three indirections over near-identical record constructions. The class IS the constructor — `Gaussian(rank=..., num_freqs=..., sigma=..., key=...)`, `LearnedGaussian(rank=..., num_freqs=..., key=...)`, `Dyadic(rank=..., num_bands=...)`, `Identity(in_dim=...)`. The `sigma_from_shape: Callable` field is gone too — that was a shape-rule mechanism for per-leaf encoding construction, which is the *renderer's* responsibility (loom calls `Gaussian(sigma=nyquist_sigma(leaf.shape), ...)` per leaf), not the encoding's.

This is a permanent policy. Future contributors hit it at PR review: any new variant family is a new subclass, never a new `kind` value.

**Downstream consumers type against `Body` or `BasisModule`, never the concrete union.** Two valid choices: `Body` for nominal typing (the concrete public base; what every shipped basis body inherits) or `BasisModule` for structural typing (the `Protocol`; matches any duck-typed body including user-defined wrappers). Both are public and documented. What's forbidden is `SIREN | HSIREN | WIRE` (the anti-pattern relapse this section closes — the union grows every time a new basis lands, every downstream consumer pays the cost). `BasisModule` is `@runtime_checkable`, so `isinstance(body, BasisModule)` works at runtime when static typing isn't enough.

**Scannability: enabled but not implemented (verified).** Within one body, all layers share an identical pytree structure — the `is_first` kwarg that `siren_init` uses to pick the init bound (1/in_dim vs sqrt(6/in_dim)/omega) is consumed in `__init__` and discarded; it is *not* a field on the layer. This makes `jax.lax.scan` over the layer stack mechanically clean: array leaves stack along a leading axis, the scan body is one `eqx.combine` step. The only residual constraint is *shape*: layer 0's `W` is `(hidden_dim, in_dim)` while layers 1..N-1 are `(hidden_dim, hidden_dim)`. When `in_dim == hidden_dim` the whole stack scans uniformly; in the typical INR case (`in_dim` is 2 or 3 for coords, `hidden_dim` is 64+), the realistic scan pattern is layer 0 eager then scan over layers 1..N-1. `tests/test_scannability.py` demonstrates both paths, across `SIREN`/`HSIREN`/`WIRE` with and without FiLM, asserting numerical match with the eager for-loop to within float32 fusion-reorder noise (~1e-4). `ondes` does not ship a scan implementation — `Body.trunk` uses a plain Python loop — because (a) downstream renderers (loom) can choose between scan, unrolled loop, or `jax.lax.fori_loop` per their compile-budget needs, and (b) for typical 4-8 layer INRs the unrolled forward is what XLA produces anyway. The scannability is a *capability*, not a default. Forward-pass microbenchmark for the canonical `SIREN(in_dim=2, hidden_dim=64, num_hidden_layers=4)` × `batch=1024` config sits at ~0.5 ms median (jit+vmap, CPU, JAX 0.10) — no measurable difference from a hypothetical pre-refactor `BasisBody(kind="siren")` version, since `kind` was already static and XLA traced the conditional away.

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
