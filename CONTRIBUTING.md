# Contributing to ondes

`ondes` is a JAX/Equinox library of implicit-neural-representation (INR) primitives: basis-MLP trunks (SIREN, H-SIREN, WIRE, FINER, RFF, BACON, PNF, Fourier-MFN, Gabor-MFN), Fourier-feature encodings, and the spectral-init machinery they share. It is the "INR mechanism" layer — it does *not* ship datasets, concrete training scripts, distribution heads, or downstream task glue. Those live in `examples/` (CI-tested demos) or in downstream consumers (`loom` for rendering, `samgria` for second-order optimisation, `xptrack` for experiment tracking).

We welcome contributions that fit that scope:

- **New basis families** with a paper-cited init scheme (Gabor variants, hash-grid backbones, learned-activation INRs, etc.).
- **New encodings** that compose under the existing `Encoding` ABC.
- **Tests** that pin a paper-claimed property (init bound, scannability, spectral behaviour) we don't already cover.
- **Bug fixes**, type-checker fixes, and docstring sharpening.

If you're not sure whether your idea fits — open a GitHub issue first and ask. The three-AND-gate in `DECISIONS.md` (spectral coupling / closed-form trap / multi-consumer) governs whether something belongs in the library at all versus in an `examples/` script; we'd rather discuss that upfront than after you've written the code.

## Quick start

```bash
git clone https://github.com/DarkbyteAT/ondes.git
cd ondes
source scripts/enable-venv.sh
uv run pytest tests/
```

A fresh clone should reach a green test run in under thirty seconds on a recent machine.

`scripts/enable-venv.sh` does meaningful setup work beyond what `uv run` covers on its own: it creates `.venv/` on first invocation, runs `uv sync --group dev` to pull in the dev-group dependencies (`uv run` resolves only the project's main dependencies), and installs the `pre-commit` git hooks. The pre-commit install step is load-bearing — without it the silent-no-op gotcha described in the next section can't trigger because the hooks aren't wired up at all, and formatting violations only surface when CI fails.

## The three quality gates

Every PR must clear **all three** of the following before requesting human review. No half-and-half. No "I'll fix the test later." All three or no review request.

```bash
uv run ruff check ondes/ tests/ examples/
uv run pyright ondes/
uv run pytest tests/
```

Equivalently:

```bash
make all
```

which runs `format-check`, `lint`, `typecheck`, and `test` in sequence and bails on the first failure. CI runs the same set on every push.

Note this is a *different* "three-AND-gate" from the one in `DECISIONS.md`. That one governs whether a new function belongs in the library's public surface; this one governs whether a PR is ready for review. They share the "all three or none" discipline but answer different questions.

### A note on the pre-commit hook

`ruff format` runs as a pre-commit hook. When it rewrites a staged file, the commit *silently no-ops* — the rewritten file becomes unstaged and the commit ends up empty. If a `git commit` appears to succeed but `git log` shows nothing new, re-`git add` the affected files and retry.

## Code conventions

Each convention is one line plus its rationale. Where the convention is non-obvious, the source-of-truth file is named for the canonical example.

### Polymorphism over discriminators

No `kind=` constructor argument, no factory function, no enum-keyed dict-of-classes. One concrete class per basis family (`SIREN`, `HSIREN`, `WIRE`, `FINER`, `RFF`, `BACON`, `PNF`, `FourierMFN`, `GaborMFN`) and one concrete class per encoding family (`Identity`, `Gaussian`, `LearnedGaussian`, `Dyadic`). Consumers dispatch by `isinstance(...)` or by the concrete type out of the constructor.

The same prohibition applies at field granularity: a field is `T`, not `T₁ | T₂` where the consumer runtime-checks which variant arrived. If two semantics need different consumer behaviour, they need different classes.

Full reasoning lives in `DECISIONS.md` §"Structural design — polymorphism over discriminators". The example subcommand layout in `examples/fit_image.py` mirrors this — one Typer subcommand per basis, each owning the kwargs that *that* basis takes.

### Value-function framing

Every INR is a value function: `(coord, channel) → amplitude` returning a scalar. RGB is `(x, y, c) → amplitude` (channel is a coord), not `(x, y) → (r, g, b)` (channel as output dim). `out_features=1` is canonicalised to `out_features=None` at construction so the two scalar-yielding forms produce identical pytrees — relevant for serialisation, jit caching, and `tree_structure` equality.

`out_features = N > 1` is supported when the downstream genuinely needs `N` independent readouts (multi-task heads, vector fields with a fixed output dimensionality known at construction).

### No `Head` wrapper

`Body.trunk(coord)` returns pre-readout hidden features and is the extension point for any post-trunk transformation. The library owns the linear readout; there is no `head=` kwarg, no `ondes.Head` type, and no `ondes.heads` namespace.

To add a distribution head (Gaussian, mixture, normalised flow, …), a rotation parameterisation, a vector field, or anything else built on top of the trunk, wrap a concrete `Body` inside your own `eqx.Module` and call `inr.trunk(coord)` (or `inr(coord)`) from it. See the README's composition snippet and the `Model` class in `examples/fit_image.py` for the canonical shape.

Five independent rationales for the no-head decision (no neutral default exists in the literature, heads are post-trunk and have no coupling to ondes internals, heads are port-fragile while trunks are port-survivable, cohort-norm with `samgria`/`rltrain`/`xptrack`, additive reversibility asymmetry) are documented in `DECISIONS.md` §"Why no heads".

### FiLM contract

Every concrete `trunk` calls `self._check_film_shape(film)` as its first line. The validator is defined once on `Body` and raises `ValueError` if `film` is not `None` and its shape is not `(num_hidden_layers, 2 * hidden_dim)`. `ValueError`, not `assert`, because the contract is part of the user-facing API and must survive `python -O`.

`tests/test_film_validation.py` parametrises across every shipped body class to lock the contract in. When you add a new body, add it to the `_BODY_CLASSES` tuple in that file.

### Init-only kwargs don't pollute the pytree

If a constructor kwarg is consumed only at construction time (its value influences `W`/`b` sampling but the forward pass never reads it again), take it as a `__init__` parameter and *do not* store it as a field. The canonical example is `is_first` on `SIRENLayer`/`HSIRENLayer`/`WIRELayer`/`FINERLayer` — it selects the SIREN init bound (`1/in_dim` for the first layer, `sqrt(6/in_dim)/omega` for the rest), the bound bakes into `W` and `b`, and the layer never needs `is_first` again. Storing it would add a static field that discriminates layer 0 from the rest at the pytree level, which breaks `jax.lax.scan` over the layer stack.

Conversely, a kwarg that *is* read at forward time (e.g. `FINERLayer.scale_req_grad`, which gates a `stop_gradient`) belongs on the pytree as a static field. All layers in one body must share the same value of any such static field so pytree-structural homogeneity across layers is preserved (`FINER.__init__` enforces this by passing the same value to every `FINERLayer`).

`tests/test_scannability.py` and the homogeneity assertion in `test_layer_pytree_structure_is_homogeneous_across_body` lock this invariant in.

### Repetition over confusing indirection

Three near-identical `__init__` bodies are preferable to an abstraction that hides the structure (virtual init methods, `ClassVar` dispatch tables, mixin hierarchies). Each body's `__init__` is written out explicitly even when the resulting code is repetitive — see `SIREN.__init__`, `HSIREN.__init__`, `WIRE.__init__` as the reference. Trying to dedupe them via a shared `_init_layers` helper that branches on a class attribute is exactly the discriminator anti-pattern at lower granularity.

### Library scope: machinery, not instantiations

`ondes` owns protocols, ABCs, value types, and the spectral primitives that the basis families themselves use. Concrete datasets, concrete targets, concrete training loops, and concrete renderers do not belong here. They belong in `examples/` (when they're useful demos of the library's intended composition pattern) or in downstream consumers.

The three-AND-gate in `DECISIONS.md` makes this concrete for proposed additions: a function belongs in `ondes` only if its design has *spectral or init coupling* to the basis machinery, *and* it has a closed-form trap (a naive implementation would plausibly be wrong), *and* it serves multiple consumers with no plausible variant on the horizon. Convenience wrappers fail at least one of the three.

### Typing convention

- `jaxtyping.Float[Array, "..."]` for shape-bearing array parameters and returns. Shape annotations are part of the API contract — `Float[Array, "out in"]`, `Float[Array, "n_layers two_hidden"]`, etc.
- `jaxtyping.Key[Array, ""]` for PRNG keys.
- Stdlib `int` / `float` / `bool` for scalars passed by value.
- PEP-604 union syntax (`T | None`, never `Optional[T]`).
- Explicit return type annotations on every public function and method.
- Generic stdlib aliases (`list[T]`, `dict[K, V]`, `tuple[T, ...]`), not `typing.List` / `typing.Dict` / `typing.Tuple`.

Pyright runs in strict-ish mode; check `pyrightconfig.json` for the exact configuration.

### Docstrings

Google-style with LaTeX-friendly math:

```python
from jaxtyping import Array, Float, Key


def siren_init(
    in_dim: int,
    out_dim: int,
    omega: float,
    is_first: bool,
    key: Key[Array, ""],
) -> tuple[Float[Array, "out in"], Float[Array, "out"]]:
    """Sample (W, b) under the SIREN initialisation scheme.

    First-layer weights are drawn uniformly from $[-1/\\text{in\\_dim}, 1/\\text{in\\_dim}]$;
    subsequent layers from $[-\\sqrt{6/\\text{in\\_dim}}/\\omega, +\\sqrt{6/\\text{in\\_dim}}/\\omega]$.

    Args:
        in_dim: Input dimension of the linear map.
        ...

    Returns:
        Tuple ``(W, b)`` with shapes ``(out_dim, in_dim)`` and ``(out_dim,)``.
    """
```

Use `$...$` for inline math and `$$...$$` for display math. Cite the paper (author + year) when the routine implements a paper-specific recipe.

### Tests

Plain `def test_*` (or `async def test_*`) functions at module top level. No classes. Given/When/Then structure as inline comments. `tests/` mirrors `ondes/` — `tests/test_<file>.py` for `ondes/<file>.py` where the file scope is right, plus cross-cutting files like `test_film_validation.py` and `test_scannability.py` for properties that parametrise across the whole basis family.

Test what you *claim*, not what you *assume*. "Runs without error" is not proof. Write the test that would catch the lie: assert the init bound matches the paper formula, assert the FiLM shape contract raises on wrong input, assert that WIRE has an `s` leaf and SIREN does not.

### Tool configs in dedicated files

`pyproject.toml` stays minimal. Tool configs live in dedicated files: `ruff.toml`, `pytest.ini`, `pyrightconfig.json`. Don't fold them back into `pyproject.toml`.

## How to add a new basis family

The canonical reference is the FINER basis (PR #11). Walk through `ondes/basis/finer.py`, `examples/fit_image.py`'s `finer` subcommand, and `tests/test_finer.py` to see the full shape. The recipe:

### 1. Implement the basis module

Create `ondes/basis/<name>.py` containing:

- A `<Name>Layer(Basis)` subclass — fields specific to *this* basis (e.g. `WIRELayer.s`, `FINERLayer.scale_req_grad`), an `__init__` that consumes any init-only kwargs and bakes them into `W`/`b`, and an `_activate(pre)` method.
- A `<Name>(Body)` subclass — owns layer construction, the readout, and the structural fields (`out_features`, `hidden_dim`, `num_hidden_layers`). Call `_validate_body_args(num_hidden_layers, out_features)` to canonicalise `out_features=1` to `None`. If your basis reuses the SIREN init family (SIREN, H-SIREN, WIRE, FINER all do), import `siren_init` and `_build_readout` from `ondes.basis.siren`. If your basis has its own init scheme (RFF uses Kaiming-uniform; MFN uses a Gamma prior for filter scales), write a helper next to your basis class — don't try to fold it into `siren_init`.
- Module docstring citing the paper and any reference implementation URL.

Use `ondes/basis/finer.py` as a structural template; use `ondes/basis/rff.py` if your basis has its own init scheme; use `ondes/basis/wire.py` if your basis carries an extra learnable scalar; use `ondes/basis/mfn.py` if your basis has a non-trivial recurrence in `trunk()` (overriding `Body.trunk` is fine when needed — RFF and MFN both do because their trunk shape differs from the SIREN-family layer-stack pattern).

### 2. Export it from the package

Add the public names to `ondes/basis/__init__.py` (alphabetical in the `__all__` list and in the imports) and to `ondes/__init__.py` (same).

### 3. Write the tests

Create `tests/test_<name>.py` mirroring `tests/test_basis.py`'s coverage for the canonical bases. At minimum, your suite should pin:

- Forward-pass output shape under default kwargs.
- Init-scheme correctness — sample the weights and assert the bound matches the paper formula. The `siren_init` tests in `tests/test_basis.py` are the template.
- `out_features` canonicalisation — confirm `out_features=1` and `out_features=None` produce identical pytrees.
- Scalar vs vector return — `out_features=None` returns shape `()` (0-d scalar), `out_features=N>1` returns shape `(N,)`.
- FiLM modulation actually changes the output (cross-check that the contract isn't a no-op).
- Gradient through both scalar and vector paths is finite and non-zero.
- JIT-compilability of `body()` and `body.trunk()`.

Then update the cross-cutting suites:

- Add your body class to `_BODY_CLASSES` in `tests/test_film_validation.py` so the FiLM-shape contract is exercised against it.
- Add your body to the appropriate parametrisations in `tests/test_scannability.py` if your basis follows the SIREN-family layer-stack pattern (skip if you override `trunk()` non-trivially).

### 4. Add an example subcommand

Add a Typer subcommand to `examples/fit_image.py` mirroring the existing ones. Defaults should be the paper-recommended hparams; include a help string per kwarg naming the paper and the default's source. Pass the basis-specific kwargs through `basis_extras` so `config.json` records them. Do not collapse the per-basis subcommands into a `--basis` enum — the polymorphism convention applies in the example exactly as it does in the library.

### 5. Run the three quality gates

```bash
uv run ruff check ondes/ tests/ examples/
uv run pyright ondes/
uv run pytest tests/
```

All three green, no exceptions, before requesting review.

## Pull request workflow

1. **Branch** off `main` with a descriptive name. Prefixes: `feat/` (new functionality), `fix/` (bug fix), `chore/` (housekeeping), `docs/` (documentation), `refactor/` (no behaviour change), `test/` (test-only).
2. **Open the PR** against `main`. Use the description template:

   ```
   Trello Card: [<name>](<link>)   # if applicable

   ## What?

   <one paragraph: what does this change do, observably>

   ## Why?

   <one paragraph: the motivating constraint, paper, bug, or convention>

   ## How?

   <design choices a reviewer would otherwise have to reverse-engineer>

   ## Changes

   - <bullet per touched module / category>

   ## Test Plan

   - [ ] uv run ruff check passes
   - [ ] uv run pyright passes
   - [ ] uv run pytest passes
   - [ ] <any additional manual checks>
   ```

3. **Run `/gemini review`** on the PR (or your preferred automated reviewer) and iterate until convergence — zero new substantive comments. Don't merge with unresolved automated review comments.

4. **Address every review finding** (automated or human). Three legitimate resolutions:
   - Fix it in the PR.
   - Open a follow-up issue / Trello card and link it from the resolution comment.
   - Push back with specific reasoning if you disagree.

   "Noted, will follow up" without a tracking link is not a resolution.

5. **Wait for human review** before merging. Never `gh pr merge` without explicit sign-off.

6. **Squash-merge with branch deletion** once approved:

   ```bash
   gh pr merge <N> --squash --delete-branch --repo DarkbyteAT/ondes
   ```

   Keeps `main`'s history one-commit-per-PR; avoids stale branches piling up.

7. **Bring branches up to date** with `git fetch origin && git merge origin/main`, not `git rebase`. The fetch step is necessary — `git merge origin/main` without a prior fetch merges the *locally cached* state of `origin/main`, which can be hours or days stale. Rebases rewrite history, require force-pushes, and produce more conflict iterations than merges; rebase only when there's a clear advantage (single-commit branch with no merge history worth preserving).

## Design records

Every significant design decision lives in `DECISIONS.md` with the reasoning chain that produced it. Read it before proposing a structural change to the public API. The recurring themes:

- **The library exposes mechanism, not policy.** No interpretation of user intent baked into enums or factories.
- **The three-AND-gate for adding library symbols.** Spectral coupling, closed-form trap, multi-consumer with no horizon variant.
- **Heads, distributions, and parameterisations are user-owned.** No `ondes.heads`, no `head=` kwarg, no bundled distribution library.
- **Scannability is a capability, not a default.** The library doesn't ship a `scan`-based forward pass; downstream renderers (`loom`) choose between unrolled, scanned, or `fori_loop` per their compile budget. The library's job is to make sure the layer pytree is structurally homogeneous so scan *works* when chosen.

If you find yourself proposing a change that conflicts with one of these, please open an issue first and reference the relevant `DECISIONS.md` section so we can have the design conversation before the code conversation.

## Where to ask

- **Bugs, feature requests, design discussion**: GitHub issues on [DarkbyteAT/ondes](https://github.com/DarkbyteAT/ondes).
- **Anything else**: open an issue. There is no separate chat channel.
