"""Forward-pass microbenchmark for SIREN.

Times the canonical post-refactor configuration (hidden_dim=64,
num_hidden_layers=4, in_dim=2, batch=1024) under JIT-compiled ``jax.vmap``.
The benchmark exists to *document the measurement*, not to chase a speedup.

Per ml-engineer's F7 prediction the refactor compiles to the same HLO as the
hypothetical pre-refactor ``BasisBody(kind="siren")`` version (the ``kind``
field was already a static string, so XLA traced the conditional away). The
expected result is "no measurable difference from a hypothetical comparison";
the *load-bearing* output is the absolute median ± std for the current
implementation, which downstream consumers (loom) can compare against their
own configurations.

Total runtime ~0.5s on CPU (50 warmup + 100 timed iters of a 1024-batch
jit+vmap forward), small enough to leave in the default suite.
"""

import gc
import statistics
import time

import equinox as eqx
import jax
import pytest

from ondes import SIREN


def _time_jit_vmap(body, batch, n_warmup=50, n_iters=100):
    """Return (median_ms, stdev_ms) of jit(vmap(body))(batch).

    The function is built once and compiled by the warmup pass; the timed
    iterations only measure the dispatch + device computation. GC is disabled
    inside the timing window so collection cycles don't leak into the
    measurement; warmup is generous enough that the JIT cache is stable by
    sample 0.
    """
    # eqx.filter_vmap composes with eqx.filter_jit cleanly; using jax.vmap
    # directly on the Module emits a (correct but noisy) Equinox warning
    # about untracked attribute assignment on the resulting wrapper.
    f = eqx.filter_jit(eqx.filter_vmap(body))
    for _ in range(n_warmup):
        jax.block_until_ready(f(batch))

    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        samples = []
        for _ in range(n_iters):
            t0 = time.perf_counter_ns()
            jax.block_until_ready(f(batch))
            t1 = time.perf_counter_ns()
            samples.append((t1 - t0) / 1e6)
    finally:
        if gc_was_enabled:
            gc.enable()

    return statistics.median(samples), statistics.stdev(samples)


@pytest.mark.benchmark
def test_forward_pass_microbenchmark(capsys):
    # Given: a canonical SIREN body (in_dim=2, hidden_dim=64, num_hidden_layers=4)
    # When: timing JIT-compiled jax.vmap over a batch of 1024 coords
    # Then: report median + stdev over 100 trials post-warmup. The test does
    # not assert a numeric threshold — the value is the *measurement* itself,
    # which the maintainer compares against historical numbers. A future
    # regression that 10× the time would be caught by reading the captured
    # stdout, not by a hard-coded assert (which would fragile on noisy CPUs).
    body = SIREN(in_dim=2, hidden_dim=64, num_hidden_layers=4, key=jax.random.key(0))
    coords = jax.random.uniform(jax.random.key(1), (1024, 2), minval=-1.0, maxval=1.0)

    median_ms, std_ms = _time_jit_vmap(body, coords)

    # Print via capsys.disabled() so the output reaches the terminal even when
    # pytest captures stdout by default.
    with capsys.disabled():
        print()
        print("  SIREN(in_dim=2, hidden_dim=64, num_hidden_layers=4) on batch=1024")
        print(f"  JAX {jax.__version__}, default backend: {jax.default_backend()}")
        print(f"  jit+vmap forward pass: median {median_ms:.3f} ms ± {std_ms:.3f} (100 trials)")

    # Sanity assertion: timings should be positive and not absurdly large.
    # ~50ms would indicate something is wrong (e.g. compilation leaked into
    # the timing window) on any reasonable CPU.
    assert median_ms > 0.0
    assert median_ms < 100.0, f"median {median_ms} ms is suspiciously large"
