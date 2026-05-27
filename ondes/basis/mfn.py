"""MFN basis: Multiplicative Filter Networks (Fathony+ 2021, ICLR).

Two flavours of multiplicative filter network from the paper:

- ``FourierMFN`` — sinusoidal filters ``g_i(x) = sin(omega_i x + phi_i)``;
  the network output is provably a linear combination of a known Fourier basis.
- ``GaborMFN``   — Gabor filters ``g_i(x) = sin(omega_i x + phi_i) *
  exp(-0.5 * gamma_i * ||x - mu_i||^2)``; spatially-localised filters with
  learnable centres ``mu`` and scales ``gamma``.

Both share the multiplicative-composition recurrence

```
    z_0 = g_0(x)
    z_{i+1} = g_{i+1}(x) * (W_i z_i + b_i)        for i = 0..N-1
    y = W_out z_N + b_out
```

so the body owns ``N+1`` filter layers and ``N`` recurrence-linear pairs in
addition to the readout. This shape doesn't fit the SIREN-style "one linear +
one pointwise activation per layer" mould, so the body subclasses ``Body``
directly and ships its own filter/recurrence layer types rather than mapping
onto the ``Basis`` ABC's ``W, b, omega`` triple.

Reference implementation (Bosch Research):
https://github.com/boschresearch/multiplicative-filter-networks
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ondes.basis._base import Body, _validate_body_args


class FourierFilter(eqx.Module):
    """Sinusoidal filter ``sin(W @ x + b)`` (Fathony+ 2021, eq. 6).

    The filter's pre-activation linear layer is initialised under the paper's
    convention: weights uniform ``[-bound, bound]`` with
    ``bound = input_scale / sqrt(n_layers + 1)`` so the per-filter variance
    matches the overall ``input_scale`` after the multiplicative composition
    is unrolled. Biases are uniform on ``[-pi, pi]`` to randomise phase.
    """

    W: Float[Array, "hidden in"]
    b: Float[Array, "hidden"]

    def __init__(self, in_dim, hidden_dim, input_scale, n_layers, *, key):
        """Sample filter weights at ``input_scale / sqrt(n_layers + 1)`` and uniform-phase bias."""
        bound = float(input_scale) / jnp.sqrt(n_layers + 1.0)
        kw, kb = jax.random.split(key)
        self.W = jax.random.uniform(kw, (hidden_dim, in_dim), minval=-bound, maxval=bound)
        self.b = jax.random.uniform(kb, (hidden_dim,), minval=-jnp.pi, maxval=jnp.pi)

    def __call__(self, x):
        """Apply the sinusoidal filter pointwise."""
        return jnp.sin(self.W @ x + self.b)


class GaborFilter(eqx.Module):
    """Gabor filter ``sin(W @ x + b) * exp(-0.5 * gamma * ||x - mu||^2)`` (Fathony+ 2021, eq. 9).

    Carries per-filter centres ``mu`` (uniform in ``[-1, 1]``) and per-filter
    scales ``gamma`` (Gamma-distributed with shape ``alpha / (n_layers + 1)``
    and rate ``beta``). Linear weights are Gaussian with std ``sqrt(gamma)``
    in the paper's formulation, which gives each filter a per-output
    bandwidth matched to its envelope width — the reference implementation
    samples them as ``randn * sqrt(gamma)`` rather than the uniform bound
    used by the sinusoidal MFN.
    """

    W: Float[Array, "hidden in"]
    b: Float[Array, "hidden"]
    mu: Float[Array, "hidden in"]
    gamma: Float[Array, "hidden"]

    def __init__(self, in_dim, hidden_dim, n_layers, *, key, alpha=6.0, beta=1.0):
        """Sample mu uniformly in [-1, 1], gamma from Gamma(alpha/(n_layers+1), beta), and weights ~ N(0, gamma)."""
        k_mu, k_gamma, k_w, k_b = jax.random.split(key, 4)
        self.mu = jax.random.uniform(k_mu, (hidden_dim, in_dim), minval=-1.0, maxval=1.0)
        shape = float(alpha) / (n_layers + 1.0)
        # jax.random.gamma returns samples from Gamma(shape) with rate 1; divide by beta to apply the rate.
        self.gamma = jax.random.gamma(k_gamma, shape, (hidden_dim,)) / float(beta)
        # Weights drawn from N(0, gamma): per-row std matches that row's gamma so the
        # filter's spatial frequency couples to its envelope width.
        std = jnp.sqrt(self.gamma)
        self.W = jax.random.normal(k_w, (hidden_dim, in_dim)) * std[:, None]
        self.b = jax.random.uniform(k_b, (hidden_dim,), minval=-jnp.pi, maxval=jnp.pi)

    def __call__(self, x):
        """Apply the Gabor (sin * Gaussian envelope) filter pointwise."""
        diff = x[None, :] - self.mu  # (hidden_dim, in_dim)
        sq = jnp.sum(diff * diff, axis=-1)  # (hidden_dim,)
        envelope = jnp.exp(-0.5 * self.gamma * sq)
        return envelope * jnp.sin(self.W @ x + self.b)


def _mfn_recurrence_init(in_dim, out_dim, weight_scale, key):
    """Uniform init for the recurrence-linear matrices ``(W_i, b_i)``.

    Bound ``sqrt(weight_scale / in_dim)`` matches the reference implementation.
    The same routine is shared between Fourier and Gabor MFN because the
    recurrence step is identical across both flavours.
    """
    bound = jnp.sqrt(float(weight_scale) / in_dim)
    kw, kb = jax.random.split(key)
    W = jax.random.uniform(kw, (out_dim, in_dim), minval=-bound, maxval=bound)
    b = jax.random.uniform(kb, (out_dim,), minval=-bound, maxval=bound)
    return W, b


class _MFNBody(Body):
    """Shared trunk for FourierMFN / GaborMFN.

    Owns the recurrence-linear stack and the readout; concrete subclasses
    own the filter list (with the paper-specific filter init scheme).

    Override ``layers`` field semantics: in this body ``self.layers`` holds
    the recurrence-linear ``(W_i, b_i)`` pairs as a stacked-array nested
    pytree; ``self.filters`` is the per-step filter modules. The two-stream
    layout is the natural shape for the MFN recurrence (filters and
    recurrence-linears alternate but live in different parameter families).
    """

    filters: tuple
    recurrence_W: Float[Array, "n_layers hidden hidden"]
    recurrence_b: Float[Array, "n_layers hidden"]

    def _build_recurrence(self, num_hidden_layers, hidden_dim, weight_scale, keys):
        Ws = []
        bs = []
        for i in range(num_hidden_layers):
            W, b = _mfn_recurrence_init(hidden_dim, hidden_dim, weight_scale, keys[i])
            Ws.append(W)
            bs.append(b)
        return jnp.stack(Ws), jnp.stack(bs)

    def trunk(self, coord, *, film=None):
        """MFN multiplicative-composition recurrence over filters.

        ``film`` is applied per recurrence step (i.e. to the post-(W_i z + b_i)
        output before multiplication by the next filter). The first filter's
        output is unmodulated — there's no prior recurrence-linear for FiLM
        to gate — which mirrors the SIREN/H-SIREN/WIRE convention of FiLM
        being a per-hidden-layer gate.
        """
        z = self.filters[0](coord)
        for i in range(self.num_hidden_layers):
            pre = self.recurrence_W[i] @ z + self.recurrence_b[i]
            if film is not None:
                gamma = film[i, : self.hidden_dim]
                beta = film[i, self.hidden_dim :]
                pre = gamma * pre + beta
            z = self.filters[i + 1](coord) * pre
        return z

    # Override layers field: MFN doesn't have the SIREN-style "Basis subclass per layer"
    # so we leave self.layers empty. Downstream code that iterates over body.layers
    # will see an empty tuple — that's intentional, the structural layers are in
    # self.filters + self.recurrence_W/b, exposed under their own names because
    # they carry distinct semantics (filter vs recurrence linear).


class FourierMFN(_MFNBody):
    """Sinusoidal MFN body (Fathony+ 2021).

    Each filter is ``sin(W_i @ x + b_i)``. The network output is provably an
    exact linear combination of a Fourier basis whose frequencies are
    convolutions of the per-filter ``W_i`` rows (paper eq. 7).
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_hidden_layers,
        *,
        key,
        out_features=None,
        input_scale=256.0,
        weight_scale=1.0,
    ):
        """Initialise ``num_hidden_layers + 1`` Fourier filters and the recurrence stack.

        Args:
            in_dim: Coordinate (input) dimension.
            hidden_dim: Width of the recurrence and each filter.
            num_hidden_layers: Number of recurrence steps (the network has
                ``num_hidden_layers + 1`` filters and ``num_hidden_layers``
                recurrence-linear pairs).
            key: JAX PRNG key.
            out_features: Readout width; see :class:`ondes.basis.siren.SIREN`.
            input_scale: Bandwidth of the per-filter linear map. Paper uses
                ``input_scale = 256`` for natural-image fitting; reduce for
                smoother targets.
            weight_scale: Scale on the recurrence-linear weight uniform-init bound.
        """
        out_features = _validate_body_args(num_hidden_layers, out_features)
        n_filters = num_hidden_layers + 1
        keys = jax.random.split(key, n_filters + num_hidden_layers + 1)
        filter_keys = keys[:n_filters]
        rec_keys = keys[n_filters : n_filters + num_hidden_layers]
        readout_key = keys[-1]

        self.filters = tuple(
            FourierFilter(in_dim, hidden_dim, input_scale, num_hidden_layers, key=k) for k in filter_keys
        )
        self.recurrence_W, self.recurrence_b = self._build_recurrence(
            num_hidden_layers, hidden_dim, weight_scale, rec_keys
        )

        out_dim = 1 if out_features is None else out_features
        # Readout uses the same uniform-init bound as the recurrence; the paper
        # treats the output projection as the (N+1)-th linear in the recurrence.
        rw, rb = _mfn_recurrence_init(hidden_dim, out_dim, weight_scale, readout_key)
        self.readout_W = rw
        self.readout_b = rb
        self.layers = ()
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers


class GaborMFN(_MFNBody):
    """Gabor MFN body (Fathony+ 2021).

    Each filter is ``sin(W_i @ x + b_i) * exp(-0.5 * gamma_i * ||x - mu_i||^2)``.
    Centres ``mu``, scales ``gamma``, and linear weights are learnable per the
    paper (every filter parameter is in the optimisation state).
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_hidden_layers,
        *,
        key,
        out_features=None,
        alpha=6.0,
        beta=1.0,
        weight_scale=1.0,
    ):
        """Initialise ``num_hidden_layers + 1`` Gabor filters and the recurrence stack.

        Args:
            in_dim: Coordinate (input) dimension.
            hidden_dim: Width of the recurrence and each filter.
            num_hidden_layers: Number of recurrence steps.
            key: JAX PRNG key.
            out_features: Readout width.
            alpha: Shape parameter for the Gamma prior on ``gamma``. Paper
                default is ``alpha = 6.0``.
            beta: Rate parameter for the Gamma prior on ``gamma``. Paper
                default is ``beta = 1.0``.
            weight_scale: Scale on the recurrence-linear weight uniform-init bound.
        """
        out_features = _validate_body_args(num_hidden_layers, out_features)
        n_filters = num_hidden_layers + 1
        keys = jax.random.split(key, n_filters + num_hidden_layers + 1)
        filter_keys = keys[:n_filters]
        rec_keys = keys[n_filters : n_filters + num_hidden_layers]
        readout_key = keys[-1]

        self.filters = tuple(
            GaborFilter(in_dim, hidden_dim, num_hidden_layers, key=k, alpha=alpha, beta=beta) for k in filter_keys
        )
        self.recurrence_W, self.recurrence_b = self._build_recurrence(
            num_hidden_layers, hidden_dim, weight_scale, rec_keys
        )

        out_dim = 1 if out_features is None else out_features
        rw, rb = _mfn_recurrence_init(hidden_dim, out_dim, weight_scale, readout_key)
        self.readout_W = rw
        self.readout_b = rb
        self.layers = ()
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
