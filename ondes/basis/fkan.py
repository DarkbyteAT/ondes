r"""FKAN basis: first-layer per-edge learnable Fourier-series features (Mehrabian+ 2024).

Reference: "Implicit Neural Representations with Fourier Kolmogorov-Arnold
Networks" (Mehrabian, Adi, Heidari, Hacihaliloglu, arXiv:2409.09323v3, 2025).
Code: https://github.com/Ali-Meh619/FKAN.

The FKAN mechanism is a **learnable feature map on the first layer only**: each
(input-dim, output-neuron) edge owns its own truncated Fourier series over the
raw input coordinate,

$$y[o] = \text{bias}[o] + \sum_{j}\sum_{k=1}^{K}
      A[o,j,k]\,\cos(k\,x_j) + B[o,j,k]\,\sin(k\,x_j),$$

with $A, B$ of shape $(\text{out}, \text{in}, K)$. Subsequent layers are a plain
MLP with a fixed scalar activation. Because the mechanism produces the first
hidden features rather than being a per-layer pointwise non-linearity, it does
**not** fit the ``Basis`` (``W, b, omega`` + ``_activate``) mould. The first
layer is therefore its own ``eqx.Module`` (:class:`FKANFirstLayer`) and the body
subclasses :class:`~ondes.basis._base.Body` directly, overriding ``trunk`` — the
same shape RFF (encoding + MLP) and the MFN family use. The Fourier feature map
is layer 0 (unmodulated by FiLM, matching RFF's encoding and MFN's ``filters[0]``);
``num_hidden_layers`` counts the plain hidden layers *after* it, mirroring the
paper's own "FKAN block, then $L$ hidden layers" description (p. 3, Table I arch).

**Every integer harmonic $k = 1 \ldots K$, both sine and cosine, fundamental
fixed at 1** (frequencies are the raw integers times the coordinate — there is
no learnable or shared $\omega$ inside the series, and no $\omega_0$ prefactor on
the first layer). This is the full Fourier series, and the fixed-unit fundamental
is exactly what distinguishes FKAN from the odd-harmonic comb (which locks to odd
harmonics of a single *shared, learnable* fundamental $\omega$); see
:mod:`ondes.basis.comb`.

Canonical hyperparameters (Table I, image task): the paper uses ``n_freqs=270``
with a first-layer latent width of 128. **The default ``n_freqs`` here is the
canonical 270** — do not silently diverge from the paper's grid size. The
head-to-head equal-parameter arm (comb paper, DoD 2) sets a much smaller
``n_freqs`` explicitly so the first-layer cost ($2 K \cdot \text{in} \cdot
\text{hidden}$ coefficients) matches the comb's tiny activation overhead; that is
a caller choice, not a default.

Two paper-vs-code divergences, both behind flags (both change the forward pass):

- ``use_layernorm`` — the released code applies ``LayerNorm`` over the first
  layer's outputs; the paper never mentions it, yet it is present in the notebook
  that reproduces Table I. Code-only.
- ``gated_activation`` — the paper's Eq. (4) hidden activation is $\tanh(\omega_0
  u)$; the shipped code uses the gated $(u + \tanh(\omega_0 u))\,\sigma(u)$
  instead. They differ; label arms explicitly.

**Default configuration reproduces the paper's *reported-result* mechanism, not
its written equations.** ``gated_activation=True`` and ``use_layernorm=True`` are
the released-code configuration behind the paper's headline 37.91 dB. This
follows the imported-baseline defaults rule: a reimplemented baseline defaults to
the configuration that reproduces the paper's *reported results*, with the
paper-text form (``gated_activation=False, use_layernorm=False`` — plain
$\tanh(\omega_0 u)$, no norm) behind flags. The split is deliberate: the canonical
*results* come from the code, so the default follows the code and the docstring
names the paper-text form.

This port reproduces FKAN's *mechanism* at a uniform ``hidden_dim`` — the exact
first-layer per-edge coefficient count $2 K\, d_\text{in}\,\text{hidden}$ (Table
I's 138,240 at the canonical $K = 270$, $d_\text{in} = 2$, width 128), pinned in
the tests. It does **not** reproduce the paper's 436,367-parameter *total*: that
comes from the paper's non-uniform trunk (widths 128, 256, 256, 256, 512), which
this uniform-width ``Body`` API does not express. Match a parameter budget
downstream by choosing ``hidden_dim`` / ``n_freqs``, not by expecting this class
to mirror the paper's width schedule.

Initialisation of the Fourier coefficients — $A, B \sim \mathcal{N}(0, 1/(d_\text{in}
K))$, bias zeros — is **not stated in the paper**; it is inherited from the
reference code (a variance-preservation comment, no analysis). The hidden layers
follow the code's SIREN-style $U(\pm\sqrt{6/d_\text{in}}/\omega_0)$ (the $/\omega_0$
divisor is code-only; the paper omits it), reused from
:func:`ondes.basis.siren.siren_init`.

FKAN has **no SIREN special case** — the hidden activation is $\tanh$-based, not
sinusoidal, so there is no $c = e_0$ corner analogous to the comb family.

The hidden-layer $\omega_0$ is stored as a learnable leaf (the ``Basis``
convention), which **departs by default from the paper's fixed $\omega_0 = 30$**.
Two cautions follow. First, the ``Basis`` ABC justifies direct (non-log)
parameterisation of ``omega`` by the activations being *even* in ``omega`` (sin,
sinh-then-sin, cos), so its sign is immaterial — that argument does **not**
transfer here: $\tanh(\omega_0 u)$ and the gated form are *odd* in $\omega_0$, so
a trainable $\omega_0$ can drift in magnitude *and flip sign*, changing the
activation. Second, to recover the paper's fixed-$\omega_0$ regime, freeze the
``omega`` leaves before the optimiser step with
``eqx.partition(body, body.omega_mask())`` (see :meth:`FKAN.omega_mask`) — the
same frozen-parameter pattern RFF/BACON use. A head-to-head harness must pin one
regime across arms (freeze $\omega_0$ on the FKAN arm, or train it on all arms).

The linear readout reuses :func:`ondes.basis.siren._build_readout` at the body's
*true* ``hidden_dim`` fan-in. This is a conscious divergence from the reference
code, which hardcodes a fan-in of 128 in the readout bound even when the last
hidden width differs; the true fan-in is the more principled variance-preserving
choice and is intentional here.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key, PyTree

from ondes.basis._base import Basis, Body, _validate_body_args
from ondes.basis.siren import _build_readout, siren_init


def _validate_n_freqs(n_freqs: int) -> int:
    """Reject ``n_freqs < 1`` (mirrors ``_validate_body_args``'s layer check).

    A non-positive grid size gives an empty harmonic axis, which would silently
    zero the entire first-layer feature map. ``assert`` matches the constructor-
    precondition convention in ``_base.py``.
    """
    assert n_freqs >= 1, f"n_freqs must be >= 1, got {n_freqs}"
    return int(n_freqs)


class FKANFirstLayer(eqx.Module):
    r"""Per-edge learnable Fourier-series feature map (Mehrabian+ 2024, first layer).

    Computes ``y[o] = bias[o] + sum_{j,k} A[o,j,k] cos(k x_j) + B[o,j,k] sin(k x_j)``
    for ``k = 1 .. n_freqs`` (every integer harmonic, sine and cosine, fundamental
    fixed at 1). Each ``(input-dim j, output-neuron o)`` edge owns its own ``2 *
    n_freqs`` coefficients — per-edge granularity, finer than per-neuron.

    Carries ``A`` (cosine) and ``B`` (sine) coefficient tensors of shape
    ``(out_dim, in_dim, n_freqs)`` and a per-output ``bias``. (Note: the paper's
    Eq. 3 letters them the other way — ``a_k`` on ``sin``, ``b_k`` on ``cos``;
    here ``A`` is the cosine coefficient and ``B`` the sine, which reads more
    naturally against the ``cos``/``sin`` call order.) This is a plain feature map
    (no ``W, b, omega`` triple, no pointwise ``_activate``), so it is an
    ``eqx.Module`` rather than a ``Basis`` subclass — analogous to
    :class:`~ondes.basis.mfn.FourierFilter` and
    :class:`~ondes.basis.bacon.BACONFilter`.
    """

    A: Float[Array, "out in n_freqs"]
    B: Float[Array, "out in n_freqs"]
    bias: Float[Array, "out"]
    n_freqs: int = eqx.field(static=True)

    def __init__(self, in_dim: int, out_dim: int, n_freqs: int, *, key: Key[Array, ""]) -> None:
        r"""Sample the Fourier coefficients and zero the bias.

        ``A, B ~ N(0, 1/(in_dim * n_freqs))`` — i.e. ``randn / sqrt(in_dim *
        n_freqs)`` — inherited from the reference code (the paper states no
        coefficient init). ``bias`` is zeros: the code absorbs the constant
        (``k = 0``) term into the bias, so the harmonic sum starts at ``k = 1``.

        Args:
            in_dim: Coordinate (input) dimension.
            out_dim: First-layer latent width (number of output neurons).
            n_freqs: Grid size ``K`` — the number of integer harmonics. Paper
                canonical value is 270.
            key: JAX PRNG key.
        """
        self.n_freqs = _validate_n_freqs(n_freqs)
        std = 1.0 / jnp.sqrt(float(in_dim * self.n_freqs))
        k_a, k_b = jax.random.split(key)
        self.A = jax.random.normal(k_a, (out_dim, in_dim, self.n_freqs)) * std
        self.B = jax.random.normal(k_b, (out_dim, in_dim, self.n_freqs)) * std
        self.bias = jnp.zeros((out_dim,))

    def __call__(self, coord: Float[Array, "in"]) -> Float[Array, "out"]:
        """Evaluate the per-edge Fourier series at ``coord``.

        Args:
            coord: Coordinate vector of shape ``(in_dim,)``.

        Returns:
            First-layer features of shape ``(out_dim,)``.
        """
        k = jnp.arange(1, self.n_freqs + 1, dtype=coord.dtype)  # integer harmonics 1..K
        angles = coord[:, None] * k[None, :]  # (in, n_freqs)
        cos = jnp.cos(angles)
        sin = jnp.sin(angles)
        # y[o] = bias[o] + sum_{j,k} A[o,j,k] cos[j,k] + B[o,j,k] sin[j,k]
        return self.bias + jnp.einsum("ojk,jk->o", self.A, cos) + jnp.einsum("ojk,jk->o", self.B, sin)


class FKANHiddenLayer(Basis):
    r"""FKAN hidden layer: linear followed by ``tanh(omega * pre)`` or the gated variant.

    A plain ``Basis`` layer (``W, b, omega`` + pointwise ``_activate``) used for
    every layer *after* the first-layer Fourier feature map. The activation is
    selected by the static ``gated`` flag:

    - ``gated=False`` (paper Eq. 4): ``tanh(omega * pre)``.
    - ``gated=True`` (released code): ``(pre + tanh(omega * pre)) * sigmoid(pre)``.

    ``omega`` is the paper's ``omega_0`` (default 30), stored as a learnable leaf
    per the ``Basis`` convention. ``gated`` is a static field read at forward
    time; every hidden layer in a body shares the same value, so the per-layer
    pytree structure stays homogeneous.
    """

    gated: bool = eqx.field(static=True)

    def __init__(self, in_dim: int, out_dim: int, omega_init: float, *, key: Key[Array, ""], gated: bool) -> None:
        r"""Initialise the linear weights (SIREN hidden-layer bound) and ``omega``.

        Weights use ``siren_init(..., is_first=False)`` — bound ``sqrt(6/in_dim)/omega``
        — matching the reference code's ``U(+-sqrt(6/in_dim)/omega_0)`` (the
        ``/omega_0`` divisor is code-only; the paper omits it). These layers are
        never the network's first layer (the Fourier map is), so ``is_first`` is
        fixed ``False``.

        Args:
            in_dim: Input dimension of the linear map.
            out_dim: Output dimension of the linear map.
            omega_init: The activation frequency ``omega_0`` (paper default 30).
            key: JAX PRNG key.
            gated: Select the gated (code) activation over ``tanh`` (paper).
        """
        self.W, self.b = siren_init(in_dim, out_dim, omega_init, is_first=False, key=key)
        self.omega = jnp.array(float(omega_init))
        self.gated = bool(gated)

    def _activate(self, pre: Float[Array, "out"]) -> Float[Array, "out"]:
        """Apply the paper (``tanh``) or code (gated) hidden activation pointwise."""
        if self.gated:
            return (pre + jnp.tanh(self.omega * pre)) * jax.nn.sigmoid(pre)
        return jnp.tanh(self.omega * pre)


class FKAN(Body):
    r"""FKAN body: first-layer per-edge Fourier series then a plain MLP (Mehrabian+ 2024).

    Layer 0 is the :class:`FKANFirstLayer` feature map (optionally ``LayerNorm``-ed);
    ``num_hidden_layers`` :class:`FKANHiddenLayer` s follow, then the linear readout.
    See the module docstring for the placement rationale, the canonical
    ``n_freqs``, the two paper-vs-code flags, and the default-reproduces-code note.

    FiLM modulates the hidden layers only (the Fourier feature map is unmodulated,
    matching RFF's encoding); ``film`` therefore has ``num_hidden_layers`` rows,
    one per :class:`FKANHiddenLayer`.
    """

    first: FKANFirstLayer
    layernorm: eqx.nn.LayerNorm | None

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        *,
        key: Key[Array, ""],
        out_features: int | None = None,
        n_freqs: int = 270,
        omega_hidden: float = 30.0,
        gated_activation: bool = True,
        use_layernorm: bool = True,
    ) -> None:
        r"""Initialise the FKAN body.

        Args:
            in_dim: Coordinate (input) dimension.
            hidden_dim: Width of the first-layer feature map and every hidden layer.
            num_hidden_layers: Number of :class:`FKANHiddenLayer` s *after* the
                Fourier feature map (which is layer 0, not counted here).
            key: JAX PRNG key.
            out_features: Readout width; see :class:`ondes.basis.siren.SIREN`.
            n_freqs: Grid size ``K`` — number of integer harmonics per edge.
                **Default 270 is the paper's canonical value.** The equal-parameter
                head-to-head arm sets this much smaller explicitly.
            omega_hidden: The hidden-layer activation frequency ``omega_0`` (paper
                default 30). Stored as a learnable leaf per the ``Basis`` convention.
            gated_activation: ``True`` (default, released code) uses the gated
                hidden activation ``(u + tanh(omega_0 u)) sigmoid(u)``; ``False``
                (paper Eq. 4) uses ``tanh(omega_0 u)``.
            use_layernorm: ``True`` (default, released code) applies ``LayerNorm``
                over the first-layer outputs; ``False`` (paper text) omits it.
        """
        out_features = _validate_body_args(num_hidden_layers, out_features)
        # keys: [first, hidden_0 .. hidden_{L-1}, readout]
        keys = jax.random.split(key, num_hidden_layers + 2)
        self.first = FKANFirstLayer(in_dim, hidden_dim, n_freqs, key=keys[0])
        self.layernorm = eqx.nn.LayerNorm(hidden_dim) if use_layernorm else None
        self.layers = tuple(
            FKANHiddenLayer(hidden_dim, hidden_dim, omega_hidden, key=keys[i + 1], gated=gated_activation)
            for i in range(num_hidden_layers)
        )
        self.readout_W, self.readout_b = _build_readout(hidden_dim, omega_hidden, out_features, keys[-1])
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

    def omega_mask(self) -> PyTree:
        r"""Return a pytree mask selecting the learnable (non-``omega``) leaves.

        Use with ``eqx.partition(body, body.omega_mask())`` to split the body into
        ``(learnable, frozen)`` halves before an optimiser step, holding every
        hidden layer's ``omega`` fixed at its initial value. This recovers the
        paper's fixed $\omega_0 = 30$ regime — the trainable-leaf default departs
        from it (see the module docstring). Symmetric with
        :meth:`ondes.basis.rff.RFF.fix_encoding_mask` and
        :meth:`ondes.basis.bacon.BACON.fix_filters_mask`.
        """
        # Start from "everything learnable", then flip each hidden layer's omega off.
        mask = jax.tree_util.tree_map(lambda x: eqx.is_array(x), self)
        frozen_layers = tuple(eqx.tree_at(lambda la: la.omega, m, False) for m in mask.layers)
        return eqx.tree_at(lambda b: b.layers, mask, frozen_layers)

    def trunk(
        self,
        coord: Float[Array, "in"],
        *,
        film: Float[Array, "n_layers two_hidden"] | None = None,
    ) -> Float[Array, "hidden"]:
        """Apply the Fourier feature map (optionally LayerNorm-ed) then the hidden MLP.

        ``film`` threads through the hidden layers exactly as in the SIREN-family
        bodies; the Fourier feature map is unmodulated (it's a first-layer feature
        map, like RFF's encoding). ``film`` row ``i`` gates hidden layer ``i``.
        """
        self._check_film_shape(film)
        h = self.first(coord)
        if self.layernorm is not None:
            h = self.layernorm(h)
        for i, layer in enumerate(self.layers):
            if film is not None:
                gamma = film[i, : self.hidden_dim]
                beta = film[i, self.hidden_dim :]
                h = layer(h, gamma=gamma, beta=beta)
            else:
                h = layer(h)
        return h
