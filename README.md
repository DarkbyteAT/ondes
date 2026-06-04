# ondes

Functional implementations of implicit neural representations (SIREN, H-SIREN, WIRE) in JAX/Equinox.

## Installation

```bash
uv add ondes
```

Or with pip:

```bash
pip install ondes
```

## Composition

`ondes` ships the basis-MLP trunks (`SIREN`, `HSIREN`, `WIRE`) and the spectral init machinery. Anything post-trunk — distribution heads, parameterisations, rotation maps, vector fields, loss-specific transforms — lives in user code, wrapped around a concrete body inside your own `eqx.Module`. Type-annotate against the `ondes.Body` base class if your wrapper should accept any basis kind:

```python
import equinox as eqx
import jax

import ondes


class Model(eqx.Module):
    """Compose an ondes INR with whatever readout/head you want."""

    inr: ondes.Body

    def __call__(self, coord):
        features = self.inr(coord)
        # Apply your wrapping here: distribution, vector field,
        # rotation parameterisation, anything. The point is ondes
        # owns the body; you own composition.
        return features


key = jax.random.key(0)
inr = ondes.SIREN(in_dim=2, hidden_dim=64, num_hidden_layers=4, key=key)
model = Model(inr=inr)
```

Set `out_features=N` on the body if you want `N` raw outputs to feed into your wrapping; the default is a single scalar.

## Examples

For concrete composition recipes (Gaussian outputs, image fitting, SDF, rotation fields, etc.), see `examples/`. Each example is CI-tested and cites its parameterisation choice with at least one alternative in active use.

## Why no `ondes.heads` module?

See [`DECISIONS.md`](DECISIONS.md) — short answer: the literature has no neutral default for most output transformations, so picking one would take a methodological side on users' behalf.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, code conventions, and the three quality gates every PR must clear.

## License

MIT
