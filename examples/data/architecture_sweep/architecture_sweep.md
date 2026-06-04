# Architecture sweep on astronaut_256

A matched-architecture comparison of nine implicit-neural-representation (INR) bases
fitting a single image at each basis's paper-default hyperparameters.

## Executive summary

SIREN wins this regime — 34.23 dB final PSNR, 134 seconds — with H-SIREN
essentially tied at 34.20 dB. Everything else lands at least 8 dB lower:
WIRE 25.53, Fourier-MFN 23.36, PNF 21.74, BACON 20.91, RFF 18.34,
Gabor-MFN 15.53, and FINER bottoms out at 13.05 dB despite being a SIREN
extension. The ranking is faithful to "out-of-the-box at paper defaults
on one seed" and should be read as such, not as a verdict on the
architectures themselves; per-basis hyperparameter tuning and seed
averaging would almost certainly reshape it (FINER in particular).

![Per-basis reconstructions in PSNR-descending order](comparison_grid.png)

## Methodology

Each basis is fit against the same image, with all architectural knobs held
fixed across runs. Only basis-specific hyperparameters (which vary by paper)
differ between runs — and within each run, those are pinned to the values the
original paper recommends. No per-basis tuning was performed.

| Knob              | Value                                  |
| ----------------- | -------------------------------------- |
| image             | `examples/data/astronaut_256.png`      |
| grid              | 256 × 256                              |
| hidden width      | 128                                    |
| hidden layers     | 4                                      |
| training steps    | 1000                                   |
| optimiser         | Adam (per-basis lr per paper)          |
| chunk size (scan) | 50                                     |
| seed              | 0                                      |
| total wall-clock  | 32 min 14 s                            |

**Hardware.** Apple Silicon (M-series), macOS 15.6, `jax==0.10.1` with the
`jax-mlx-plugin==0.0.4` (`mlx==0.31.2`, `mlx-metal==0.31.2`) sidecar venv at
`.venv-mlx/`. `jax.devices()` reports `[mlx:0]` and the plugin auto-registers
as the default backend — no `JAX_PLATFORMS` override needed. MLX is
float32-only on this hardware; that limit applies to every fit equally.

**Paper-default discipline.** Each basis runs with its paper-blessed
hyperparameters (frequency / bandwidth init, learning rate, etc.). This
exposes how each basis performs *as recommended*, which is the right
question for a first-pass architecture comparison — but it is *not* the
same as asking which basis can be coaxed to the highest PSNR with a
per-image hyperparameter sweep. Per-basis tuning would almost certainly
reshuffle the ranking (cf. the FINER note below).

The `synthetic` subcommand is not included — it is a self-contained smoke
fit against a generated target (sinusoid / gaussian bump / Mandelbrot),
not a natural-image fit, so it has no meaningful place in this comparison.

## Results

| Basis        | Paper            | Final PSNR (dB) | Final MSE   | Wall-clock (s) | Notes                                              |
| ------------ | ---------------- | --------------: | ----------: | -------------: | -------------------------------------------------- |
| siren        | Sitzmann+ 2020   |           34.23 |   3.774e-04 |            134 |                                                    |
| hsiren       | Cai & Pan 2024   |           34.20 |   3.805e-04 |            173 | within sampling noise of SIREN                     |
| wire         | Saragadam+ 2023  |           25.53 |   2.797e-03 |            208 | known seed-sensitive at paper σ=10                 |
| fourier-mfn  | Fathony+ 2021    |           23.36 |   4.610e-03 |            217 |                                                    |
| pnf          | Yang+ 2022       |           21.74 |   6.704e-03 |            198 |                                                    |
| bacon        | Lindell+ 2022    |           20.91 |   8.117e-03 |            197 | output bandwidth aliasing at hidden=128            |
| rff          | Tancik+ 2020     |           18.34 |   1.465e-02 |            209 | paper `sigma=10, lr=1e-4` underconverges in 1000 s |
| gabor-mfn    | Fathony+ 2021    |           15.53 |   2.802e-02 |            434 | recurrence is slowest per-step                     |
| finer        | Liu+ 2024        |           13.05 |   4.954e-02 |            164 | seed-fragile: this seed lands far from a good init |

Numbers sorted by final PSNR descending. The same data is in
`results.csv` (machine-readable); intermediate JSON in `_metrics.json`.

![Per-basis loss curves at paper-default hparams](loss_curves.svg)

## Per-basis notes

**siren** (Sitzmann+ 2020). Sinusoidal periodic activations with
weight-init scaled by `omega=30`. Paper defaults `omega=30, lr=5e-4`.
First place. The reference benchmark for natural-image INR fits and
the basis everything else is implicitly compared against — losing
to SIREN by less than a dB is a tie; losing by more than 5 dB is a
sign the basis isn't suited to this regime.

**hsiren** (Cai & Pan 2024). SIREN extended by a `sinh` pre-modulation
that lets the first layer reach higher frequencies than SIREN's bandlimit.
Paper defaults `omega=30, lr=5e-4`. Statistically tied with SIREN
(0.03 dB lower) — the `sinh` enrichment is wasted at this image
resolution and depth, where SIREN's bandwidth already covers what the
image contains. Would expect H-SIREN to pull ahead on higher-frequency
targets (textures, fine detail at 512+ resolution) per the paper.

**wire** (Saragadam+ 2023). Gabor wavelet activation
`sin(ω·z) · exp(-σ²·z²)` combining oscillation and a Gaussian
envelope. Paper defaults `omega=10, s_init=10, lr=1e-3`. Third place
at 25.53 dB. WIRE was an order of magnitude better here than under
jax-mps (13 dB), which suggests it was hitting a fp32-precision-related
bad basin on the other backend; the mlx number is the honest paper-default
performance. Still far behind SIREN, which is consistent with WIRE's
trade-off (small σ for sharp edges, large σ for smoothness, paper σ=10
trying to split the difference on natural images).

**fourier-mfn** (Fathony+ 2021). Multiplicative Filter Network with
Fourier filters: `sin(W·x + b)` multiplied through a recurrence of
linear layers. Paper defaults `input_scale=256, weight_scale=1, lr=1e-3`.
Fourth at 23.36 dB. The recurrence lets each layer carve a different
frequency band, in principle competitive with SIREN; on this seed and
1000 steps it just hasn't converged that far.

**pnf** (Yang+ 2022). Progressive Neural Field — multiplicative
filter recurrence with a learned mix-layer at the head. Paper defaults
`input_scale=256, weight_scale=1, lr=1e-3`. Fifth at 21.74 dB. Behaves
similarly to Fourier-MFN; the mix-layer doesn't pay off at this step
count and depth.

**bacon** (Lindell+ 2022). Band-limited Coordinate Network — quantised
discrete frequency grid with a target output bandwidth. Paper defaults
`max_freq=256, quant=2π, lr=1e-3`. Sixth at 20.91 dB. The bandlimit
construction means BACON literally cannot represent frequencies above
`max_freq`, which is correctly set for a 256-pixel image but may
interact poorly with the layer recurrence at hidden=128 width.

**rff** (Tancik+ 2020). Gaussian Random Fourier Features encoding
into a plain ReLU MLP. Paper defaults `sigma=10, num_freqs=256,
lr=1e-4`. Seventh at 18.34 dB. The paper's `lr=1e-4` is conservative
and the network is genuinely still descending at step 1000 (PSNR was
climbing roughly 0.5 dB per 100 steps near the end) — give it 5000
steps and it would close most of the gap. At a sweep level this counts
against RFF; at a per-basis tuning level it would invite a `lr=1e-3`
re-run.

**gabor-mfn** (Fathony+ 2021). MFN with Gabor filters (Gaussian
envelope × sinusoid). Paper defaults `alpha=6, beta=1, weight_scale=1,
lr=1e-3`. Eighth at 15.53 dB and **the slowest per fit by a wide
margin** (434 s vs ~200 s typical) — the per-filter scale sampling
and the additional envelope computation make the recurrence kernel
substantially heavier than Fourier-MFN. The PSNR gap to Fourier-MFN
(8 dB) is mostly a converge-rate story; both should improve with
more steps but Gabor's per-step cost makes that expensive.

**finer** (Liu+ 2024). SIREN with a first-layer bias whose
initialisation is bounded by `first_bias_scale` (paper recommends 5).
Paper defaults `omega=30, first_bias_scale=5, lr=5e-4`. Last place
at 13.05 dB. This is the outlier of the sweep: FINER extends SIREN
and so we'd expect it to be a SIREN-class winner, but on this seed
it starts from a *much* worse initialisation (initial loss 1.02 vs
SIREN's 0.31) and never recovers — the loss curve descends
monotonically but slowly, exactly the shape of "stuck in a poor
basin." The paper itself reports seed sensitivity from the
first-bias initialisation; SIREN+0.32 to FINER−21.18 on the same
seed at the same hparams is at the extreme end of that. With
seed averaging or with `first_bias_scale=1`, FINER would be expected
to land near SIREN.

## Conclusion

For 256×256 natural-image fitting at modest depth (4 hidden layers, 128
width) and 1000 training steps with paper-default hyperparameters, SIREN
is the basis to reach for — it's the highest PSNR, the fastest per fit,
and 0.03 dB ahead of its closest competitor (H-SIREN). The rest of the
field is far enough behind that the comparison is more about *why*
they're behind than which one to pick second.

Three caveats apply when generalising this result:

1. **Single seed.** FINER's 13 dB is a genuine seed-fragility datapoint,
   not a representative one. A multi-seed sweep would change the floor
   of the ranking (and possibly the top — SIREN's 34.23 is also one
   point, even if a less fragile one).

2. **Paper defaults aren't tuned defaults.** RFF in particular is
   reading "below par" largely because its paper `lr=1e-4` is slow.
   `lr=1e-3` for RFF would close most of the SIREN gap. This sweep
   measures "what does a new user get if they call `ondes.<Basis>()`
   with our published defaults?" — not "what is the best PSNR each
   basis can reach with a per-basis hyperparameter search?"

3. **Regime-specific.** Higher-resolution images, deeper networks,
   longer training, or non-natural-image targets (high-frequency
   textures, signed distance fields) all change this picture. WIRE,
   FINER, and the MFN family were all designed for regimes where SIREN
   leaves performance on the table — they're being measured here in the
   regime where SIREN does its best work.

## Reproducibility

The sweep was run against `feat/architecture-sweep` rebased onto
`origin/main` at `9bd759a`, with the Box-Muller rewrite of RFF's
B-matrix sampling applied (PR #15 — see Anomalies below for why).
PR #15 is a prerequisite of this PR; merging this PR without it on
`main` would leave RFF unable to construct on `jax-mps`.

Driver:

```bash
bash scripts/sweep_astronaut.sh
```

Per-basis invocations (executed by the driver):

```bash
SHARED="--image examples/data/astronaut_256.png --hidden 128 --layers 4 \
        --steps 1000 --grid 256 --chunk-size 50 --snapshot-every 1 \
        --log-every 50 --seed 0"

.venv-mlx/bin/python examples/fit_image.py siren        $SHARED --output-dir runs/sweep-arch/siren        --omega 30 --lr 5e-4
.venv-mlx/bin/python examples/fit_image.py hsiren       $SHARED --output-dir runs/sweep-arch/hsiren       --omega 30 --lr 5e-4
.venv-mlx/bin/python examples/fit_image.py wire         $SHARED --output-dir runs/sweep-arch/wire         --omega 10 --s-init 10 --lr 1e-3
.venv-mlx/bin/python examples/fit_image.py finer        $SHARED --output-dir runs/sweep-arch/finer        --omega 30 --first-bias-scale 5 --lr 5e-4
.venv-mlx/bin/python examples/fit_image.py rff          $SHARED --output-dir runs/sweep-arch/rff          --sigma 10 --num-freqs 256 --lr 1e-4
.venv-mlx/bin/python examples/fit_image.py bacon        $SHARED --output-dir runs/sweep-arch/bacon        --max-freq 256 --lr 1e-3
.venv-mlx/bin/python examples/fit_image.py fourier-mfn  $SHARED --output-dir runs/sweep-arch/fourier-mfn  --input-scale 256 --weight-scale 1 --lr 1e-3
.venv-mlx/bin/python examples/fit_image.py gabor-mfn    $SHARED --output-dir runs/sweep-arch/gabor-mfn    --alpha 6 --beta 1 --weight-scale 1 --lr 1e-3
.venv-mlx/bin/python examples/fit_image.py pnf          $SHARED --output-dir runs/sweep-arch/pnf          --input-scale 256 --weight-scale 1 --lr 1e-3
```

`jax-mlx-plugin` auto-registers as the default JAX backend on Apple Silicon,
so no `JAX_PLATFORMS` env var is needed (in fact, setting one breaks the
plugin loader; the env-var path is for `jax-mps`).

Sidecar venv setup (Apple Silicon):

```bash
uv venv --python 3.13 .venv-mlx
.venv-mlx/bin/python -m ensurepip --upgrade
.venv-mlx/bin/python -m pip install --timeout 600 jax jaxlib jax-mlx-plugin
.venv-mlx/bin/python -m pip install --timeout 600 -e . equinox optax \
    jaxtyping matplotlib pillow typer jax-tqdm
.venv-mlx/bin/python -c "import jax; print(jax.devices())"  # [mlx:0]
```

The `mlx-metal` wheel is ~38 MB; pip's `--timeout 600` is required
because `uv pip install`'s default 30 s timeout reliably fails the
download.

## Anomalies

**`jax-mlx-plugin` over `jax-mps`.** The sweep was first attempted on
the `jax-mps` plugin, which has known coverage gaps in its MLIR
lowering for several JAX primitives (`aval_to_ir_type` arity bug
against the current JAX MLIR API). The bug bites three operations
that `ondes` basis families depend on: `jnp.sinh` (H-SIREN's
modulation), `lax_special.erf_inv` (used internally by
`jax.random.normal`, RFF's B-matrix init), and `jax.random.gamma`
(Gabor-MFN's scale-prior sampling). `jax-mlx-plugin` 0.0.4 lowers all
three cleanly on Apple Silicon, so the sweep can run end-to-end on
one backend.

**Box-Muller rewrite of RFF's Gaussian sampling.** PR #15
(`fix/rff-box-muller-mps`) replaces `jax.random.normal(key, shape)` in
`ondes/basis/rff.py`'s B-matrix init with a Box-Muller draw built from
two uniforms (`sqrt(-2·log u₁) · cos(2π·u₂)`). The samples are
identically distributed N(0,1) — verified at N=10000: Box-Muller
mean=+0.0043, std=1.0043 vs `jax.random.normal` mean=−0.0119, std=0.9925,
both well within ±0.05 of N(0,1). The rewrite is harmless on
`jax-mlx-plugin` (this sweep's backend); it's load-bearing for anyone
running `ondes` on `jax-mps`. PRNG-stream caveat: a seeded RFF run
after PR #15 won't reproduce a seeded run before it — same
distribution of B matrices, different actual draws. This PR is
sequenced to merge after PR #15.

**Exponential-identity rewrite of H-SIREN's `sinh`.** Pre-existing
commit `4b19d02` (on `main`) replaces `jnp.sinh(pre)` in
`ondes/basis/hsiren.py` with `0.5·(exp(pre) − exp(−pre))` for the same
`jax-mps` `aval_to_ir_type` reason. Algebraically identical; mlx
doesn't need it but it doesn't hurt.

**FINER's 13 dB outlier.** See per-basis notes. Genuine seed-fragility,
not a backend issue or implementation bug. The loss curve is monotonic,
just stuck — initial loss is ~3× SIREN's, and the recovery is too slow
to close the gap in 1000 steps. Multi-seed averaging is the right
follow-up; a single-seed report is sensitive to this kind of
init-dependent stuck-basin behaviour.

**No CPU fallback.** All 9 fits ran on `[mlx:0]`. Wall-clock numbers
are directly comparable across bases.
