#!/usr/bin/env bash
# Architecture sweep: fit examples/data/astronaut_256.png with 9 INR bases
# at matched architecture (hidden=128, layers=4, 1000 steps, seed=0) and
# each basis's paper-default hyperparameters.
#
# Produces per-run artifacts under runs/sweep-arch/<basis>/ (gitignored)
# and aggregated outputs under examples/data/architecture_sweep/.
#
# Hardware: jax-mlx-plugin sidecar venv at .venv-mlx (Apple Silicon).
# Expected wall-clock: ~25-35 min total.
#
# Re-running is idempotent — each per-basis run dir is recreated cleanly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The sidecar venv's editable install of `ondes` was created from this
# worktree, so its .pth already points here. We still prepend `PYTHONPATH`
# defensively — re-running this script from a different checkout that
# shares the venv would otherwise resolve `import ondes` to whichever path
# the .pth was first written with.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# jax-mlx-plugin auto-registers as the default backend on Apple Silicon —
# no `JAX_PLATFORMS=mlx` needed (in fact setting it confuses the plugin
# loader; the env var path is for jax-mps).
PY=".venv-mlx/bin/python"
IMG="examples/data/astronaut_256.png"
OUT="runs/sweep-arch"
SHARED=(--image "$IMG" --hidden 128 --layers 4 --steps 1000 --grid 256
        --chunk-size 50 --snapshot-every 1 --log-every 50 --seed 0)

# Resume control: set RESUME=<basis> to skip earlier bases (idempotent — each
# run dir is wiped fresh on re-entry). Wallclock file is preserved on resume.
RESUME="${RESUME:-}"
_skipping=0
if [[ -n "$RESUME" ]]; then
    _skipping=1
fi

run_basis() {
    local name="$1"; shift
    if (( _skipping )); then
        if [[ "$name" == "$RESUME" ]]; then
            _skipping=0
        else
            echo "=== skip: $name (RESUME=$RESUME) ==="
            return 0
        fi
    fi
    local dir="$OUT/$name"
    rm -rf "$dir"
    echo "=== sweep: $name ==="
    local start=$SECONDS
    "$PY" examples/fit_image.py "$name" \
        "${SHARED[@]}" --output-dir "$dir" "$@"
    local elapsed=$((SECONDS - start))
    echo "$name $elapsed" >> "$OUT/_wallclock.txt"
    echo "=== $name done in ${elapsed}s ==="
}

mkdir -p "$OUT"
if [[ -z "$RESUME" ]]; then
    : > "$OUT/_wallclock.txt"
fi

run_basis siren        --omega 30 --lr 5e-4
run_basis hsiren       --omega 30 --lr 5e-4
run_basis wire         --omega 10 --s-init 10 --lr 1e-3
run_basis finer        --omega 30 --first-bias-scale 5 --lr 5e-4
run_basis rff          --sigma 10 --num-freqs 256 --lr 1e-4
run_basis bacon        --max-freq 256 --lr 1e-3
run_basis fourier-mfn  --input-scale 256 --weight-scale 1 --lr 1e-3
run_basis gabor-mfn    --alpha 6 --beta 1 --weight-scale 1 --lr 1e-3
run_basis pnf          --input-scale 256 --weight-scale 1 --lr 1e-3

echo
echo "=== aggregating ==="
"$PY" scripts/aggregate_sweep.py

echo
echo "=== sweep complete ==="
echo "per-run dirs: $OUT/<basis>/"
echo "aggregated:   examples/data/architecture_sweep/"
