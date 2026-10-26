#!/usr/bin/env bash
# Run the full pipeline (steps 1 to 4) on a small synthetic dataset and check the
# results against the planted cell types. No download needed; about 1 to 2
# minutes on a laptop.
#
# Usage (from anywhere):  bash example/run_example.sh
# Outputs go to example/work/ so they never overwrite a real run in data/ and results/.

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$EXAMPLE_DIR")"
WORK_DIR="$EXAMPLE_DIR/work"
SCRIPTS="$REPO_DIR/scripts"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
export MPLBACKEND=Agg

echo "==> Generating toy data (3 patients x 2 libraries, 5 planted cell types)"
python "$EXAMPLE_DIR/make_toy_data.py" --out "$EXAMPLE_DIR/toy_data"

cd "$WORK_DIR"
echo "==> 1. preprocess";       python "$SCRIPTS/preprocess.py" --raw-dir "$EXAMPLE_DIR/toy_data"
echo "==> 2. clustering";       python "$SCRIPTS/clustering.py" --skip-stability-check
echo "==> 3. annotate";         python "$SCRIPTS/annotate.py" --skip-celltypist
echo "==> 4. machine_learning"; python "$SCRIPTS/machine_learning.py"

echo "==> Checking results against the planted truth"
python "$EXAMPLE_DIR/check_example.py" --work "$WORK_DIR" --truth "$EXAMPLE_DIR/toy_data/truth.csv"

echo ""
echo "Toy outputs: $WORK_DIR/data (h5ad), $WORK_DIR/results (CSV, JSON, figures/)"
