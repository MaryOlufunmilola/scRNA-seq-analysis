#!/usr/bin/env bash
# Run the test suite, then each pipeline step in order, logging output to logs/.
#
# Usage:
#   bash run_pipeline.sh                            # tests + the four core steps
#   bash run_pipeline.sh --all                      # also signature export and cell-cell communication
#   bash run_pipeline.sh --all --copy-signature-to ../TCGA-Cancer-Analysis/data
#   bash run_pipeline.sh --skip-tests               # skip pytest (e.g. when re-running a single change)
#
# Run from the repo root inside the environment built from requirements.txt
# (or requirements-lock.txt to reproduce the README results exactly).

set -euo pipefail

RUN_ALL=false
RUN_TESTS=true
SIGNATURE_DEST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) RUN_ALL=true; shift ;;
    --skip-tests) RUN_TESTS=false; shift ;;
    --copy-signature-to) SIGNATURE_DEST="$2"; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")"
mkdir -p logs

run_step() {
  local name="$1"; shift
  echo ""
  echo "==> ${name}"
  "$@" 2>&1 | tee "logs/${name}.log"
}

if $RUN_TESTS; then
  run_step tests pytest --cov=scripts --cov-report=term-missing
fi

# Pipeline settings used for the README results. min_genes=100 is the script default;
# NYU_UCEC2's own Scrublet-estimated doublet rate (~15-19%) is well above the 6% default.
run_step preprocess python scripts/preprocess.py 
run_step clustering python scripts/clustering.py
run_step annotate python scripts/annotate.py
run_step machine_learning python scripts/machine_learning.py

if $RUN_ALL; then
  run_step export_signature_matrix python scripts/export_signature_matrix.py
  run_step cell_communication python scripts/cell_communication.py
fi

if [[ -n "$SIGNATURE_DEST" ]]; then
  if [[ ! -f results/cell_type_signature_matrix.csv ]]; then
    echo "No signature matrix found; run with --all to create it." >&2
    exit 1
  fi
  mkdir -p "$SIGNATURE_DEST"
  cp results/cell_type_signature_matrix.csv "$SIGNATURE_DEST"/
  echo "Copied signature matrix to $SIGNATURE_DEST"
fi

echo ""
echo "Done. Outputs are in results/ and results/figures/; logs are in logs/."
