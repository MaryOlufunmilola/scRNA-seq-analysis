#!/usr/bin/env bash
# Run the test suite, then each pipeline step in order, logging output to logs/.
#
# Usage:
#   bash run_pipeline.sh                            # tests + the four core steps
#   bash run_pipeline.sh --all                      # also signature export and cell to cell communication
#   bash run_pipeline.sh --all --copy-signature-to ../TCGA-Cancer-Analysis/data
#   bash run_pipeline.sh --skip-tests               # skip pytest (e.g. when rerunning a single change)
#
# Run from the repo root inside the environment built from requirements.txt
# (or requirements-lock.txt to reproduce the README results exactly).
#
# Each step's run time and peak memory are recorded in logs/timing.tsv and
# printed at the end.

set -euo pipefail

RUN_ALL=false
RUN_TESTS=true
SIGNATURE_DEST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) RUN_ALL=true; shift ;;
    --skip-tests) RUN_TESTS=false; shift ;;
    --copy-signature-to) SIGNATURE_DEST="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")"
mkdir -p logs

TIMING="logs/timing.tsv"
printf "step\tseconds\tpeak_memory_mb\n" > "$TIMING"
TOTAL_START=$SECONDS

# Peak memory comes from /usr/bin/time: -l on macOS (bytes), -v on Linux (kilobytes).
if [[ "$(uname)" == "Darwin" ]]; then TIME_FLAG="-l"; else TIME_FLAG="-v"; fi
if [[ ! -x /usr/bin/time ]]; then TIME_FLAG=""; fi

run_step() {
  local name="$1"; shift
  local log="logs/${name}.log"
  echo ""
  echo "==> ${name}"
  local start=$SECONDS
  if [[ -n "$TIME_FLAG" ]]; then
    /usr/bin/time "$TIME_FLAG" "$@" 2>&1 | tee "$log"
  else
    "$@" 2>&1 | tee "$log"
  fi
  local secs=$(( SECONDS - start ))
  local mem="NA"
  if [[ "$TIME_FLAG" == "-l" ]]; then
    mem=$(awk '/maximum resident set size/ {printf "%.0f", $1/1048576}' "$log")
  elif [[ "$TIME_FLAG" == "-v" ]]; then
    mem=$(awk -F: '/Maximum resident set size/ {printf "%.0f", $2/1024}' "$log")
  fi
  printf "%s\t%s\t%s\n" "$name" "$secs" "${mem:-NA}" >> "$TIMING"
  echo "    ${name}: ${secs} s, peak memory ${mem:-NA} MB"
}

if $RUN_TESTS; then
  run_step tests pytest --cov=scripts --cov-report=term-missing
fi

# Pipeline settings used for the README results. min_genes=100 is the script default;
# NYU_UCEC2's own doublet rate estimated by Scrublet (about 15 to 19%) is well above the 6% default.
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

TOTAL_SECS=$(( SECONDS - TOTAL_START ))
printf "total\t%s\tNA\n" "$TOTAL_SECS" >> "$TIMING"

echo ""
echo "Timing (also in ${TIMING}):"
column -t -s $'\t' "$TIMING" 2>/dev/null || cat "$TIMING"
echo "Total: $(( TOTAL_SECS / 60 )) min $(( TOTAL_SECS % 60 )) s"
echo ""
echo "Done. Outputs are in results/ and results/figures/; logs are in logs/."
