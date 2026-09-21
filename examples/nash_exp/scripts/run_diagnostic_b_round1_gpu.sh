#!/usr/bin/env bash
set -Eeuo pipefail

NASH_EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$NASH_EXPERIMENT_ROOT"
NASH_RUN_OUTPUT="${NASH_OUTPUT_DIR:-$NASH_EXPERIMENT_ROOT/artifacts/diagnostic_b/round1_qwen25math7b_math500_test500_p1024_g3072}"
NASH_RUN_CONFIG="${NASH_CONFIG:-$NASH_EXPERIMENT_ROOT/configs/diagnostic_b_round1.yaml}"
NASH_RUN_PYTHON="${NASH_PYTHON:-python3}"
mkdir -p "$NASH_RUN_OUTPUT"
NASH_RUN_LOG="$NASH_RUN_OUTPUT/diagnostic_b_round1.log"
exec > >(tee -a "$NASH_RUN_LOG") 2>&1
export NASH_LOG_TEE=1
trap 'rc=$?; printf "Launcher failed: exit=%s line=%s\nReturn this one log file: %s\nResume: bash scripts/run_diagnostic_b_round1_gpu.sh\n" "$rc" "$LINENO" "$NASH_RUN_LOG"; exit "$rc"' ERR
trap 'printf "Launcher interrupted. Caches preserved. Return: %s\n" "$NASH_RUN_LOG"; exit 130' INT TERM
printf 'UTC: %s\nLocal: %s\nCommand:' "$(date -u --iso-8601=seconds)" "$(date --iso-8601=seconds)"
printf ' %q' bash scripts/run_diagnostic_b_round1_gpu.sh "$@"
printf '\nResume: bash scripts/run_diagnostic_b_round1_gpu.sh\n'
"$NASH_RUN_PYTHON" -m diagnostic_b preflight --config "$NASH_RUN_CONFIG" --output-dir "$NASH_RUN_OUTPUT"
"$NASH_RUN_PYTHON" -m diagnostic_b all --config "$NASH_RUN_CONFIG" --output-dir "$NASH_RUN_OUTPUT" --resume "$@"
printf 'Completed. Results: %s\nSingle log: %s\n' "$NASH_RUN_OUTPUT" "$NASH_RUN_LOG"

# source .venv/bin/activate  # From the veRL root, after setup_server.sh.
# Eight independent TP1 engines: four train replicas and four held-out replicas.
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# NASH_TENSOR_PARALLEL_SIZE=1 \
# NASH_PARALLEL_SPLITS=1 \
# NASH_REPLICAS_PER_SPLIT=4 \
# NASH_OUTPUT_DIR=/root/my_gpro/examples/nash_exp/artifacts/diagnostic_b/round1_qwen25math7b_math500_test500_p1024_g3072 \
# bash examples/nash_exp/scripts/run_diagnostic_b_round1_gpu.sh

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# NASH_TENSOR_PARALLEL_SIZE=1 \
# NASH_PARALLEL_SPLITS=1 \
# NASH_REPLICAS_PER_SPLIT=4 \
# bash examples/nash_exp/scripts/run_diagnostic_b_round1_gpu.sh
