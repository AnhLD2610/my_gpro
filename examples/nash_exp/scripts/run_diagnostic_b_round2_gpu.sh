#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${1:-}" != "--enable-round2" ]]; then
    printf '%s\n' 'Round 2 is disabled. Run only after explicit author approval, with --enable-round2 as the first argument.' >&2
    exit 2
fi
shift
NASH_EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$NASH_EXPERIMENT_ROOT"
NASH_RUN_OUTPUT="${NASH_OUTPUT_DIR:-$NASH_EXPERIMENT_ROOT/artifacts/diagnostic_b/round2}"
NASH_RUN_CONFIG="${NASH_CONFIG:-$NASH_EXPERIMENT_ROOT/configs/diagnostic_b_round2.yaml}"
NASH_RUN_PYTHON="${NASH_PYTHON:-python3}"
mkdir -p "$NASH_RUN_OUTPUT"
NASH_RUN_LOG="$NASH_RUN_OUTPUT/diagnostic_b_round2.log"
exec > >(tee -a "$NASH_RUN_LOG") 2>&1
export NASH_LOG_TEE=1
trap 'rc=$?; printf "Launcher failed: exit=%s line=%s\nReturn this one log file: %s\nResume: bash scripts/run_diagnostic_b_round2_gpu.sh --enable-round2\n" "$rc" "$LINENO" "$NASH_RUN_LOG"; exit "$rc"' ERR
trap 'printf "Launcher interrupted. Caches preserved. Return: %s\n" "$NASH_RUN_LOG"; exit 130' INT TERM
printf 'UTC: %s\nLocal: %s\nCommand:' "$(date -u --iso-8601=seconds)" "$(date --iso-8601=seconds)"
printf ' %q' bash scripts/run_diagnostic_b_round2_gpu.sh --enable-round2 "$@"
printf '\n'
"$NASH_RUN_PYTHON" -m diagnostic_b preflight --config "$NASH_RUN_CONFIG" --output-dir "$NASH_RUN_OUTPUT" --enable-round2
"$NASH_RUN_PYTHON" -m diagnostic_b all --config "$NASH_RUN_CONFIG" --output-dir "$NASH_RUN_OUTPUT" --enable-round2 --resume "$@"
printf 'Completed. Single log: %s\n' "$NASH_RUN_LOG"
