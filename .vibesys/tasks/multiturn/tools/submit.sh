#!/usr/bin/env bash
#
# Submit a harness sbatch script through the site config.
#
# Usage:
#   tools/submit.sh SCRIPT [ARGS...]
#
# Evaluates `config/loader.py --shell` for the selected site
# ($VIBESYS_SITE, default "example"; see config/sites/<site>.toml) and
# exports the resulting SITE_*/PLATFORM_* variables so SCRIPT (and anything
# it srun's) can read them, then submits SCRIPT via sbatch with
# --account/--partition/--output taken from the site config. SCRIPT is no
# longer expected to hardcode those.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 SCRIPT [ARGS...]" >&2
  exit 2
fi

SCRIPT="$1"
shift

TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

eval "$(python3 "$TASK_DIR/config/loader.py" --shell)"
export SITE_NAME SITE_MODEL_PATH SITE_SHARDED_ARTIFACT SITE_SHARDED_ARTIFACT_STRIPED \
  SITE_AITER_JIT_DIR SITE_TMPFS_ROOT SITE_LOG_DIR SITE_CHECKOUT SITE_SLURM_ACCOUNT \
  SITE_SLURM_PARTITION SITE_SLURM_EDF SITE_LUSTRE_STRIPE_COUNT SITE_LUSTRE_STRIPE_SIZE \
  SITE_STARTUP_TIMEOUT_S PLATFORM_NAME PLATFORM_TP PLATFORM_ATTENTION_BACKEND \
  PLATFORM_PAGE_SIZE PLATFORM_MEM_FRACTION_STATIC PLATFORM_MAX_TOTAL_TOKENS \
  PLATFORM_HF_LOADER_DISABLE_MMAP PLATFORM_HF_LOADER_EXTRA_CONFIG

mkdir -p -- "$SITE_LOG_DIR"

exec sbatch \
  --account="$SITE_SLURM_ACCOUNT" \
  --partition="$SITE_SLURM_PARTITION" \
  --output="$SITE_LOG_DIR/%j.out" \
  "$SCRIPT" "$@"
