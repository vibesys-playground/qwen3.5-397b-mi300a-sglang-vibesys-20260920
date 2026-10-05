#!/bin/bash
#SBATCH --job-name=vs-savesharded
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=03:00:00

# Runs tools/save_sharded.py against the checkout at $SITE_CHECKOUT,
# additionally teeing its output to $SITE_LOG_DIR. Submit via tools/submit.sh
# (not `sbatch` directly), which exports SITE_*/PLATFORM_* from the site
# config (see config/sites/<site>.toml) and supplies
# --account/--partition/--output:
#
#   tools/submit.sh tools/save_sharded.sbatch
#
# save_sharded.py itself defaults --source/--output/--tp/--max-part-bytes to
# the site/platform config; override those on the script's own command line
# below if regenerating a different artifact.

: "${SITE_CHECKOUT:?run this via tools/submit.sh, which exports SITE_* vars}"
: "${SITE_LOG_DIR:?run this via tools/submit.sh, which exports SITE_* vars}"
: "${SITE_SLURM_EDF:?run this via tools/submit.sh, which exports SITE_* vars}"
: "${SITE_AITER_JIT_DIR:?run this via tools/submit.sh, which exports SITE_* vars}"

set -x
date -u
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SITE_CHECKOUT=${SITE_CHECKOUT}"
echo "SITE_LOG_DIR=${SITE_LOG_DIR}"

mkdir -p -- "$SITE_LOG_DIR"

srun --overlap --environment="$SITE_SLURM_EDF" bash -c '
set -x
set -o pipefail
cd "'"$SITE_CHECKOUT"'" || exit 1
export AITER_JIT_DIR="'"$SITE_AITER_JIT_DIR"'"
date -u
python3 -c "import sglang; print(sglang.__file__)"
python3 "'"$SITE_CHECKOUT"'/.vibesys/tasks/multiturn/tools/save_sharded.py" 2>&1 | tee "'"$SITE_LOG_DIR"'/'"${SLURM_JOB_ID}"'.save.log"
echo SAVE_EXIT=${PIPESTATUS[0]}
date -u
'
