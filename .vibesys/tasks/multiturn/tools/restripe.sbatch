#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --job-name=vs-restripe

# Copies a sharded-state artifact from SRC into a pre-striped DST and
# verifies the copy (file count, per-file sizes, stripe count, spot cmp).
# Submit via tools/submit.sh (not `sbatch` directly), which exports
# SITE_*/PLATFORM_* from the site config (see config/sites/<site>.toml) and
# supplies --account/--partition/--output:
#
#   tools/submit.sh tools/restripe.sbatch
#
# DST must already exist and be striped before this runs, using this site's
# stripe count/size ($SITE_LUSTRE_STRIPE_COUNT / $SITE_LUSTRE_STRIPE_SIZE):
#   mkdir DST && lfs setstripe -c "$SITE_LUSTRE_STRIPE_COUNT" -S "$SITE_LUSTRE_STRIPE_SIZE" DST
# Striping evens out per-rank read time (each reader's stream spreads across
# more OSTs instead of hammering the single OST an unstriped 4 GiB part
# lands on), but it does not raise the per-node Lustre read cap (~3.5 GB/s
# aggregate on the test cluster); it only makes ranks finish closer together.
set -euo pipefail

: "${SITE_LUSTRE_STRIPE_COUNT:?run this via tools/submit.sh, which exports SITE_* vars}"
: "${SITE_LUSTRE_STRIPE_SIZE:?run this via tools/submit.sh, which exports SITE_* vars}"
: "${SITE_SHARDED_ARTIFACT:?run this via tools/submit.sh, which exports SITE_* vars}"
: "${SITE_SHARDED_ARTIFACT_STRIPED:?run this via tools/submit.sh, which exports SITE_* vars}"

SRC="${SRC:-$SITE_SHARDED_ARTIFACT}"
DST="${DST:-$SITE_SHARDED_ARTIFACT_STRIPED}"

echo "=== $(date -Is) start ==="
echo "SRC=$SRC"
echo "DST=$DST"
echo "expected stripe: count=$SITE_LUSTRE_STRIPE_COUNT size=$SITE_LUSTRE_STRIPE_SIZE"

lfs getstripe -d "$DST"
DST_STRIPE_COUNT="$(lfs getstripe -d "$DST" | awk '/lmm_stripe_count:/ {print $2}')"
if [ "$DST_STRIPE_COUNT" != "$SITE_LUSTRE_STRIPE_COUNT" ]; then
  echo "STRIPE COUNT MISMATCH: DST has $DST_STRIPE_COUNT, site config expects $SITE_LUSTRE_STRIPE_COUNT"
else
  echo "stripe count OK: $DST_STRIPE_COUNT"
fi

echo "=== copying 56 weight files (8-way parallel) ==="
T0=$(date +%s.%N)
ls "$SRC"/model-rank-*.safetensors | xargs -n1 -P8 -I{} bash -c '
  src="{}"
  base=$(basename "$src")
  dst="'"$DST"'/$base"
  cp "$src" "$dst"
  echo "copied $base"
'
T1=$(date +%s.%N)
echo "=== weight copy done in $(echo "$T1 - $T0" | bc) s ==="

echo "=== copying small config/tokenizer files ==="
T2=$(date +%s.%N)
find "$SRC" -maxdepth 1 -type f -not -name '*.safetensors' -print0 | xargs -0 -n1 -P8 -I{} cp {} "$DST"/
T3=$(date +%s.%N)
echo "=== small-file copy done in $(echo "$T3 - $T2" | bc) s ==="

TOTAL=$(echo "$T3 - $T0" | bc)
echo "=== TOTAL COPY TIME: $TOTAL s ==="

echo "=== verification: file count ==="
SRC_COUNT=$(find "$SRC" -maxdepth 1 -type f | wc -l)
DST_COUNT=$(find "$DST" -maxdepth 1 -type f | wc -l)
echo "src=$SRC_COUNT dst=$DST_COUNT"
if [ "$SRC_COUNT" != "$DST_COUNT" ]; then
  echo "FILE COUNT MISMATCH"
else
  echo "file count OK: $DST_COUNT"
fi

echo "=== verification: per-file sizes ==="
MISMATCH=0
for f in $(find "$SRC" -maxdepth 1 -type f -printf '%f\n'); do
  ssz=$(stat -c %s "$SRC/$f")
  dsz=$(stat -c %s "$DST/$f" 2>/dev/null || echo "MISSING")
  if [ "$ssz" != "$dsz" ]; then
    echo "SIZE MISMATCH: $f src=$ssz dst=$dsz"
    MISMATCH=1
  fi
done
if [ "$MISMATCH" = "0" ]; then
  echo "all sizes match"
else
  echo "SIZE VERIFICATION FAILED"
fi

echo "=== verification: stripe count on 2 files ==="
lfs getstripe "$DST"/model-rank-0-part-0.safetensors | grep lmm_stripe_count
lfs getstripe "$DST"/model-rank-2-part-7.safetensors | grep lmm_stripe_count

echo "=== verification: cmp 3 random weight files ==="
for f in model-rank-0-part-3.safetensors model-rank-1-part-9.safetensors model-rank-3-part-12.safetensors; do
  if cmp "$SRC/$f" "$DST/$f"; then
    echo "cmp OK: $f"
  else
    echo "CMP FAILED: $f"
  fi
done

echo "=== $(date -Is) done ==="
