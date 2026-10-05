#!/usr/bin/env bash
#
# Extract a checkout tarball into node-local tmpfs and print the resulting
# workspace path.
#
# Usage:
#   WORKSPACE=$(stage_workspace.sh TARBALL [DEST_ROOT])
#
# Why: importing sglang from a checkout on Lustre costs ~85-107 s cold on
# the test cluster (client-side metadata cache misses over ~5700 files), versus ~20 s
# from tmpfs or from a warm cache (job 631029). Copying the tree file by
# file is slower still (177 s), but a single tarball is one sequential read,
# so extracting it into /dev/shm costs seconds. The tarball is produced by
# `git archive --format=tar HEAD | gzip -1 > <name>.tar.gz` at ship time.
#
# DEST_ROOT defaults to the site config's paths.tmpfs_root (see
# config/sites/<site>.toml, selected via $VIBESYS_SITE); the workspace is
# created as DEST_ROOT/<tarball basename without .tar.gz>-<pid> so concurrent
# jobs on a shared filesystem do not collide (tmpfs is per node anyway).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 TARBALL [DEST_ROOT]" >&2
  exit 2
fi

TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

TARBALL="$1"
if [[ $# -ge 2 ]]; then
  DEST_ROOT="$2"
else
  DEST_ROOT="$(python3 "$TASK_DIR/config/loader.py" paths.tmpfs_root)"
fi
name="$(basename -- "$TARBALL")"
name="${name%.tar.gz}"
name="${name%.tgz}"
name="${name%.tar}"
WORKSPACE="$DEST_ROOT/$name-$$"

mkdir -p -- "$WORKSPACE"
case "$TARBALL" in
  *.tar.gz|*.tgz) tar -xzf "$TARBALL" -C "$WORKSPACE" ;;
  *) tar -xf "$TARBALL" -C "$WORKSPACE" ;;
esac

if [[ ! -e "$WORKSPACE/sglang" ]]; then
  echo "$0: $TARBALL did not contain the checkout root (no sglang/ entry)" >&2
  exit 1
fi
echo "$WORKSPACE"
