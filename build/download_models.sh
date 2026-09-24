#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

output_dir=$(realpath -m "$1")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/versions.env"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is required: https://huggingface.co/docs/huggingface_hub/guides/cli" >&2
  exit 1
fi
mkdir -p "$output_dir"
if find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "model directory is not empty: $output_dir" >&2
  exit 1
fi

hf download "$TARGET_MODEL_REPO" \
  --revision "$TARGET_MODEL_REVISION" \
  --local-dir "$output_dir/target"
hf cache verify "$TARGET_MODEL_REPO" \
  --revision "$TARGET_MODEL_REVISION" \
  --local-dir "$output_dir/target" \
  --fail-on-missing-files --fail-on-extra-files

hf download "$DFLASH_MODEL_REPO" \
  --revision "$DFLASH_MODEL_REVISION" \
  --local-dir "$output_dir/dflash2"
hf cache verify "$DFLASH_MODEL_REPO" \
  --revision "$DFLASH_MODEL_REVISION" \
  --local-dir "$output_dir/dflash2" \
  --fail-on-missing-files --fail-on-extra-files

test -f "$output_dir/target/config.json"
test -f "$output_dir/dflash2/config.json"
echo "downloaded and verified both pinned model revisions in $output_dir"
