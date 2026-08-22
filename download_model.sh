#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="model"
MODEL_FILE="${MODEL_DIR}/exp-qwen3-coa-q4_k_m.gguf"
PUBLIC_URL="https://storage.googleapis.com/ledgerlite-indaba-public/exp-qwen3-coa-q4_k_m.gguf"
EXPECTED_SIZE_MIN=1900000000

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
  FILE_SIZE=$(wc -c < "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null || echo 0)
  if [ "$FILE_SIZE" -ge "$EXPECTED_SIZE_MIN" ]; then
    echo "Model weights already present at ${MODEL_FILE} (${FILE_SIZE} bytes). Skipping download."
    exit 0
  else
    echo "Existing model file is incomplete (${FILE_SIZE} bytes). Re-downloading..."
    rm -f "$MODEL_FILE"
  fi
fi

echo "Downloading LedgerLite GGUF model from ${PUBLIC_URL}..."

if command -v curl >/dev/null 2>&1; then
  curl -L --retry 3 -o "$MODEL_FILE" "$PUBLIC_URL"
elif command -v wget >/dev/null 2>&1; then
  wget --tries=3 -O "$MODEL_FILE" "$PUBLIC_URL"
else
  echo "Error: Neither curl nor wget is available." >&2
  exit 1
fi

echo "Successfully downloaded ${MODEL_FILE}"
