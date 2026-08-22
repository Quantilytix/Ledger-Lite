#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ID="${PROJECT_ID:-tpu-builder1}"
export ZONE="${ZONE:-us-west4-a}"
export TPU_NAME="${TPU_NAME:-ledgerlite-sft-v5e8}"
export ACCEL="${ACCEL:-v5litepod-8}"
export RUNTIME="${RUNTIME:-v2-alpha-tpuv5-lite}"
gcloud config set project "$PROJECT_ID"
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --zone="$ZONE" \
  --accelerator-type="$ACCEL" \
  --version="$RUNTIME"
