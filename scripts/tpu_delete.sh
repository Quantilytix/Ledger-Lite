#!/usr/bin/env bash
set -euo pipefail
export ZONE="${ZONE:-us-west4-a}"
export TPU_NAME="${TPU_NAME:-ledgerlite-sft-v5e8}"
gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --quiet
