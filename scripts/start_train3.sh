#!/bin/bash
set -euo pipefail
LOG=/home/Tinevimbo/qx-foundational-model/outputs/exp-qwen3-coa/train.log
PY=/home/Tinevimbo/qx-venv/bin/python
mkdir -p /home/Tinevimbo/qx-foundational-model/outputs/exp-qwen3-coa
: > "$LOG"
cd /home/Tinevimbo/qx-foundational-model
export PYTHONPATH=src
nohup "$PY" -u scripts/train_sft.py --config configs/sft_qwen3.yaml > "$LOG" 2>&1 < /dev/null &
echo STARTED
sleep 2
head -n 8 "$LOG"
