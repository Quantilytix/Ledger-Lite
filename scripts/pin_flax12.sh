#!/bin/bash
set -euo pipefail
UV=/home/Tinevimbo/.local/bin/uv
PY=/home/Tinevimbo/qx-venv/bin/python
"$UV" pip install --python "$PY" flax==0.12.8
"$PY" -c 'import jax, flax, qwix; from tunix.sft import peft_trainer; from tunix.models.qwen2 import model as qwen2; print("ok", jax.__version__, flax.__version__, jax.device_count())'
