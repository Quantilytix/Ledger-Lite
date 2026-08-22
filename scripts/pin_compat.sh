#!/bin/bash
set -euo pipefail
UV=/home/Tinevimbo/.local/bin/uv
PY=/home/Tinevimbo/qx-venv/bin/python
"$UV" pip install --python "$PY" --upgrade \
  'jax[tpu]==0.5.3' flax==0.10.4 \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"$PY" -c 'import jax, flax; from tunix.sft import peft_trainer; from tunix.models.qwen2 import model; print("ok", jax.__version__, flax.__version__, jax.device_count())'
