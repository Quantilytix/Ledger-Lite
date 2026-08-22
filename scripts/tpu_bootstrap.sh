#!/bin/bash
set -euo pipefail
UV=/home/Tinevimbo/.local/bin/uv
if [ ! -d "/home/Tinevimbo/qx-venv" ]; then
  "$UV" venv /home/Tinevimbo/qx-venv --python 3.12
fi
PY=/home/Tinevimbo/qx-venv/bin/python
"$UV" pip install --python "$PY" -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"$UV" pip install --python "$PY" 'google-tunix[prod]' transformers datasets pyyaml huggingface_hub qwix optax flax safetensors numpy
"$UV" pip install --python "$PY" --reinstall jax==0.10.2 jaxlib==0.10.2 \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"$PY" -c 'import jax; print("device_count", jax.device_count()); print(jax.devices())'
