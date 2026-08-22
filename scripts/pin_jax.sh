#!/bin/bash
set -euo pipefail
UV=/home/Tinevimbo/.local/bin/uv
PY=/home/Tinevimbo/qx-venv/bin/python
"$UV" pip install --python "$PY" --upgrade \
  'jax[tpu]==0.6.2' \
  flax==0.10.6 \
  optax==0.2.4 \
  qwix==0.1.5 \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"$PY" -c 'import jax, flax, tunix; print("jax", jax.__version__, "flax", flax.__version__, "devices", jax.device_count())'
