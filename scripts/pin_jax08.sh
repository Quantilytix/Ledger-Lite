#!/bin/bash
set -euo pipefail
UV=/home/Tinevimbo/.local/bin/uv
PY=/home/Tinevimbo/qx-venv/bin/python
# flax 0.12.8 needs jax.experimental.hijax.MutableHiType (not in jax 0.11).
"$UV" pip install --python "$PY" --upgrade \
  'jax[tpu]==0.8.1' \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"$PY" -c 'import jax; from jax.experimental import hijax; print("jax", jax.__version__, hasattr(hijax,"MutableHiType")); print(jax.device_count())'
