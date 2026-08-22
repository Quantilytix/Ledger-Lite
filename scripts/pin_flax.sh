#!/bin/bash
set -euo pipefail
UV=/home/Tinevimbo/.local/bin/uv
PY=/home/Tinevimbo/qx-venv/bin/python
"$UV" pip install --python "$PY" flax==0.10.7
"$PY" -c 'import jax, flax, tunix, qwix; print("ok", jax.__version__, flax.__version__, jax.device_count())'
