#!/bin/bash
set -euo pipefail
UV=/home/Tinevimbo/.local/bin/uv
PY=/home/Tinevimbo/qx-venv/bin/python
"$UV" pip install --python "$PY" flax==0.12.8
"$UV" pip install --python "$PY" --reinstall jax==0.8.1 jaxlib==0.8.1 libtpu==0.0.30 \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"$PY" -c 'import jax, flax, qwix; from jax.experimental import hijax; from tunix.sft import peft_trainer; print("ok", jax.__version__, flax.__version__, hasattr(hijax,"MutableHiType"), jax.device_count())'
