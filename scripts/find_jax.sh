#!/bin/bash
set -euo pipefail
UV=/home/Tinevimbo/.local/bin/uv
PY=/home/Tinevimbo/qx-venv/bin/python
for V in 0.10.2 0.10.0 0.9.2 0.9.0 0.7.2; do
  echo "TRY jax==$V"
  if "$UV" pip install --python "$PY" --reinstall "jax==$V" "jaxlib==$V" \
      -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; then
    if "$PY" -c 'import jax, flax, qwix; from tunix.sft import peft_trainer; print("OK", jax.__version__, flax.__version__, jax.device_count())'; then
      echo "LOCKED jax==$V"
      exit 0
    fi
  fi
done
echo FAIL
exit 1
