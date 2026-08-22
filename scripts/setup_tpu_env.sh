#!/usr/bin/env bash
set -euo pipefail
pip install -U pip
# Prefer the 3.12 venv created by tpu_ctl.py setup (Tunix needs Python >=3.11).
if [ -x "$HOME/qx-venv/bin/pip" ]; then
  PIP="$HOME/qx-venv/bin/pip"
  PY="$HOME/qx-venv/bin/python"
else
  PIP="pip"
  PY="python3"
fi
"$PIP" install -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"$PIP" install 'google-tunix[prod]' transformers datasets pyyaml huggingface_hub qwix optax flax safetensors numpy
"$PY" -c "import jax; print('device_count', jax.device_count()); print(jax.devices())"
