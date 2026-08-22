#!/bin/bash
# Separate venv so we do not touch the running Tunix/JAX train env.
set -euo pipefail
UV=/home/Tinevimbo/.local/bin/uv
VENV=/home/Tinevimbo/gguf-venv
LLAMA=/home/Tinevimbo/llama.cpp
"$UV" python install 3.12
"$UV" venv --python 3.12 "$VENV"
"$UV" pip install --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
"$UV" pip install --python "$VENV/bin/python" transformers safetensors sentencepiece numpy gguf protobuf huggingface_hub
if [ ! -d "$LLAMA/.git" ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA"
fi
"$VENV/bin/python" - <<'PY'
import json, tarfile, urllib.request
from pathlib import Path
dest = Path("/home/Tinevimbo/llama.cpp/prebuilt")
dest.mkdir(parents=True, exist_ok=True)
if any(dest.rglob("llama-quantize")):
    print("quantize already present")
    raise SystemExit(0)
api = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
with urllib.request.urlopen(api, timeout=60) as resp:
    release = json.loads(resp.read().decode())
url = name = None
for asset in release.get("assets", []):
    n = asset.get("name", "")
    if n.endswith("bin-ubuntu-x64.tar.gz") and "vulkan" not in n and "sycl" not in n:
        url, name = asset["browser_download_url"], n
        break
if not url:
    raise SystemExit("no ubuntu-x64 tar.gz in latest llama.cpp release")
archive = dest / name
print("downloading", url)
urllib.request.urlretrieve(url, archive)
with tarfile.open(archive, "r:gz") as tf:
    tf.extractall(dest)
print("extracted", dest)
print("quantize", next(dest.rglob("llama-quantize"), None))
print("cli", next(dest.rglob("llama-cli"), None))
PY
"$VENV/bin/python" -c "import torch, transformers, gguf; print('gguf-env-ok', torch.__version__)"
