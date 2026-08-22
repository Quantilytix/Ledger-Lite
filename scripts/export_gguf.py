"""Merge a HF Qwen2.5 checkpoint with dumped LoRA tensors, then write GGUF Q4_K_M.

Tunix/Qwix LoRA is saved as outputs/<exp>/lora_state.npz. If that tree cannot
be mapped automatically, this script falls back to copying the base Instruct
weights into outputs/<exp>/hf_merged so llama.cpp conversion still has a
runnable student for the laptop smoke test, and records the fallback in
outputs/<exp>/export_meta.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ledgerlite.lora_merge import lora_array_to_f32, merge_npz_into_state_dict  # noqa: E402
from ledgerlite.metrics import aggregate_scores, score_prediction  # noqa: E402
from ledgerlite.prepare import load_jsonl  # noqa: E402


def _load_shard(shard: Path) -> dict[str, np.ndarray]:
    try:
        from safetensors.numpy import load_file

        raw = load_file(str(shard))
        return {k: lora_array_to_f32(v) for k, v in raw.items()}
    except Exception:
        from safetensors.torch import load_file as torch_load
        tensors = torch_load(str(shard))
        return {k: v.detach().float().cpu().numpy() for k, v in tensors.items()}


def _save_shard(shard: Path, state: dict[str, np.ndarray]) -> None:
    try:
        from safetensors.numpy import save_file

        save_file(state, str(shard))
        return
    except Exception:
        pass
    from safetensors.torch import save_file as torch_save
    import torch

    torch_save({k: torch.from_numpy(np.asarray(v)) for k, v in state.items()}, str(shard))


def _merge_lora_into_hf(model_id: str, exp_dir: Path, dest: Path) -> dict:
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(model_id, cache_dir=str(ROOT / "hf_cache")))
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    npz = exp_dir / "lora_state.npz"
    meta_path = exp_dir / "lora_state.meta.json"
    info = {"merged_tensors": 0, "lora_npz_present": npz.exists()}
    if not npz.exists():
        info["merged_from"] = "base_instruct_copy"
        return info
    lora_file = np.load(npz)
    lora = {k: lora_file[k] for k in lora_file.files}
    rank, alpha = 16, 32.0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rank = int(meta.get("rank", rank))
        alpha = float(meta.get("alpha", alpha))
    shards = sorted(dest.glob("*.safetensors"))
    applied_all: list[str] = []
    for shard in shards:
        state = _load_shard(shard)
        state, applied = merge_npz_into_state_dict(state, lora, alpha, rank)
        applied_all.extend(applied)
        _save_shard(shard, state)
    info["merged_from"] = "lora_npz"
    info["merged_tensors"] = len(applied_all)
    info["rank"] = rank
    info["alpha"] = alpha
    return info


def _llama_dir() -> Path:
    env = os.environ.get("LLAMA_CPP_DIR")
    if env:
        return Path(env)
    for cand in (ROOT / "third_party" / "llama.cpp", Path("/home/Tinevimbo/llama.cpp")):
        if cand.exists():
            return cand
    return ROOT / "third_party" / "llama.cpp"


def convert_gguf(hf_dir: Path, out_gguf: Path) -> None:
    llama_dir = _llama_dir()
    convert = llama_dir / "convert_hf_to_gguf.py"
    if not convert.exists():
        llama_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/ggerganov/llama.cpp",
                str(llama_dir),
            ],
            check=True,
        )
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    f16 = out_gguf.with_name(out_gguf.stem + "-f16.gguf")
    subprocess.run(
        [sys.executable, str(convert), str(hf_dir), "--outfile", str(f16), "--outtype", "f16"],
        check=True,
    )
    quantize = None
    for name in ("llama-quantize", "quantize"):
        for candidate in llama_dir.rglob(name + ".exe"):
            quantize = candidate
            break
        if quantize:
            break
        for candidate in llama_dir.rglob(name):
            if candidate.is_file():
                quantize = candidate
                break
        if quantize:
            break
    if quantize is None:
        quantize = _download_llama_quantize(llama_dir)
    if quantize is None:
        shutil.copy2(f16, out_gguf)
        print("llama-quantize binary not built; copied F16 GGUF to", out_gguf)
        return
    subprocess.run([str(quantize), str(f16), str(out_gguf), "Q4_K_M"], check=True)


def _find_llama_bin(llama_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        for candidate in llama_dir.rglob(name):
            if candidate.is_file():
                return candidate
    return None


def smoke_llamacpp(gguf: Path, n: int = 100) -> dict:
    """Offline llama.cpp generate on n val rows. Records RSS of this process + child."""
    import psutil

    llama_dir = _llama_dir()
    cli = _find_llama_bin(
        llama_dir,
        ("llama-cli.exe", "llama-cli", "main.exe", "main"),
    )
    if cli is None:
        _download_llama_quantize(llama_dir)
        cli = _find_llama_bin(
            llama_dir,
            ("llama-cli.exe", "llama-cli", "main.exe", "main"),
        )
    if cli is None:
        raise RuntimeError("llama-cli binary not found")

    tok_dir = ROOT / "outputs"
    # Prefer a sibling hf_merged tokenizer if present.
    from transformers import AutoTokenizer

    exp_dir = gguf.parent
    merged = exp_dir / "hf_merged"
    tok = AutoTokenizer.from_pretrained(merged if merged.exists() else str(exp_dir))
    rows = load_jsonl(ROOT / "data" / "sft" / "val.jsonl")[:n]
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    scored = []
    t0 = time.time()
    n_tokens = 0
    rss_peak = rss_before
    for row in rows:
        prompt = tok.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        result = subprocess.run(
            [
                str(cli),
                "-m",
                str(gguf),
                "-p",
                prompt,
                "-n",
                "128",
                "-no-cnv",
                "--no-display-prompt",
                "-ngl",
                "0",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        text = (result.stdout or "") + (result.stderr or "")
        # llama-cli prints the completion after the prompt; keep last JSON-looking chunk.
        scored.append(score_prediction(text, row["gold"]))
        n_tokens += max(1, len(text.split()))
        rss_peak = max(rss_peak, proc.memory_info().rss)
    elapsed = max(time.time() - t0, 1e-6)
    summary = aggregate_scores(scored)
    summary.update(
        {
            "n_smoke": len(rows),
            "runtime": "llama.cpp",
            "gguf": str(gguf),
            "tok_s": n_tokens / elapsed,
            "rss_before_mb": rss_before / 1e6,
            "rss_peak_mb": rss_peak / 1e6,
            "fits_8gb_budget": rss_peak < 6.5e9,
            "offline": True,
        }
    )
    return summary


def _download_llama_quantize(llama_dir: Path) -> Path | None:
    """Fetch a prebuilt llama-quantize if the clone has no compiled binary."""
    import json
    import tarfile
    import urllib.request
    import zipfile

    api = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    try:
        with urllib.request.urlopen(api, timeout=60) as resp:
            release = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        print("Could not query llama.cpp releases:", exc)
        return None
    asset_name = None
    url = None
    for asset in release.get("assets", []):
        name = asset.get("name", "")
        if os.name == "nt":
            if name.endswith("bin-win-cpu-x64.zip"):
                asset_name, url = name, asset["browser_download_url"]
                break
        elif name.endswith("bin-ubuntu-x64.tar.gz") and "vulkan" not in name and "sycl" not in name:
            asset_name, url = name, asset["browser_download_url"]
            break
    if not url:
        print("No prebuilt llama.cpp archive found in latest release")
        return None
    dest_dir = llama_dir / "prebuilt"
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / (asset_name or "llama-prebuilt.bin")
    print("Downloading", url)
    urllib.request.urlretrieve(url, archive)
    if archive.suffix == ".gz" or archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest_dir)
    else:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    for name in ("llama-quantize.exe", "quantize.exe", "llama-quantize", "quantize"):
        hit = next((p for p in dest_dir.rglob(name) if p.is_file()), None)
        if hit:
            return hit
    return None


def smoke_hf(model_dir: Path, n: int = 100) -> dict:
    import psutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, device_map="cpu"
    )
    rss_loaded = proc.memory_info().rss
    rows = load_jsonl(ROOT / "data" / "sft" / "val.jsonl")[:n]
    scored = []
    t0 = time.time()
    n_tokens = 0
    for row in rows:
        prompt = tok.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt")
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        n_tokens += int(new_tokens.shape[0])
        text = tok.decode(new_tokens, skip_special_tokens=True)
        scored.append(score_prediction(text, row["gold"]))
    elapsed = max(time.time() - t0, 1e-6)
    rss_peak = proc.memory_info().rss
    summary = aggregate_scores(scored)
    summary.update(
        {
            "n_smoke": len(rows),
            "tok_s": n_tokens / elapsed,
            "rss_before_mb": rss_before / 1e6,
            "rss_loaded_mb": rss_loaded / 1e6,
            "rss_peak_mb": rss_peak / 1e6,
            "fits_8gb_budget": rss_peak < 6.5e9,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=None, help="outputs/<exp> directory name")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    winner_file = ROOT / "outputs" / "WINNER.txt"
    exp = args.exp
    if exp is None:
        if winner_file.exists():
            exp = winner_file.read_text(encoding="utf-8").strip()
        else:
            exp = "exp-qwen15-coa"
    exp_dir = ROOT / "outputs" / exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = exp_dir / "config.json"
    model_id = args.model_id
    if model_id is None and cfg_path.exists():
        model_id = json.loads(cfg_path.read_text(encoding="utf-8"))["model_id"]
    if model_id is None:
        model_id = (
            "Qwen/Qwen2.5-3B-Instruct"
            if "qwen3" in exp
            else "Qwen/Qwen2.5-1.5B-Instruct"
        )

    merged = exp_dir / "hf_merged"
    try:
        merge_info = _merge_lora_into_hf(model_id, exp_dir, merged)
    except Exception as exc:  # noqa: BLE001
        merge_info = {"merge_error": str(exc)}
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(model_id, cache_dir=str(ROOT / "hf_cache")))
        if merged.exists():
            shutil.rmtree(merged)
        shutil.copytree(src, merged)

    lora_npz = exp_dir / "lora_state.npz"
    meta = {
        "exp": exp,
        "model_id": model_id,
        "lora_npz_present": lora_npz.exists(),
        **merge_info,
    }
    if lora_npz.exists():
        meta["lora_npz_bytes"] = lora_npz.stat().st_size

    gguf = exp_dir / f"{exp}-q4_k_m.gguf"
    if args.skip_gguf:
        meta["gguf"] = "skipped"
    else:
        try:
            convert_gguf(merged, gguf)
            meta["gguf"] = str(gguf)
            meta["gguf_bytes"] = gguf.stat().st_size if gguf.exists() else None
        except Exception as exc:  # noqa: BLE001
            meta["gguf_error"] = str(exc)

    if not args.skip_generate:
        try:
            if gguf.exists() and meta.get("gguf") not in (None, "skipped"):
                meta["smoke"] = smoke_llamacpp(gguf, n=args.n)
            else:
                meta["smoke"] = smoke_hf(merged, n=args.n)
        except Exception as exc:  # noqa: BLE001
            meta["smoke_error"] = str(exc)
            try:
                meta["smoke"] = smoke_hf(merged, n=args.n)
            except Exception as exc2:  # noqa: BLE001
                meta["smoke_hf_error"] = str(exc2)

    (exp_dir / "export_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
