"""Offline llama.cpp smoke test on a val subsample (Gate-1 8GB check)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ledgerlite.metrics import aggregate_scores, score_prediction  # noqa: E402
from ledgerlite.prepare import load_jsonl  # noqa: E402


def _find_bin(names: tuple[str, ...]) -> Path | None:
    roots = [
        ROOT / "third_party" / "llama.cpp",
        Path("/home/Tinevimbo/llama.cpp"),
    ]
    env = os.environ.get("LLAMA_CPP_DIR")
    if env:
        roots.insert(0, Path(env))
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            hit = next((p for p in root.rglob(name) if p.is_file()), None)
            if hit:
                return hit
    return None


def _wait_health(url: str, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5)
    raise SystemExit(f"llama-server did not become healthy: {last}")


def _complete(url: str, prompt: str, n_predict: int) -> str:
    payload = json.dumps(
        {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": True}
    ).encode()
    req = urllib.request.Request(
        url + "/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode())
    return str(body.get("content") or body.get("completion") or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--tok", default=None, help="HF tokenizer dir or model id")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    gguf = Path(args.gguf)
    if not gguf.exists():
        raise SystemExit(f"missing {gguf}")
    server_bin = _find_bin(("llama-server.exe", "llama-server"))
    if server_bin is None:
        raise SystemExit("llama-server not found")

    from transformers import AutoTokenizer

    tok_src = args.tok or str(gguf.parent / "hf_merged")
    if not Path(tok_src).exists():
        tok_src = (
            "Qwen/Qwen2.5-3B-Instruct"
            if "qwen3" in gguf.name
            else "Qwen/Qwen2.5-1.5B-Instruct"
        )
    tok = AutoTokenizer.from_pretrained(tok_src)
    rows = load_jsonl(ROOT / "data" / "sft" / "val.jsonl")[: args.n]

    try:
        import psutil

        proc = psutil.Process(os.getpid())
        rss_before = proc.memory_info().rss
    except Exception:
        proc = None
        rss_before = 0

    url = f"http://127.0.0.1:{args.port}"
    env = os.environ.copy()
    libdir = str(server_bin.parent)
    env["LD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = libdir + os.pathsep + env.get("PATH", "")
    server = subprocess.Popen(
        [
            str(server_bin),
            "-m",
            str(gguf),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "-ngl",
            "0",
            "--log-disable",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        _wait_health(url)
        scored = []
        n_tokens = 0
        t0 = time.time()
        rss_peak = rss_before
        for i, row in enumerate(rows):
            prompt = tok.apply_chat_template(
                row["messages"][:-1], tokenize=False, add_generation_prompt=True
            )
            text = _complete(url, prompt, args.max_new)
            scored.append(score_prediction(text, row["gold"]))
            n_tokens += max(1, len(text.split()))
            if proc is not None:
                try:
                    rss_peak = max(rss_peak, proc.memory_info().rss)
                    rss_peak = max(rss_peak, psutil.Process(server.pid).memory_info().rss)
                except Exception:
                    pass
            if (i + 1) % 10 == 0:
                print(f"smoke {i + 1}/{len(rows)}", flush=True)
        elapsed = max(time.time() - t0, 1e-6)
        summary = aggregate_scores(scored)
        summary.update(
            {
                "n_smoke": len(rows),
                "runtime": "llama.cpp",
                "gguf": str(gguf),
                "gguf_bytes": gguf.stat().st_size,
                "tok_s": n_tokens / elapsed,
                "elapsed_s": elapsed,
                "rss_before_mb": rss_before / 1e6,
                "rss_peak_mb": rss_peak / 1e6,
                "fits_8gb_budget": (rss_peak < 6.5e9) if rss_peak else None,
                "offline": True,
            }
        )
        out = args.out or (gguf.parent / "smoke_gguf.json")
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()
