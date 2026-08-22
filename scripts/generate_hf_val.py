"""Generate val_preds.jsonl from a merged HuggingFace checkpoint (CPU)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ledgerlite.prepare import load_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    model.eval()
    rows = load_jsonl(ROOT / "data" / "sft" / "val.jsonl")[: args.n]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            prompt = tok.apply_chat_template(
                row["messages"][:-1], tokenize=False, add_generation_prompt=True
            )
            inputs = tok(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(
                out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            handle.write(
                json.dumps(
                    {
                        "tenant_id": row.get("tenant_id"),
                        "gold": row["gold"],
                        "completion": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (i + 1) % 16 == 0:
                print(f"generated {i + 1}/{len(rows)}", flush=True)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
