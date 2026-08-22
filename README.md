"""LedgerLite student — Qwen2.5 CoA-conditioned SFT on TPU.

Text-only (not multimodal). OCR/PDF is upstream. This repo fine-tunes
Qwen2.5-1.5B and Qwen2.5-3B with Tunix LoRA so a quantized GGUF can run
offline on an 8GB laptop for the Africa Deep Tech Challenge 2026.
"""

## Data

Chris's Gemini-labeled cut lives in `data/raw/` (gitignored). Prepare SFT chats:

```bash
python scripts/prepare_sft.py
```

That writes `data/sft/{train,val,test}.jsonl` plus `data/sft/meta.json`.
Account `id`s are dropped. Date/amount stay in the user prompt.

## Experiments

| Run | Base | Recipe |
|---|---|---|
| `exp-qwen15-coa` | Qwen2.5-1.5B-Instruct | CoA-conditioned LoRA SFT |
| `exp-qwen3-coa` | Qwen2.5-3B-Instruct | same data, larger student |

TPU: legacy Cloud TPU API, `v5litepod-8`, `us-west4-a`.

```bash
python scripts/tpu_ctl.py create
python scripts/tpu_ctl.py setup
python scripts/tpu_ctl.py sync
python scripts/tpu_ctl.py train configs/sft_qwen15.yaml
python scripts/tpu_ctl.py train configs/sft_qwen3.yaml
python scripts/eval_ledger.py --pred outputs/exp-qwen15-coa/val_preds.jsonl --gold data/sft/val.jsonl
```

Do not score `test` until a winner is frozen on val.
