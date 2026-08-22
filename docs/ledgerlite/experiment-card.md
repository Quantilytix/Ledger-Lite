# LedgerLite experiment card (Gate 1)

**Project:** LedgerLite / Africa Deep Tech Challenge 2026  
**Student:** Qwen2.5 Instruct, text-only CoA-conditioned SFT  
**Hardware train:** TPU v5e-8 (`v5litepod-8`, `us-west4-a`)  
**Hardware submit:** consumer laptop, 8GB DDR4, offline GGUF

## Why not multimodal

The Qx Analytix export is `{input_text, target_json}`. `input_text` is the
already-extracted bank line. Raw OCR/PDF is not retained. Dates and amounts
are often absent from the memo, so they are passed in as structured fields
(the parser already has them). The student predicts debit/credit **code,
name, type** — never tenant-specific account `id`s.

## Runs

1. `exp-qwen15-coa` — Qwen/Qwen2.5-1.5B-Instruct, LoRA r=16 α=32, 1 epoch
2. `exp-qwen3-coa` — Qwen/Qwen2.5-3B-Instruct, same recipe

Winner rule: higher val exact-match of `(debit.code, credit.code)` with JSON
parse rate ≥ 95%. Tie-break: macro-F1 on normalized debit name. Watch
Miscellaneous Expense (~13.9% base rate).

## Frozen winner (val, n=256, test untouched)

| exp | JSON parse | schema | code exact | name-norm | macro-F1 debit | Misc pred |
|---|---|---|---|---|---|---|
| exp-qwen15-coa | 100% | 100% | 62.1% | 62.1% | 0.419 | 14.8% |
| **exp-qwen3-coa** | 100% | 100% | **65.6%** | 65.6% | **0.438** | 14.1% |

Winner: `exp-qwen3-coa` (Qwen2.5-3B-Instruct LoRA). Export: GGUF Q4_K_M ~1.8GB,
252 LoRA tensors merged.

- TPU path: `/home/Tinevimbo/qx-foundational-model/outputs/exp-qwen3-coa/exp-qwen3-coa-q4_k_m.gguf`
- GCS: `gs://tpu-builder1-indaba-ckpts/ledgerlite/exp-qwen3-coa-q4_k_m.gguf`
- Laptop: `outputs/exp-qwen3-coa/exp-qwen3-coa-q4_k_m.gguf` (copy with
  `gcloud storage cp gs://tpu-builder1-indaba-ckpts/ledgerlite/exp-qwen3-coa-q4_k_m.gguf outputs/exp-qwen3-coa/`)

Do not score `data/sft/test.jsonl` until this freeze is intentional.

## GGUF smoke (100 val rows, llama.cpp Q4_K_M, CPU)

Offline `llama-server` on the merged 3B student:

- JSON parse / schema: **100%**
- code exact-match: **86%** (100-row subsample; Tunix greedy on 256 was 65.6%)
- Misc Expense prediction rate: 37% (watch vs 13.9% base; subsample)
- Wall time: 159s (~1.6 s/row after one model load)
- Peak RSS: **5.5 GB** (`fits_8gb_budget: true`)
- Runtime: llama.cpp, `ngl=0`, no network at infer

## Data caveats

- Tenant-hash split; no business appears in more than one split
- Per-tenant charts of accounts; names are not a global taxonomy
- Exact-row duplicates are dropped before SFT
- Bank/Bank identity postings are held out of the primary mix
- No Gemini-vs-human preference labels in this cut (no DPO)

## Files

Prepared chats: `data/sft/` (gitignored). Metrics land in `outputs/<exp>/metrics.json`.
