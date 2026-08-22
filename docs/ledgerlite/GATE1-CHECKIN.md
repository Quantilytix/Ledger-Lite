# LedgerLite — Gate 1 check-in

**Project:** LedgerLite / Africa Deep Tech Challenge 2026 (Laptop LLM)  
**Date:** 19 August 2026  
**Owner:** Tinevimbo  
**Status:** Text-only student trained, winner frozen on val, Q4 GGUF exported. Waiting on multimodal (PDF/OCR) data.

This note is a project check-in, not a test-set score. Everything below is from runs that actually executed on a Google TPU, not a mock or dry-run.

---

## How we are going

Gate 1 is **25 August 2026**. We are on the **text posting student**, not vision. The TPU is already torn down; weights live in this repo (LoRA) and GCS (GGUF).

| stage | status | what that means |
|---|---|---|
| Data prep (Gemini JSONL → CoA chats) | **done** | train 34,165 / val 6,886 / test 8,780. JSONL gitignored. |
| TPU LoRA SFT — Qwen2.5-1.5B | **done** | 4,270 steps, 1 epoch, real `v5litepod-8` |
| TPU LoRA SFT — Qwen2.5-3B | **done** | 8,541 steps, 1 epoch, same recipe |
| Val pick winner (n=256) | **done** | **3B wins** (65.6% code exact-match, 100% JSON). Test **not** scored. |
| Merge LoRA → HF → GGUF Q4_K_M | **done** | 252 tensors merged, ~1.8 GB |
| llama.cpp smoke (100 val rows) | **done on TPU CPU** | 100% JSON, 86% code match on that slice, 5.5 GB RSS |
| Copy GGUF onto the submission laptop | **not yet** | SCP kept dropping. Pull from GCS (commands below). |
| Full val (6,886) then test | **not yet** | freeze stays on the 256-row val until we re-run generate |
| 0.5B / 7B / 2nd epoch / constrained JSON | **next (text)** | same recipe, different size or decode |
| Vision / PDF student | **blocked** | waiting on multimodal dump. This text model stays the posting head. |
| DPO / GRPO | **blocked** | no Gemini-vs-human pairs in this export |

**One-line:** trained, winner frozen, GGUF exists in the bucket; laptop copy and vision data are the open items.

---

## 1. What we are building

A small **offline** student that, given an extracted bank line plus that tenant’s chart of accounts, posts a double-entry journal:

```json
{
  "transaction_type": "DEBIT",
  "debit_account":  {"code": "6500", "name": "Fuel Expense", "type": "Expense"},
  "credit_account": {"code": "1000", "name": "Bank Account", "type": "Asset"}
}
```

Inference target: consumer laptop, **8GB RAM**, llama.cpp / GGUF, no cloud at serve time.

---

## 2. Why this week is text-only (not a mock of vision)

Chris’s cut is `{tenant_id, input_text, target_json}`.

- `input_text` is the **already-extracted** memo (`"Fuel"`, `"Payshap …"`), not a statement PDF.
- OCR/layout is upstream and **not retained** in this export.
- Dates and amounts are often missing from the memo (`"Fuel"` still has `amount: 600`). The parser already has them, so they stay in the **prompt**. The student is not asked to invent them.
- Account `id`s are per-tenant and not transferable. We train `code` / `name` / `type` only.

So the student task this week is **ledger posting given structured extraction + CoA**, not “read a PDF.” We are waiting on a multimodal dataset (raw statements / OCR pages) before standing up a vision student. This text student should remain the posting head when that data arrives.

---

## 3. What was a full real run

### Data prep (local)

Gemini-labeled JSONL → CoA-conditioned chat JSONL (`scripts/prepare_sft.py`):

| split | n in | dups dropped | Bank/Bank dropped | n out | tenants |
|---|---|---|---|---|---|
| train | 46,437 | 9,214 | 3,058 | **34,165** | 233 |
| val | 10,240 | 2,489 | 865 | **6,886** | 51 |
| test | 13,770 | 3,752 | 1,238 | **8,780** | 65 |

CoA is rebuilt per tenant from that split, capped at ~40 lines so 1.5B/3B fit `max_seq_len=2048`. Bank/Bank identity postings are held out of the primary mix (logged in `data/diagnostics/`). Names in free text; JSONL is gitignored and must not go to the public GitHub remote.

### Training (real TPU, not simulated)

Hardware: Cloud TPU **`v5litepod-8`**, zone **`us-west4-a`**, name `ledgerlite-sft-v5e8` (legacy TPU API, runtime `v2-alpha-tpuv5-lite`). **VM has been deleted** (chip-hour billing). JAX reported 8 devices. Stack that actually imported Tunix: `jax==0.10.2` + `flax==0.12.8` + `google-tunix[prod]`.

Both runs used the **same data recipe**. Only model size changed. LoRA rank 16, α 32, lr 2e-4, 1 epoch.

| run | base | mesh | batch | steps | outcome |
|---|---|---|---|---|---|
| `exp-qwen15-coa` | `Qwen/Qwen2.5-1.5B-Instruct` | (8, 1) fsdp/tp | 8 | **4,270** | LoRA saved |
| `exp-qwen3-coa` | `Qwen/Qwen2.5-3B-Instruct` | (4, 2) fsdp/tp | 4 | **8,541** | LoRA saved |

These were full one-epoch SFT jobs on the prepared train set, not toy steps and not mocked loss curves.

### Val selection (winner frozen; test untouched)

256 greedy completions per model. Rule: JSON parse ≥ 95%, then higher exact-match of `(debit.code, credit.code)`; tie-break macro-F1 on normalized debit name.

| exp | JSON parse | schema | code exact | name-norm | macro-F1 debit | Misc pred |
|---|---|---|---|---|---|---|
| exp-qwen15-coa | 100% | 100% | 62.1% | 62.1% | 0.419 | 14.8% |
| **exp-qwen3-coa** | 100% | 100% | **65.6%** | 65.6% | **0.438** | 14.1% |

**Winner: `exp-qwen3-coa`.** Frozen in `outputs/WINNER.txt`. **`data/sft/test.jsonl` has not been scored.**

Misc Expense base rate in this cut is ~13.9%. The 3B val rate (14.1%) did not collapse to Misc; the 100-row GGUF subsample did (37%) — that subsample is not the freeze metric.

### Export (real merge + real quantize)

Winner LoRA merged into HuggingFace Qwen2.5-3B weights (**252 tensors**), then llama.cpp **Q4_K_M** GGUF (~1.80 GB, 4.99 BPW).

Offline llama.cpp smoke, 100 val rows, CPU, model loaded once (`llama-server`):

| | |
|---|---|
| JSON parse / schema | **100%** |
| Code exact-match (this 100-row slice) | 86% |
| Wall time | 159 s (~1.6 s/row after load) |
| Peak RSS | **5.5 GB** (`fits_8gb_budget: true`) |
| Network at infer | none |

---

## 4. Real, but not the whole universe

None of this was fake. These limits are still true:

- Val generate is **256 rows**, not all 6,886 val chats.
- GGUF smoke ran on the **TPU host CPU** (8GB-class RSS). It has not yet been re-run on the submission laptop — copying the 1.8 GB file over `tpu-vm scp` kept dropping. Canonical copy is GCS (below).
- No DPO / GRPO: this export has no Gemini-vs-human preference pairs.
- No 7B / MoE / v6e / vLLM. Training was v5e-8 only; submit runtime is llama.cpp.
- No vision: the PDF is not in this JSONL.

---

## 5. How to get the weights

The TPU VM (`ledgerlite-sft-v5e8`) is **deleted**. Do not SSH looking for `/home/Tinevimbo/...`. Use GCS for the GGUF and the laptop repo for LoRA + metrics.

### What to pull

| artifact | size | where | who already has it |
|---|---|---|---|
| 1.5B LoRA | ~29 MB | `outputs/exp-qwen15-coa/lora_state.npz` + `lora_state.meta.json` | this laptop checkout |
| 3B LoRA (winner) | ~47 MB | `outputs/exp-qwen3-coa/lora_state.npz` + `lora_state.meta.json` | this laptop checkout |
| **Winner GGUF Q4_K_M** | **~1.80 GB** | `gs://tpu-builder1-indaba-ckpts/ledgerlite/exp-qwen3-coa-q4_k_m.gguf` | GCS only (canonical) |
| Val comparison | small | `outputs/val_comparison.json` | this laptop checkout |
| Frozen winner name | tiny | `outputs/WINNER.txt` | this laptop checkout |
| GGUF smoke | small | `outputs/exp-qwen3-coa/smoke_gguf.json` | this laptop checkout |
| Prepared chats | — | `data/sft/` | **gitignored** — do not push (names in free text) |

Incomplete local GGUF temps (`*.gguf_.gstmp`) are **not** usable. Delete them and copy again from GCS.

### Copy the GGUF (the one people actually need)

From the repo root, GCP project **`tpu-builder1`**. On Windows use **`gcloud.cmd`**, not the Python `gcloud.py` wrapper (`storage cp` via `gcloud.py` dies on missing protobuf).

**Windows (PowerShell):**

```powershell
& "C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" storage cp `
  gs://tpu-builder1-indaba-ckpts/ledgerlite/exp-qwen3-coa-q4_k_m.gguf `
  outputs/exp-qwen3-coa/
```

**Linux / macOS / Cloud Shell:**

```bash
gcloud storage cp \
  gs://tpu-builder1-indaba-ckpts/ledgerlite/exp-qwen3-coa-q4_k_m.gguf \
  outputs/exp-qwen3-coa/
```

Expect ~1,929,903,008 bytes. If the copy dies near the end, retry the same command (sliced downloads often fail on a flaky laptop link).

### What you do **not** need from GCS

LoRA npz files are already under `outputs/`. Training JSONL stays local and gitignored. There is no live TPU path to scp.

---

## 6. Current status while we wait for multimodal data

**Now:** text CoA student is trained, 3B wins on val, Q4 GGUF exists, TPU torn down.

**Blocked on:** a dataset that still has statement PDFs / page images / OCR boxes, so a vision student has something to look at. Until then we will not pretend this repo is multimodal.

**Can run next on the same text recipe (different model / size / algorithm):**

- Same CoA SFT on Qwen2.5-0.5B (tighter 8GB) or 7B if v5e HBM allows.
- Second epoch on 3B only if val is still improving.
- Constrained JSON decoding at eval; full-val generate (6,886) then freeze-then-test.
- Laptop llama.cpp RSS / tok/s on the GCS GGUF (the missing local copy).
- When PDFs arrive: vision encoder or VLM student → this text head for posting.
- DPO/GRPO only if preference labels show up.

Recipe files to clone for a new size: `configs/sft_qwen15.yaml` / `configs/sft_qwen3.yaml`, `scripts/train_sft.py`, `scripts/eval_ledger.py`.

---

## 7. Bottom line for Gate 1

We did **two real TPU LoRA SFT runs** on Chris’s Gemini-labeled **text** cut, picked **Qwen2.5-3B** on val (65.6% code exact-match, 100% JSON), and exported a **~1.8 GB Q4 GGUF** that decoded JSON offline at **5.5 GB RSS**. We did **not** mock training. We did **not** score test. We did **not** train vision, because the current files are not images. Next research track is more text ablations and, when the multimodal dump lands, a vision student on top of this posting head.
