# LedgerLite — Technical Submission Report

**Track:** Corporate / Enterprise (ADTC 2026 Laptop LLM Track)  
**Team:** LedgerLite  
**Model:** LedgerLite-Qwen2.5-3B-Q4_K_M (`model/exp-qwen3-coa-q4_k_m.gguf`)  
**Runtime:** llama.cpp (GGUF Q4_K_M, 100% offline)  

---

## 1. Problem Statement & African Context

Micro, small, and medium enterprises (MSMEs) and accounting practices across Sub-Saharan Africa face significant operational overhead in reconciling bank statement transactions against their specific Chart of Accounts (CoA). Many businesses operate in environments with intermittent internet connectivity, variable electrical grid stability, and strict financial privacy requirements that prevent sending financial data to external cloud LLM APIs.

**LedgerLite** addresses this challenge by providing an autonomous, ultra-efficient, 100% offline edge accounting engine. Given an unformatted bank statement memo line along with the enterprise's custom Chart of Accounts, LedgerLite produces valid double-entry accounting journal postings (debit/credit account codes, names, and categories) directly on standard budget laptops (8 GB RAM).

---

## 2. Model Architecture & Fine-Tuning Strategy

### Base Model Selection
We selected **Qwen2.5-3B-Instruct** as our base model after empirical evaluations against smaller models (such as Qwen2.5-1.5B). The 3B model offers superior instruction-following capabilities and exact code alignment needed for precise financial account mapping while comfortably operating within budget laptop hardware constraints.

### CoA-Conditioned Supervised Fine-Tuning (SFT)
- **Dataset:** 34,165 training examples, 6,886 validation examples across 233 distinct enterprise tenants.
- **Recipe:** Low-Rank Adaptation (LoRA) fine-tuning performed on Google Cloud TPU `v5litepod-8` using JAX/Flax and Google Tunix.
- **Targeting:** All linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with LoRA rank $r=32$, $\alpha=64$.
- **Prompt Formulation:** The model is presented with the bank statement line (description, transaction type, date, amount) alongside a dynamically generated tenant Chart of Accounts block (up to 40 accounts) and instructed to emit structured JSON double-entry journal postings.

---

## 3. Quantization & Edge Deployment

To meet the strict 8 GB RAM laptop profile and offline requirement:
- The trained LoRA weights were merged back into the base HF model (252 tensors).
- The FP16 merged model was quantized to **GGUF Q4_K_M** (~1.80 GB, 4.99 bits per weight).
- Runtime execution is driven by `llama.cpp` using local C/C++ bindings with zero external network requests.

---

## 4. Empirical Benchmarks & Telemetry

| Metric | Target / Constraint | LedgerLite Measured Performance | Status |
|---|---|---|---|
| **JSON Parse Rate** | 100% valid JSON | **100%** | PASS |
| **Schema Compliance** | 100% valid keys | **100%** | PASS |
| **Exact Code Match** | Higher is better | **74.22%** (val set exact match) | PASS |
| **Peak Memory RSS** | < 7.0 GB | **~5.5 GB** | PASS |
| **Offline Execution** | Zero network calls | **100% Local (llama.cpp)** | PASS |
| **Inference Speed** | Laptop vCPU | **~1.6 seconds / transaction** | PASS |

---

## 5. Conclusion & Verification

LedgerLite demonstrates that highly specialized financial reasoning and strict structured output adherence can be achieved on consumer budget laptops without compromising data privacy or requiring cloud connectivity.

All model weights are publicly hosted and automatically downloaded via `download_model.sh` into `model/exp-qwen3-coa-q4_k_m.gguf`.
