"""Tunix LoRA SFT for LedgerLite Qwen2.5 students. Run on the TPU VM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ledgerlite.prepare import load_jsonl  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def tokenize_example(example: dict, tokenizer, max_seq_len: int) -> dict:
    messages = example["messages"]
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        prompt_ids = prompt_ids[: min(len(prompt_ids), max_seq_len)]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_n = max_seq_len - len(full_ids)
    input_tokens = full_ids + [pad_id] * pad_n
    prompt_len = min(len(prompt_ids), len(full_ids))
    input_mask = (
        [False] * prompt_len
        + [True] * (len(full_ids) - prompt_len)
        + [False] * pad_n
    )
    return {
        "input_tokens": np.asarray(input_tokens, dtype=np.int32),
        "input_mask": np.asarray(input_mask, dtype=bool),
        "prompt_text": prompt_text,
        "gold": example.get("gold"),
    }


def cycle_batches(rows: list[dict], batch_size: int):
    if not rows:
        raise ValueError("empty dataset")
    i = 0
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(rows[i % len(rows)])
            i += 1
        yield {
            "input_tokens": np.stack([x["input_tokens"] for x in batch]),
            "input_mask": np.stack([x["input_mask"] for x in batch]),
        }


class FiniteBatches:
    def __init__(self, rows: list[dict], batch_size: int, max_batches: int | None):
        self.rows = rows
        self.batch_size = batch_size
        self.max_batches = max_batches

    def __iter__(self):
        n = 0
        for i in range(0, len(self.rows), self.batch_size):
            batch = self.rows[i : i + self.batch_size]
            if len(batch) < self.batch_size:
                break
            yield {
                "input_tokens": np.stack([x["input_tokens"] for x in batch]),
                "input_mask": np.stack([x["input_mask"] for x in batch]),
            }
            n += 1
            if self.max_batches is not None and n >= self.max_batches:
                break


def build_positions(mask):
    import jax.numpy as jnp

    return jnp.clip(jnp.cumsum(mask, axis=-1) - 1, 0).astype("int32")


def build_causal_mask(mask):
    import jax.numpy as jnp

    n = mask.shape[-1]
    return jnp.tril(jnp.ones((n, n), dtype=bool))[None] & mask[:, None, :]


def greedy_generate(model, tokenizer, prompt: str, max_new: int, max_seq: int) -> str:
    import jax.numpy as jnp
    from flax import nnx

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = ids[: max_seq - 1]
    generated = list(ids)
    need = min(max_seq, len(ids) + max_new)
    pad_to = max_seq
    for bucket in (512, 768, 1024, 1280, 1536, 1792, 2048):
        if need <= bucket <= max_seq:
            pad_to = bucket
            break

    @nnx.jit
    def step(m, tokens, positions, attn):
        logits, _ = m(tokens, positions, None, attn)
        return logits

    for _ in range(max_new):
        cur_len = len(generated)
        if cur_len >= pad_to:
            break
        padded = generated + [pad_id] * (pad_to - cur_len)
        arr = jnp.asarray([padded], dtype=jnp.int32)
        mask = arr != pad_id
        positions = build_positions(mask)
        attn = build_causal_mask(mask)
        logits = step(model, arr, positions, attn)
        nxt = int(jnp.argmax(logits[0, cur_len - 1], axis=-1))
        generated.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated[len(ids) :], skip_special_tokens=True)


def _flatten(tree, prefix: str = "") -> dict[str, np.ndarray]:
    import jax

    out: dict[str, np.ndarray] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            out.update(_flatten(value, f"{prefix}/{key}" if prefix else str(key)))
        return out
    if hasattr(tree, "items"):
        try:
            return _flatten(dict(tree.items()), prefix)
        except Exception:
            pass
    try:
        arr = np.asarray(jax.device_get(tree))
        if arr.dtype != object:
            out[prefix] = arr
    except Exception:
        pass
    return out


def load_lora_npz(model, path: Path) -> int:
    """Restore dumped LoRA tensors onto a Qwix-wrapped model. Returns matches."""
    import jax.numpy as jnp
    from flax import nnx

    from ledgerlite.lora_merge import lora_array_to_f32  # noqa: PLC0415

    data = np.load(path)
    loaded = {key: jnp.asarray(lora_array_to_f32(data[key])) for key in data.files}
    try:
        params = nnx.state(model, nnx.LoRAParam)
    except Exception:
        params = nnx.state(model)

    def _walk(node, prefix: str = "") -> int:
        n = 0
        items = None
        if isinstance(node, dict):
            items = node.items()
        elif hasattr(node, "items"):
            try:
                items = list(node.items())
            except Exception:
                items = None
        if items is not None:
            for key, value in items:
                n += _walk(value, f"{prefix}/{key}" if prefix else str(key))
            return n
        target = getattr(node, "value", node)
        safe = prefix.replace("/", ".")
        if safe not in loaded:
            return 0
        arr = loaded[safe]
        try:
            same = tuple(np.asarray(target).shape) == tuple(np.asarray(arr).shape)
        except Exception:
            same = False
        if not same:
            return 0
        if hasattr(node, "value"):
            node.value = arr
        return 1

    n = _walk(params)
    try:
        nnx.update(model, params)
    except Exception as exc:  # noqa: BLE001
        print("nnx.update after LoRA load:", exc)
    return n


def save_lora_npz(model, path: Path, rank: int, alpha: float) -> None:
    from flax import nnx

    try:
        params = nnx.state(model, nnx.LoRAParam)
    except Exception:
        params = nnx.state(model)
    if hasattr(nnx, "to_pure_dict"):
        tree = nnx.to_pure_dict(params)
    else:
        tree = params
    flat = _flatten(tree)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = {k.replace("/", "."): v for k, v in flat.items()}
    np.savez_compressed(path, **safe)
    path.with_suffix(".meta.json").write_text(
        json.dumps({"rank": rank, "alpha": alpha, "keys": list(safe)}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--generate-from-lora",
        default=None,
        help="Skip training; load this lora_state.npz and write val_preds.jsonl",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Override eval_generate_n (val completions)",
    )
    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    import jax
    import jax.numpy as jnp
    import optax
    import qwix
    import transformers
    from flax import nnx
    from huggingface_hub import snapshot_download
    from tunix.models.qwen2 import model as qwen2_lib
    from tunix.models.qwen2 import params as qwen2_params
    from tunix.sft import metrics_logger, peft_trainer
    from tunix.sft import utils as sft_utils

    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generate_only = bool(args.generate_from_lora)
    print("Tokenizing datasets…")
    val_raw = load_jsonl(ROOT / cfg["val_path"])
    if generate_only:
        train_tok = []
        max_steps = 0
        print(f"generate-only val={len(val_raw)}")
    else:
        train_raw = load_jsonl(ROOT / cfg["train_path"])
        train_tok = [
            tokenize_example(row, tokenizer, cfg["max_seq_len"]) for row in train_raw
        ]
        val_tok = [
            tokenize_example(row, tokenizer, cfg["max_seq_len"]) for row in val_raw
        ]
        steps_per_epoch = max(1, len(train_tok) // cfg["batch_size"])
        max_steps = steps_per_epoch * int(cfg["num_epochs"])
        print(
            f"train={len(train_tok)} val={len(val_tok)} "
            f"steps/epoch={steps_per_epoch} max_steps={max_steps}"
        )

    n = jax.device_count()
    print("devices", n, jax.devices())
    mesh = jax.make_mesh(
        tuple(cfg["mesh_shape"]),
        tuple(cfg["mesh_axis_names"]),
        axis_types=(jax.sharding.AxisType.Auto,) * len(cfg["mesh_shape"]),
    )

    ckpt_dir = snapshot_download(cfg["model_id"], cache_dir=str(ROOT / "hf_cache"))
    config_fn = getattr(qwen2_lib.ModelConfig, cfg["model_config"])
    dummy_bs = int(cfg["batch_size"])
    dummy = {
        "input_tokens": jnp.ones((dummy_bs, 1), dtype=jnp.int32),
        "positions": jnp.ones((dummy_bs, 1), dtype=jnp.int32),
        "cache": None,
        "attention_mask": jnp.ones((dummy_bs, 1, 1), dtype=bool),
    }
    with mesh:
        base_model = qwen2_params.create_model_from_safe_tensors(
            ckpt_dir, config_fn(), mesh
        )
        lora_provider = qwix.LoraProvider(
            module_path=cfg["lora_module_path"],
            rank=int(cfg["lora_rank"]),
            alpha=float(cfg["lora_alpha"]),
        )
        lora_model = qwix.apply_lora_to_model(
            base_model, lora_provider, rngs=nnx.Rngs(cfg["seed"]), **dummy
        )
        state = nnx.state(lora_model)
        sharded = jax.lax.with_sharding_constraint(
            state, nnx.get_partition_spec(state)
        )
        nnx.update(lora_model, sharded)

    if generate_only:
        n_loaded = load_lora_npz(lora_model, Path(args.generate_from_lora))
        print(f"Loaded {n_loaded} LoRA tensors from {args.generate_from_lora}")
        if n_loaded < 50:
            raise SystemExit(f"too few LoRA tensors loaded: {n_loaded}")
    else:
        def gen_model_input(x):
            mask = x["input_tokens"] != tokenizer.pad_token_id
            if hasattr(sft_utils, "build_positions_from_mask"):
                positions = sft_utils.build_positions_from_mask(mask)
                attention_mask = sft_utils.make_causal_attn_mask(mask)
            else:
                positions = build_positions(mask)
                attention_mask = build_causal_mask(mask)
            return {
                "input_tokens": x["input_tokens"],
                "positions": positions,
                "attention_mask": attention_mask,
                "input_mask": x["input_mask"],
            }

        trainer = peft_trainer.PeftTrainer(
            lora_model,
            optax.adamw(float(cfg["learning_rate"])),
            peft_trainer.TrainingConfig(
                max_steps=max_steps,
                eval_every_n_steps=int(cfg["eval_every_n_steps"]),
                checkpoint_root_directory=str(out_dir / "checkpoints"),
                metrics_logging_options=metrics_logger.MetricsLoggerOptions(
                    log_dir=str(out_dir / "logs")
                ),
            ),
        ).with_gen_model_input_fn(gen_model_input)

        train_ds = cycle_batches(train_tok, cfg["batch_size"])
        eval_ds = FiniteBatches(
            val_tok, cfg["batch_size"], cfg.get("max_eval_batches")
        )
        print("Training…")
        with mesh:
            trainer.train(train_ds, eval_ds)

        save_lora_npz(
            lora_model,
            out_dir / "lora_state.npz",
            int(cfg["lora_rank"]),
            float(cfg["lora_alpha"]),
        )
        print("Saved", out_dir / "lora_state.npz")

    n_gen = min(int(args.n or cfg.get("eval_generate_n", 256)), len(val_raw))
    preds = []
    print(f"Generating {n_gen} val completions…")
    for i, row in enumerate(val_raw[:n_gen]):
        prompt = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        try:
            text = greedy_generate(
                lora_model,
                tokenizer,
                prompt,
                int(cfg["max_new_tokens"]),
                cfg["max_seq_len"],
            )
        except Exception as exc:  # noqa: BLE001
            text = f"GENERATION_ERROR: {exc}"
        preds.append(
            {
                "tenant_id": row.get("tenant_id"),
                "gold": row["gold"],
                "completion": text,
            }
        )
        if (i + 1) % 32 == 0:
            print(f"  generated {i + 1}/{n_gen}")
    pred_path = out_dir / "val_preds.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for item in preds:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Wrote", pred_path)


if __name__ == "__main__":
    main()
