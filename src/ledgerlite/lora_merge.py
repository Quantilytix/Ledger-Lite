"""Map Tunix/Qwix LoRA tensors onto HuggingFace Qwen2.5 Linear weights."""

from __future__ import annotations

import re
from typing import Mapping

import numpy as np

_LORA_A_RE = re.compile(r"(.*?)(_lora_a|lora_a)$")
_LAYER_RE = re.compile(r"layers[./](\d+)[./]attn[./](q_proj|k_proj|v_proj|o_proj)")
_MLP_RE = re.compile(r"layers[./](\d+)[./]mlp[./](gate_proj|up_proj|down_proj)")


def lora_array_to_f32(arr: np.ndarray) -> np.ndarray:
    """Decode Tunix np.savez dumps (often bfloat16 stored as void V2) to float32."""
    arr = np.asarray(arr)
    if arr.dtype.itemsize == 2 and (
        arr.dtype == np.dtype("V2")
        or np.issubdtype(arr.dtype, np.void)
        or str(arr.dtype) in {"V2", "|V2", "bfloat16"}
    ):
        bits = arr.view(np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).reshape(arr.shape).copy()
    return np.asarray(arr, dtype=np.float32)


def apply_delta(a: np.ndarray, b: np.ndarray, alpha: float, rank: int) -> np.ndarray:
    scale = float(alpha) / float(rank)
    return (a @ b) * scale


def merge_lora_into_weight(
    weight_hf: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    alpha: float,
    rank: int,
) -> np.ndarray:
    """weight_hf is (out, in). Tunix LoRA a is (in, r), b is (r, out)."""
    delta_in_out = apply_delta(a, b, alpha, rank)
    if delta_in_out.T.shape == weight_hf.shape:
        return weight_hf + delta_in_out.T
    if delta_in_out.shape == weight_hf.shape:
        return weight_hf + delta_in_out
    flat_w = weight_hf.reshape(-1)
    flat_d = delta_in_out.reshape(-1)
    if flat_w.size == flat_d.size:
        return (flat_w + flat_d).reshape(weight_hf.shape)
    raise ValueError(
        f"shape mismatch weight={weight_hf.shape} a={a.shape} b={b.shape} delta={delta_in_out.shape}"
    )


def hf_qwen_name(lora_key: str) -> str | None:
    key = lora_key.replace(".", "/")
    layer = _LAYER_RE.search(key)
    if layer:
        idx, proj = layer.group(1), layer.group(2)
        return f"model.layers.{idx}.self_attn.{proj}.weight"
    mlp = _MLP_RE.search(key)
    if mlp:
        idx, proj = mlp.group(1), mlp.group(2)
        return f"model.layers.{idx}.mlp.{proj}.weight"
    return None


def pair_lora_keys(keys: Mapping[str, np.ndarray]) -> list[tuple[str, str]]:
    pairs = []
    key_set = set(keys)
    for key in keys:
        if "lora_a" not in key:
            continue
        b_key = key.replace("lora_a", "lora_b")
        if b_key in key_set:
            pairs.append((key, b_key))
    return pairs


def _as_2d(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    if b.ndim > 2:
        b = b.reshape(b.shape[0], -1)
    return a, b


def merge_npz_into_state_dict(
    state: dict[str, np.ndarray],
    lora: Mapping[str, np.ndarray],
    alpha: float,
    rank: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    applied: list[str] = []
    for a_key, b_key in pair_lora_keys(lora):
        hf_name = hf_qwen_name(a_key)
        if hf_name is None or hf_name not in state:
            continue
        a, b = _as_2d(lora_array_to_f32(lora[a_key]), lora_array_to_f32(lora[b_key]))
        state[hf_name] = merge_lora_into_weight(
            np.asarray(state[hf_name], dtype=np.float32), a, b, alpha, rank
        )
        applied.append(hf_name)
    return state, applied
