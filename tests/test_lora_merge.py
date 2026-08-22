from ledgerlite.lora_merge import (
    apply_delta,
    hf_qwen_name,
    merge_lora_into_weight,
    merge_npz_into_state_dict,
    pair_lora_keys,
)


def test_apply_delta_rank1():
    import numpy as np

    a = np.ones((4, 1), dtype=np.float32)
    b = np.ones((1, 6), dtype=np.float32)
    delta = apply_delta(a, b, alpha=16, rank=16)
    assert delta.shape == (4, 6)
    assert np.allclose(delta, 1.0)


def test_merge_lora_into_weight_adds_scaled_delta():
    import numpy as np

    weight = np.zeros((6, 4), dtype=np.float32)  # HF (out, in)
    a = np.eye(4, 2, dtype=np.float32)  # (in, rank)
    b = np.ones((2, 6), dtype=np.float32)  # (rank, out)
    merged = merge_lora_into_weight(weight, a, b, alpha=2, rank=2)
    assert merged.shape == (6, 4)


def test_hf_qwen_name_maps_q_proj():
    assert (
        hf_qwen_name("layers/0/attn/q_proj/w_lora_a")
        == "model.layers.0.self_attn.q_proj.weight"
    )


def test_lora_array_to_f32_decodes_bf16_void():
    import numpy as np

    from ledgerlite.lora_merge import lora_array_to_f32

    values = np.array([1.0, -2.5, 0.0], dtype=np.float32)
    # Truncate f32 to bf16 bits, store as void V2 like Tunix np.savez.
    bits = values.view(np.uint32) >> 16
    void = bits.astype(np.uint16).view(np.dtype("V2"))
    decoded = lora_array_to_f32(void)
    assert decoded.dtype == np.float32
    assert np.allclose(decoded, values, atol=0.01)


def test_merge_npz_updates_matching_hf_key():
    import numpy as np

    state = {"model.layers.0.self_attn.q_proj.weight": np.zeros((6, 4), np.float32)}
    lora = {
        "layers.0.attn.q_proj.w_lora_a": np.ones((4, 2), np.float32),
        "layers.0.attn.q_proj.w_lora_b": np.ones((2, 6), np.float32),
    }
    new_state, applied = merge_npz_into_state_dict(state, lora, alpha=2, rank=2)
    assert applied == ["model.layers.0.self_attn.q_proj.weight"]
    assert new_state["model.layers.0.self_attn.q_proj.weight"].sum() != 0
    assert pair_lora_keys(lora) == [
        ("layers.0.attn.q_proj.w_lora_a", "layers.0.attn.q_proj.w_lora_b")
    ]
