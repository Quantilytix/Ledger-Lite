from ledgerlite.prompts import build_user_prompt, format_coa_block, SYSTEM_PROMPT


def test_format_coa_block_caps_and_keeps_tenant_names():
    coa = [
        {"code": "1000", "name": "Bank Account", "type": "Asset", "count": 90},
        {"code": "6500", "name": "Fuel Expense", "type": "Expense", "count": 40},
        {"code": "7900", "name": "Miscellaneous Expense", "type": "Expense", "count": 10},
    ]
    block = format_coa_block(coa, cap=2)
    assert "Bank Account" in block
    assert "Fuel Expense" in block
    assert "Miscellaneous Expense" not in block
    assert block.count("\n") == 1


def test_build_user_prompt_includes_structured_fields_not_just_memo():
    prompt = build_user_prompt(
        description="Fuel",
        date="2026-02-15T22:00:00.000Z",
        amount=600,
        transaction_type="CREDIT",
        coa=[
            {"code": "1000", "name": "Bank Account", "type": "Asset", "count": 2},
            {"code": "6500", "name": "Fuel Expense", "type": "Expense", "count": 1},
        ],
    )
    assert "Fuel" in prompt
    assert "600" in prompt
    assert "2026-02-15" in prompt
    assert "CREDIT" in prompt
    assert "6500" in prompt
    assert "Bank Account" in prompt


def test_system_prompt_asks_for_json_without_ids():
    assert "JSON" in SYSTEM_PROMPT
    assert "id" in SYSTEM_PROMPT.lower()
