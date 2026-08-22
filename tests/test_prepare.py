from ledgerlite.prepare import (
    build_tenant_coa,
    fingerprint,
    prepare_records,
)


def _row(tenant, text, debit, credit, amount=100, tx="CREDIT"):
    return {
        "tenant_id": tenant,
        "input_text": text,
        "target_json": {
            "date": "2026-02-15T22:00:00.000Z",
            "amount": amount,
            "transaction_type": tx,
            "debit_account": debit,
            "credit_account": credit,
        },
    }


BANK = {"id": 1, "code": "1000", "name": "Bank Account", "type": "Asset"}
FUEL = {"id": 2, "code": "6500", "name": "Fuel Expense", "type": "Expense"}
MISC = {"id": 3, "code": "7900", "name": "Miscellaneous Expense", "type": "Expense"}


def test_fingerprint_collapses_exact_duplicates():
    a = _row("t1", "Fuel", FUEL, BANK)
    b = _row("t1", "Fuel", FUEL, BANK)
    assert fingerprint(a) == fingerprint(b)


def test_build_tenant_coa_frequency_sorted():
    rows = [
        _row("t1", "Fuel", FUEL, BANK),
        _row("t1", "Fuel2", FUEL, BANK),
        _row("t1", "Misc", MISC, BANK),
    ]
    coa = build_tenant_coa(rows)
    assert coa[0]["code"] == "1000"
    assert {item["code"] for item in coa} == {"1000", "6500", "7900"}


def test_prepare_records_dedups_drops_bank_bank_and_emits_chat():
    bank_bank = _row("t1", "Transfer", BANK, BANK)
    fuel = _row("t1", "Fuel", FUEL, BANK)
    fuel_dup = _row("t1", "Fuel", FUEL, BANK)
    other = _row("t2", "Fuel", FUEL, BANK)

    prepared, meta = prepare_records(
        [bank_bank, fuel, fuel_dup, other],
        drop_bank_bank=True,
        coa_cap=40,
    )
    assert meta["n_in"] == 4
    assert meta["n_dups_dropped"] == 1
    assert meta["n_bank_bank_dropped"] == 1
    assert meta["n_out"] == 2
    assert len(prepared) == 2
    sample = prepared[0]
    assert sample["messages"][0]["role"] == "system"
    assert sample["messages"][1]["role"] == "user"
    assert sample["messages"][2]["role"] == "assistant"
    assert "id" not in sample["messages"][2]["content"]
    assert "Fuel Expense" in sample["messages"][1]["content"]
    assert sample["gold"]["debit_account"]["code"] == "6500"


def test_prepare_records_keeps_bank_bank_when_not_dropped():
    rows = [_row("t1", "Transfer", BANK, BANK)]
    prepared, meta = prepare_records(rows, drop_bank_bank=False)
    assert meta["n_out"] == 1
    assert len(prepared) == 1
