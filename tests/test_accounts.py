from ledgerlite.accounts import (
    account_tuple,
    is_bank_bank,
    normalize_account_name,
    strip_account_ids,
    target_payload,
)


FUEL = {
    "id": 10551,
    "code": "6500",
    "name": "Fuel Expense",
    "type": "Expense",
}
BANK = {
    "id": 10517,
    "code": "1000",
    "name": "Bank Account",
    "type": "Asset",
}


def test_account_tuple_drops_tenant_id():
    assert account_tuple(FUEL) == ("6500", "Fuel Expense", "Expense")


def test_is_bank_bank_detects_identity_posting():
    target = {
        "debit_account": BANK,
        "credit_account": {
            "id": 999,
            "code": "1000",
            "name": "Bank Account",
            "type": "Asset",
        },
    }
    assert is_bank_bank(target) is True


def test_is_bank_bank_false_for_real_expense():
    assert is_bank_bank({"debit_account": FUEL, "credit_account": BANK}) is False


def test_normalize_account_name_collapses_general_expense_variants():
    assert normalize_account_name("General Expenses") == normalize_account_name(
        "General expense"
    )
    assert normalize_account_name("General Expense") == normalize_account_name(
        "General Expenses"
    )


def test_target_payload_omits_ids():
    payload = target_payload(
        {
            "date": "2026-02-15T22:00:00.000Z",
            "amount": 600,
            "transaction_type": "CREDIT",
            "debit_account": FUEL,
            "credit_account": BANK,
        }
    )
    assert payload == {
        "transaction_type": "CREDIT",
        "debit_account": {
            "code": "6500",
            "name": "Fuel Expense",
            "type": "Expense",
        },
        "credit_account": {
            "code": "1000",
            "name": "Bank Account",
            "type": "Asset",
        },
    }
    assert "id" not in payload["debit_account"]
    assert "date" not in payload


def test_strip_account_ids():
    stripped = strip_account_ids(FUEL)
    assert stripped == {"code": "6500", "name": "Fuel Expense", "type": "Expense"}
