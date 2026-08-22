from ledgerlite.metrics import (
    code_pair,
    extract_json_object,
    score_prediction,
)


GOLD = {
    "transaction_type": "CREDIT",
    "debit_account": {"code": "6500", "name": "Fuel Expense", "type": "Expense"},
    "credit_account": {"code": "1000", "name": "Bank Account", "type": "Asset"},
}


def test_extract_json_object_from_fenced_completion():
    text = "Sure.\n```json\n" + '{"transaction_type":"CREDIT"}' + "\n```"
    parsed = extract_json_object(text)
    assert parsed["transaction_type"] == "CREDIT"


def test_extract_json_object_returns_none_when_missing():
    assert extract_json_object("no json here") is None


def test_score_prediction_exact_code_match():
    result = score_prediction(
        '{"transaction_type":"CREDIT","debit_account":{"code":"6500","name":"Fuel Expense","type":"Expense"},"credit_account":{"code":"1000","name":"Bank Account","type":"Asset"}}',
        GOLD,
    )
    assert result["json_ok"] is True
    assert result["schema_ok"] is True
    assert result["code_exact"] is True
    assert result["name_norm_match"] is True


def test_score_prediction_name_norm_survives_general_expense_spelling():
    pred = '{"transaction_type":"CREDIT","debit_account":{"code":"7910","name":"General Expenses","type":"Expense"},"credit_account":{"code":"1000","name":"Bank Account","type":"Asset"}}'
    gold = {
        "transaction_type": "CREDIT",
        "debit_account": {
            "code": "7910",
            "name": "General expense",
            "type": "Expense",
        },
        "credit_account": {
            "code": "1000",
            "name": "Bank Account",
            "type": "Asset",
        },
    }
    result = score_prediction(pred, gold)
    assert result["code_exact"] is True
    assert result["name_norm_match"] is True


def test_code_pair_from_payload():
    assert code_pair(GOLD) == ("6500", "1000")


def test_score_prediction_flags_invalid_json():
    result = score_prediction("not json", GOLD)
    assert result["json_ok"] is False
    assert result["schema_ok"] is False
    assert result["code_exact"] is False
