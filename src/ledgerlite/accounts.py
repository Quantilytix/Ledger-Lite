"""Chart-of-accounts helpers. Tenant `id`s are never training targets."""

from __future__ import annotations

from typing import Any, Mapping


def account_tuple(account: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(account["code"]), str(account["name"]), str(account["type"]))


def strip_account_ids(account: Mapping[str, Any]) -> dict[str, str]:
    return {
        "code": str(account["code"]),
        "name": str(account["name"]),
        "type": str(account["type"]),
    }


def is_bank_bank(target: Mapping[str, Any]) -> bool:
    debit = target["debit_account"]
    credit = target["credit_account"]
    same_code = str(debit.get("code")) == str(credit.get("code"))
    same_name = str(debit.get("name", "")).strip().casefold() == str(
        credit.get("name", "")
    ).strip().casefold()
    bankish = "bank" in str(debit.get("name", "")).casefold()
    return bool(same_code and same_name and bankish)


def normalize_account_name(name: str) -> str:
    collapsed = " ".join(str(name).replace("’", "'").split()).casefold()
    return collapsed.replace("expenses", "expense")


def target_payload(target_json: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "transaction_type": str(target_json["transaction_type"]),
        "debit_account": strip_account_ids(target_json["debit_account"]),
        "credit_account": strip_account_ids(target_json["credit_account"]),
    }
