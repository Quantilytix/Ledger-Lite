"""CoA-conditioned SFT prompts. Date/amount stay in the user turn, not the target."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


SYSTEM_PROMPT = (
    "You are LedgerLite, an offline bookkeeper. "
    "Given a bank-statement line, posting date, amount, bank direction, and the "
    "business chart of accounts, reply with a single JSON object only. "
    "Do not include tenant-specific account id fields. "
    "Required keys: transaction_type, debit_account, credit_account. "
    "Each account object must have code, name, and type copied from the chart."
)


def format_coa_block(coa: Sequence[Mapping[str, Any]], cap: int = 40) -> str:
    lines = []
    for item in list(coa)[:cap]:
        lines.append(
            f"- {item['code']} | {item['name']} | {item['type']}"
        )
    return "\n".join(lines)


def build_user_prompt(
    *,
    description: str,
    date: str,
    amount: Any,
    transaction_type: str,
    coa: Sequence[Mapping[str, Any]],
    cap: int = 40,
) -> str:
    date_str = str(date)
    if "T" in date_str:
        date_str = date_str.split("T", 1)[0]
    block = format_coa_block(coa, cap=cap)
    return (
        "Classify this bank transaction into a double-entry journal.\n"
        f"Date: {date_str}\n"
        f"Amount: {amount}\n"
        f"Bank direction: {transaction_type}\n"
        f"Description: {description}\n"
        "Chart of accounts:\n"
        f"{block}\n"
        "Return JSON only."
    )
