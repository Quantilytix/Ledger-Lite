"""Turn Chris's {input_text, target_json} rows into CoA-conditioned SFT chats."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ledgerlite.accounts import (
    account_tuple,
    is_bank_bank,
    strip_account_ids,
    target_payload,
)
from ledgerlite.prompts import SYSTEM_PROMPT, build_user_prompt


def fingerprint(row: Mapping[str, Any]) -> tuple[Any, ...]:
    target = row["target_json"]
    debit = account_tuple(target["debit_account"])
    credit = account_tuple(target["credit_account"])
    return (
        row["tenant_id"],
        row["input_text"],
        target.get("amount"),
        target.get("transaction_type"),
        debit,
        credit,
    )


def build_tenant_coa(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        target = row["target_json"]
        for key in ("debit_account", "credit_account"):
            counts[account_tuple(target[key])] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [
        {"code": code, "name": name, "type": typ, "count": count}
        for (code, name, typ), count in ranked
    ]


def prepare_records(
    rows: list[Mapping[str, Any]],
    *,
    drop_bank_bank: bool = True,
    coa_cap: int = 40,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_in = len(rows)
    seen: set[tuple[Any, ...]] = set()
    unique: list[Mapping[str, Any]] = []
    n_dups = 0
    for row in rows:
        key = fingerprint(row)
        if key in seen:
            n_dups += 1
            continue
        seen.add(key)
        unique.append(row)

    by_tenant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in unique:
        by_tenant[str(row["tenant_id"])].append(row)
    coa_by_tenant = {tid: build_tenant_coa(items) for tid, items in by_tenant.items()}

    prepared: list[dict[str, Any]] = []
    n_bank = 0
    for row in unique:
        if drop_bank_bank and is_bank_bank(row["target_json"]):
            n_bank += 1
            continue
        target = row["target_json"]
        gold = target_payload(target)
        coa = coa_by_tenant[str(row["tenant_id"])]
        user = build_user_prompt(
            description=str(row["input_text"]),
            date=str(target.get("date", "")),
            amount=target.get("amount"),
            transaction_type=str(target.get("transaction_type", "")),
            coa=coa,
            cap=coa_cap,
        )
        assistant = json.dumps(gold, ensure_ascii=False, separators=(",", ":"))
        prepared.append(
            {
                "tenant_id": row["tenant_id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "gold": gold,
                "coa_size": min(len(coa), coa_cap),
            }
        )

    coa_sizes = [item["coa_size"] for item in prepared]
    meta = {
        "n_in": n_in,
        "n_dups_dropped": n_dups,
        "n_bank_bank_dropped": n_bank,
        "n_out": len(prepared),
        "n_tenants": len(coa_by_tenant),
        "coa_cap": coa_cap,
        "coa_size_mean": (sum(coa_sizes) / len(coa_sizes)) if coa_sizes else 0,
        "drop_bank_bank": drop_bank_bank,
    }
    return prepared, meta


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
