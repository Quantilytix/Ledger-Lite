"""Val metrics: JSON parse, schema, code exact-match, normalized names."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from ledgerlite.accounts import normalize_account_name, strip_account_ids


_REQUIRED = ("transaction_type", "debit_account", "credit_account")
_ACCOUNT_KEYS = ("code", "name", "type")
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    fenced = _FENCE_RE.search(text)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        match = _OBJECT_RE.search(text)
        candidate = match.group(0) if match else None
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _account_ok(value: Any) -> bool:
    return isinstance(value, Mapping) and all(k in value for k in _ACCOUNT_KEYS)


def schema_ok(payload: Mapping[str, Any] | None) -> bool:
    if not payload:
        return False
    if any(k not in payload for k in _REQUIRED):
        return False
    return _account_ok(payload["debit_account"]) and _account_ok(
        payload["credit_account"]
    )


def code_pair(payload: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(payload["debit_account"]["code"]),
        str(payload["credit_account"]["code"]),
    )


def _name_pair(payload: Mapping[str, Any]) -> tuple[str, str]:
    return (
        normalize_account_name(str(payload["debit_account"]["name"])),
        normalize_account_name(str(payload["credit_account"]["name"])),
    )


def score_prediction(completion: str, gold: Mapping[str, Any]) -> dict[str, Any]:
    parsed = extract_json_object(completion)
    ok_json = parsed is not None
    ok_schema = schema_ok(parsed)
    gold_codes = code_pair(gold)
    gold_names = _name_pair(gold)
    pred_codes = code_pair(parsed) if ok_schema else None
    pred_names = _name_pair(parsed) if ok_schema else None
    debit_name = (
        normalize_account_name(str(parsed["debit_account"]["name"]))
        if ok_schema
        else None
    )
    return {
        "json_ok": ok_json,
        "schema_ok": ok_schema,
        "code_exact": bool(pred_codes == gold_codes),
        "name_norm_match": bool(pred_names == gold_names),
        "debit_name_norm": debit_name,
        "gold_debit_name_norm": gold_names[0],
        "parsed": parsed,
        "gold": {
            "transaction_type": gold["transaction_type"],
            "debit_account": strip_account_ids(gold["debit_account"]),
            "credit_account": strip_account_ids(gold["credit_account"]),
        },
    }


def aggregate_scores(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    json_ok = sum(1 for r in rows if r["json_ok"])
    schema_ok_n = sum(1 for r in rows if r["schema_ok"])
    code_exact = sum(1 for r in rows if r["code_exact"])
    name_norm = sum(1 for r in rows if r["name_norm_match"])
    labels = sorted({r["gold_debit_name_norm"] for r in rows})
    f1s: list[float] = []
    for label in labels:
        tp = sum(
            1
            for r in rows
            if r["debit_name_norm"] == label and r["gold_debit_name_norm"] == label
        )
        fp = sum(
            1
            for r in rows
            if r["debit_name_norm"] == label and r["gold_debit_name_norm"] != label
        )
        fn = sum(
            1
            for r in rows
            if r["debit_name_norm"] != label and r["gold_debit_name_norm"] == label
        )
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    misc_pred = sum(
        1
        for r in rows
        if r["debit_name_norm"] and "miscellaneous expense" in r["debit_name_norm"]
    )
    return {
        "n": len(rows),
        "json_parse_rate": json_ok / n,
        "schema_rate": schema_ok_n / n,
        "code_exact_match": code_exact / n,
        "name_norm_match": name_norm / n,
        "macro_f1_debit_name": sum(f1s) / len(f1s) if f1s else 0.0,
        "misc_expense_pred_rate": misc_pred / n,
    }
