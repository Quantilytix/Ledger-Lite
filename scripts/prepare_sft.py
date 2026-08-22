"""CLI: Chris JSONL → CoA-conditioned chat JSONL."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ledgerlite.prepare import load_jsonl, prepare_records, write_jsonl
RAW_CANDIDATES = {
    "train": ["data/raw/train.jsonl", "train.jsonl", "train (2).jsonl"],
    "val": ["data/raw/val.jsonl", "val.jsonl"],
    "test": ["data/raw/test.jsonl", "test.jsonl", "test (2).jsonl"],
}


def _find(split: str) -> Path:
    for rel in RAW_CANDIDATES[split]:
        path = ROOT / rel
        if path.exists():
            return path
    raise FileNotFoundError(f"No raw file for split {split}: {RAW_CANDIDATES[split]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coa-cap", type=int, default=40)
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "raw"
    sft_dir = ROOT / "data" / "sft"
    diag_dir = ROOT / "data" / "diagnostics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sft_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    meta_all: dict = {"coa_cap": args.coa_cap, "splits": {}}
    for split, drop_bank in (("train", True), ("val", True), ("test", True)):
        src = _find(split)
        dest = raw_dir / f"{split}.jsonl"
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        rows = load_jsonl(dest)
        prepared, meta = prepare_records(
            rows, drop_bank_bank=drop_bank, coa_cap=args.coa_cap
        )
        write_jsonl(sft_dir / f"{split}.jsonl", prepared)
        kept, _ = prepare_records(rows, drop_bank_bank=False, coa_cap=args.coa_cap)
        prepared_keys = {(r["tenant_id"], r["messages"][2]["content"]) for r in prepared}
        bank_rows = [
            row
            for row in kept
            if (row["tenant_id"], row["messages"][2]["content"]) not in prepared_keys
        ]
        write_jsonl(diag_dir / f"{split}_bank_bank.jsonl", bank_rows)
        meta_all["splits"][split] = meta
        print(f"{split}: {meta}")

    (sft_dir / "meta.json").write_text(
        json.dumps(meta_all, indent=2), encoding="utf-8"
    )
    print(f"Wrote {sft_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
