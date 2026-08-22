"""Score val completions; freeze a winner. Do not touch test until then."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ledgerlite.metrics import aggregate_scores, score_prediction  # noqa: E402
from ledgerlite.prepare import load_jsonl  # noqa: E402


def score_file(pred_path: Path, gold_path: Path | None) -> dict:
    preds = load_jsonl(pred_path)
    gold_by_idx = None
    if gold_path and gold_path.exists():
        gold_rows = load_jsonl(gold_path)
        gold_by_idx = [row.get("gold") for row in gold_rows]
    scored = []
    for i, row in enumerate(preds):
        gold = row.get("gold")
        if gold is None and gold_by_idx is not None and i < len(gold_by_idx):
            gold = gold_by_idx[i]
        if gold is None:
            continue
        item = score_prediction(row.get("completion", ""), gold)
        scored.append(item)
    summary = aggregate_scores(scored)
    summary["pred_path"] = str(pred_path)
    return summary


def pick_winner(summaries: list[dict]) -> dict:
    eligible = [s for s in summaries if s["json_parse_rate"] >= 0.95]
    pool = eligible or summaries
    return max(
        pool,
        key=lambda s: (s["code_exact_match"], s["macro_f1_debit_name"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", action="append", dest="preds", default=[])
    parser.add_argument("--gold", type=Path, default=ROOT / "data" / "sft" / "val.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "val_comparison.json")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()

    if not args.allow_test and "test" in str(args.gold).replace("\\", "/").split("/"):
        raise SystemExit("Refusing to score test until --allow-test (winner must be frozen).")

    if not args.preds:
        default_dir = ROOT / "outputs"
        args.preds = sorted(str(p) for p in default_dir.glob("*/val_preds.jsonl"))
    if not args.preds:
        raise SystemExit("No val_preds.jsonl files found. Pass --pred.")

    summaries = []
    for pred in args.preds:
        summary = score_file(Path(pred), args.gold)
        summary["exp"] = Path(pred).parent.name
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    winner = pick_winner(summaries)
    payload = {"summaries": summaries, "winner": winner["exp"], "winner_metrics": winner}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    freeze = ROOT / "outputs" / "WINNER.txt"
    freeze.write_text(winner["exp"] + "\n", encoding="utf-8")
    print("Winner:", winner["exp"])
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
