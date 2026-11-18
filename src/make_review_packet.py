"""Assemble an unannotated small audit packet for the primary agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrate import read_jsonl
from runtime_tools import rows_for_lane


ROOT = Path(__file__).resolve().parents[1]


def build(max_cases: int = 8) -> list[dict]:
    catalog = read_jsonl(ROOT / "data" / "rate_catalog.jsonl")
    chosen: list[dict] = []
    used: set[str] = set()
    criteria = [
        ("test", lambda case, out: bool(case["source_visible"])
         and out.get("candidate", {}).get("decision") == "quote", "visible rate card"),
        ("test", lambda case, out: not case["source_visible"]
         and out.get("candidate", {}).get("decision") == "need_data", "rates absent from first-round context"),
        ("test", lambda case, out: not case["source_visible"]
         and out.get("candidate", {}).get("carrier_id") is not None
         and out.get("candidate", {}).get("amount_cents") is not None, "complete candidate without visible rate card"),
        ("test", lambda case, out: out.get("finish_reason") == "length", "provider finish_reason=length"),
        ("shift", lambda case, out: out.get("candidate", {}).get("decision") == "hold",
         "shift-split hold candidate"),
        ("shift", lambda case, out: out.get("candidate", {}).get("decision") == "quote",
         "shift-split quote candidate"),
        ("dev", lambda case, out: bool(case["source_visible"]), "development visible-card sample"),
        ("dev", lambda case, out: not case["source_visible"], "development query-required sample"),
    ]
    for split, predicate, stratum in criteria:
        request_path = ROOT / "data" / f"{split}_requests.jsonl"
        output_path = ROOT / "data" / "model_outputs" / f"{split}_first_round_api.jsonl"
        if not output_path.exists():
            raise FileNotFoundError(f"API output cache is not ready: {output_path}")
        cases = {row["request_id"]: row for row in read_jsonl(request_path)}
        outputs = {row["request_id"]: row for row in read_jsonl(output_path)}
        if set(outputs) != set(cases):
            raise ValueError(f"API output cache for {split} is incomplete or has extra IDs")
        candidate = next((outputs[rid] for rid, case in cases.items()
                          if rid not in used and predicate(case, outputs[rid])), None)
        if candidate is None:
            continue
        rid = candidate["request_id"]
        case = cases[rid]
        gold = {row["request_id"]: row for row in read_jsonl(ROOT / "data" / "oracle" / f"{split}_gold.jsonl")}[rid]
        chosen.append({
            "request_id": rid,
            "split": split,
            "selection_stratum": stratum,
            "request_visible_to_model": {
                "user_text": case["user_text"],
                "lane_id": case["lane_id"],
                "weight_kg": case["weight_kg"],
                "latest_days": case["latest_days"],
                "service": case["service"],
                "carrier_ids": case["carrier_ids"],
                "visible_rate_card": case["visible_rate_card"],
            },
            "retrieved_catalog_for_manual_inspection": rows_for_lane(catalog, case["lane_id"]),
            "unannotated_model_output": {
                "requested_model": candidate.get("requested_model"),
                "server_model": candidate.get("server_model"),
                "finish_reason": candidate.get("finish_reason"),
                "truncated": candidate.get("truncated"),
                "raw_content": candidate.get("raw_content"),
                "parsed_candidate": candidate.get("candidate"),
                "usage": candidate.get("usage"),
                "api_elapsed_ms": candidate.get("api_elapsed_ms"),
            },
            "independent_program_target_for_review": gold,
            "primary_manual_annotation": None,
        })
        used.add(rid)
        if len(chosen) >= max_cases:
            break
    out_path = ROOT / "results" / "manual_review_packet.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                        for row in chosen), encoding="utf-8")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cases", type=int, default=8)
    args = parser.parse_args()
    result = build(args.max_cases)
    print(json.dumps({"review_cases": len(result),
                      "request_ids": [row["request_id"] for row in result],
                      "path": "results/manual_review_packet.jsonl"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

