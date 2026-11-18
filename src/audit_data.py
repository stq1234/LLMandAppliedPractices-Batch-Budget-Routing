"""Independently cross-check generated labels and data split invariants."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def independent_quote(req: dict, lane_rows: list[dict]) -> dict:
    """Separate implementation from the label builder; no shared helper calls."""
    accepted = []
    for row in lane_rows:
        if row["capacity_kg"] < req["weight_kg"]:
            continue
        if row["eta_days"] > req["latest_days"]:
            continue
        if req["service"] not in row["services"]:
            continue
        total = row["base_cents"] + (row["per_kg_cents"] * req["weight_kg"])
        if req["service"] == "express":
            total += row["express_surcharge_cents"]
        accepted.append((total, row["eta_days"], row["carrier_id"]))
    if len(accepted) == 0:
        return {"request_id": req["request_id"], "decision": "hold", "carrier_id": None,
                "amount_cents": None, "eta_days": None, "feasible_count": 0}
    chosen_price, chosen_eta, chosen_carrier = sorted(accepted)[0]
    return {"request_id": req["request_id"], "decision": "quote",
            "carrier_id": chosen_carrier, "amount_cents": chosen_price,
            "eta_days": chosen_eta, "feasible_count": len(accepted)}


def audit() -> dict:
    lane_to_rows = {}
    for row in read_rows(ROOT / "data" / "rate_catalog.jsonl"):
        lane_to_rows.setdefault(row["lane_id"], []).append(row)
    ids, lanes = set(), set()
    report = {"oracle_crosscheck": {}, "split_counts": {}, "passed": True}
    for split in ("dev", "test", "shift"):
        requests = read_rows(ROOT / "data" / f"{split}_requests.jsonl")
        gold = read_rows(ROOT / "data" / "oracle" / f"{split}_gold.jsonl")
        if len(requests) != len(gold):
            raise AssertionError(f"{split}: request/gold row count differs")
        mismatch_ids = []
        for request, label in zip(requests, gold):
            if request["request_id"] in ids or request["lane_id"] in lanes:
                raise AssertionError(f"cross-split duplicate request/lane: {request['request_id']}")
            ids.add(request["request_id"])
            lanes.add(request["lane_id"])
            forbidden = {"gold", "oracle", "truth", "answer_key"} & set(request)
            if forbidden:
                raise AssertionError(f"runtime request leaks labels: {forbidden}")
            expected = independent_quote(request, lane_to_rows.get(request["lane_id"], []))
            if expected != label:
                mismatch_ids.append(request["request_id"])
        report["split_counts"][split] = {
            "requests": len(requests),
            "quotes": sum(row["decision"] == "quote" for row in gold),
            "holds": sum(row["decision"] == "hold" for row in gold),
            "visible_cards": sum(bool(row["source_visible"]) for row in requests),
            "mismatch_ids": mismatch_ids,
        }
        if mismatch_ids:
            report["passed"] = False
        report["oracle_crosscheck"][split] = "pass" if not mismatch_ids else "fail"
    if report["passed"] and len(read_rows(ROOT / "data" / "test_requests.jsonl")) < 40:
        raise AssertionError("acceptance requires at least 40 test cases")
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))

