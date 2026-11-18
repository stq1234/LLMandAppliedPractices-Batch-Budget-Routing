"""Runtime-visible source query, quote execution, noisy verification, and repair.

This module never reads anything below data/oracle and does not import the
ground-truth implementation. Every output is derived from the catalog or the
current candidate.
"""

from __future__ import annotations

import hashlib


def rows_for_lane(catalog: list[dict], lane_id: str) -> list[dict]:
    return [row for row in catalog if row["lane_id"] == lane_id]


def query_rate_card(case: dict, catalog: list[dict]) -> dict:
    rows = rows_for_lane(catalog, case["lane_id"])
    return {"lane_id": case["lane_id"], "rows": rows,
            "source": "synthetic_carrier_catalog", "found": bool(rows)}


def _price(row: dict, case: dict) -> int:
    return (row["base_cents"] + case["weight_kg"] * row["per_kg_cents"]
            + (case["service"] == "express") * row["express_surcharge_cents"])


def execute_quote(case: dict, candidate: dict, visible_rows: list[dict]) -> dict:
    carrier = candidate.get("carrier_id")
    if candidate.get("decision") == "hold" and carrier is None:
        feasible = [r for r in visible_rows
                    if r["capacity_kg"] >= case["weight_kg"]
                    and r["eta_days"] <= case["latest_days"]
                    and case["service"] in r["services"]]
        return {"status": "hold_rejected" if feasible else "hold_confirmed",
                "carrier_id": None, "amount_cents": None, "eta_days": None,
                "reason": "an eligible route exists" if feasible else "no eligible route"}
    row = next((r for r in visible_rows if r["carrier_id"] == carrier), None)
    if row is None:
        return {"status": "rejected", "carrier_id": carrier,
                "reason": "carrier missing from the currently available rate card",
                "amount_cents": None, "eta_days": None}
    if row["capacity_kg"] < case["weight_kg"]:
        reason = "capacity_exceeded"
    elif row["eta_days"] > case["latest_days"]:
        reason = "arrival_after_deadline"
    elif case["service"] not in row["services"]:
        reason = "service_not_supported"
    else:
        return {"status": "quoted", "carrier_id": carrier,
                "amount_cents": _price(row, case), "eta_days": row["eta_days"],
                "reason": "quote calculated from the selected catalog row"}
    return {"status": "rejected", "carrier_id": carrier, "amount_cents": None,
            "eta_days": row["eta_days"], "reason": reason}


def _candidate_matches_receipt(candidate: dict, receipt: dict) -> bool:
    if candidate.get("decision") == "hold" and candidate.get("carrier_id") is None:
        return receipt.get("status") == "hold_confirmed"
    return (
        receipt.get("status") == "quoted"
        and candidate.get("carrier_id") == receipt.get("carrier_id")
        and candidate.get("amount_cents") == receipt.get("amount_cents")
        and candidate.get("eta_days") == receipt.get("eta_days")
        and candidate.get("decision") == "quote"
    )


def verifier_label(candidate: dict, receipt: dict) -> bool:
    """Private calibration/evaluation label; never returned to the router."""
    return _candidate_matches_receipt(candidate, receipt)


def _unit_interval(seed_text: str) -> float:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / (2**64)


def noisy_verify(case: dict, candidate: dict, receipt: dict,
                 sensitivity: float = 0.90, specificity: float = 0.88) -> dict:
    """Return only an imperfect runtime signal, never the hidden checker label."""
    consistent = _candidate_matches_receipt(candidate, receipt)
    u = _unit_interval(f"{case['request_id']}|{candidate}|{receipt}|verify-v1")
    passes = (u < specificity) if consistent else (u < 1.0 - sensitivity)
    return {"signal": "pass" if passes else "flag",
            "checker": "noisy_quote_consistency_v1"}


def revise_from_evidence(case: dict, rate_rows: list[dict]) -> dict:
    """Deterministic local repair using only the task card and queried rate rows."""
    feasible = [r for r in rate_rows
                if r["capacity_kg"] >= case["weight_kg"]
                and r["eta_days"] <= case["latest_days"]
                and case["service"] in r["services"]]
    if not feasible:
        return {"decision": "hold", "carrier_id": None, "amount_cents": None,
                "eta_days": None, "reason": "no feasible carrier in retrieved rate card"}
    best = min(feasible, key=lambda r: (_price(r, case), r["eta_days"], r["carrier_id"]))
    return {"decision": "quote", "carrier_id": best["carrier_id"],
            "amount_cents": _price(best, case), "eta_days": best["eta_days"],
            "reason": "locally revised from retrieved constraints and rate card"}


def offline_first_round(case: dict) -> dict:
    """Deterministic stand-in for offline replay when no API cache is present.

    The stand-in follows visible facts on direct cases. Without the rate card it
    emits a deliberately incomplete first-pass candidate and asks for evidence.
    It never sees catalog rows or oracle labels.
    """
    visible = case.get("visible_rate_card")
    if visible:
        candidates = [r for r in visible
                      if r["capacity_kg"] >= case["weight_kg"]
                      and r["eta_days"] <= case["latest_days"]
                      and case["service"] in r["services"]]
        if not candidates:
            return {"decision": "hold", "carrier_id": None, "amount_cents": None,
                    "eta_days": None, "reason": "no visible feasible route"}
        best = min(candidates, key=lambda r: (_price(r, case), r["eta_days"], r["carrier_id"]))
        return {"decision": "quote", "carrier_id": best["carrier_id"],
                "amount_cents": _price(best, case), "eta_days": best["eta_days"],
                "reason": "selected from visible rate card"}
    return {"decision": "need_data", "carrier_id": case["carrier_ids"][0],
            "amount_cents": None, "eta_days": None,
            "reason": "rate and capacity information required"}

