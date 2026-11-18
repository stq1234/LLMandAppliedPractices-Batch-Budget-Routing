"""Fit small development-only routing priors from observable tool outcomes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from model_api import parse_candidate
from runtime_tools import (execute_quote, noisy_verify, offline_first_round,
                           revise_from_evidence, rows_for_lane, verifier_label)


ROOT = Path(__file__).resolve().parents[1]
ACTION_LATENCY_MS = {"generate": 2300, "query": 45, "execute": 25,
                     "validate": 35, "revise": 120, "rollback": 35}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _exact(candidate: dict, receipt: dict | None, gold: dict) -> bool:
    if candidate.get("decision") == "hold" and candidate.get("carrier_id") is None:
        return gold["decision"] == "hold"
    # The executor output is evidence for verification, not silently merged into
    # the assistant response. A local revision must incorporate it explicitly.
    return all(candidate.get(k) == gold.get(k)
               for k in ("decision", "carrier_id", "amount_cents", "eta_days"))


def load_first_round(split: str, cases: list[dict]) -> tuple[dict[str, dict], str]:
    path = ROOT / "data" / "model_outputs" / f"{split}_first_round_api.jsonl"
    if path.exists():
        outputs = {row["request_id"]: row for row in read_jsonl(path)}
        if set(outputs) != {case["request_id"] for case in cases}:
            raise ValueError(f"API output cache for {split} does not exactly match request IDs")
        return {rid: parse_candidate(row["raw_content"]) for rid, row in outputs.items()}, "deepseek-v4.1-flash"
    return {case["request_id"]: offline_first_round(case) for case in cases}, "deterministic_offline_stub"


def calibrate() -> dict:
    cases = read_jsonl(ROOT / "data" / "dev_requests.jsonl")
    labels = {r["request_id"]: r for r in read_jsonl(ROOT / "data" / "oracle" / "dev_gold.jsonl")}
    catalog = read_jsonl(ROOT / "data" / "rate_catalog.jsonl")
    candidates, base_model = load_first_round("dev", cases)
    groups = defaultdict(lambda: {"n": 0, "initial_correct": 0, "after_execution_correct": 0,
                                  "after_adaptive_path_correct": 0, "flag_count": 0,
                                  "flag_recovery_correct": 0, "validation_label_true": 0,
                                  "validation_pass_true": 0, "validation_pass_false": 0,
                                  "validation_label_false": 0, "validation_eligible": 0,
                                  "validation_gain_sum": 0})
    counters = defaultdict(int)
    for case in cases:
        rid = case["request_id"]
        gold = labels[rid]
        candidate = candidates[rid]
        visible = bool(case["visible_rate_card"])
        group_key = "visible" if visible else "lookup_required"
        stat = groups[group_key]
        stat["n"] += 1
        initial_ok = _exact(candidate, None, gold)
        stat["initial_correct"] += int(initial_ok)
        rows = case["visible_rate_card"] or rows_for_lane(catalog, case["lane_id"])
        has_route_candidate = bool(candidate.get("carrier_id")) or (
            candidate.get("decision") == "hold" and candidate.get("carrier_id") is None)
        receipt = execute_quote(case, candidate, rows) if has_route_candidate else None
        after_exec_ok = _exact(candidate, receipt, gold)
        stat["after_execution_correct"] += int(after_exec_ok)
        if not has_route_candidate:
            # Missing route facts cause a query followed by an evidence-based
            # local repair. This recovery gain is not attributed to the noisy
            # validator, which runs only after an executable candidate exists.
            repaired = revise_from_evidence(case, rows)
            new_receipt = execute_quote(case, repaired, rows) if repaired.get("carrier_id") or repaired.get("decision") == "hold" else None
            counters["revised_without_candidate"] += 1
        elif receipt and receipt.get("status") in ("rejected", "hold_rejected"):
            # A trusted executor rejection is already an observable hard
            # constraint failure, so the controller repairs without asking the
            # noisy validator to re-discover it.
            repaired = revise_from_evidence(case, rows)
            new_receipt = execute_quote(case, repaired, rows) if repaired.get("carrier_id") or repaired.get("decision") == "hold" else None
            counters["revised_after_executor_rejection"] += 1
        elif receipt:
            label = verifier_label(candidate, receipt)
            signal = noisy_verify(case, candidate, receipt)["signal"]
            stat["validation_eligible"] += 1
            stat["validation_label_true"] += int(label)
            stat["validation_label_false"] += int(not label)
            stat["validation_pass_true"] += int(label and signal == "pass")
            stat["validation_pass_false"] += int((not label) and signal == "pass")
            flagged = signal == "flag"
            stat["flag_count"] += int(flagged)
            if flagged:
                repaired = revise_from_evidence(case, rows)
                new_receipt = execute_quote(case, repaired, rows) if repaired.get("carrier_id") else None
                recovered = _exact(repaired, new_receipt, gold)
                stat["flag_recovery_correct"] += int(recovered)
                counters["revised_after_flag"] += 1
            else:
                repaired = candidate
                new_receipt = receipt
            stat["validation_gain_sum"] += int(_exact(repaired, new_receipt, gold)) - int(_exact(candidate, receipt, gold))
        else:
            repaired = candidate
            new_receipt = receipt
        stat["after_adaptive_path_correct"] += int(_exact(repaired, new_receipt, gold))
        # Always measure the tool-grounded repair as a development counterfactual.
        best = revise_from_evidence(case, rows)
        best_receipt = execute_quote(case, best, rows) if best.get("carrier_id") else None
        counters[f"full_repair_correct_{group_key}"] += int(_exact(best, best_receipt, gold))

    rates = {}
    for key, stat in groups.items():
        n = max(stat["n"], 1)
        valid = stat["validation_label_true"]
        invalid = stat["validation_label_false"]
        tpr = stat["validation_pass_true"] / valid if valid else None
        fpr = stat["validation_pass_false"] / invalid if invalid else None
        rates[key] = {
            "n": stat["n"],
            "initial_exact_rate": stat["initial_correct"] / n,
            "after_execution_exact_rate": stat["after_execution_correct"] / n,
            "after_development_policy_path_exact_rate": stat["after_adaptive_path_correct"] / n,
            "validator_pass_rate_on_consistent_candidates": tpr,
            "validator_false_pass_rate_on_inconsistent_candidates": fpr,
            "validator_eligible_cases": stat["validation_eligible"],
            "validator_expected_exact_gain_per_eligible_case": (
                stat["validation_gain_sum"] / stat["validation_eligible"]
                if stat["validation_eligible"] else 0.0),
            "validator_flag_count": stat["flag_count"],
            "flagged_cases_repaired_exactly": stat["flag_recovery_correct"],
        }

    initial = {k: v["initial_exact_rate"] for k, v in rates.items()}
    full_repair = {
        key: counters[f"full_repair_correct_{key}"] / max(groups[key]["n"], 1)
        for key in groups
    }
    gain = {
        "generate": 1.0,
        "query": max(0.0, full_repair.get("lookup_required", 0.0)
                     - initial.get("lookup_required", 0.0)),
        "execute": 0.05,
        "validate": max((v["validator_expected_exact_gain_per_eligible_case"] for v in rates.values()), default=0.0),
        "revise": max(0.0, full_repair.get("lookup_required", 0.0)
                      - rates.get("lookup_required", {}).get("after_execution_exact_rate", 0)),
    }
    result = {
        "calibration_split": "dev",
        "first_round_source": base_model,
        "n_dev": len(cases),
        "quality_rates_by_observable_source_state": rates,
        "action_gain_priors": gain,
        "policy_parameters": {"minimum_marginal_value_points": 8.0,
                              "exploration_fraction": 0.08,
                              "max_local_revisions_per_request": 1,
                              "action_cost_units": {name: 1 for name in ACTION_LATENCY_MS},
                              "simulated_latency_mean_ms": ACTION_LATENCY_MS},
        "note": "Gold labels were used only here to fit aggregate development priors; runtime routing reads this summary, never dev/test labels.",
    }
    out = ROOT / "results" / "dev_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(calibrate(), ensure_ascii=False, indent=2))

