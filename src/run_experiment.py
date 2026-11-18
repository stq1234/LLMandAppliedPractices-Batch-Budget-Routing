"""Deterministic finite-budget policy replay for the synthetic quote workload."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from calibrate import ACTION_LATENCY_MS, load_first_round, read_jsonl
from runtime_tools import (execute_quote, noisy_verify, revise_from_evidence,
                           rows_for_lane)


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("fifo", "value_first", "adaptive", "adaptive_noisy_signal_off")


def _latency(case: dict, action: str, attempt: int = 0) -> int:
    key = f"{case['request_id']}|{action}|{attempt}|sim-v1".encode()
    jitter = int.from_bytes(hashlib.sha256(key).digest()[:2], "big") % 451
    base = ACTION_LATENCY_MS[action]
    if action in ("query", "execute", "validate", "revise"):
        return base + jitter % 81
    return base + jitter


def _empty_state(case: dict, first: dict, cached: dict | None) -> dict:
    visible = case.get("visible_rate_card") or []
    return {
        "case": case,
        "candidate": None,
        "candidate_version": 0,
        "generated": False,
        "query_done": bool(visible),
        "rate_rows": visible,
        "receipts": {},
        "signals": {},
        "revision_count": 0,
        "rollback_candidate": None,
        "use_validator": True,
        "elapsed_ms": 0,
        "units": 0,
        "actions": [],
        "terminal": False,
        "stop_reason": None,
        "first_round": first,
        "api_record": cached,
        "group": "source_visible" if visible else "lookup_required",
    }


def _current_receipt(state: dict) -> dict | None:
    return state["receipts"].get(state["candidate_version"])


def _current_signal(state: dict) -> dict | None:
    return state["signals"].get(state["candidate_version"])


def _next_action(state: dict) -> str | None:
    if state["terminal"]:
        return None
    if not state["generated"]:
        return "generate"
    if not state["rate_rows"] and not state["query_done"]:
        return "query"
    candidate = state["candidate"] or {}
    receipt = _current_receipt(state)
    signal = _current_signal(state)
    if candidate.get("decision") == "need_data" and candidate.get("carrier_id") is None:
        return "revise" if state["revision_count"] < 1 else None
    if candidate.get("decision") == "unparsed" or not candidate.get("carrier_id") and candidate.get("decision") != "hold":
        return "revise" if state["revision_count"] < 1 else None
    if receipt is None:
        return "execute"
    if receipt.get("status") in ("rejected", "hold_rejected"):
        return "revise" if state["revision_count"] < 1 else "rollback"
    if not state["use_validator"]:
        state["terminal"] = True
        state["stop_reason"] = "signal_ablation_stopped_after_executor"
        return None
    if signal is None:
        return "validate"
    if signal.get("signal") == "flag" and state["revision_count"] < 1:
        return "revise"
    state["terminal"] = True
    state["stop_reason"] = "validated_pass" if signal.get("signal") == "pass" else "validation_flag_after_repair"
    return None


def _adaptive_gain(action: str, state: dict, calibration: dict) -> float:
    priors = calibration["action_gain_priors"]
    visible_rates = calibration["quality_rates_by_observable_source_state"]
    group = "visible" if state["group"] == "source_visible" else "lookup_required"
    if action == "generate":
        return max(0.05, visible_rates.get(group, {}).get("initial_exact_rate", 0.05))
    if action == "query":
        return max(0.05, priors.get("query", 0.05))
    if action == "execute":
        validation_gain = max(
            0.0,
            visible_rates.get(group, {}).get("validator_expected_exact_gain_per_eligible_case", 0.0),
        )
        return max(priors.get("execute", 0.02), validation_gain)
    if action == "validate":
        return max(
            0.0,
            visible_rates.get(group, {}).get("validator_expected_exact_gain_per_eligible_case", 0.0),
        )
    if action == "revise":
        return max(0.05, priors.get("revise", 0.05))
    if action == "rollback":
        return 1.0
    return 0.0


def _action_score(policy: str, action: str, state: dict, calibration: dict) -> float:
    case = state["case"]
    value = float(case["value_weight"])
    deadline = max(float(case["service_time_limit_ms"]), 1.0)
    elapsed_including_queue = state["elapsed_ms"] + state.get("queue_wait_ms", 0)
    urgency = 1.0 + 0.25 * min(elapsed_including_queue / deadline, 1.0)
    if policy == "fifo":
        return -float(case["batch_position"])
    if policy == "value_first":
        return value * urgency
    return value * _adaptive_gain(action, state, calibration) * urgency


def _answer(state: dict) -> dict:
    candidate = state["candidate"] or {}
    return {k: candidate.get(k) for k in
            ("decision", "carrier_id", "amount_cents", "eta_days", "reason")}


def _perform_action(state: dict, action: str, catalog: list[dict],
                    latency_override_ms: int | None = None,
                    queue_wait_ms: int | None = None,
                    service_start_ms: int | None = None) -> None:
    case = state["case"]
    output: dict
    if action == "generate":
        state["candidate"] = dict(state["first_round"])
        state["generated"] = True
        state["candidate_version"] = 1
        output = {"candidate": state["candidate"], "model_source":
                  "deepseek-v4.1-flash" if state["api_record"] else "deterministic_offline_stub"}
    elif action == "query":
        query = {"rows": rows_for_lane(catalog, case["lane_id"]),
                 "source": "synthetic_carrier_catalog"}
        state["rate_rows"] = query["rows"]
        state["query_done"] = True
        output = {"lane_id": case["lane_id"], "rows": query["rows"],
                  "source": query["source"], "row_count": len(query["rows"])}
    elif action == "execute":
        receipt = execute_quote(case, state["candidate"], state["rate_rows"])
        state["receipts"][state["candidate_version"]] = receipt
        output = receipt
    elif action == "validate":
        receipt = _current_receipt(state)
        result = noisy_verify(case, state["candidate"], receipt or {})
        state["signals"][state["candidate_version"]] = result
        output = result
    elif action == "revise":
        state["rollback_candidate"] = dict(state["candidate"])
        updated = revise_from_evidence(case, state["rate_rows"])
        state["candidate"] = updated
        state["candidate_version"] += 1
        state["revision_count"] += 1
        output = {"candidate": updated, "method": "local_evidence_patch"}
    elif action == "rollback":
        state["candidate"] = dict(state["rollback_candidate"] or state["candidate"])
        state["candidate_version"] += 1
        state["terminal"] = True
        state["stop_reason"] = "rollback_after_repair_rejected"
        output = {"candidate": state["candidate"],
                  "reason": "revised route was rejected; restored prior answer"}
    else:
        raise ValueError(f"unknown action: {action}")

    latency = latency_override_ms if latency_override_ms is not None else _latency(case, action, state["units"])
    state["elapsed_ms"] += latency
    state["units"] += 1
    event = {"step": len(state["actions"]) + 1, "action": action,
             "resource_units": 1, "simulated_latency_ms": latency,
             "elapsed_simulated_ms": state["elapsed_ms"], "output": output}
    if queue_wait_ms is not None:
        state["total_queue_wait_ms"] = state.get("total_queue_wait_ms", 0) + queue_wait_ms
        event["queue_wait_ms"] = queue_wait_ms
        event["service_start_ms"] = service_start_ms
        event["service_finish_ms"] = service_start_ms + latency
    if action == "generate" and state["api_record"]:
        usage = state["api_record"].get("usage") or {}
        event["captured_api_cost"] = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "api_elapsed_ms": state["api_record"].get("api_elapsed_ms"),
        }
    state["actions"].append(event)


def _truth_metrics(state: dict, gold: dict, catalog: list[dict]) -> dict:
    answer = _answer(state)
    exact = all(answer.get(k) == gold.get(k)
                for k in ("decision", "carrier_id", "amount_cents", "eta_days"))
    quoted = answer.get("decision") == "quote" and answer.get("carrier_id") is not None
    rows = rows_for_lane(catalog, state["case"]["lane_id"])
    selected = next((r for r in rows if r["carrier_id"] == answer.get("carrier_id")), None)
    feasible = bool(selected
                    and selected["capacity_kg"] >= state["case"]["weight_kg"]
                    and selected["eta_days"] <= state["case"]["latest_days"]
                    and state["case"]["service"] in selected["services"])
    if quoted and selected:
        amount = (selected["base_cents"]
                  + state["case"]["weight_kg"] * selected["per_kg_cents"]
                  + (state["case"]["service"] == "express") * selected["express_surcharge_cents"])
        regret = amount - gold["amount_cents"] if gold["decision"] == "quote" else None
    else:
        regret = None
    return {"exact": exact, "quoted": quoted, "feasible_quote": feasible,
            "regret_cents": regret,
            "answer_completed": answer.get("decision") in ("quote", "hold")
            and (not quoted or answer.get("amount_cents") is not None
                 and answer.get("eta_days") is not None)}


def _set_variant(cases: list[dict], variant: str) -> list[dict]:
    out = [dict(case) for case in cases]
    if variant == "normal":
        return out
    if variant == "reversed_order":
        out.reverse()
        for pos, case in enumerate(out):
            case["batch_position"] = pos
        return out
    if variant == "value_order_swap":
        values = sorted((c["value_weight"] for c in out), reverse=True)
        rng = random.Random(786)
        rng.shuffle(out)
        for pos, (case, value) in enumerate(zip(out, values)):
            case["batch_position"] = pos
            case["value_weight"] = value
        return out
    raise ValueError(variant)


def run_one(split: str, variant: str, policy: str, budget: int,
            cases: list[dict], candidates: dict[str, dict], api_records: dict[str, dict],
            catalog: list[dict], labels: dict[str, dict], calibration: dict) -> list[dict]:
    ordered = _set_variant(cases, variant)
    states = [_empty_state(case, candidates[case["request_id"]],
                           api_records.get(case["request_id"])) for case in ordered]
    for state in states:
        state["use_validator"] = policy != "adaptive_noisy_signal_off"
    used = 0
    explored_group_counts = Counter()
    quota = math.ceil(budget * calibration["policy_parameters"]["exploration_fraction"]) if policy.startswith("adaptive") else 0

    while used < budget:
        choices = []
        for state in states:
            action = _next_action(state)
            if action is None:
                continue
            case = state["case"]
            latency = _latency(case, action, state["units"])
            if state["elapsed_ms"] + latency > case["service_time_limit_ms"]:
                state["terminal"] = True
                state["stop_reason"] = "service_time_limit_exceeded_before_next_action"
                continue
            score = _action_score(policy, action, state, calibration)
            if policy.startswith("adaptive") and action in ("validate", "revise") and score < calibration["policy_parameters"]["minimum_marginal_value_points"]:
                state["terminal"] = True
                state["stop_reason"] = "low_expected_marginal_value"
                continue
            choices.append((score, case["request_id"], action, state))
        if not choices:
            break

        if policy.startswith("adaptive") and used < quota:
            least_sampled = min(explored_group_counts.get(s[3]["group"], 0) for s in choices)
            pool = [choice for choice in choices
                    if explored_group_counts.get(choice[3]["group"], 0) == least_sampled]
            chosen = max(pool, key=lambda item: (item[3]["case"]["value_weight"], -item[3]["case"]["batch_position"]))
            explored_group_counts[chosen[3]["group"]] += 1
        else:
            chosen = max(choices, key=lambda item: (item[0], -item[3]["case"]["batch_position"], item[1]))
        _, _, action, state = chosen
        _perform_action(state, action, catalog)
        used += 1

    traces = []
    for state in states:
        if not state["terminal"]:
            state["stop_reason"] = "batch_budget_exhausted"
        gold = labels[state["case"]["request_id"]]
        outcome = _truth_metrics(state, gold, catalog)
        trace = {
            "experiment": split,
            "variant": variant,
            "policy": policy,
            "budget_units": budget,
            "request_id": state["case"]["request_id"],
            "batch_position": state["case"]["batch_position"],
            "value_weight": state["case"]["value_weight"],
            "service_time_limit_ms": state["case"]["service_time_limit_ms"],
            "observable_group": state["group"],
            "actions": state["actions"],
            "answer": _answer(state),
            "stop_reason": state["stop_reason"] or "no_branch_available",
            "resource_units_used": state["units"],
            "simulated_latency_ms": state["elapsed_ms"],
            "evaluation": outcome,
        }
        traces.append(trace)
    return traces


def summarize(traces: list[dict], dataset_capture: int, base_model: str) -> dict:
    n = len(traces)
    total_value = sum(row["value_weight"] for row in traces) or 1
    action_counts = Counter(action["action"] for row in traces for action in row["actions"])
    exact = sum(row["evaluation"]["exact"] for row in traces)
    weighted = sum(row["value_weight"] * row["evaluation"]["exact"] for row in traces)
    latencies = [row["simulated_latency_ms"] for row in traces]
    p95 = sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1] if latencies else 0
    api_rows = [event["captured_api_cost"] for row in traces for event in row["actions"]
                if "captured_api_cost" in event]
    reg = [row["evaluation"]["regret_cents"] for row in traces
           if row["evaluation"]["regret_cents"] is not None]
    executed = sum(row["evaluation"]["quoted"] for row in traces)
    completed = sum(row["evaluation"]["answer_completed"] for row in traces)
    flags = sum(action.get("output", {}).get("signal") == "flag"
                for row in traces for action in row["actions"] if action["action"] == "validate")
    generated = action_counts["generate"]
    value_cut = sorted(row["value_weight"] for row in traces)[math.ceil(0.75 * n) - 1]
    top_value_spend = sum(row["resource_units_used"] for row in traces
                          if row["value_weight"] >= value_cut)
    total_spend = sum(row["resource_units_used"] for row in traces)
    return {
        "n_requests": n,
        "exact_rate": exact / max(n, 1),
        "value_weighted_exact_rate": weighted / total_value,
        "answer_completion_rate": completed / max(n, 1),
        "quote_coverage": executed / max(n, 1),
        "feasible_quote_rate_all": sum(row["evaluation"]["feasible_quote"] for row in traces) / max(n, 1),
        "mean_regret_cents_given_quote": statistics.mean(reg) if reg else None,
        "resource_units_used": sum(row["resource_units_used"] for row in traces),
        "mean_resource_units_per_request": statistics.mean(row["resource_units_used"] for row in traces),
        "spend_on_top_quartile_value_fraction": top_value_spend / total_spend if total_spend else 0,
        "action_counts": dict(action_counts),
        "validation_flags": flags,
        "simulated_latency_mean_ms": statistics.mean(latencies) if latencies else 0,
        "simulated_latency_p95_ms": p95,
        "service_time_limit_stop_rate": sum(row["stop_reason"].startswith("service_time_limit") for row in traces) / max(n, 1),
        "api_first_round_replay_calls_imputed": generated if base_model == "deepseek-v4.1-flash" else 0,
        "api_prompt_tokens_replayed_imputed": sum(int(x.get("prompt_tokens") or 0) for x in api_rows),
        "api_completion_tokens_replayed_imputed": sum(int(x.get("completion_tokens") or 0) for x in api_rows),
        "api_elapsed_ms_captured_on_selected_requests": round(sum(float(x.get("api_elapsed_ms") or 0) for x in api_rows), 2),
        "api_output_capture_calls_for_split": dataset_capture,
        "first_round_source": base_model,
    }


def _read_cache(split: str, cases: list[dict]) -> tuple[dict[str, dict], dict[str, dict], str, int]:
    candidates, source = load_first_round(split, cases)
    path = ROOT / "data" / "model_outputs" / f"{split}_first_round_api.jsonl"
    if path.exists():
        records = {row["request_id"]: row for row in read_jsonl(path)}
        return candidates, records, source, len(records)
    return candidates, {}, source, 0


def _write_traces(path: Path, traces: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in traces:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(budgets: list[int] | None = None) -> dict:
    calibration_path = ROOT / "results" / "dev_calibration.json"
    if not calibration_path.exists():
        raise FileNotFoundError("Run `python src/calibrate.py` before replay")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    budgets = budgets or [48, 96, 144, 240]
    catalog = read_jsonl(ROOT / "data" / "rate_catalog.jsonl")
    splits = {name: read_jsonl(ROOT / "data" / f"{name}_requests.jsonl")
              for name in ("test", "shift")}
    labels = {name: {row["request_id"]: row for row in read_jsonl(ROOT / "data" / "oracle" / f"{name}_gold.jsonl")}
              for name in splits}
    rows_out, traces_written = [], 0
    trace_dir = ROOT / "results" / "traces"
    for split, cases in splits.items():
        candidates, api_records, base_model, capture_count = _read_cache(split, cases)
        variants = [("normal", budgets)]
        if split == "test":
            variants += [("reversed_order", [96, 144]), ("value_order_swap", [96, 144])]
        for variant, variant_budgets in variants:
            for budget in variant_budgets:
                for policy in POLICIES:
                    traces = run_one(split, variant, policy, budget, cases, candidates,
                                     api_records, catalog, labels[split], calibration)
                    path = trace_dir / f"{split}_{variant}_{policy}_b{budget}.jsonl"
                    _write_traces(path, traces)
                    traces_written += len(traces)
                    metrics = summarize(traces, capture_count, base_model)
                    row = {"experiment": split, "variant": variant, "policy": policy,
                           "budget_units": budget, **metrics}
                    rows_out.append(row)
    result_dir = ROOT / "results"
    _write_csv(result_dir / "summary.csv", rows_out)
    (result_dir / "summary.json").write_text(
        json.dumps({"calibration_file": "dev_calibration.json",
                    "policies": list(POLICIES), "budgets": budgets,
                    "trace_records_written": traces_written, "summary": rows_out},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    allocation = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        trace_rows = read_jsonl(path)
        spend = sum(row["resource_units_used"] for row in trace_rows)
        top = sorted(row["value_weight"] for row in trace_rows)[math.ceil(0.75 * len(trace_rows)) - 1]
        top_spend = sum(row["resource_units_used"] for row in trace_rows
                        if row["value_weight"] >= top)
        allocation.append({"trace_file": path.name, "requests": len(trace_rows),
                           "resource_units_used": spend, "top_quartile_value_cutoff": top,
                           "spend_on_top_quartile_value_fraction": top_spend / spend if spend else 0})
    _write_csv(result_dir / "allocation_diagnostics.csv", allocation)
    return {"summary_rows": len(rows_out), "trace_records": traces_written,
            "base_model_by_split": {split: _read_cache(split, cases)[2] for split, cases in splits.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", default="48,96,144,240",
                        help="Comma-separated policy resource budgets in action units")
    args = parser.parse_args()
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    print(json.dumps(run(budgets), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

