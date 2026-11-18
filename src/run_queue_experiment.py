"""Shared-service queue replay with response deadlines and four service slots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from calibrate import load_first_round, read_jsonl
from run_experiment import (_action_score, _answer, _empty_state, _latency,
                           _next_action, _perform_action, _truth_metrics)


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("fifo", "value_first", "adaptive", "adaptive_noisy_signal_off")
CONDITIONS = {"standard": 1.0, "tight": 0.40}
QUEUE_SLOTS = 4
MINI_BATCH_SIZE = 8


def run_batch(batch: list[dict], split: str, deadline_condition: str,
              policy: str, budget: int, candidates: dict[str, dict],
              api_records: dict[str, dict], catalog: list[dict],
              labels: dict[str, dict], calibration: dict) -> list[dict]:
    scale = CONDITIONS[deadline_condition]
    states = []
    for case0 in batch:
        case = dict(case0)
        case["response_deadline_ms"] = round(case["response_deadline_ms"] * scale)
        state = _empty_state(case, candidates[case["request_id"]],
                             api_records.get(case["request_id"]))
        state["use_validator"] = policy != "adaptive_noisy_signal_off"
        state["arrival_ms"] = 0
        state["response_time_ms"] = None
        state["total_queue_wait_ms"] = 0
        states.append(state)

    active: dict[str, dict] = {}
    now = 0
    units_used = 0
    explored_groups = Counter()
    exploration_quota = (
        math.ceil(budget * calibration["policy_parameters"]["exploration_fraction"])
        if policy.startswith("adaptive") else 0
    )

    while True:
        # Complete all actions ending at the next event time, then update the
        # observable state before dispatching another action.
        if active:
            now = min(item["finish_ms"] for item in active.values())
            due = sorted((rid, item) for rid, item in active.items()
                         if item["finish_ms"] == now)
            for rid, item in due:
                state = item["state"]
                wait_ms = max(0, item["start_ms"] - state["elapsed_ms"]
                              - state["total_queue_wait_ms"])
                _perform_action(state, item["action"], catalog,
                                latency_override_ms=item["service_ms"],
                                queue_wait_ms=wait_ms,
                                service_start_ms=item["start_ms"])
                del active[rid]
                if policy == "first_round_only" and item["action"] == "generate":
                    state["terminal"] = True
                    state["stop_reason"] = "first_round_only_stop"
                    state["response_time_ms"] = now
                elif now > state["case"]["response_deadline_ms"]:
                    state["terminal"] = True
                    state["stop_reason"] = "timed_out_in_service"
                    state["response_time_ms"] = now

        # Terminal decisions and deadline misses are applied to idle requests.
        for state in states:
            rid = state["case"]["request_id"]
            if state["terminal"] or rid in active:
                continue
            if now >= state["case"]["response_deadline_ms"]:
                state["terminal"] = True
                state["stop_reason"] = "deadline_exceeded_before_dispatch"
                state["response_time_ms"] = now
                continue
            if policy == "first_round_only" and state["generated"]:
                state["terminal"] = True
                state["stop_reason"] = "first_round_only_stop"
                state["response_time_ms"] = now
                continue
            action = "generate" if policy == "first_round_only" and not state["generated"] else _next_action(state)
            if action is None:
                if state["terminal"]:
                    state["response_time_ms"] = now

        # Dispatch into all free service slots. Each operation occupies one
        # slot until its deterministic simulated service completes.
        while len(active) < QUEUE_SLOTS and units_used < budget:
            choices = []
            for state in states:
                rid = state["case"]["request_id"]
                if state["terminal"] or rid in active:
                    continue
                if policy == "first_round_only":
                    action = "generate" if not state["generated"] else None
                else:
                    action = _next_action(state)
                if action is None:
                    if state["terminal"] and state["response_time_ms"] is None:
                        state["response_time_ms"] = now
                    continue
                deadline = state["case"]["response_deadline_ms"]
                if now >= deadline:
                    state["terminal"] = True
                    state["stop_reason"] = "deadline_exceeded_before_dispatch"
                    state["response_time_ms"] = now
                    continue
                state["queue_wait_ms"] = max(
                    0, now - state["elapsed_ms"] - state["total_queue_wait_ms"])
                duration = _latency(state["case"], action, state["units"])
                if now + duration > deadline:
                    state["terminal"] = True
                    state["stop_reason"] = "deadline_exceeded_before_dispatch"
                    state["response_time_ms"] = now
                    continue
                score = _action_score(policy, action, state, calibration)
                if (policy.startswith("adaptive") and action in ("validate", "revise")
                        and score < calibration["policy_parameters"]["minimum_marginal_value_points"]):
                    state["terminal"] = True
                    state["stop_reason"] = "low_expected_marginal_value"
                    state["response_time_ms"] = now
                    continue
                choices.append((score, rid, action, state, duration))
            if not choices:
                break
            if policy.startswith("adaptive") and units_used < exploration_quota:
                least_seen = min(explored_groups.get(item[3]["group"], 0) for item in choices)
                pool = [item for item in choices
                        if explored_groups.get(item[3]["group"], 0) == least_seen]
                chosen = max(pool, key=lambda item: (item[3]["case"]["value_weight"],
                                                     -item[3]["case"]["batch_position"]))
                explored_groups[chosen[3]["group"]] += 1
            else:
                chosen = max(choices, key=lambda item: (item[0],
                                                         -item[3]["case"]["batch_position"],
                                                         item[1]))
            _, rid, action, state, duration = chosen
            active[rid] = {"state": state, "action": action, "service_ms": duration,
                           "start_ms": now, "finish_ms": now + duration}
            units_used += 1

        if active:
            continue
        if units_used >= budget:
            for state in states:
                if not state["terminal"]:
                    state["terminal"] = True
                    state["stop_reason"] = "batch_budget_exhausted"
                    state["response_time_ms"] = now
            break
        if all(state["terminal"] for state in states):
            break
        # No active job and no dispatchable action means the scheduler stopped
        # with a low-gain or invalid branch; record the actual current time.
        for state in states:
            if not state["terminal"]:
                state["terminal"] = True
                state["stop_reason"] = "unscheduled_no_dispatchable_action"
                state["response_time_ms"] = now
        break

    traces = []
    for state in states:
        if state["response_time_ms"] is None:
            state["response_time_ms"] = now if state["actions"] else None
        gold = labels[state["case"]["request_id"]]
        outcome = _truth_metrics(state, gold, catalog)
        deadline = state["case"]["response_deadline_ms"]
        on_time = state["response_time_ms"] is not None and state["response_time_ms"] <= deadline
        trace = {
            "experiment": split,
            "deadline_condition": deadline_condition,
            "policy": policy,
            "batch_id": state["case"]["batch_position"] // MINI_BATCH_SIZE,
            "batch_size": len(batch),
            "service_slots": QUEUE_SLOTS,
            "batch_budget_units": budget,
            "request_id": state["case"]["request_id"],
            "batch_position": state["case"]["batch_position"],
            "value_weight": state["case"]["value_weight"],
            "response_deadline_ms": deadline,
            "response_time_ms": state["response_time_ms"],
            "queue_wait_ms": state.get("total_queue_wait_ms", 0),
            "service_time_ms": state["elapsed_ms"],
            "on_time": on_time,
            "timed_out": state["stop_reason"] in (
                "deadline_exceeded_before_dispatch", "timed_out_in_service"),
            "actions": state["actions"],
            "answer": _answer(state),
            "stop_reason": state["stop_reason"],
            "resource_units_used": state["units"],
            "evaluation": outcome,
        }
        traces.append(trace)
    return traces


def _batches(cases: list[dict]) -> list[list[dict]]:
    ordered = sorted(cases, key=lambda case: case["batch_position"])
    return [ordered[i:i + MINI_BATCH_SIZE]
            for i in range(0, len(ordered), MINI_BATCH_SIZE)]


def _summarize(traces: list[dict], budget: int, split: str,
               condition: str, policy: str, capture_calls: int) -> dict:
    n = len(traces)
    value_total = sum(row["value_weight"] for row in traces) or 1
    exact_ontime = sum(row["evaluation"]["exact"] and row["on_time"] for row in traces)
    completed_ontime = sum(row["evaluation"]["answer_completed"] and row["on_time"] for row in traces)
    value_exact_ontime = sum(row["value_weight"] for row in traces
                             if row["evaluation"]["exact"] and row["on_time"])
    value_completed_ontime = sum(row["value_weight"] for row in traces
                                 if row["evaluation"]["answer_completed"] and row["on_time"])
    response = [row["response_time_ms"] for row in traces
                if row["response_time_ms"] is not None and row["resource_units_used"] > 0]
    wait = [row["queue_wait_ms"] for row in traces if row["resource_units_used"] > 0]
    service = [row["service_time_ms"] for row in traces]
    actions = Counter(event["action"] for row in traces for event in row["actions"])
    api_events = [event["captured_api_cost"] for row in traces for event in row["actions"]
                  if "captured_api_cost" in event]
    sorted_response = sorted(response)
    p95 = sorted_response[math.ceil(0.95 * len(sorted_response)) - 1] if sorted_response else None
    total_budget = budget * (n // MINI_BATCH_SIZE)
    return {
        "experiment": split,
        "deadline_condition": condition,
        "policy": policy,
        "requests": n,
        "mini_batches": n // MINI_BATCH_SIZE,
        "service_slots_per_batch": QUEUE_SLOTS,
        "budget_units_per_batch": budget,
        "budget_units_total": total_budget,
        "resource_units_used": sum(row["resource_units_used"] for row in traces),
        "resource_utilization": sum(row["resource_units_used"] for row in traces) / max(total_budget, 1),
        "action_counts": json.dumps(dict(actions), sort_keys=True),
        "answer_completion_rate": sum(row["evaluation"]["answer_completed"] for row in traces) / max(n, 1),
        "on_time_completion_rate": completed_ontime / max(n, 1),
        "exact_rate_all": sum(row["evaluation"]["exact"] for row in traces) / max(n, 1),
        "on_time_exact_rate": exact_ontime / max(n, 1),
        "on_time_value_coverage": value_completed_ontime / value_total,
        "on_time_exact_value_coverage": value_exact_ontime / value_total,
        "timeout_rate": sum(row["timed_out"] for row in traces) / max(n, 1),
        "timeout_count": sum(row["timed_out"] for row in traces),
        "undispatched_count": sum(row["resource_units_used"] == 0 for row in traces),
        "mean_queue_wait_ms_per_request": round(statistics.mean(wait), 2) if wait else 0,
        "mean_simulated_service_ms_per_request": round(statistics.mean(service), 2) if service else 0,
        "mean_response_time_ms": round(statistics.mean(response), 2) if response else None,
        "p95_response_time_ms": p95,
        "first_round_model_calls_imputed": actions["generate"] if api_events else 0,
        "first_round_prompt_tokens_imputed": sum(int(x.get("prompt_tokens") or 0) for x in api_events),
        "first_round_completion_tokens_imputed": sum(int(x.get("completion_tokens") or 0) for x in api_events),
        "captured_actual_api_elapsed_ms_selected_sum": round(sum(float(x.get("api_elapsed_ms") or 0)
                                                                  for x in api_events), 2),
        "api_output_capture_calls_for_split": capture_calls,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(budgets: list[int] | None = None) -> dict:
    budgets = budgets or [8, 16, 24, 40]
    calibration = json.loads((ROOT / "results" / "dev_calibration.json").read_text(encoding="utf-8"))
    catalog = read_jsonl(ROOT / "data" / "rate_catalog.jsonl")
    summary_rows = []
    trace_dir = ROOT / "results" / "queue_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for split in ("test", "shift"):
        cases = read_jsonl(ROOT / "data" / f"{split}_requests.jsonl")
        labels = {r["request_id"]: r for r in read_jsonl(ROOT / "data" / "oracle" / f"{split}_gold.jsonl")}
        candidates, source = load_first_round(split, cases)
        cache_path = ROOT / "data" / "model_outputs" / f"{split}_first_round_api.jsonl"
        api_records = ({r["request_id"]: r for r in read_jsonl(cache_path)}
                       if cache_path.exists() else {})
        capture_calls = len(api_records)
        batches = _batches(cases)
        for condition in CONDITIONS:
            for policy in POLICIES:
                for budget in budgets:
                    traces = []
                    for batch in batches:
                        traces.extend(run_batch(batch, split, condition, policy, budget,
                                                candidates, api_records, catalog, labels,
                                                calibration))
                    filename = f"{split}_{condition}_{policy}_b{budget}.jsonl"
                    with (trace_dir / filename).open("w", encoding="utf-8", newline="\n") as f:
                        for row in traces:
                            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    summary_rows.append(_summarize(traces, budget, split, condition,
                                                   policy, capture_calls))

            # Cost/quality reference: stop after the actual first model response.
            traces = []
            for batch in batches:
                traces.extend(run_batch(batch, split, condition, "first_round_only", MINI_BATCH_SIZE,
                                        candidates, api_records, catalog, labels, calibration))
            filename = f"{split}_{condition}_first_round_only_b{MINI_BATCH_SIZE}.jsonl"
            with (trace_dir / filename).open("w", encoding="utf-8", newline="\n") as f:
                for row in traces:
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            summary_rows.append(_summarize(traces, MINI_BATCH_SIZE, split, condition,
                                           "first_round_only", capture_calls))

    _write_csv(ROOT / "results" / "queue_summary.csv", summary_rows)
    (ROOT / "results" / "queue_summary.json").write_text(
        json.dumps({"service_slots": QUEUE_SLOTS, "mini_batch_size": MINI_BATCH_SIZE,
                    "deadline_scale": CONDITIONS, "summary": summary_rows},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"summary_rows": len(summary_rows),
            "trace_records": sum(row["requests"] for row in summary_rows),
            "mini_batches_per_split": len(_batches(read_jsonl(ROOT / "data" / "test_requests.jsonl"))),
            "service_slots": QUEUE_SLOTS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", default="8,16,24,40",
                        help="Resource units per synchronous mini-batch")
    args = parser.parse_args()
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    print(json.dumps(run(budgets), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

