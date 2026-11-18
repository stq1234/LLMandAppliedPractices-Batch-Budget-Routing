"""Summarize actual captured first-round model outputs against held-out labels."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from calibrate import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def analyze() -> list[dict]:
    rows = []
    all_records = []
    for split in ("dev", "test", "shift"):
        requests = {r["request_id"]: r for r in read_jsonl(ROOT / "data" / f"{split}_requests.jsonl")}
        labels = {r["request_id"]: r for r in read_jsonl(ROOT / "data" / "oracle" / f"{split}_gold.jsonl")}
        path = ROOT / "data" / "model_outputs" / f"{split}_first_round_api.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing first-round API output: {path}")
        outputs = read_jsonl(path)
        if len(outputs) != len(requests) or {r["request_id"] for r in outputs} != set(requests):
            raise ValueError(f"Incomplete or duplicate model output IDs for split {split}")
        all_records.extend(outputs)
        for visibility in ("visible", "lookup_required"):
            selected = [record for record in outputs
                        if bool(requests[record["request_id"]]["source_visible"]) == (visibility == "visible")]
            if not selected:
                continue
            n = len(selected)
            exact = 0
            complete = 0
            quote = 0
            feasible = 0
            need_data = 0
            unparsed = 0
            trunc = 0
            for record in selected:
                case = requests[record["request_id"]]
                gold = labels[record["request_id"]]
                candidate = record["candidate"]
                exact += int(all(candidate.get(k) == gold.get(k)
                                 for k in ("decision", "carrier_id", "amount_cents", "eta_days")))
                complete += int(candidate.get("decision") == "quote"
                                 and candidate.get("carrier_id") is not None
                                 and candidate.get("amount_cents") is not None
                                 and candidate.get("eta_days") is not None)
                need_data += int(candidate.get("decision") == "need_data")
                unparsed += int(candidate.get("decision") == "unparsed")
                trunc += int(record.get("truncated") or record.get("finish_reason") == "length")
                if candidate.get("decision") == "quote":
                    quote += 1
                    row = next((r for r in case["visible_rate_card"] or []
                                if r["carrier_id"] == candidate.get("carrier_id")), None)
                    if row and row["capacity_kg"] >= case["weight_kg"] \
                            and row["eta_days"] <= case["latest_days"] \
                            and case["service"] in row["services"]:
                        feasible += 1
            usages = [record.get("usage") or {} for record in selected]
            elapsed = [float(record.get("api_elapsed_ms") or 0) for record in selected]
            server_models = sorted({str(record.get("server_model")) for record in selected})
            finish_counts = {}
            for record in selected:
                reason = str(record.get("finish_reason"))
                finish_counts[reason] = finish_counts.get(reason, 0) + 1
            rows.append({
                "split": split,
                "context_group": visibility,
                "requests": n,
                "server_models": ";".join(server_models),
                "finish_reason_counts": json.dumps(finish_counts, sort_keys=True),
                "truncated": trunc,
                "unparsed_json": unparsed,
                "need_data": need_data,
                "complete_quote_answers": complete,
                "quote_answers": quote,
                "feasible_quotes_with_visible_card": feasible if visibility == "visible" else None,
                "exact_answer_count": exact,
                "exact_answer_rate": exact / n,
                "prompt_tokens_total": sum(int(u.get("prompt_tokens") or 0) for u in usages),
                "completion_tokens_total": sum(int(u.get("completion_tokens") or 0) for u in usages),
                "api_elapsed_ms_mean": round(statistics.mean(elapsed), 2),
                "api_elapsed_ms_p95": round(sorted(elapsed)[int(0.95 * (n - 1))], 2),
            })
    out = ROOT / "results" / "model_output_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "results" / "model_output_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finish_counts = {}
    for record in all_records:
        reason = str(record.get("finish_reason"))
        finish_counts[reason] = finish_counts.get(reason, 0) + 1
    api_times = [float(record.get("api_elapsed_ms") or 0) for record in all_records]
    usage = [record.get("usage") or {} for record in all_records]
    manifest = {
        "requested_model": "deepseek-v4.1-flash",
        "server_models": sorted({str(record.get("server_model")) for record in all_records}),
        "captured_first_round_calls": len(all_records),
        "prompt_tokens_total": sum(int(item.get("prompt_tokens") or 0) for item in usage),
        "completion_tokens_total": sum(int(item.get("completion_tokens") or 0) for item in usage),
        "actual_api_elapsed_ms_sum_of_individual_calls": round(sum(api_times), 2),
        "actual_api_elapsed_ms_mean_per_call": round(statistics.mean(api_times), 2) if api_times else None,
        "actual_api_elapsed_ms_p95_per_call": round(sorted(api_times)[int(0.95 * (len(api_times) - 1))], 2)
        if api_times else None,
        "finish_reason_counts": finish_counts,
        "truncated_outputs": sum(record.get("truncated") is True for record in all_records),
        "unparsed_json_outputs": sum(record.get("candidate", {}).get("decision") == "unparsed"
                                     for record in all_records),
        "api_currency_cost": None,
        "note": "The elapsed values are the sum/statistics of per-call API timers; they are not replay wall time or simulated service/queue latency.",
    }
    (ROOT / "results" / "api_collection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))

