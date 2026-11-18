"""Build a deterministic, synthetic quote-routing benchmark.

The request and catalog files are the only runtime inputs. Oracle labels are
written to data/oracle and are loaded only by calibration/evaluation code.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _eligible(row: dict, req: dict) -> bool:
    return (
        row["capacity_kg"] >= req["weight_kg"]
        and row["eta_days"] <= req["latest_days"]
        and req["service"] in row["services"]
    )


def oracle_v1(req: dict, rows: list[dict]) -> dict:
    """Primary ground-truth calculation; intentionally outside runtime code."""
    feasible = [r for r in rows if _eligible(r, req)]
    if not feasible:
        return {"request_id": req["request_id"], "decision": "hold", "carrier_id": None,
                "amount_cents": None, "eta_days": None, "feasible_count": 0}
    ranked = sorted(
        feasible,
        key=lambda r: (
            r["base_cents"] + req["weight_kg"] * r["per_kg_cents"]
            + (req["service"] == "express") * r["express_surcharge_cents"],
            r["eta_days"], r["carrier_id"],
        ),
    )
    best = ranked[0]
    price = (best["base_cents"] + req["weight_kg"] * best["per_kg_cents"]
             + (req["service"] == "express") * best["express_surcharge_cents"])
    return {"request_id": req["request_id"], "decision": "quote",
            "carrier_id": best["carrier_id"], "amount_cents": price,
            "eta_days": best["eta_days"], "feasible_count": len(feasible)}


def make_case(rng: random.Random, split: str, index: int, shifted: bool) -> tuple[dict, list[dict], dict]:
    rid = f"{split}_{index:03d}"
    lane = f"LANE_{split.upper()}_{index:03d}"
    n_carriers = rng.randint(4, 6) if shifted else rng.randint(3, 5)
    rows = []
    for c in range(n_carriers):
        cid = f"C{c + 1:02d}"
        if shifted:
            capacity = rng.choice([500, 900, 1200, 1700, 2400, 3600])
            eta = rng.choice([1, 2, 3, 4, 5, 6])
            base = rng.randint(300, 1800)
            perkg = rng.randint(2, 11)
            services = ["standard"]
            if rng.random() < 0.58:
                services.append("express")
        else:
            capacity = rng.choice([900, 1400, 2100, 3000, 4500])
            eta = rng.choice([1, 2, 2, 3, 4])
            base = rng.randint(250, 1400)
            perkg = rng.randint(2, 9)
            services = ["standard"]
            if rng.random() < 0.78:
                services.append("express")
        rows.append({
            "lane_id": lane,
            "carrier_id": cid,
            "base_cents": base,
            "per_kg_cents": perkg,
            "capacity_kg": capacity,
            "eta_days": eta,
            "services": services,
            "express_surcharge_cents": rng.randint(180, 760),
        })

    if shifted:
        weight = rng.randint(850, 3200)
        service = "express" if rng.random() < 0.66 else "standard"
        latest = rng.choice([1, 2, 2, 3, 4])
        value = rng.choice([40, 60, 80, 200, 260, 320, 400])
        deadline_ms = rng.choice([5200, 6500, 8000, 11000, 18000])
        known_rate_table = rng.random() < 0.14
    else:
        weight = rng.randint(250, 2600)
        service = "express" if rng.random() < 0.33 else "standard"
        latest = rng.choice([2, 3, 4, 5, 6])
        value = rng.randint(60, 360)
        deadline_ms = rng.choice([6500, 9000, 12000, 18000, 26000])
        known_rate_table = rng.random() < 0.30

    req = {"request_id": rid, "lane_id": lane, "weight_kg": weight,
           "latest_days": latest, "service": service}
    visible_rows = rows if known_rate_table else None
    carrier_ids = [r["carrier_id"] for r in rows]
    text = (
        f"Please quote shipment {weight} kg on lane {lane}. It must arrive within "
        f"{latest} days and requires {service} service. Choose the lowest-cost "
        "eligible carrier. If the needed rate or capacity facts are missing, say so "
        "instead of inventing a quote. Return one JSON object with decision, "
        "carrier_id, amount_cents, eta_days, and reason."
    )
    request = {
        **req,
        "split": split,
        "batch_position": index,
        "value_weight": value,
        "service_time_limit_ms": deadline_ms,
        "response_deadline_ms": deadline_ms,
        "carrier_ids": carrier_ids,
        "visible_rate_card": visible_rows,
        "user_text": text,
        "source_visible": known_rate_table,
    }
    truth = oracle_v1(req, rows)
    return request, rows, truth


def build() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "oracle").mkdir(parents=True, exist_ok=True)
    counts = {"dev": 24, "test": 48, "shift": 48}
    seeds = {"dev": 761, "test": 762, "shift": 763}
    summary = {"seed_by_split": seeds, "count_by_split": counts,
               "distribution": {"dev": "baseline", "test": "baseline",
                                "shift": "tight SLA, heavier loads, fewer visible rate cards, changed values"}}
    (DATA / "generation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    all_catalog = []
    totals = {}
    for split, count in counts.items():
        rng = random.Random(seeds[split])
        requests, labels = [], []
        for i in range(count):
            request, catalog_rows, truth = make_case(rng, split, i, split == "shift")
            requests.append(request)
            labels.append(truth)
            all_catalog.extend(catalog_rows)
        _write_jsonl(DATA / f"{split}_requests.jsonl", requests)
        _write_jsonl(DATA / "oracle" / f"{split}_gold.jsonl", labels)
        totals[split] = {"requests": len(requests), "catalog_rows": sum(
            1 for row in all_catalog if row["lane_id"].startswith(f"LANE_{split.upper()}_")),
            "gold_quotes": sum(x["decision"] == "quote" for x in labels),
            "gold_holds": sum(x["decision"] == "hold" for x in labels),
            "visible_rate_cards": sum(x["source_visible"] for x in requests)}
    _write_jsonl(DATA / "rate_catalog.jsonl", all_catalog)
    summary["realized"] = totals
    (DATA / "generation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
