"""Explicit, one-pass collection of model first-round outputs.

This script is never invoked by the offline replay command. It requires
CASDAO_API_KEY in the parent process and never writes that value to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_api import call_chat, messages_for_request, parse_candidate


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def collect(splits: list[str], overwrite: bool = False) -> None:
    output_dir = ROOT / "data" / "model_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        source = ROOT / "data" / f"{split}_requests.jsonl"
        target = output_dir / f"{split}_first_round_api.jsonl"
        rows = read_jsonl(source)
        existing = {}
        if target.exists() and not overwrite:
            existing = {row["request_id"]: row for row in read_jsonl(target)}
            unexpected = set(existing) - {case["request_id"] for case in rows}
            if unexpected:
                raise ValueError(f"Existing output has unknown request IDs: {sorted(unexpected)[:5]}")
        mode = "w" if overwrite or not target.exists() else "a"
        with target.open(mode, encoding="utf-8", newline="\n") as f:
            for i, case in enumerate(rows, start=1):
                if case["request_id"] in existing:
                    continue
                result = call_chat(messages_for_request(case))
                record = {
                    "request_id": case["request_id"],
                    "requested_model": result["requested_model"],
                    "server_model": result["server_model"],
                    "finish_reason": result["finish_reason"],
                    "truncated": result["finish_reason"] == "length",
                    "candidate": parse_candidate(result["content"]),
                    "raw_content": result["content"],
                    "usage": result["usage"],
                    "api_elapsed_ms": result["api_elapsed_ms"],
                    "http_status": result["http_status"],
                }
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()
                print(f"{split}: {i}/{len(rows)} request_id={case['request_id']} "
                      f"server_model={result['server_model']} finish={result['finish_reason']} "
                      f"elapsed_ms={result['api_elapsed_ms']}", flush=True)
    print(f"Saved first-round outputs under {output_dir.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True,
                        help="Comma-separated split names, e.g. dev,test,shift")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing output files instead of resuming missing IDs")
    args = parser.parse_args()
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    unknown = sorted(set(splits) - {"dev", "test", "shift"})
    if unknown:
        parser.error(f"unknown split(s): {unknown}")
    collect(splits, args.overwrite)


if __name__ == "__main__":
    main()

