"""CASDAO chat-completions client and safe prompt/response adapters."""

from __future__ import annotations

import json
import os
import re
import time
from urllib.request import (HTTPRedirectHandler, ProxyHandler, Request,
                            build_opener)


ENDPOINT = "https://maasapi.casdao.com/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def messages_for_request(case: dict) -> list[dict]:
    """Build a prompt from observable task fields only; exclude value and gold."""
    visible = {
        "lane_id": case["lane_id"],
        "weight_kg": case["weight_kg"],
        "latest_days": case["latest_days"],
        "service": case["service"],
        "carrier_ids": case["carrier_ids"],
        "rate_card": case["visible_rate_card"],
    }
    return [
        {"role": "system", "content": (
            "You are a warehouse freight quote assistant. Use only the information "
            "in the user message. Do not invent missing rates, capacities, or transit "
            "times. If the best eligible carrier cannot be determined from visible "
            "facts, return decision=\"need_data\" with null carrier_id, amount_cents, "
            "and eta_days. Otherwise choose the lowest-price carrier satisfying the "
            "weight, service, and arrival constraints. Output exactly one JSON object "
            "with keys decision, carrier_id, amount_cents, eta_days, reason."
        )},
        {"role": "user", "content": json.dumps(
            {"request": case["user_text"], "visible_facts": visible},
            ensure_ascii=False, separators=(",", ":")),
        },
    ]


def call_chat(messages: list[dict], timeout_s: int = 90) -> dict:
    key = os.environ.get("CASDAO_API_KEY")
    if not key:
        raise RuntimeError("CASDAO_API_KEY is not set; refusing to call the model API")
    payload = json.dumps({"model": MODEL, "messages": messages,
                          "temperature": 0, "max_tokens": 250,
                          "thinking": {"type": "disabled"},
                          "stream": False}, ensure_ascii=False).encode("utf-8")
    req = Request(ENDPOINT, data=payload, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }, method="POST")
    start = time.perf_counter()
    # The configured direct endpoint must receive the request; neither system
    # proxy settings nor HTTP redirects are allowed to change its destination.
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    with opener.open(req, timeout=timeout_s) as response:
        body = response.read()
        status = response.status
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    obj = json.loads(body.decode("utf-8"))
    choices = obj.get("choices") or []
    if not choices:
        raise RuntimeError(f"API returned no choices (HTTP {status})")
    choice = choices[0]
    content = choice.get("message", {}).get("content", "")
    usage = obj.get("usage") or {}
    return {"requested_model": MODEL, "server_model": obj.get("model"),
            "finish_reason": choice.get("finish_reason"), "content": content, "usage": usage,
            "api_elapsed_ms": elapsed_ms, "http_status": status}


def parse_candidate(content: str) -> dict:
    """Parse a JSON answer without changing its claims or fabricating fields."""
    if isinstance(content, dict):
        obj = content
    else:
        text = str(content).strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return {"decision": "unparsed", "carrier_id": None,
                        "amount_cents": None, "eta_days": None,
                        "reason": "model output was not a JSON object"}
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {"decision": "unparsed", "carrier_id": None,
                        "amount_cents": None, "eta_days": None,
                        "reason": "model output contained malformed JSON"}
    amount = obj.get("amount_cents")
    eta = obj.get("eta_days")
    try:
        amount = int(amount) if amount is not None else None
    except (TypeError, ValueError):
        amount = None
    try:
        eta = int(eta) if eta is not None else None
    except (TypeError, ValueError):
        eta = None
    carrier = obj.get("carrier_id")
    carrier = str(carrier).strip() if carrier not in (None, "", "null") else None
    return {"decision": str(obj.get("decision", "quote")).strip().lower(),
            "carrier_id": carrier, "amount_cents": amount, "eta_days": eta,
            "reason": str(obj.get("reason", ""))[:240]}

