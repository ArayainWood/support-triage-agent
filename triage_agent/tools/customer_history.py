"""Tool: look up a customer's account and recent support history (mocked).

In production this would call a CRM / billing service; here it reads a local JSON
fixture so the agent is fully runnable offline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import DATA_DIR
from .registry import Tool


class CustomerHistoryTool(Tool):
    name = "lookup_customer_history"
    description = (
        "Fetch a customer's account details and recent support history by "
        "customer_id (plan, tenure, region, billing state, prior tickets). Use it "
        "to gauge account value and risk before deciding urgency and routing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "The customer's unique id, e.g. 'cus_free_001'.",
            }
        },
        "required": ["customer_id"],
    }

    def __init__(self, source: Path | None = None):
        path = source or (DATA_DIR / "customers.json")
        self._db: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    def run(self, customer_id: str) -> dict[str, Any]:
        record = self._db.get(customer_id)
        if record is None:
            return {"found": False, "customer_id": customer_id}
        return {"found": True, "customer_id": customer_id, **record}
