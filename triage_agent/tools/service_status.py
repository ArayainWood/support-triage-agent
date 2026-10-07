"""Tool: check the internal real-time service status (mocked).

Mirrors a real triage need: the public status page says "all operational" while
an internal check reveals a regional incident. The Asia region is intentionally
degraded to match sample ticket #2.
"""
from __future__ import annotations

from typing import Any

from .registry import Tool

# Internal status that deliberately diverges from the public page for Asia.
_STATUS: dict[str, dict[str, Any]] = {
    "asia": {
        "status": "degraded",
        "component": "api-gateway",
        "incident_id": "INC-4821",
        "detail": "Elevated HTTP 5xx in ap-southeast-1 since 13:10 ICT; mitigation in progress.",
        "public_status_page": "operational",
    },
    "europe": {"status": "operational"},
    "us": {"status": "operational"},
    "global": {"status": "operational"},
}


class ServiceStatusTool(Tool):
    name = "check_service_status"
    description = (
        "Check the INTERNAL real-time status of a region (asia, europe, us, "
        "global). More authoritative than the public status page. Use it when a "
        "customer reports an outage or errors (e.g. HTTP 500) to confirm whether "
        "the fault is on our side."
    )
    parameters = {
        "type": "object",
        "properties": {
            "region": {
                "type": "string",
                "description": "Region to check: asia, europe, us, or global.",
            }
        },
        "required": ["region"],
    }

    def run(self, region: str) -> dict[str, Any]:
        key = (region or "").strip().lower()
        result = _STATUS.get(key, {"status": "unknown", "region": region})
        return {"region": key or region, **result}
