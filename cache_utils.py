import hashlib
import json
import os
import tempfile
from typing import Any, Dict, Optional


VOLATILE_KEYS = {"request_id", "timestamp", "ts", "nonce"}


def _deep_sort(obj: Any) -> Any:
    """
    Recursively sort lists and dicts to ensure stable hashing of semantically-equal payloads.
    """
    if isinstance(obj, dict):
        # Remove volatile keys that shouldn't impact business logic
        return {k: _deep_sort(v) for k, v in sorted(obj.items(), key=lambda kv: kv[0]) if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        # Sort list of hashable items; if not hashable, keep order but normalize children
        try:
            return sorted((_deep_sort(v) for v in obj), key=lambda x: json.dumps(x, sort_keys=True))
        except Exception:
            return [_deep_sort(v) for v in obj]
    return obj


def normalized_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return _deep_sort(payload or {})
    except Exception:
        # Fallback: shallow filter only
        return {k: v for k, v in (payload or {}).items() if k not in VOLATILE_KEYS}


def payload_hash(payload: Dict[str, Any]) -> str:
    norm = normalized_payload(payload)
    data = json.dumps(norm, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class LastResultCache:
    """
    Single-entry cache persisted to the OS temp dir.
    Stores the last request's fingerprint and full JSON response.
    """
    def __init__(self, name: str = "schedule_last_result.json") -> None:
        self.path = os.path.join(tempfile.gettempdir(), name)

    def load(self) -> Optional[Dict[str, Any]]:
        try:
            if not os.path.exists(self.path):
                return None
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def save(self, req_hash: str, response: Dict[str, Any]) -> None:
        try:
            blob = {"hash": req_hash, "response": response}
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False)
        except Exception:
            # Swallow cache failures; they're non-critical
            pass
