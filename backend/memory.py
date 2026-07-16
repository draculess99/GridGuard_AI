from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonMemoryStore:
    """Small JSON-backed memory store loaded into process memory on each operation.

    It is intentionally lightweight for the MVP. It persists operator/assistant exchanges
    and decision briefings across local restarts, while keeping a bounded history.
    """

    def __init__(self, path: str | Path = "data/runtime/decision_memory.json", max_records: int = 200) -> None:
        self.path = Path(path)
        self.max_records = max(1, int(max_records))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _read(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _write(self, records: list[dict[str, Any]]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(records[-self.max_records :], indent=2, default=str), encoding="utf-8")
        temporary.replace(self.path)

    def append(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        record = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }
        with self.lock:
            records = self._read()
            records.append(record)
            self._write(records)
        return record

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._read()[-max(1, int(limit)) :]

    def conversation_messages(self, limit: int = 8) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for record in self.list(limit=limit):
            role = str(record.get("role", "user"))
            if role not in {"user", "assistant"}:
                continue
            messages.append({"role": role, "content": str(record.get("content", ""))})
        return messages

    def clear(self) -> None:
        with self.lock:
            self._write([])

    def status(self) -> dict[str, Any]:
        records = self._read()
        return {
            "mode": "json_memory",
            "path": str(self.path),
            "records": len(records),
            "max_records": self.max_records,
            "reachable": self.path.parent.exists(),
        }
