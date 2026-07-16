from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROVIDERS = ("grok", "groq", "gemini")


class TokenMeter:
    """Tracks tokens reported by provider responses.

    This is an application-local ledger, not the provider's billing system. Resetting it
    clears only GridGuard's counters and does not reset an API quota or billing balance.
    """

    def __init__(self, path: str | Path = "data/runtime/token_usage.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        if not self.path.exists():
            self._write(self._empty())

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "providers": {
                provider: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0}
                for provider in PROVIDERS
            },
        }

    def _read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("providers"), dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
        return self._empty()

    def _write(self, payload: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def add(self, provider: str, prompt_tokens: int, completion_tokens: int, total_tokens: int | None = None) -> dict[str, Any]:
        normalized = provider.strip().lower()
        if normalized not in PROVIDERS:
            raise ValueError(f"Unsupported metered provider: {provider}")
        prompt = max(0, int(prompt_tokens or 0))
        completion = max(0, int(completion_tokens or 0))
        total = max(0, int(total_tokens if total_tokens is not None else prompt + completion))
        with self.lock:
            payload = self._read()
            counters = payload["providers"].setdefault(
                normalized,
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0},
            )
            counters["prompt_tokens"] += prompt
            counters["completion_tokens"] += completion
            counters["total_tokens"] += total
            counters["requests"] += 1
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write(payload)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return self._read()

    def reset(self, provider: str | None = None) -> None:
        with self.lock:
            payload = self._read()
            if provider is None:
                payload = self._empty()
            else:
                normalized = provider.strip().lower()
                if normalized not in PROVIDERS:
                    raise ValueError(f"Unsupported metered provider: {provider}")
                payload["providers"][normalized] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "requests": 0,
                }
                payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write(payload)
