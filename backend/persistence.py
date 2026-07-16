from __future__ import annotations

import json
import os
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Column, DateTime, MetaData, String, Table, create_engine, insert, select, text


def normalize_database_url(url: str) -> str:
    value = (url or "").strip()
    if value.startswith("postgres://"):
        return "postgresql+psycopg://" + value[len("postgres://") :]
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value[len("postgresql://") :]
    return value


class DecisionStore(ABC):
    @abstractmethod
    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def status(self) -> dict[str, Any]:
        raise NotImplementedError


class JsonDecisionStore(DecisionStore):
    def __init__(self, path: str | Path = "data/runtime/decisions.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _read(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        item = dict(record)
        item.setdefault("id", str(uuid.uuid4()))
        item.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        with self.lock:
            records = self._read()
            records.append(item)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
            temporary.replace(self.path)
        return item

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        return list(reversed(self._read()))[:limit]

    def status(self) -> dict[str, Any]:
        return {"mode": "json", "shared": False, "reachable": self.path.parent.exists()}


class PostgreSQLDecisionStore(DecisionStore):
    def __init__(self, database_url: str) -> None:
        normalized = normalize_database_url(database_url)
        if not normalized:
            raise ValueError("DATABASE_URL is required for PostgreSQL persistence.")
        self.engine = create_engine(normalized, pool_pre_ping=True, pool_recycle=1800)
        metadata = MetaData()
        self.decisions = Table(
            "gridguard_decisions",
            metadata,
            Column("id", String(36), primary_key=True),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("payload", JSON, nullable=False),
        )
        metadata.create_all(self.engine)

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        item = dict(record)
        item.setdefault("id", str(uuid.uuid4()))
        raw_created = item.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        created_at = datetime.fromisoformat(str(raw_created).replace("Z", "+00:00"))
        with self.engine.begin() as connection:
            connection.execute(
                insert(self.decisions).values(id=item["id"], created_at=created_at, payload=item)
            )
        return item

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        statement = select(self.decisions.c.payload).order_by(self.decisions.c.created_at.desc()).limit(limit)
        with self.engine.connect() as connection:
            return [dict(row[0]) for row in connection.execute(statement).all()]

    def status(self) -> dict[str, Any]:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            reachable = True
        except Exception:
            reachable = False
        return {"mode": "postgresql", "shared": True, "reachable": reachable}


_STORE: DecisionStore | None = None


def get_decision_store(force_new: bool = False) -> DecisionStore:
    global _STORE
    if _STORE is not None and not force_new:
        return _STORE

    mode = os.getenv("GRIDGUARD_PERSISTENCE_MODE", "json").strip().lower()
    if mode == "postgresql":
        _STORE = PostgreSQLDecisionStore(os.getenv("DATABASE_URL", ""))
    elif mode == "json":
        _STORE = JsonDecisionStore(os.getenv("GRIDGUARD_JSON_PATH", "data/runtime/decisions.json"))
    else:
        raise ValueError("GRIDGUARD_PERSISTENCE_MODE must be 'json' or 'postgresql'.")
    return _STORE
