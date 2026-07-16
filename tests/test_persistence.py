from pathlib import Path

from backend.persistence import JsonDecisionStore, normalize_database_url


def test_json_decision_store(tmp_path: Path):
    store = JsonDecisionStore(tmp_path / "decisions.json")
    created = store.append({"decision_status": "approved"})
    assert created["id"]
    records = store.list()
    assert records[0]["decision_status"] == "approved"
    assert store.status()["reachable"] is True


def test_railway_postgres_url_normalization():
    result = normalize_database_url("postgresql://user:pass@host:5432/db")
    assert result == "postgresql+psycopg://user:pass@host:5432/db"
    result_legacy = normalize_database_url("postgres://user:pass@host:5432/db")
    assert result_legacy == "postgresql+psycopg://user:pass@host:5432/db"
