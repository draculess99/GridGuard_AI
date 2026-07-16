from pathlib import Path


def test_intelligence_and_token_endpoints(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GRIDGUARD_MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setenv("GRIDGUARD_TOKEN_LEDGER_PATH", str(tmp_path / "tokens.json"))
    monkeypatch.setenv("GRIDGUARD_RAG_DOCS_DIR", str(Path(__file__).resolve().parents[1] / "docs" / "rag"))
    monkeypatch.setenv("GRIDGUARD_JSON_PATH", str(tmp_path / "decisions.json"))
    monkeypatch.setenv("GRIDGUARD_PERSISTENCE_MODE", "json")

    from backend import persistence
    persistence._STORE = None
    from backend.api import create_app

    client = create_app().test_client()
    status = client.get("/api/intelligence/status")
    assert status.status_code == 200
    payload = status.get_json()
    assert payload["internal_expert_system"]["configured"] is True
    assert payload["rag"]["ready"] is True

    tokens = client.get("/api/tokens")
    assert tokens.status_code == 200
    assert "groq" in tokens.get_json()["providers"]

    reset = client.delete("/api/tokens?provider=groq")
    assert reset.status_code == 200
    assert reset.get_json()["provider"] == "groq"


def test_data_sources_endpoint(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GRIDGUARD_MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setenv("GRIDGUARD_TOKEN_LEDGER_PATH", str(tmp_path / "tokens.json"))
    monkeypatch.setenv("GRIDGUARD_RAG_DOCS_DIR", str(Path(__file__).resolve().parents[1] / "docs" / "rag"))
    monkeypatch.setenv("GRIDGUARD_JSON_PATH", str(tmp_path / "decisions.json"))
    monkeypatch.setenv("GRIDGUARD_PERSISTENCE_MODE", "json")
    monkeypatch.setenv("GRIDGUARD_KAGGLE_DATA_PATH", str(tmp_path / "not-present.csv"))

    from backend import persistence
    persistence._STORE = None
    from backend.api import create_app

    response = create_app().test_client().get("/api/data/sources")
    assert response.status_code == 200
    payload = response.get_json()
    assert [source["id"] for source in payload["sources"]] == [
        "synthetic",
        "kaggle_historical",
        "eia_live",
    ]
    assert "data_quality_status" in payload["canonical_schema"]
