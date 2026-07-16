from pathlib import Path

from backend.memory import JsonMemoryStore
from backend.rag import LocalRagIndex
from backend.token_meter import TokenMeter


def test_json_memory_is_bounded_and_clearable(tmp_path: Path):
    memory = JsonMemoryStore(tmp_path / "memory.json", max_records=3)
    for index in range(5):
        memory.append("user", f"message {index}")
    records = memory.list(limit=10)
    assert len(records) == 3
    assert records[-1]["content"] == "message 4"
    memory.clear()
    assert memory.list() == []


def test_local_rag_retrieves_matching_policy(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "grid.md").write_text(
        "A negative reserve margin requires critical escalation and human approval.", encoding="utf-8"
    )
    (docs / "model.md").write_text(
        "XGBoost must be compared with a seasonal naive baseline.", encoding="utf-8"
    )
    rag = LocalRagIndex(docs, chunk_size=300)
    hits = rag.retrieve("What happens with a negative reserve margin?", top_k=2)
    assert hits
    assert hits[0].source == "grid.md"
    assert "negative reserve margin" in hits[0].text.lower()


def test_token_meter_add_and_reset(tmp_path: Path):
    meter = TokenMeter(tmp_path / "tokens.json")
    meter.add("groq", 100, 30, 130)
    meter.add("groq", 20, 10, 30)
    counters = meter.snapshot()["providers"]["groq"]
    assert counters == {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160, "requests": 2}
    meter.reset("groq")
    assert meter.snapshot()["providers"]["groq"]["total_tokens"] == 0
