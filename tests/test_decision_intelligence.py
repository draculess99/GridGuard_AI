from pathlib import Path

from backend.decision_intelligence import run_decision_intelligence
from backend.expert_system import build_expert_decision
from backend.memory import JsonMemoryStore
from backend.rag import LocalRagIndex
from backend.token_meter import TokenMeter


def sample_risk():
    return {
        "level": "ELEVATED",
        "headline": "Reserve margin is thin during the forecast peak.",
        "recommendation": "Activate targeted demand response and confirm fast-start resources.",
        "peak_mw": 25000.0,
        "peak_time": "2026-07-16T18:00:00+00:00",
        "effective_capacity_mw": 26000.0,
        "reserve_margin_pct": 4.0,
        "high_risk_hours": 3,
    }


def test_expert_system_returns_fired_rules():
    decision = build_expert_decision(
        sample_risk(),
        {"mae_improvement_pct": 8.5},
        {"temperature_delta": 12.0, "outage_mw": 1000.0, "demand_shock_pct": 6.0},
    )
    rule_ids = {item["rule"] for item in decision["rules_fired"]}
    assert "R2_THIN_RESERVE" in rule_ids
    assert "R7_GENERATION_OUTAGE" in rule_ids
    assert "R8_DEMAND_SHOCK" in rule_ids
    assert decision["requires_human_approval"] is True


def test_internal_decision_uses_rag_and_zero_tokens(tmp_path: Path):
    docs = tmp_path / "rag"
    docs.mkdir()
    (docs / "policy.md").write_text(
        "Thin reserve margins require targeted demand response and human operator approval.",
        encoding="utf-8",
    )
    rag = LocalRagIndex(docs, chunk_size=300)
    memory = JsonMemoryStore(tmp_path / "memory.json")
    meter = TokenMeter(tmp_path / "tokens.json")
    result = run_decision_intelligence(
        provider="internal_expert_system",
        model="deterministic-rules-v1",
        risk=sample_risk(),
        model_metrics={"mae_improvement_pct": 8.5},
        scenario={"temperature_delta": 0.0, "outage_mw": 0.0, "demand_shock_pct": 0.0},
        operator_question="What should the operator do for a thin reserve margin?",
        rag=rag,
        memory=memory,
        meter=meter,
    )
    assert result["usage"]["total_tokens"] == 0
    assert result["rag_sources"]
    assert "X-Decision briefing" in result["text"]
    assert memory.status()["records"] == 2
