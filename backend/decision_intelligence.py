from __future__ import annotations

from typing import Any

from backend.expert_system import build_expert_decision, render_expert_brief
from backend.llm_providers import ProviderResponse, generate
from backend.memory import JsonMemoryStore
from backend.rag import LocalRagIndex
from backend.token_meter import TokenMeter

SYSTEM_PROMPT = """You are the advisory explanation layer for GridGuard AI, an electricity-grid decision-support demonstration.
Use only the supplied forecast facts, expert-system rules, scenario assumptions, and retrieved policy context.
Do not invent grid conditions or claim that any control action was executed. Preserve human approval.
Return a concise operational briefing with: situation, evidence, recommended action, uncertainties, and human checkpoint.
Cite retrieved context using its bracketed chunk identifiers when it is relevant."""


def build_decision_context(
    risk: dict[str, Any],
    model_metrics: dict[str, float],
    scenario: dict[str, float],
    operator_question: str,
    rag: LocalRagIndex,
) -> tuple[dict[str, Any], str, list[str]]:
    expert = build_expert_decision(risk, model_metrics, scenario)
    query = (
        f"{operator_question} Risk {risk['level']}; reserve margin {risk['reserve_margin_pct']:.1f}%; "
        f"high-risk hours {risk['high_risk_hours']}; outage {scenario.get('outage_mw', 0)} MW; "
        f"demand shock {scenario.get('demand_shock_pct', 0)} percent; demand response escalation human approval"
    )
    rag_context, hits = rag.context(query, top_k=4)
    sources = [hit.chunk_id for hit in hits]
    facts = f"""Operator question: {operator_question}

Forecast/risk facts:
- Risk level: {risk['level']}
- Peak demand: {risk['peak_mw']:.0f} MW at {risk['peak_time']}
- Effective capacity: {risk['effective_capacity_mw']:.0f} MW
- Reserve margin: {risk['reserve_margin_pct']:.1f}%
- High-risk hours: {risk['high_risk_hours']}
- Model MAE improvement versus seasonal naive: {model_metrics.get('mae_improvement_pct', 0):.1f}%
- Scenario: {scenario}

Internal expert-system result:
{render_expert_brief(expert, sources)}

Retrieved local policy context:
{rag_context or 'No local RAG chunks matched.'}
"""
    return expert, facts, sources


def run_decision_intelligence(
    provider: str,
    model: str,
    risk: dict[str, Any],
    model_metrics: dict[str, float],
    scenario: dict[str, float],
    operator_question: str,
    rag: LocalRagIndex,
    memory: JsonMemoryStore,
    meter: TokenMeter,
    max_completion_tokens: int = 700,
) -> dict[str, Any]:
    expert, facts, sources = build_decision_context(
        risk=risk,
        model_metrics=model_metrics,
        scenario=scenario,
        operator_question=operator_question,
        rag=rag,
    )
    normalized = provider.strip().lower()
    memory.append("user", operator_question, {"provider": normalized, "rag_sources": sources})

    if normalized == "internal_expert_system":
        text = render_expert_brief(expert, sources)
        memory.append("assistant", text, {"provider": normalized, "tokens": 0})
        return {
            "provider": normalized,
            "model": "deterministic-rules-v1",
            "text": text,
            "expert": expert,
            "rag_sources": sources,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(memory.conversation_messages(limit=6)[:-1])
    messages.append({"role": "user", "content": facts})
    response: ProviderResponse = generate(
        provider=normalized,
        model=model,
        messages=messages,
        meter=meter,
        max_completion_tokens=max_completion_tokens,
    )
    memory.append(
        "assistant",
        response.text,
        {
            "provider": response.provider,
            "model": response.model,
            "rag_sources": sources,
            "tokens": response.total_tokens,
        },
    )
    return {
        "provider": response.provider,
        "model": response.model,
        "text": response.text,
        "expert": expert,
        "rag_sources": sources,
        "usage": {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
        },
    }
