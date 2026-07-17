from __future__ import annotations

from typing import Any

from backend.expert_system import build_expert_decision, render_expert_brief
from backend.llm_providers import ProviderResponse, generate, configured
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

    if normalized == "debate_committee":
        available = [p for p in ["groq", "gemini"] if configured(p)]
        if not available:
            raise ValueError("Debate Committee requires at least one configured LLM provider.")
        
        # Use the user's selected primary provider for all agents in the committee
        primary_prov = "gemini" if "gemini" in model.lower() else "groq"
        
        analyst_prov = primary_prov
        compliance_prov = primary_prov
        chief_prov = primary_prov
        
        analyst_model = model
        compliance_model = model
        chief_model = model

        transcript = []
        total_prompt = 0
        total_comp = 0

        # Agent 1: Analyst
        analyst_system = "You are the Quantitative Analyst Agent. Focus strictly on the statistical risk, capacity limits, and forecast confidence. Keep it very concise (3-4 sentences). Do not make final operational decisions."
        resp1 = generate(analyst_prov, analyst_model, [{"role": "system", "content": analyst_system}, {"role": "user", "content": facts}], meter, max_completion_tokens)
        transcript.append({"role": "Quantitative Analyst", "provider": analyst_prov.upper(), "content": resp1.text})
        total_prompt += resp1.prompt_tokens; total_comp += resp1.completion_tokens

        # Agent 2: Compliance
        compliance_system = "You are the Regulatory Compliance Agent. Review the Analyst's assessment and the retrieved policy context. Point out any regulatory violations or required procedures. Keep it very concise (3-4 sentences)."
        comp_prompt = facts + f"\n\nAnalyst Assessment:\n{resp1.text}"
        resp2 = generate(compliance_prov, compliance_model, [{"role": "system", "content": compliance_system}, {"role": "user", "content": comp_prompt}], meter, max_completion_tokens)
        transcript.append({"role": "Compliance Officer", "provider": compliance_prov.upper(), "content": resp2.text})
        total_prompt += resp2.prompt_tokens; total_comp += resp2.completion_tokens

        # Agent 3: Chief Dispatcher
        chief_system = "You are the Chief Dispatcher. Synthesize the Analyst's data and the Compliance Officer's rules into a final, human-readable operational briefing and recommendation. Follow the format: Situation, Evidence, Recommended Action."
        chief_prompt = facts + f"\n\nAnalyst Assessment:\n{resp1.text}\n\nCompliance Officer Assessment:\n{resp2.text}"
        resp3 = generate(chief_prov, chief_model, [{"role": "system", "content": chief_system}, {"role": "user", "content": chief_prompt}], meter, max_completion_tokens)
        transcript.append({"role": "Chief Dispatcher", "provider": chief_prov.upper(), "content": resp3.text})
        total_prompt += resp3.prompt_tokens; total_comp += resp3.completion_tokens

        final_text = f"### Debate Committee Final Decision\n\n{resp3.text}"
        memory.append("assistant", final_text, {"provider": "debate_committee", "rag_sources": sources, "tokens": total_prompt + total_comp})
        
        return {
            "provider": "debate_committee",
            "model": chief_model,
            "text": final_text,
            "expert": expert,
            "rag_sources": sources,
            "committee_transcript": transcript,
            "usage": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_comp,
                "total_tokens": total_prompt + total_comp,
            },
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
