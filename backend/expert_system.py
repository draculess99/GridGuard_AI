from __future__ import annotations

from typing import Any


def build_expert_decision(risk: dict[str, Any], model_metrics: dict[str, float], scenario: dict[str, float]) -> dict[str, Any]:
    """Explainable rule engine that remains available without an LLM."""
    fired: list[dict[str, str]] = []
    level = str(risk["level"]).upper()
    reserve = float(risk["reserve_margin_pct"])
    high_risk_hours = int(risk["high_risk_hours"])
    outage = float(scenario.get("outage_mw", 0.0))
    shock = float(scenario.get("demand_shock_pct", 0.0))
    temperature_delta = float(scenario.get("temperature_delta", 0.0))
    improvement = float(model_metrics.get("mae_improvement_pct", 0.0))

    if reserve < 0:
        fired.append({"rule": "R1_CAPACITY_DEFICIT", "evidence": f"Reserve margin is {reserve:.1f}%.", "effect": "Critical escalation required."})
    elif reserve < 5:
        fired.append({"rule": "R2_THIN_RESERVE", "evidence": f"Reserve margin is {reserve:.1f}%.", "effect": "Targeted demand response should be prepared."})
    elif reserve < 12:
        fired.append({"rule": "R3_WATCH_MARGIN", "evidence": f"Reserve margin is {reserve:.1f}%.", "effect": "Enhanced monitoring is required."})
    else:
        fired.append({"rule": "R4_NORMAL_MARGIN", "evidence": f"Reserve margin is {reserve:.1f}%.", "effect": "Routine monitoring is sufficient."})

    if high_risk_hours >= 4:
        fired.append({"rule": "R5_SUSTAINED_STRESS", "evidence": f"{high_risk_hours} hours exceed the high-risk utilization threshold.", "effect": "Escalation timing should cover the full stress window."})
    elif high_risk_hours > 0:
        fired.append({"rule": "R6_PEAK_WINDOW", "evidence": f"{high_risk_hours} high-risk hour(s) detected.", "effect": "Schedule operator reviews around the peak window."})

    if outage > 0:
        fired.append({"rule": "R7_GENERATION_OUTAGE", "evidence": f"Scenario removes {outage:,.0f} MW of capacity.", "effect": "Verify contingency and transfer resources."})
    if shock >= 5:
        fired.append({"rule": "R8_DEMAND_SHOCK", "evidence": f"Demand shock is {shock:.1f}%.", "effect": "Shorten forecast refresh interval."})
    if abs(temperature_delta) >= 10:
        fired.append({"rule": "R9_EXTREME_TEMPERATURE", "evidence": f"Temperature adjustment is {temperature_delta:+.1f}°F.", "effect": "Treat weather-sensitive demand as elevated uncertainty."})
    if improvement <= 0:
        fired.append({"rule": "R10_MODEL_GUARDRAIL", "evidence": f"XGBoost MAE improvement is {improvement:.1f}% versus seasonal naive.", "effect": "Do not rely on XGBoost alone; retain baseline comparison."})

    confidence = 0.86
    if improvement <= 0:
        confidence -= 0.16
    if abs(temperature_delta) >= 10:
        confidence -= 0.06
    if shock >= 10:
        confidence -= 0.05
    confidence = max(0.45, min(confidence, 0.95))

    action = str(risk["recommendation"])
    return {
        "engine": "internal_expert_system",
        "level": level,
        "headline": str(risk["headline"]),
        "recommended_action": action,
        "confidence": confidence,
        "rules_fired": fired,
        "requires_human_approval": True,
        "guardrail": "The expert system and any LLM are advisory; no control action is executed automatically.",
    }


def render_expert_brief(decision: dict[str, Any], rag_sources: list[str] | None = None) -> str:
    rules = "\n".join(
        f"- {item['rule']}: {item['evidence']} {item['effect']}" for item in decision.get("rules_fired", [])
    )
    sources = ", ".join(rag_sources or []) or "No RAG sources retrieved"
    return (
        f"### X-Decision briefing\n"
        f"**Risk:** {decision['level']}  \n"
        f"**Confidence:** {decision['confidence']:.0%}  \n"
        f"**Recommendation:** {decision['recommended_action']}\n\n"
        f"**Rules fired**\n{rules}\n\n"
        f"**RAG sources:** {sources}\n\n"
        f"**Human control:** {decision['guardrail']}"
    )
