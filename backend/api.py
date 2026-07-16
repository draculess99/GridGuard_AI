from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request

from backend.data import configured_kaggle_path, data_source_catalog
from backend.llm_providers import PROVIDER_MODELS, configured
from backend.memory import JsonMemoryStore
from backend.persistence import get_decision_store
from backend.rag import LocalRagIndex
from backend.token_meter import TokenMeter


def create_app() -> Flask:
    app = Flask(__name__)
    memory = JsonMemoryStore(
        os.getenv("GRIDGUARD_MEMORY_PATH", "data/runtime/decision_memory.json"),
        max_records=int(os.getenv("GRIDGUARD_MEMORY_MAX_RECORDS", "200")),
    )
    meter = TokenMeter(os.getenv("GRIDGUARD_TOKEN_LEDGER_PATH", "data/runtime/token_usage.json"))
    rag = LocalRagIndex(os.getenv("GRIDGUARD_RAG_DOCS_DIR", "docs/rag"))

    @app.get("/")
    def root():
        return jsonify({"service": "GridGuard AI API", "status": "ok"})

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "healthy",
                "service": "gridguard-api",
                "time": datetime.now(timezone.utc).isoformat(),
            }
        )

    @app.get("/ready")
    def ready():
        store = get_decision_store()
        persistence = store.status()
        rag_status = rag.status()
        ready_state = bool(persistence["reachable"] and rag_status["ready"])
        return (
            jsonify(
                {
                    "status": "ready" if ready_state else "degraded",
                    "persistence": persistence,
                    "rag": rag_status,
                    "memory": memory.status(),
                }
            ),
            200 if ready_state else 503,
        )

    @app.get("/api/status")
    def api_status():
        return jsonify(
            {
                "application": "GridGuard AI",
                "stage": "MVP-3 three-source decision intelligence",
                "forecast_model": "XGBoost",
                "decision_system": "X-Decision hybrid expert system + RAG + optional LLM",
                "persistence": get_decision_store().status(),
                "memory": memory.status(),
                "rag": rag.status(),
                "data_sources": data_source_catalog(),
            }
        )


    @app.get("/api/data/sources")
    def data_sources():
        configured_path = configured_kaggle_path()
        return jsonify(
            {
                "sources": data_source_catalog(kaggle_available=Path(configured_path).exists()),
                "canonical_schema": [
                    "timestamp",
                    "demand_mw",
                    "temperature_f",
                    "region",
                    "is_holiday",
                    "source",
                    "data_quality_status",
                ],
                "kaggle_configured_path": configured_path,
            }
        )

    @app.get("/api/intelligence/status")
    def intelligence_status():
        return jsonify(
            {
                "internal_expert_system": {"configured": True, "tokens": False, "model": "deterministic-rules-v1"},
                "grok": {"configured": configured("grok"), "models": PROVIDER_MODELS["grok"]},
                "groq": {"configured": configured("groq"), "models": PROVIDER_MODELS["groq"]},
                "gemini": {"configured": configured("gemini"), "models": PROVIDER_MODELS["gemini"]},
                "rag": rag.status(),
                "memory": memory.status(),
            }
        )

    @app.get("/api/tokens")
    def token_status():
        return jsonify(meter.snapshot())

    @app.delete("/api/tokens")
    def reset_tokens():
        provider = request.args.get("provider")
        meter.reset(provider=provider)
        return jsonify({"status": "reset", "provider": provider or "all", "usage": meter.snapshot()})

    @app.get("/api/memory")
    def list_memory():
        limit = min(max(int(request.args.get("limit", 20)), 1), 200)
        return jsonify({"records": memory.list(limit=limit)})

    @app.delete("/api/memory")
    def clear_memory():
        memory.clear()
        return jsonify({"status": "cleared"})

    @app.get("/api/decisions")
    def list_decisions():
        limit = min(max(int(request.args.get("limit", 100)), 1), 500)
        return jsonify({"records": get_decision_store().list(limit=limit)})

    @app.post("/api/decisions")
    def create_decision():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "A JSON object is required."}), 400
        return jsonify(get_decision_store().append(payload)), 201

    return app
