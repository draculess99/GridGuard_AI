# GridGuard AI: Feature Roadmap

This document outlines the current state of features in the GridGuard AI MVP and the strategic roadmap for future development.

## ✅ Completed Features (MVP Phase)

- **[x] Multi-Agent Debate Committee**
  - Integrated a multi-agent decision intelligence pipeline using Groq and Gemini.
  - Implemented three distinct agent personas: Quantitative Analyst, Compliance Officer, and Chief Dispatcher.
  - Added a "Committee Transcript" audit tab to the UI for full transparency.
- **[x] XGBoost Forecasting Pipeline**
  - Autoregressive, rolling-statistic, and calendar-based feature engineering.
  - 12-48 hour recursive load forecasting.
- **[x] Expert Risk Engine**
  - Deterministic rules engine to evaluate operational capacity pressure and demand shocks.
- **[x] Local TF-IDF RAG**
  - In-memory policy retrieval without needing an external vector database.
- **[x] Operator Control Tower UI**
  - Interactive Streamlit dashboard with "Human-in-the-Loop" approval gates.

---

## 🚀 Planned Features (V2 & Beyond)

### 1. Data Integration & Streaming
- **[ ] Live Grid Telemetry Webhooks:** Transition from polling EIA APIs to processing real-time websocket streams.
- **[ ] Advanced Weather API Integration:** Automatically pull live localized weather forecasts (temperature, humidity, wind) into the feature pipeline.

### 2. Advanced Scenario Analytics
- **[ ] Automated Stress Tests:** 1-click simulations for extreme edge cases (e.g., Polar Vortex, sudden generator trip, severe heatwave).
- **[ ] Interactive SHAP Visualizations:** Allow operators to dynamically filter and drill down into the SHAP waterfall charts to see localized feature impacts.

### 3. Enterprise Governance & Security
- **[ ] Role-Based Access Control (RBAC):** Differentiate UI views and approval privileges between 'Dispatch Operators', 'Compliance Officers', and 'System Admins'.
- **[ ] PostgreSQL Audit Ledger:** Move from the local JSON ledger to a secure, remote relational database with cryptographic hashing of approval records.

### 4. Continuous Learning
- **[ ] Automated Model Retraining (MLOps):** Build a pipeline that periodically pulls recent accurate data from the audit ledger to fine-tune the XGBoost forecaster automatically.
- **[ ] Agentic Policy Updates:** Allow the Compliance Agent to proactively suggest updates to the Local RAG policy corpus based on past successful (or overridden) decisions.
