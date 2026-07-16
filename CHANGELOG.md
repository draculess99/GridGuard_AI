# Changelog

## 0.3.0 — Three-source data ingestion

- Added a three-way Streamlit data switch: Synthetic Demo, Kaggle Historical, and EIA Live.
- Added a dedicated Data Sources tab showing all three inbound sources and active/readiness status.
- Added a common canonical hourly schema for every data adapter.
- Added Kaggle CSV upload and local-file support.
- Added Kaggle ZIP support with CSV selection.
- Added automatic timestamp, MW demand-series, and optional temperature-column detection.
- Added support for PJM-style multi-region hourly-energy datasets.
- Added source timezone conversion to UTC.
- Added missing-hour interpolation, drop, and error policies.
- Added duplicate, invalid-row, missing-hour, interpolation, and temperature-method profiling.
- Added data-quality status flags to normalized rows.
- Added data source and region provenance to human decision audit records.
- Added `/api/data/sources` and data-source details to `/api/status`.
- Updated README, environment variables, project structure, screenshots, limitations, tests, and roadmap.
- Added Windows `tzdata` dependency for IANA timezone support.
- Added Kaggle data-directory instructions without redistributing third-party datasets.

## 0.2.0 — X-Decision, RAG and multi-provider intelligence

- Added X-Decision hybrid decision system.
- Added deterministic internal expert rules and fired-rule trace.
- Added local TF-IDF RAG over Markdown/text documents.
- Added JSON-backed bounded decision memory.
- Added Grok/xAI, GroqCloud, and Gemini provider adapters.
- Added provider model dropdowns.
- Added prompt, completion, total-token, and request counters.
- Added selected-provider and global token reset controls.
- Added intelligence, memory, and token Flask endpoints.

## 0.1.0 — GridGuard MVP

- Added Streamlit grid-operations control tower.
- Added Flask health and readiness API.
- Added synthetic and EIA hourly demand sources.
- Added XGBoost model with chronological holdout evaluation.
- Added seasonal-naive baseline comparison.
- Added recursive demand forecast.
- Added peak-risk and reserve-margin decision engine.
- Added scenario controls, human approval, JSON persistence, and optional PostgreSQL.
