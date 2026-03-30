# MeridianIQ — OpsIQ: Industrial Fleet Intelligence

> Intelligence at the intersection of data and action.

**Live Demo:** https://bodezqmqfhy7oxkvwuusde.streamlit.app

A production-grade multi-agent AI system that analyzes industrial
robot fleet telemetry and predicts equipment failures before they happen.
Built by Meridian AI Consulting to demonstrate agentic AI architecture
for manufacturing operations.

---

## The Problem

Manufacturing operations teams are drowning in data but starved for
intelligence. AGV fleets generate thousands of telemetry events per
shift. Maintenance engineers manually review sensor data to predict
failures. Plant managers piece together status from multiple systems.

OpsIQ replaces that manual work with a 4-agent AI pipeline that
analyzes fleet health, predicts failures, routes alerts, and produces
plain-English operations briefs — automatically.

---

## What It Does

- Analyzes robot fleet telemetry (3,000+ events per shift)
- Predicts equipment failures using NASA CMAPSS sensor data
- Detects deadlocks, congestion hotspots, and battery emergencies
- Routes prioritized alerts to the right personnel
- Produces executive summaries, operations briefs, and chat interface

---

## Architecture

### 4-Agent LangGraph Pipeline
```
Raw Telemetry Data
        ↓
┌─────────────────────────────────────────────────┐
│                  OpsIQ Pipeline                  │
│                                                  │
│  [Telemetry Agent]  →  [Fault Agent]            │
│   Fleet health           RUL estimation          │
│   Deadlock detection     Sensor anomalies        │
│   Congestion analysis    Cascade risks           │
│         ↓                      ↓                │
│  [Alerting Agent]  →  [Reporting Agent]         │
│   Alert lifecycle        Executive summary       │
│   Escalation routing     Operations brief        │
│   Recurring patterns     Chat interface          │
└─────────────────────────────────────────────────┘
        ↓
   Three Output Surfaces:
   • Line Manager Dashboard (live alerts)
   • Plant Manager Report (predictive maintenance)
   • Executive Brief (downloadable report)
```

### State Schema (Pydantic V2)
Every agent reads from and writes to a shared typed state object.
No direct agent-to-agent communication — all coordination through state.

### Tools Layer (Deterministic Python)
8 analysis functions handle all computation. Agents call tools,
Claude reasons over results. Thresholds are transparent and configurable.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph |
| LLM | Claude (Anthropic) |
| State schema | Pydantic V2 |
| UI | Streamlit |
| Data processing | Pandas, NumPy |
| Deployment | Streamlit Cloud |

---

## Data Sources

- **NASA CMAPSS Turbofan Dataset** — 20,631 sensor records, 100 engines
- **Synthetic AGV Telemetry** — 85,800 events, 10 robots, 30-day history

---

## Three Output Surfaces

**Line Manager / Ops Manager**
Real-time alert feed with severity filtering, escalation routing,
and recommended actions. What needs attention right now.

**Plant Manager**
Predictive maintenance view — engine RUL estimates, sensor anomaly
detection, cascade risk analysis, and 30-day fleet health trends.

**Executive**
Plain-English operations brief. Downloadable report.
No technical jargon. Paired actions with every finding.

---

## Project Structure
```
opsiq/
├── agents/
│   ├── state.py           # Pydantic state schema
│   ├── pipeline.py        # LangGraph graph definition
│   ├── telemetry_agent.py # Fleet health analysis
│   ├── fault_agent.py     # CMAPSS fault detection
│   ├── alerting_agent.py  # Alert lifecycle management
│   └── reporting_agent.py # Plain-English report generation
├── tools/
│   └── analysis_tools.py  # 8 deterministic analysis functions
├── data/
│   ├── data_generator.py  # Synthetic AGV telemetry generator
│   ├── data_loader.py     # CMAPSS loader and normalizer
│   ├── raw/               # Source data files
│   └── processed/         # Computed profiles and history
└── app/
    └── streamlit_app.py   # 4-page Streamlit application
```

---

## Built By

**Meridian AI Consulting**
Agentic AI systems for industrial operations.
[MeridianIQ.co](https://meridianiq.co)
