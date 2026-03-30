# OpsIQ Architecture Deep Dive

## Design Philosophy

Three principles guide every architectural decision in OpsIQ:

**1. Deterministic tools, intelligent agents**
All computation lives in analysis_tools.py — transparent, auditable,
configurable Python functions. Agents call tools and reason over results.
Claude never makes the severity determination alone.

**2. State as the single source of truth**
No direct agent-to-agent communication. Every agent reads from and
writes to a shared Pydantic state object. Like a data warehouse where
every ETL job reads from and writes to defined tables.

**3. Fail gracefully, not loudly**
Every agent checks for missing upstream data and writes to state.errors
rather than crashing the pipeline. The Reporting Agent notes incomplete
data rather than silently producing wrong output.

## State Schema

The OpsIQ state is a Pydantic V2 model with four sections:

| Section | Owner | Purpose |
|---------|-------|---------|
| raw_data | Orchestrator | Input telemetry, available to all agents |
| telemetry | Telemetry Agent | Fleet metrics, deadlocks, congestion, battery |
| fault | Fault Agent | RUL estimates, sensor anomalies, cascade risks |
| alerts | Alerting Agent | Active alerts, escalations, recurring patterns |
| report | Reporting Agent | Executive summary, findings, recommendations |

## Agent Responsibilities

### Telemetry Agent
- Calls: get_run_metrics, detect_congestion, detect_deadlocks, battery_analysis
- Writes: TelemetryOutput to state.telemetry
- Claude task: Interpret findings, write summary for Fault Agent

### Fault Agent
- Reads: state.telemetry (deadlocks, congestion for cascade analysis)
- Calls: detect_sensor_anomaly, estimate_rul, identify_cascade_risks
- Writes: FaultOutput to state.fault
- Claude task: Distinguish NOW failures from SOON predictions

### Alerting Agent
- Reads: state.telemetry + state.fault
- Checks: alert_history.json for recurring patterns
- Applies: ESCALATION_MAP business rules
- Writes: AlertOutput to state.alerts
- Claude task: Write actionable notification summary

### Reporting Agent
- Reads: Full state (telemetry + fault + alerts)
- Makes: 4 Claude API calls (exec summary, findings, recommendations, brief)
- Writes: ReportOutput to state.report
- Output: Plain English for non-technical audiences

## Conditional Routing

After Telemetry Agent runs, LangGraph evaluates fleet status:
- CRITICAL or WARNING → full pipeline (Fault → Alerting → Reporting)
- NORMAL → skip Fault Agent, go directly to Reporting

This prevents expensive CMAPSS analysis on healthy fleets.

## Production Deployment Architecture
```
Data Sources (AGV historian, SCADA)
        ↓
Connector Layer (custom per client)
        ↓
OpsIQ Pipeline (containerized)
        ↓
PostgreSQL (results, alert history)
        ↓
Streamlit App (dashboard)
        ↓
Email/Slack (scheduled reports)
```

## Data Warehouse Analogy

For data architects, OpsIQ maps directly to familiar patterns:

| Data Warehouse | OpsIQ Equivalent |
|----------------|-----------------|
| Source systems | AGV telemetry + CMAPSS |
| ETL pipeline | 4-agent LangGraph pipeline |
| Shared schema | OpsIQState Pydantic model |
| Staging tables | Tool function outputs |
| Curated layer | Validated Pydantic models |
| Business rules | ESCALATION_MAP, thresholds |
| Report layer | ReportOutput, Streamlit app |
