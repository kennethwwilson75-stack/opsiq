import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents.state import (
    OpsIQState,
    ReportOutput
)

# ── SYSTEM PROMPT ───────────────────────────────────────────
REPORTING_SYSTEM_PROMPT = """You are the Reporting Agent for OpsIQ, an industrial
fleet intelligence system.

You are writing for plant managers, operations supervisors, and executives
who are not technical and do not have time to read lengthy reports.
They need to know three things immediately:
1. Is everything OK or is there a problem?
2. If there is a problem, how bad is it?
3. What should they do about it right now?

Your writing rules:
- Lead with the most critical finding — never bury it
- Use plain business language — no sensor names, no technical jargon
- Translate technical findings into business impact
  - Not "RUL=4 cycles" — say "Engine 68 is expected to fail within 4 shifts"
  - Not "deadlock_detected" — say "Robot R4 has stopped and is blocking the floor"
  - Not "cascade risk" — say "if R4 is not cleared, three other robots will also stop"
- Every finding must have a recommended action attached
- Be concise — executives read the first two sentences and decide if they care
- Calibrate urgency honestly — not everything is a five-alarm fire
- Never use markdown, bullet symbols, or formatting characters
- Write in short paragraphs, not lists"""

def run_reporting_agent(state: OpsIQState) -> dict:
    """
    Reporting Agent node for LangGraph.
    Reads full pipeline state, produces three output formats:
    executive summary, critical findings, and full operations brief.
    """
    print("\n[Reporting Agent] Starting report generation...")

    # ── READ FROM STATE ─────────────────────────────────────
    telemetry = state.telemetry
    fault = state.fault
    alerts = state.alerts

    if not all([telemetry, fault, alerts]):
        missing = []
        if not telemetry: missing.append("telemetry")
        if not fault: missing.append("fault")
        if not alerts: missing.append("alerts")
        error_msg = f"Reporting Agent: missing {', '.join(missing)} in state"
        print(f"[Reporting Agent] ERROR: {error_msg}")
        return {"errors": state.errors + [error_msg]}

    print(f"[Reporting Agent] Processing {alerts.escalation_count} escalations")
    print(f"[Reporting Agent] Pipeline errors: {len(state.errors)}")

    import anthropic
    from dotenv import load_dotenv
    load_dotenv()

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ── BUILD FULL CONTEXT ──────────────────────────────────
    # Compile everything the Reporting Agent needs
    critical_alerts = [
        a for a in alerts.active_alerts
        if a.severity == "critical"
    ]
    high_alerts = [
        a for a in alerts.active_alerts
        if a.severity == "high"
    ]

    escalation_summary = "\n".join([
        f"  {e['severity'].upper()} — {e['asset']} — notify {e['escalate_to']}"
        for e in alerts.escalations[:10]
    ])

    rul_summary = "\n".join([
        f"  Engine {r.engine_id}: {r.estimated_rul} shifts remaining "
        f"({r.risk_level.upper()}) — {r.recommendation}"
        for r in fault.rul_estimates
    ])

    full_context = f"""
PIPELINE RUN: {state.run_id}
TIMESTAMP: {state.analysis_timestamp}
DATA SOURCE: {state.data_source.upper()}

FLEET STATUS:
- Overall status: {telemetry.status.upper()}
- Fleet availability: {telemetry.fleet_metrics.fleet_availability:.0%}
- Active robots: {telemetry.fleet_metrics.active_robots} of {telemetry.fleet_metrics.total_robots}
- Average health score: {telemetry.fleet_metrics.avg_health_score}
- Deadlocked robots: {[r['robot_id'] for r in telemetry.deadlocks.deadlocked_robots]}
- Congestion hotspots: {telemetry.congestion.hotspot_count}
- Fleet avg battery: {telemetry.battery.fleet_avg_battery}%

FAULT ANALYSIS:
- Critical engines: {fault.critical_engines}
- Engines analyzed: {len(fault.rul_estimates)}
- Sensor anomaly engines: {len(fault.sensor_anomalies)}
- Cascade risks active: {fault.cascade_risks.risk_count}

ENGINE HEALTH (predicted remaining life):
{rul_summary}

ALERTS:
- Total alerts: {len(alerts.active_alerts)}
- Critical: {len(critical_alerts)}
- High: {len(high_alerts)}
- New alerts: {alerts.new_alert_count}
- Recurring alerts: {alerts.recurring_alert_count}

ESCALATIONS:
{escalation_summary}

AGENT SUMMARIES:
Telemetry: {telemetry.summary}
Fault: {fault.summary}
Alerts: {alerts.summary}

PIPELINE ERRORS: {state.errors if state.errors else 'None'}
"""

    # ── GENERATE EXECUTIVE SUMMARY ──────────────────────────
    print("[Reporting Agent] Generating executive summary...")

    exec_message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=REPORTING_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"{full_context}\n\nWrite a 2-3 sentence executive summary. "
                      f"This goes at the top of the report. "
                      f"A plant manager reads this and knows immediately whether "
                      f"they need to act right now or can wait until their next meeting."
        }]
    )
    executive_summary = exec_message.content[0].text

    # ── GENERATE CRITICAL FINDINGS ──────────────────────────
    print("[Reporting Agent] Generating critical findings...")

    findings_message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=REPORTING_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"{full_context}\n\nList the top 3-5 critical findings. "
                      f"Each finding is one sentence stating the problem "
                      f"and one sentence stating the required action. "
                      f"Order by urgency. Plain sentences, no bullet symbols."
        }]
    )
    findings_text = findings_message.content[0].text
    # Split into list by newlines
    critical_findings = [
        f.strip() for f in findings_text.split("\n")
        if f.strip() and len(f.strip()) > 20
    ]

    # ── GENERATE RECOMMENDATIONS ────────────────────────────
    print("[Reporting Agent] Generating recommendations...")

    rec_message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=REPORTING_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"{full_context}\n\nProvide 3-5 prioritized recommendations. "
                      f"Each recommendation is one action sentence. "
                      f"Start with the most urgent. "
                      f"Be specific — name the asset and the action. "
                      f"Plain sentences, no bullet symbols, no numbering."
        }]
    )
    rec_text = rec_message.content[0].text
    recommendations = [
        r.strip() for r in rec_text.split("\n")
        if r.strip() and len(r.strip()) > 20
    ]

    # ── GENERATE FULL REPORT ────────────────────────────────
    print("[Reporting Agent] Generating full operations brief...")

    report_message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system=REPORTING_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"{full_context}\n\nWrite a full operations brief. "
                      f"Structure it as: "
                      f"(1) Situation overview — what is the current state of the fleet "
                      f"(2) Active problems — what needs attention right now "
                      f"(3) Predicted problems — what will need attention soon "
                      f"(4) Actions taken — what has been escalated and to whom "
                      f"(5) Recommended next steps — what the manager should do "
                      f"Write in short paragraphs. Plain business language. "
                      f"No markdown. No bullet symbols. No technical jargon."
        }]
    )
    full_report = report_message.content[0].text

    # ── BUILD SEVERITY SUMMARY ──────────────────────────────
    severity_summary = {
        "critical": len(critical_alerts),
        "high": len(high_alerts),
        "moderate": len([a for a in alerts.active_alerts if a.severity == "moderate"]),
        "total": len(alerts.active_alerts)
    }

    # ── DETERMINE STATUS ────────────────────────────────────
    if severity_summary["critical"] > 0:
        overall_status = "critical"
    elif severity_summary["high"] > 0:
        overall_status = "warning"
    else:
        overall_status = "normal"

    # ── BUILD OUTPUT ────────────────────────────────────────
    report_output = ReportOutput(
        executive_summary=executive_summary,
        critical_findings=critical_findings,
        recommendations=recommendations,
        severity_summary=severity_summary,
        full_report=full_report,
        status=overall_status
    )

    print(f"[Reporting Agent] Complete — status: {overall_status.upper()}")

    return {
        "report": report_output,
        "current_agent": "complete",
        "completed_agents": state.completed_agents + ["reporting"]
    }


# ── TEST ────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    print("\n" + "="*55)
    print("Reporting Agent — Test Run")
    print("="*55)

    # Load data
    df = pd.read_csv("data/raw/robot_runs_current.csv")
    raw_data = df.to_dict(orient="records")

    # Build initial state
    initial_state = OpsIQState(
        raw_data=raw_data,
        data_source="agv",
        analysis_timestamp=datetime.now().isoformat(),
        run_id=str(uuid.uuid4())[:8]
    )

    # Run full pipeline
    from agents.telemetry_agent import run_telemetry_agent
    from agents.fault_agent import run_fault_agent
    from agents.alerting_agent import run_alerting_agent

    print("Running Telemetry Agent...")
    telemetry_result = run_telemetry_agent(initial_state)
    state_1 = initial_state.model_copy(update=telemetry_result)

    print("\nRunning Fault Agent...")
    fault_result = run_fault_agent(state_1)
    state_2 = state_1.model_copy(update=fault_result)

    print("\nRunning Alerting Agent...")
    alert_result = run_alerting_agent(state_2)
    state_3 = state_2.model_copy(update=alert_result)

    print("\nRunning Reporting Agent...")
    report_result = run_reporting_agent(state_3)

    # Show output
    report = report_result["report"]
    print("\n" + "="*55)
    print("FINAL OPERATIONS REPORT")
    print("="*55)
    print(f"\nSeverity summary: {report.severity_summary}")
    print(f"Status: {report.status.upper()}")
    print(f"\nEXECUTIVE SUMMARY:")
    print(report.executive_summary)
    print(f"\nCRITICAL FINDINGS:")
    for i, finding in enumerate(report.critical_findings, 1):
        print(f"  {i}. {finding}")
    print(f"\nRECOMMENDATIONS:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")
    print(f"\nFULL OPERATIONS BRIEF:")
    print(report.full_report)
    print(f"\nCompleted agents: {report_result['completed_agents']}")