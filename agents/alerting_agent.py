import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents.state import (
    OpsIQState,
    AlertOutput,
    Alert
)

# ── SYSTEM PROMPT ───────────────────────────────────────────
ALERTING_SYSTEM_PROMPT = """You are the Alerting Agent for OpsIQ, an industrial
fleet intelligence system.

You receive findings from the Telemetry Agent and Fault Agent.
Your job is to:
1. Determine which findings warrant alerts
2. Assess whether each alert is new or a recurring pattern
3. Assign escalation routing based on severity and recurrence
4. Write a concise description for each alert that tells the recipient
   exactly what is wrong and what to do

Be specific. Name the asset. State the severity. Give one clear action.
Do not use markdown. No bullet points. Plain sentences only."""

# ── ESCALATION ROUTING ──────────────────────────────────────
ESCALATION_MAP = {
    "critical_new":       "Maintenance Manager",
    "critical_recurring": "Operations Supervisor",
    "high_new":           "Fleet Engineer",
    "high_recurring":     "Maintenance Manager",
    "moderate_new":       None,
    "moderate_recurring": "Fleet Engineer"
}

def get_escalation(severity: str, is_recurring: bool) -> str | None:
    key = f"{severity.lower()}_{'recurring' if is_recurring else 'new'}"
    return ESCALATION_MAP.get(key)

# ── CHECK ALERT HISTORY ─────────────────────────────────────
def check_alert_history(alert_history: list, asset_id: str, alert_type: str) -> tuple:
    """
    Check if this alert has fired before.
    Returns (is_recurring, recurrence_count)
    """
    matching = [
        a for a in alert_history
        if a.get("robot_id") == asset_id or str(a.get("engine_id")) == str(asset_id)
    ]
    type_matching = [
        a for a in matching
        if alert_type.lower() in a.get("event", "").lower() or
           alert_type.lower() in a.get("severity", "").lower()
    ]
    count = len(type_matching)
    return count > 0, count

# ── MAIN AGENT FUNCTION ─────────────────────────────────────
def run_alerting_agent(state: OpsIQState) -> dict:
    """
    Alerting Agent node for LangGraph.
    Reads telemetry and fault findings from state.
    Produces prioritized alert list with escalation routing.
    """
    print("\n[Alerting Agent] Starting alert processing...")

    # ── READ FROM STATE ─────────────────────────────────────
    telemetry = state.telemetry
    fault = state.fault

    if telemetry is None or fault is None:
        error_msg = "Alerting Agent: missing telemetry or fault output in state"
        print(f"[Alerting Agent] ERROR: {error_msg}")
        return {"errors": state.errors + [error_msg]}

    # ── LOAD ALERT HISTORY ──────────────────────────────────
    alert_history_path = Path("data/processed/alert_history.json")
    with open(alert_history_path) as f:
        alert_history = json.load(f)

    print(f"[Alerting Agent] Loaded {len(alert_history)} historical alerts")

    # ── BUILD ALERT LIST ────────────────────────────────────
    alerts = []
    alert_counter = 1

    # ── ROBOT DEADLOCK ALERTS ───────────────────────────────
    for robot in telemetry.deadlocks.deadlocked_robots:
        robot_id = robot["robot_id"]
        is_recurring, count = check_alert_history(
            alert_history, robot_id, "deadlock"
        )
        severity = "critical"
        escalate_to = get_escalation(severity, is_recurring)

        alerts.append(Alert(
            alert_id=f"ALT-{state.run_id}-{alert_counter:03d}",
            robot_id=robot_id,
            severity=severity,
            alert_type="deadlock",
            description=f"{robot_id} is in active deadlock with {robot.get('deadlock_events', 0)} "
                       f"confirmed events. Robot is non-operational. "
                       f"{'Recurring issue — ' + str(count) + ' previous occurrences.' if is_recurring else 'New event.'} "
                       f"Dispatch technician for manual intervention immediately.",
            is_recurring=is_recurring,
            recurrence_count=count,
            recommended_action="Dispatch technician for manual deadlock resolution",
            escalate_to=escalate_to
        ))
        alert_counter += 1

    # ── BATTERY CRITICAL ALERTS ─────────────────────────────
    for battery_alert in telemetry.battery.battery_alerts:
        if battery_alert["severity"] == "critical":
            robot_id = battery_alert["robot_id"]
            is_recurring, count = check_alert_history(
                alert_history, robot_id, "low_battery"
            )
            severity = "critical"
            escalate_to = get_escalation(severity, is_recurring)

            alerts.append(Alert(
                alert_id=f"ALT-{state.run_id}-{alert_counter:03d}",
                robot_id=robot_id,
                severity=severity,
                alert_type="battery_critical",
                description=f"{robot_id} battery at {battery_alert['current_battery']}% — "
                           f"critical threshold breached. "
                           f"{'Recurring issue — ' + str(count) + ' previous occurrences.' if is_recurring else 'New event.'} "
                           f"Route to nearest charging station immediately.",
                is_recurring=is_recurring,
                recurrence_count=count,
                recommended_action="Route robot to charging station immediately",
                escalate_to=escalate_to
            ))
            alert_counter += 1

    # ── ENGINE RUL CRITICAL ALERTS ──────────────────────────
    for rul in fault.rul_estimates:
        if rul.risk_level in ["critical", "high"]:
            engine_id = rul.engine_id
            is_recurring, count = check_alert_history(
                alert_history, str(engine_id), "critical"
            )
            severity = "critical" if rul.risk_level == "critical" else "high"
            escalate_to = get_escalation(severity, is_recurring)

            alerts.append(Alert(
                alert_id=f"ALT-{state.run_id}-{alert_counter:03d}",
                engine_id=engine_id,
                severity=severity,
                alert_type="rul_critical",
                description=f"Engine {engine_id} has {rul.estimated_rul} cycles remaining "
                           f"at {rul.confidence} confidence. "
                           f"{'Recurring risk — flagged ' + str(count) + ' times previously.' if is_recurring else 'New risk identification.'} "
                           f"{rul.recommendation}",
                is_recurring=is_recurring,
                recurrence_count=count,
                recommended_action=rul.recommendation,
                escalate_to=escalate_to
            ))
            alert_counter += 1

    # ── SENSOR ANOMALY ALERTS ───────────────────────────────
    for anomaly in fault.sensor_anomalies:
        if anomaly.status == "critical":
            engine_id = anomaly.engine_id
            is_recurring, count = check_alert_history(
                alert_history, str(engine_id), "critical"
            )
            severity = "high"
            escalate_to = get_escalation(severity, is_recurring)

            alerts.append(Alert(
                alert_id=f"ALT-{state.run_id}-{alert_counter:03d}",
                engine_id=engine_id,
                severity=severity,
                alert_type="sensor_anomaly",
                description=f"Engine {engine_id} showing {anomaly.anomaly_count} sensor anomalies. "
                           f"Sensor readings have drifted significantly from healthy baseline. "
                           f"{'Recurring pattern — ' + str(count) + ' previous flags.' if is_recurring else 'New anomaly detected.'} "
                           f"Inspect sensor array and schedule diagnostic.",
                is_recurring=is_recurring,
                recurrence_count=count,
                recommended_action="Inspect sensor array and schedule diagnostic",
                escalate_to=escalate_to
            ))
            alert_counter += 1

    # ── CASCADE RISK ALERT ──────────────────────────────────
    if fault.cascade_risks.cascade_risks_detected:
        alerts.append(Alert(
            alert_id=f"ALT-{state.run_id}-{alert_counter:03d}",
            severity="critical",
            alert_type="cascade_risk",
            description=f"{fault.cascade_risks.risk_count} cascade risks active. "
                       f"Current failures are propagating to other assets. "
                       f"Fleet-wide intervention required to prevent total operational collapse.",
            is_recurring=False,
            recurrence_count=0,
            recommended_action="Initiate fleet-wide intervention protocol",
            escalate_to="Operations Supervisor"
        ))
        alert_counter += 1

    # ── SORT BY SEVERITY ────────────────────────────────────
    severity_order = {"critical": 0, "high": 1, "moderate": 2}
    alerts.sort(key=lambda a: severity_order.get(a.severity, 3))

    # ── CALL CLAUDE FOR ALERT SUMMARY ───────────────────────
    print(f"[Alerting Agent] Generated {len(alerts)} alerts")
    print("[Alerting Agent] Calling Claude for notification summary...")

 import anthropic
import os
from dotenv import load_dotenv
load_dotenv(override=False)

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment")
client = anthropic.Anthropic(api_key=api_key)

    alert_context = "\n".join([
        f"Alert {a.alert_id}: [{a.severity.upper()}] {a.alert_type} — "
        f"{'Robot ' + a.robot_id if a.robot_id else 'Engine ' + str(a.engine_id) if a.engine_id else 'Fleet'} — "
        f"Escalate to: {a.escalate_to or 'Log only'} — "
        f"Recurring: {a.is_recurring} ({a.recurrence_count} previous)"
        for a in alerts
    ])

    context = f"""TELEMETRY SUMMARY:
{telemetry.summary}

FAULT SUMMARY:
{fault.summary}

ALERTS GENERATED ({len(alerts)} total):
{alert_context}

Write a 2-3 sentence notification summary for the Reporting Agent.
State total alert count, highest severity, who has been notified,
and the single most urgent action required right now."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=ALERTING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}]
    )

    summary = message.content[0].text
    print("[Alerting Agent] Summary generated")

    # ── BUILD COUNTS ────────────────────────────────────────
    new_alerts = [a for a in alerts if not a.is_recurring]
    recurring_alerts = [a for a in alerts if a.is_recurring]
    escalations = [
        {
            "alert_id": a.alert_id,
            "escalate_to": a.escalate_to,
            "severity": a.severity,
            "asset": a.robot_id if a.robot_id else f"Engine {a.engine_id}" if a.engine_id else "Fleet-wide"
        }
        for a in alerts if a.escalate_to
    ]

    # ── DETERMINE STATUS ────────────────────────────────────
    if any(a.severity == "critical" for a in alerts):
        overall_status = "critical"
    elif any(a.severity == "high" for a in alerts):
        overall_status = "warning"
    else:
        overall_status = "normal"

    # ── BUILD OUTPUT ────────────────────────────────────────
    alert_output = AlertOutput(
        active_alerts=alerts,
        new_alert_count=len(new_alerts),
        recurring_alert_count=len(recurring_alerts),
        escalation_count=len(escalations),
        escalations=escalations,
        summary=summary, 
        status=overall_status
    )

    print(f"[Alerting Agent] Complete — {len(alerts)} alerts, "
          f"{len(escalations)} escalations, status: {overall_status.upper()}")

    return {
        "alerts": alert_output,
        "current_agent": "reporting",
        "completed_agents": state.completed_agents + ["alerting"]
    }


# ── TEST ────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    print("\n" + "="*55)
    print("Alerting Agent — Test Run")
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

    # Run Telemetry Agent
    from agents.telemetry_agent import run_telemetry_agent
    print("Running Telemetry Agent...")
    telemetry_result = run_telemetry_agent(initial_state)
    state_with_telemetry = initial_state.model_copy(update=telemetry_result)

    # Run Fault Agent
    from agents.fault_agent import run_fault_agent
    print("\nRunning Fault Agent...")
    fault_result = run_fault_agent(state_with_telemetry)
    state_with_fault = state_with_telemetry.model_copy(update=fault_result)

    # Run Alerting Agent
    print("\nRunning Alerting Agent...")
    alert_result = run_alerting_agent(state_with_fault)

    # Show output
    alerts = alert_result["alerts"]
    print("\n" + "="*55)
    print("ALERTING AGENT OUTPUT")
    print("="*55)
    print(f"Total alerts:      {len(alerts.active_alerts)}")
    print(f"New alerts:        {alerts.new_alert_count}")
    print(f"Recurring alerts:  {alerts.recurring_alert_count}")
    print(f"Escalations:       {alerts.escalation_count}")
    print(f"Overall status:    {alerts.status.upper()}")
    print(f"\nEscalation routing:")
    for e in alerts.escalations:
        print(f"  [{e['severity'].upper()}] {e['asset']} → {e['escalate_to']}")
    print(f"\nClaude's alert summary:")
    print(alert_result["alerts"].summary)
    print(f"\nCompleted agents: {alert_result['completed_agents']}")