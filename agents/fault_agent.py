import pandas as pd
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tools.analysis_tools import (
    detect_sensor_anomaly,
    estimate_rul,
    identify_cascade_risks,
    load_support_files
)
from agents.state import (
    OpsIQState,
    FaultOutput,
    SensorAnomaly,
    RULEstimate,
    CascadeRisk
)

# ── SYSTEM PROMPT ───────────────────────────────────────────
FAULT_SYSTEM_PROMPT = """You are the Fault Agent for OpsIQ, an industrial 
fleet intelligence system.

You receive findings from the Telemetry Agent and sensor analysis results.
Your job is to:
1. Assess the severity of detected faults and anomalies
2. Identify which assets are at highest risk of failure
3. Confirm or escalate cascade risks identified by the Telemetry Agent
4. Write a concise 2-3 sentence summary for the Alerting Agent

Focus on actionable intelligence. Name specific assets. State urgency clearly.
Distinguish between what is failing NOW versus what is predicted to fail SOON.
Do not use markdown. Be direct."""

def run_fault_agent(state: OpsIQState) -> dict:
    """
    Fault Agent node for LangGraph.
    Reads telemetry findings from state, runs CMAPSS analysis,
    writes FaultOutput back to state.
    """
    print("\n[Fault Agent] Starting fault analysis...")

    # ── READ FROM STATE ─────────────────────────────────────
    df = pd.DataFrame(state.raw_data)
    telemetry = state.telemetry

    if telemetry is None:
        error_msg = "Fault Agent: no telemetry output found in state"
        print(f"[Fault Agent] ERROR: {error_msg}")
        return {"errors": state.errors + [error_msg]}

    print(f"[Fault Agent] Telemetry status: {telemetry.status.upper()}")
    print(f"[Fault Agent] Deadlocked robots: {[r['robot_id'] for r in telemetry.deadlocks.deadlocked_robots]}")

    # ── LOAD CMAPSS DATA ────────────────────────────────────
    print("[Fault Agent] Loading CMAPSS sensor data...")
    cmapss_df = pd.read_csv("data/processed/cmapss_normalized.csv")

    # ── IDENTIFY AT-RISK ENGINES ────────────────────────────
    # Focus on engines in the RUL warning window (10-100 cycles remaining)
    engine_profiles_path = Path("data/processed/engine_profiles.json")
    import json
    with open(engine_profiles_path) as f:
        engine_profiles = json.load(f)

    # Find engines approaching failure
    at_risk_engines = [
        int(eid) for eid, profile in engine_profiles.items()
        if 0 < profile["current_rul"] <= 100
    ]

    # If too many just take top 5 most critical
    at_risk_engines = sorted(
        at_risk_engines,
        key=lambda e: engine_profiles[str(e)]["current_rul"]
    )[:5]

    print(f"[Fault Agent] At-risk engines identified: {at_risk_engines}")

    # ── RUN SENSOR ANOMALY DETECTION ────────────────────────
    print("[Fault Agent] Running sensor anomaly detection...")
    sensor_anomalies = []
    for engine_id in at_risk_engines:
        anomaly_raw = detect_sensor_anomaly(cmapss_df, engine_id)
        if anomaly_raw.get("anomaly_count", 0) > 0:
            sensor_anomalies.append(SensorAnomaly(
                engine_id=anomaly_raw["engine_id"],
                anomaly_count=anomaly_raw["anomaly_count"],
                anomalies=anomaly_raw["anomalies"],
                status=anomaly_raw["status"]
            ))

    print(f"[Fault Agent] Engines with sensor anomalies: {len(sensor_anomalies)}")

    # ── RUN RUL ESTIMATION ──────────────────────────────────
    print("[Fault Agent] Running RUL estimation...")
    rul_estimates = []
    critical_engines = []

    for engine_id in at_risk_engines:
        rul_raw = estimate_rul(cmapss_df, engine_id)
        rul_estimates.append(RULEstimate(
            engine_id=rul_raw["engine_id"],
            current_cycle=rul_raw["current_cycle"],
            estimated_rul=rul_raw["estimated_rul"],
            confidence=rul_raw["confidence"],
            risk_level=rul_raw["risk_level"],
            recommendation=rul_raw["recommendation"],
            status=rul_raw["status"]
        ))
        if rul_raw["risk_level"] in ["critical", "high"]:
            critical_engines.append(engine_id)

    print(f"[Fault Agent] Critical/high risk engines: {critical_engines}")

    # ── RUN CASCADE RISK ANALYSIS ───────────────────────────
    print("[Fault Agent] Running cascade risk analysis...")
    cascade_raw = identify_cascade_risks(
        df,
       telemetry.deadlocks.model_dump(),
telemetry.congestion.model_dump()
    )
    cascade_risks = CascadeRisk(
        cascade_risks_detected=cascade_raw["cascade_risks_detected"],
        risk_count=cascade_raw["risk_count"],
        cascade_risks=cascade_raw["cascade_risks"],
        status=cascade_raw["status"]
    )

    # ── CALL CLAUDE FOR FAULT INTERPRETATION ────────────────
    print("[Fault Agent] Calling Claude for fault interpretation...")

import anthropic
import os
from dotenv import load_dotenv
load_dotenv(override=False)
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment")
client = anthropic.Anthropic(api_key=api_key)

    # Build context — include telemetry summary as background
    rul_context = "\n".join([
        f"  Engine {r.engine_id}: RUL={r.estimated_rul} cycles, "
        f"risk={r.risk_level.upper()}, confidence={r.confidence}, "
        f"action={r.recommendation}"
        for r in rul_estimates
    ])

    anomaly_context = "\n".join([
        f"  Engine {a.engine_id}: {a.anomaly_count} sensor anomalies, "
        f"status={a.status.upper()}"
        for a in sensor_anomalies
    ])

    context = f"""TELEMETRY AGENT SUMMARY (context from previous agent):
{telemetry.summary}

FAULT ANALYSIS RESULTS:

RUL ESTIMATES FOR AT-RISK ENGINES:
{rul_context if rul_context else "No engines in critical RUL window"}

SENSOR ANOMALIES DETECTED:
{anomaly_context if anomaly_context else "No significant sensor anomalies"}

CASCADE RISKS:
- Cascade risks detected: {cascade_risks.cascade_risks_detected}
- Risk count: {cascade_risks.risk_count}
- Status: {cascade_risks.status.upper()}

CURRENT FLEET STATUS:
- Deadlocked robots: {[r['robot_id'] for r in telemetry.deadlocks.deadlocked_robots]}
- Fleet availability: {telemetry.fleet_metrics.fleet_availability:.0%}

Write a 2-3 sentence fault analysis summary for the Alerting Agent.
Distinguish clearly between what is failing NOW versus what is predicted to fail SOON."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=FAULT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}]
    )

    summary = message.content[0].text
    print("[Fault Agent] Summary generated")

    # ── DETERMINE OVERALL FAULT STATUS ──────────────────────
    statuses = [cascade_risks.status]
    statuses += [a.status for a in sensor_anomalies]
    statuses += [r.status for r in rul_estimates]

    if "critical" in statuses:
        overall_status = "critical"
    elif "high" in statuses:
        overall_status = "critical"
    elif "warning" in statuses or "moderate" in statuses:
        overall_status = "warning"
    else:
        overall_status = "normal"

    # ── BUILD OUTPUT ────────────────────────────────────────
    fault_output = FaultOutput(
        sensor_anomalies=sensor_anomalies,
        rul_estimates=rul_estimates,
        cascade_risks=cascade_risks,
        critical_engines=critical_engines,
        summary=summary,
        status=overall_status
    )

    print(f"[Fault Agent] Complete — status: {overall_status.upper()}")
    print(f"[Fault Agent] Summary: {summary[:100]}...")

    return {
        "fault": fault_output,
        "current_agent": "alerting",
        "completed_agents": state.completed_agents + ["fault"]
    }


# ── TEST ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uuid

    print("\n" + "="*55)
    print("Fault Agent — Test Run")
    print("="*55)

    # Load current shift data
    df = pd.read_csv("data/raw/robot_runs_current.csv")
    raw_data = df.to_dict(orient="records")

    # We need telemetry output in state first
    # Import and run telemetry agent to populate state
    from agents.telemetry_agent import run_telemetry_agent

    initial_state = OpsIQState(
        raw_data=raw_data,
        data_source="agv",
        analysis_timestamp=datetime.now().isoformat(),
        run_id=str(uuid.uuid4())[:8]
    )

    print("Running Telemetry Agent first to populate state...")
    telemetry_result = run_telemetry_agent(initial_state)

    # Update state with telemetry output
    state_with_telemetry = initial_state.model_copy(
        update=telemetry_result
    )

    print(f"\nTelemetry complete. Now running Fault Agent...")
    print("="*55)

    # Run fault agent
    fault_result = run_fault_agent(state_with_telemetry)

    # Show output
    fault = fault_result["fault"]
    print("\n" + "="*55)
    print("FAULT AGENT OUTPUT")
    print("="*55)
    print(f"At-risk engines:     {[r.engine_id for r in fault.rul_estimates]}")
    print(f"Critical engines:    {fault.critical_engines}")
    print(f"Sensor anomalies:    {len(fault.sensor_anomalies)} engines affected")
    print(f"Cascade risks:       {fault.cascade_risks.risk_count}")
    print(f"Overall status:      {fault.status.upper()}")
    print(f"\nClaude's fault summary:")
    print(fault.summary)
    print(f"\nCompleted agents: {fault_result['completed_agents']}")