import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import uuid
from datetime import datetime

from tools.analysis_tools import (
    get_run_metrics,
    detect_congestion,
    detect_deadlocks,
    battery_analysis,
    load_support_files
)
from .state import (
    OpsIQState,
    TelemetryOutput,
    FleetMetrics,
    CongestionResult,
    DeadlockResult,
    BatteryResult
)

# ── SYSTEM PROMPT ───────────────────────────────────────────
TELEMETRY_SYSTEM_PROMPT = """You are the Telemetry Agent for OpsIQ, an industrial 
fleet intelligence system.

Your job is to analyze robot fleet telemetry data and produce a concise, 
accurate summary of fleet health for the next agent in the pipeline.

You have already received structured findings from analysis tools. Your job is to:
1. Interpret what the numbers mean in operational terms
2. Identify the most significant finding
3. Write a 2-3 sentence plain English summary that the Fault Agent can use as context

Be direct and specific. Name the robots. State the severity.
Do not use markdown. Do not repeat the raw numbers — interpret them."""

def run_telemetry_agent(state: OpsIQState) -> dict:
    import anthropic
    import os
    from dotenv import load_dotenv
    load_dotenv(override=False)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    print(f"[Telemetry Agent] API key found: {bool(api_key)}, length: {len(api_key) if api_key else 0}")
    client = anthropic.Anthropic(api_key=api_key)

    print("\n[Telemetry Agent] Starting analysis...")

    # Load raw data from state
    df = pd.DataFrame(state.raw_data)
    print(f"[Telemetry Agent] Analyzing {len(df)} events across {df['robot_id'].nunique()} robots")

    # ── CALL ALL 4 TELEMETRY TOOLS ──────────────────────────
    print("[Telemetry Agent] Running get_run_metrics...")
    metrics_raw = get_run_metrics(df)

    print("[Telemetry Agent] Running detect_congestion...")
    congestion_raw = detect_congestion(df)

    print("[Telemetry Agent] Running detect_deadlocks...")
    deadlocks_raw = detect_deadlocks(df)

    print("[Telemetry Agent] Running battery_analysis...")
    battery_raw = battery_analysis(df)

    # ── PARSE INTO PYDANTIC MODELS ──────────────────────────
    fleet_metrics = FleetMetrics(
        total_robots=metrics_raw["total_robots"],
        active_robots=metrics_raw["active_robots"],
        fleet_availability=metrics_raw["fleet_availability"],
        avg_health_score=metrics_raw["avg_health_score"],
        task_completion_rate=metrics_raw["task_completion_rate"],
        deadlock_count=metrics_raw["deadlock_count"],
        path_blocked_count=metrics_raw["path_blocked_count"],
        downtime_pct=metrics_raw["downtime_pct"],
        status=metrics_raw["status"]
    )

    congestion = CongestionResult(
        congestion_detected=congestion_raw["congestion_detected"],
        hotspot_count=congestion_raw["hotspot_count"],
        hotspots=congestion_raw["hotspots"],
        total_congestion_events=congestion_raw["total_congestion_events"],
        status=congestion_raw["status"]
    )

    deadlocks = DeadlockResult(
        deadlocks_detected=deadlocks_raw["deadlocks_detected"],
        deadlocked_robot_count=deadlocks_raw["deadlocked_robot_count"],
        deadlocked_robots=deadlocks_raw["deadlocked_robots"],
        status=deadlocks_raw["status"]
    )

    battery = BatteryResult(
        robots_analyzed=battery_raw["robots_analyzed"],
        critical_battery_count=battery_raw["critical_battery_count"],
        warning_battery_count=battery_raw["warning_battery_count"],
        battery_alerts=battery_raw["battery_alerts"],
        fleet_avg_battery=battery_raw["fleet_avg_battery"],
        status=battery_raw["status"]
    )

    # ── CALL CLAUDE FOR INTERPRETATION ─────────────────────
    print("[Telemetry Agent] Calling Claude for interpretation...")



    # Build context for Claude
    context = f"""Fleet telemetry analysis results:

FLEET METRICS:
- Total robots: {fleet_metrics.total_robots}
- Fleet availability: {fleet_metrics.fleet_availability:.0%}
- Average health score: {fleet_metrics.avg_health_score}
- Task completion rate: {fleet_metrics.task_completion_rate:.0%}
- Deadlock events: {fleet_metrics.deadlock_count}
- Path blocked events: {fleet_metrics.path_blocked_count}
- Overall status: {fleet_metrics.status.upper()}

DEADLOCKS:
- Deadlocks detected: {deadlocks.deadlocks_detected}
- Deadlocked robots: {[r['robot_id'] for r in deadlocks.deadlocked_robots]}
- Status: {deadlocks.status.upper()}

CONGESTION:
- Hotspots found: {congestion.hotspot_count}
- Total congestion events: {congestion.total_congestion_events}
- Status: {congestion.status.upper()}

BATTERY:
- Fleet average battery: {battery.fleet_avg_battery}%
- Critical battery alerts: {battery.critical_battery_count}
- Status: {battery.status.upper()}

Write a 2-3 sentence operational summary for the Fault Agent."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=TELEMETRY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}]
    )

    summary = message.content[0].text
    print(f"[Telemetry Agent] Summary generated")

    # ── DETERMINE OVERALL STATUS ────────────────────────────
    statuses = [
        fleet_metrics.status,
        congestion.status,
        deadlocks.status,
        battery.status
    ]
    if "critical" in statuses:
        overall_status = "critical"
    elif "warning" in statuses:
        overall_status = "warning"
    else:
        overall_status = "normal"

    # ── BUILD OUTPUT ────────────────────────────────────────
    telemetry_output = TelemetryOutput(
        fleet_metrics=fleet_metrics,
        congestion=congestion,
        deadlocks=deadlocks,
        battery=battery,
        summary=summary,
        status=overall_status
    )

    print(f"[Telemetry Agent] Complete — status: {overall_status.upper()}")
    print(f"[Telemetry Agent] Summary: {summary[:100]}...")

    # Return state update
    return {
        "telemetry": telemetry_output,
        "current_agent": "fault",
        "completed_agents": state.completed_agents + ["telemetry"]
    }


# ── TEST ────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("\n" + "="*55)
    print("Telemetry Agent — Test Run")
    print("="*55)

    # Load current shift data
    df = pd.read_csv("data/raw/robot_runs_current.csv")
    raw_data = df.to_dict(orient="records")

    # Build initial state
    initial_state = OpsIQState(
        raw_data=raw_data,
        data_source="agv",
        analysis_timestamp=datetime.now().isoformat(),
        run_id=str(uuid.uuid4())[:8]
    )

    print(f"Run ID: {initial_state.run_id}")
    print(f"Records: {len(raw_data)}")

    # Run the agent
    result = run_telemetry_agent(initial_state)

    # Show output
    telemetry = result["telemetry"]
    print("\n" + "="*55)
    print("TELEMETRY AGENT OUTPUT")
    print("="*55)
    print(f"Fleet availability:  {telemetry.fleet_metrics.fleet_availability:.0%}")
    print(f"Avg health score:    {telemetry.fleet_metrics.avg_health_score}")
    print(f"Deadlocked robots:   {[r['robot_id'] for r in telemetry.deadlocks.deadlocked_robots]}")
    print(f"Congestion hotspots: {telemetry.congestion.hotspot_count}")
    print(f"Overall status:      {telemetry.status.upper()}")
    print(f"\nClaude's summary:")
    print(telemetry.summary)
    print(f"\nCompleted agents: {result['completed_agents']}")