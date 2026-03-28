import os
import streamlit as st

# Set API key from secrets (Streamlit Cloud) or env
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
    os.environ["ANTHROPIC_API_KEY"] = api_key
except Exception:
    pass  # Fall back to env var silently

# Now import everything else
import pandas as pd
import json
import uuid
import sys
from datetime import datetime
from pathlib import Path

# Add root to path for cloud deployment
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from agents.pipeline import run_pipeline, build_pipeline
from agents.state import (
    OpsIQState, TelemetryOutput, FaultOutput, AlertOutput, ReportOutput,
    FleetMetrics, CongestionResult, DeadlockResult, BatteryResult,
    SensorAnomaly, RULEstimate, CascadeRisk, Alert,
)

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="MeridianIQ — Industrial Fleet Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #1D9E75;
    }
    .critical { border-left-color: #dc3545; background: #fff5f5; }
    .warning  { border-left-color: #ffc107; background: #fffdf0; }
    .normal   { border-left-color: #28a745; background: #f0fff4; }
    .alert-critical { background: #fff5f5; border-radius: 6px; padding: .75rem; margin: .375rem 0; border-left: 3px solid #dc3545; }
    .alert-high     { background: #fffdf0; border-radius: 6px; padding: .75rem; margin: .375rem 0; border-left: 3px solid #ffc107; }
    .alert-moderate { background: #f0f8ff; border-radius: 6px; padding: .75rem; margin: .375rem 0; border-left: 3px solid #17a2b8; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #1D9E75; margin: 1rem 0 .5rem; }
    .report-text { background: #f8f9fa; border-radius: 8px; padding: 1.5rem; line-height: 1.7; font-size: .95rem; }
    .chat-message-user      { background: #E1F5EE; border-radius: 8px; padding: .75rem 1rem; margin: .5rem 0; }
    .chat-message-assistant { background: #f8f9fa; border-radius: 8px; padding: .75rem 1rem; margin: .5rem 0; }
    .brand-tagline { color: #888780; font-size: .95rem; margin-top: -.5rem; margin-bottom: 1rem; }
    .about-card { background: #E1F5EE; border-radius: 10px; padding: 2rem; margin: 1rem 0; }
    .about-card h3 { color: #1D9E75; }
    .agent-status { background: #E1F5EE; border-radius: 6px; padding: .5rem 1rem; margin: .25rem 0; font-size: .9rem; }
    .agent-status.done { opacity: 0.6; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    logo_path = Path(__file__).parent / "logo.svg"
    if logo_path.exists():
        st.image(str(logo_path), width=220)
    else:
        st.markdown("### 🌐 MeridianIQ")
    st.markdown('<p class="brand-tagline">Intelligence at the intersection of data and action</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Select view",
        ["Live Analysis", "Plant Dashboard", "Executive Report", "Chat Interface", "About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Powered by Claude + LangGraph")

# ── SESSION STATE ───────────────────────────────────────────
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False

# ── HELPER FUNCTIONS ────────────────────────────────────────
def severity_color(severity: str) -> str:
    return {"critical": "🔴", "high": "🟡", "moderate": "🔵", "normal": "🟢"}.get(severity.lower(), "⚪")

def status_badge(status: str) -> str:
    colors = {"critical": "#dc3545", "warning": "#ffc107", "normal": "#28a745"}
    color = colors.get(status.lower(), "#6c757d")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:600">{status.upper()}</span>'

def run_analysis(df: pd.DataFrame) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
        except Exception:
            pass
    if not api_key:
        st.error("No API key found. Set ANTHROPIC_API_KEY in environment or Streamlit secrets.")
        st.stop()

    raw_data = df.to_dict(orient="records")
    run_id = str(uuid.uuid4())[:8]

    initial_state = OpsIQState(
        raw_data=raw_data,
        data_source="agv",
        analysis_timestamp=datetime.now().isoformat(),
        run_id=run_id
    )

    app = build_pipeline()

    # Show per-agent progress
    agent_steps = [
        ("Telemetry Agent", "Analyzing fleet metrics, deadlocks, congestion..."),
        ("Fault Agent", "Running predictive maintenance & RUL estimation..."),
        ("Alerting Agent", "Generating alerts and escalation rules..."),
        ("Reporting Agent", "Compiling executive report..."),
    ]
    progress_bar = st.progress(0, text="Starting pipeline...")
    status_container = st.empty()

    for i, (name, desc) in enumerate(agent_steps):
        progress_bar.progress((i) / len(agent_steps), text=f"Running {name}...")
        status_container.markdown(
            f'<div class="agent-status">⏳ <strong>{name}</strong> — {desc}</div>',
            unsafe_allow_html=True,
        )

    result = app.invoke(initial_state)

    progress_bar.progress(1.0, text="Pipeline complete!")
    status_container.markdown(
        '<div class="agent-status done">✅ All agents complete</div>',
        unsafe_allow_html=True,
    )

    st.session_state.pipeline_result = result
    st.session_state.run_id = run_id
    st.session_state.demo_mode = False
    return result


def load_demo_results() -> dict:
    """Load pre-computed demo results — a critical fleet scenario."""
    telemetry = TelemetryOutput(
        fleet_metrics=FleetMetrics(
            total_robots=12,
            active_robots=9,
            fleet_availability=0.75,
            avg_health_score=62,
            task_completion_rate=0.71,
            deadlock_count=3,
            path_blocked_count=7,
            downtime_pct=0.25,
            status="critical",
        ),
        congestion=CongestionResult(
            congestion_detected=True,
            hotspot_count=2,
            hotspots=[
                {"location": "Zone B — Charging Bay", "event_count": 14, "robots_affected": 4, "severity": "high"},
                {"location": "Zone D — Loading Dock", "event_count": 9, "robots_affected": 3, "severity": "moderate"},
            ],
            total_congestion_events=23,
            status="warning",
        ),
        deadlocks=DeadlockResult(
            deadlocks_detected=True,
            deadlocked_robot_count=3,
            deadlocked_robots=[
                {"robot_id": "AGV-007", "deadlock_events": 5, "avg_speed": 0.02, "battery": 34, "health_score": 28, "severity": "critical"},
                {"robot_id": "AGV-003", "deadlock_events": 3, "avg_speed": 0.08, "battery": 41, "health_score": 45, "severity": "high"},
                {"robot_id": "AGV-011", "deadlock_events": 2, "avg_speed": 0.12, "battery": 55, "health_score": 52, "severity": "moderate"},
            ],
            status="critical",
        ),
        battery=BatteryResult(
            robots_analyzed=12,
            critical_battery_count=2,
            warning_battery_count=3,
            battery_alerts=[
                {"robot_id": "AGV-007", "battery": 34, "status": "critical"},
                {"robot_id": "AGV-003", "battery": 41, "status": "warning"},
            ],
            fleet_avg_battery=67.3,
            status="warning",
        ),
        summary="Fleet in CRITICAL state — 3 robots deadlocked, 2 congestion hotspots, 75% availability. Immediate intervention required for AGV-007.",
        status="critical",
    )

    fault = FaultOutput(
        sensor_anomalies=[
            SensorAnomaly(engine_id=14, anomaly_count=8, anomalies=[
                {"sensor": "s_11", "value": 47.8, "expected_range": "38-43", "deviation": "critical"},
                {"sensor": "s_15", "value": 22.1, "expected_range": "18-20", "deviation": "high"},
            ], status="critical"),
            SensorAnomaly(engine_id=27, anomaly_count=4, anomalies=[
                {"sensor": "s_04", "value": 1412, "expected_range": "1380-1400", "deviation": "moderate"},
            ], status="warning"),
        ],
        rul_estimates=[
            RULEstimate(engine_id=14, current_cycle=187, estimated_rul=12, confidence="high", risk_level="critical", recommendation="Schedule immediate replacement — estimated 12 cycles remaining", status="critical"),
            RULEstimate(engine_id=27, current_cycle=145, estimated_rul=38, confidence="medium", risk_level="high", recommendation="Plan maintenance within next 2 shifts", status="warning"),
            RULEstimate(engine_id=6, current_cycle=89, estimated_rul=95, confidence="high", risk_level="moderate", recommendation="Monitor — schedule inspection next maintenance window", status="normal"),
        ],
        cascade_risks=CascadeRisk(
            cascade_risks_detected=True,
            risk_count=2,
            cascade_risks=[
                {"trigger_type": "deadlock_battery_drain", "severity": "critical", "risk_description": "AGV-007 deadlock is draining battery below safe threshold. If battery fails in Zone B, it will block 3 other robots causing a cascade deadlock across the charging bay corridor."},
                {"trigger_type": "engine_failure_propagation", "severity": "high", "risk_description": "Engine 14 failure during peak shift could force rerouting through Zone D, compounding existing congestion and creating 15+ minute delays fleet-wide."},
            ],
            status="critical",
        ),
        critical_engines=[14, 27],
        summary="Engine 14 is in critical condition with only 12 cycles remaining. Engine 27 needs attention within 2 shifts. Cascade risk detected between deadlocked AGV-007 and charging bay congestion.",
        status="critical",
    )

    alerts = AlertOutput(
        active_alerts=[
            Alert(alert_id="ALT-001", robot_id="AGV-007", severity="critical", alert_type="deadlock_battery", description="AGV-007 deadlocked with critically low battery (34%). Risk of total shutdown blocking Zone B corridor.", is_recurring=True, recurrence_count=3, recommended_action="Dispatch technician immediately to manually reposition AGV-007 and connect to charger", escalate_to="Shift Supervisor"),
            Alert(alert_id="ALT-002", engine_id=14, severity="critical", alert_type="rul_critical", description="Engine 14 has only 12 estimated cycles remaining. 8 sensor anomalies detected across critical parameters.", is_recurring=False, recurrence_count=0, recommended_action="Schedule engine replacement before next production shift", escalate_to="Maintenance Lead"),
            Alert(alert_id="ALT-003", robot_id="AGV-003", severity="high", alert_type="deadlock", description="AGV-003 experiencing repeated deadlocks (3 events). Health score degrading.", is_recurring=True, recurrence_count=2, recommended_action="Reroute AGV-003 away from Zone B and run diagnostic", escalate_to="Line Supervisor"),
            Alert(alert_id="ALT-004", engine_id=27, severity="high", alert_type="rul_warning", description="Engine 27 showing early degradation — 38 cycles remaining with 4 sensor anomalies.", is_recurring=False, recurrence_count=0, recommended_action="Plan maintenance within next 2 shifts", escalate_to="Maintenance Lead"),
            Alert(alert_id="ALT-005", severity="moderate", alert_type="congestion", description="Zone D Loading Dock congestion affecting 3 robots with 9 blocked events this shift.", is_recurring=True, recurrence_count=5, recommended_action="Adjust traffic routing to distribute load across Zones C and D", escalate_to=None),
        ],
        new_alert_count=2,
        recurring_alert_count=3,
        escalation_count=4,
        escalations=[
            {"asset": "AGV-007", "severity": "critical", "escalate_to": "Shift Supervisor"},
            {"asset": "Engine 14", "severity": "critical", "escalate_to": "Maintenance Lead"},
            {"asset": "AGV-003", "severity": "high", "escalate_to": "Line Supervisor"},
            {"asset": "Engine 27", "severity": "high", "escalate_to": "Maintenance Lead"},
        ],
        summary="5 active alerts — 2 critical, 2 high, 1 moderate. 4 escalations generated.",
        status="critical",
    )

    report = ReportOutput(
        executive_summary="Fleet is in CRITICAL condition. AGV-007 is deadlocked with critically low battery posing cascade risk across Zone B. Engine 14 has only 12 cycles of estimated remaining life and requires immediate replacement scheduling. Fleet availability has dropped to 75% with 3 robots deadlocked and 2 congestion hotspots. Immediate action is required on 2 critical escalations to prevent further degradation.",
        critical_findings=[
            "AGV-007 deadlocked with 34% battery — cascade risk will block Zone B charging corridor if battery fails",
            "Engine 14 has only 12 estimated cycles remaining with 8 sensor anomalies across critical parameters",
            "Fleet availability at 75% — 3 of 12 robots deadlocked, well below 90% operational target",
            "Zone B Charging Bay congestion (14 events, 4 robots affected) compounding deadlock situation",
            "Cascade risk: AGV-007 failure would trigger secondary deadlocks affecting 3 additional robots",
        ],
        recommendations=[
            "IMMEDIATE: Dispatch technician to manually reposition AGV-007 and connect to emergency charger",
            "URGENT: Schedule Engine 14 replacement before next production shift — only 12 cycles remaining",
            "HIGH: Reroute AGV-003 and AGV-011 away from Zone B to break deadlock chain",
            "HIGH: Plan Engine 27 maintenance within next 2 shifts to prevent escalation to critical",
            "MODERATE: Adjust Zone D traffic routing to distribute load and reduce congestion events",
        ],
        severity_summary={"critical": 2, "high": 2, "moderate": 1},
        full_report="MERIDIANIQ OPERATIONS BRIEF\n\nFleet Status: CRITICAL\nAnalysis Timestamp: Demo Mode\nRun ID: demo-001\n\n1. FLEET HEALTH\nFleet availability has dropped to 75%, significantly below the 90% operational target. Of 12 robots, 9 are actively operating. Three robots (AGV-007, AGV-003, AGV-011) are in various stages of deadlock. Average fleet health score is 62/100, pulled down primarily by AGV-007 (28) and AGV-003 (45).\n\n2. CRITICAL DEADLOCK — AGV-007\nAGV-007 is the highest-priority issue. It has experienced 5 deadlock events this shift with near-zero movement (avg speed 0.02). Its battery is at 34% and dropping. If the battery fails while AGV-007 is blocking the Zone B charging corridor, it will prevent 3 other robots from reaching their chargers, creating a cascade deadlock that could take down 50% of the fleet within 2 hours.\n\n3. PREDICTIVE MAINTENANCE\nEngine 14 is in critical condition with an estimated 12 cycles of remaining useful life. Eight sensor parameters are outside normal ranges, with s_11 showing the most severe deviation. This engine must be replaced before the next production shift. Engine 27 is showing early signs of degradation (38 cycles remaining) and should be scheduled for maintenance within 2 shifts.\n\n4. CONGESTION\nTwo congestion hotspots are active: Zone B Charging Bay (14 events, 4 robots) and Zone D Loading Dock (9 events, 3 robots). The Zone B congestion is directly linked to the deadlock situation and will resolve once AGV-007 is cleared. Zone D congestion can be mitigated through traffic routing adjustments.\n\n5. RECOMMENDED ACTIONS\n1. Dispatch technician to AGV-007 immediately\n2. Schedule Engine 14 replacement\n3. Reroute AGV-003 and AGV-011 away from Zone B\n4. Plan Engine 27 maintenance\n5. Adjust Zone D traffic routing",
        status="critical",
    )

    return {
        "telemetry": telemetry,
        "fault": fault,
        "alerts": alerts,
        "report": report,
    }


# ── PAGE 1 — LIVE ANALYSIS (LINE MANAGER) ──────────────────
if page == "Live Analysis":
    st.title("🌐 MeridianIQ — Live Fleet Analysis")
    st.markdown('<p class="brand-tagline">Intelligence at the intersection of data and action</p>', unsafe_allow_html=True)
    st.caption("For operations managers and line supervisors — what needs attention right now")

    # File upload or use default
    col1, col2, col3 = st.columns([2, 0.8, 0.8])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload robot telemetry CSV",
            type=["csv"],
            help="Upload a robot_runs CSV file or use the default current shift data"
        )
        # Download sample CSV
        sample_csv_path = Path(__file__).parent.parent / "data" / "raw" / "robot_runs_current.csv"
        if sample_csv_path.exists():
            with open(sample_csv_path, "rb") as f:
                st.download_button(
                    "📥 Download sample CSV",
                    data=f,
                    file_name="robot_runs_current.csv",
                    mime="text/csv",
                )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_default = st.button(
            "Run with current shift data",
            type="primary",
            use_container_width=True
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        use_demo = st.button(
            "⚡ Demo Mode",
            use_container_width=True,
            help="Load pre-computed results instantly — no API key or pipeline run needed",
        )

    # Trigger analysis
    if use_demo:
        result = load_demo_results()
        st.session_state.pipeline_result = result
        st.session_state.run_id = "demo-001"
        st.session_state.demo_mode = True
        st.rerun()

    if uploaded_file or use_default:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv("data/raw/robot_runs_current.csv")

        result = run_analysis(df)

    # Show results
    if st.session_state.pipeline_result:
        result = st.session_state.pipeline_result
        telemetry = result.get("telemetry")
        alerts = result.get("alerts")
        report = result.get("report")

        if not all([telemetry, alerts, report]):
            st.error("Pipeline completed with errors. Check logs.")
            st.stop()

        if st.session_state.demo_mode:
            st.info("⚡ **Demo Mode** — showing pre-computed results for a critical fleet scenario. Run the pipeline with real data for live analysis.")

        # Status banner
        status = report.status
        st.markdown(f"""
        <div class="metric-card {status}">
            <strong>Fleet Status: {status_badge(status)}</strong> &nbsp;
            Run ID: {st.session_state.run_id} &nbsp;|&nbsp;
            {datetime.now().strftime('%B %d, %Y %H:%M')}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # KPI metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric(
                "Fleet Availability",
                f"{telemetry.fleet_metrics.fleet_availability:.0%}",
                delta=None
            )
        with m2:
            st.metric(
                "Active Robots",
                f"{telemetry.fleet_metrics.active_robots}/{telemetry.fleet_metrics.total_robots}"
            )
        with m3:
            st.metric(
                "Avg Health Score",
                f"{telemetry.fleet_metrics.avg_health_score:.0f}"
            )
        with m4:
            st.metric(
                "Active Alerts",
                len(alerts.active_alerts),
                delta=f"{alerts.new_alert_count} new"
            )
        with m5:
            st.metric(
                "Deadlocked Robots",
                telemetry.deadlocks.deadlocked_robot_count
            )

        st.markdown("---")

        # Active alerts — the line manager's primary view
        col_alerts, col_summary = st.columns([1.5, 1])

        with col_alerts:
            st.markdown('<div class="section-header">Active Alerts</div>',
                       unsafe_allow_html=True)

            # Filter controls
            severity_filter = st.multiselect(
                "Filter by severity",
                ["critical", "high", "moderate"],
                default=["critical", "high"],
                label_visibility="collapsed"
            )

            filtered_alerts = [
                a for a in alerts.active_alerts
                if a.severity in severity_filter
            ]

            for alert in filtered_alerts:
                css_class = f"alert-{alert.severity}"
                robot_or_engine = (
                    f"Robot {alert.robot_id}" if alert.robot_id
                    else f"Engine {alert.engine_id}" if alert.engine_id
                    else "Fleet-wide"
                )
                recurring_tag = "🔁 Recurring" if alert.is_recurring else "🆕 New"
                escalate_tag = f"→ {alert.escalate_to}" if alert.escalate_to else ""

                st.markdown(f"""
                <div class="{css_class}">
                    <strong>{severity_color(alert.severity)} {robot_or_engine}</strong>
                    &nbsp; {recurring_tag} &nbsp; <em>{escalate_tag}</em><br>
                    <small>{alert.description}</small><br>
                    <small><strong>Action:</strong> {alert.recommended_action}</small>
                </div>
                """, unsafe_allow_html=True)

            if not filtered_alerts:
                st.success("No alerts matching selected severity filters")

        with col_summary:
            st.markdown('<div class="section-header">Alert Summary</div>',
                       unsafe_allow_html=True)

            # Severity breakdown
            sev = report.severity_summary
            st.markdown(f"""
            <div class="metric-card">
                🔴 Critical: <strong>{sev.get('critical', 0)}</strong><br>
                🟡 High: <strong>{sev.get('high', 0)}</strong><br>
                🔵 Moderate: <strong>{sev.get('moderate', 0)}</strong><br>
                🔁 Recurring: <strong>{alerts.recurring_alert_count}</strong><br>
                🆕 New: <strong>{alerts.new_alert_count}</strong>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="section-header">Escalations</div>',
                       unsafe_allow_html=True)
            for esc in alerts.escalations[:8]:
                st.markdown(f"""
                <div class="metric-card" style="margin:.25rem 0;padding:.5rem">
                    {severity_color(esc['severity'])}
                    <strong>{esc['asset']}</strong> → {esc['escalate_to']}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Deadlock and congestion detail
        col_dead, col_cong = st.columns(2)

        with col_dead:
            st.markdown('<div class="section-header">Deadlocked Robots</div>',
                       unsafe_allow_html=True)
            if telemetry.deadlocks.deadlocked_robots:
                dead_df = pd.DataFrame(telemetry.deadlocks.deadlocked_robots)
                st.dataframe(
                    dead_df[["robot_id", "deadlock_events", "avg_speed",
                             "battery", "health_score", "severity"]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("No deadlocked robots")

        with col_cong:
            st.markdown('<div class="section-header">Congestion Hotspots</div>',
                       unsafe_allow_html=True)
            if telemetry.congestion.hotspots:
                cong_df = pd.DataFrame(telemetry.congestion.hotspots)
                st.dataframe(
                    cong_df[["location", "event_count",
                             "robots_affected", "severity"]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("No congestion hotspots detected")

    else:
        st.info("Upload a telemetry file, click 'Run with current shift data', or try **Demo Mode** for instant results")


# ── PAGE 2 — PLANT DASHBOARD (PLANT MANAGER) ───────────────
elif page == "Plant Dashboard":
    st.title("🏭 MeridianIQ — Plant Dashboard")
    st.caption("For plant managers — fleet health, predictive maintenance, and trends")

    if not st.session_state.pipeline_result:
        st.warning("No analysis results yet. Go to Live Analysis and run the pipeline first.")
        st.stop()

    result = st.session_state.pipeline_result
    telemetry = result.get("telemetry")
    fault = result.get("fault")
    report = result.get("report")

    # Fleet health overview
    st.markdown('<div class="section-header">Fleet Health Overview</div>',
               unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fleet Availability", f"{telemetry.fleet_metrics.fleet_availability:.0%}")
    with col2:
        st.metric("Avg Health Score", f"{telemetry.fleet_metrics.avg_health_score:.0f}/100")
    with col3:
        st.metric("Total Downtime (30d)",
                  f"{sum(r.get('total_downtime_hours',0) for r in telemetry.fleet_metrics.__dict__.get('robot_details', [])):.0f} hrs"
                  if hasattr(telemetry.fleet_metrics, 'robot_details') else "N/A")
    with col4:
        st.metric("Critical Engines", len(fault.critical_engines) if fault else 0)

    st.markdown("---")

    # Predictive maintenance — the plant manager's key view
    col_rul, col_anomaly = st.columns(2)

    with col_rul:
        st.markdown('<div class="section-header">Engine RUL — Predicted Remaining Life</div>',
                   unsafe_allow_html=True)
        if fault and fault.rul_estimates:
            rul_data = []
            for r in fault.rul_estimates:
                rul_data.append({
                    "Engine": r.engine_id,
                    "Cycles Remaining": r.estimated_rul,
                    "Risk Level": r.risk_level.upper(),
                    "Confidence": r.confidence,
                    "Action Required": r.recommendation
                })
            rul_df = pd.DataFrame(rul_data)
            st.dataframe(rul_df, use_container_width=True, hide_index=True)
        else:
            st.info("No RUL data available")

    with col_anomaly:
        st.markdown('<div class="section-header">Sensor Anomalies Detected</div>',
                   unsafe_allow_html=True)
        if fault and fault.sensor_anomalies:
            for anomaly in fault.sensor_anomalies:
                st.markdown(f"""
                <div class="alert-{'critical' if anomaly.status == 'critical' else 'high'}">
                    <strong>Engine {anomaly.engine_id}</strong>
                    — {anomaly.anomaly_count} sensor anomalies
                    — Status: {anomaly.status.upper()}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No significant sensor anomalies detected")

    st.markdown("---")

    # Cascade risks
    st.markdown('<div class="section-header">Cascade Risk Analysis</div>',
               unsafe_allow_html=True)
    if fault and fault.cascade_risks.cascade_risks_detected:
        for risk in fault.cascade_risks.cascade_risks[:5]:
            severity = risk.get("severity", "moderate")
            st.markdown(f"""
            <div class="alert-{severity}">
                {severity_color(severity)}
                <strong>{risk.get('trigger_type', '').replace('_', ' ').title()}</strong><br>
                <small>{risk.get('risk_description', '')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No cascade risks detected")

    st.markdown("---")

    # Full findings
    st.markdown('<div class="section-header">Critical Findings</div>',
               unsafe_allow_html=True)
    if report:
        for finding in report.critical_findings:
            if len(finding) > 30:
                st.markdown(f"• {finding}")


# ── PAGE 3 — EXECUTIVE REPORT ───────────────────────────────
elif page == "Executive Report":
    st.title("📊 MeridianIQ — Executive Report")
    st.caption("For plant directors and executives — status, risk, and recommended actions")

    if not st.session_state.pipeline_result:
        st.warning("No analysis results yet. Go to Live Analysis and run the pipeline first.")
        st.stop()

    result = st.session_state.pipeline_result
    report = result.get("report")
    alerts = result.get("alerts")

    if not report:
        st.error("Report generation failed.")
        st.stop()

    # Executive summary — the only thing an exec needs to read
    st.markdown('<div class="section-header">Executive Summary</div>',
               unsafe_allow_html=True)

    status_html = status_badge(report.status)
    st.markdown(f"""
    <div class="metric-card {report.status}" style="padding:1.5rem">
        {status_html}<br><br>
        <p style="font-size:1.05rem;line-height:1.7">{report.executive_summary}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    sev = report.severity_summary
    with col1:
        st.metric("Critical Alerts", sev.get("critical", 0))
    with col2:
        st.metric("High Alerts", sev.get("high", 0))
    with col3:
        st.metric("Total Escalations", alerts.escalation_count)
    with col4:
        st.metric("Recurring Issues", alerts.recurring_alert_count)

    st.markdown("---")

    col_rec, col_brief = st.columns([1, 1.5])

    with col_rec:
        st.markdown('<div class="section-header">Recommended Actions</div>',
                   unsafe_allow_html=True)
        for i, rec in enumerate(report.recommendations, 1):
            if len(rec) > 20:
                st.markdown(f"""
                <div class="metric-card" style="margin:.375rem 0;padding:.75rem">
                    <strong>{i}.</strong> {rec}
                </div>
                """, unsafe_allow_html=True)

    with col_brief:
        st.markdown('<div class="section-header">Full Operations Brief</div>',
                   unsafe_allow_html=True)
        st.markdown(f"""
        <div class="report-text">{report.full_report.replace(chr(10), '<br>')}</div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Download report
    report_text = f"""MERIDIANIQ OPERATIONS REPORT
Run ID: {st.session_state.run_id}
Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}

EXECUTIVE SUMMARY
{report.executive_summary}

CRITICAL FINDINGS
{chr(10).join(f'• {f}' for f in report.critical_findings if len(f) > 20)}

RECOMMENDATIONS
{chr(10).join(f'{i+1}. {r}' for i, r in enumerate(report.recommendations) if len(r) > 20)}

FULL OPERATIONS BRIEF
{report.full_report}
"""
    st.download_button(
        label="Download Report as TXT",
        data=report_text,
        file_name=f"meridianiq_report_{st.session_state.run_id}.txt",
        mime="text/plain"
    )


# ── PAGE 4 — CHAT INTERFACE ─────────────────────────────────
elif page == "Chat Interface":
    st.title("💬 MeridianIQ — Ask the Fleet")
    st.caption("Ask questions about the current analysis in plain English")

    if not st.session_state.pipeline_result:
        st.warning("No analysis results yet. Go to Live Analysis and run the pipeline first.")
        st.stop()

    result = st.session_state.pipeline_result
    telemetry = result.get("telemetry")
    fault = result.get("fault")
    alerts = result.get("alerts")
    report = result.get("report")

    # Build context from pipeline results
    chat_context = f"""You are MeridianIQ, an industrial fleet intelligence assistant.
You have just completed a full analysis of the robot fleet. Here are the findings:

FLEET STATUS: {telemetry.status.upper()}
Fleet availability: {telemetry.fleet_metrics.fleet_availability:.0%}
Active robots: {telemetry.fleet_metrics.active_robots}/{telemetry.fleet_metrics.total_robots}
Deadlocked robots: {[r['robot_id'] for r in telemetry.deadlocks.deadlocked_robots]}
Congestion hotspots: {telemetry.congestion.hotspot_count}

FAULT ANALYSIS:
Critical engines: {fault.critical_engines if fault else []}
Cascade risks: {fault.cascade_risks.risk_count if fault else 0}

ALERTS:
Total: {len(alerts.active_alerts)}
Critical: {report.severity_summary.get('critical', 0)}

EXECUTIVE SUMMARY:
{report.executive_summary}

FULL REPORT:
{report.full_report}

Answer questions about this analysis in plain English.
Be specific — name robots and engines. Give actionable answers.
If asked something outside the analysis, say so honestly.
Do not use markdown formatting."""

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message-user">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message-assistant">
                <strong>MeridianIQ:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)

    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Which robot needs attention first?"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "Which robot needs attention first?"
                })
                st.rerun()
        with col2:
            if st.button("Which engine is most at risk?"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "Which engine is most at risk of failure?"
                })
                st.rerun()
        with col3:
            if st.button("What caused the cascade risk?"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "What caused the cascade risk and how do I stop it?"
                })
                st.rerun()

    # Chat input
    user_input = st.chat_input("Ask a question about the fleet...")

    if user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        # Call Claude with full context — create client inside the handler
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            try:
                api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            except Exception:
                pass
        if not api_key:
            st.error("No API key found. Set ANTHROPIC_API_KEY in environment or Streamlit secrets.")
            st.stop()

        client = anthropic.Anthropic(api_key=api_key)

        with st.spinner("Thinking..."):
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": chat_context + f"\n\nUser question: {user_input}"
                }]
            )
            answer = response.content[0].text

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })
        st.rerun()

    # Clear chat
    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()


# ── PAGE 5 — ABOUT ──────────────────────────────────────────
elif page == "About":
    st.title("🌐 MeridianIQ")
    st.markdown('<p class="brand-tagline" style="font-size:1.2rem">Intelligence at the intersection of data and action</p>', unsafe_allow_html=True)

    st.markdown("---")

    col_about, col_method = st.columns(2)

    with col_about:
        st.markdown("""
        <div class="about-card">
            <h3>What is MeridianIQ?</h3>
            <p>MeridianIQ is a multi-agent industrial fleet intelligence platform that transforms
            raw telemetry and sensor data into actionable operational insights.</p>
            <p>Built on a pipeline of specialized AI agents, MeridianIQ analyzes robot fleet health,
            predicts equipment failures, detects cascade risks, and delivers prioritized
            recommendations — from the shop floor to the executive suite.</p>
            <h3>Who is it for?</h3>
            <ul>
                <li><strong>Line Supervisors</strong> — real-time alerts and deadlock resolution</li>
                <li><strong>Plant Managers</strong> — predictive maintenance and fleet health trends</li>
                <li><strong>Executives</strong> — status reports and strategic recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_method:
        st.markdown("""
        <div class="about-card">
            <h3>Methodology</h3>
            <p>MeridianIQ runs a four-stage agent pipeline powered by Claude and orchestrated with LangGraph:</p>
            <ol>
                <li><strong>Telemetry Agent</strong> — Ingests AGV robot data. Computes fleet metrics,
                detects deadlocks, identifies congestion hotspots, and monitors battery health.</li>
                <li><strong>Fault Agent</strong> — Analyzes NASA CMAPSS turbofan sensor data. Estimates
                remaining useful life (RUL), detects sensor anomalies, and models cascade failure risks.</li>
                <li><strong>Alerting Agent</strong> — Generates prioritized alerts with severity classification,
                recurrence tracking, and escalation routing.</li>
                <li><strong>Reporting Agent</strong> — Synthesizes all findings into role-appropriate reports
                with executive summaries, critical findings, and actionable recommendations.</li>
            </ol>
            <p><strong>Conditional routing:</strong> If the fleet is healthy, the pipeline skips the
            expensive fault analysis — saving time and cost when it's not needed.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="about-card" style="text-align:center">
        <h3>Get in Touch</h3>
        <p style="font-size:1.05rem">Interested in MeridianIQ for your fleet operations?</p>
        <p style="color:#888780">Contact us to discuss a pilot program or custom deployment.</p>
        <p style="font-size:1.1rem"><strong style="color:#1D9E75">hello@meridianiq.com</strong></p>
    </div>
    """, unsafe_allow_html=True)
