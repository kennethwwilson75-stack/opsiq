import streamlit as st
import pandas as pd
import json
import uuid
import sys
import os

# Load API key from Streamlit secrets or .env
if "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()
from datetime import datetime
from pathlib import Path



sys.path.append(str(Path(__file__).parent.parent))

from agents.pipeline import run_pipeline, build_pipeline
from agents.state import OpsIQState

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="OpsIQ — Industrial Fleet Intelligence",
    page_icon="⚙️",
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
        border-left: 4px solid #1F4E79;
    }
    .critical { border-left-color: #dc3545; background: #fff5f5; }
    .warning  { border-left-color: #ffc107; background: #fffdf0; }
    .normal   { border-left-color: #28a745; background: #f0fff4; }
    .alert-critical { background: #fff5f5; border-radius: 6px; padding: .75rem; margin: .375rem 0; border-left: 3px solid #dc3545; }
    .alert-high     { background: #fffdf0; border-radius: 6px; padding: .75rem; margin: .375rem 0; border-left: 3px solid #ffc107; }
    .alert-moderate { background: #f0f8ff; border-radius: 6px; padding: .75rem; margin: .375rem 0; border-left: 3px solid #17a2b8; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #1F4E79; margin: 1rem 0 .5rem; }
    .report-text { background: #f8f9fa; border-radius: 8px; padding: 1.5rem; line-height: 1.7; font-size: .95rem; }
    .chat-message-user      { background: #e8f4fd; border-radius: 8px; padding: .75rem 1rem; margin: .5rem 0; }
    .chat-message-assistant { background: #f8f9fa; border-radius: 8px; padding: .75rem 1rem; margin: .5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1F4E79/FFFFFF?text=OpsIQ", width=200)
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Select view",
        ["Live Analysis", "Plant Dashboard", "Executive Report", "Chat Interface"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### About OpsIQ")
    st.markdown("""
    Multi-agent industrial fleet intelligence platform.

    **Data sources:**
    - AGV robot telemetry
    - NASA CMAPSS sensor data

    **Pipeline:**
    Telemetry → Fault → Alerting → Reporting
    """)
    st.markdown("---")
    st.caption("Powered by Claude + LangGraph")

# ── SESSION STATE ───────────────────────────────────────────
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "run_id" not in st.session_state:
    st.session_state.run_id = None

# ── HELPER FUNCTIONS ────────────────────────────────────────
def severity_color(severity: str) -> str:
    return {"critical": "🔴", "high": "🟡", "moderate": "🔵", "normal": "🟢"}.get(severity.lower(), "⚪")

def status_badge(status: str) -> str:
    colors = {"critical": "#dc3545", "warning": "#ffc107", "normal": "#28a745"}
    color = colors.get(status.lower(), "#6c757d")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:600">{status.upper()}</span>'

def run_analysis(df: pd.DataFrame) -> dict:
    """Run the full OpsIQ pipeline against uploaded or default data."""
    with st.spinner("Running OpsIQ pipeline... this takes 30-60 seconds"):
        raw_data = df.to_dict(orient="records")
        run_id = str(uuid.uuid4())[:8]

        initial_state = OpsIQState(
            raw_data=raw_data,
            data_source="agv",
            analysis_timestamp=datetime.now().isoformat(),
            run_id=run_id
        )

        app = build_pipeline()
        result = app.invoke(initial_state)
        st.session_state.pipeline_result = result
        st.session_state.run_id = run_id
        return result

# ── PAGE 1 — LIVE ANALYSIS (LINE MANAGER) ──────────────────
if page == "Live Analysis":
    st.title("⚙️ OpsIQ — Live Fleet Analysis")
    st.caption("For operations managers and line supervisors — what needs attention right now")

    # File upload or use default
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload robot telemetry CSV",
            type=["csv"],
            help="Upload a robot_runs CSV file or use the default current shift data"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_default = st.button(
            "Run with current shift data",
            type="primary",
            use_container_width=True
        )

    # Trigger analysis
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
        st.info("Upload a telemetry file or click 'Run with current shift data' to start analysis")


# ── PAGE 2 — PLANT DASHBOARD (PLANT MANAGER) ───────────────
elif page == "Plant Dashboard":
    st.title("🏭 OpsIQ — Plant Dashboard")
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
    st.title("📊 OpsIQ — Executive Report")
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
    report_text = f"""OPSIQ OPERATIONS REPORT
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
        file_name=f"opsiq_report_{st.session_state.run_id}.txt",
        mime="text/plain"
    )


# ── PAGE 4 — CHAT INTERFACE ─────────────────────────────────
elif page == "Chat Interface":
    st.title("💬 OpsIQ — Ask the Fleet")
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
    chat_context = f"""You are OpsIQ, an industrial fleet intelligence assistant.
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
                <strong>OpsIQ:</strong> {message["content"]}
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

        # Call Claude with full context
        import anthropic
        import os
        from dotenv import load_dotenv
        load_dotenv()

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        messages = [{"role": "user", "content": chat_context + f"\n\nUser question: {user_input}"}]

        # Add history
        for msg in st.session_state.chat_history[:-1]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_input})

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