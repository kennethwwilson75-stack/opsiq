from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

# ── TELEMETRY AGENT OUTPUT ─────────────────────────────────
class FleetMetrics(BaseModel):
    total_robots: int
    active_robots: int
    fleet_availability: float
    avg_health_score: float
    task_completion_rate: float
    deadlock_count: int
    path_blocked_count: int
    downtime_pct: float
    status: str

class CongestionResult(BaseModel):
    congestion_detected: bool
    hotspot_count: int
    hotspots: List[Dict[str, Any]]
    total_congestion_events: int
    status: str

class DeadlockResult(BaseModel):
    deadlocks_detected: bool
    deadlocked_robot_count: int
    deadlocked_robots: List[Dict[str, Any]]
    status: str

class BatteryResult(BaseModel):
    robots_analyzed: int
    critical_battery_count: int
    warning_battery_count: int
    battery_alerts: List[Dict[str, Any]]
    fleet_avg_battery: float
    status: str

class TelemetryOutput(BaseModel):
    fleet_metrics: FleetMetrics
    congestion: CongestionResult
    deadlocks: DeadlockResult
    battery: BatteryResult
    summary: str
    status: str

# ── FAULT AGENT OUTPUT ─────────────────────────────────────
class SensorAnomaly(BaseModel):
    engine_id: int
    anomaly_count: int
    anomalies: List[Dict[str, Any]]
    status: str

class RULEstimate(BaseModel):
    engine_id: int
    current_cycle: int
    estimated_rul: int
    confidence: str
    risk_level: str
    recommendation: str
    status: str

class CascadeRisk(BaseModel):
    cascade_risks_detected: bool
    risk_count: int
    cascade_risks: List[Dict[str, Any]]
    status: str

class FaultOutput(BaseModel):
    sensor_anomalies: List[SensorAnomaly]
    rul_estimates: List[RULEstimate]
    cascade_risks: CascadeRisk
    critical_engines: List[int]
    summary: str
    status: str

# ── ALERTING AGENT OUTPUT ──────────────────────────────────
class Alert(BaseModel):
    alert_id: str
    robot_id: Optional[str] = None
    engine_id: Optional[int] = None
    severity: str
    alert_type: str
    description: str
    is_recurring: bool
    recurrence_count: int
    recommended_action: str
    escalate_to: Optional[str] = None

class AlertOutput(BaseModel):
    active_alerts: List[Alert]
    new_alert_count: int
    recurring_alert_count: int
    escalation_count: int
    escalations: List[Dict[str, Any]]
    status: str

# ── REPORTING AGENT OUTPUT ─────────────────────────────────
class ReportOutput(BaseModel):
    executive_summary: str
    critical_findings: List[str]
    recommendations: List[str]
    severity_summary: Dict[str, int]
    full_report: str
    status: str

# ── FULL PIPELINE STATE ────────────────────────────────────
class OpsIQState(BaseModel):
    # Input
    raw_data: List[Dict[str, Any]]
    data_source: str
    analysis_timestamp: str
    run_id: str

    # Agent outputs — None until each agent runs
    telemetry: Optional[TelemetryOutput] = None
    fault: Optional[FaultOutput] = None
    alerts: Optional[AlertOutput] = None
    report: Optional[ReportOutput] = None

    # Pipeline control
    current_agent: str = "orchestrator"
    errors: List[str] = []
    completed_agents: List[str] = []