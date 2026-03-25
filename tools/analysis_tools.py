import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# ── CONFIG ─────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")

# Alert thresholds
BATTERY_LOW = 15
BATTERY_CRITICAL = 5
SPEED_STOPPED = 0.1
HEALTH_WARNING = 70
HEALTH_CRITICAL = 40
RUL_WARNING = 50
RUL_CRITICAL = 20
CONGESTION_WAIT_THRESHOLD = 3  # 3+ robots at same location = congestion

# ── HELPER ─────────────────────────────────────────────────
def load_support_files():
    with open(PROCESSED_DIR / "fleet_baselines.json") as f:
        baselines = json.load(f)
    with open(PROCESSED_DIR / "mtbf_mttr.json") as f:
        mtbf_mttr = json.load(f)
    with open(PROCESSED_DIR / "engine_profiles.json") as f:
        engine_profiles = json.load(f)
    with open(PROCESSED_DIR / "alert_history.json") as f:
        alert_history = json.load(f)
    return baselines, mtbf_mttr, engine_profiles, alert_history

# ══════════════════════════════════════════════════════════
# FUNCTION 1 — get_run_metrics
# ══════════════════════════════════════════════════════════
def get_run_metrics(df):
    """
    Overall fleet performance metrics for a given run dataset.
    Returns a structured summary dict.
    """
    total_events = len(df)
    total_robots = df["robot_id"].nunique()
    
    # Task completion rate
    completed = len(df[df["event"] == "task_completed"])
    started = len(df[df["event"] == "robot_started_task"])
    completion_rate = round(completed / started, 3) if started > 0 else 0
    
    # Problem events
    deadlocks = len(df[df["event"] == "deadlock_detected"])
    blocked = len(df[df["event"] == "path_blocked"])
    low_battery = len(df[df["event"] == "low_battery_warning"])
    
    # Fleet availability — robots with any task_completed vs total
    active_robots = df[df["event"] == "task_completed"]["robot_id"].nunique()
    availability = round(active_robots / total_robots, 3) if total_robots > 0 else 0
    
    # Average health score
    avg_health = round(df["health_score"].mean(), 1)
    
    # Downtime estimate — events where speed = 0 and not charging
    stopped = df[
        (df["speed"] <= SPEED_STOPPED) &
        (~df["event"].isin(["charging_started", "charging_completed"]))
    ]
    downtime_pct = round(len(stopped) / total_events, 3) if total_events > 0 else 0
    
    return {
        "function": "get_run_metrics",
        "total_events": total_events,
        "total_robots": total_robots,
        "active_robots": active_robots,
        "fleet_availability": availability,
        "avg_health_score": avg_health,
        "task_completion_rate": completion_rate,
        "deadlock_count": deadlocks,
        "path_blocked_count": blocked,
        "low_battery_count": low_battery,
        "downtime_pct": downtime_pct,
        "status": "critical" if availability < 0.7 else "warning" if availability < 0.85 else "normal"
    }

# ══════════════════════════════════════════════════════════
# FUNCTION 2 — detect_congestion
# ══════════════════════════════════════════════════════════
def detect_congestion(df):
    """
    Identifies location hotspots where multiple robots slow or stop.
    Returns congestion zones with severity.
    """
    # Round coordinates to create location buckets
    df = df.copy()
    df["loc_x"] = df["x"].round(0)
    df["loc_y"] = df["y"].round(0)
    
    # Find locations with path_blocked or intersection_wait events
    problem_events = df[df["event"].isin(["path_blocked", "intersection_wait", "deadlock_detected"])]
    
    if len(problem_events) == 0:
        return {
            "function": "detect_congestion",
            "congestion_detected": False,
            "hotspots": [],
            "total_congestion_events": 0,
            "status": "normal"
        }
    
    # Group by location
    location_counts = problem_events.groupby(["loc_x", "loc_y"]).agg(
        event_count=("event", "count"),
        robots_affected=("robot_id", "nunique"),
        avg_speed=("speed", "mean"),
        events=("event", lambda x: x.value_counts().to_dict())
    ).reset_index()
    
    # Filter to significant hotspots
    hotspots = location_counts[
        location_counts["robots_affected"] >= 1
    ].sort_values("event_count", ascending=False)
    
    hotspot_list = []
    for _, row in hotspots.head(5).iterrows():
        severity = "critical" if row["robots_affected"] >= 3 else \
                   "high" if row["robots_affected"] >= 2 else "moderate"
        hotspot_list.append({
            "location": f"({row['loc_x']}, {row['loc_y']})",
            "event_count": int(row["event_count"]),
            "robots_affected": int(row["robots_affected"]),
            "avg_speed": round(row["avg_speed"], 2),
            "severity": severity
        })
    
    return {
        "function": "detect_congestion",
        "congestion_detected": len(hotspot_list) > 0,
        "hotspot_count": len(hotspot_list),
        "hotspots": hotspot_list,
        "total_congestion_events": int(len(problem_events)),
        "status": "critical" if any(h["severity"] == "critical" for h in hotspot_list) else
                  "warning" if len(hotspot_list) > 0 else "normal"
    }

# ══════════════════════════════════════════════════════════
# FUNCTION 3 — detect_deadlocks
# ══════════════════════════════════════════════════════════
def detect_deadlocks(df):
    """
    Identifies robots that are stationary or in deadlock state.
    Uses position change and speed analysis.
    """
    deadlock_events = df[df["event"] == "deadlock_detected"]

    # Position-based detection — robots that haven't moved
    position_analysis = df.groupby("robot_id").agg(
        x_range=("x", lambda x: x.max() - x.min()),
        y_range=("y", lambda y: y.max() - y.min()),
        avg_speed=("speed", "mean"),
        min_battery=("battery", "min"),
        avg_health=("health_score", "mean"),
        event_count=("event", "count")
    )

    # Only flag as stuck if robot also has explicit problem events
    problem_robots = df[
        df["event"].isin(["deadlock_detected", "path_blocked"])
    ]["robot_id"].unique()

    stuck = position_analysis[
        (position_analysis["x_range"] < 0.2) &
        (position_analysis["avg_speed"] < SPEED_STOPPED) &
        (position_analysis.index.isin(problem_robots))
    ]

    deadlocked_robots = []

    # From explicit deadlock events
    for robot_id in deadlock_events["robot_id"].unique():
        robot_data = df[df["robot_id"] == robot_id]
        deadlocked_robots.append({
            "robot_id": robot_id,
            "detection_method": "event_based",
            "deadlock_events": int(len(deadlock_events[deadlock_events["robot_id"] == robot_id])),
            "avg_speed": round(robot_data["speed"].mean(), 2),
            "battery": round(robot_data["battery"].min(), 1),
            "health_score": round(robot_data["health_score"].mean(), 1),
            "severity": "critical"
        })

    # From position analysis — catch robots not flagged by events
    for robot_id in stuck.index:
        if robot_id not in [r["robot_id"] for r in deadlocked_robots]:
            deadlocked_robots.append({
                "robot_id": robot_id,
                "detection_method": "position_based",
                "deadlock_events": 0,
                "avg_speed": round(stuck.loc[robot_id, "avg_speed"], 2),
                "battery": round(stuck.loc[robot_id, "min_battery"], 1),
                "health_score": round(stuck.loc[robot_id, "avg_health"], 1),
                "severity": "high"
            })

    return {
        "function": "detect_deadlocks",
        "deadlocks_detected": len(deadlocked_robots) > 0,
        "deadlocked_robot_count": len(deadlocked_robots),
        "deadlocked_robots": deadlocked_robots,
        "status": "critical" if any(r["severity"] == "critical" for r in deadlocked_robots) else
                  "warning" if len(deadlocked_robots) > 0 else "normal"
    }

# ══════════════════════════════════════════════════════════
# FUNCTION 4 — battery_analysis
# ══════════════════════════════════════════════════════════
def battery_analysis(df):
    """
    Analyzes battery health and drain rates across the fleet.
    Flags robots at risk of running out during shift.
    """
    battery_stats = df.groupby("robot_id").agg(
        current_battery=("battery", "last"),
        min_battery=("battery", "min"),
        avg_battery=("battery", "mean"),
        battery_start=("battery", "first"),
        low_battery_events=("event", lambda x: (x == "low_battery_warning").sum())
    ).reset_index()
    
    battery_stats["drain_rate"] = round(
        battery_stats["battery_start"] - battery_stats["current_battery"], 1
    )
    
    alerts = []
    for _, row in battery_stats.iterrows():
        if row["min_battery"] <= BATTERY_CRITICAL:
            alerts.append({
                "robot_id": row["robot_id"],
                "current_battery": round(row["current_battery"], 1),
                "min_battery": round(row["min_battery"], 1),
                "drain_rate": row["drain_rate"],
                "low_battery_events": int(row["low_battery_events"]),
                "severity": "critical"
            })
        elif row["min_battery"] <= BATTERY_LOW:
            alerts.append({
                "robot_id": row["robot_id"],
                "current_battery": round(row["current_battery"], 1),
                "min_battery": round(row["min_battery"], 1),
                "drain_rate": row["drain_rate"],
                "low_battery_events": int(row["low_battery_events"]),
                "severity": "warning"
            })
    
    return {
        "function": "battery_analysis",
        "robots_analyzed": len(battery_stats),
        "critical_battery_count": len([a for a in alerts if a["severity"] == "critical"]),
        "warning_battery_count": len([a for a in alerts if a["severity"] == "warning"]),
        "battery_alerts": alerts,
        "fleet_avg_battery": round(battery_stats["avg_battery"].mean(), 1),
        "status": "critical" if any(a["severity"] == "critical" for a in alerts) else
                  "warning" if len(alerts) > 0 else "normal"
    }

# ══════════════════════════════════════════════════════════
# FUNCTION 5 — detect_sensor_anomaly
# ══════════════════════════════════════════════════════════
def detect_sensor_anomaly(cmapss_df, engine_id):
    """
    Compares current sensor readings to healthy baseline.
    Flags sensors that have drifted significantly.
    """
    engine_data = cmapss_df[cmapss_df["engine_id"] == engine_id].sort_values("cycle")
    
    if len(engine_data) == 0:
        return {"function": "detect_sensor_anomaly", "error": f"Engine {engine_id} not found"}
    
    sensor_cols = [c for c in cmapss_df.columns if c.startswith("sensor_")]
    useful_sensors = [s for s in sensor_cols if cmapss_df[s].std() > 0.01]
    
    # Baseline = first 20% of cycles for this engine
    baseline_cycles = int(len(engine_data) * 0.2)
    baseline = engine_data.head(baseline_cycles)
    recent = engine_data.tail(20)
    
    anomalies = []
    for sensor in useful_sensors:
        baseline_mean = baseline[sensor].mean()
        recent_mean = recent[sensor].mean()
        baseline_std = baseline[sensor].std()
        
        if baseline_std == 0:
            continue
            
        # Z-score style deviation
        deviation = abs(recent_mean - baseline_mean) / baseline_std
        
        if deviation > 2.0:
            anomalies.append({
                "sensor": sensor,
                "baseline_mean": round(baseline_mean, 4),
                "recent_mean": round(recent_mean, 4),
                "deviation_score": round(deviation, 2),
                "direction": "increasing" if recent_mean > baseline_mean else "decreasing",
                "severity": "critical" if deviation > 3.0 else "warning"
            })
    
    anomalies.sort(key=lambda x: x["deviation_score"], reverse=True)
    
    return {
        "function": "detect_sensor_anomaly",
        "engine_id": int(engine_id),
        "total_cycles": int(engine_data["cycle"].max()),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies[:5],  # top 5 most deviant sensors
        "status": "critical" if any(a["severity"] == "critical" for a in anomalies) else
                  "warning" if len(anomalies) > 0 else "normal"
    }

# ══════════════════════════════════════════════════════════
# FUNCTION 6 — estimate_rul
# ══════════════════════════════════════════════════════════
def estimate_rul(cmapss_df, engine_id):
    """
    Estimates remaining useful life for a given engine.
    Uses degradation rate and historical patterns.
    """
    engine_data = cmapss_df[cmapss_df["engine_id"] == engine_id].sort_values("cycle")
    
    if len(engine_data) == 0:
        return {"function": "estimate_rul", "error": f"Engine {engine_id} not found"}
    
    current_cycle = int(engine_data["cycle"].max())
    actual_rul = int(engine_data["rul"].min())
    
    # Mean engine lifespan from fleet history
    mean_lifespan = int(cmapss_df.groupby("engine_id")["cycle"].max().mean())
    
    # Simple RUL estimate — remaining cycles based on mean lifespan
    estimated_rul = max(0, mean_lifespan - current_cycle)
    
    # Confidence based on how close to mean lifespan
    if current_cycle < mean_lifespan * 0.5:
        confidence = "low"      # early in life, hard to predict
    elif current_cycle < mean_lifespan * 0.75:
        confidence = "medium"
    else:
        confidence = "high"     # near end of life, pattern is clear
    
    # Risk level
    if estimated_rul <= RUL_CRITICAL:
        risk = "critical"
        recommendation = "Schedule immediate maintenance. Failure imminent."
    elif estimated_rul <= RUL_WARNING:
        risk = "high"
        recommendation = "Plan maintenance within next 2 shifts."
    elif estimated_rul <= 100:
        risk = "moderate"
        recommendation = "Monitor closely. Schedule maintenance this week."
    else:
        risk = "low"
        recommendation = "No immediate action required."
    
    return {
        "function": "estimate_rul",
        "engine_id": int(engine_id),
        "current_cycle": current_cycle,
        "estimated_rul": estimated_rul,
        "actual_rul": actual_rul,
        "mean_fleet_lifespan": mean_lifespan,
        "confidence": confidence,
        "risk_level": risk,
        "recommendation": recommendation,
        "status": risk
    }

# ══════════════════════════════════════════════════════════
# FUNCTION 7 — get_fleet_health_summary
# ══════════════════════════════════════════════════════════
def get_fleet_health_summary(agv_df, baselines, mtbf_mttr):
    """
    Unified fleet health picture combining AGV telemetry
    with historical baselines and MTBF data.
    """
    robot_summaries = []
    
    for robot_id in agv_df["robot_id"].unique():
        robot_data = agv_df[agv_df["robot_id"] == robot_id]
        baseline = baselines.get(robot_id, {})
        mtbf = mtbf_mttr.get(robot_id, {})
        
        current_health = round(robot_data["health_score"].mean(), 1)
        baseline_health = baseline.get("avg_health_score", 100)
        health_delta = round(current_health - baseline_health, 1)
        
        current_speed = round(robot_data["speed"].mean(), 2)
        baseline_speed = baseline.get("avg_speed", 1.3)
        speed_delta = round(current_speed - baseline_speed, 2)
        
        failure_count = mtbf.get("failure_count", 0)
        mtbf_days = mtbf.get("mtbf_days", None)
        total_downtime = mtbf.get("total_downtime_hours", 0)
        
        # Overall robot status
        if current_health < HEALTH_CRITICAL or health_delta < -30:
            status = "critical"
        elif current_health < HEALTH_WARNING or health_delta < -15:
            status = "warning"
        else:
            status = "normal"
        
        robot_summaries.append({
            "robot_id": robot_id,
            "current_health": current_health,
            "health_delta_from_baseline": health_delta,
            "current_avg_speed": current_speed,
            "speed_delta_from_baseline": speed_delta,
            "failure_count_30d": failure_count,
            "mtbf_days": mtbf_days,
            "total_downtime_hours": total_downtime,
            "status": status
        })
    
    # Fleet-level aggregates
    critical = [r for r in robot_summaries if r["status"] == "critical"]
    warning = [r for r in robot_summaries if r["status"] == "warning"]
    normal = [r for r in robot_summaries if r["status"] == "normal"]
    
    total_downtime = sum(r["total_downtime_hours"] for r in robot_summaries)
    avg_health = round(sum(r["current_health"] for r in robot_summaries) / len(robot_summaries), 1)
    
    return {
        "function": "get_fleet_health_summary",
        "total_robots": len(robot_summaries),
        "critical_count": len(critical),
        "warning_count": len(warning),
        "normal_count": len(normal),
        "fleet_avg_health": avg_health,
        "total_downtime_hours_30d": round(total_downtime, 1),
        "critical_robots": [r["robot_id"] for r in critical],
        "warning_robots": [r["robot_id"] for r in warning],
        "robot_details": robot_summaries,
        "status": "critical" if len(critical) > 0 else
                  "warning" if len(warning) > 0 else "normal"
    }

# ══════════════════════════════════════════════════════════
# FUNCTION 8 — identify_cascade_risks
# ══════════════════════════════════════════════════════════
def identify_cascade_risks(agv_df, deadlock_result, congestion_result):
    """
    Identifies situations where one failure could cause others.
    Cross-references deadlocks with congestion hotspots.
    """
    cascade_risks = []
    
    deadlocked_robots = [r["robot_id"] for r in deadlock_result.get("deadlocked_robots", [])]
    hotspots = congestion_result.get("hotspots", [])
    
    # If deadlocked robots are near congestion hotspots — cascade risk
    for robot_id in deadlocked_robots:
        robot_data = agv_df[agv_df["robot_id"] == robot_id]
        robot_location = f"({robot_data['x'].mean().round(0)}, {robot_data['y'].mean().round(0)})"
        
        # Check if other robots are nearby
        nearby_robots = []
        for other_robot in agv_df["robot_id"].unique():
            if other_robot == robot_id:
                continue
            other_data = agv_df[agv_df["robot_id"] == other_robot]
            other_events = other_data["event"].value_counts().to_dict()
            if other_events.get("path_blocked", 0) > 0 or \
               other_events.get("intersection_wait", 0) > 0:
                nearby_robots.append(other_robot)
        
        if nearby_robots:
            cascade_risks.append({
                "trigger_robot": robot_id,
                "trigger_type": "deadlock",
                "affected_robots": nearby_robots,
                "affected_count": len(nearby_robots),
                "risk_description": f"{robot_id} deadlock is likely blocking {', '.join(nearby_robots)}",
                "severity": "critical" if len(nearby_robots) >= 2 else "high"
            })
    
    # Congestion hotspots with multiple robots
    for hotspot in hotspots:
        if hotspot["robots_affected"] >= 2:
            cascade_risks.append({
                "trigger_robot": None,
                "trigger_type": "congestion_hotspot",
                "location": hotspot["location"],
                "affected_count": hotspot["robots_affected"],
                "risk_description": f"Congestion at {hotspot['location']} affecting {hotspot['robots_affected']} robots",
                "severity": hotspot["severity"]
            })
    
    return {
        "function": "identify_cascade_risks",
        "cascade_risks_detected": len(cascade_risks) > 0,
        "risk_count": len(cascade_risks),
        "cascade_risks": cascade_risks,
        "status": "critical" if any(r["severity"] == "critical" for r in cascade_risks) else
                  "warning" if len(cascade_risks) > 0 else "normal"
    }

# ══════════════════════════════════════════════════════════
# TEST — run all functions against current shift data
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import pandas as pd

    print("\n" + "="*55)
    print("OpsIQ Analysis Tools — Test Run")
    print("="*55)

    # Load data
    agv_df = pd.read_csv("data/raw/robot_runs_current.csv")
    cmapss_df = pd.read_csv("data/processed/cmapss_normalized.csv")
    baselines, mtbf_mttr, engine_profiles, alert_history = load_support_files()

    print(f"\nData loaded: {len(agv_df)} AGV events, {len(cmapss_df)} CMAPSS records\n")

    # Run all 8 functions
    print("─"*55)
    metrics = get_run_metrics(agv_df)
    print(f"1. get_run_metrics")
    print(f"   Fleet availability: {metrics['fleet_availability']:.0%}")
    print(f"   Avg health score:   {metrics['avg_health_score']}")
    print(f"   Deadlocks:          {metrics['deadlock_count']}")
    print(f"   Status:             {metrics['status'].upper()}")

    print("─"*55)
    congestion = detect_congestion(agv_df)
    print(f"2. detect_congestion")
    print(f"   Hotspots found:     {congestion['hotspot_count']}")
    print(f"   Total events:       {congestion['total_congestion_events']}")
    print(f"   Status:             {congestion['status'].upper()}")

    print("─"*55)
    deadlocks = detect_deadlocks(agv_df)
    print(f"3. detect_deadlocks")
    print(f"   Deadlocked robots:  {deadlocks['deadlocked_robot_count']}")
    print(f"   Robots:             {[r['robot_id'] for r in deadlocks['deadlocked_robots']]}")
    print(f"   Status:             {deadlocks['status'].upper()}")

    print("─"*55)
    battery = battery_analysis(agv_df)
    print(f"4. battery_analysis")
    print(f"   Fleet avg battery:  {battery['fleet_avg_battery']}%")
    print(f"   Critical alerts:    {battery['critical_battery_count']}")
    print(f"   Warning alerts:     {battery['warning_battery_count']}")
    print(f"   Status:             {battery['status'].upper()}")

    print("─"*55)
    anomaly = detect_sensor_anomaly(cmapss_df, engine_id=50)
    print(f"5. detect_sensor_anomaly (Engine 50)")
    print(f"   Anomalies found:    {anomaly['anomaly_count']}")
    print(f"   Status:             {anomaly['status'].upper()}")

    print("─"*55)
    rul = estimate_rul(cmapss_df, engine_id=50)
    print(f"6. estimate_rul (Engine 50)")
    print(f"   Current cycle:      {rul['current_cycle']}")
    print(f"   Estimated RUL:      {rul['estimated_rul']} cycles")
    print(f"   Confidence:         {rul['confidence']}")
    print(f"   Recommendation:     {rul['recommendation']}")
    print(f"   Status:             {rul['status'].upper()}")

    print("─"*55)
    health_summary = get_fleet_health_summary(agv_df, baselines, mtbf_mttr)
    print(f"7. get_fleet_health_summary")
    print(f"   Total robots:       {health_summary['total_robots']}")
    print(f"   Critical:           {health_summary['critical_count']}")
    print(f"   Warning:            {health_summary['warning_count']}")
    print(f"   Normal:             {health_summary['normal_count']}")
    print(f"   Fleet avg health:   {health_summary['fleet_avg_health']}")
    print(f"   Total downtime 30d: {health_summary['total_downtime_hours_30d']} hrs")
    print(f"   Status:             {health_summary['status'].upper()}")

    print("─"*55)
    cascade = identify_cascade_risks(agv_df, deadlocks, congestion)
    print(f"8. identify_cascade_risks")
    print(f"   Cascade risks:      {cascade['risk_count']}")
    print(f"   Status:             {cascade['status'].upper()}")
    if cascade["cascade_risks"]:
        for risk in cascade["cascade_risks"][:2]:
            print(f"   → {risk['risk_description']}")

    print("\n" + "="*55)
    print("ALL 8 TOOLS OPERATIONAL")
    print("="*55)