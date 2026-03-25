import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────
FLEET_SIZE = 10
DAYS_HISTORY = 30
EVENTS_PER_ROBOT_PER_SHIFT = 300
SHIFT_START = "06:00:00"
SHIFT_HOURS = 10
SEED = 42
np.random.seed(SEED)

# Output paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── ROBOT PROFILES ─────────────────────────────────────────
ROBOT_PROFILES = {
    "R1":  {"type": "healthy",       "base_battery_drain": 0.8,  "base_speed": 1.3, "failure_day": None, "degradation_start": None},
    "R2":  {"type": "healthy",       "base_battery_drain": 0.9,  "base_speed": 1.2, "failure_day": None, "degradation_start": None},
    "R3":  {"type": "healthy",       "base_battery_drain": 0.85, "base_speed": 1.4, "failure_day": None, "degradation_start": None},
    "R4":  {"type": "degrading",     "base_battery_drain": 0.9,  "base_speed": 1.2, "failure_day": None, "degradation_start": 15},
    "R5":  {"type": "degrading",     "base_battery_drain": 0.85, "base_speed": 1.1, "failure_day": None, "degradation_start": 22},
    "R6":  {"type": "failure_prone", "base_battery_drain": 1.1,  "base_speed": 1.1, "failure_day": [8, 19], "degradation_start": None},
    "R7":  {"type": "failure_prone", "base_battery_drain": 1.0,  "base_speed": 1.0, "failure_day": [12, 25], "degradation_start": None},
    "R8":  {"type": "healthy",       "base_battery_drain": 0.75, "base_speed": 1.3, "failure_day": None, "degradation_start": None},
    "R9":  {"type": "degrading",     "base_battery_drain": 1.2,  "base_speed": 1.0, "failure_day": None, "degradation_start": 10},
    "R10": {"type": "healthy",       "base_battery_drain": 0.8,  "base_speed": 1.5, "failure_day": None, "degradation_start": None},
}

EVENT_TYPES = [
    "task_completed", "task_completed", "task_completed",
    "robot_started_task", "robot_started_task",
    "charging_started", "charging_completed",
    "path_blocked", "intersection_wait",
    "low_battery_warning"
]

LOCATIONS = [
    (10.0, 5.0), (20.0, 5.0), (30.0, 5.0), (40.0, 5.0),
    (10.0, 15.0), (20.0, 15.0), (30.0, 15.0), (40.0, 15.0),
    (15.0, 10.0), (25.0, 10.0), (35.0, 10.0),  # intersection hotspots
]

# ── HELPER FUNCTIONS ───────────────────────────────────────
def get_degradation_factor(robot_id, day):
    profile = ROBOT_PROFILES[robot_id]
    if profile["type"] == "healthy":
        return 1.0
    if profile["type"] == "failure_prone":
        failure_days = profile["failure_day"] or []
        for fd in failure_days:
            if abs(day - fd) <= 2:
                return 2.5  # significant degradation around failure days
        return 1.1
    if profile["type"] == "degrading":
        start = profile["degradation_start"]
        if start is None or day < start:
            return 1.0
        days_degrading = day - start
        return 1.0 + (days_degrading * 0.08)  # 8% worse per day
    return 1.0

def get_health_score(robot_id, day):
    factor = get_degradation_factor(robot_id, day)
    score = max(0, 100 - ((factor - 1.0) * 60))
    return round(score, 1)

def is_robot_active(robot_id, day):
    if robot_id == "R8" and day < 10:
        return False  # R8 not deployed until day 10
    profile = ROBOT_PROFILES[robot_id]
    failure_days = profile.get("failure_day") or []
    for fd in failure_days:
        if day == fd:
            return False  # down on failure day
    return True

def generate_event(robot_id, day, event_num, battery, deg_factor):
    profile = ROBOT_PROFILES[robot_id]
    base_date = datetime(2025, 1, 1) + timedelta(days=day)
    shift_start = datetime.combine(base_date.date(),
                  datetime.strptime(SHIFT_START, "%H:%M:%S").time())
    timestamp = shift_start + timedelta(seconds=event_num * (SHIFT_HOURS * 3600 / EVENTS_PER_ROBOT_PER_SHIFT))

    # Speed degrades with factor
    speed = max(0.0, profile["base_speed"] / deg_factor + np.random.normal(0, 0.1))

    # Battery drains faster with degradation
    battery_drain = profile["base_battery_drain"] * deg_factor
    battery = max(0, battery - battery_drain + np.random.normal(0, 0.2))

    # Event selection weighted by health
    # Event selection weighted by health
    if deg_factor > 2.0:
        # Severely degraded — deadlocks and blocks only
        event = np.random.choice(["deadlock_detected", "path_blocked", "low_battery_warning"], p=[0.4, 0.4, 0.2])
        speed = 0.0
    elif deg_factor > 1.5:
        # Degrading — more path blocks, no deadlocks
        event = np.random.choice(EVENT_TYPES + ["path_blocked", "path_blocked", "low_battery_warning"])
    else:
        # Healthy — normal events only, never deadlocks
        healthy_events = [e for e in EVENT_TYPES if e != "deadlock_detected"]
        event = np.random.choice(healthy_events)

    # Location — degraded robots tend to get stuck at intersections
    if deg_factor > 1.5:
        loc = LOCATIONS[np.random.randint(8, 11)]  # intersection hotspots
    else:
        loc = LOCATIONS[np.random.randint(0, len(LOCATIONS))]

    return {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "day": day,
        "robot_id": robot_id,
        "x": round(loc[0] + np.random.normal(0, 0.5), 2),
        "y": round(loc[1] + np.random.normal(0, 0.5), 2),
        "speed": round(speed, 2),
        "battery": round(battery, 1),
        "event": event,
        "health_score": get_health_score(robot_id, day),
        "degradation_factor": round(deg_factor, 3),
        "shift": "day"
    }

# ── GENERATE HISTORY ───────────────────────────────────────
def generate_history():
    print("Generating 30-day fleet history...")
    all_events = []
    failure_history = []
    downtime_history = []

    for day in range(DAYS_HISTORY):
        for robot_id in ROBOT_PROFILES:
            if not is_robot_active(robot_id, day):
                # Record downtime
                base_date = datetime(2025, 1, 1) + timedelta(days=day)
                failure_history.append({
                    "robot_id": robot_id,
                    "failure_date": base_date.strftime("%Y-%m-%d"),
                    "day": day,
                    "downtime_hours": round(np.random.uniform(4, 10), 1)
                })
                continue

            deg_factor = get_degradation_factor(robot_id, day)
            battery = 100.0

            for event_num in range(EVENTS_PER_ROBOT_PER_SHIFT):
                event = generate_event(robot_id, day, event_num, battery, deg_factor)
                battery = event["battery"]
                all_events.append(event)

    df = pd.DataFrame(all_events)
    df.to_csv(RAW_DIR / "robot_runs_history.csv", index=False)
    print(f"  Generated {len(df):,} historical events")

    with open(PROCESSED_DIR / "failure_history.json", "w") as f:
        json.dump(failure_history, f, indent=2)
    print(f"  Recorded {len(failure_history)} failure events")

    return df, failure_history

# ── GENERATE TODAY'S SHIFT ─────────────────────────────────
def generate_current_shift():
    print("Generating current shift data...")
    events = []
    today = DAYS_HISTORY  # day 30 = today

    for robot_id in ROBOT_PROFILES:
        if not is_robot_active(robot_id, today):
            continue

        deg_factor = get_degradation_factor(robot_id, today)
        battery = 100.0

        for event_num in range(EVENTS_PER_ROBOT_PER_SHIFT):
            event = generate_event(robot_id, today, event_num, battery, deg_factor)
            battery = event["battery"]
            events.append(event)

    df = pd.DataFrame(events)
    df.to_csv(RAW_DIR / "robot_runs_current.csv", index=False)
    print(f"  Generated {len(df):,} current shift events")
    return df

# ── CALCULATE BASELINES ────────────────────────────────────
def calculate_baselines(history_df):
    print("Calculating fleet baselines...")
    # Use first 10 days as healthy baseline (before degradation starts)
    baseline_data = history_df[history_df["day"] < 10]

    baselines = {}
    for robot_id in ROBOT_PROFILES:
        robot_data = baseline_data[baseline_data["robot_id"] == robot_id]
        if len(robot_data) == 0:
            continue
        baselines[robot_id] = {
            "avg_speed": round(robot_data["speed"].mean(), 3),
            "avg_battery_drain": round(100 - robot_data["battery"].min(), 2),
            "avg_health_score": round(robot_data["health_score"].mean(), 1),
            "task_completion_rate": round(
                len(robot_data[robot_data["event"] == "task_completed"]) / len(robot_data), 3
            )
        }

    with open(PROCESSED_DIR / "fleet_baselines.json", "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"  Baselines calculated for {len(baselines)} robots")
    return baselines

# ── CALCULATE MTBF / MTTR ──────────────────────────────────
def calculate_mtbf_mttr(failure_history):
    print("Calculating MTBF and MTTR...")
    mtbf_mttr = {}

    for robot_id in ROBOT_PROFILES:
        failures = [f for f in failure_history if f["robot_id"] == robot_id]
        if len(failures) < 2:
            mtbf_mttr[robot_id] = {
                "failure_count": len(failures),
                "mtbf_days": None,
                "avg_mttr_hours": round(failures[0]["downtime_hours"], 1) if failures else 0,
                "total_downtime_hours": sum(f["downtime_hours"] for f in failures)
            }
            continue

        failure_days = sorted([f["day"] for f in failures])
        intervals = [failure_days[i+1] - failure_days[i] for i in range(len(failure_days)-1)]
        mtbf = sum(intervals) / len(intervals)
        avg_mttr = sum(f["downtime_hours"] for f in failures) / len(failures)

        mtbf_mttr[robot_id] = {
            "failure_count": len(failures),
            "mtbf_days": round(mtbf, 1),
            "avg_mttr_hours": round(avg_mttr, 1),
            "total_downtime_hours": round(sum(f["downtime_hours"] for f in failures), 1)
        }

    with open(PROCESSED_DIR / "mtbf_mttr.json", "w") as f:
        json.dump(mtbf_mttr, f, indent=2)
    print(f"  MTBF/MTTR calculated for {len(mtbf_mttr)} robots")
    return mtbf_mttr

# ── GENERATE ALERT HISTORY ─────────────────────────────────
def generate_alert_history(history_df):
    print("Generating alert history...")
    alerts = []
    alert_id = 1

    degraded = history_df[
        history_df["event"].isin(["deadlock_detected", "path_blocked", "low_battery_warning"])
    ]

    for _, row in degraded.iterrows():
        severity = "Critical" if row["event"] == "deadlock_detected" else \
                   "High" if row["health_score"] < 50 else "Moderate"
        alerts.append({
            "alert_id": f"ALT{alert_id:04d}",
            "timestamp": row["timestamp"],
            "day": int(row["day"]),
            "robot_id": row["robot_id"],
            "event": row["event"],
            "severity": severity,
            "health_score": row["health_score"],
            "status": "Resolved",
            "resolution_time_minutes": int(np.random.uniform(15, 120))
        })
        alert_id += 1

    with open(PROCESSED_DIR / "alert_history.json", "w") as f:
        json.dump(alerts[:500], f, indent=2)  # cap at 500 for file size
    print(f"  Generated {len(alerts)} historical alerts (saved top 500)")
    return alerts

# ── MAIN ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("OpsIQ Data Generator")
    print("="*50 + "\n")

    history_df, failure_history = generate_history()
    current_df = generate_current_shift()
    baselines = calculate_baselines(history_df)
    mtbf_mttr = calculate_mtbf_mttr(failure_history)
    alert_history = generate_alert_history(history_df)

    print("\n" + "="*50)
    print("GENERATION COMPLETE")
    print("="*50)
    print(f"Historical events:     {len(history_df):,}")
    print(f"Current shift events:  {len(current_df):,}")
    print(f"Failure events:        {len(failure_history)}")
    print(f"Baselines calculated:  {len(baselines)} robots")
    print(f"Alert history:         {len(alert_history)} alerts")
    print("\nFiles written to data/raw/ and data/processed/")