# Default arguments make functions flexible
def analyze_robot(robot_id, battery, speed, event, threshold=50):
    status = "normal"
    alerts = []
    
    if battery < threshold:
        alerts.append(f"Low battery: {battery}%")
        status = "warning"
    
    if speed == 0.0 and event != "charging_started":
        alerts.append(f"Robot stopped unexpectedly")
        status = "critical"
    
    if event == "deadlock_detected":
        alerts.append("Deadlock detected")
        status = "critical"
    
    return {
        "robot_id": robot_id,
        "status": status,
        "alerts": alerts,
        "alert_count": len(alerts)
    }

# Test it
print(analyze_robot("R1", 92, 1.3, "task_completed"))
print(analyze_robot("R3", 12, 0.0, "deadlock_detected"))
print(analyze_robot("R5", 31, 0.0, "path_blocked"))

# This is exactly the pattern get_fleet_health_summary() will use
def fleet_health_summary(fleet_data):
    results = []
    
    for robot in fleet_data:
        result = analyze_robot(
            robot["robot_id"],
            robot["battery"],
            robot["speed"],
            robot["event"]
        )
        results.append(result)
    
    # Aggregate the results
    critical = [r for r in results if r["status"] == "critical"]
    warnings = [r for r in results if r["status"] == "warning"]
    normal = [r for r in results if r["status"] == "normal"]
    
    return {
        "total_robots": len(results),
        "critical_count": len(critical),
        "warning_count": len(warnings),
        "normal_count": len(normal),
        "critical_robots": [r["robot_id"] for r in critical],
        "all_alerts": [a for r in results for a in r["alerts"]]
    }

# Test fleet
fleet = [
    {"robot_id": "R1", "battery": 92, "speed": 1.3, "event": "task_completed"},
    {"robot_id": "R2", "battery": 45, "speed": 0.2, "event": "path_blocked"},
    {"robot_id": "R3", "battery": 12, "speed": 0.0, "event": "deadlock_detected"},
    {"robot_id": "R4", "battery": 88, "speed": 1.4, "event": "task_completed"},
    {"robot_id": "R5", "battery": 31, "speed": 0.0, "event": "path_blocked"},
]

summary = fleet_health_summary(fleet)
print("\n=== FLEET HEALTH SUMMARY ===")
for key, value in summary.items():
    print(f"{key}: {value}")

    import json

# Save your summary to a file
with open("../data/fleet_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== SAVED TO JSON ===")

# Read it back
with open("../data/fleet_summary.json", "r") as f:
    loaded = json.load(f)

print("Loaded from file:")
print(json.dumps(loaded, indent=2))