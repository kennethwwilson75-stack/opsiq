# A robot telemetry event looks exactly like this
event = {
    "robot_id": "R7",
    "x": 23.4,
    "y": 11.8,
    "speed": 1.3,
    "battery": 78,
    "event": "path_blocked"
}

print(event["robot_id"])        # access a value
print(event.get("battery"))     # safer way to access
event["status"] = "flagged"     # add a new key
print(event)                    # print the whole thing
# A fleet of robots is just a list of dicts
fleet = [
    {"robot_id": "R1", "battery": 92, "event": "task_completed"},
    {"robot_id": "R2", "battery": 45, "event": "path_blocked"},
    {"robot_id": "R3", "battery": 12, "event": "charging_started"},
]

# Loop through and print each robot
for robot in fleet:
    print(robot["robot_id"], "—", robot["event"])

# Find robots with low battery
for robot in fleet:
    if robot["battery"] < 50:
        print(robot["robot_id"], "has low battery:", robot["battery"])
# This is the exact pattern your analysis_tools.py uses in Week 3
def get_run_metrics(fleet):
    total = len(fleet)
    low_battery = [r for r in fleet if r["battery"] < 50]
    blocked = [r for r in fleet if r["event"] == "path_blocked"]
    
    return {
        "total_robots": total,
        "low_battery_count": len(low_battery),
        "blocked_count": len(blocked)
    }

result = get_run_metrics(fleet)
print(result)

import pandas as pd

# Create a tiny test CSV first
import csv
with open("test_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["robot_id", "battery", "event"])
    writer.writerow(["R1", 92, "task_completed"])
    writer.writerow(["R2", 45, "path_blocked"])
    writer.writerow(["R3", 12, "charging_started"])

# Now read it back with pandas
df = pd.read_csv("test_data.csv")
print(df)
print(df.describe())
print(df[df["battery"] < 50])