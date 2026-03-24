import pandas as pd
import os

# Build a more realistic fleet dataset
data = {
    "timestamp": ["10:21:03","10:21:08","10:21:15","10:21:22","10:21:30",
                  "10:21:35","10:21:42","10:21:50","10:21:55","10:22:01"],
    "robot_id": ["R1","R2","R3","R1","R2","R3","R1","R2","R3","R1"],
    "x":        [12.1, 23.4, 8.7, 12.3, 23.4, 8.7, 12.6, 23.4, 8.7, 12.9],
    "y":        [5.2,  11.8, 3.1, 5.4,  11.8, 3.1, 5.7,  11.8, 3.1, 6.0],
    "speed":    [1.2,  1.3,  0.0, 1.1,  0.0,  0.0, 1.3,  0.0,  0.0, 1.2],
    "battery":  [88,   72,   45,  87,   71,   44,  86,   70,   43,  85],
    "event":    ["task_completed","task_completed","path_blocked",
                 "task_completed","path_blocked","path_blocked",
                 "task_completed","deadlock_detected","deadlock_detected",
                 "task_completed"]
}

df = pd.DataFrame(data)
print("=== RAW DATA ===")
print(df)
print()
print("=== FILTERING ===")

# Find all blocked or deadlocked robots
problems = df[df["event"].isin(["path_blocked", "deadlock_detected"])]
print("Problem events:")
print(problems[["timestamp", "robot_id", "event", "battery"]])
print()

# Count events by type
print("Event counts:")
print(df["event"].value_counts())
print()

# Which robots appear in problems?
print("Robots with issues:", problems["robot_id"].unique())
print()
print("=== GROUPBY ===")

# Average battery per robot
avg_battery = df.groupby("robot_id")["battery"].mean()
print("Average battery by robot:")
print(avg_battery)
print()

# Count events per robot
event_counts = df.groupby("robot_id")["event"].count()
print("Event count by robot:")
print(event_counts)
print()

# Average speed per robot — low speed = potential problem
avg_speed = df.groupby("robot_id")["speed"].mean().round(2)
print("Average speed by robot:")
print(avg_speed)
print()

print("=== STATIONARY DETECTION ===")

# A robot that hasn't moved across multiple readings is stuck
# Compare x/y position — if same across rows, it hasn't moved
position_changes = df.groupby("robot_id").agg(
    x_range=("x", lambda x: x.max() - x.min()),
    y_range=("y", lambda y: y.max() - y.min()),
    avg_speed=("speed", "mean")
)

print("Position range and avg speed per robot:")
print(position_changes)
print()

# Flag robots that haven't moved
stuck = position_changes[
    (position_changes["x_range"] < 0.1) & 
    (position_changes["avg_speed"] < 0.1)
]
print("Potentially stuck robots:")
print(stuck)