import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are OpsIQ, an expert operations analyst for industrial robot fleets.

Your job is to analyze telemetry data and provide clear, actionable intelligence to operations managers.

You always:
- Prioritize by severity (Critical → High → Moderate)
- Give a specific recommendation, not general advice
- Flag cascade risks when one failure could cause others
- Keep responses concise — operations managers are busy

You respond in plain text only — no markdown, no tables, no bullet symbols.
Just clear numbered lists and short paragraphs."""

# A small fleet with a problem baked in
fleet = [
    {"robot_id": "R1", "battery": 92, "speed": 1.3, "event": "task_completed"},
    {"robot_id": "R2", "battery": 45, "speed": 0.2, "event": "path_blocked"},
    {"robot_id": "R3", "battery": 12, "speed": 0.0, "event": "deadlock_detected"},
    {"robot_id": "R4", "battery": 88, "speed": 1.4, "event": "task_completed"},
    {"robot_id": "R5", "battery": 31, "speed": 0.1, "event": "path_blocked"},
]

# Convert fleet data to a readable string for Claude
fleet_summary = "\n".join([
    f"Robot {r['robot_id']}: battery={r['battery']}%, speed={r['speed']}, event={r['event']}"
    for r in fleet
])

print("Fleet data being sent to Claude:")
print(fleet_summary)
print("\n" + "="*50 + "\n")

# Send it to Claude with a clear instruction
message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=SYSTEM_PROMPT,
    messages=[
        {
            "role": "user",
            "content": f"""You are an operations analyst for a warehouse robot fleet.
            
Analyze this telemetry data and identify operational issues:

{fleet_summary}

Provide:
1. A summary of fleet health
2. Any robots that need immediate attention
3. Your top recommendation
Keep your response concise and actionable."""
        }
    ]
)

print("Claude's analysis:")
print(message.content[0].text)