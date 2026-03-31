import sys
import json
import pandas as pd
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

sys.path.append(str(Path(__file__).parent))

from tools.analysis_tools import (
    get_run_metrics,
    detect_congestion,
    detect_deadlocks,
    battery_analysis,
    detect_sensor_anomaly,
    estimate_rul,
    identify_cascade_risks,
    get_fleet_health_summary
)

# ── INITIALIZE SERVER ────────────────────────────────────────
server = Server("opsiq-tools")

# ── LOAD DATA ONCE ───────────────────────────────────────────
def load_data():
    df = pd.read_csv("data/raw/robot_runs_current.csv")
    cmapss_df = pd.read_csv("data/processed/cmapss_normalized.csv")
    return df, cmapss_df

# ── REGISTER TOOLS ───────────────────────────────────────────
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_run_metrics",
            description="Analyze current shift AGV telemetry. Returns fleet availability, health score, deadlock count, and overall status.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="detect_congestion",
            description="Identify congestion hotspots in the robot fleet. Returns hotspot locations, event counts, and severity.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="detect_deadlocks",
            description="Detect deadlocked robots that are blocking operations. Returns list of deadlocked robots and severity.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="battery_analysis",
            description="Analyze battery levels across the robot fleet. Returns critical and warning battery alerts per robot.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="get_fleet_health_summary",
            description="Get overall fleet health summary including robot status breakdown and total downtime.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="detect_sensor_anomaly",
            description="Detect sensor anomalies for a specific engine using NASA CMAPSS data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "engine_id": {
                        "type": "integer",
                        "description": "Engine ID to analyze (1-100)"
                    }
                },
                "required": ["engine_id"]
            }
        ),
        types.Tool(
            name="estimate_rul",
            description="Estimate Remaining Useful Life (RUL) for a specific engine. Returns cycles remaining, risk level, and maintenance recommendation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "engine_id": {
                        "type": "integer",
                        "description": "Engine ID to analyze (1-100)"
                    }
                },
                "required": ["engine_id"]
            }
        ),
        types.Tool(
            name="identify_cascade_risks",
            description="Identify cascade failure risks based on current deadlocks and congestion patterns.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
    ]

# ── HANDLE TOOL CALLS ────────────────────────────────────────
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    df, cmapss_df = load_data()

    try:
        if name == "get_run_metrics":
            result = get_run_metrics(df)

        elif name == "detect_congestion":
            result = detect_congestion(df)

        elif name == "detect_deadlocks":
            result = detect_deadlocks(df)

        elif name == "battery_analysis":
            result = battery_analysis(df)

        elif name == "get_fleet_health_summary":
            result = get_fleet_health_summary(df)

        elif name == "detect_sensor_anomaly":
            engine_id = arguments.get("engine_id", 50)
            result = detect_sensor_anomaly(cmapss_df, engine_id)

        elif name == "estimate_rul":
            engine_id = arguments.get("engine_id", 50)
            result = estimate_rul(cmapss_df, engine_id)

        elif name == "identify_cascade_risks":
            deadlocks = detect_deadlocks(df)
            congestion = detect_congestion(df)
            result = identify_cascade_risks(df, deadlocks, congestion)

        else:
            result = {"error": f"Unknown tool: {name}"}

        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]

# ── RUN SERVER ───────────────────────────────────────────────
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())