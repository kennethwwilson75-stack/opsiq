import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_server():
    print("\n" + "="*50)
    print("OpsIQ MCP Server — Test")
    print("="*50)

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            print("\n✓ Connected to OpsIQ MCP server")

            # List available tools
            tools = await session.list_tools()
            print(f"\n✓ Tools available: {len(tools.tools)}")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description[:60]}...")

            # Call get_run_metrics
            print("\n→ Calling get_run_metrics...")
            result = await session.call_tool("get_run_metrics", {})
            data = json.loads(result.content[0].text)
            print(f"  Fleet availability: {data.get('fleet_availability', 'N/A')}")
            print(f"  Status: {data.get('status', 'N/A').upper()}")

            # Call detect_deadlocks
            print("\n→ Calling detect_deadlocks...")
            result = await session.call_tool("detect_deadlocks", {})
            data = json.loads(result.content[0].text)
            print(f"  Deadlocked robots: {data.get('deadlocked_robot_count', 0)}")
            print(f"  Status: {data.get('status', 'N/A').upper()}")

            # Call estimate_rul for engine 68
            print("\n→ Calling estimate_rul for engine 68...")
            result = await session.call_tool("estimate_rul", {"engine_id": 68})
            data = json.loads(result.content[0].text)
            print(f"  RUL: {data.get('estimated_rul', 'N/A')} cycles")
            print(f"  Risk level: {data.get('risk_level', 'N/A').upper()}")
            print(f"  Recommendation: {data.get('recommendation', 'N/A')}")

            print("\n" + "="*50)
            print("✓ All MCP tool calls successful")
            print("="*50)

if __name__ == "__main__":
    asyncio.run(test_mcp_server())