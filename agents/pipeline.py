import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import uuid
import pandas as pd
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import OpsIQState
from .telemetry_agent import run_telemetry_agent
from .fault_agent import run_fault_agent
from .alerting_agent import run_alerting_agent
from .reporting_agent import run_reporting_agent

# ── CONDITIONAL ROUTING ─────────────────────────────────────
def route_after_telemetry(state: OpsIQState) -> str:
    """
    After Telemetry Agent runs decide next step.
    If fleet is normal skip Fault Agent — no point running
    expensive CMAPSS analysis on a healthy fleet.
    If critical or warning run full pipeline.
    """
    if state.telemetry is None:
        print("[Pipeline] Telemetry output missing — routing to reporting")
        return "reporting"

    status = state.telemetry.status
    print(f"[Pipeline] Telemetry status: {status.upper()} — routing to: "
          f"{'fault' if status in ['critical', 'warning'] else 'reporting'}")

    if status in ["critical", "warning"]:
        return "fault"
    else:
        return "reporting"

# ── BUILD THE GRAPH ─────────────────────────────────────────
def build_pipeline() -> StateGraph:
    """
    Assembles the OpsIQ LangGraph pipeline.
    Returns a compiled app ready for invocation.
    """
    print("[Pipeline] Building OpsIQ agent graph...")

    # Initialize graph with state schema
    graph = StateGraph(OpsIQState)

    # ── REGISTER NODES ──────────────────────────────────────
    graph.add_node("telemetry", run_telemetry_agent)
    graph.add_node("fault", run_fault_agent)
    graph.add_node("alerting", run_alerting_agent)
    graph.add_node("reporting", run_reporting_agent)

    # ── SET ENTRY POINT ─────────────────────────────────────
    graph.set_entry_point("telemetry")

    # ── DEFINE EDGES ────────────────────────────────────────
    # Conditional edge after telemetry — critical/warning goes to fault
    # normal skips fault and goes straight to reporting
    graph.add_conditional_edges(
        "telemetry",
        route_after_telemetry,
        {
            "fault": "fault",
            "reporting": "reporting"
        }
    )

    # Linear edges for the rest of the pipeline
    graph.add_edge("fault", "alerting")
    graph.add_edge("alerting", "reporting")
    graph.add_edge("reporting", END)

    # ── ADD MEMORY / CHECKPOINTER ───────────────────────────
    # MemorySaver keeps state in memory during the session
    # In production replace with SqliteSaver or PostgresSaver
# ── COMPILE WITHOUT CHECKPOINTER FOR NOW ───────────────
    # MemorySaver has serialization issues with Pydantic V2 + Python 3.14
    # Will add SqliteSaver with proper serialization in polish week
    app = graph.compile()
    print("[Pipeline] Graph compiled successfully")

    return app

# ── RUN PIPELINE ────────────────────────────────────────────
def run_pipeline(data_source: str = "agv", verbose: bool = True) -> OpsIQState:
    """
    Main entry point for running the full OpsIQ pipeline.
    Loads current shift data, builds initial state,
    invokes the compiled graph, returns final state.
    """
    print("\n" + "="*55)
    print("OpsIQ Pipeline — Starting Run")
    print("="*55)

    # ── LOAD DATA ───────────────────────────────────────────
    print(f"\n[Pipeline] Loading {data_source.upper()} data...")
    df = pd.read_csv("data/raw/robot_runs_current.csv")
    raw_data = df.to_dict(orient="records")
    print(f"[Pipeline] Loaded {len(raw_data)} events")

    # ── BUILD INITIAL STATE ─────────────────────────────────
    run_id = str(uuid.uuid4())[:8]
    initial_state = OpsIQState(
        raw_data=raw_data,
        data_source=data_source,
        analysis_timestamp=datetime.now().isoformat(),
        run_id=run_id
    )
    print(f"[Pipeline] Run ID: {run_id}")

    # ── BUILD AND INVOKE GRAPH ──────────────────────────────
    app = build_pipeline()

    # Thread ID for checkpointer — unique per run
    config = {"configurable": {"thread_id": run_id}}

    print(f"\n[Pipeline] Invoking graph...")
    final_state = app.invoke(initial_state)

    # ── PRINT SUMMARY ───────────────────────────────────────
    print("\n" + "="*55)
    print("OpsIQ Pipeline — Run Complete")
    print("="*55)

    if verbose and final_state.get("report"):
        report = final_state["report"]
        print(f"\nRun ID:             {run_id}")
        print(f"Completed agents:   {final_state.get('completed_agents', [])}")
        print(f"Pipeline errors:    {final_state.get('errors', [])}")
        print(f"Overall status:     {report.status.upper()}")
        print(f"Total alerts:       {final_state['alerts'].escalation_count if final_state.get('alerts') else 'N/A'}")
        print(f"\n{'='*55}")
        print("EXECUTIVE SUMMARY")
        print("="*55)
        print(report.executive_summary)
        print(f"\n{'='*55}")
        print("RECOMMENDATIONS")
        print("="*55)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")
        print(f"\n{'='*55}")
        print("FULL OPERATIONS BRIEF")
        print("="*55)
        print(report.full_report)

    return final_state


# ── MAIN ────────────────────────────────────────────────────
if __name__ == "__main__":
    final_state = run_pipeline(data_source="agv", verbose=True)
    print(f"\n[Pipeline] Pipeline complete.")
    print(f"[Pipeline] Completed agents: {final_state.get('completed_agents', [])}")
    print(f"[Pipeline] Errors: {final_state.get('errors', [])}")