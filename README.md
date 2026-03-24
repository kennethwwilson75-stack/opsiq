# OpsIQ — Industrial Fleet Intelligence Platform

Multi-agent AI system for analyzing robot fleet telemetry and 
detecting operational issues in real time.

## What it does
- Ingests warehouse AGV telemetry and NASA CMAPSS sensor data
- Routes data through a 4-agent LangGraph pipeline
- Detects congestion, deadlocks, battery failures, and anomalies
- Delivers findings via dashboard, chat interface, and scheduled reports

## Architecture
- **Planner Agent** — determines analysis steps based on data profile
- **Telemetry Agent** — analyzes raw data using tool functions
- **Root Cause Agent** — explains why problems exist
- **Report Agent** — generates the operations brief

## Tech Stack
Python · LangGraph · Claude API · Streamlit · Langfuse · Pandas

## Data Sources
- NASA CMAPSS Turbofan Dataset (public)
- Synthetic AGV fleet telemetry (generated)

## Status
🔨 Week 1 — Environment and foundations complete

## Author
Kenneth W. Wilson — Agentic AI Architect  
https://github.com/kennethwwilson75-stack/opsiq