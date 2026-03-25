import pandas as pd
import numpy as np
import json
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# CMAPSS column names — no headers in the raw files
CMAPSS_COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21"
]

# ── LOAD AGV DATA ───────────────────────────────────────────
def load_agv_history():
    print("Loading AGV history...")
    df = pd.read_csv(RAW_DIR / "robot_runs_history.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  Loaded {len(df):,} historical events")
    print(f"  Robots: {sorted(df['robot_id'].unique())}")
    print(f"  Days: {df['day'].min()} to {df['day'].max()}")
    return df

def load_agv_current():
    print("Loading current shift data...")
    df = pd.read_csv(RAW_DIR / "robot_runs_current.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  Loaded {len(df):,} current shift events")
    return df

# ── LOAD CMAPSS DATA ────────────────────────────────────────
def load_cmapss_train():
    print("Loading NASA CMAPSS training data...")
    df = pd.read_csv(
        RAW_DIR / "train_FD001.txt",
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS
    )

    # Drop last two columns — always NaN in FD001
    df = df.dropna(axis=1, how="all")

    # Add RUL — remaining useful life
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycles, on="engine_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df = df.drop("max_cycle", axis=1)

    print(f"  Loaded {len(df):,} CMAPSS records")
    print(f"  Engines: {df['engine_id'].nunique()}")
    print(f"  Max cycles: {df['cycle'].max()}")
    print(f"  RUL range: {df['rul'].min()} to {df['rul'].max()} cycles")
    return df

# ── NORMALIZE CMAPSS SENSORS ────────────────────────────────
def normalize_cmapss(df):
    print("Normalizing CMAPSS sensor data...")
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    # Drop sensors with near-zero variance
    useful_sensors = []
    dropped_sensors = []
    for col in sensor_cols:
        if df[col].std() > 0.01:
            useful_sensors.append(col)
        else:
            dropped_sensors.append(col)

    print(f"  Useful sensors: {len(useful_sensors)}")
    print(f"  Dropped (no variance): {dropped_sensors}")

    # Min-max normalize
    df_norm = df.copy()
    for col in useful_sensors:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            df_norm[col] = (df[col] - col_min) / (col_max - col_min)

    # Save normalized data
    df_norm.to_csv(PROCESSED_DIR / "cmapss_normalized.csv", index=False)
    print(f"  Saved normalized data to data/processed/cmapss_normalized.csv")

    return df_norm, useful_sensors

# ── CALCULATE ENGINE HEALTH PROFILES ───────────────────────
def calculate_engine_profiles(df_norm, useful_sensors):
    print("Calculating engine health profiles...")
    profiles = {}

    for engine_id in df_norm["engine_id"].unique():
        engine_data = df_norm[df_norm["engine_id"] == engine_id].sort_values("cycle")
        total_cycles = int(engine_data["cycle"].max())

        # Simulate engine at 70-90% of life — currently running, not yet failed
        # Varies per engine so we get a spread of RUL values
        snapshot_pct = 0.30 + (int(engine_id) % 70) * 0.01
        snapshot_cycle = int(total_cycles * snapshot_pct)
        snapshot_data = engine_data[engine_data["cycle"] <= snapshot_cycle]
        current_cycle = int(snapshot_data["cycle"].max())
        actual_rul = int(snapshot_data["rul"].min())

        # Health score based on remaining life percentage
        health_score = round((actual_rul / total_cycles) * 100, 1)

        # Degradation rate — slope of sensor trends in last 20 cycles
        recent = snapshot_data.tail(20)
        early = snapshot_data.head(20)

        sensor_deltas = {}
        for s in useful_sensors[:6]:
            early_mean = early[s].mean()
            recent_mean = recent[s].mean()
            sensor_deltas[s] = round(recent_mean - early_mean, 4)

        profiles[str(engine_id)] = {
            "engine_id": int(engine_id),
            "total_cycles": total_cycles,
            "current_cycle": current_cycle,
            "current_rul": actual_rul,
            "health_score": health_score,
            "sensor_deltas": sensor_deltas,
            "degradation_rate": round(sum(abs(v) for v in sensor_deltas.values()), 4)
        }

    # Save profiles
    with open(PROCESSED_DIR / "engine_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"  Profiles calculated for {len(profiles)} engines")

    # Show top 5 most at-risk
    at_risk = [p for p in profiles.values() if 0 < p["current_rul"] <= 100]
    at_risk_sorted = sorted(at_risk, key=lambda x: x["current_rul"])
    print(f"\n  Engines with RUL <= 100 cycles: {len(at_risk)}")
    print("  Top 5 most at-risk:")
    for p in at_risk_sorted[:5]:
        print(f"    Engine {p['engine_id']:3d}: RUL={p['current_rul']:3d} cycles, "
              f"health={p['health_score']:5.1f}%, "
              f"cycle={p['current_cycle']}/{p['total_cycles']}")

    return profiles

# ── LOAD PROCESSED FILES ────────────────────────────────────
def load_processed_files():
    print("Loading processed support files...")
    with open(PROCESSED_DIR / "fleet_baselines.json") as f:
        baselines = json.load(f)
    with open(PROCESSED_DIR / "mtbf_mttr.json") as f:
        mtbf_mttr = json.load(f)
    with open(PROCESSED_DIR / "failure_history.json") as f:
        failure_history = json.load(f)
    with open(PROCESSED_DIR / "alert_history.json") as f:
        alert_history = json.load(f)
    print(f"  Baselines: {len(baselines)} robots")
    print(f"  MTBF/MTTR: {len(mtbf_mttr)} robots")
    print(f"  Failure history: {len(failure_history)} events")
    print(f"  Alert history: {len(alert_history)} alerts")
    return baselines, mtbf_mttr, failure_history, alert_history

# ── MAIN ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("OpsIQ Data Loader")
    print("="*50 + "\n")

    # Load all data sources
    agv_history = load_agv_history()
    print()
    agv_current = load_agv_current()
    print()
    cmapss_raw = load_cmapss_train()
    print()
    cmapss_norm, useful_sensors = normalize_cmapss(cmapss_raw)
    print()
    engine_profiles = calculate_engine_profiles(cmapss_norm, useful_sensors)
    print()
    baselines, mtbf_mttr, failure_history, alert_history = load_processed_files()

    print("\n" + "="*50)
    print("ALL DATA LOADED SUCCESSFULLY")
    print("="*50)
    print(f"\nAGV history:        {len(agv_history):,} events")
    print(f"AGV current shift:  {len(agv_current):,} events")
    print(f"CMAPSS records:     {len(cmapss_raw):,} records")
    print(f"CMAPSS engines:     {cmapss_raw['engine_id'].nunique()}")
    print(f"Useful sensors:     {len(useful_sensors)}")
    print(f"Engine profiles:    {len(engine_profiles)}")
    print(f"\nData layer is ready for agent ingestion.")