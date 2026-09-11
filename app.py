"""
Virtual Race Engineer — F1 25 / F1 26 telemetry analyzer (Streamlit)

Upload a tab-separated (or comma) telemetry export matching the F1 game
logger schema. Diagnoses balance / braking / traction with frequency, then
ranks setup changes.

Modes:
  Conservative — from your current car, max 3 knobs, 1 click each
  Aggressive   — from your current car, max 6 knobs, 1–2 clicks
  Best guess   — unconstrained full target sheet from a track-type baseline
                 plus the trace. Allowed to look nothing like the car you ran.

Usage:
  pip install streamlit pandas numpy plotly
  streamlit run app.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SENTINEL = -1.0
FL, FR, RL, RR = 0, 1, 2, 3
V_LOW = 120.0
V_HIGH = 200.0
US_ALPHA_THRESH = 0.03
OS_ALPHA_THRESH = 0.03
LOCK_SLIP = 0.08
SPIN_SLIP = 0.12
STEER_BUSY = 0.35
TIRE_COLD = 70.0
TIRE_HOT = 95.0
PHASE_ENTRY, PHASE_MID, PHASE_EXIT, PHASE_STRAIGHT = "entry", "mid", "exit", "straight"

SETUP_LIMITS: dict[str, dict[str, Any]] = {
    "Front wing": {"field": "wing_setup_0", "min": 0, "max": 50, "step": 1, "unit": "clicks", "source": "user"},
    "Rear wing": {"field": "wing_setup_1", "min": 0, "max": 50, "step": 1, "unit": "clicks", "source": "user"},
    "Front ARB": {"field": "arb_setup_0", "min": 1, "max": 21, "step": 1, "unit": "clicks", "source": "user"},
    "Rear ARB": {"field": "arb_setup_1", "min": 1, "max": 21, "step": 1, "unit": "clicks", "source": "user"},
    "On-throttle differential": {
        "field": "diff_onThrottle_setup", "min": 0.10, "max": 1.00, "step": 0.05,
        "unit": "fraction (UI % = value×100)", "display_as_pct": True, "source": "user",
    },
    "Off-throttle differential": {
        "field": "diff_offThrottle_setup", "min": 0.10, "max": 1.00, "step": 0.05,
        "unit": "fraction (UI % = value×100)", "display_as_pct": True, "source": "user",
    },
    "Brake bias (% front)": {
        "field": "front_brake_bias", "alt_fields": ["brake_bias_setup"],
        "min": 0.50, "max": 0.70, "step": 0.01,
        "unit": "% front (70% = more forward, 50% = more rearward)",
        "display_as_pct": True,
        "min_label": "more rearward 50%", "max_label": "more forward 70%",
        "increase_means": "more forward (toward 70% front)",
        "decrease_means": "more rearward (toward 50% front)",
        "source": "user",
    },
    "Brake pressure": {
        "field": "brake_press_setup", "min": 0.80, "max": 1.00, "step": 0.01,
        "unit": "percent", "display_as_pct": True, "min_label": "80%", "max_label": "100%", "source": "user",
    },
    "Front tire pressure": {
        "field": "tyre_press_setup_0", "alt_fields": ["tyre_press_setup_1"],
        "min": 22.5, "max": 29.5, "step": 0.1, "unit": "psi (in-game)",
        "telemetry_in_pascals": True, "min_label": "22.5 psi", "max_label": "29.5 psi", "source": "user",
    },
    "Rear tire pressure": {
        "field": "tyre_press_setup_2", "alt_fields": ["tyre_press_setup_3"],
        "min": 20.5, "max": 26.5, "step": 0.1, "unit": "psi (in-game)",
        "telemetry_in_pascals": True, "min_label": "20.5 psi", "max_label": "26.5 psi", "source": "user",
    },
    "Front spring": {"field": "susp_spring_setup_0", "min": 1, "max": 41, "step": 1, "unit": "clicks", "source": "user"},
    "Rear spring": {"field": "susp_spring_setup_2", "min": 1, "max": 41, "step": 1, "unit": "clicks", "source": "user"},
    "Front camber": {
        "field": "camber_setup_0", "alt_fields": ["camber_setup_1"],
        "min": -3.5, "max": -2.5, "step": 0.1, "unit": "degrees (in-game)",
        "telemetry_in_radians": True, "source": "user",
    },
    "Rear camber": {
        "field": "camber_setup_2", "alt_fields": ["camber_setup_3"],
        "min": -2.0, "max": -1.0, "step": 0.1, "unit": "degrees (in-game)",
        "telemetry_in_radians": True, "source": "user",
    },
    "Front toe out": {
        "field": "toe_setup_0", "alt_fields": ["toe_setup_1"],
        "min": 0.0, "max": 0.2, "step": 0.01, "unit": "degrees (in-game)",
        "telemetry_in_radians": True, "source": "user",
    },
    "Rear toe in": {
        "field": "toe_setup_2", "alt_fields": ["toe_setup_3"],
        "min": 0.1, "max": 0.25, "step": 0.01, "unit": "degrees (in-game)",
        "telemetry_in_radians": True, "source": "user",
    },
    "Front ride height": {"field": "susp_height_setup_0", "min": 15, "max": 35, "step": 1, "unit": "clicks", "source": "user"},
    "Rear ride height": {"field": "susp_height_setup_2", "min": 40, "max": 60, "step": 1, "unit": "clicks", "source": "user"},
}

ISSUE_IMPACT = {
    "lock_front": 1.35, "lock_rear": 1.35, "os_entry": 1.3, "os_high": 1.25, "os_exit": 1.2,
    "traction_spin": 1.15, "us_high": 1.15, "aero_us_hs": 1.1, "us_entry": 1.1, "us_low": 1.05,
    "us_mid_speed": 1.05, "us_exit": 1.0, "os_low": 1.05, "os_mid_speed": 1.05,
    "steer_corrections": 0.95, "tires_hot": 0.9, "tires_cold": 0.75, "tires_axle_imbalance": 0.8,
    "aero_os_or_mech": 0.95,
}
SAFETY_IDS = {"lock_front", "lock_rear", "os_entry", "os_high", "os_exit", "traction_spin"}
ISSUE_NAMES = {
    "us_low": "Low-speed understeer", "us_high": "High-speed understeer",
    "us_mid_speed": "Medium-speed understeer", "os_low": "Low-speed oversteer",
    "os_high": "High-speed oversteer", "os_mid_speed": "Medium-speed oversteer",
    "us_entry": "Entry understeer", "os_entry": "Entry oversteer",
    "os_exit": "Exit oversteer", "us_exit": "Exit understeer",
    "traction_spin": "Exit traction limitation (wheelspin)",
    "lock_front": "Front lockup", "lock_rear": "Rear lockup",
    "steer_corrections": "High steering correction (instability)",
    "aero_us_hs": "Aero imbalance (high-speed understeer trend)",
    "aero_os_or_mech": "Low-speed mechanical understeer vs aero",
    "tires_cold": "Tires below temperature window",
    "tires_hot": "Tires above temperature window",
    "tires_axle_imbalance": "Front/rear tire temp imbalance",
}


@dataclass
class IssueEvent:
    issue_id: str
    name: str
    lap: float
    distance_m: float
    speed_kph: float
    phase: str
    severity: float
    detail: str


@dataclass
class IssueSummary:
    issue_id: str
    name: str
    count: int
    events_per_lap: float
    laps_present: int
    total_laps: int
    lap_presence_pct: float
    hot_corners_m: list
    mean_severity: float
    max_severity: float
    sample_details: list
    confidence: float
    criticality: float = 0.0
    tier: str = "C"


@dataclass
class SetupChange:
    parameter: str
    direction: str
    amount_hint: str
    reason: str
    linked_issues: list
    priority: float
    validation_metric: str
    current: Optional[float] = None
    min_v: Optional[float] = None
    max_v: Optional[float] = None
    feasible: bool = True
    blocked_reason: str = ""
    issue_id: str = ""
    option_label: str = ""


def _detect_sep(sample: bytes) -> str:
    return "\t" if b"\t" in sample[:4000] else ","


def load_telemetry(file) -> pd.DataFrame:
    if hasattr(file, "read"):
        raw = file.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8", errors="replace")
        sep = _detect_sep(raw)
        df = pd.read_csv(io.BytesIO(raw), sep=sep, low_memory=False)
    else:
        with open(file, "rb") as f:
            sample = f.read(4000)
        sep = _detect_sep(sample)
        df = pd.read_csv(file, sep=sep, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return clean_telemetry(df)


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _assign_g_axes(df: pd.DataFrame, gx: pd.Series, gy: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = df.index
    steer = df["steering"] if "steering" in df.columns else pd.Series(np.nan, index=idx)
    g_long, g_lat = gx.copy(), gy.copy()
    try:
        s = steer.fillna(0.0)
        if s.abs().mean() > 0.02:
            c_x = float(np.corrcoef(s, gx.fillna(0.0))[0, 1])
            c_y = float(np.corrcoef(s, gy.fillna(0.0))[0, 1])
            if abs(c_x) > abs(c_y):
                g_lat, g_long = gx, gy
            else:
                g_lat, g_long = gy, gx
    except Exception:
        pass
    return g_long, g_lat


def clean_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    keep_as_is = {"carId", "trackId"}
    num_cols = [c for c in df.columns if c not in keep_as_is]
    if num_cols:
        coerced = {c: _to_num(df[c]) for c in num_cols}
        id_part = df[[c for c in df.columns if c in keep_as_is]]
        df = pd.concat([id_part, pd.DataFrame(coerced, index=idx)], axis=1)
        ordered = [c for c in list(id_part.columns) + num_cols if c in df.columns]
        df = df.reindex(columns=ordered)

    sentinel_cols = [
        "throttle", "brake", "clutch", "steering", "gear", "rpm", "rpm_perc", "fuel",
        "lap_number", "lap_distance", "lap_time", "pit_status",
        "wheel_speed_0", "wheel_speed_1", "wheel_speed_2", "wheel_speed_3",
        "tyre_temp_0", "tyre_temp_1", "tyre_temp_2", "tyre_temp_3",
        "tyre_wear_0", "tyre_wear_1", "tyre_wear_2", "tyre_wear_3",
        "tyre_press_0", "tyre_press_1", "tyre_press_2", "tyre_press_3",
        "wing_setup_0", "wing_setup_1", "drs", "ers_store", "track_temp", "air_temp",
        "front_brake_bias", "brake_bias_setup", "tyres_age",
        "velocity_X", "velocity_Y", "velocity_Z",
    ]
    for c in sentinel_cols:
        if c in df.columns:
            df.loc[df[c] == SENTINEL, c] = np.nan

    derived: dict[str, pd.Series] = {}
    vx = df["velocity_X"].fillna(0.0) if "velocity_X" in df.columns else pd.Series(0.0, index=idx)
    vy = df["velocity_Y"].fillna(0.0) if "velocity_Y" in df.columns else pd.Series(0.0, index=idx)
    vz = df["velocity_Z"].fillna(0.0) if "velocity_Z" in df.columns else pd.Series(0.0, index=idx)
    derived["speed_ms"] = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    if "speed" in df.columns:
        spd = df["speed"]
        derived["speed_kph"] = np.where(spd.notna() & (spd.abs() > derived["speed_ms"] * 3.6 * 0.5), spd, derived["speed_ms"] * 3.6)
    else:
        derived["speed_kph"] = derived["speed_ms"] * 3.6

    gx = df["gforce_X"] if "gforce_X" in df.columns else pd.Series(np.nan, index=idx)
    gy = df["gforce_Y"] if "gforce_Y" in df.columns else pd.Series(np.nan, index=idx)
    tmp = df.copy()
    for i in range(4):
        sa, sr = f"wheel_slip_angle_{i}", f"wheel_slip_ratio_{i}"
        if sa not in tmp.columns:
            tmp[sa] = np.nan
        if sr not in tmp.columns:
            tmp[sr] = np.nan
    g_long, g_lat = _assign_g_axes(tmp, gx, gy)
    derived["g_long"] = g_long
    derived["g_lat"] = g_lat
    derived["g_lat_abs"] = g_lat.abs()

    work = pd.concat([df, pd.DataFrame({k: v for k, v in derived.items() if k.startswith("wheel_")}, index=idx)], axis=1)
    for i in range(4):
        sa, sr = f"wheel_slip_angle_{i}", f"wheel_slip_ratio_{i}"
        if sa not in work.columns:
            work[sa] = np.nan
        if sr not in work.columns:
            work[sr] = np.nan

    derived["alpha_f"] = work[["wheel_slip_angle_0", "wheel_slip_angle_1"]].mean(axis=1)
    derived["alpha_r"] = work[["wheel_slip_angle_2", "wheel_slip_angle_3"]].mean(axis=1)
    derived["alpha_balance"] = derived["alpha_f"].abs() - derived["alpha_r"].abs()
    derived["kappa_f"] = work[["wheel_slip_ratio_0", "wheel_slip_ratio_1"]].mean(axis=1)
    derived["kappa_r"] = work[["wheel_slip_ratio_2", "wheel_slip_ratio_3"]].mean(axis=1)

    t0 = work["tyre_temp_0"] if "tyre_temp_0" in work.columns else pd.Series(np.nan, index=idx)
    t1 = work["tyre_temp_1"] if "tyre_temp_1" in work.columns else pd.Series(np.nan, index=idx)
    t2 = work["tyre_temp_2"] if "tyre_temp_2" in work.columns else pd.Series(np.nan, index=idx)
    t3 = work["tyre_temp_3"] if "tyre_temp_3" in work.columns else pd.Series(np.nan, index=idx)
    derived["tyre_temp_f"] = pd.concat([t0, t1], axis=1).mean(axis=1)
    derived["tyre_temp_r"] = pd.concat([t2, t3], axis=1).mean(axis=1)

    thr = work["throttle"].fillna(0.0) if "throttle" in work.columns else pd.Series(0.0, index=idx)
    brk = work["brake"].fillna(0.0) if "brake" in work.columns else pd.Series(0.0, index=idx)
    steer = work["steering"].fillna(0.0) if "steering" in work.columns else pd.Series(0.0, index=idx)
    derived["throttle"] = thr
    derived["brake"] = brk
    derived["steering"] = steer
    derived["d_steer"] = steer.diff().abs().fillna(0.0)

    g_lat_abs = derived["g_lat_abs"].fillna(0.0)
    g_long_s = derived["g_long"].fillna(0.0)
    entry = (brk > 0.12) & ((g_long_s < -0.15) | (g_lat_abs > 0.4))
    exit_ = (thr > 0.25) & (brk < 0.08) & (g_long_s > -0.05) & (g_lat_abs > 0.35)
    mid = (brk < 0.08) & (thr < 0.3) & (g_lat_abs > 0.55)
    phase = np.where(mid & (brk < 0.05), PHASE_MID, np.where(entry, PHASE_ENTRY, np.where(mid, PHASE_MID, np.where(exit_, PHASE_EXIT, PHASE_STRAIGHT))))
    derived["phase"] = pd.Series(phase, index=idx)

    lap_n = work["lap_number"] if "lap_number" in work.columns else pd.Series(0, index=idx)
    pit = work["pit_status"] if "pit_status" in work.columns else pd.Series(0, index=idx)
    derived["valid_sample"] = (derived["speed_kph"].fillna(0) > 5) & (lap_n.fillna(-1) >= 0) & (pit.fillna(0) <= 0)

    out = pd.concat([df, pd.DataFrame(derived, index=idx)], axis=1)
    return out


def extract_setup(df: pd.DataFrame) -> dict[str, Any]:
    raw: dict[str, float] = {}
    named: dict[str, Optional[float]] = {}
    table = []
    for param, meta in SETUP_LIMITS.items():
        fields = [meta["field"]] + list(meta.get("alt_fields") or [])
        val = None
        used = None
        for f in fields:
            if f in df.columns and df[f].notna().any():
                val = float(df[f].median())
                used = f
                break
        if val is not None:
            raw[used] = val
        named[param] = val
        table.append({
            "Parameter": param,
            "Channel": used or meta["field"],
            "Raw": f"{val:g}" if val is not None else "—",
            "Current": format_setup_value(param, _to_limit_units(param, val)),
        })
    return {"raw": raw, "named": named, "table": table}


def _to_limit_units(parameter: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    meta = SETUP_LIMITS.get(parameter) or {}
    v = float(value)
    if meta.get("telemetry_in_radians") and abs(v) < 0.5:
        return v * (180.0 / np.pi)
    if meta.get("telemetry_in_pascals") and v > 500:
        return v / 6894.757
    if meta.get("display_as_pct") and abs(v) > 1.5:
        return v / 100.0
    return v


def setup_current(setup: dict[str, Any], parameter: str) -> Optional[float]:
    named = setup.get("named") or {}
    raw_val: Optional[float] = None
    if parameter in named and named[parameter] is not None:
        raw_val = float(named[parameter])
    else:
        meta = SETUP_LIMITS.get(parameter)
        if not meta:
            return None
        raw = setup.get("raw") or {}
        if meta["field"] in raw:
            raw_val = float(raw[meta["field"]])
        else:
            for alt in meta.get("alt_fields") or []:
                if alt in raw:
                    raw_val = float(raw[alt])
                    break
    return _to_limit_units(parameter, raw_val)


def format_setup_value(parameter: str, value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    meta = SETUP_LIMITS.get(parameter) or {}
    if meta.get("display_as_pct"):
        pct = value * 100.0
        if "Brake bias" in parameter:
            if pct >= 65:
                label = "more forward"
            elif pct <= 55:
                label = "more rearward"
            else:
                label = "mid-range"
            return f"{label} {pct:.0f}%"
        return f"{pct:.0f}%"
    if meta.get("telemetry_in_radians") or str(meta.get("unit", "")).startswith("degrees"):
        return f"{value:.2f}°"
    if meta.get("telemetry_in_pascals") or "psi" in str(meta.get("unit", "")).lower():
        return f"{value:.1f} psi"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def feasibility(parameter: str, direction: str, setup: dict[str, Any]):
    meta = SETUP_LIMITS.get(parameter)
    cur = setup_current(setup, parameter)
    if meta is None:
        return True, "", cur, None, None
    lo, hi = float(meta["min"]), float(meta["max"])
    if cur is None:
        return True, "Current value not in telemetry; verify in-game before changing.", None, lo, hi
    if direction == "increase" and cur >= hi:
        extra = f" (increase would mean: {meta['increase_means']})" if meta.get("increase_means") else ""
        return False, f"Already at maximum ({format_setup_value(parameter, cur)}). Cannot increase.{extra}", cur, lo, hi
    if direction == "decrease" and cur <= lo:
        extra = f" (decrease would mean: {meta['decrease_means']})" if meta.get("decrease_means") else ""
        return False, f"Already at minimum ({format_setup_value(parameter, cur)}). Cannot decrease.{extra}", cur, lo, hi
    return True, "", cur, lo, hi


def _clamp_setup_value(parameter: str, value: float) -> float:
    meta = SETUP_LIMITS.get(parameter) or {}
    lo, hi = meta.get("min"), meta.get("max")
    v = float(value)
    if lo is not None:
        v = max(float(lo), v)
    if hi is not None:
        v = min(float(hi), v)
    step = float(meta.get("step", 1) or 1)
    if step >= 1 and not meta.get("display_as_pct") and not meta.get("telemetry_in_radians"):
        v = round(v)
    elif step:
        v = round(v / step) * step
    return v


def _apply_clicks(parameter: str, value: float, clicks: int) -> float:
    step = float((SETUP_LIMITS.get(parameter) or {}).get("step", 1) or 1)
    return _clamp_setup_value(parameter, float(value) + int(clicks) * step)


def _clicks_between(parameter: str, from_v: float, to_v: float) -> int:
    step = float((SETUP_LIMITS.get(parameter) or {}).get("step", 1) or 1)
    if not step:
        return 0
    return int(round((float(to_v) - float(from_v)) / step))


def format_lap_time(sec: Optional[float]) -> str:
    if sec is None or not np.isfinite(sec):
        return "—"
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m}:{s:06.3f}"


def _corner_bin(distance_m: float, bin_m: float = 50.0) -> float:
    if pd.isna(distance_m):
        return np.nan
    return float(bin_m * round(distance_m / bin_m))


def extract_completed_lap_times(df: pd.DataFrame) -> list[tuple[float, float]]:
    rejected = []
    if "lap_number" not in df.columns or "lap_time" not in df.columns:
        extract_completed_lap_times.last_rejected = rejected
        return []
    candidates = []
    for lap, g in df.groupby("lap_number"):
        if pd.isna(lap) or lap < 0:
            continue
        times = g["lap_time"].dropna()
        times = times[times >= 0]
        if times.empty:
            continue
        lt = float(times.max())
        if lt < 55:
            rejected.append({"lap": lap, "lap_time": lt, "reason": f"too short ({lt:.3f}s < 55s)"})
            continue
        candidates.append((float(lap), lt))
    if not candidates:
        extract_completed_lap_times.last_rejected = rejected
        return []
    med = float(np.median([t for _, t in candidates]))
    kept = []
    for lap, lt in candidates:
        if med >= 55 and lt < med * 0.88:
            rejected.append({"lap": lap, "lap_time": lt, "reason": f"outlier fast vs median {med:.3f}s"})
            continue
        if med >= 55 and lt > med * 1.2:
            rejected.append({"lap": lap, "lap_time": lt, "reason": f"outlier slow vs median {med:.3f}s"})
            continue
        kept.append((lap, lt))
    extract_completed_lap_times.last_rejected = rejected
    return kept


def run_diagnostics(df, us_alpha=US_ALPHA_THRESH, os_alpha=OS_ALPHA_THRESH, lock_slip=LOCK_SLIP, spin_slip=SPIN_SLIP):
    d = df[df["valid_sample"]].copy() if "valid_sample" in df.columns else df.copy()
    if d.empty:
        return [], []
    events: list[IssueEvent] = []

    def add(issue_id, row, severity, detail):
        events.append(IssueEvent(
            issue_id=issue_id, name=ISSUE_NAMES.get(issue_id, issue_id),
            lap=float(row.get("lap_number", 0) or 0),
            distance_m=float(row.get("lap_distance", 0) or 0),
            speed_kph=float(row.get("speed_kph", 0) or 0),
            phase=str(row.get("phase", "")),
            severity=float(severity),
            detail=detail,
        ))

    for _, row in d.iterrows():
        phase = row.get("phase")
        bal = row.get("alpha_balance")
        spd = float(row.get("speed_kph") or 0)
        gabs = float(row.get("g_lat_abs") or 0)
        if pd.notna(bal) and phase == PHASE_MID and gabs > 0.6:
            if bal > us_alpha:
                iid = "us_low" if spd < V_LOW else ("us_high" if spd > V_HIGH else "us_mid_speed")
                add(iid, row, min(2, bal / us_alpha), f"α_f−α_r={bal:.3f} rad, v={spd:.0f} kph")
            elif bal < -os_alpha:
                iid = "os_low" if spd < V_LOW else ("os_high" if spd > V_HIGH else "os_mid_speed")
                add(iid, row, min(2, abs(bal) / os_alpha), f"α_bal={bal:.3f}, v={spd:.0f}")
        if pd.notna(bal) and phase == PHASE_ENTRY and gabs > 0.5:
            if bal > us_alpha * 1.1:
                add("us_entry", row, min(2, bal / us_alpha), f"Turn-in push α_bal={bal:.3f}")
            elif bal < -os_alpha * 1.1:
                add("os_entry", row, min(2, abs(bal) / os_alpha), f"Rear rotates on entry α_bal={bal:.3f}")
        thr = float(row.get("throttle") or 0)
        brk = float(row.get("brake") or 0)
        if phase == PHASE_EXIT and thr > 0.4:
            kr = row.get("kappa_r")
            if pd.notna(kr) and kr > spin_slip:
                add("traction_spin", row, min(2, kr / spin_slip), f"κ_r={kr:.3f}, throttle={thr:.2f}")
            if pd.notna(bal) and bal < -os_alpha and thr > 0.5:
                add("os_exit", row, min(2, abs(bal) / os_alpha), f"Power oversteer α_bal={bal:.3f}")
            if pd.notna(bal) and bal > us_alpha and thr > 0.55:
                add("us_exit", row, min(2, bal / us_alpha), f"Push on power α_bal={bal:.3f}")
        if brk > 0.35 and spd > 40:
            kf, kr = row.get("kappa_f"), row.get("kappa_r")
            if pd.notna(kf) and abs(kf) > lock_slip and kf < 0:
                add("lock_front", row, min(2, abs(kf) / lock_slip), f"κ_f={kf:.3f}")
            if pd.notna(kr) and abs(kr) > lock_slip and kr < 0:
                add("lock_rear", row, min(2, abs(kr) / lock_slip), f"κ_r={kr:.3f}")
        ds = float(row.get("d_steer") or 0)
        if phase in (PHASE_MID, PHASE_EXIT) and gabs > 0.7 and ds > 0.04:
            add("steer_corrections", row, min(2, ds / 0.04), f"Δsteer={ds:.3f}")
        tf, tr = row.get("tyre_temp_f"), row.get("tyre_temp_r")
        if pd.notna(tf) and pd.notna(tr):
            if tf < TIRE_COLD or tr < TIRE_COLD:
                add("tires_cold", row, max(0.5, (TIRE_COLD - min(tf, tr)) / 10), f"T_f={tf:.0f} T_r={tr:.0f}")
            if tf > TIRE_HOT or tr > TIRE_HOT:
                add("tires_hot", row, max(0.5, (max(tf, tr) - TIRE_HOT) / 10), f"T_f={tf:.0f} T_r={tr:.0f}")
            if abs(tf - tr) > 12:
                add("tires_axle_imbalance", row, min(2, abs(tf - tr) / 12), f"ΔT={tf - tr:.0f}")

    clusters: dict[tuple, IssueEvent] = {}
    for e in events:
        key = (e.issue_id, round(e.lap), _corner_bin(e.distance_m))
        prev = clusters.get(key)
        if prev is None or e.severity > prev.severity:
            clusters[key] = e
    clustered = list(clusters.values())

    lap_times = extract_completed_lap_times(df)
    total_laps = max(1, len(lap_times) or int(d["lap_number"].nunique()) if "lap_number" in d.columns else 1)
    by_id: dict[str, list[IssueEvent]] = {}
    for e in clustered:
        by_id.setdefault(e.issue_id, []).append(e)

    summaries: list[IssueSummary] = []
    for iid, evs in by_id.items():
        laps_hit = {round(e.lap) for e in evs}
        sevs = [e.severity for e in evs]
        bins: dict[float, int] = {}
        for e in evs:
            b = _corner_bin(e.distance_m)
            if np.isfinite(b):
                bins[b] = bins.get(b, 0) + 1
        hot = [k for k, _ in sorted(bins.items(), key=lambda x: -x[1])[:5]]
        count = len(evs)
        mean_sev = float(np.mean(sevs))
        presence = 100.0 * len(laps_hit) / total_laps
        conf = min(1.0, 0.35 + 0.1 * min(count, 5) + 0.4 * (len(laps_hit) / total_laps) + 0.1 * mean_sev)
        w = ISSUE_IMPACT.get(iid, 1.0)
        crit = (count / total_laps) * max(mean_sev, 0.05) * max(conf, 0.05) * w * (0.35 + presence / 100)
        tier = "C"
        if (iid in SAFETY_IDS and presence >= 20 and mean_sev >= 0.85) or (crit >= 2.5 and presence >= 30):
            tier = "S"
        elif presence >= 45 and mean_sev >= 0.7:
            tier = "A"
        elif presence >= 25 or count >= 5:
            tier = "B"
        if crit >= 1.8 and tier == "B":
            tier = "A"
        if crit >= 3.0 and tier in ("A", "B"):
            tier = "S"
        summaries.append(IssueSummary(
            issue_id=iid, name=ISSUE_NAMES.get(iid, iid), count=count,
            events_per_lap=count / total_laps, laps_present=len(laps_hit),
            total_laps=total_laps, lap_presence_pct=presence, hot_corners_m=hot,
            mean_severity=mean_sev, max_severity=float(max(sevs)),
            sample_details=[e.detail for e in evs[:3]], confidence=conf,
            criticality=crit, tier=tier,
        ))
    order = {"S": 0, "A": 1, "B": 2, "C": 3}
    summaries.sort(key=lambda s: (order.get(s.tier, 9), -s.criticality))
    return clustered, summaries


def session_overview(df: pd.DataFrame) -> dict[str, Any]:
    d = df[df["valid_sample"]] if "valid_sample" in df.columns else df
    lap_times = extract_completed_lap_times(df)
    laps = [ln for ln, _ in lap_times]
    best = min(lap_times, key=lambda x: x[1]) if lap_times else None
    return {
        "track": df["trackId"].dropna().iloc[0] if "trackId" in df.columns and df["trackId"].notna().any() else "?",
        "car": df["carId"].dropna().iloc[0] if "carId" in df.columns and df["carId"].notna().any() else "?",
        "laps": laps, "lap_times": lap_times, "best_lap": best,
        "samples": len(d),
        "vmax_kph": float(d["speed_kph"].max()) if len(d) and "speed_kph" in d.columns else 0.0,
        "track_temp": float(d["track_temp"].median()) if "track_temp" in d.columns and d["track_temp"].notna().any() else None,
        "air_temp": float(d["air_temp"].median()) if "air_temp" in d.columns and d["air_temp"].notna().any() else None,
    }


def session_grade(df, summaries, driver) -> dict[str, Any]:
    lap_times = extract_completed_lap_times(df)
    raw = [t for _, t in lap_times]
    laps = raw
    if len(raw) >= 2:
        best0 = min(raw)
        push = [t for t in raw if t <= best0 * 1.03]
        if len(push) >= 2:
            laps = push
    components = {}
    if len(laps) >= 2:
        best, mean_t = min(laps), float(np.mean(laps))
        pace = max(40, min(100, 100 - (mean_t - best) * 20))
        components["Pace"] = {"score": pace, "detail": f"Best {best:.3f}s · mean {mean_t:.3f}s"}
    elif len(laps) == 1:
        components["Pace"] = {"score": 80, "detail": f"Single timed lap {laps[0]:.3f}s"}
    else:
        components["Pace"] = {"score": 60, "detail": "No usable lap times"}
    if len(laps) >= 3:
        sd = float(np.std(laps))
        spread = max(laps) - min(laps)
        cons = max(45, min(100, 100 - sd * 25))
        components["Consistency"] = {"score": cons, "detail": f"σ ≈ {sd:.3f}s · spread {spread:.3f}s"}
    else:
        components["Consistency"] = {"score": 75, "detail": "Need 2+ push laps"}
    epl = sum(s.events_per_lap for s in summaries if s.issue_id in SAFETY_IDS)
    components["Cleanliness"] = {"score": max(60, min(100, 100 - epl * 4)), "detail": f"Lock/spin/entry-OS per lap ≈ {epl:.2f}"}
    bal = sum(s.events_per_lap for s in summaries if s.issue_id.startswith("us_") or s.issue_id.startswith("os_"))
    components["Balance"] = {"score": max(65, min(100, 100 - bal * 2)), "detail": f"US/OS per lap ≈ {bal:.2f}"}
    tire = sum(s.events_per_lap for s in summaries if "tire" in s.issue_id)
    components["Tires"] = {"score": max(70, min(100, 100 - tire * 3)), "detail": f"Tire events/lap ≈ {tire:.2f}"}
    weights = {"Pace": 0.40, "Consistency": 0.20, "Cleanliness": 0.20, "Balance": 0.10, "Tires": 0.10}
    total = sum(components[k]["score"] * weights[k] for k in weights)
    total = max(0, min(100, total))
    letter = "A" if total >= 90 else "B" if total >= 80 else "C" if total >= 70 else "D" if total >= 60 else "F"
    return {
        "score": round(total, 1), "letter": letter, "components": components,
        "disclaimer": "Session-relative grade (push laps within 3% of best). Not a world ranking.",
    }


def analyze_driver(df: pd.DataFrame) -> dict[str, Any]:
    lap_times = extract_completed_lap_times(df)
    notes, brake_notes, throttle_notes, scrub_notes = [], [], [], []
    if not lap_times:
        return {"notes": ["No completed lap times found (need flying laps ≥ 55s)."],
                "brake_notes": [], "throttle_notes": [], "scrub_notes": [],
                "time_loss_zones": [], "delta_series": None}
    best_lap, best_t = min(lap_times, key=lambda x: x[1])
    d = df[df["valid_sample"]].copy() if "valid_sample" in df.columns else df.copy()
    ref = d[d["lap_number"] == best_lap].sort_values("lap_distance") if "lap_number" in d.columns else d
    others = [ln for ln, _ in lap_times if ln != best_lap]
    cmp = d[d["lap_number"].isin(others)] if others and "lap_number" in d.columns else ref
    notes.append(f"Reference: best lap L{best_lap:.0f} ({best_t:.3f}s). Time-loss is a speed-bin proxy.")

    def profile(sub):
        if sub.empty or "lap_distance" not in sub.columns:
            return pd.DataFrame()
        sub = sub.copy()
        sub["bin"] = (sub["lap_distance"] / 25).round() * 25
        return sub.groupby("bin")[["speed_kph", "throttle", "brake", "steering", "g_lat_abs", "alpha_balance"]].mean().reset_index()

    rp, cp = profile(ref), profile(cmp)
    zones, merged_rows = [], []
    if not rp.empty and not cp.empty:
        m = cp.merge(rp, on="bin", suffixes=("_c", "_r"))
        vref = np.maximum(1, m["speed_kph_r"] / 3.6)
        vcmp = np.maximum(1, m["speed_kph_c"] / 3.6)
        m["time_loss_pos"] = np.maximum(0, 25 * (1 / vcmp - 1 / vref))
        merged_rows = m
        top = m.sort_values("time_loss_pos", ascending=False).head(8)
        for _, z in top.iterrows():
            if z["time_loss_pos"] < 0.01:
                continue
            zones.append({
                "distance_m": float(z["bin"]), "time_loss_s": float(z["time_loss_pos"]),
                "speed_ref_kph": float(z["speed_kph_r"]), "speed_cmp_kph": float(z["speed_kph_c"]),
                "speed_delta_kph": float(z["speed_kph_r"] - z["speed_kph_c"]),
            })
        if zones:
            t0 = zones[0]
            notes.append(f"Largest proxy time loss: +{t0['time_loss_s']:.3f}s around {t0['distance_m']:.0f} m.")
    brake_notes.append("No strong brake-point issues detected vs best lap." if not zones else "Check heavy-brake bins vs best lap in the table.")
    throttle_notes.append("No strong exit-speed loss pattern vs best lap.")
    scrub_notes.append("No clear scrub signature vs best lap.")
    return {
        "notes": notes, "brake_notes": brake_notes, "throttle_notes": throttle_notes,
        "scrub_notes": scrub_notes, "time_loss_zones": zones,
        "delta_series": merged_rows if len(merged_rows) else None,
    }


ISSUE_TO_CHANGES = {
    "us_low": [
        {"parameter": "Front wing", "direction": "increase", "amount_hint": "+1 to +2 clicks",
         "reason": "Low-speed mid-corner understeer (front slip angle > rear).",
         "validation_metric": "α_balance mid-corner low-speed closer to 0", "weight": 1},
        {"parameter": "Front ARB", "direction": "decrease", "amount_hint": "−1 to −2",
         "reason": "Softer front anti-roll adds front mechanical grip in slow corners.",
         "validation_metric": "Fewer us_low events per lap", "weight": 1},
        {"parameter": "Rear ARB", "direction": "increase", "amount_hint": "+1",
         "reason": "Stiffer rear can rotate the car mid-corner at low speed.",
         "validation_metric": "No rise in os_low frequency", "weight": 0.85},
        {"parameter": "Off-throttle differential", "direction": "decrease", "amount_hint": "−5% to −10%",
         "reason": "Less coast locking can free rotation into/through slow corners.",
         "validation_metric": "us_low / us_entry frequency", "weight": 0.65},
        {"parameter": "Front spring", "direction": "decrease", "amount_hint": "−1 to −2",
         "reason": "Softer front spring can increase front mechanical grip mid-corner.",
         "validation_metric": "us_low events per lap", "weight": 0.55},
    ],
    "us_high": [
        {"parameter": "Front wing", "direction": "increase", "amount_hint": "+1 to +3",
         "reason": "High-speed understeer — aero front load shortfall.",
         "validation_metric": "us_high count down", "weight": 1.2},
        {"parameter": "Rear wing", "direction": "decrease", "amount_hint": "−1 (if top speed allows)",
         "reason": "Reduces rear aero dominance that pushes the car wide in fast corners.",
         "validation_metric": "Trap speed acceptable", "weight": 1},
        {"parameter": "Front ARB", "direction": "decrease", "amount_hint": "−1 to −2",
         "reason": "Mechanical front grip if front wing is already maxed.",
         "validation_metric": "us_high frequency", "weight": 0.85},
    ],
    "us_mid_speed": [
        {"parameter": "Front wing", "direction": "increase", "amount_hint": "+1",
         "reason": "Medium-speed mid-corner push.", "validation_metric": "us_mid_speed frequency", "weight": 0.8},
        {"parameter": "Front ARB", "direction": "decrease", "amount_hint": "−1",
         "reason": "Mechanical front grip for mid-speed corners.", "validation_metric": "α_balance mid phase", "weight": 0.9},
        {"parameter": "Rear ARB", "direction": "increase", "amount_hint": "+1",
         "reason": "Rotate mid-corner without adding front wing.", "validation_metric": "us_mid_speed", "weight": 0.7},
    ],
    "os_low": [
        {"parameter": "On-throttle differential", "direction": "decrease", "amount_hint": "−10% to −20%",
         "reason": "Low-speed oversteer — less locking on power helps rear stability.",
         "validation_metric": "os_low and os_exit counts", "weight": 0.7},
        {"parameter": "Rear ARB", "direction": "decrease", "amount_hint": "−1 to −2",
         "reason": "Softer rear anti-roll reduces low-speed rear slip.",
         "validation_metric": "α_balance not rear-dominant", "weight": 1},
    ],
    "os_high": [
        {"parameter": "Rear wing", "direction": "increase", "amount_hint": "+1 to +2",
         "reason": "High-speed rear instability / oversteer.", "validation_metric": "os_high count", "weight": 1.2},
        {"parameter": "Rear ARB", "direction": "decrease", "amount_hint": "−1",
         "reason": "More rear mechanical compliance in fast direction changes.",
         "validation_metric": "steer_corrections high-speed", "weight": 0.8},
    ],
    "os_entry": [
        {"parameter": "Brake bias (% front)", "direction": "increase", "amount_hint": "more forward +1% to +2% (toward 70%)",
         "reason": "Entry oversteer — more forward 70% to stabilize the rear on turn-in.",
         "validation_metric": "os_entry frequency", "weight": 1.1},
        {"parameter": "Off-throttle differential", "direction": "increase", "amount_hint": "+5% to +10%",
         "reason": "More coast locking can stabilize entry rotation.", "validation_metric": "os_entry events", "weight": 0.6},
    ],
    "us_entry": [
        {"parameter": "Brake bias (% front)", "direction": "decrease", "amount_hint": "more rearward −1% to −2% (toward 50%)",
         "reason": "Entry understeer — more rearward 50% to help rotation on trail-brake.",
         "validation_metric": "us_entry count", "weight": 1},
        {"parameter": "Front wing", "direction": "increase", "amount_hint": "+1",
         "reason": "More front aero turn-in bite.", "validation_metric": "us_entry frequency", "weight": 0.7},
        {"parameter": "Front ARB", "direction": "decrease", "amount_hint": "−1",
         "reason": "Softer front ARB helps turn-in if wing cannot go up.", "validation_metric": "us_entry frequency", "weight": 0.85},
    ],
    "os_exit": [
        {"parameter": "On-throttle differential", "direction": "decrease", "amount_hint": "−10% to −20%",
         "reason": "Exit oversteer — open on-throttle diff reduces inside-wheel drive spike.",
         "validation_metric": "os_exit and traction_spin counts", "weight": 1.2},
        {"parameter": "Rear ARB", "direction": "decrease", "amount_hint": "−1",
         "reason": "Helps put power down with a calmer rear.", "validation_metric": "κ_r on exit", "weight": 0.8},
    ],
    "us_exit": [
        {"parameter": "On-throttle differential", "direction": "increase", "amount_hint": "+5% to +10%",
         "reason": "Exit understeer — more locking can help rotate (if not spinning).",
         "validation_metric": "us_exit", "weight": 0.7},
        {"parameter": "Rear ARB", "direction": "increase", "amount_hint": "+1",
         "reason": "Can free the rear slightly on exit to reduce push.", "validation_metric": "us_exit frequency", "weight": 0.6},
    ],
    "traction_spin": [
        {"parameter": "On-throttle differential", "direction": "decrease", "amount_hint": "−10% to −20%",
         "reason": "Rear wheelspin on exit (high κ_r).", "validation_metric": "Mean peak κ_r on exit", "weight": 1.3},
        {"parameter": "Rear wing", "direction": "increase", "amount_hint": "+1",
         "reason": "Slightly more rear load for traction.", "validation_metric": "traction_spin frequency", "weight": 0.4},
    ],
    "lock_front": [
        {"parameter": "Brake bias (% front)", "direction": "decrease", "amount_hint": "more rearward −1% to −2% (toward 50%)",
         "reason": "Front lockups — more rearward 50% to unload the fronts.", "validation_metric": "lock_front count", "weight": 1.2},
        {"parameter": "Brake pressure", "direction": "decrease", "amount_hint": "−1% to −2%",
         "reason": "Softer initial bite if still locking after bias change.", "validation_metric": "lock_front severity", "weight": 0.8},
        {"parameter": "Driving / out-lap", "direction": "adjust", "amount_hint": "Slightly earlier brake / less initial spike",
         "reason": "Driver input often fixes lock without setup change.", "validation_metric": "lock_front frequency", "weight": 0.45},
    ],
    "lock_rear": [
        {"parameter": "Brake bias (% front)", "direction": "increase", "amount_hint": "more forward +1% to +2% (toward 70%)",
         "reason": "Rear lockups — more forward 70% to unload the rears.", "validation_metric": "lock_rear count", "weight": 1.2},
        {"parameter": "Brake pressure", "direction": "decrease", "amount_hint": "−1% to −2%",
         "reason": "Softer overall bite if bias alone does not stop rear lock.", "validation_metric": "lock_rear count", "weight": 0.7},
    ],
    "steer_corrections": [
        {"parameter": "Rear wing", "direction": "increase", "amount_hint": "+1",
         "reason": "High mid/exit steering corrections suggest rear nervousness.", "validation_metric": "steer_corrections frequency", "weight": 0.6},
        {"parameter": "Rear ARB", "direction": "decrease", "amount_hint": "−1",
         "reason": "Calmer rear mechanical response.", "validation_metric": "steer_corrections + os_*", "weight": 0.7},
    ],
    "aero_us_hs": [
        {"parameter": "Front wing", "direction": "increase", "amount_hint": "+1 to +2",
         "reason": "Steer-per-G rises with speed (aero understeer trend).", "validation_metric": "U_high vs U_low", "weight": 1.1},
        {"parameter": "Rear wing", "direction": "decrease", "amount_hint": "−1 to −2",
         "reason": "Alternative when front wing is already at max: reduce rear aero.", "validation_metric": "trap speed", "weight": 1.05},
    ],
    "tires_cold": [
        {"parameter": "Driving / out-lap", "direction": "adjust", "amount_hint": "Warm-up: weave, later brake, build load",
         "reason": "Tires below window — setup changes are secondary until temps rise.",
         "validation_metric": "tyre temps into window", "weight": 0.9},
    ],
    "tires_hot": [
        {"parameter": "Driving / out-lap", "direction": "adjust", "amount_hint": "Reduce sliding / scrubbing",
         "reason": "Over-temp often from sustained slide — fix balance issues first.",
         "validation_metric": "Peak tyre temp", "weight": 0.8},
    ],
    "tires_axle_imbalance": [
        {"parameter": "Front ARB", "direction": "decrease", "amount_hint": "−1 if fronts much hotter",
         "reason": "Front much hotter often means front sliding (understeer).", "validation_metric": "|T_f − T_r|", "weight": 0.6},
        {"parameter": "Rear ARB", "direction": "decrease", "amount_hint": "−1 if rears much hotter",
         "reason": "Rear much hotter often means rear sliding (oversteer).", "validation_metric": "|T_f − T_r|", "weight": 0.6},
    ],
}


def recommend_setup(summaries: list[IssueSummary], setup: dict[str, Any]) -> list[SetupChange]:
    all_changes: list[SetupChange] = []
    for s in summaries:
        templates = ISSUE_TO_CHANGES.get(s.issue_id, [])
        if not templates:
            continue
        freq_factor = min(2.0, 0.5 + s.events_per_lap + 0.01 * s.lap_presence_pct)
        issue_opts: list[SetupChange] = []
        for t in templates:
            pr = t["weight"] * s.mean_severity * s.confidence * freq_factor
            ok, blocked, cur, lo, hi = feasibility(t["parameter"], t["direction"], setup)
            cur_txt = format_setup_value(t["parameter"], cur)
            meta = SETUP_LIMITS.get(t["parameter"]) or {}
            range_txt = ""
            if lo is not None and hi is not None:
                lo_l = meta.get("min_label") or format_setup_value(t["parameter"], lo)
                hi_l = meta.get("max_label") or format_setup_value(t["parameter"], hi)
                range_txt = f" [range {lo_l} … {hi_l}]"
            dir_txt = ""
            if t["direction"] == "increase" and meta.get("increase_means"):
                dir_txt = f" → {meta['increase_means']}"
            elif t["direction"] == "decrease" and meta.get("decrease_means"):
                dir_txt = f" → {meta['decrease_means']}"
            amount = t["amount_hint"]
            amount = f"{amount} (current {cur_txt}{range_txt}){dir_txt}" if cur is not None else f"{amount}{range_txt}{dir_txt}"
            issue_opts.append(SetupChange(
                parameter=t["parameter"], direction=t["direction"], amount_hint=amount,
                reason=t["reason"] + f" [seen {s.count}×, {s.lap_presence_pct:.0f}% of laps]",
                linked_issues=[s.name], priority=pr if ok else pr * 0.01,
                validation_metric=t["validation_metric"], current=cur, min_v=lo, max_v=hi,
                feasible=ok, blocked_reason=blocked, issue_id=s.issue_id,
            ))
        issue_opts.sort(key=lambda c: (not c.feasible, -c.priority))
        for i, c in enumerate(issue_opts):
            c.option_label = f"Option {'ABCDEFGH'[i] if i < 8 else i + 1}"
            all_changes.append(c)
    return all_changes


def recommendations_by_issue(changes: list[SetupChange]) -> dict[str, list[SetupChange]]:
    by: dict[str, list[SetupChange]] = {}
    for c in changes:
        key = c.linked_issues[0] if c.linked_issues else c.issue_id
        by.setdefault(key, []).append(c)
    return by


def _n_steps_for_mode(mode: str, severity: float, weight: float) -> int:
    if mode == "aggressive":
        return 2 if severity >= 1.3 or weight >= 1.15 else 1
    return 1


def _setup_diff_rows(current, suggested):
    rows = []
    for param in SETUP_LIMITS:
        cur, sug = current.get(param), suggested.get(param)
        if cur is None and sug is None:
            continue
        changed = cur is not None and sug is not None and abs(float(sug) - float(cur)) > 1e-9
        delta_str = "—"
        if changed:
            dlt = float(sug) - float(cur)
            meta = SETUP_LIMITS[param]
            if meta.get("display_as_pct"):
                delta_str = f"{dlt * 100:+.1f}%"
            elif meta.get("telemetry_in_radians") or "degree" in meta.get("unit", ""):
                delta_str = f"{dlt:+.2f}°"
            elif meta.get("telemetry_in_pascals") or "psi" in meta.get("unit", ""):
                delta_str = f"{dlt:+.1f} psi"
            else:
                delta_str = f"{dlt:+g}"
        rows.append({
            "Parameter": param, "Current": format_setup_value(param, cur),
            "Suggested": format_setup_value(param, sug),
            "Delta": delta_str if changed else "—", "Changed": "yes" if changed else "",
        })
    return rows


def build_suggested_setup(setup, summaries, mode="conservative"):
    mode = (mode or "conservative").lower()
    if mode not in ("conservative", "aggressive"):
        mode = "conservative"
    max_changes = 3 if mode == "conservative" else 6
    allowed_tiers = {"S", "A"} if mode == "conservative" else {"S", "A", "B"}
    current = {param: setup_current(setup, param) for param in SETUP_LIMITS}
    suggested = dict(current)
    applied, skipped = [], []
    issues = [s for s in summaries if s.tier in allowed_tiers and s.issue_id in ISSUE_TO_CHANGES]
    issues.sort(key=lambda s: (-s.criticality, s.tier))
    used_params, param_dir = set(), {}
    for s in issues:
        if len(applied) >= max_changes:
            break
        templates = sorted(ISSUE_TO_CHANGES.get(s.issue_id, []), key=lambda t: -t.get("weight", 0))
        picked = None
        for t in templates:
            param, direction = t["parameter"], t["direction"]
            if direction == "adjust" or param not in SETUP_LIMITS or param in used_params:
                continue
            if param in param_dir and param_dir[param] != direction:
                continue
            ok, blocked, cur, lo, hi = feasibility(param, direction, setup)
            if not ok or cur is None:
                continue
            n_steps = _n_steps_for_mode(mode, s.mean_severity, t.get("weight", 1.0))
            step = float(SETUP_LIMITS[param].get("step", 1))
            delta = n_steps * step * (1 if direction == "increase" else -1)
            new_val = _clamp_setup_value(param, cur + delta)
            if abs(new_val - cur) < step * 0.25:
                continue
            picked = {"parameter": param, "direction": direction, "from": cur, "to": new_val,
                      "steps": n_steps, "issue": s.name, "issue_id": s.issue_id, "tier": s.tier,
                      "reason": t["reason"], "weight": t.get("weight", 1.0)}
            break
        if not picked:
            skipped.append(f"{s.name}: no feasible numeric lever left")
            continue
        suggested[picked["parameter"]] = picked["to"]
        used_params.add(picked["parameter"])
        param_dir[picked["parameter"]] = picked["direction"]
        applied.append(picked)
    return {
        "mode": mode, "current": current, "suggested": suggested, "applied": applied,
        "skipped": skipped, "rows": _setup_diff_rows(current, suggested),
        "max_changes": max_changes, "tiers_used": sorted(allowed_tiers),
    }


# ----- Best guess (the new function) -----
TRACK_TYPE_BASELINES = {
    "high": {
        "Front wing": 40, "Rear wing": 43, "On-throttle differential": 0.55, "Off-throttle differential": 0.50,
        "Front camber": -3.0, "Rear camber": -1.6, "Front toe out": 0.06, "Rear toe in": 0.22,
        "Front spring": 18, "Rear spring": 8, "Front ARB": 8, "Rear ARB": 12,
        "Front ride height": 24, "Rear ride height": 50, "Brake pressure": 1.00,
        "Brake bias (% front)": 0.54, "Front tire pressure": 23.4, "Rear tire pressure": 22.4,
    },
    "med": {
        "Front wing": 28, "Rear wing": 30, "On-throttle differential": 0.60, "Off-throttle differential": 0.50,
        "Front camber": -2.9, "Rear camber": -1.5, "Front toe out": 0.05, "Rear toe in": 0.20,
        "Front spring": 24, "Rear spring": 8, "Front ARB": 10, "Rear ARB": 16,
        "Front ride height": 21, "Rear ride height": 46, "Brake pressure": 1.00,
        "Brake bias (% front)": 0.55, "Front tire pressure": 24.0, "Rear tire pressure": 23.0,
    },
    "low": {
        "Front wing": 8, "Rear wing": 6, "On-throttle differential": 0.70, "Off-throttle differential": 0.40,
        "Front camber": -3.2, "Rear camber": -1.8, "Front toe out": 0.03, "Rear toe in": 0.14,
        "Front spring": 34, "Rear spring": 4, "Front ARB": 9, "Rear ARB": 18,
        "Front ride height": 20, "Rear ride height": 48, "Brake pressure": 1.00,
        "Brake bias (% front)": 0.56, "Front tire pressure": 26.5, "Rear tire pressure": 24.5,
    },
}
TRACK_ID_TYPES = {
    0: ("Melbourne", "med"), 1: ("Paul Ricard", "med"), 2: ("Shanghai", "med"), 3: ("Bahrain", "med"),
    4: ("Spain", "med"), 5: ("Monaco", "high"), 6: ("Canada", "low"), 7: ("Silverstone", "med"),
    8: ("Hockenheim", "med"), 9: ("Hungary", "high"), 10: ("Spa", "low"), 11: ("Monza", "low"),
    12: ("Singapore", "high"), 13: ("Suzuka", "med"), 14: ("Abu Dhabi", "med"), 15: ("Austin", "med"),
    16: ("Interlagos", "med"), 17: ("Austria", "low"), 19: ("Mexico", "low"), 20: ("Baku", "low"),
    26: ("Zandvoort", "high"), 27: ("Imola", "med"), 29: ("Jeddah", "low"), 30: ("Miami", "med"),
    31: ("Las Vegas", "low"), 32: ("Qatar", "med"),
}


def guess_track_type(track_id=None, vmax_kph=None, choice="auto") -> str:
    c = (choice or "auto").strip().lower()
    if c.startswith("high"):
        return "high"
    if c.startswith("low"):
        return "low"
    if c.startswith("med"):
        return "med"
    try:
        tid = int(float(track_id)) if track_id not in (None, "", "?") else None
    except (TypeError, ValueError):
        tid = None
    if tid is not None and tid in TRACK_ID_TYPES:
        return TRACK_ID_TYPES[tid][1]
    name = str(track_id or "").lower()
    if any(k in name for k in ("monaco", "hungary", "singapore", "zandvoort")):
        return "high"
    if any(k in name for k in ("monza", "spa", "baku", "vegas", "jeddah", "mexico", "canada")):
        return "low"
    if vmax_kph is not None:
        if vmax_kph >= 330:
            return "low"
        if vmax_kph <= 285:
            return "high"
    return "med"


def _sev(summaries, *ids):
    by = {s.issue_id: s for s in summaries}
    best = 0.0
    for i in ids:
        s = by.get(i)
        if s is None:
            continue
        v = min(1.0, float(s.mean_severity) / 2.0 + float(s.lap_presence_pct) / 200.0)
        best = max(best, v)
    return best


def _ideal_add(target, current, parameter, clicks, reason, acc):
    n = int(round(float(clicks)))
    if n == 0 or parameter not in SETUP_LIMITS:
        return
    base = target.get(parameter)
    if base is None:
        base = current.get(parameter)
    if base is None:
        return
    target[parameter] = _apply_clicks(parameter, float(base), n)
    from_cur = current.get(parameter)
    to_v = float(target[parameter])
    if from_cur is None:
        actual, from_v = n, float(base)
    else:
        from_v = float(from_cur)
        actual = _clicks_between(parameter, from_v, to_v)
    if actual == 0:
        return
    acc[parameter] = {
        "parameter": parameter, "direction": "increase" if actual > 0 else "decrease",
        "from": from_v, "to": to_v, "steps": abs(actual), "issue": "Best guess",
        "issue_id": "ideal", "tier": "—", "reason": reason, "weight": 1.0,
    }


def build_ideal_setup(setup, summaries, track_type="med", weather="dry"):
    """Full target sheet. Not capped at 1–2 clicks. Allowed to look nothing like the car you ran."""
    tt = track_type if track_type in TRACK_TYPE_BASELINES else "med"
    wet = (weather or "dry").strip().lower().startswith("wet")
    type_label = {"high": "high-downforce", "low": "low-downforce", "med": "medium-downforce"}[tt]
    current = {param: setup_current(setup, param) for param in SETUP_LIMITS}
    baseline = dict(TRACK_TYPE_BASELINES[tt])
    if wet:
        baseline["Front wing"] = _clamp_setup_value("Front wing", baseline["Front wing"] + 6)
        baseline["Rear wing"] = _clamp_setup_value("Rear wing", baseline["Rear wing"] + 8)
        baseline["On-throttle differential"] = _clamp_setup_value("On-throttle differential", baseline["On-throttle differential"] - 0.08)
        baseline["Off-throttle differential"] = _clamp_setup_value("Off-throttle differential", max(baseline["Off-throttle differential"] - 0.06, 0.20))
        baseline["Brake bias (% front)"] = _clamp_setup_value("Brake bias (% front)", baseline["Brake bias (% front)"] - 0.01)
    target = dict(baseline)
    for param in SETUP_LIMITS:
        if target.get(param) is None:
            target[param] = current.get(param)
    acc = {}
    u_mid = _sev(summaries, "us_low", "us_mid_speed")
    u_high = _sev(summaries, "us_high", "aero_us_hs")
    u_ent = _sev(summaries, "us_entry")
    o_ex = _sev(summaries, "os_exit", "traction_spin")
    o_mid = _sev(summaries, "os_low")
    o_high = _sev(summaries, "os_high")
    f_lock = _sev(summaries, "lock_front")
    r_lock = _sev(summaries, "lock_rear")
    traction = _sev(summaries, "traction_spin")
    nervous = _sev(summaries, "steer_corrections")
    tires_hot = _sev(summaries, "tires_hot")
    tires_cold = _sev(summaries, "tires_cold")
    axle_imb = _sev(summaries, "tires_axle_imbalance")
    _ideal_add(target, current, "Front wing", u_mid * 10 + u_high * 8 + u_ent * 5 - o_ex * 5 - o_high * 3,
               "Load the nose. Front-limited traces get real wing, not a courtesy click.", acc)
    _ideal_add(target, current, "Rear wing", o_ex * 8 + o_mid * 4 + o_high * 7 + nervous * 4 - u_mid * 5 - u_high * 6,
               "Rear wing for stability vs drag.", acc)
    _ideal_add(target, current, "Front ARB", -u_mid * 7 - u_high * 4 + o_mid * 3 + o_ex * 2,
               "Softer front bar if the mid-corner is a push.", acc)
    _ideal_add(target, current, "Rear ARB", -o_ex * 7 - o_mid * 6 - o_high * 4 - traction * 5 + u_mid * 4,
               "Rear bar for rotation vs traction.", acc)
    _ideal_add(target, current, "On-throttle differential", -o_ex * 14 - traction * 12 + u_mid * 5,
               "On-throttle locking vs spin.", acc)
    _ideal_add(target, current, "Off-throttle differential", -u_ent * 10 + o_mid * 6 + r_lock * 4,
               "Off-throttle for entry rotation.", acc)
    _ideal_add(target, current, "Brake bias (% front)", -u_ent * 4 - f_lock * 4 + o_ex * 1 + r_lock * 3,
               "Bias for entry rotation and locking.", acc)
    _ideal_add(target, current, "Front ride height", -u_mid * 3 - u_high * 3 + tires_hot * 2, "Front ride height / rake.", acc)
    _ideal_add(target, current, "Rear ride height", u_mid * 4 - o_ex * 3, "Rear ride height / rake.", acc)
    _ideal_add(target, current, "Front spring", -u_ent * 5 - u_mid * 3 + o_ex * 3, "Softer front spring if the car will not turn.", acc)
    _ideal_add(target, current, "Rear spring", -traction * 7 - o_ex * 4 + u_mid * 2, "Rear spring for traction.", acc)
    _ideal_add(target, current, "Front camber", -u_mid * 3 + tires_hot * 2, "Front camber.", acc)
    _ideal_add(target, current, "Rear camber", -o_ex * 2 - tires_hot * 1, "Rear camber.", acc)
    _ideal_add(target, current, "Rear toe in", nervous * 8 + o_ex * 4 - u_mid * 3, "Rear toe-in for stability.", acc)
    _ideal_add(target, current, "Front toe out", u_mid * 4 + u_ent * 3, "A touch of front toe-out for turn-in.", acc)
    _ideal_add(target, current, "Front tire pressure", tires_hot * 6 - tires_cold * 8 + axle_imb * 2, "Front pressure from temperature.", acc)
    _ideal_add(target, current, "Rear tire pressure", tires_hot * 6 - tires_cold * 8 - axle_imb * 2, "Rear pressure from temperature.", acc)
    bp = target.get("Brake pressure")
    if f_lock > 0.45 and bp is not None and bp > 0.95:
        _ideal_add(target, current, "Brake pressure", -2, "Take pressure out if the fronts are locking.", acc)
    for param in SETUP_LIMITS:
        if param in acc:
            continue
        cur, sug = current.get(param), target.get(param)
        if cur is None or sug is None:
            continue
        clicks = _clicks_between(param, float(cur), float(sug))
        if not clicks:
            continue
        acc[param] = {
            "parameter": param, "direction": "increase" if clicks > 0 else "decrease",
            "from": float(cur), "to": float(sug), "steps": abs(clicks),
            "issue": "Track baseline", "issue_id": "ideal", "tier": "—",
            "reason": f"Came with the {type_label} baseline, not a click from your current car.",
            "weight": 1.0,
        }
    applied = [v for v in acc.values() if v.get("steps")]
    applied.sort(key=lambda a: -abs(float(a["to"]) - float(a["from"])))
    total_clicks = sum(abs(_clicks_between(a["parameter"], a["from"], a["to"])) for a in applied)
    shake = min(1.0, total_clicks / 40.0)
    ranked = sorted(summaries, key=lambda s: -s.criticality)
    dominant = ranked[0] if ranked else None
    if dominant:
        philosophy = (f"This is not a 1-click list. I started from a {type_label} baseline "
                      f"and rebuilt around {dominant.name.lower()}. It is allowed to look nothing like the car you just ran.")
        headline = f"Best guess: rebuild around {dominant.name.lower()}."
    else:
        philosophy = (f"This is not a 1-click list. I started from a {type_label} baseline "
                      "and wrote a full target sheet from the trace. It is allowed to look nothing like the car you just ran.")
        headline = "Best guess: a clean baseline for this track type."
    notes = ["Wet overlay: more wing, less on-throttle lock, bias a click rearward." if wet
             else "Dry sheet. If the circuit is actually wet, flip weather before you copy this."]
    if dominant:
        notes.append(f"Loudest thing in the trace: {dominant.name.lower()}.")
    if shake > 0.4:
        notes.append("Shake-up sheet. Put it on in practice, do three laps, then go back to Conservative clicks if you want.")
    rows = _setup_diff_rows(current, target)
    return {
        "mode": "best guess", "current": current, "suggested": target, "applied": applied,
        "skipped": [], "rows": rows, "max_changes": sum(1 for r in rows if r.get("Changed")),
        "tiers_used": ["S", "A", "B", "C"], "philosophy": philosophy, "headline": headline,
        "notes": notes, "track_type_used": tt, "weather": "wet" if wet else "dry", "shake_factor": shake,
    }


def main():
    st.set_page_config(page_title="Virtual Race Engineer", page_icon="🏎️", layout="wide")
    st.title("🏎️ Virtual Race Engineer")
    st.caption("F1 25 / F1 26 telemetry → diagnostics with frequency → ranked setup changes. Conservative / Aggressive / Best guess.")

    with st.sidebar:
        st.header("Session")
        uploaded = st.file_uploader("Telemetry file (TSV/CSV)", type=["csv", "tsv", "txt"])
        st.markdown("---")
        st.subheader("Thresholds")
        us_alpha = st.slider("Understeer α threshold (rad)", 0.01, 0.10, 0.03, 0.005)
        os_alpha = st.slider("Oversteer α threshold (rad)", 0.01, 0.10, 0.03, 0.005)
        lock_slip = st.slider("Lockup |κ| threshold", 0.03, 0.25, 0.08, 0.01)
        spin_slip = st.slider("Wheelspin κ threshold", 0.05, 0.35, 0.12, 0.01)
        min_count = st.number_input("Min event count to show issue", 1, 50, 2)
        min_presence = st.slider("Min lap presence %", 0, 100, 15)
        st.markdown("---")
        st.subheader("Suggested setup")
        setup_mode = st.radio(
            "Build mode",
            ["Conservative", "Aggressive", "Best guess"],
            index=0,
            help="Conservative: 1 click, 3 knobs. Aggressive: 1–2 clicks, 6 knobs. Best guess: full unconstrained sheet.",
        )
        track_choice, weather_choice = "Auto", "Dry"
        if setup_mode == "Best guess":
            track_choice = st.selectbox("Track type (best guess)", ["Auto", "High downforce", "Medium", "Low downforce"])
            weather_choice = st.radio("Weather", ["Dry", "Wet"], horizontal=True)
            st.caption("Best guess starts from a track-type baseline and rebuilds the whole sheet. Not 1-click advice.")
        else:
            st.caption("Starts from **your current setup**, applies non-conflicting clicks only, clamped to min/max.")

    if uploaded is None:
        st.info("Upload a telemetry export to begin. Logger format (tab-separated, 266 columns) is supported.")
        with st.expander("What this app assesses"):
            st.markdown(
                """
- **Balance:** low/med/high-speed understeer & oversteer (slip angles)
- **Phases:** entry / mid / exit
- **Brakes:** front & rear lockup
- **Traction:** exit wheelspin & power oversteer
- **Tires:** cold / hot / axle imbalance
- **Criticality tiers** S / A / B / C
- **Driver coach** and **session grade**
- **Suggested setup:** Conservative / Aggressive / **Best guess**
                """
            )
        return

    with st.spinner("Loading and analyzing…"):
        try:
            df = load_telemetry(uploaded)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return
        overview = session_overview(df)
        setup = extract_setup(df)
        events, summaries = run_diagnostics(df, us_alpha=us_alpha, os_alpha=os_alpha, lock_slip=lock_slip, spin_slip=spin_slip)
        shown = [s for s in summaries if s.count >= min_count and s.lap_presence_pct >= min_presence]
        if not shown:
            shown = summaries[:15]
        driver = analyze_driver(df)
        grade = session_grade(df, summaries, driver)
        changes = recommend_setup(shown if shown else summaries[:5], setup)
        if setup_mode == "Best guess":
            suggested = build_ideal_setup(
                setup, shown if shown else summaries,
                track_type=guess_track_type(overview.get("track"), overview.get("vmax_kph"), track_choice),
                weather=weather_choice,
            )
        else:
            suggested = build_suggested_setup(setup, shown if shown else summaries, mode=setup_mode.lower())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Track", str(overview["track"]))
    c2.metric("Completed laps", f"{len(overview['laps'])}")
    c3.metric("Best lap", f"L{overview['best_lap'][0]:.0f}  {format_lap_time(overview['best_lap'][1])}" if overview["best_lap"] else "—")
    c4.metric("Vmax", f"{overview['vmax_kph']:.0f} kph")
    c5.metric("Samples", f"{overview['samples']:,}")
    c6.metric("Session grade", f"{grade['score']:.0f}/100 ({grade['letter']})")

    if overview.get("lap_times"):
        st.markdown("**Completed lap times**")
        st.dataframe(pd.DataFrame([{"Lap": int(ln), "Time": format_lap_time(lt), "Seconds": round(lt, 3)} for ln, lt in overview["lap_times"]]), width="stretch", hide_index=True)
    else:
        st.warning("No completed lap times detected (need ≥55s flying laps).")

    rejected = getattr(extract_completed_lap_times, "last_rejected", None) or []
    if rejected:
        with st.expander(f"Ignored laps ({len(rejected)})"):
            st.dataframe(pd.DataFrame(rejected), width="stretch", hide_index=True)

    st.subheader("Session grade")
    st.caption(grade["disclaimer"])
    gcols = st.columns(len(grade["components"]))
    for col, (name, comp) in zip(gcols, grade["components"].items()):
        col.metric(name, f"{comp['score']:.0f}")
        col.caption(comp["detail"])

    st.subheader("Fix first (criticality tiers)")
    if shown:
        st.dataframe(pd.DataFrame([{
            "Tier": s.tier, "Issue": s.name, "Criticality": round(s.criticality, 2),
            "Presence %": round(s.lap_presence_pct, 1), "Per lap": round(s.events_per_lap, 2),
            "Severity": round(s.mean_severity, 2),
            "Hot spots (m)": ", ".join(f"{h:.0f}" for h in s.hot_corners_m[:3]),
        } for s in shown]), width="stretch", hide_index=True)

    st.subheader("Current setup (from telemetry)")
    if setup.get("table"):
        st.dataframe(pd.DataFrame(setup["table"]), width="stretch", hide_index=True)
        n = setup.get("named") or {}
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Front wing", format_setup_value("Front wing", _to_limit_units("Front wing", n.get("Front wing"))))
        a2.metric("Rear wing", format_setup_value("Rear wing", _to_limit_units("Rear wing", n.get("Rear wing"))))
        a3.metric("Front ARB", format_setup_value("Front ARB", _to_limit_units("Front ARB", n.get("Front ARB"))))
        a4.metric("Rear ARB", format_setup_value("Rear ARB", _to_limit_units("Rear ARB", n.get("Rear ARB"))))

    st.subheader(f"Suggested setup ({suggested['mode'].title()})")
    if suggested.get("mode") == "best guess":
        st.info(suggested.get("headline") or "Best guess target sheet.")
        st.write(suggested.get("philosophy") or "")
        for nline in suggested.get("notes") or []:
            st.caption(nline)
    else:
        st.caption(f"From your current car · tiers {', '.join(suggested['tiers_used'])} · max {suggested['max_changes']} knobs.")
    if suggested.get("applied"):
        st.markdown("**Changes applied to build this suggestion**")
        for a in suggested["applied"]:
            st.markdown(
                f"- **{a['parameter']}**: {format_setup_value(a['parameter'], a['from'])} → "
                f"**{format_setup_value(a['parameter'], a['to'])}** "
                f"(`{a['direction']}` ×{a['steps']}) — *[{a['tier']}] {a['issue']}*: {a['reason']}"
            )
    changed_rows = [r for r in suggested.get("rows") or [] if r.get("Changed")]
    if changed_rows:
        st.markdown("**Delta only**")
        st.dataframe(pd.DataFrame([{k: r[k] for k in ("Parameter", "Current", "Suggested", "Delta")} for r in changed_rows]), width="stretch", hide_index=True)
    with st.expander("Full setup card (current vs suggested)"):
        st.dataframe(pd.DataFrame([{k: r[k] for k in ("Parameter", "Current", "Suggested", "Delta")} for r in suggested.get("rows") or []]), width="stretch", hide_index=True)

    st.subheader("Driver coach (vs best lap)")
    for nline in driver.get("notes") or []:
        st.markdown(f"- {nline}")
    z = driver.get("time_loss_zones") or []
    if z:
        st.dataframe(pd.DataFrame([{
            "Distance (m)": int(x["distance_m"]), "Proxy loss (s)": round(x["time_loss_s"], 3),
            "Best lap kph": round(x["speed_ref_kph"], 1), "Other kph": round(x["speed_cmp_kph"], 1),
            "Δ kph": round(x["speed_delta_kph"], 1),
        } for x in z]), width="stretch", hide_index=True)

    st.subheader("Recommended setup changes")
    if not changes:
        st.success("No strong setup signals.")
    else:
        tier_by_name = {s.name: s.tier for s in shown}
        crit_by_name = {s.name: s.criticality for s in shown}
        by_issue = recommendations_by_issue(changes)
        ordered_names = sorted(by_issue.keys(), key=lambda n: ({"S": 0, "A": 1, "B": 2, "C": 3}.get(tier_by_name.get(n, "C"), 9), -crit_by_name.get(n, 0)))
        for issue_name in ordered_names:
            opts = by_issue[issue_name]
            feasible_opts = [o for o in opts if o.feasible]
            blocked_opts = [o for o in opts if not o.feasible]
            with st.container(border=True):
                st.markdown(f"### [{tier_by_name.get(issue_name, '?')}] {issue_name}")
                for ch in feasible_opts:
                    st.markdown(f"**{ch.option_label}: {ch.parameter}** — `{ch.direction}` · {ch.amount_hint}")
                    st.write(ch.reason)
                    st.caption(f"Validate: {ch.validation_metric}")
                if blocked_opts:
                    with st.expander(f"Blocked options ({len(blocked_opts)})"):
                        for ch in blocked_opts:
                            st.markdown(f"**{ch.option_label}: {ch.parameter}** — `{ch.direction}`")
                            st.caption(ch.blocked_reason or "Not feasible.")

    st.subheader("Issues (with frequency & criticality)")
    if shown:
        table = pd.DataFrame([{
            "Tier": s.tier, "Issue": s.name, "Criticality": round(s.criticality, 2), "Count": s.count,
            "Per lap": round(s.events_per_lap, 2), "Laps": f"{s.laps_present}/{s.total_laps}",
            "Presence %": round(s.lap_presence_pct, 1), "Avg severity": round(s.mean_severity, 2),
            "Confidence": round(s.confidence, 2),
            "Hot spots (m)": ", ".join(f"{h:.0f}" for h in s.hot_corners_m),
            "Example": s.sample_details[0] if s.sample_details else "",
        } for s in shown])
        st.dataframe(table, width="stretch", hide_index=True)
        fig = px.bar(table, x="Issue", y="Count", color="Tier", title="Issue frequency by tier",
                     category_orders={"Tier": ["S", "A", "B", "C"]})
        fig.update_layout(xaxis_tickangle=-35, height=400)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Telemetry snapshots")
    d = df[df["valid_sample"]].copy() if "valid_sample" in df.columns else df.copy()
    if not d.empty and "lap_number" in d.columns:
        lap_opts = sorted([int(x) for x in d["lap_number"].dropna().unique() if x >= 0])
        if lap_opts:
            lap_sel = st.selectbox("Lap for trace", lap_opts, index=len(lap_opts) - 1)
            ld = d[d["lap_number"] == lap_sel].sort_values("lap_distance")
            if len(ld) > 5:
                t1, t2 = st.tabs(["Speed / inputs", "Balance (slip angles)"])
                with t1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ld["lap_distance"], y=ld["speed_kph"], name="Speed kph", yaxis="y1"))
                    fig.add_trace(go.Scatter(x=ld["lap_distance"], y=ld["throttle"] * 100, name="Throttle %", yaxis="y2", opacity=0.7))
                    fig.add_trace(go.Scatter(x=ld["lap_distance"], y=ld["brake"] * 100, name="Brake %", yaxis="y2", opacity=0.7))
                    fig.update_layout(height=380, yaxis=dict(title="kph"),
                                      yaxis2=dict(title="Input %", overlaying="y", side="right", range=[0, 100]),
                                      legend=dict(orientation="h"), margin=dict(l=40, r=40, t=30, b=40))
                    st.plotly_chart(fig, width="stretch")
                with t2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=ld["lap_distance"], y=ld["alpha_balance"], name="α_f − α_r"))
                    fig2.add_hline(y=us_alpha, line_dash="dot", annotation_text="US")
                    fig2.add_hline(y=-os_alpha, line_dash="dot", annotation_text="OS")
                    fig2.update_layout(height=380, xaxis_title="Distance (m)", yaxis_title="Slip angle balance (rad)")
                    st.plotly_chart(fig2, width="stretch")

    with st.expander("Engineer notes / method"):
        st.markdown(
            f"""
**Phases:** entry (brake + long G), mid (low throttle/brake + high |ay|), exit (throttle + residual lateral).

**Understeer / oversteer:** `α_balance = |α_f| − |α_r|` with thresholds {us_alpha} / {os_alpha} rad.

**Frequency:** clustered by lap + 50 m so one long push corner is one event.

**Setup advice:** Conservative = 1 click from your car. Aggressive = 1–2. **Best guess** = full unconstrained sheet from a track-type baseline.
            """
        )


if __name__ == "__main__":
    main()
