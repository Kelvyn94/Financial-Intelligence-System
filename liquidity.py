"""
Liquidity Level Detection for Financial Intelligence System

Identifies resting liquidity pools that ICT-style price action is drawn to:
  - Equal highs / equal lows  (repeated swing points -> clustered stops)
  - Major swing highs / lows  (structural highs/lows, wider fractal lookback
    than the minor swing structure used for SMT divergence)
  - Previous session high/low (Asia, London, New York, New York PM -- session
    boundaries are read in New York time, the ICT convention, regardless of
    what timezone the instrument itself trades in)
  - Previous day / week / month high/low

This module adds routes onto the existing `agent_app` (defined in agent.py).
main.py mounts agent_app at /agent, so these become reachable at
/agent/liquidity/detect and /agent/liquidity/detect_all. It reuses
fetch_asset_data / find_swing_points / now_local from agent.py rather than
duplicating them.
"""

import asyncio
import logging
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from agent import (
    agent_app,
    fetch_asset_data,
    find_swing_points,
    now_local,
    PERIOD_MAP,
    INTERVAL_MAP,
)

logger = logging.getLogger("liquidity")

# ICT session / kill-zone boundaries are conventionally quoted in New York
# time regardless of where the instrument itself trades -- that's the
# reference clock this module uses to bucket candles into sessions.
NY_TZ = ZoneInfo("America/New_York")

# (session_name, start_time, end_time) in New York local time.
# end <= start means the session crosses midnight (e.g. Asia 20:00 -> 00:00).
SESSIONS: List[Tuple[str, dt_time, dt_time]] = [
    ("asia", dt_time(20, 0), dt_time(0, 0)),
    ("london", dt_time(2, 0), dt_time(5, 0)),
    ("new_york", dt_time(7, 0), dt_time(10, 0)),
    ("new_york_pm", dt_time(13, 30), dt_time(16, 0)),
]

# Assets this endpoint understands, mirroring the symbols already used
# elsewhere in the app (agent.py's SMT_GROUPS, main.py's ASSET_SYMBOLS).
LIQUIDITY_SYMBOLS: Dict[str, str] = {
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "GBPUSD": "GBPUSD=X",
    "EURUSD": "EURUSD=X",
    "NASDAQ": "NQ=F",
    "ES": "ES=F",
    "YM": "YM=F",
}


def _resolve_symbol(asset: str) -> Optional[str]:
    return LIQUIDITY_SYMBOLS.get(asset.upper())


# ==========================================
# EQUAL HIGHS / LOWS
# ==========================================

def find_equal_levels(points: List[Tuple[int, float]], tolerance_pct: float) -> List[Dict[str, Any]]:
    """
    Group swing points that sit within `tolerance_pct`% of each other into
    liquidity pools ("equal highs" / "equal lows"). ICT equal highs/lows are
    swing points repeated at (near) the same price -- each repeat adds resting
    stop liquidity at that level, which is why 2+ touches define a pool and a
    lone swing point does not.

    Returns clusters sorted by touch count (most-touched first): price
    (cluster average), touches, the min/max price actually seen in the
    cluster, and the bar indices involved.
    """
    if len(points) < 2:
        return []

    ordered = sorted(points, key=lambda p: p[1])
    clusters: List[List[Tuple[int, float]]] = []
    current = [ordered[0]]

    for point in ordered[1:]:
        cluster_ref_price = current[0][1]
        if cluster_ref_price != 0 and abs(point[1] - cluster_ref_price) / abs(cluster_ref_price) * 100 <= tolerance_pct:
            current.append(point)
        else:
            clusters.append(current)
            current = [point]
    clusters.append(current)

    pools = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        prices = [p[1] for p in cluster]
        indices = [p[0] for p in cluster]
        pools.append({
            "price": round(sum(prices) / len(prices), 5),
            "touches": len(cluster),
            "price_range": [round(min(prices), 5), round(max(prices), 5)],
            "bar_indices": indices,
        })
    pools.sort(key=lambda p: p["touches"], reverse=True)
    return pools


# ==========================================
# SESSION WINDOWS
# ==========================================

def _most_recent_completed_window(
    session_start: dt_time, session_end: dt_time, reference_ny: datetime
) -> Tuple[datetime, datetime]:
    """
    Return the (start, end) datetimes (tz-aware, America/New_York) of the
    most recently *completed* occurrence of a daily session window, relative
    to `reference_ny` (also tz-aware, America/New_York).
    """
    anchor_date = reference_ny.date()
    start_dt = datetime.combine(anchor_date, session_start, tzinfo=NY_TZ)
    if session_end <= session_start:
        end_dt = datetime.combine(anchor_date, session_end, tzinfo=NY_TZ) + timedelta(days=1)
    else:
        end_dt = datetime.combine(anchor_date, session_end, tzinfo=NY_TZ)

    while end_dt > reference_ny:
        start_dt -= timedelta(days=1)
        end_dt -= timedelta(days=1)

    return start_dt, end_dt


def _ensure_ny_index(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `data` whose index is tz-aware in America/New_York."""
    df = data.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(NY_TZ)
    return df


def previous_session_ranges(data: pd.DataFrame, reference_ny: datetime) -> Dict[str, Any]:
    """
    Compute the high/low of the most recently completed occurrence of each
    ICT session (Asia, London, New York, New York PM), reading candle
    timestamps against New York time.
    """
    if data.empty:
        return {name: None for name, _, _ in SESSIONS}

    df = _ensure_ny_index(data)
    results: Dict[str, Any] = {}

    for name, start_t, end_t in SESSIONS:
        start_dt, end_dt = _most_recent_completed_window(start_t, end_t, reference_ny)
        window = df[(df.index >= start_dt) & (df.index < end_dt)]
        if window.empty:
            results[name] = None
            continue
        results[name] = {
            "high": round(float(window["High"].max()), 5),
            "low": round(float(window["Low"].min()), 5),
            "session_start": start_dt.isoformat(),
            "session_end": end_dt.isoformat(),
        }

    return results


# ==========================================
# PREVIOUS DAY / WEEK / MONTH
# ==========================================

async def fetch_daily_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    def _fetch():
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval="1d")
        return data if not data.empty else pd.DataFrame()

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as e:
        logger.error(f"Error fetching daily history for {symbol}: {e}")
        return pd.DataFrame()


def _previous_completed_bar(resampled: pd.DataFrame, reference_ny: datetime) -> Optional[Dict[str, Any]]:
    """
    Given OHLC data resampled to day/week/month buckets, return the most
    recently *fully closed* bucket relative to `reference_ny` -- i.e. skip a
    trailing bucket that's still in progress.
    """
    if resampled.empty:
        return None

    df = resampled.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)

    completed = df[df.index <= reference_ny]
    if completed.empty:
        return None
    if len(completed) < 2:
        row = completed.iloc[-1]
        period_start = completed.index[-1]
    else:
        row = completed.iloc[-2]
        period_start = completed.index[-2]

    return {
        "high": round(float(row["High"]), 5),
        "low": round(float(row["Low"]), 5),
        "period_start": period_start.isoformat(),
    }


def previous_day_week_month(daily_data: pd.DataFrame, reference_ny: datetime) -> Dict[str, Any]:
    if daily_data.empty:
        return {"previous_day": None, "previous_week": None, "previous_month": None}

    df = daily_data.copy()
    weekly = df.resample("W-FRI").agg({"High": "max", "Low": "min"}).dropna()
    monthly = df.resample("ME").agg({"High": "max", "Low": "min"}).dropna()

    return {
        "previous_day": _previous_completed_bar(df[["High", "Low"]], reference_ny),
        "previous_week": _previous_completed_bar(weekly, reference_ny),
        "previous_month": _previous_completed_bar(monthly, reference_ny),
    }


# ==========================================
# MAJOR SWINGS
# ==========================================

def find_major_swings(data: pd.DataFrame, lookback: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Major swing highs/lows use a wider fractal lookback than the lookback=2
    used for SMT's minor market structure -- fewer, more structurally
    significant points, which is what ICT liquidity draws (buy-side /
    sell-side liquidity) sit above/below.
    """
    swing_highs, swing_lows = find_swing_points(data, lookback=lookback)
    idx = data.index

    def _fmt(points):
        out = []
        for i, price in points:
            out.append({
                "price": round(price, 5),
                "time": idx[i].isoformat() if i < len(idx) else None,
            })
        return out

    return {"major_swing_highs": _fmt(swing_highs), "major_swing_lows": _fmt(swing_lows)}


# ==========================================
# ENDPOINTS
# ==========================================

async def _detect_liquidity_for_asset(
    asset: str, timeframe: str, tolerance_pct: float, major_lookback: int
) -> Dict[str, Any]:
    symbol = _resolve_symbol(asset)
    if not symbol:
        return {"error": f"Unknown asset '{asset}'. Available: {list(LIQUIDITY_SYMBOLS.keys())}"}

    period = PERIOD_MAP.get(timeframe, "5d")
    interval = INTERVAL_MAP.get(timeframe, "30m")

    intraday = await fetch_asset_data(symbol, period, interval)
    daily = await fetch_daily_history(symbol, period="1y")

    if intraday.empty:
        return {"error": f"No {timeframe} data available for {asset}", "asset": asset}

    reference_ny = now_local().astimezone(NY_TZ)

    swing_highs, swing_lows = find_swing_points(intraday, lookback=2)
    equal_highs = find_equal_levels(swing_highs, tolerance_pct)
    equal_lows = find_equal_levels(swing_lows, tolerance_pct)

    major = find_major_swings(intraday, lookback=major_lookback)
    sessions = previous_session_ranges(intraday, reference_ny)
    dwm = previous_day_week_month(daily, reference_ny)

    return {
        "asset": asset,
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": now_local().isoformat(),
        "liquidity": {
            "equal_highs": equal_highs,
            "equal_lows": equal_lows,
            **major,
            "previous_session": sessions,
            **dwm,
        },
    }


@agent_app.get("/liquidity/detect")
async def detect_liquidity(
    asset: str = "XAUUSD",
    timeframe: str = "30m",
    tolerance_pct: float = 0.05,
    major_lookback: int = 6,
) -> Dict[str, Any]:
    """
    Identify resting liquidity for a single asset: equal highs/lows, major
    swing highs/lows, previous session ranges (Asia/London/New York/New York
    PM in NY time), and previous day/week/month ranges.

    Query Parameters:
        asset: One of XAUUSD, XAGUSD, GBPUSD, EURUSD, NASDAQ, ES, YM
        timeframe: Intraday timeframe used for equal-highs/lows, major swings,
                   and session ranges (default 30m)
        tolerance_pct: % price tolerance for grouping "equal" highs/lows
        major_lookback: fractal lookback (bars each side) used to define a
                        "major" swing point, vs the tighter lookback=2 used
                        for SMT's minor market structure
    """
    return await _detect_liquidity_for_asset(asset, timeframe, tolerance_pct, major_lookback)


@agent_app.get("/liquidity/detect_all")
async def detect_liquidity_all(
    timeframe: str = "30m",
    tolerance_pct: float = 0.05,
    major_lookback: int = 6,
) -> Dict[str, Any]:
    """Run /liquidity/detect across every asset this app tracks."""
    results = {}
    for asset in LIQUIDITY_SYMBOLS:
        results[asset] = await _detect_liquidity_for_asset(asset, timeframe, tolerance_pct, major_lookback)
    return {"timeframe": timeframe, "timestamp": now_local().isoformat(), "assets": results}
