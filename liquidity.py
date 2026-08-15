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
import json as _json
import logging
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
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


SESSION_LABELS: Dict[str, str] = {
    "asia": "Asia",
    "london": "London",
    "new_york": "New York",
    "new_york_pm": "New York PM",
}


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

def _compute_liquidity(
    intraday: pd.DataFrame, daily: pd.DataFrame, tolerance_pct: float, major_lookback: int
) -> Dict[str, Any]:
    """Pure computation over already-fetched data -- shared by the JSON
    endpoints and the chart builder so both read off the *same* intraday
    dataframe (and therefore the same bar indices/timestamps) rather than
    each triggering their own network fetch, which could drift by a bar."""
    reference_ny = now_local().astimezone(NY_TZ)

    swing_highs, swing_lows = find_swing_points(intraday, lookback=2)
    equal_highs = find_equal_levels(swing_highs, tolerance_pct)
    equal_lows = find_equal_levels(swing_lows, tolerance_pct)

    major = find_major_swings(intraday, lookback=major_lookback)
    sessions = previous_session_ranges(intraday, reference_ny)
    dwm = previous_day_week_month(daily, reference_ny)

    return {
        "equal_highs": equal_highs,
        "equal_lows": equal_lows,
        **major,
        "previous_session": sessions,
        **dwm,
    }


async def _fetch_liquidity_inputs(asset: str, timeframe: str) -> Tuple[Optional[str], pd.DataFrame, pd.DataFrame]:
    """Resolve the asset and fetch both the intraday and daily series once.
    Returns (symbol_or_None, intraday, daily); symbol is None for an
    unrecognized asset, intraday is empty if no data came back."""
    symbol = _resolve_symbol(asset)
    if not symbol:
        return None, pd.DataFrame(), pd.DataFrame()

    period = PERIOD_MAP.get(timeframe, "5d")
    interval = INTERVAL_MAP.get(timeframe, "30m")

    intraday = await fetch_asset_data(symbol, period, interval)
    daily = await fetch_daily_history(symbol, period="1y")
    return symbol, intraday, daily


async def _detect_liquidity_for_asset(
    asset: str, timeframe: str, tolerance_pct: float, major_lookback: int
) -> Dict[str, Any]:
    symbol, intraday, daily = await _fetch_liquidity_inputs(asset, timeframe)
    if not symbol:
        return {"error": f"Unknown asset '{asset}'. Available: {list(LIQUIDITY_SYMBOLS.keys())}"}
    if intraday.empty:
        return {"error": f"No {timeframe} data available for {asset}", "asset": asset}

    liquidity = _compute_liquidity(intraday, daily, tolerance_pct, major_lookback)

    return {
        "asset": asset,
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": now_local().isoformat(),
        "liquidity": liquidity,
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


# ==========================================
# TABLE PRESENTATION (flat rows, for an OpenBB Workspace TABLE widget)
# ==========================================

_DWM_LABELS: Dict[str, str] = {
    "previous_day": "Previous Day",
    "previous_week": "Previous Week",
    "previous_month": "Previous Month",
}


def _flatten_liquidity_levels(asset: str, timeframe: str, liq: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn the nested /liquidity/detect payload into flat rows -- one per
    level -- sorted highest price first, so a table reads like a price
    ladder: what's above, what's below, and by how much."""
    rows: List[Dict[str, Any]] = []

    for pool in liq["equal_highs"]:
        rows.append({
            "asset": asset, "timeframe": timeframe,
            "category": "Equal Highs/Lows", "label": "Equal High",
            "side": "high", "price": pool["price"],
            "detail": f"{pool['touches']} touches, {pool['price_range'][0]}-{pool['price_range'][1]}",
        })
    for pool in liq["equal_lows"]:
        rows.append({
            "asset": asset, "timeframe": timeframe,
            "category": "Equal Highs/Lows", "label": "Equal Low",
            "side": "low", "price": pool["price"],
            "detail": f"{pool['touches']} touches, {pool['price_range'][0]}-{pool['price_range'][1]}",
        })

    for point in liq["major_swing_highs"]:
        rows.append({
            "asset": asset, "timeframe": timeframe,
            "category": "Major Swings", "label": "Major Swing High",
            "side": "high", "price": point["price"], "detail": point["time"] or "",
        })
    for point in liq["major_swing_lows"]:
        rows.append({
            "asset": asset, "timeframe": timeframe,
            "category": "Major Swings", "label": "Major Swing Low",
            "side": "low", "price": point["price"], "detail": point["time"] or "",
        })

    for key, session in liq["previous_session"].items():
        if not session:
            continue
        label = SESSION_LABELS.get(key, key)
        for side in ("high", "low"):
            rows.append({
                "asset": asset, "timeframe": timeframe,
                "category": "Previous Session H/L", "label": f"{label} {side.title()}",
                "side": side, "price": session[side],
                "detail": f"{session['session_start']} to {session['session_end']}",
            })

    for key, dwm_label in _DWM_LABELS.items():
        bar = liq.get(key)
        if not bar:
            continue
        for side in ("high", "low"):
            rows.append({
                "asset": asset, "timeframe": timeframe,
                "category": "Previous Day/Week/Month H/L", "label": f"{dwm_label} {side.title()}",
                "side": side, "price": bar[side],
                "detail": f"period starting {bar['period_start']}",
            })

    rows.sort(key=lambda r: r["price"], reverse=True)
    return rows


@agent_app.get("/liquidity/levels")
async def liquidity_levels_table(
    asset: str = "XAUUSD",
    timeframe: str = "30m",
    tolerance_pct: float = 0.05,
    major_lookback: int = 6,
) -> List[Dict[str, Any]]:
    """
    Flat table of every detected liquidity level for one asset: category,
    label, side (high/low), price, and a human-readable detail column. Built
    for an OpenBB Workspace TABLE widget, but useful as plain tabular JSON
    for any consumer.
    """
    result = await _detect_liquidity_for_asset(asset, timeframe, tolerance_pct, major_lookback)
    if "error" in result:
        return [{"error": result["error"]}]
    return _flatten_liquidity_levels(asset, timeframe, result["liquidity"])


# ==========================================
# CHART PRESENTATION (Plotly figure, for an OpenBB Workspace CHART widget)
# ==========================================
#
# Colors below are the validated dark-mode categorical hues from the
# dataviz skill's reference palette (references/palette.md), restricted to
# the first three slots -- the only set that clears the palette's all-pairs
# CVD gate in both light and dark mode. A 4th hue (yellow, next to orange)
# fails that gate, so "previous session" and "previous day/week/month" share
# one hue (aqua) and are told apart by line dash + an always-visible direct
# label instead of a dedicated color -- identity never rests on color alone.
COLOR_EQUAL = "#3987e5"   # blue   -- equal highs/lows (repeated-swing pools)
COLOR_MAJOR = "#d95926"   # orange -- major swing highs/lows
COLOR_RANGE = "#199e70"   # aqua   -- previous session H/L + previous D/W/M H/L
COLOR_UP = "#0ca30c"      # status "good"     -- bullish candle
COLOR_DOWN = "#d03b3b"    # status "critical" -- bearish candle

CHART_BG = "#1a1a19"      # dark chart surface (palette.md) -- OpenBB Workspace
GRID_COLOR = "#2c2c2a"    # dark-mode gridline
INK_PRIMARY = "#ffffff"   # dark-mode primary ink (title)
INK_SECONDARY = "#c3c2b7"  # dark-mode secondary ink (legend/hover text)
INK_MUTED = "#898781"     # dark-mode muted ink (axis ticks)


# A chart reads as clutter past a handful of lines per family -- so unlike
# /liquidity/detect and /liquidity/levels (which return everything found),
# the chart keeps only the most relevant few per category: highest-touch
# equal-level pools, and the most recent major swings. Nothing is silently
# lost -- the fuller lists stay one call away via /liquidity/levels.
MAX_EQUAL_LINES_PER_SIDE = 3
MAX_MAJOR_LINES_PER_SIDE = 3


def _level_trace(x0: str, x1: str, price: float, color: str, dash: str,
                  legendgroup: str, legend_label: str, hover_label: str,
                  show_legend: bool) -> "go.Scatter":
    """One horizontal reference line running from where the level was set
    (x0) to the right edge of the visible chart (x1) -- it reads as "this
    liquidity has been resting here since x0," matching how ICT-style
    charting tools draw these levels."""
    return go.Scatter(
        x=[x0, x1],
        y=[price, price],
        mode="lines",
        line=dict(color=color, width=2, dash=dash),
        legendgroup=legendgroup,
        name=legend_label,
        showlegend=show_legend,
        hovertemplate=f"{hover_label}: %{{y:.5f}}<extra></extra>",
    )


async def _build_liquidity_chart(
    asset: str, timeframe: str, tolerance_pct: float, major_lookback: int
) -> Dict[str, Any]:
    symbol, candles, daily = await _fetch_liquidity_inputs(asset, timeframe)
    if not symbol:
        return {"error": f"Unknown asset '{asset}'. Available: {list(LIQUIDITY_SYMBOLS.keys())}"}
    if candles.empty:
        return {"error": f"No {timeframe} data available for {asset}"}

    liq = _compute_liquidity(candles, daily, tolerance_pct, major_lookback)
    idx = candles.index
    chart_start, chart_end = idx[0].isoformat(), idx[-1].isoformat()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=idx, open=candles["Open"], high=candles["High"],
        low=candles["Low"], close=candles["Close"],
        increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN,
        increasing_fillcolor=COLOR_UP, decreasing_fillcolor=COLOR_DOWN,
        name=asset, showlegend=False,
    ))

    # Equal highs/lows: keep the highest-touch pools (already sorted
    # touches-descending by find_equal_levels); anchor each line at its
    # most recent touch, since that's when the pool was last reinforced.
    eq_shown = False
    for pool in (liq["equal_highs"][:MAX_EQUAL_LINES_PER_SIDE]
                 + liq["equal_lows"][:MAX_EQUAL_LINES_PER_SIDE]):
        last_touch = idx[max(pool["bar_indices"])].isoformat()
        fig.add_trace(_level_trace(
            last_touch, chart_end, pool["price"], COLOR_EQUAL, "solid", "equal",
            "Equal Highs/Lows", f"Equal level ({pool['touches']} touches)", not eq_shown,
        ))
        eq_shown = True

    # Major swings: keep the most recent few per side (find_major_swings
    # returns them in chronological order), anchored at their own bar.
    major_shown = False
    for point in liq["major_swing_highs"][-MAX_MAJOR_LINES_PER_SIDE:]:
        fig.add_trace(_level_trace(
            point["time"] or chart_start, chart_end, point["price"], COLOR_MAJOR, "dot", "major",
            "Major Swings", "Major swing high", not major_shown,
        ))
        major_shown = True
    for point in liq["major_swing_lows"][-MAX_MAJOR_LINES_PER_SIDE:]:
        fig.add_trace(_level_trace(
            point["time"] or chart_start, chart_end, point["price"], COLOR_MAJOR, "dot", "major",
            "Major Swings", "Major swing low", not major_shown,
        ))
        major_shown = True

    # Previous session H/L and prior D/W/M H/L: anchor at when that period
    # started (may fall before the visible window for week/month -- Plotly
    # simply clips the line to the plot area, which is the desired look).
    session_shown = False
    for key, session in liq["previous_session"].items():
        if not session:
            continue
        label = SESSION_LABELS.get(key, key)
        for side in ("high", "low"):
            fig.add_trace(_level_trace(
                session["session_start"], chart_end, session[side], COLOR_RANGE, "dashdot", "session",
                "Previous Session H/L", f"{label} {side}", not session_shown,
            ))
            session_shown = True

    dwm_shown = False
    for key, dwm_label in _DWM_LABELS.items():
        bar = liq.get(key)
        if not bar:
            continue
        for side in ("high", "low"):
            fig.add_trace(_level_trace(
                bar["period_start"], chart_end, bar[side], COLOR_RANGE, "longdash", "dwm",
                "Prior Day/Week/Month H/L", f"{dwm_label} {side}", not dwm_shown,
            ))
            dwm_shown = True

    fig.update_layout(
        title=f"{asset} -- {timeframe} Liquidity Map",
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=INK_SECONDARY),
        title_font=dict(color=INK_PRIMARY),
        # Pin the visible x-range to the candles themselves. Session/D-W-M
        # lines are anchored at their true (often much earlier) start so
        # they read as "resting since x" on hover, but letting THAT drive
        # autorange would squeeze the actual price action into a sliver --
        # Plotly still clips those lines cleanly at this fixed left edge.
        xaxis=dict(gridcolor=GRID_COLOR, rangeslider=dict(visible=False),
                    color=INK_MUTED, range=[chart_start, chart_end]),
        yaxis=dict(gridcolor=GRID_COLOR, color=INK_MUTED),
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return _json.loads(fig.to_json())


@agent_app.get("/liquidity/chart")
async def liquidity_chart(
    asset: str = "XAUUSD",
    timeframe: str = "30m",
    tolerance_pct: float = 0.05,
    major_lookback: int = 6,
) -> Dict[str, Any]:
    """
    Candlestick chart with liquidity levels drawn as horizontal reference
    lines: the top equal highs/lows pools by touch count, the most recent
    major swings, previous session H/L, and previous day/week/month H/L.
    Levels are capped per category so the chart stays readable -- use
    /liquidity/levels or /liquidity/detect for the complete, uncapped list.
    Returns a full Plotly figure as JSON, the format an OpenBB Workspace
    CHART widget expects
    (docs.openbb.co/workspace/developers/widget-types/plotly-charts).
    """
    return await _build_liquidity_chart(asset, timeframe, tolerance_pct, major_lookback)
