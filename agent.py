"""
SMT Detection Agent for Financial Intelligence System
Monitors correlations and divergences across assets and timeframes
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional
import httpx
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Email notifications are disabled (see SMTMonitor.send_alerts below);
# no import from notifications.py needed here anymore.

# All timestamps shown to the user (API responses, logs) use Kenya time,
# regardless of what timezone the host server itself runs in (Render runs UTC).
NAIROBI_TZ = ZoneInfo("Africa/Nairobi")

def now_local() -> datetime:
    """Current time in Africa/Nairobi (EAT, UTC+3)."""
    return datetime.now(NAIROBI_TZ)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI for agent
agent_app = FastAPI(
    title="SMT Detection Agent",
    description="Monitors SMT divergences across assets and timeframes",
    version="1.0.0"
)

# Enable CORS
agent_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# CONFIGURATION
# ==========================================

# Asset correlation groups for SMT detection
SMT_GROUPS = {
    "precious_metals": {
        "base": "XAUUSD",
        "correlated": ["XAGUSD", "XAUEUR", "XAUGBP"],
        "symbols": {
            "XAUUSD": "GC=F",
            "XAGUSD": "SI=F",
            "XAUEUR": "XAUUSD=X",
            "XAUGBP": "XAUUSD=X"
        }
    },
    "forex": {
        "base": "GBPUSD",
        "correlated": ["EURUSD"],
        "symbols": {
            "GBPUSD": "GBPUSD=X",
            "EURUSD": "EURUSD=X"
        }
    },
    "futures": {
        "base": "Nasdaq",
        "correlated": ["ES", "YM"],
        "symbols": {
            "Nasdaq": "NQ=F",
            "ES": "ES=F",
            "YM": "YM=F"
        }
    }
}

# Timeframes to monitor
TIMEFRAMES = ["30m", "1h", "4h", "1d", "1w"]

# Yahoo Finance period mapping
PERIOD_MAP = {
    "30m": "5d",
    "1h": "5d",
    "4h": "5d",
    "1d": "6mo",
    "1w": "1y"
}

# Interval mapping for yfinance
INTERVAL_MAP = {
    "30m": "30m",
    "1h": "60m",
    "4h": "1h",  # yfinance doesn't have 4h, we'll resample
    "1d": "1d",
    "1w": "1wk"
}

# ==========================================
# DATA FETCHING FUNCTIONS
# ==========================================

async def fetch_asset_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch historical data for an asset using yfinance.

    yfinance's calls are blocking (plain network I/O), so this runs them on
    a background thread via asyncio.to_thread(). Without that, one blocking
    call would freeze the whole app's event loop — every other request would
    stall until this one finished.

    Args:
        symbol: Yahoo Finance symbol
        period: Period to fetch (e.g., '5d', '6mo')
        interval: Data interval (e.g., '30m', '60m', '1d')

    Returns:
        DataFrame with OHLCV data
    """
    def _fetch():
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)

        if data.empty:
            logger.warning(f"No data for {symbol} with period={period}, interval={interval}")
            return pd.DataFrame()

        # Resample to 4h if needed
        if interval == "1h" and period == "5d":
            # Resample 1h data to 4h
            data = data.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

        return data

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def find_swing_points(data: pd.DataFrame, lookback: int = 2):
    """
    Identify swing highs and swing lows using a fractal method.

    In ICT (Inner Circle Trader) methodology, market structure is read from
    swing highs and swing lows directly — not from moving averages or
    oscillators. A swing high is a candle whose high is greater than the
    `lookback` candles on either side of it; a swing low is the mirror case.
    These points mark resting liquidity (buy-side above swing highs,
    sell-side below swing lows) that price is drawn back to.

    Returns:
        (swing_highs, swing_lows) — each a list of (index, price) tuples,
        in chronological order.
    """
    highs = data['High'].values
    lows = data['Low'].values
    n = len(highs)

    swing_highs = []
    swing_lows = []

    for i in range(lookback, n - lookback):
        left_highs = highs[i - lookback:i]
        right_highs = highs[i + 1:i + lookback + 1]
        if highs[i] > left_highs.max() and highs[i] > right_highs.max():
            swing_highs.append((i, float(highs[i])))

        left_lows = lows[i - lookback:i]
        right_lows = lows[i + 1:i + lookback + 1]
        if lows[i] < left_lows.min() and lows[i] < right_lows.min():
            swing_lows.append((i, float(lows[i])))

    return swing_highs, swing_lows

def detect_smt_divergence(data1: pd.DataFrame, data2: pd.DataFrame,
                          timeframe: str) -> Dict[str, Any]:
    """
    Detect genuine SMT (Smart Money Technique / "Smart Money Divergence")
    between two correlated assets, per ICT methodology.

    SMT divergence is a market-structure concept, not an indicator signal.
    Two historically correlated instruments (e.g. GBPUSD/EURUSD, ES/NQ,
    XAUUSD/XAGUSD) are expected to confirm each other's swing highs and
    swing lows. SMT divergence happens when one instrument sweeps to a new
    high/low while the correlated instrument FAILS to confirm — it prints a
    lower high (or higher low) instead. That non-confirmation is read as a
    sign smart money isn't behind the move that did make the new extreme,
    which is why SMT divergence at a liquidity point is treated as an early
    reversal signal.

    This deliberately does NOT use RSI, MACD, or any other lagging
    oscillator — ICT's approach is price-action / market-structure based,
    working directly off swing highs and swing lows rather than derived
    indicators.

    Args:
        data1: DataFrame for the base asset
        data2: DataFrame for the correlated asset
        timeframe: Timeframe being analyzed

    Returns:
        Dict with divergence detection results
    """
    MIN_BARS = 15  # need enough bars to form at least two confirmed swings
    if data1.empty or data2.empty or len(data1) < MIN_BARS or len(data2) < MIN_BARS:
        return {"divergence": False, "reason": "Insufficient data for swing structure"}

    try:
        swing_highs1, swing_lows1 = find_swing_points(data1)
        swing_highs2, swing_lows2 = find_swing_points(data2)

        close1 = data1['Close'].values
        close2 = data2['Close'].values
        change1 = ((close1[-1] - close1[-5]) / close1[-5]) * 100 if len(close1) >= 5 else 0.0
        change2 = ((close2[-1] - close2[-5]) / close2[-5]) * 100 if len(close2) >= 5 else 0.0

        bearish_smt = False  # base sweeps to a higher high, correlated fails to confirm
        bullish_smt = False  # base sweeps to a lower low, correlated fails to confirm

        # --- Bearish SMT: base prints a higher high, correlated asset doesn't ---
        if len(swing_highs1) >= 2 and len(swing_highs2) >= 2:
            _, last1 = swing_highs1[-1]
            _, prev1 = swing_highs1[-2]
            _, last2 = swing_highs2[-1]
            _, prev2 = swing_highs2[-2]

            asset1_higher_high = last1 > prev1
            asset2_higher_high = last2 > prev2

            if asset1_higher_high and not asset2_higher_high:
                bearish_smt = True

        # --- Bullish SMT: base prints a lower low, correlated asset doesn't ---
        if len(swing_lows1) >= 2 and len(swing_lows2) >= 2:
            _, last1 = swing_lows1[-1]
            _, prev1 = swing_lows1[-2]
            _, last2 = swing_lows2[-1]
            _, prev2 = swing_lows2[-2]

            asset1_lower_low = last1 < prev1
            asset2_lower_low = last2 < prev2

            if asset1_lower_low and not asset2_lower_low:
                bullish_smt = True

        divergence_detected = bullish_smt or bearish_smt
        div_type = "bullish" if bullish_smt else "bearish" if bearish_smt else "none"

        return {
            "divergence": divergence_detected,
            "type": div_type,
            "change1": float(change1),
            "change2": float(change2),
            "price1": float(close1[-1]),
            "price2": float(close2[-1]),
            "message": (
                f"SMT divergence ({div_type}) on {timeframe}: correlated asset failed to "
                f"confirm the {'swing low' if bullish_smt else 'swing high'} sweep"
                if divergence_detected else "No SMT divergence"
            )
        }

    except Exception as e:
        logger.error(f"Error detecting divergence: {e}")
        return {"divergence": False, "reason": str(e)}

# ==========================================
# SMT MONITORING ENGINE
# ==========================================

class SMTMonitor:
    """Main monitoring engine for SMT detection"""
    
    def __init__(self):
        self.detections = []
        self.last_check = None
        
    async def check_timeframe(self, group_name: str, group_config: Dict, 
                              timeframe: str) -> List[Dict]:
        """
        Check SMT divergence for a group on a specific timeframe
        """
        results = []
        base_symbol = group_config["symbols"][group_config["base"]]
        correlated_symbols = [group_config["symbols"][c] for c in group_config["correlated"]]
        
        period = PERIOD_MAP.get(timeframe, "5d")
        interval = INTERVAL_MAP.get(timeframe, "30m")
        
        # Fetch base asset data
        base_data = await fetch_asset_data(base_symbol, period, interval)
        if base_data.empty:
            return results
        
        # Check against each correlated asset
        for i, corr_symbol in enumerate(correlated_symbols):
            corr_data = await fetch_asset_data(corr_symbol, period, interval)
            if corr_data.empty:
                continue
            
            # Align data
            min_len = min(len(base_data), len(corr_data))
            base_aligned = base_data.iloc[-min_len:]
            corr_aligned = corr_data.iloc[-min_len:]
            
            # Detect divergence
            divergence = detect_smt_divergence(base_aligned, corr_aligned, timeframe)
            
            if divergence.get("divergence", False):
                results.append({
                    "group": group_name,
                    "base": group_config["base"],
                    "correlated": group_config["correlated"][i],
                    "timeframe": timeframe,
                    "type": divergence.get("type", "unknown"),
                    "message": divergence.get("message", ""),
                    "change1": divergence.get("change1", 0),
                    "change2": divergence.get("change2", 0),
                    "price1": divergence.get("price1", 0),
                    "price2": divergence.get("price2", 0),
                    "timestamp": now_local().isoformat()
                })
        
        return results
    
    async def run_full_check(self) -> Dict[str, Any]:
        """
        Run full SMT check across all groups and timeframes
        """
        logger.info("Starting SMT detection scan...")
        
        all_results = []
        
        for group_name, group_config in SMT_GROUPS.items():
            for timeframe in TIMEFRAMES:
                try:
                    results = await self.check_timeframe(group_name, group_config, timeframe)
                    if results:
                        all_results.extend(results)
                        logger.info(f"Found {len(results)} divergences in {group_name} on {timeframe}")
                except Exception as e:
                    logger.error(f"Error checking {group_name} on {timeframe}: {e}")
        
        self.last_check = now_local()
        
        # Send notifications for any detections
        if all_results:
            await self.send_alerts(all_results)
        
        return {
            "timestamp": self.last_check.isoformat(),
            "detections": all_results,
            "total_count": len(all_results)
        }
    
    async def send_alerts(self, detections: List[Dict]):
        """
        Email notifications are disabled — this just logs what would have
        been sent. See notifications.py if you want to re-enable email later.
        """
        if not detections:
            return

        logger.info(f"{len(detections)} divergence(s) detected (email notifications are disabled)")

# ==========================================
# FASTAPI ENDPOINTS
# ==========================================

monitor = SMTMonitor()

@agent_app.get("/agents.json")
async def get_agent_config():
    """Agent configuration for OpenBB Workspace"""
    return {
        "smt_detector": {
            "name": "SMT Detector",
            "description": "Detects SMT divergences across correlated assets",
            "endpoint": "smt/detect",
            "type": "agent"
        }
    }

@agent_app.get("/smt/detect")
async def detect_smt(
    group: str = None,
    timeframe: str = None
) -> Dict[str, Any]:
    """
    Trigger SMT detection manually
    
    Query Parameters:
        group: Specific group to check (precious_metals, forex, futures)
        timeframe: Specific timeframe (30m, 1h, 4h, 1d, 1w)
    """
    if group and group not in SMT_GROUPS:
        return {"error": f"Invalid group. Available: {list(SMT_GROUPS.keys())}"}
    
    if timeframe and timeframe not in TIMEFRAMES:
        return {"error": f"Invalid timeframe. Available: {TIMEFRAMES}"}
    
    groups_to_check = [group] if group else list(SMT_GROUPS.keys())
    timeframes_to_check = [timeframe] if timeframe else TIMEFRAMES
    
    all_results = []
    
    for g in groups_to_check:
        group_config = SMT_GROUPS[g]
        for tf in timeframes_to_check:
            results = await monitor.check_timeframe(g, group_config, tf)
            if results:
                all_results.extend(results)
    
    return {
        "groups": groups_to_check,
        "timeframes": timeframes_to_check,
        "detections": all_results,
        "total": len(all_results),
        "timestamp": now_local().isoformat()
    }

@agent_app.get("/smt/status")
async def get_status():
    """Get current monitoring status"""
    return {
        "status": "running",
        "last_check": monitor.last_check.isoformat() if monitor.last_check else None,
        "timeframes_monitored": TIMEFRAMES,
        "groups_monitored": list(SMT_GROUPS.keys()),
        "asset_pairs": [
            {
                "base": config["base"],
                "correlated": config["correlated"]
            }
            for config in SMT_GROUPS.values()
        ]
    }

@agent_app.post("/smt/check")
async def force_check():
    """Force an immediate full scan"""
    result = await monitor.run_full_check()
    return result

@agent_app.get("/")
async def root():
    return {
        "service": "SMT Detection Agent",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "agents.json": "/agents.json",
            "detect": "/smt/detect",
            "status": "/smt/status",
            "force_check": "/smt/check (POST)"
        }
    }

# ==========================================
# RUN THE AGENT
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:agent_app", host="0.0.0.0", port=6901, reload=True)