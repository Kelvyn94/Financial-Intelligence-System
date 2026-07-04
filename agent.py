"""
SMT Detection Agent for Financial Intelligence System
Monitors correlations and divergences across assets and timeframes
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import httpx
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import notification functions
from notifications import send_notification, send_bulk_notification

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

def fetch_asset_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch historical data for an asset using yfinance
    
    Args:
        symbol: Yahoo Finance symbol
        period: Period to fetch (e.g., '5d', '6mo')
        interval: Data interval (e.g., '30m', '60m', '1d')
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
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
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def detect_smt_divergence(data1: pd.DataFrame, data2: pd.DataFrame, 
                          timeframe: str) -> Dict[str, Any]:
    """
    Detect Smart Money Technique (SMT) divergence between two assets
    
    Args:
        data1: DataFrame for asset 1
        data2: DataFrame for asset 2
        timeframe: Timeframe being analyzed
    
    Returns:
        Dict with divergence detection results
    """
    if data1.empty or data2.empty or len(data1) < 5 or len(data2) < 5:
        return {"divergence": False, "reason": "Insufficient data"}
    
    try:
        # Get last 5 closing prices
        close1 = data1['Close'].values[-5:]
        close2 = data2['Close'].values[-5:]
        
        # Calculate changes
        change1 = ((close1[-1] - close1[0]) / close1[0]) * 100
        change2 = ((close2[-1] - close2[0]) / close2[0]) * 100
        
        # Detect bullish divergence (asset1 making higher high, asset2 lower high)
        bullish_divergence = False
        bearish_divergence = False
        
        # Check for price peaks
        peak1 = max(close1)
        peak2 = max(close2)
        peak1_idx = len(close1) - 1 - list(reversed(close1)).index(peak1)
        peak2_idx = len(close2) - 1 - list(reversed(close2)).index(peak2)
        
        # Simple SMT logic: compare recent peaks
        if peak1 > close1[0] and peak2 < close2[0]:
            bullish_divergence = True
        elif peak1 < close1[0] and peak2 > close2[0]:
            bearish_divergence = True
        
        # Also check RSI divergence if data available
        rsi1 = calculate_rsi(data1['Close'].values)
        rsi2 = calculate_rsi(data2['Close'].values)
        
        rsi_divergence = False
        if rsi1 and rsi2:
            if (data1['Close'].iloc[-1] > data1['Close'].iloc[-5] and 
                rsi1[-1] < rsi1[-5] and
                data2['Close'].iloc[-1] < data2['Close'].iloc[-5] and
                rsi2[-1] > rsi2[-5]):
                rsi_divergence = True
        
        divergence_detected = bullish_divergence or bearish_divergence or rsi_divergence
        
        return {
            "divergence": divergence_detected,
            "type": "bullish" if bullish_divergence else "bearish" if bearish_divergence else "rsi" if rsi_divergence else "none",
            "change1": float(change1),
            "change2": float(change2),
            "price1": float(close1[-1]),
            "price2": float(close2[-1]),
            "message": f"SMT divergence detected on {timeframe} timeframe"
        }
        
    except Exception as e:
        logger.error(f"Error detecting divergence: {e}")
        return {"divergence": False, "reason": str(e)}

def calculate_rsi(values, period=14):
    """Calculate RSI for divergence detection"""
    try:
        if len(values) < period + 1:
            return None
        
        deltas = pd.Series(values).diff()
        gain = (deltas.where(deltas > 0, 0)).rolling(window=period).mean()
        loss = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    except Exception:
        return None

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
        base_data = fetch_asset_data(base_symbol, period, interval)
        if base_data.empty:
            return results
        
        # Check against each correlated asset
        for i, corr_symbol in enumerate(correlated_symbols):
            corr_data = fetch_asset_data(corr_symbol, period, interval)
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
                    "timestamp": datetime.now().isoformat()
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
        
        self.last_check = datetime.now()
        
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
        Send email notifications for detections
        """
        if not detections:
            return
        
        # Group by timeframe for better readability
        by_timeframe = {}
        for d in detections:
            tf = d.get("timeframe", "unknown")
            if tf not in by_timeframe:
                by_timeframe[tf] = []
            by_timeframe[tf].append(d)
        
        # Build email content
        subject = f"SMT Alert: {len(detections)} divergence(s) detected"
        
        body = "SMT Divergence Detection Report\n"
        body += "=" * 50 + "\n\n"
        
        for timeframe, items in by_timeframe.items():
            body += f"📊 {timeframe.upper()} TIMEFRAME\n"
            body += "-" * 30 + "\n"
            
            for item in items:
                body += f"  • {item['group'].upper()}: {item['base']} vs {item['correlated']}\n"
                body += f"    Type: {item['type'].upper()} divergence\n"
                body += f"    {item['base']} change: {item.get('change1', 0):+.2f}%\n"
                body += f"    {item['correlated']} change: {item.get('change2', 0):+.2f}%\n"
                body += f"    Message: {item.get('message', '')}\n\n"
        
        body += f"\n---\nChecked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send email notification
        try:
            await send_notification(
                subject=subject,
                body=body,
                recipients=["admin@example.com"]  # Configure as needed
            )
            logger.info(f"Sent alert email for {len(detections)} detections")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

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
        "timestamp": datetime.now().isoformat()
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