from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import pandas as pd
from pathlib import Path
import yfinance as yf

# Initialize FastAPI
app = FastAPI(
    title="Financial Intelligence System",
    description="AI-powered financial analysis and market intelligence",
    version="1.0.0"
)

# Enable CORS for OpenBB Workspace
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pro.openbb.co", "http://localhost:6900", "http://127.0.0.1:6900"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# SYMBOL MAPPING
# ==========================================

ASSET_SYMBOLS = {
    "XAUUSD": "GC=F",
    "GOLD": "GC=F",
    "XAGUSD": "SI=F",
    "SILVER": "SI=F",
    "XAUEUR": "XAUUSD=X",
    "XAUGBP": "XAUUSD=X",
    "GBPUSD": "GBPUSD=X",
    "EURUSD": "EURUSD=X",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "NASDAQ": "NQ=F",
    "ES": "ES=F",
    "YM": "YM=F",
}

def get_symbol(asset: str) -> str:
    """Get Yahoo Finance symbol for an asset"""
    return ASSET_SYMBOLS.get(asset.upper(), asset)

def fetch_yahoo_data(symbol: str, period: str = "5d", start: str = None, end: str = None):
    """Safely fetch data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        if start and end:
            data = ticker.history(start=start, end=end)
        else:
            data = ticker.history(period=period)
        return data
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

# ==========================================
# WIDGETS CONFIGURATION ENDPOINT
# ==========================================

@app.get("/widgets.json")
async def get_widgets():
    """Serve the widgets configuration file"""
    try:
        widgets_path = Path(__file__).parent / "widgets.json"
        # Explicitly use UTF-8 encoding to handle special characters
        with open(widgets_path, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))
    except FileNotFoundError:
        # Fallback configuration
        return JSONResponse(content={
            "xauusd_current": {
                "name": "Gold Price (Live)",
                "description": "Current XAUUSD price with change",
                "endpoint": "xauusd/current",
                "category": "Commodities",
                "gridData": {"w": 8, "h": 4}
            },
            "silver_current": {
                "name": "Silver Price (Live)",
                "description": "Current XAGUSD price with change",
                "endpoint": "silver/current",
                "category": "Commodities",
                "gridData": {"w": 8, "h": 4}
            },
            "forex_current": {
                "name": "Forex Rates",
                "description": "Live forex rates for major pairs",
                "endpoint": "forex/current",
                "category": "Forex",
                "gridData": {"w": 12, "h": 6}
            },
            "nasdaq_futures": {
                "name": "Futures Dashboard",
                "description": "Nasdaq, S&P 500, and Dow Jones futures",
                "endpoint": "nasdaq/futures",
                "category": "Futures",
                "gridData": {"w": 16, "h": 6}
            },
            "xauusd_correlation": {
                "name": "Correlation Matrix",
                "description": "Correlation between gold, silver, and forex",
                "endpoint": "xauusd/correlation",
                "category": "Analysis",
                "gridData": {"w": 20, "h": 8}
            },
            "xauusd_divergence": {
                "name": "Divergence Detector",
                "description": "Detects divergence between gold and silver",
                "endpoint": "xauusd/divergence",
                "category": "Analysis",
                "gridData": {"w": 12, "h": 6}
            },
            "xauusd_insight": {
                "name": "Market Insight",
                "description": "AI-generated analysis and outlook",
                "endpoint": "xauusd/insight",
                "category": "AI & Analytics",
                "gridData": {"w": 20, "h": 12}
            },
            "xauusd_historical": {
                "name": "Historical Price",
                "description": "Gold price with date range selector",
                "endpoint": "xauusd/historical",
                "category": "Commodities",
                "gridData": {"w": 20, "h": 10},
                "params": {
                    "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                    "end_date": datetime.now().strftime("%Y-%m-%d")
                }
            },
            "all_assets": {
                "name": "All Assets Dashboard",
                "description": "Complete market overview",
                "endpoint": "assets/all",
                "category": "Dashboard",
                "gridData": {"w": 20, "h": 8}
            }
        })

# ==========================================
# DATA ENDPOINTS
# ==========================================

@app.get("/xauusd/current")
async def get_current_price() -> Dict[str, Any]:
    """Current XAUUSD Price - Returns a METRIC widget"""
    try:
        data = fetch_yahoo_data("GC=F", period="5d")
        
        if data.empty:
            return {"value": "N/A", "delta": "0%", "label": "XAUUSD (Unavailable)"}
        
        current = float(data["Close"].iloc[-1])
        
        if len(data) > 1:
            previous = float(data["Close"].iloc[-2])
            change = ((current - previous) / previous) * 100
        else:
            change = 0.0
            
        return {
            "value": f"${current:.2f}",
            "delta": f"{change:+.2f}%",
            "label": "XAUUSD Gold"
        }
    except Exception as e:
        return {"value": "N/A", "delta": "0%", "label": "XAUUSD (Error)", "error": str(e)}

@app.get("/silver/current")
async def get_silver_current() -> Dict[str, Any]:
    """Current XAGUSD Price - Returns a METRIC widget"""
    try:
        data = fetch_yahoo_data("SI=F", period="5d")
        
        if data.empty:
            return {"value": "N/A", "delta": "0%", "label": "XAGUSD (Unavailable)"}
        
        current = float(data["Close"].iloc[-1])
        
        if len(data) > 1:
            previous = float(data["Close"].iloc[-2])
            change = ((current - previous) / previous) * 100
        else:
            change = 0.0
            
        return {
            "value": f"${current:.2f}",
            "delta": f"{change:+.2f}%",
            "label": "XAGUSD Silver"
        }
    except Exception as e:
        return {"value": "N/A", "delta": "0%", "label": "XAGUSD (Error)", "error": str(e)}

@app.get("/forex/current")
async def get_forex_current() -> List[Dict[str, Any]]:
    """Current Forex Rates - Returns a TABLE widget"""
    pairs = {"GBPUSD": "GBPUSD=X", "EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X"}
    result = []
    
    for name, symbol in pairs.items():
        try:
            data = fetch_yahoo_data(symbol, period="2d")
            if not data.empty:
                current = float(data["Close"].iloc[-1])
                previous = float(data["Close"].iloc[-2]) if len(data) > 1 else current
                change = ((current - previous) / previous) * 100 if previous != 0 else 0
                result.append({
                    "pair": name,
                    "price": round(current, 4),
                    "change_%": round(change, 2),
                    "direction": "▲" if change > 0 else "▼" if change < 0 else "—"
                })
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            continue
    
    return result if result else [{"error": "No forex data available"}]

@app.get("/nasdaq/futures")
async def get_futures_data() -> List[Dict[str, Any]]:
    """Futures Data: Nasdaq, ES, YM"""
    futures = {"Nasdaq": "NQ=F", "S&P 500": "ES=F", "Dow Jones": "YM=F"}
    result = []
    
    for name, symbol in futures.items():
        try:
            data = fetch_yahoo_data(symbol, period="2d")
            if not data.empty:
                current = float(data["Close"].iloc[-1])
                previous = float(data["Close"].iloc[-2]) if len(data) > 1 else current
                change = ((current - previous) / previous) * 100 if previous != 0 else 0
                result.append({
                    "name": name,
                    "symbol": symbol,
                    "price": round(current, 2),
                    "change": round(change, 2),
                    "direction": "▲" if change > 0 else "▼" if change < 0 else "—"
                })
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            continue
    
    return result if result else [{"error": "No futures data available"}]

@app.get("/xauusd/historical")
async def get_xauusd_historical(
    start_date: str = Query(default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")),
    end_date: str = Query(default=datetime.now().strftime("%Y-%m-%d"))
) -> List[Dict[str, Any]]:
    """XAUUSD Historical Price Data"""
    try:
        data = fetch_yahoo_data("GC=F", start=start_date, end=end_date)
        if data.empty:
            return [{"error": "No data available"}]
        
        data = data.reset_index()
        data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
        data = data.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        
        records = data[["date", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, (int, float)):
                    record[key] = float(value)
        return records
    except Exception as e:
        return [{"error": str(e)}]

@app.get("/xauusd/correlation")
async def get_correlation(
    start_date: str = Query(default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")),
    end_date: str = Query(default=datetime.now().strftime("%Y-%m-%d"))
) -> List[Dict[str, Any]]:
    """Correlation Matrix"""
    assets = {"Gold": "GC=F", "Silver": "SI=F", "GBPUSD": "GBPUSD=X", "EURUSD": "EURUSD=X"}
    data_dict = {}
    
    for name, symbol in assets.items():
        try:
            data = fetch_yahoo_data(symbol, start=start_date, end=end_date)
            if not data.empty and "Close" in data.columns:
                data_dict[name] = data["Close"]
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            continue
    
    if data_dict:
        df_prices = pd.DataFrame(data_dict).dropna(axis=1, how='all')
        if df_prices.empty:
            return [{"error": "No data available"}]
        
        correlation_matrix = df_prices.corr().reset_index().rename(columns={"index": "Asset"})
        records = correlation_matrix.to_dict(orient="records")
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, (int, float)):
                    record[key] = round(float(value), 3)
        return records
    return [{"error": "No data available"}]

@app.get("/xauusd/divergence")
async def check_divergence() -> Dict[str, Any]:
    """Divergence Detection: XAUUSD vs XAGUSD"""
    try:
        gold = fetch_yahoo_data("GC=F", period="30d")
        silver = fetch_yahoo_data("SI=F", period="30d")
        
        if gold.empty or silver.empty:
            return {"error": "No data available", "divergence": False}
        
        price_gold = float(gold["Close"].iloc[-1])
        price_silver = float(silver["Close"].iloc[-1])
        gold_change = ((price_gold - float(gold["Close"].iloc[-5])) / float(gold["Close"].iloc[-5])) * 100 if len(gold) >= 5 else 0
        silver_change = ((price_silver - float(silver["Close"].iloc[-5])) / float(silver["Close"].iloc[-5])) * 100 if len(silver) >= 5 else 0
        
        divergence_detected = abs(gold_change - silver_change) > 2
        direction = "bullish" if gold_change > silver_change else "bearish"
        
        return {
            "asset1": "XAUUSD (Gold)",
            "asset2": "XAGUSD (Silver)",
            "price_gold": round(price_gold, 2),
            "price_silver": round(price_silver, 2),
            "gold_change": f"{gold_change:+.2f}%",
            "silver_change": f"{silver_change:+.2f}%",
            "divergence": divergence_detected,
            "direction": direction if divergence_detected else "none",
            "message": f"{direction.capitalize()} divergence detected!" if divergence_detected else "No divergence detected",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": str(e), "divergence": False}

@app.get("/xauusd/insight")
async def get_insight() -> str:
    """AI-Generated Market Insight - Returns MARKDOWN"""
    try:
        data = fetch_yahoo_data("GC=F", period="6mo")
        if data.empty:
            return "No data available for analysis"
        
        current = float(data["Close"].iloc[-1])
        high_6m = float(data["Close"].max())
        low_6m = float(data["Close"].min())
        avg_50d = float(data["Close"].tail(50).mean()) if len(data) >= 50 else current
        avg_200d = float(data["Close"].tail(200).mean()) if len(data) >= 200 else current
        
        return f"""
## XAUUSD Gold Market Insight

### Key Levels
- Current Price: ${current:.2f}
- 6-Month High: ${high_6m:.2f}
- 6-Month Low: ${low_6m:.2f}
- 50-Day Average: ${avg_50d:.2f}
- 200-Day Average: ${avg_200d:.2f}

### Analysis
- Gold is trading **{'ABOVE' if current > avg_50d else 'BELOW'}** its 50-day average
- Gold is trading **{'ABOVE' if current > avg_200d else 'BELOW'}** its 200-day average
- **{'BULLISH' if current > avg_50d else 'BEARISH'}** short-term momentum
- Price is **{((current - low_6m) / (high_6m - low_6m) * 100):.1f}%** of its 6-month range

*Data from Yahoo Finance | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    except Exception as e:
        return f"Error generating insight: {str(e)}"

@app.get("/assets/all")
async def get_all_assets() -> List[Dict[str, Any]]:
    """All Assets Dashboard"""
    assets = {
        "Gold": "GC=F", "Silver": "SI=F",
        "GBPUSD": "GBPUSD=X", "EURUSD": "EURUSD=X",
        "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X",
        "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD",
        "Nasdaq": "NQ=F", "S&P 500": "ES=F", "Dow Jones": "YM=F"
    }
    result = []
    
    for name, symbol in assets.items():
        try:
            data = fetch_yahoo_data(symbol, period="2d")
            if not data.empty:
                current = float(data["Close"].iloc[-1])
                previous = float(data["Close"].iloc[-2]) if len(data) > 1 else current
                change = ((current - previous) / previous) * 100 if previous != 0 else 0
                result.append({
                    "name": name,
                    "symbol": symbol,
                    "price": float(current),
                    "change": round(change, 2),
                    "direction": "▲" if change > 0 else "▼" if change < 0 else "—"
                })
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            continue
    
    return result if result else [{"error": "No data available"}]

# ==========================================
# ROOT AND HEALTH ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    return {
        "service": "Financial Intelligence System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "widgets": "/widgets.json",
            "xauusd_current": "/xauusd/current",
            "xauusd_historical": "/xauusd/historical",
            "xauusd_correlation": "/xauusd/correlation",
            "xauusd_divergence": "/xauusd/divergence",
            "xauusd_insight": "/xauusd/insight",
            "silver_current": "/silver/current",
            "forex_current": "/forex/current",
            "nasdaq_futures": "/nasdaq/futures",
            "assets_all": "/assets/all"
        }
    }

@app.get("/favicon.ico")
async def favicon():
    favicon_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    return Response(content=favicon_data, media_type="image/png")

# ==========================================
# RUN THE APPLICATION
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=6900, reload=True)