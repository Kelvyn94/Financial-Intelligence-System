"""
Standalone sanity check for liquidity.py using synthetic OHLC data --
the sandbox this was written in has no network path to Yahoo Finance, so
this monkeypatches the two network-fetch functions liquidity.py depends on
(agent.fetch_asset_data and liquidity.fetch_daily_history) and exercises the
pure logic (session windows, equal-level clustering, major swings, prior
day/week/month) against hand-built data with known answers.

Run with: venv/bin/python test_liquidity_synthetic.py
"""
import asyncio
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import agent  # noqa: E402
import liquidity  # noqa: E402

NY_TZ = ZoneInfo("America/New_York")


def make_intraday(days: int = 5) -> pd.DataFrame:
    """30m bars for `days` days ending now, tz-aware in America/New_York
    (mirrors how yfinance returns futures data)."""
    end = datetime.now(NY_TZ).replace(second=0, microsecond=0)
    end = end - timedelta(minutes=end.minute % 30)
    start = end - timedelta(days=days)
    idx = pd.date_range(start=start, end=end, freq="30min", tz=NY_TZ)

    rng = np.random.default_rng(42)
    n = len(idx)
    base = 2000 + np.cumsum(rng.normal(0, 1.5, n))
    high = base + rng.uniform(0.5, 2.0, n)
    low = base - rng.uniform(0.5, 2.0, n)
    open_ = base + rng.normal(0, 0.5, n)
    close = base + rng.normal(0, 0.5, n)

    plant_a, plant_b = n // 3, (2 * n) // 3
    high[plant_a] = 2050.00
    high[plant_b] = 2050.03

    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                        "Volume": rng.integers(100, 1000, n)}, index=idx)
    return df


def make_daily(days: int = 400) -> pd.DataFrame:
    """Daily bars, tz-naive index (mirrors yfinance daily data)."""
    end = datetime.now(NY_TZ).date()
    idx = pd.bdate_range(end=end, periods=days)
    rng = np.random.default_rng(7)
    n = len(idx)
    base = 2000 + np.cumsum(rng.normal(0, 5, n))
    high = base + rng.uniform(2, 10, n)
    low = base - rng.uniform(2, 10, n)
    open_ = base + rng.normal(0, 2, n)
    close = base + rng.normal(0, 2, n)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                        "Volume": rng.integers(1000, 10000, n)}, index=idx)
    return df


async def fake_fetch_asset_data(symbol, period, interval):
    return make_intraday(days=5)


async def fake_fetch_daily_history(symbol, period="1y"):
    return make_daily(days=400)


async def main():
    liquidity.fetch_asset_data = fake_fetch_asset_data
    liquidity.fetch_daily_history = fake_fetch_daily_history

    result = await liquidity._detect_liquidity_for_asset("XAUUSD", "30m", 0.05, 6)

    import json
    print(json.dumps(result, indent=2, default=str))

    liq = result["liquidity"]
    assert result["asset"] == "XAUUSD"
    assert result["symbol"] == "GC=F"

    eq_highs = liq["equal_highs"]
    assert len(eq_highs) >= 1, "expected at least one equal-highs pool"
    planted = [p for p in eq_highs if 2049.9 <= p["price"] <= 2050.1]
    assert planted, f"planted equal-high pool not detected: {eq_highs}"
    assert planted[0]["touches"] == 2, f"expected 2 touches, got {planted[0]}"
    print("\nOK: equal-highs clustering found the planted pool:", planted[0])

    minor_highs, minor_lows = agent.find_swing_points(make_intraday(5), lookback=2)
    assert len(liq["major_swing_highs"]) <= len(minor_highs) + 1
    print("OK: major swing highs:", len(liq["major_swing_highs"]), "vs minor:", len(minor_highs))

    for name in ["asia", "london", "new_york", "new_york_pm"]:
        assert name in liq["previous_session"], f"missing session {name}"
        s = liq["previous_session"][name]
        if s is not None:
            assert s["high"] >= s["low"]
            start = datetime.fromisoformat(s["session_start"])
            end = datetime.fromisoformat(s["session_end"])
            assert start < end
            assert end <= datetime.now(NY_TZ), "session window must be in the past"
    print("OK: previous_session windows all in the past and internally consistent:")
    for name in ["asia", "london", "new_york", "new_york_pm"]:
        print(" ", name, liq["previous_session"][name])

    for key in ["previous_day", "previous_week", "previous_month"]:
        v = liq[key]
        assert v is not None, f"{key} should not be None with 400 days of daily data"
        assert v["high"] >= v["low"]
    print("OK: previous_day/week/month:")
    for key in ["previous_day", "previous_week", "previous_month"]:
        print(" ", key, liq[key])

    print("\nALL SYNTHETIC CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
