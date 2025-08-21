# src/data/polygon_client.py
from __future__ import annotations
import os
from typing import Literal, Optional
import pandas as pd
import requests

from normalize.ohlcv_schema import coerce_schema, validate_ohlcv

_POLYGON_BASE = "https://api.polygon.io"

def _ticker_for_market(symbol: str, market: Literal["stocks", "crypto", "fx"]) -> str:
    s = symbol.strip().upper()
    if market == "stocks":
        # Works for equities/ETFs. Examples: AAPL, MSFT, SPY, BRK.B, RDS.A
        return s
    elif market == "crypto":
        if s.startswith("X:"):
            return s
        core = s.replace("-", "").replace(":", "")
        return f"X:{core}"
    elif market == "fx":
        if s.startswith("C:"):
            return s
        core = s.replace(":", "")
        return f"C:{core}"
    raise ValueError(f"unknown market: {market}")

def fetch_polygon_ohlcv(
    symbol: str,
    market: Literal["stocks", "crypto", "fx"] = "stocks",   # <-- default now stocks
    start: str = "2023-01-01",
    end: str = "2023-03-31",
    timespan: Literal["minute", "hour", "day", "week", "month"] = "day",
    multiplier: int = 1,
    adjusted: bool = True,   # for stocks: True = split/div adjusted
    api_key: Optional[str] = 'hzOJT91LLAflxA4B73FY1ST5r0ccZMTR',
) -> pd.DataFrame:
    api_key = api_key or os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("Set POLYGON_API_KEY env var or pass api_key=...")

    ticker = _ticker_for_market(symbol, market)
    url = f"{_POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {
        "adjusted": str(adjusted).lower(),
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    r = requests.get(url, params=params, timeout=30)
    try:
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Polygon HTTP error: {e}\nBody: {r.text[:400]}")

    data = r.json()
    if data.get("status") != "OK" or not data.get("results"):
        return coerce_schema(pd.DataFrame([]))

    df = pd.DataFrame.from_records(data["results"]).rename(
        columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")  # <-- add unit="ms"
    df["symbol"] = ticker
    df = coerce_schema(df)
    return validate_ohlcv(df)