# src/data/cmc_client.py
from __future__ import annotations
import os
import requests
import pandas as pd
from datetime import timedelta
import json
from normalize.ohlcv_schema import coerce_schema, validate_ohlcv

CMC_BASE = "https://pro-api.coinmarketcap.com/v2"
DEFAULT_MAX_MONTHS = 24
api_key='e64efce5-b78d-4175-9f35-9626265824e0'
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _clamp_cmc_window(start: str, end: str, max_months: int = DEFAULT_MAX_MONTHS):
    end_dt = pd.to_datetime(end, utc=True).date()
    start_dt = pd.to_datetime(start, utc=True).date()
    cutoff = end_dt - timedelta(days=30 * max_months)
    if start_dt < cutoff:
        start_dt = cutoff
    if start_dt > end_dt:
        start_dt = end_dt
    return start_dt.isoformat(), end_dt.isoformat()


def _cmc_key(env=os.getenv) -> str:
    # Accept any of these; your .env used COINMKTCAP_KEY
    key = (
        env("CMC_API_KEY")
        or env("COINMARKETCAP_API_KEY")
        or env("COINMKTCAP_KEY")
        or ""
    )
    key = key.strip()
    if not key:
        raise RuntimeError(
            "CoinMarketCap API key not found. Set CMC_API_KEY or COINMARKETCAP_API_KEY (or COINMKTCAP_KEY)."
        )
    return key


def _cmc_get(path: str, params: dict) -> dict:
    url = f"{CMC_BASE}{path}"
    headers = {"X-CMC_PRO_API_KEY": _cmc_key()}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"CMC HTTP error: {e}\nURL: {r.url}\nBody: {r.text[:600]}")
    data = r.json()
    status = data.get("status") or {}
    # CMC often returns HTTP 200 + error_code in the status block when the plan forbids an endpoint
    if isinstance(status, dict) and status.get("error_code") not in (0, None):
        code = status.get("error_code")
        msg = status.get("error_message")
        raise RuntimeError(f"CMC status error {code}: {msg}\nURL: {r.url}")
    return data


def fetch_cmc_quote_latest(symbol: str, convert: str = "USD") -> pd.DataFrame:
    params = {"symbol": symbol, "convert": convert}
    data = _cmc_get("/cryptocurrency/quotes/latest", params)
    d = data.get("data", {})

    item = None
    if isinstance(d, dict) and symbol in d:
        # d[symbol] could be a dict or a list
        symbol_data = d[symbol]
        if isinstance(symbol_data, list) and symbol_data:
            item = symbol_data[0]  # Get first item from list
        elif isinstance(symbol_data, dict):
            item = symbol_data
    elif isinstance(d, list) and d:
        item = d[0]

    if not item:
        return pd.DataFrame([])

    # Now item should be a dict
    if not isinstance(item, dict):
        return pd.DataFrame([])
    
    quote = item.get("quote", {}).get(convert, {})
    out = {
        "timestamp": pd.Timestamp.now(tz="UTC"),
        "symbol": f"X:{symbol}{convert}",
        "price": quote.get("price"),
        "market_cap": quote.get("market_cap"),
        "volume_24h": quote.get("volume_24h"),
    }
    return pd.DataFrame([out])


def fetch_cmc_ohlcv(
    symbol: str,
    convert: str,
    start: str,
    end: str,
    interval: str = "daily",
    max_months: int = DEFAULT_MAX_MONTHS,
) -> pd.DataFrame:

    start, end = _clamp_cmc_window(start, end, max_months=max_months)
    params = {
        "symbol": symbol,
        "convert": convert,
        "time_start": start,
        "time_end": end,
        "interval": interval,
    }
    data = _cmc_get("/cryptocurrency/ohlcv/historical", params)

    d = data.get("data", {})
    if "quotes" in d:
        rows = d["quotes"]
    elif isinstance(d, dict) and symbol in d and isinstance(d[symbol], dict) and "quotes" in d[symbol]:
        rows = d[symbol]["quotes"]
    else:
        rows = []

    if not rows:
        # Typed-empty
        return coerce_schema(pd.DataFrame([]))

    df = pd.DataFrame(
        {
            "timestamp": [row.get("time_open") for row in rows],
            "symbol": [f"X:{symbol}USD"] * len(rows),
            "open": [row["quote"][convert]["open"] for row in rows],
            "high": [row["quote"][convert]["high"] for row in rows],
            "low": [row["quote"][convert]["low"] for row in rows],
            "close": [row["quote"][convert]["close"] for row in rows],
            "volume": [row["quote"][convert]["volume"] for row in rows],
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = coerce_schema(df)
    return validate_ohlcv(df)


def fetch_cmc_ohlcv_safe(*args, **kwargs) -> pd.DataFrame:
    try:
        return fetch_cmc_ohlcv(*args, **kwargs)
    except Exception as e:
        # Capture the specific reason for operators/logs
        msg = str(e)
        df = coerce_schema(pd.DataFrame([]))
        df.attrs["cmc_error"] = msg
        return df