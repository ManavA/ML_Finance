# src/data/crypto_data.py
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from normalize.ohlcv_schema import REQUIRED_COLS, validate_ohlcv, coerce_schema
import ccxt
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dateutil import parser as dateparser
from tqdm import tqdm

MS_IN_MIN = 60_000
TIMEFRAME_MS = {
    "1m": MS_IN_MIN,
    "5m": 5 * MS_IN_MIN,
    "15m": 15 * MS_IN_MIN,
    "30m": 30 * MS_IN_MIN,
    "1h": 60 * MS_IN_MIN,
    "4h": 240 * MS_IN_MIN,
    "1d": 1440 * MS_IN_MIN,
}


@dataclass
class FetchWindow:
    since_ms: int
    until_ms: Optional[int] = None


class RateLimit(Exception):
    pass


def _to_ms(ts) -> int:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return int(ts)
    # strings/datetime
    return int(pd.Timestamp(ts, tz="UTC").value // 1_000_000)


def _to_utc_index(df: pd.DataFrame, ts_col="timestamp") -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df[ts_col], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


def _align_cols(df: pd.DataFrame, symbol: str, kind: str) -> pd.DataFrame:
    if kind == "ohlcv":
        df.columns = ["open", "high", "low", "close", "volume"]
    elif kind == "funding":
        # standardize funding columns
        # for binance/bybit we'll have: fundingRate, fundingTime, symbol
        pass
    df["symbol"] = symbol
    return df


class CryptoDataClient:

    def __init__(self, exchange_id: str = "binance", type_hint: str = "swap", enable_rate_limit: bool = True):

        self.exchange_id = exchange_id
        self.type_hint = type_hint
        klass = getattr(ccxt, exchange_id)
        # enable built-in throttling too
        self.ex = klass({"enableRateLimit": enable_rate_limit})
        self.markets = self.ex.load_markets()

    # -------- Symbol helpers --------
    def resolve_market(self, base: str, quote: str = "USDT") -> str:

        candidates = []
        for m, info in self.markets.items():
            if info.get("base") == base and info.get("quote") == quote:
                if self.type_hint == "spot" and info.get("spot"):
                    candidates.append(m)
                elif self.type_hint == "swap" and info.get("swap"):
                    candidates.append(m)
                elif self.type_hint == "future" and info.get("future"):
                    candidates.append(m)
        if not candidates:
            raise ValueError(f"No market found for {base}/{quote} on {self.exchange_id} with type={self.type_hint}")
        # prefer linear USDT swaps where multiple exist
        if self.type_hint == "swap":
            usdt_like = [m for m in candidates if ":USDT" in m or "/USDT" in m]
            if usdt_like:
                return usdt_like[0]
        return candidates[0]

    # -------- Core fetchers --------
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30),
           retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeNotAvailable)))
    def _fetch_ohlcv_page(self, symbol: str, timeframe: str, since_ms: Optional[int], limit: int = 1500):
        return self.ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)

    def fetch_ohlcv(self, symbol: str, timeframe: str, window: FetchWindow, limit_per_call: int = 1500) -> pd.DataFrame:

        if timeframe not in TIMEFRAME_MS:
            raise ValueError(f"Unsupported timeframe {timeframe}")
        frame_ms = TIMEFRAME_MS[timeframe]
        since = int(window.since_ms)
        until = int(window.until_ms) if window.until_ms else None

        all_rows: List[List] = []
        pbar = tqdm(total=None, desc=f"{self.exchange_id}:{symbol}:{timeframe}", leave=False)
        while True:
            batch = self._fetch_ohlcv_page(symbol, timeframe, since, limit=limit_per_call)
            if not batch:
                break
            all_rows.extend(batch)
            last_ts = batch[-1][0]
            pbar.set_postfix_str(pd.to_datetime(last_ts, unit="ms", utc=True).isoformat())
            # advance since by number of full candles fetched
            since = last_ts + frame_ms
            # exit if we passed 'until'
            if until and since >= until:
                break
            # polite sleep if CCXT throttle is off
            if not self.ex.enableRateLimit:
                time.sleep(self.ex.rateLimit / 1000)
        pbar.close()

        if not all_rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "symbol"])

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = _to_utc_index(df)
        # drop the very last row if it is the current forming candle
        now_floor = (pd.Timestamp.utcnow().floor(timeframe)).tz_localize("UTC")
        df = df[df.index < now_floor]
        df = _align_cols(df, symbol, "ohlcv")
        return df

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30),
           retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeNotAvailable)))
    def _fetch_funding_page(self, symbol: str, since_ms: Optional[int] = None, limit: int = 500):

        if not getattr(self.ex, "has", {}).get("fetchFundingRateHistory", False):
            raise NotImplementedError(f"{self.exchange_id} does not support fetchFundingRateHistory in CCXT.")
        return self.ex.fetch_funding_rate_history(symbol, since=since_ms, limit=limit)

    def fetch_funding_rates(self, symbol: str, window: FetchWindow) -> pd.DataFrame:

        rows = []
        since = int(window.since_ms)
        until = int(window.until_ms) if window.until_ms else None
        pbar = tqdm(total=None, desc=f"{self.exchange_id}:{symbol}:funding", leave=False)
        while True:
            page = self._fetch_funding_page(symbol, since_ms=since, limit=1000)
            if not page:
                break
            rows.extend(page)
            last_ts = page[-1]["timestamp"]
            pbar.set_postfix_str(pd.to_datetime(last_ts, unit="ms", utc=True).isoformat())
            since = last_ts + MS_IN_MIN  # funding is 8h or 1h; +1m keeps us moving
            if until and since >= until:
                break
            if not self.ex.enableRateLimit:
                time.sleep(self.ex.rateLimit / 1000)
        pbar.close()

        if not rows:
            return pd.DataFrame(columns=["funding_rate", "symbol"])

        # normalize -> DataFrame
        df = pd.DataFrame(rows)
        # CCXT returns 'info' blob per exchange; we standardize key fields
        # Common fields: 'timestamp', 'fundingRate', 'symbol'
        col_map = {}
        if "fundingRate" in df.columns:
            col_map["fundingRate"] = "funding_rate"
        elif "rate" in df.columns:
            col_map["rate"] = "funding_rate"
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        df = df.rename(columns=col_map)
        # enforce numeric
        df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
        df = df[["timestamp", "funding_rate", "symbol"]]
        df = _to_utc_index(df)
        return df

    def fetch_universe_ohlcv(
        self, bases: List[str], quote: str, timeframe: str, start, end=None
    ) -> Dict[str, pd.DataFrame]:
        since_ms = _to_ms(start)
        until_ms = _to_ms(end) if end else None
        out = {}
        for base in bases:
            sym = self.resolve_market(base, quote)
            df = self.fetch_ohlcv(sym, timeframe, FetchWindow(since_ms, until_ms))
            out[sym] = df
        return out

    def fetch_universe_funding(
        self, bases: List[str], quote: str, start, end=None
    ) -> Dict[str, pd.DataFrame]:
        since_ms = _to_ms(start)
        until_ms = _to_ms(end) if end else None
        out = {}
        for base in bases:
            sym = self.resolve_market(base, quote)
            try:
                df = self.fetch_funding_rates(sym, FetchWindow(since_ms, until_ms))
            except NotImplementedError:
                warnings.warn(f"{self.exchange_id} has no funding history in CCXT; skipping {sym}")
                df = pd.DataFrame(columns=["funding_rate", "symbol"])
            out[sym] = df
        return out

    @staticmethod
    def save_parquet_per_symbol(data: Dict[str, pd.DataFrame], path: str):
        path = str(path)
        for sym, df in data.items():
            if df is None or df.empty:
                continue
            safe = sym.replace("/", "_").replace(":", "_")
            fp = f"{path}/{safe}.parquet"
            df.to_parquet(fp, engine="pyarrow")

    @staticmethod
    def join_as_panel(data: Dict[str, pd.DataFrame], column: str) -> pd.DataFrame:

        pieces = []
        for sym, df in data.items():
            if df is None or df.empty or column not in df.columns:
                continue
            sub = df[[column]].copy()
            sub.columns = pd.MultiIndex.from_tuples([(sym, column)])
            pieces.append(sub)
        if not pieces:
            return pd.DataFrame()
        panel = pd.concat(pieces, axis=1).sort_index()
        return panel

    @staticmethod
    def basic_quality(panel: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if panel.empty:
            return pd.DataFrame()
        frame_ms = TIMEFRAME_MS[timeframe]
        # expected index should be continuous between min and max
        full_index = pd.date_range(panel.index.min(), panel.index.max(), freq=timeframe, tz="UTC")
        expect_n = len(full_index)
        stats = []
        for sym in panel.columns.get_level_values(0).unique():
            s = panel[(sym, panel.columns.levels[1][0])]
            present = s.dropna().shape[0]
            stats.append({"symbol": sym, "coverage_pct": 100 * present / expect_n})
        return pd.DataFrame(stats).sort_values("coverage_pct", ascending=False)