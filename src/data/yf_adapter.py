# src/data/yf_adapter.py
import pandas as pd, yfinance as yf

def fetch_yf_ohlcv(symbol, start=None, end=None, interval="1d"):
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    if df.empty: return df
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
    df = df.reset_index().rename(columns={"Date":"timestamp"})
    df["symbol"] = symbol
    return df