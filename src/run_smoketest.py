# gtrade/run_smoketest.py
from __future__ import annotations

import os
import datetime as dt

from data.polygon_client import fetch_polygon_ohlcv
from data.cmc_client import fetch_cmc_ohlcv, fetch_cmc_ohlcv_safe, fetch_cmc_quote_latest
from normalize.ohlcv_schema import validate_ohlcv

def _recent_window(days: int = 90) -> tuple[str, str]:
    # end = yesterday UTC to avoid partial current day
    end_dt = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    start_dt = end_dt - dt.timedelta(days=days - 1)
    return start_dt.isoformat(), end_dt.isoformat()

# run_smoketest.py
def main():
    start, end = _recent_window(90)
    # Map your .env variable names to the ones the adapters expect
    os.environ.setdefault("POLYGON_API_KEY", os.getenv("POLYGONIO_KEY", ""))        # from .env
    os.environ.setdefault("CMC_API_KEY", os.getenv("COINMKTCAP_KEY", ""))           # from .env

    # 1) Polygon crypto (BTCUSD daily)
    df_poly = fetch_polygon_ohlcv(
        symbol="BTCUSD", market="crypto", start=start, end=end, timespan="day", adjusted=False,
    )
    print("Polygon BTC rows:", len(df_poly), "|", df_poly.symbol.iloc[0])
    print(df_poly.head(3))

    # 2) CMC crypto (BTC/USD daily)
    try:
        latest = fetch_cmc_quote_latest("BTC", convert="USD")
        ok = not latest.empty and pd.notna(latest["price"]).any()
        print("CMC quotes/latest OK:", ok)
        if ok:
            print(latest.head(1))
    except Exception as e:
        print("CMC quotes/latest failed:", e)

    # 3) CMC historical with graceful fallback
    df_cmc = fetch_cmc_ohlcv_safe(symbol="BTC", start=start, end=end, convert="USD", interval="daily")
    if getattr(df_cmc, "empty", True):
        reason = df_cmc.attrs.get("cmc_error", "no rows returned")
        print("CMC OHLCV unavailable (falling back). Reason:", reason)
    else:
        print("CMC BTC rows:", len(df_cmc), "|", df_cmc.symbol.iloc[0])
        print(df_cmc.head(3))

    # 4) Final summary
    poly_range = (df_poly.timestamp.min(), df_poly.timestamp.max())
    cmc_range = (df_cmc.timestamp.min() if not df_cmc.empty else None,
                 df_cmc.timestamp.max() if not df_cmc.empty else None)
    print("Date ranges -> Polygon:", poly_range[0], "to", poly_range[1], "CMC:", cmc_range[0], "to", cmc_range[1])
    "CMC:", (df_cmc.timestamp.min() if not df_cmc.empty else None), "to", (df_cmc.timestamp.max() if not df_cmc.empty else None)

if __name__ == "__main__":
    main()