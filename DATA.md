# Data & Sources

## Sources

- **yfinance** — free historical OHLCV and reference data.  
- **Polygon.io** (optional) — higher‑resolution market data for crypto/equities.  
- **CoinMarketCap** (optional) — crypto reference/market data.

## Frequency & coverage

The notebooks target multi‑year coverage (e.g., 2023‑2025) for both crypto (24/7) and equities (regular trading hours).

## Quality checks

- Missing data detection and backfilling rules where appropriate.  
- Cross‑validation that respects session boundaries.  
- Cache & retry with fallbacks when one provider fails.

## Credentials

Store API keys in `.env` (never commit secrets):

```
POLYGONIO_KEY=...
COINMARKETCAP_KEY=...
```

