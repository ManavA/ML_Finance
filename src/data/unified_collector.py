"""
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Literal
import hashlib
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.polygon_client import fetch_polygon_ohlcv
from data.cmc_client import fetch_cmc_quote_latest, fetch_cmc_ohlcv_safe


class UnifiedDataCollector:
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure data sources in priority order
        self.sources = {
            'polygon_s3': self._fetch_polygon_s3,
            'polygon': self._fetch_polygon,
            'coinmarketcap': self._fetch_cmc,
            # Add more sources as needed
        }
        
        # Initialize S3 collector (lazy loading)
        self._s3_collector = None
        
    def _generate_cache_key(self, symbol: str, start: str, end: str, 
                           timespan: str, source: str) -> str:
        key_string = f"{source}_{symbol}_{start}_{end}_{timespan}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.csv"
    
    def _get_cache_metadata_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}_meta.json"
    
    def _is_cache_valid(self, cache_key: str, max_age_hours: int = 24) -> bool:
        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_cache_metadata_path(cache_key)
        
        if not cache_path.exists() or not meta_path.exists():
            return False
        
        # Check cache age
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            cached_time = datetime.fromisoformat(metadata['cached_at'])
            age = datetime.now() - cached_time
            
            return age.total_seconds() < (max_age_hours * 3600)
        except Exception:
            return False
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str, 
                      source: str, symbol: str) -> None:
        if data.empty:
            return
        
        # Save data
        cache_path = self._get_cache_path(cache_key)
        data.to_csv(cache_path, index=False)
        
        # Save metadata
        metadata = {
            'cached_at': datetime.now().isoformat(),
            'source': source,
            'symbol': symbol,
            'rows': len(data),
            'date_range': {
                'start': str(data['timestamp'].min()) if 'timestamp' in data.columns else None,
                'end': str(data['timestamp'].max()) if 'timestamp' in data.columns else None
            }
        }
        
        meta_path = self._get_cache_metadata_path(cache_key)
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        cache_path = self._get_cache_path(cache_key)
        
        try:
            data = pd.read_csv(cache_path)
            # Ensure timestamp is datetime
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
            return data
        except Exception as e:
            print(f"Cache load error: {e}")
            return None
    
    def _fetch_polygon_s3(self, symbol: str, start: str, end: str,
                          timespan: str = "hour", **kwargs) -> pd.DataFrame:
        try:
            # Lazy load S3 collector
            if self._s3_collector is None:
                from data.polygon_s3_collector import PolygonS3Collector
                try:
                    self._s3_collector = PolygonS3Collector()
                except Exception as e:
                    print(f"S3 collector initialization failed: {e}")
                    return pd.DataFrame()
            
            # Map timespan to S3 format
            timeframe_map = {
                'hourly': 'hour',
                '1h': 'hour',
                'daily': 'day',
                '1d': 'day',
                'minute': 'minute',
                '1m': 'minute'
            }
            s3_timeframe = timeframe_map.get(timespan, timespan)
            
            # Determine market
            market = 'crypto' if 'USD' in symbol.upper() else 'stocks'
            
            return self._s3_collector.fetch_aggregated_data(
                symbol=symbol,
                start=start,
                end=end,
                market=market,
                timeframe=s3_timeframe
            )
        except Exception as e:
            print(f"Polygon S3 fetch error: {e}")
            return pd.DataFrame()
    
    def _fetch_polygon(self, symbol: str, start: str, end: str, 
                      timespan: str = "hour", **kwargs) -> pd.DataFrame:
        try:
            # Map common timespan values to Polygon format
            timespan_map = {
                'hourly': 'hour',
                '1h': 'hour',
                'daily': 'day',
                '1d': 'day',
                'minute': 'minute',
                '1m': 'minute'
            }
            polygon_timespan = timespan_map.get(timespan, timespan)
            
            # Determine market type based on symbol
            if symbol.upper() in ['BTCUSD', 'ETHUSD', 'SOLUSD'] or 'USD' in symbol.upper():
                market = 'crypto'
            else:
                market = 'stocks'  # Default to stocks for other symbols
            
            return fetch_polygon_ohlcv(
                symbol=symbol,
                market=market,
                start=start,
                end=end,
                timespan=polygon_timespan,
                multiplier=1
            )
        except Exception as e:
            print(f"Polygon fetch error: {e}")
            return pd.DataFrame()
    
    def _fetch_cmc(self, symbol: str, start: str, end: str, 
                  timespan: str = "hour", **kwargs) -> pd.DataFrame:
        try:
            # CMC only supports daily historical data on paid plans
            if timespan in ['hour', 'hourly', '1h', 'minute', '1m']:
                print("CMC only supports daily data on free plan")
                return pd.DataFrame()
            
            # Clean symbol for CMC (remove USD suffix if present)
            cmc_symbol = symbol.replace('USD', '').replace('-', '')
            
            return fetch_cmc_ohlcv_safe(
                symbol=cmc_symbol,
                convert="USD",
                start=start,
                end=end,
                interval="daily"
            )
        except Exception as e:
            print(f"CMC fetch error: {e}")
            return pd.DataFrame()
    
    def fetch_data(self, 
                  symbol: str,
                  start: Optional[str] = None,
                  end: Optional[str] = None,
                  timespan: str = "hour",
                  source_priority: List[str] = None,
                  use_cache: bool = True,
                  cache_max_age_hours: int = 24) -> pd.DataFrame:

        # Set defaults
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')
        if start is None:
            start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if source_priority is None:
            # Prioritize S3 for bulk data, then API
            source_priority = ['polygon_s3', 'polygon', 'coinmarketcap']
        
        # Try each source in priority order
        for source in source_priority:
            if source not in self.sources:
                print(f"Unknown source: {source}")
                continue
            
            # Generate cache key
            cache_key = self._generate_cache_key(symbol, start, end, timespan, source)
            
            # Check cache first
            if use_cache and self._is_cache_valid(cache_key, cache_max_age_hours):
                print(f"Loading {symbol} from cache (source: {source})")
                cached_data = self._load_from_cache(cache_key)
                if cached_data is not None and not cached_data.empty:
                    return cached_data
            
            # Fetch from source
            print(f"Fetching {symbol} from {source}...")
            try:
                data = self.sources[source](symbol, start, end, timespan)
                
                if not data.empty:
                    print(f"Success! Retrieved {len(data)} rows from {source}")
                    
                    # Save to cache
                    if use_cache:
                        self._save_to_cache(data, cache_key, source, symbol)
                    
                    return data
                else:
                    print(f"No data returned from {source}")
                    
            except Exception as e:
                print(f"Error with {source}: {e}")
        
        # If all sources failed, return empty DataFrame
        print(f"All sources failed for {symbol}")
        return pd.DataFrame()
    
    def fetch_latest_quote(self, symbol: str) -> Optional[dict]:

        try:
            # Clean symbol for CMC
            cmc_symbol = symbol.replace('USD', '').replace('-', '')
            
            quote_df = fetch_cmc_quote_latest(cmc_symbol, "USD")
            
            if not quote_df.empty:
                return {
                    'symbol': symbol,
                    'price': quote_df['price'].iloc[0],
                    'market_cap': quote_df['market_cap'].iloc[0],
                    'volume_24h': quote_df['volume_24h'].iloc[0],
                    'timestamp': quote_df['timestamp'].iloc[0]
                }
        except Exception as e:
            print(f"Error fetching quote: {e}")
        
        return None
    
    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:

        deleted = 0
        
        for cache_file in self.cache_dir.glob("*.csv"):
            meta_file = cache_file.with_suffix('') + "_meta.json"
            
            # Check age if specified
            if older_than_hours:
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)
                        cached_time = datetime.fromisoformat(metadata['cached_at'])
                        age = datetime.now() - cached_time
                        
                        if age.total_seconds() < (older_than_hours * 3600):
                            continue  # Skip this file, it's not old enough
                    except Exception:
                        pass  # Delete if we can't read metadata
            
            # Delete cache and metadata files
            try:
                cache_file.unlink()
                deleted += 1
                if meta_file.exists():
                    meta_file.unlink()
            except Exception as e:
                print(f"Error deleting {cache_file}: {e}")
        
        print(f"Cleared {deleted} cache files")
        return deleted


# Example usage
if __name__ == "__main__":
    # Initialize collector
    collector = UnifiedDataCollector()
    
    # Fetch BTC data with automatic fallback
    print("Fetching BTC data...")
    btc_data = collector.fetch_data(
        symbol="BTCUSD",
        start="2025-08-01",
        end="2025-08-10",
        timespan="hour"
    )
    
    if not btc_data.empty:
        print(f"\nData shape: {btc_data.shape}")
        print(f"Date range: {btc_data['timestamp'].min()} to {btc_data['timestamp'].max()}")
        print(f"Price range: ${btc_data['close'].min():.2f} - ${btc_data['close'].max():.2f}")
        print("\nFirst 5 rows:")
        print(btc_data.head())
    
    # Get latest quote
    print("\n" + "="*50)
    print("Fetching latest BTC quote...")
    quote = collector.fetch_latest_quote("BTCUSD")
    if quote:
        print(f"Current price: ${quote['price']:,.2f}")
        print(f"24h Volume: ${quote['volume_24h']:,.0f}")