"""
Polygon.io S3 Flat File Data Collector
Provides access to bulk historical data without API rate limits
"""

import os
import sys
import pandas as pd
import boto3
from botocore.config import Config
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import gzip
import json
from io import BytesIO
import pyarrow.parquet as pq

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from normalize.ohlcv_schema import coerce_schema, validate_ohlcv


class PolygonS3Collector:
    """
    Collector for Polygon.io S3 flat files
    
    Flat files are organized by:
    - Asset class (stocks, options, crypto, forex)
    - Year/Month/Day
    - Available in CSV and Parquet formats
    """
    
    def __init__(self, cache_dir: str = "data/s3_cache"):
        """Initialize S3 client with Polygon credentials"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load credentials from environment
        from dotenv import load_dotenv
        load_dotenv()
        
        # S3 credentials
        self.access_key = os.getenv('POLYGONIO_S3_ACCESS_KEY_ID')
        self.secret_key = os.getenv('POLYGONIO_S3_SECRET_ACCESS_KEY')
        self.endpoint = os.getenv('POLYGONIO_S3_ENDPOINT', 'https://files.polygon.io')
        self.bucket = os.getenv('POLYGONIO_S3_BUCKET', 'flatfiles')
        
        if not self.access_key or not self.secret_key:
            raise ValueError("Polygon S3 credentials not found in environment")
        
        # Initialize S3 client with proper configuration
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name='us-east-1',  # Polygon uses us-east-1
            config=Config(
                signature_version='s3v4',
                s3={'addressing_style': 'path'}
            )
        )
        
        print(f"Initialized Polygon S3 client")
        print(f"Endpoint: {self.endpoint}")
        print(f"Bucket: {self.bucket}")
    
    def list_available_dates(self, market: str = "crypto", 
                           year: Optional[int] = None) -> List[str]:
        """
        List available dates for a given market
        
        Args:
            market: Market type (crypto, stocks, options, forex)
            year: Optional year to filter
            
        Returns:
            List of available date paths
        """
        prefix = f"us_{market}_aggs_minute/"
        if year:
            prefix += f"{year}/"
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                Delimiter='/'
            )
            
            if 'CommonPrefixes' in response:
                return [p['Prefix'] for p in response['CommonPrefixes']]
            else:
                return []
                
        except Exception as e:
            print(f"Error listing S3 objects: {e}")
            return []
    
    def fetch_minute_data(self, date: str, market: str = "crypto",
                         symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch minute-level data for a specific date
        
        Args:
            date: Date in YYYY-MM-DD format
            market: Market type (crypto, stocks, options, forex)
            symbols: Optional list of symbols to filter
            
        Returns:
            DataFrame with minute-level OHLCV data
        """
        # Convert date to path format
        date_obj = pd.to_datetime(date)
        year = date_obj.year
        month = str(date_obj.month).zfill(2)
        day = str(date_obj.day).zfill(2)
        
        # S3 key for the data file - use CSV.GZ format for crypto
        if market == "crypto":
            s3_key = f"global_crypto/minute_aggs_v1/{year}/{month}/{date_obj.strftime('%Y-%m-%d')}.csv.gz"
        else:
            s3_key = f"us_{market}_sip/minute_aggs_v1/{year}/{month}/{date_obj.strftime('%Y-%m-%d')}.csv.gz"
        
        # Check cache first
        cache_file = self.cache_dir / f"{market}_{date}.parquet"
        if cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
        else:
            try:
                print(f"Fetching from S3: {s3_key}")
                
                # Download from S3
                response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
                data = response['Body'].read()
                
                # Read data based on file extension
                if s3_key.endswith('.parquet'):
                    df = pd.read_parquet(BytesIO(data))
                elif s3_key.endswith('.csv.gz'):
                    import gzip
                    with gzip.GzipFile(fileobj=BytesIO(data)) as gz:
                        df = pd.read_csv(gz)
                
                # Save to cache
                df.to_parquet(cache_file)
                print(f"Cached to: {cache_file}")
                
            except self.s3_client.exceptions.NoSuchKey:
                print(f"No data available for {date} in {market}")
                return pd.DataFrame()
            except Exception as e:
                print(f"Error fetching S3 data: {e}")
                return pd.DataFrame()
        
        # Filter by symbols if provided
        if symbols and 'ticker' in df.columns:
            # Map symbols to Polygon format
            polygon_symbols = [self._to_polygon_symbol(s, market) for s in symbols]
            df = df[df['ticker'].isin(polygon_symbols)]
        
        # Standardize column names
        df = self._standardize_columns(df, market)
        
        return df
    
    def fetch_aggregated_data(self, symbol: str, start: str, end: str,
                            market: str = "crypto", 
                            timeframe: str = "hour") -> pd.DataFrame:
        """
        Fetch and aggregate data for a symbol over a date range
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            market: Market type
            timeframe: Target timeframe (minute, hour, day)
            
        Returns:
            Aggregated DataFrame
        """
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        
        all_data = []
        current_date = start_date
        
        # Fetch data day by day
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Fetch minute data for the day
            day_data = self.fetch_minute_data(date_str, market, [symbol])
            
            if not day_data.empty:
                all_data.append(day_data)
            
            current_date += timedelta(days=1)
        
        if not all_data:
            print(f"No data found for {symbol} from {start} to {end}")
            return pd.DataFrame()
        
        # Combine all data
        df = pd.concat(all_data, ignore_index=True)
        
        # Aggregate to requested timeframe
        if timeframe != "minute":
            df = self._aggregate_timeframe(df, timeframe)
        
        return df
    
    def fetch_bulk_historical(self, symbols: List[str], 
                            start: str, end: str,
                            market: str = "crypto") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols efficiently
        
        Args:
            symbols: List of symbols
            start: Start date
            end: End date
            market: Market type
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        
        # Initialize result dictionary
        result = {symbol: [] for symbol in symbols}
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Fetch all data for the day (more efficient than per-symbol)
            day_data = self.fetch_minute_data(date_str, market)
            
            if not day_data.empty:
                # Split by symbol
                for symbol in symbols:
                    polygon_symbol = self._to_polygon_symbol(symbol, market)
                    symbol_data = day_data[day_data['symbol'] == polygon_symbol]
                    if not symbol_data.empty:
                        result[symbol].append(symbol_data)
            
            current_date += timedelta(days=1)
        
        # Combine data for each symbol
        final_result = {}
        for symbol, data_list in result.items():
            if data_list:
                final_result[symbol] = pd.concat(data_list, ignore_index=True)
            else:
                final_result[symbol] = pd.DataFrame()
        
        return final_result
    
    def _to_polygon_symbol(self, symbol: str, market: str) -> str:
        """Convert symbol to Polygon format"""
        if market == "crypto":
            # Handle various formats - Polygon uses X:BTC-USD format
            symbol = symbol.upper()
            if 'USD' in symbol and not '-' in symbol:
                # Convert BTCUSD to BTC-USD
                symbol = symbol.replace('USD', '-USD')
            if not symbol.startswith('X:'):
                symbol = f"X:{symbol}"
            return symbol
        elif market == "forex":
            if not symbol.startswith('C:'):
                symbol = f"C:{symbol}"
            return symbol
        else:
            return symbol.upper()
    
    def _standardize_columns(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Standardize column names to our schema"""
        # Polygon flat file columns mapping (CSV format)
        column_map = {
            'ticker': 'symbol',
            'window_start': 'timestamp',  # CSV uses window_start
            't': 'timestamp',  # Parquet uses t
            'open': 'open',
            'o': 'open',
            'high': 'high', 
            'h': 'high',
            'low': 'low',
            'l': 'low',
            'close': 'close',
            'c': 'close',
            'volume': 'volume',
            'v': 'volume',
            'vw': 'vwap',
            'transactions': 'num_trades',
            'n': 'num_trades'
        }
        
        # Rename columns
        df = df.rename(columns=column_map)
        
        # Convert timestamp (Polygon uses Unix nanoseconds in CSV, milliseconds in Parquet)
        if 'timestamp' in df.columns:
            # Check if timestamp is too large (nanoseconds)
            if df['timestamp'].iloc[0] > 1e12:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True)
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Select and order required columns
        required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols]
        
        # Coerce to standard schema
        df = coerce_schema(df)
        
        return df
    
    def _aggregate_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Aggregate minute data to higher timeframes"""
        if df.empty or 'timestamp' not in df.columns:
            return df
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # Define aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Map timeframe to pandas frequency
        freq_map = {
            'hour': 'h',  # Use lowercase 'h' to avoid deprecation warning
            'day': 'D',
            'week': 'W',
            'month': 'M'
        }
        
        freq = freq_map.get(timeframe, 'h')  # Default to 'h' (hour)
        
        # Group by symbol if present
        if 'symbol' in df.columns:
            symbol = df['symbol'].iloc[0]
            df = df.drop('symbol', axis=1)
            
            # Resample and aggregate
            df_resampled = df.resample(freq).agg(agg_rules)
            
            # Remove rows with all NaN
            df_resampled = df_resampled.dropna(how='all')
            
            # Add symbol back
            df_resampled['symbol'] = symbol
        else:
            df_resampled = df.resample(freq).agg(agg_rules)
            df_resampled = df_resampled.dropna(how='all')
        
        # Reset index
        df_resampled = df_resampled.reset_index()
        
        return df_resampled
    
    def get_data_availability(self) -> Dict:
        """Check what data is available in S3"""
        markets = ['crypto', 'stocks', 'options', 'forex']
        availability = {}
        
        for market in markets:
            try:
                # List years available
                prefix = f"us_{market}_aggs_minute/"
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=prefix,
                    Delimiter='/',
                    MaxKeys=10
                )
                
                if 'CommonPrefixes' in response:
                    years = [p['Prefix'].split('/')[-2] for p in response['CommonPrefixes']]
                    availability[market] = {
                        'available': True,
                        'years': sorted(years),
                        'sample_path': f"{prefix}{years[-1]}/" if years else None
                    }
                else:
                    availability[market] = {'available': False}
                    
            except Exception as e:
                availability[market] = {'available': False, 'error': str(e)}
        
        return availability

    def fetch_crypto_day(self, ticker: str, date: str) -> pd.DataFrame:
        """
        Fetch daily crypto data for a specific date
        
        Args:
            ticker: Crypto ticker in format "X:BTCUSD"
            date: Date in format "YYYY-MM-DD"
            
        Returns:
            DataFrame with OHLCV data for the day
        """
        try:
            # Convert date string to datetime
            target_date = pd.to_datetime(date)
            
            # Remove X: prefix if present for symbol parameter
            symbol = ticker.replace("X:", "") if ticker.startswith("X:") else ticker
            
            # Use fetch_aggregated_data for daily data
            data = self.fetch_aggregated_data(
                symbol=symbol,
                start=date,
                end=date,
                market='crypto',
                timeframe='day'
            )
            
            if data is not None and not data.empty:
                return data
            
            # Fallback to minute data aggregated to daily
            minute_data = self.fetch_minute_data(
                date=date,
                market='crypto',
                symbols=[symbol]
            )
            
            if minute_data is not None and not minute_data.empty:
                # Aggregate to daily
                daily = minute_data.resample('D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                return daily
                
        except Exception as e:
            print(f"Error fetching crypto day data: {e}")
            
        return pd.DataFrame()
    
    def fetch_stock_day(self, ticker: str, date: str) -> pd.DataFrame:
        """
        Fetch daily stock data for a specific date
        
        Args:
            ticker: Stock ticker symbol (e.g., "SPY")
            date: Date in format "YYYY-MM-DD"
            
        Returns:
            DataFrame with OHLCV data for the day
        """
        try:
            # Use fetch_aggregated_data for daily data
            data = self.fetch_aggregated_data(
                symbol=ticker,
                start=date,
                end=date,
                market='stocks',
                timeframe='day'
            )
            
            if data is not None and not data.empty:
                return data
            
            # Fallback to minute data aggregated to daily
            minute_data = self.fetch_minute_data(
                date=date,
                market='stocks',
                symbols=[ticker]
            )
            
            if minute_data is not None and not minute_data.empty:
                # Aggregate to daily
                daily = minute_data.resample('D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                return daily
                
        except Exception as e:
            print(f"Error fetching stock day data: {e}")
            
        return pd.DataFrame()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Polygon S3 Flat File Collector")
    print("=" * 50)
    
    # Initialize collector
    collector = PolygonS3Collector()
    
    # Check data availability
    print("\nChecking S3 data availability...")
    availability = collector.get_data_availability()
    for market, info in availability.items():
        if info.get('available'):
            print(f"  {market}: Available (years: {', '.join(info.get('years', [])[:5])}...)")
        else:
            print(f"  {market}: Not available")
    
    # Test fetching recent crypto data
    print("\n" + "=" * 50)
    print("Testing crypto data fetch...")
    
    # Get data for last 3 days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    
    print(f"Fetching BTCUSD from {start_date} to {end_date}")
    
    btc_data = collector.fetch_aggregated_data(
        symbol='BTCUSD',
        start=start_date,
        end=end_date,
        market='crypto',
        timeframe='hour'
    )
    
    if not btc_data.empty:
        print(f"\nSuccess! Retrieved {len(btc_data)} hourly records")
        print(f"Date range: {btc_data['timestamp'].min()} to {btc_data['timestamp'].max()}")
        print(f"Price range: ${btc_data['close'].min():,.2f} - ${btc_data['close'].max():,.2f}")
        print("\nSample data:")
        print(btc_data.head())
    else:
        print("No data retrieved")
    
    # Test bulk fetch for multiple symbols
    print("\n" + "=" * 50)
    print("Testing bulk fetch for multiple symbols...")
    
    symbols = ['BTCUSD', 'ETHUSD']
    bulk_data = collector.fetch_bulk_historical(
        symbols=symbols,
        start=start_date,
        end=end_date,
        market='crypto'
    )
    
    for symbol, data in bulk_data.items():
        if not data.empty:
            print(f"\n{symbol}: {len(data)} records")
            print(f"  Latest price: ${data['close'].iloc[-1]:,.2f}")