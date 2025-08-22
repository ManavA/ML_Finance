# src/data/data_collection.py
"""
Simplified data collection utilities for research.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import os
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)

class DataCollector:
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_crypto_data(self, 
                         symbol: str, 
                         start_date: str, 
                         end_date: str,
                         interval: str = '1d',
                         use_cache: bool = True) -> pd.DataFrame:

        # Create cache key
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}.pkl"
        cache_path = self.cache_dir / cache_key
        
        # Check cache
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached data for {symbol}")
            return pd.read_pickle(cache_path)
        
        # Fetch from yfinance
        logger.info(f"Fetching {symbol} from yfinance")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        # Clean and standardize
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index.name = 'date'
        
        # Add returns and log returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Save to cache
        if use_cache:
            df.to_pickle(cache_path)
            logger.info(f"Cached data for {symbol}")
        
        return df
    
    def fetch_stock_data(self,
                        symbol: str,
                        start_date: str,
                        end_date: str,
                        interval: str = '1d') -> pd.DataFrame:

        # Same as crypto but with different symbols
        return self.fetch_crypto_data(symbol, start_date, end_date, interval)
    
    def fetch_multiple_assets(self,
                            symbols: List[str],
                            start_date: str,
                            end_date: str,
                            interval: str = '1d') -> Dict[str, pd.DataFrame]:

        data = {}
        for symbol in symbols:
            try:
                if symbol.startswith('^'):
                    # Index
                    df = self.fetch_stock_data(symbol, start_date, end_date, interval)
                elif '-USD' in symbol or symbol in ['BTC-USD', 'ETH-USD']:
                    # Crypto
                    df = self.fetch_crypto_data(symbol, start_date, end_date, interval)
                else:
                    # Stock
                    df = self.fetch_stock_data(symbol, start_date, end_date, interval)
                
                if not df.empty:
                    data[symbol] = df
                    
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                
        return data
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        
        # Simple Moving Averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_30'] = df['close'].rolling(window=30).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['atr'] = self.calculate_atr(df)
        
        # Price position
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def prepare_ml_features(self, 
                           df: pd.DataFrame,
                           lookback: int = 20) -> pd.DataFrame:
        df = df.copy()
        
        # Add lagged returns
        for i in range(1, lookback + 1):
            df[f'returns_lag_{i}'] = df['returns'].shift(i)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
            df[f'returns_kurt_{window}'] = df['returns'].rolling(window).kurt()
        
        # Price ratios
        df['price_to_sma20'] = df['close'] / df['sma_20']
        df['price_to_sma50'] = df['close'] / df['sma_50']
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['high_volume'] = (df['volume'] > df['volume_sma']).astype(int)
        
        # Target variable (next day return)
        df['target'] = df['returns'].shift(-1)
        
        # Drop NaN rows
        df = df.dropna()
        
        return df
    
    def get_risk_free_rate(self, start_date: str, end_date: str) -> float:

        try:
            # Fetch 3-month Treasury yield
            treasury = yf.Ticker('^IRX')
            df = treasury.history(start=start_date, end=end_date)
            if not df.empty:
                # Convert from percentage to decimal
                return df['Close'].mean() / 100
        except:
            pass
        
        # Default to 2% if can't fetch
        return 0.02


def create_sample_dataset(start_date: str = '2020-01-01',
                         end_date: str = '2024-01-01') -> Dict[str, pd.DataFrame]:

    collector = DataCollector()
    
    # Assets to analyze
    symbols = [
        'BTC-USD',   # Bitcoin
        'ETH-USD',   # Ethereum
        '^GSPC',     # S&P 500
        'GLD',       # Gold ETF (as alternative baseline)
    ]
    
    # Fetch all data
    data = collector.fetch_multiple_assets(symbols, start_date, end_date)
    
    # Add technical indicators to each
    for symbol in data:
        data[symbol] = collector.add_technical_indicators(data[symbol])
    
    return data