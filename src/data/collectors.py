# src/data/collectors.py
import ccxt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import asyncio
import aiohttp
import concurrent.futures
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)


class CryptoDataCollector:
    
    
    def __init__(self, exchange_name: str = 'binance', 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 testnet: bool = False):
        
        self.exchange_name = exchange_name
        self.exchange = self._init_exchange(exchange_name, api_key, api_secret, testnet)
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        
    def _init_exchange(self, exchange_name: str, api_key: str, 
                      api_secret: str, testnet: bool) -> ccxt.Exchange:
        
        exchange_class = getattr(ccxt, exchange_name)
        
        config = {
            'enableRateLimit': True,
            'rateLimit': 50,
        }
        
        if api_key and api_secret:
            config['apiKey'] = api_key
            config['secret'] = api_secret
            
        if testnet and exchange_name == 'binance':
            config['options'] = {'defaultType': 'future', 'testnet': True}
            
        exchange = exchange_class(config)
        exchange.load_markets()
        
        return exchange
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h',
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   limit: int = 1000) -> pd.DataFrame:
        
        cache_key = f"{self.exchange_name}_{symbol.replace('/', '_')}_{timeframe}_{start_date}_{end_date}.pkl"
        cache_path = self.cache_dir / cache_key
        
        # Check cache first
        if cache_path.exists():
            logger.info(f"Loading cached data for {symbol}")
            return pd.read_pickle(cache_path)
        
        logger.info(f"Fetching {symbol} data from {self.exchange_name}")
        
        # Convert dates to timestamps
        since = None
        if start_date:
            since = self.exchange.parse8601(f"{start_date}T00:00:00Z")
            
        end_timestamp = None
        if end_date:
            end_timestamp = self.exchange.parse8601(f"{end_date}T23:59:59Z")
        
        all_ohlcv = []
        
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe, 
                    since=since, 
                    limit=limit
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                # Update since to last timestamp
                since = ohlcv[-1][0] + 1
                
                # Check if we've reached the end date
                if end_timestamp and since >= end_timestamp:
                    break
                    
                # Rate limiting
                if len(ohlcv) < limit:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Filter by end date if specified
        if end_date:
            df = df[df.index <= end_date]
        
        # Cache the data
        df.to_pickle(cache_path)
        logger.info(f"Cached data to {cache_path}")
        
        return df
    
    def fetch_multiple_symbols(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        
        data = {}
        for symbol in tqdm(symbols, desc="Fetching symbols"):
            try:
                data[symbol] = self.fetch_ohlcv(symbol, **kwargs)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        return data
    
    def fetch_multiple_symbols_parallel(self, symbols: List[str], max_workers: int = 5, **kwargs) -> Dict[str, pd.DataFrame]:
        
        data = {}
        
        def fetch_single_symbol(symbol):
            try:
                return symbol, self.fetch_ohlcv(symbol, **kwargs)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                return symbol, pd.DataFrame()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_single_symbol, symbol) for symbol in symbols]
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(symbols), desc="Fetching symbols"):
                symbol, df = future.result()
                if not df.empty:
                    data[symbol] = df
        
        return data
    
    def fetch_order_book(self, symbol: str, limit: int = 100) -> Dict:
        
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return {}
    
    def fetch_trades(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            df = pd.DataFrame(trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return pd.DataFrame()


# src/data/preprocessors.py
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import ta
from ta.utils import dropna
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    
    
    def __init__(self, sequence_length: int = 168, 
                 prediction_horizon: int = 24,
                 feature_config: Optional[Dict] = None):
        
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_config = feature_config or {}
        self.scalers = {}
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        
        # Trend indicators
        df['sma_7'] = ta.trend.sma_indicator(df['close'], window=7)
        df['sma_21'] = ta.trend.sma_indicator(df['close'], window=21)
        df['ema_7'] = ta.trend.ema_indicator(df['close'], window=7)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = bb.bollinger_wband()
        
        # ATR (Average True Range)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Volume indicators
        df['volume_sma'] = ta.volume.volume_weighted_average_price(
            df['high'], df['low'], df['close'], df['volume']
        )
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Remove NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def create_sequences(self, data: np.ndarray, 
                        targets: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        
        sequences = []
        target_values = []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            seq = data[i:i + self.sequence_length]
            
            if targets is not None:
                target = targets[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
            else:
                # Use closing price as target
                target = data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, 3]
            
            sequences.append(seq)
            target_values.append(target)
        
        return np.array(sequences), np.array(target_values)
    
    def scale_data(self, data: pd.DataFrame, 
                  scaler_type: str = 'standard',
                  fit: bool = True) -> np.ndarray:
        
        if scaler_type not in self.scalers and fit:
            if scaler_type == 'standard':
                self.scalers[scaler_type] = StandardScaler()
            elif scaler_type == 'robust':
                self.scalers[scaler_type] = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        scaler = self.scalers[scaler_type]
        
        if fit:
            scaled_data = scaler.fit_transform(data)
        else:
            scaled_data = scaler.transform(data)
        
        return scaled_data
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        
        # Add technical indicators
        if self.feature_config.get('technical_indicators', True):
            df = self.add_technical_indicators(df)
        
        # Select features
        feature_columns = self._select_features(df)
        features = df[feature_columns].values
        
        # Scale features
        features = self.scale_data(features, fit=True)
        
        # Create sequences
        X, y = self.create_sequences(features)
        
        return X, y
    
    def _select_features(self, df: pd.DataFrame) -> List[str]:
        
        features = []
        
        if self.feature_config.get('price', True):
            features.extend(['open', 'high', 'low', 'close'])
        
        if self.feature_config.get('volume', True):
            features.append('volume')
        
        if self.feature_config.get('technical_indicators', True):
            # Add all technical indicators
            ti_features = ['sma_7', 'sma_21', 'ema_7', 'ema_21', 'macd', 
                          'macd_signal', 'macd_diff', 'rsi', 'bb_width',
                          'atr', 'returns', 'volatility']
            features.extend([f for f in ti_features if f in df.columns])
        
        return features
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        
        if 'standard' in self.scalers:
            # Assuming predictions are for closing price (index 3)
            # Create dummy array with same shape as original features
            dummy = np.zeros((predictions.shape[0], self.scalers['standard'].n_features_in_))
            dummy[:, 3] = predictions.flatten()  # Put predictions in close price column
            inversed = self.scalers['standard'].inverse_transform(dummy)
            return inversed[:, 3].reshape(predictions.shape)
        return predictions


# src/data/datasets.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CryptoDataset(Dataset):
    
    
    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 transform: Optional = None):
        
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features[idx]
        targets = self.targets[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return features, targets


def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)
    test_dataset = CryptoDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader