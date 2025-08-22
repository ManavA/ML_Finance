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
