#!/usr/bin/env python3
"""
Feature engineering pipeline for ML comparison research
Creates technical indicators and market structure features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Comprehensive feature engineering for crypto and equity data
    """
    
    def __init__(self, feature_config: Optional[Dict] = None):
        """
        Initialize feature engineer
        
        Args:
            feature_config: Configuration for feature selection
        """
        self.feature_config = feature_config or self.get_default_config()
        self.scaler = None
        
    def get_default_config(self) -> Dict:
        """Get default feature configuration"""
        return {
            'price_features': True,
            'volume_features': True,
            'technical_indicators': True,
            'volatility_features': True,
            'market_structure': True,
            'time_features': True,
            'lookback_periods': [7, 14, 30, 60],  # Days for rolling features
            'ta_periods': {
                'rsi': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'bb_std': 2,
                'adx': 14,
                'atr': 14
            }
        }
    
    def create_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Create all features for a dataframe
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name for asset-specific features
            
        Returns:
            DataFrame with engineered features
        """
        # Create copy to avoid modifying original
        data = df.copy()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"Missing required columns. Found: {data.columns.tolist()}")
            return data
        
        # Convert to float
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Basic price features
        if self.feature_config['price_features']:
            data = self.add_price_features(data)
        
        # Volume features
        if self.feature_config['volume_features']:
            data = self.add_volume_features(data)
        
        # Technical indicators
        if self.feature_config['technical_indicators']:
            data = self.add_technical_indicators(data)
        
        # Volatility features
        if self.feature_config['volatility_features']:
            data = self.add_volatility_features(data)
        
        # Market structure features
        if self.feature_config['market_structure']:
            data = self.add_market_structure_features(data)
        
        # Time-based features
        if self.feature_config['time_features']:
            data = self.add_time_features(data)
        
        # Asset-specific features
        if symbol:
            data = self.add_symbol_specific_features(data, symbol)
        
        # Clean up NaN values from feature creation
        data = self.handle_missing_values(data)
        
        return data
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Price position in range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Rolling price features
        for period in self.feature_config['lookback_periods']:
            # Moving averages
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # Price relative to MA
            df[f'close_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
            df[f'close_to_ema_{period}'] = df['close'] / df[f'ema_{period}']
            
            # Rolling min/max
            df[f'rolling_min_{period}'] = df['close'].rolling(period).min()
            df[f'rolling_max_{period}'] = df['close'].rolling(period).max()
            df[f'price_range_position_{period}'] = (
                (df['close'] - df[f'rolling_min_{period}']) / 
                (df[f'rolling_max_{period}'] - df[f'rolling_min_{period}'] + 1e-10)
            )
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume moving averages
        for period in self.feature_config['lookback_periods']:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        
        # Volume-Weighted Average Price (VWAP)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['close_to_vwap'] = df['close'] / df['vwap']
        
        # Money Flow
        df['money_flow'] = df['close'] * df['volume']
        df['money_flow_ratio'] = df['money_flow'] / df['money_flow'].rolling(14).mean()
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        config = self.feature_config['ta_periods']
        
        # RSI
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=config['rsi'])
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            df['close'].values,
            fastperiod=config['macd_fast'],
            slowperiod=config['macd_slow'],
            signalperiod=config['macd_signal']
        )
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df['close'].values,
            timeperiod=config['bb_period'],
            nbdevup=config['bb_std'],
            nbdevdn=config['bb_std']
        )
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        df['bb_position'] = (df['close'] - lower) / (upper - lower + 1e-10)
        
        # ADX (Average Directional Index)
        df['adx'] = talib.ADX(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            timeperiod=config['adx']
        )
        
        # ATR (Average True Range)
        df['atr'] = talib.ATR(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            timeperiod=config['atr']
        )
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            fastk_period=14,
            slowk_period=3,
            slowd_period=3
        )
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        # Williams %R
        df['williams_r'] = talib.WILLR(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            timeperiod=14
        )
        
        # CCI (Commodity Channel Index)
        df['cci'] = talib.CCI(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            timeperiod=20
        )
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        # Historical volatility
        for period in self.feature_config['lookback_periods']:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'volatility_ann_{period}'] = df[f'volatility_{period}'] * np.sqrt(252)
        
        # Parkinson volatility (using high-low)
        df['parkinson_vol'] = np.sqrt(
            np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))
        ).rolling(20).mean()
        
        # Garman-Klass volatility
        df['garman_klass_vol'] = np.sqrt(
            0.5 * np.log(df['high'] / df['low']) ** 2 -
            (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2
        ).rolling(20).mean()
        
        # Volatility regime
        vol_median = df['volatility_30'].median()
        df['vol_regime'] = (df['volatility_30'] > vol_median).astype(int)
        
        return df
    
    def add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Spread measures
        df['bid_ask_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Price efficiency
        df['efficiency_ratio'] = abs(
            df['close'].diff(10)
        ) / df['close'].diff().abs().rolling(10).sum()
        
        # Trend strength
        for period in [20, 50]:
            # Linear regression slope
            def calc_slope(x):
                if len(x) < 2:
                    return 0
                return np.polyfit(range(len(x)), x, 1)[0]
            
            df[f'trend_slope_{period}'] = df['close'].rolling(period).apply(calc_slope)
            
            # R-squared of trend
            def calc_r2(x):
                if len(x) < 2:
                    return 0
                y = np.array(x)
                x = np.arange(len(y))
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                return 1 - (ss_res / (ss_tot + 1e-10))
            
            df[f'trend_r2_{period}'] = df['close'].rolling(period).apply(calc_r2)
        
        # Support/Resistance levels
        for period in [20, 50]:
            df[f'distance_from_high_{period}'] = (
                df['close'] / df['high'].rolling(period).max() - 1
            )
            df[f'distance_from_low_{period}'] = (
                df['close'] / df['low'].rolling(period).min() - 1
            )
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'timestamp' in df.columns:
            # Convert to datetime if needed
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Time of day features (for hourly data)
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Trading session indicators (crypto 24/7, stocks have sessions)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['is_european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['is_us_session'] = ((df['hour'] >= 14) & (df['hour'] < 22)).astype(int)
        
        return df
    
    def add_symbol_specific_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add features specific to certain assets"""
        # Crypto-specific features
        if 'USD' in symbol:
            df['is_crypto'] = 1
            
            # Bitcoin dominance proxy (if BTC)
            if 'BTC' in symbol:
                df['is_bitcoin'] = 1
            else:
                df['is_bitcoin'] = 0
                
            # High volatility asset flag
            if 'SOL' in symbol:
                df['is_high_vol'] = 1
            else:
                df['is_high_vol'] = 0
        else:
            # Equity-specific
            df['is_crypto'] = 0
            df['is_bitcoin'] = 0
            df['is_high_vol'] = 0
            
            # Index type
            if symbol == 'SPY':
                df['is_sp500'] = 1
            elif symbol == 'QQQ':
                df['is_nasdaq'] = 1
            elif symbol == 'IWM':
                df['is_russell'] = 1
            elif symbol == 'DIA':
                df['is_dow'] = 1
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values from feature creation"""
        # Forward fill for most features
        df = df.fillna(method='ffill', limit=5)
        
        # Backward fill for any remaining
        df = df.fillna(method='bfill', limit=5)
        
        # Drop rows with too many NaN values
        nan_threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=nan_threshold)
        
        # Fill any remaining with 0
        df = df.fillna(0)
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame, horizons: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Create target variables for prediction
        
        Args:
            df: DataFrame with features
            horizons: List of prediction horizons (in periods)
        
        Returns:
            DataFrame with target variables
        """
        for horizon in horizons:
            # Future returns
            df[f'target_return_{horizon}'] = df['close'].shift(-horizon) / df['close'] - 1
            
            # Classification targets (up/down)
            df[f'target_direction_{horizon}'] = (df[f'target_return_{horizon}'] > 0).astype(int)
            
            # Multi-class targets (strong down, down, flat, up, strong up)
            returns = df[f'target_return_{horizon}']
            conditions = [
                returns < -0.02,
                (returns >= -0.02) & (returns < -0.005),
                (returns >= -0.005) & (returns < 0.005),
                (returns >= 0.005) & (returns < 0.02),
                returns >= 0.02
            ]
            choices = [0, 1, 2, 3, 4]  # strong_down, down, flat, up, strong_up
            df[f'target_multiclass_{horizon}'] = np.select(conditions, choices, default=2)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, 
                          method: str = 'standard',
                          fit: bool = True) -> pd.DataFrame:
        """
        Normalize features for ML models
        
        Args:
            df: DataFrame with features
            method: 'standard' or 'robust'
            fit: Whether to fit the scaler or use existing
        
        Returns:
            Normalized DataFrame
        """
        # Identify numeric columns (exclude targets and identifiers)
        exclude_cols = ['timestamp', 'symbol', 'data_usage'] + \
                      [col for col in df.columns if 'target' in col]
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        if fit:
            if method == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = RobustScaler()
            
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            if self.scaler is not None:
                df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        return df
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Calculate feature importance using correlation and mutual information
        
        Args:
            df: DataFrame with features
            target_col: Name of target column
        
        Returns:
            DataFrame with feature importance scores
        """
        from sklearn.feature_selection import mutual_info_regression
        
        # Get feature columns
        exclude_cols = ['timestamp', 'symbol', 'data_usage'] + \
                      [col for col in df.columns if 'target' in col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Calculate correlations
        correlations = df[feature_cols].corrwith(df[target_col]).abs()
        
        # Calculate mutual information
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Combine scores
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'correlation': correlations.values,
            'mutual_info': mi_scores
        })
        importance_df['combined_score'] = (
            importance_df['correlation'] * 0.5 + 
            importance_df['mutual_info'] / importance_df['mutual_info'].max() * 0.5
        )
        importance_df = importance_df.sort_values('combined_score', ascending=False)
        
        return importance_df