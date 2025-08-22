#!/usr/bin/env python3


import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Using fallback implementations.")

class BaseStrategy(ABC):
    
    def __init__(self, name: str, params: Optional[Dict] = None):
        self.name = name
        self.params = params or {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement generate_signals")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        return data

class BuyAndHoldStrategy(BaseStrategy):
    
    def __init__(self):
        super().__init__("BuyAndHold")
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        
        data = self.preprocess_data(data)
        return np.ones(len(data))

class MomentumStrategy(BaseStrategy):
    
    def __init__(self, lookback: int = 20, threshold: float = 0.0):
        super().__init__("Momentum", {"lookback": lookback, "threshold": threshold})
        self.lookback = lookback
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        data = self.preprocess_data(data)
        
        # Calculate momentum (rate of change)
        momentum = data['close'].pct_change(self.lookback)
        
        # Generate signals
        signals = np.zeros(len(data))
        signals[momentum > self.threshold] = 1  # Long
        signals[momentum < -self.threshold] = -1  # Short
        
        # Fill initial NaN values
        signals[:self.lookback] = 0
        
        return signals

class MeanReversionStrategy(BaseStrategy):
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__("MeanReversion", {"period": period, "num_std": num_std})
        self.period = period
        self.num_std = num_std
    
    def _calculate_bollinger_bands(self, prices: np.ndarray):
        middle = pd.Series(prices).rolling(window=self.period).mean().values
        std = pd.Series(prices).rolling(window=self.period).std().values
        upper = middle + (std * self.num_std)
        lower = middle - (std * self.num_std)
        return upper, middle, lower
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        data = self.preprocess_data(data)
        
        # Calculate Bollinger Bands
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(
                data['close'].values,
                timeperiod=self.period,
                nbdevup=self.num_std,
                nbdevdn=self.num_std
            )
        else:
            upper, middle, lower = self._calculate_bollinger_bands(data['close'].values)
        
        # Generate signals
        signals = np.zeros(len(data))
        
        # Long when price touches lower band (oversold)
        signals[data['close'].values < lower] = 1
        
        # Short when price touches upper band (overbought)
        signals[data['close'].values > upper] = -1
        
        # Exit when price crosses middle band
        for i in range(1, len(signals)):
            if signals[i] == 0:
                if signals[i-1] == 1 and data['close'].iloc[i] > middle[i]:
                    signals[i] = 0
                elif signals[i-1] == -1 and data['close'].iloc[i] < middle[i]:
                    signals[i] = 0
                else:
                    signals[i] = signals[i-1]
        
        return signals

class TrendFollowingStrategy(BaseStrategy):
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__("TrendFollowing", {"fast_period": fast_period, "slow_period": slow_period})
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def _calculate_ema(self, prices: np.ndarray, period: int):
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        data = self.preprocess_data(data)
        
        # Calculate moving averages
        if TALIB_AVAILABLE:
            fast_ma = talib.EMA(data['close'].values, timeperiod=self.fast_period)
            slow_ma = talib.EMA(data['close'].values, timeperiod=self.slow_period)
        else:
            fast_ma = self._calculate_ema(data['close'].values, self.fast_period)
            slow_ma = self._calculate_ema(data['close'].values, self.slow_period)
        
        # Generate signals
        signals = np.zeros(len(data))
        
        # Long when fast MA crosses above slow MA
        signals[(fast_ma > slow_ma)] = 1
        
        # Short when fast MA crosses below slow MA
        signals[(fast_ma < slow_ma)] = -1
        
        # Fill initial NaN values
        signals[:self.slow_period] = 0
        
        return signals

class RSIMeanReversionStrategy(BaseStrategy):
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSIMeanReversion", {
            "period": period,
            "oversold": oversold,
            "overbought": overbought
        })
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def _calculate_rsi(self, prices: np.ndarray):
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        data = self.preprocess_data(data)
        
        # Calculate RSI
        if TALIB_AVAILABLE:
            rsi = talib.RSI(data['close'].values, timeperiod=self.period)
        else:
            rsi = self._calculate_rsi(data['close'].values)
        
        # Generate signals
        signals = np.zeros(len(data))
        
        # Long when RSI < oversold
        signals[rsi < self.oversold] = 1
        
        # Short when RSI > overbought
        signals[rsi > self.overbought] = -1
        
        # Hold position until opposite signal
        for i in range(1, len(signals)):
            if signals[i] == 0:
                signals[i] = signals[i-1]
        
        return signals

class MACDStrategy(BaseStrategy):
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD", {"fast": fast, "slow": slow, "signal": signal})
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def _calculate_macd(self, prices: np.ndarray):
        prices_series = pd.Series(prices)
        ema_fast = prices_series.ewm(span=self.fast, adjust=False).mean()
        ema_slow = prices_series.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd.values, macd_signal.values, macd_hist.values
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        data = self.preprocess_data(data)
        
        # Calculate MACD
        if TALIB_AVAILABLE:
            macd, macd_signal, macd_hist = talib.MACD(
                data['close'].values,
                fastperiod=self.fast,
                slowperiod=self.slow,
                signalperiod=self.signal
            )
        else:
            macd, macd_signal, macd_hist = self._calculate_macd(data['close'].values)
        
        # Generate signals
        signals = np.zeros(len(data))
        
        # Long when MACD crosses above signal line
        signals[(macd > macd_signal) & (macd_hist > 0)] = 1
        
        # Short when MACD crosses below signal line
        signals[(macd < macd_signal) & (macd_hist < 0)] = -1
        
        # Fill NaN values
        signals[:self.slow + self.signal] = 0
        
        return signals

class BreakoutStrategy(BaseStrategy):
    
    def __init__(self, lookback: int = 20, breakout_factor: float = 1.0):
        super().__init__("Breakout", {"lookback": lookback, "breakout_factor": breakout_factor})
        self.lookback = lookback
        self.breakout_factor = breakout_factor
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        data = self.preprocess_data(data)
        
        # Calculate rolling high and low
        rolling_high = data['high'].rolling(window=self.lookback).max()
        rolling_low = data['low'].rolling(window=self.lookback).min()
        
        # Calculate breakout thresholds
        high_breakout = rolling_high * (1 + self.breakout_factor / 100)
        low_breakout = rolling_low * (1 - self.breakout_factor / 100)
        
        # Generate signals
        signals = np.zeros(len(data))
        
        # Long on upward breakout
        signals[data['close'] > high_breakout] = 1
        
        # Short on downward breakout
        signals[data['close'] < low_breakout] = -1
        
        # Hold position until opposite breakout
        for i in range(1, len(signals)):
            if signals[i] == 0:
                signals[i] = signals[i-1]
        
        return signals

class VolumeWeightedMomentumStrategy(BaseStrategy):
    
    def __init__(self, lookback: int = 20, volume_ma: int = 20):
        super().__init__("VolumeWeightedMomentum", {
            "lookback": lookback,
            "volume_ma": volume_ma
        })
        self.lookback = lookback
        self.volume_ma = volume_ma
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        data = self.preprocess_data(data)
        
        # Calculate momentum
        momentum = data['close'].pct_change(self.lookback)
        
        # Calculate volume ratio
        volume_ratio = data['volume'] / data['volume'].rolling(self.volume_ma).mean()
        
        # Weight momentum by volume
        weighted_momentum = momentum * volume_ratio
        
        # Generate signals
        signals = np.zeros(len(data))
        
        # Use percentile thresholds
        upper_threshold = np.nanpercentile(weighted_momentum, 70)
        lower_threshold = np.nanpercentile(weighted_momentum, 30)
        
        signals[weighted_momentum > upper_threshold] = 1
        signals[weighted_momentum < lower_threshold] = -1
        
        # Fill NaN values
        signals[:max(self.lookback, self.volume_ma)] = 0
        
        return signals

class AdaptiveTrendStrategy(BaseStrategy):
    
    def __init__(self, period: int = 10):
        super().__init__("AdaptiveTrend", {"period": period})
        self.period = period
    
    def _calculate_adaptive_ma(self, prices: np.ndarray):
        prices_series = pd.Series(prices)
        # Use exponential moving average with varying alpha based on volatility
        volatility = prices_series.rolling(self.period).std()
        alpha = 2.0 / (self.period + 1)
        
        # Adjust alpha based on volatility (higher volatility = faster adaptation)
        norm_vol = volatility / volatility.rolling(50).mean()
        adaptive_alpha = alpha * (1 + norm_vol.fillna(1))
        adaptive_alpha = np.clip(adaptive_alpha, 0.01, 0.99)
        
        # Calculate adaptive EMA
        ema = prices_series.ewm(alpha=adaptive_alpha, adjust=False).mean()
        return ema.values
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        data = self.preprocess_data(data)
        
        # Calculate adaptive moving average
        if TALIB_AVAILABLE:
            # Use KAMA if available
            kama = talib.KAMA(data['close'].values, timeperiod=self.period)
        else:
            kama = self._calculate_adaptive_ma(data['close'].values)
        
        # Calculate trend strength using efficiency ratio
        price_change = abs(data['close'].diff(self.period))
        volatility = data['close'].diff().abs().rolling(self.period).sum()
        efficiency_ratio = price_change / (volatility + 1e-10)
        
        # Generate signals
        signals = np.zeros(len(data))
        
        # Long when price above adaptive MA and trend is strong
        signals[(data['close'].values > kama) & (efficiency_ratio > 0.3)] = 1
        
        # Short when price below adaptive MA and trend is strong
        signals[(data['close'].values < kama) & (efficiency_ratio > 0.3)] = -1
        
        # Flat when trend is weak
        signals[efficiency_ratio <= 0.3] = 0
        
        # Fill NaN values
        signals[:self.period] = 0
        
        return signals

class CompositeStrategy(BaseStrategy):
    
    def __init__(self, strategies: Dict[BaseStrategy, float]):

        super().__init__("Composite")
        self.strategies = strategies
        
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        data = self.preprocess_data(data)
        
        # Collect signals from all strategies
        all_signals = []
        weights = []
        
        for strategy, weight in self.strategies.items():
            signals = strategy.generate_signals(data)
            all_signals.append(signals)
            weights.append(weight)
        
        # Combine signals using weighted average
        all_signals = np.array(all_signals)
        weights = np.array(weights) / np.sum(weights)  # Normalize weights
        
        combined_signals = np.average(all_signals, weights=weights, axis=0)
        
        # Convert to discrete signals
        final_signals = np.zeros(len(data))
        final_signals[combined_signals > 0.3] = 1
        final_signals[combined_signals < -0.3] = -1
        
        return final_signals

def create_strategy_suite() -> Dict[str, BaseStrategy]:
    strategies = {
        'buy_and_hold': BuyAndHoldStrategy(),
        'momentum': MomentumStrategy(lookback=20),
        'mean_reversion': MeanReversionStrategy(period=20, num_std=2),
        'trend_following': TrendFollowingStrategy(fast_period=10, slow_period=30),
        'rsi_mean_reversion': RSIMeanReversionStrategy(period=14),
        'macd': MACDStrategy(),
        'breakout': BreakoutStrategy(lookback=20),
        'volume_momentum': VolumeWeightedMomentumStrategy(),
        'adaptive_trend': AdaptiveTrendStrategy(),
    }
    
    # Add a composite strategy
    composite_strategies = {
        TrendFollowingStrategy(): 0.3,
        MomentumStrategy(): 0.3,
        RSIMeanReversionStrategy(): 0.2,
        VolumeWeightedMomentumStrategy(): 0.2
    }
    strategies['composite'] = CompositeStrategy(composite_strategies)
    
    return strategies