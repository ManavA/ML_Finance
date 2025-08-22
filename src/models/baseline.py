# src/models/baseline.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import talib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import logging

logger = logging.getLogger(__name__)


class BaselineModel(ABC):
    
    @abstractmethod
    def fit(self, data: pd.DataFrame):
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_signals(self, data: pd.DataFrame) -> np.ndarray:
        pass



class BuyAndHoldStrategy(BaselineModel):
    
    def __init__(self):
        self.name = "Buy & Hold"
        
    def fit(self, data: pd.DataFrame):
        pass
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return data['close'].values
    
    def get_signals(self, data: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(data))
        signals[0] = 1  # Buy at the beginning
        return signals


class RandomStrategy(BaselineModel):
    
    def __init__(self, seed: int = 42, trade_probability: float = 0.1):
        self.name = "Random"
        self.seed = seed
        self.trade_probability = trade_probability
        np.random.seed(seed)
        
    def fit(self, data: pd.DataFrame):
        pass
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        current_prices = data['close'].values
        random_changes = np.random.randn(len(data)) * np.std(data['close'].pct_change())
        return current_prices * (1 + random_changes)
    
    def get_signals(self, data: pd.DataFrame) -> np.ndarray:
        return np.random.choice([-1, 0, 1], size=len(data), 
                               p=[self.trade_probability/2, 
                                  1-self.trade_probability, 
                                  self.trade_probability/2])


class SMAcrossoverStrategy(BaselineModel):
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.name = f"SMA({fast_period},{slow_period})"
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def fit(self, data: pd.DataFrame):
        pass
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        sma_fast = data['close'].rolling(window=self.fast_period).mean()
        sma_slow = data['close'].rolling(window=self.slow_period).mean()
        
        # If fast > slow, predict price increase
        predictions = data['close'].values.copy()
        trend = (sma_fast > sma_slow).fillna(False)
        
        for i in range(1, len(predictions)):
            if trend.iloc[i-1]:
                predictions[i] = predictions[i-1] * 1.01  # 1% increase
            else:
                predictions[i] = predictions[i-1] * 0.99  # 1% decrease
                
        return predictions
    
    def get_signals(self, data: pd.DataFrame) -> np.ndarray:
        sma_fast = data['close'].rolling(window=self.fast_period).mean()
        sma_slow = data['close'].rolling(window=self.slow_period).mean()
        
        signals = np.zeros(len(data))
        
        # Buy when fast crosses above slow
        signals[(sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))] = 1
        
        # Sell when fast crosses below slow
        signals[(sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))] = -1
        
        return signals


class RSIStrategy(BaselineModel):
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        self.name = f"RSI({period})"
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        
    def fit(self, data: pd.DataFrame):
        pass
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        rsi = talib.RSI(data['close'].values, timeperiod=self.period)
        
        predictions = data['close'].values.copy()
        
        for i in range(self.period, len(predictions)):
            if rsi[i-1] < self.oversold:
                predictions[i] = predictions[i-1] * 1.02  # Expect bounce
            elif rsi[i-1] > self.overbought:
                predictions[i] = predictions[i-1] * 0.98  # Expect pullback
            else:
                predictions[i] = predictions[i-1]  # No change
                
        return predictions
    
    def get_signals(self, data: pd.DataFrame) -> np.ndarray:
        rsi = talib.RSI(data['close'].values, timeperiod=self.period)
        
        signals = np.zeros(len(data))
        
        # Buy when oversold
        signals[rsi < self.oversold] = 1
        
        # Sell when overbought
        signals[rsi > self.overbought] = -1
        
        return signals


class BollingerBandsStrategy(BaselineModel):
    
    def __init__(self, period: int = 20, num_std: float = 2):
        self.name = f"BB({period},{num_std})"
        self.period = period
        self.num_std = num_std
        
    def fit(self, data: pd.DataFrame):
        pass
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        upper, middle, lower = talib.BBANDS(
            data['close'].values,
            timeperiod=self.period,
            nbdevup=self.num_std,
            nbdevdn=self.num_std,
            matype=0
        )
        
        predictions = data['close'].values.copy()
        
        for i in range(self.period, len(predictions)):
            if predictions[i-1] < lower[i-1]:
                predictions[i] = middle[i-1]  # Expect reversion to mean
            elif predictions[i-1] > upper[i-1]:
                predictions[i] = middle[i-1]  # Expect reversion to mean
            else:
                predictions[i] = predictions[i-1]
                
        return predictions
    
    def get_signals(self, data: pd.DataFrame) -> np.ndarray:
        upper, middle, lower = talib.BBANDS(
            data['close'].values,
            timeperiod=self.period,
            nbdevup=self.num_std,
            nbdevdn=self.num_std,
            matype=0
        )
        
        signals = np.zeros(len(data))
        close = data['close'].values
        
        # Buy when price touches lower band
        signals[close <= lower] = 1
        
        # Sell when price touches upper band
        signals[close >= upper] = -1
        
        return signals


class MACDStrategy(BaselineModel):
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.name = f"MACD({fast},{slow},{signal})"
        self.fast = fast
        self.slow = slow
        self.signal = signal
        
    def fit(self, data: pd.DataFrame):
        pass
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        
        macd, signal, hist = talib.MACD(
            data['close'].values,
            fastperiod=self.fast,
            slowperiod=self.slow,
            signalperiod=self.signal
        )
        
        predictions = data['close'].values.copy()
        
        for i in range(self.slow, len(predictions)):
            if hist[i-1] > 0 and hist[i-2] <= 0:  # Bullish crossover
                predictions[i] = predictions[i-1] * 1.02
            elif hist[i-1] < 0 and hist[i-2] >= 0:  # Bearish crossover
                predictions[i] = predictions[i-1] * 0.98
            else:
                predictions[i] = predictions[i-1]
                
        return predictions
    
    def get_signals(self, data: pd.DataFrame) -> np.ndarray:
        macd, signal_line, hist = talib.MACD(
            data['close'].values,
            fastperiod=self.fast,
            slowperiod=self.slow,
            signalperiod=self.signal
        )
        
        signals = np.zeros(len(data))
        
        # Buy when MACD crosses above signal
        for i in range(1, len(data)):
            if i >= self.slow:
                if macd[i] > signal_line[i] and macd[i-1] <= signal_line[i-1]:
                    signals[i] = 1
                elif macd[i] < signal_line[i] and macd[i-1] >= signal_line[i-1]:
                    signals[i] = -1
                    
        return signals



class ARIMAModel(BaselineModel):
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 2)):
        self.name = f"ARIMA{order}"
        self.order = order
        self.model = None
        self.last_train_size = 0
        
    def fit(self, data: pd.DataFrame):
        try:
            self.model = ARIMA(data['close'].values, order=self.order)
            self.model_fit = self.model.fit()
            self.last_train_size = len(data)
            logger.info(f"ARIMA model fitted with order {self.order}")
        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            # Fallback to simpler model
            self.model = ARIMA(data['close'].values, order=(1, 1, 1))
            self.model_fit = self.model.fit()
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.model_fit is None:
            self.fit(data)
        
        try:
            # Forecast next periods
            forecast = self.model_fit.forecast(steps=len(data))
            return forecast
        except:
            # Return last known value as prediction
            return np.full(len(data), data['close'].iloc[-1])
    
    def get_signals(self, data: pd.DataFrame) -> np.ndarray:
        predictions = self.predict(data)
        current_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        # Buy if predicted price > current price by threshold
        signals[predictions > current_prices * 1.01] = 1
        
        # Sell if predicted price < current price by threshold
        signals[predictions < current_prices * 0.99] = -1
        
        return signals


class ProphetModel(BaselineModel):
    
    
    def __init__(self, changepoint_prior_scale: float = 0.05):
        self.name = "Prophet"
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
        
    def fit(self, data: pd.DataFrame):
        
        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': data.index,
            'y': data['close'].values
        })
        
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        self.model.fit(df_prophet)
        logger.info("Prophet model fitted")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        
        if self.model is None:
            self.fit(data)
        
        # Create future dataframe
        future = pd.DataFrame({'ds': data.index})
        forecast = self.model.predict(future)
        
        return forecast['yhat'].values
    
    def get_signals(self, data: pd.DataFrame) -> np.ndarray:
        
        predictions = self.predict(data)
        current_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        # Generate signals based on predicted trend
        for i in range(1, len(data)):
            if predictions[i] > predictions[i-1] * 1.01:
                signals[i] = 1
            elif predictions[i] < predictions[i-1] * 0.99:
                signals[i] = -1
                
        return signals


class RandomForestModel(BaselineModel):
    
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.name = f"RandomForest({n_estimators})"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.feature_importance = None
        
    def fit(self, data: pd.DataFrame):
        
        X, y = self._prepare_features(data)
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        self.feature_importance = self.model.feature_importances_
        logger.info(f"Random Forest fitted with {self.n_estimators} trees")
    
    def _prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        
        features = []
        
        # Price features
        features.append(data['close'].values)
        features.append(data['volume'].values)
        features.append(data['high'].values - data['low'].values)  # Range
        
        # Technical indicators
        features.append(talib.RSI(data['close'].values, timeperiod=14))
        features.append(talib.SMA(data['close'].values, timeperiod=20))
        
        # Returns
        features.append(data['close'].pct_change().fillna(0).values)
        
        # Stack features
        X = np.column_stack(features)[20:]  # Remove NaN rows
        
        # Target is next period's price
        y = data['close'].shift(-1).fillna(method='ffill').values[20:]
        
        return X[:-1], y[:-1]  # Remove last row (no target)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        
        if self.model is None:
            self.fit(data)
        
        X, _ = self._prepare_features(data)
        
        # Predict
        predictions = self.model.predict(X)
        
        # Pad predictions to match data length
        full_predictions = np.full(len(data), np.nan)
        full_predictions[20:-1] = predictions
        full_predictions = pd.Series(full_predictions).fillna(method='ffill').fillna(method='bfill').values
        
        return full_predictions
    
    def get_signals(self, data: pd.DataFrame) -> np.ndarray:
        
        predictions = self.predict(data)
        current_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        # Generate signals based on predicted price change
        price_change = (predictions - current_prices) / current_prices
        
        signals[price_change > 0.01] = 1  # Buy if expecting >1% gain
        signals[price_change < -0.01] = -1  # Sell if expecting >1% loss
        
        return signals


class XGBoostModel(BaselineModel):
    
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1):
        self.name = f"XGBoost({n_estimators})"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        
    def fit(self, data: pd.DataFrame):
        
        X, y = self._prepare_features(data)
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y, eval_metric='rmse', verbose=False)
        logger.info(f"XGBoost fitted with {self.n_estimators} estimators")
    
    def _prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        
        features = []
        
        # Price and volume features
        features.append(data['close'].values)
        features.append(data['volume'].values)
        features.append(data['high'].values)
        features.append(data['low'].values)
        
        # Technical indicators
        features.append(talib.RSI(data['close'].values))
        features.append(talib.MACD(data['close'].values)[0])
        features.append(talib.ATR(data['high'].values, data['low'].values, data['close'].values))
        
        # Moving averages
        for period in [7, 14, 30]:
            features.append(talib.SMA(data['close'].values, timeperiod=period))
        
        # Volatility
        features.append(data['close'].pct_change().rolling(20).std().values)
        
        # Stack and clean
        X = np.column_stack(features)
        X = pd.DataFrame(X).fillna(method='ffill').fillna(0).values[30:]
        
        # Target
        y = data['close'].shift(-1).fillna(method='ffill').values[30:]
        
        return X[:-1], y[:-1]
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        
        if self.model is None:
            self.fit(data)
        
        X, _ = self._prepare_features(data)
        predictions = self.model.predict(X)
        
        # Pad predictions
        full_predictions = np.full(len(data), np.nan)
        full_predictions[30:-1] = predictions
        full_predictions = pd.Series(full_predictions).fillna(method='ffill').fillna(method='bfill').values
        
        return full_predictions
    
    def get_signals(self, data: pd.DataFrame) -> np.ndarray:
        
        predictions = self.predict(data)
        current_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        # Calculate expected returns
        returns = (predictions - current_prices) / current_prices
        
        # Generate signals with dynamic thresholds
        threshold = np.std(returns) * 0.5
        
        signals[returns > threshold] = 1
        signals[returns < -threshold] = -1
        
        return signals


class LightGBMModel(BaselineModel):
    
    
    def __init__(self, n_estimators: int = 100, num_leaves: int = 31,
                 learning_rate: float = 0.1):
        self.name = f"LightGBM({n_estimators})"
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.model = None
        
    def fit(self, data: pd.DataFrame):
        
        X, y = self._prepare_features(data)
        
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
        self.model.fit(X, y)
        logger.info(f"LightGBM fitted with {self.n_estimators} estimators")
    
    def _prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        
        # Similar to XGBoost but with some additional features
        features = []
        
        # Basic features
        for col in ['open', 'high', 'low', 'close', 'volume']:
            features.append(data[col].values)
        
        # Price ratios
        features.append(data['high'].values / data['low'].values)
        features.append(data['close'].values / data['open'].values)
        
        # Technical indicators
        features.append(talib.RSI(data['close'].values))
        features.append(talib.MFI(data['high'].values, data['low'].values, 
                                  data['close'].values, data['volume'].values))
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(data['close'].values)
        features.append(upper)
        features.append(lower)
        features.append((data['close'].values - middle) / (upper - middle))
        
        # Volume indicators
        features.append(talib.OBV(data['close'].values, data['volume'].values))
        features.append(talib.AD(data['high'].values, data['low'].values,
                                data['close'].values, data['volume'].values))
        
        # Clean and prepare
        X = np.column_stack(features)
        X = pd.DataFrame(X).fillna(method='ffill').fillna(0).values[40:]
        
        y = data['close'].shift(-1).fillna(method='ffill').values[40:]
        
        return X[:-1], y[:-1]
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        
        if self.model is None:
            self.fit(data)
        
        X, _ = self._prepare_features(data)
        predictions = self.model.predict(X)
        
        # Pad predictions
        full_predictions = np.full(len(data), np.nan)
        full_predictions[40:-1] = predictions
        full_predictions = pd.Series(full_predictions).fillna(method='ffill').fillna(method='bfill').values
        
        return full_predictions
    
    def get_signals(self, data: pd.DataFrame) -> np.ndarray:
        
        predictions = self.predict(data)
        current_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        # Use percentile-based thresholds
        returns = (predictions - current_prices) / current_prices
        
        buy_threshold = np.percentile(returns[returns > 0], 70) if np.any(returns > 0) else 0.01
        sell_threshold = np.percentile(returns[returns < 0], 30) if np.any(returns < 0) else -0.01
        
        signals[returns > buy_threshold] = 1
        signals[returns < sell_threshold] = -1
        
        return signals



class EnsembleBaselineModel(BaselineModel):
    
    
    def __init__(self, models: Optional[List[BaselineModel]] = None):
        self.name = "Ensemble"
        
        if models is None:
            # Default ensemble
            self.models = [
                SMAcrossoverStrategy(10, 30),
                RSIStrategy(),
                MACDStrategy(),
                BollingerBandsStrategy(),
                RandomForestModel(n_estimators=50),
                XGBoostModel(n_estimators=50)
            ]
        else:
            self.models = models
            
        self.weights = None
        
    def fit(self, data: pd.DataFrame):
        
        for model in self.models:
            try:
                model.fit(data)
                logger.info(f"Fitted {model.name}")
            except Exception as e:
                logger.error(f"Failed to fit {model.name}: {e}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        
        predictions = []
        
        for model in self.models:
            try:
                pred = model.predict(data)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Prediction failed for {model.name}: {e}")
                predictions.append(np.full(len(data), data['close'].iloc[-1]))
        
        # Simple average ensemble
        return np.mean(predictions, axis=0)
    
    def get_signals(self, data: pd.DataFrame) -> np.ndarray:
        
        all_signals = []
        
        for model in self.models:
            try:
                signals = model.get_signals(data)
                all_signals.append(signals)
            except Exception as e:
                logger.error(f"Signal generation failed for {model.name}: {e}")
                all_signals.append(np.zeros(len(data)))
        
        # Majority voting
        ensemble_signals = np.zeros(len(data))
        
        for i in range(len(data)):
            votes = [sig[i] for sig in all_signals]
            
            buy_votes = sum(1 for v in votes if v > 0)
            sell_votes = sum(1 for v in votes if v < 0)
            
            if buy_votes > len(self.models) / 2:
                ensemble_signals[i] = 1
            elif sell_votes > len(self.models) / 2:
                ensemble_signals[i] = -1
                
        return ensemble_signals