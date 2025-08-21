# src/backtesting/engine.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    type: str  # 'buy' or 'sell'
    price: float
    quantity: float
    value: float
    fee: float
    position_after: float
    cash_after: float
    portfolio_value: float


@dataclass
class BacktestResults:
    """Results from backtesting."""
    trades: List[Trade]
    portfolio_values: np.ndarray
    returns: np.ndarray
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    exposure_time: float


class BacktestEngine:
    """Advanced backtesting engine for cryptocurrency strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        self.initial_capital = config.get('initial_capital', 10000)
        self.commission = config.get('commission', 0.001)  # 0.1%
        self.slippage = config.get('slippage', 0.001)  # 0.1%
        self.position_size = config.get('position_size', 1.0)  # Full capital
        self.max_positions = config.get('max_positions', 1)
        
        # State variables
        self.cash = self.initial_capital
        self.position = 0
        self.trades = []
        self.portfolio_values = []
        
    def run(self, data: pd.DataFrame, signals: pd.DataFrame,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None) -> BacktestResults:
        """
        Run backtesting on historical data.
        
        Args:
            data: Historical price data
            signals: Trading signals (-1, 0, 1)
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResults object
        """
        logger.info("Starting backtest...")
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
            signals = signals[signals.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            signals = signals[signals.index <= end_date]
        
        # Ensure alignment
        common_index = data.index.intersection(signals.index)
        data = data.loc[common_index]
        signals = signals.loc[common_index]
        
        # Reset state
        self.reset()
        
        # Run backtest
        for timestamp, signal in signals.iterrows():
            price = data.loc[timestamp, 'close']
            self._process_signal(timestamp, signal.values[0], price)
            
            # Track portfolio value
            portfolio_value = self.cash + self.position * price
            self.portfolio_values.append(portfolio_value)
        
        # Close any remaining position
        if self.position > 0:
            final_price = data.iloc[-1]['close']
            self._execute_trade(data.index[-1], 'sell', final_price, self.position)
        
        # Calculate results
        results = self._calculate_results(data)
        
        logger.info(f"Backtest complete. Total return: {results.total_return:.2%}")
        
        return results
    
    def reset(self):
        """Reset engine state."""
        self.cash = self.initial_capital
        self.position = 0
        self.trades = []
        self.portfolio_values = []
    
    def _process_signal(self, timestamp: datetime, signal: int, price: float):
        """Process a trading signal."""
        if signal == 1 and self.position == 0:  # Buy signal
            quantity = (self.cash * self.position_size) / price
            self._execute_trade(timestamp, 'buy', price, quantity)
            
        elif signal == -1 and self.position > 0:  # Sell signal
            self._execute_trade(timestamp, 'sell', price, self.position)
    
    def _execute_trade(self, timestamp: datetime, trade_type: str, 
                      price: float, quantity: float):
        """Execute a trade with fees and slippage."""
        # Apply slippage
        if trade_type == 'buy':
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)
        
        # Calculate trade value and fees
        trade_value = quantity * execution_price
        fee = trade_value * self.commission
        
        # Update positions
        if trade_type == 'buy':
            self.cash -= (trade_value + fee)
            self.position += quantity
        else:
            self.cash += (trade_value - fee)
            self.position -= quantity
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            type=trade_type,
            price=execution_price,
            quantity=quantity,
            value=trade_value,
            fee=fee,
            position_after=self.position,
            cash_after=self.cash,
            portfolio_value=self.cash + self.position * price
        )
        
        self.trades.append(trade)
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        portfolio_values = np.array(self.portfolio_values)
        
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Total return
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return
        num_years = len(data) / (365 * 24)  # Assuming hourly data
        annualized_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
        
        # Sharpe ratio (assuming 0 risk-free rate)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino_ratio = np.mean(returns) / downside_std * np.sqrt(365 * 24) if downside_std > 0 else 0
        else:
            sortino_ratio = sharpe_ratio
        
        # Maximum drawdown
        cumulative = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Trade statistics
        trade_returns = []
        for i in range(0, len(self.trades) - 1, 2):
            if i + 1 < len(self.trades):
                buy_trade = self.trades[i]
                sell_trade = self.trades[i + 1]
                trade_return = (sell_trade.price - buy_trade.price) / buy_trade.price
                trade_returns.append(trade_return)
        
        if trade_returns:
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]
            
            win_rate = len(winning_trades) / len(trade_returns)
            
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
            profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if losing_trades else float('inf')
            
            avg_trade_return = np.mean(trade_returns)
            best_trade = max(trade_returns)
            worst_trade = min(trade_returns)
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_return = 0
            best_trade = 0
            worst_trade = 0
        
        # Exposure time
        exposure_time = sum(1 for t in self.trades if t.position_after > 0) / len(data) if len(data) > 0 else 0
        
        return BacktestResults(
            trades=self.trades,
            portfolio_values=portfolio_values,
            returns=returns,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(self.trades),
            avg_trade_return=avg_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            exposure_time=exposure_time
        )


# src/strategy/signals.py
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseSignalGenerator(ABC):
    """Base class for signal generators."""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from data."""
        pass


class MLSignalGenerator(BaseSignalGenerator):
    """Generate signals using ML model predictions."""
    
    def __init__(self, model: torch.nn.Module, 
                 preprocessor: Any,
                 config: Dict[str, Any]):
        """
        Initialize ML signal generator.
        
        Args:
            model: Trained PyTorch model
            preprocessor: Data preprocessor
            config: Strategy configuration
        """
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from model predictions.
        
        Args:
            data: Historical price data
            
        Returns:
            DataFrame with signals
        """
        # Preprocess data
        features = self.preprocessor.add_technical_indicators(data)
        feature_columns = self.preprocessor._select_features(features)
        scaled_features = self.preprocessor.scale_data(features[feature_columns], fit=False)
        
        # Create sequences
        sequences, _ = self.preprocessor.create_sequences(scaled_features)
        
        # Generate predictions
        predictions = []
        
        with torch.no_grad():
            for seq in sequences:
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                pred = self.model(seq_tensor).cpu().numpy()
                predictions.append(pred[0])
        
        predictions = np.array(predictions)
        
        # Inverse transform predictions
        predictions = self.preprocessor.inverse_transform_predictions(predictions)
        
        # Generate signals based on predictions
        signals = self._predictions_to_signals(predictions, data, sequences)
        
        return signals
    
    def _predictions_to_signals(self, predictions: np.ndarray, 
                               data: pd.DataFrame,
                               sequences: np.ndarray) -> pd.DataFrame:
        """Convert predictions to trading signals."""
        seq_length = self.preprocessor.sequence_length
        
        # Get current prices (aligned with predictions)
        current_prices = data['close'].values[seq_length:seq_length + len(predictions)]
        
        # Calculate expected returns
        if len(predictions.shape) > 1:
            # Multi-horizon predictions - use average
            pred_prices = np.mean(predictions, axis=1)
        else:
            pred_prices = predictions
        
        expected_returns = (pred_prices - current_prices) / current_prices
        
        # Generate signals based on thresholds
        buy_threshold = self.config.get('buy_threshold', 0.02)  # 2% expected gain
        sell_threshold = self.config.get('sell_threshold', -0.02)  # 2% expected loss
        
        signals = np.where(expected_returns > buy_threshold, 1,
                          np.where(expected_returns < sell_threshold, -1, 0))
        
        # Apply filters
        if self.config.get('volume_filter', True):
            signals = self._apply_volume_filter(signals, data, seq_length)
        
        if self.config.get('trend_filter', True):
            signals = self._apply_trend_filter(signals, data, seq_length)
        
        # Create signals DataFrame
        signal_index = data.index[seq_length:seq_length + len(signals)]
        signals_df = pd.DataFrame(signals, index=signal_index, columns=['signal'])
        
        return signals_df
    
    def _apply_volume_filter(self, signals: np.ndarray, 
                            data: pd.DataFrame,
                            offset: int) -> np.ndarray:
        """Filter signals based on volume."""
        volumes = data['volume'].values[offset:offset + len(signals)]
        avg_volume = np.mean(volumes)
        
        # Only trade when volume is above average
        volume_filter = volumes > avg_volume * 0.8
        signals = signals * volume_filter
        
        return signals
    
    def _apply_trend_filter(self, signals: np.ndarray,
                           data: pd.DataFrame,
                           offset: int) -> np.ndarray:
        """Filter signals based on trend."""
        prices = data['close'].values[offset:offset + len(signals)]
        
        # Calculate short-term trend
        if len(prices) > 20:
            sma_20 = pd.Series(prices).rolling(20).mean().values
            
            # Only buy in uptrend, only sell in downtrend
            uptrend = prices > sma_20
            downtrend = prices < sma_20
            
            # Filter buy signals to uptrends
            signals = np.where((signals == 1) & ~uptrend, 0, signals)
            # Filter sell signals to downtrends  
            signals = np.where((signals == -1) & ~downtrend, 0, signals)
        
        return signals


# src/strategy/portfolio.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages portfolio allocation and position sizing."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize portfolio manager.
        
        Args:
            config: Portfolio configuration
        """
        self.config = config
        self.position_sizing = config.get('position_sizing', 'fixed')
        self.max_positions = config.get('max_positions', 3)
        self.position_size = config.get('position_size', 0.1)
        self.use_kelly = config.get('use_kelly', False)
        
    def calculate_position_size(self, signal_strength: float,
                               win_rate: float,
                               avg_win: float,
                               avg_loss: float,
                               current_capital: float) -> float:
        """
        Calculate position size based on strategy.
        
        Args:
            signal_strength: Strength of the signal (0-1)
            win_rate: Historical win rate
            avg_win: Average winning trade return
            avg_loss: Average losing trade return
            current_capital: Current capital
            
        Returns:
            Position size in currency units
        """
        if self.position_sizing == 'fixed':
            return current_capital * self.position_size
        
        elif self.position_sizing == 'kelly':
            kelly_fraction = self._calculate_kelly_criterion(win_rate, avg_win, avg_loss)
            # Apply Kelly fraction with safety factor
            kelly_fraction = min(kelly_fraction * 0.25, 0.25)  # Never more than 25%
            return current_capital * kelly_fraction
        
        elif self.position_sizing == 'dynamic':
            # Scale position size with signal strength
            base_size = current_capital * self.position_size
            return base_size * signal_strength
        
        else:
            return current_capital * self.position_size
    
    def _calculate_kelly_criterion(self, win_rate: float,
                                  avg_win: float,
                                  avg_loss: float) -> float:
        """Calculate Kelly criterion for position sizing."""
        if avg_loss == 0:
            return 0
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win_rate, q = 1-win_rate, b = avg_win/avg_loss
        b = avg_win / abs(avg_loss)
        q = 1 - win_rate
        
        kelly = (win_rate * b - q) / b
        
        # Ensure non-negative and reasonable bounds
        return max(0, min(kelly, 1))
    
    def rebalance_portfolio(self, current_positions: Dict[str, float],
                          target_weights: Dict[str, float],
                          current_prices: Dict[str, float],
                          capital: float) -> Dict[str, float]:
        """
        Rebalance portfolio to target weights.
        
        Args:
            current_positions: Current position quantities
            target_weights: Target portfolio weights
            current_prices: Current asset prices
            capital: Total capital
            
        Returns:
            Trade orders to execute
        """
        orders = {}
        
        # Calculate current weights
        total_value = capital
        for asset, quantity in current_positions.items():
            total_value += quantity * current_prices[asset]
        
        # Calculate target positions
        for asset, target_weight in target_weights.items():
            target_value = total_value * target_weight
            target_quantity = target_value / current_prices[asset]
            
            current_quantity = current_positions.get(asset, 0)
            order_quantity = target_quantity - current_quantity
            
            # Only rebalance if difference is significant
            if abs(order_quantity * current_prices[asset]) > total_value * 0.01:
                orders[asset] = order_quantity
        
        return orders


# src/strategy/risk.py
class RiskManager:
    """Manages risk controls and position limits."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk manager.
        
        Args:
            config: Risk management configuration
        """
        self.config = config
        self.max_drawdown = config.get('max_drawdown', 0.2)
        self.stop_loss = config.get('stop_loss', 0.05)
        self.take_profit = config.get('take_profit', 0.15)
        self.trailing_stop = config.get('trailing_stop', True)
        self.var_limit = config.get('var_limit', 0.05)
        
        # Track positions for risk management
        self.positions = {}
        self.peak_values = {}
        
    def check_stop_loss(self, position: Dict[str, Any],
                       current_price: float) -> bool:
        """Check if stop loss is triggered."""
        entry_price = position['entry_price']
        
        if position['type'] == 'long':
            loss = (entry_price - current_price) / entry_price
            return loss >= self.stop_loss
        else:  # short position
            loss = (current_price - entry_price) / entry_price
            return loss >= self.stop_loss
    
    def check_take_profit(self, position: Dict[str, Any],
                         current_price: float) -> bool:
        """Check if take profit is triggered."""
        entry_price = position['entry_price']
        
        if position['type'] == 'long':
            profit = (current_price - entry_price) / entry_price
            return profit >= self.take_profit
        else:  # short position
            profit = (entry_price - current_price) / entry_price
            return profit >= self.take_profit
    
    def update_trailing_stop(self, position_id: str,
                            current_price: float) -> Optional[float]:
        """Update trailing stop for a position."""
        if not self.trailing_stop:
            return None
        
        if position_id not in self.peak_values:
            self.peak_values[position_id] = current_price
        
        # Update peak if price has increased (for long positions)
        if current_price > self.peak_values[position_id]:
            self.peak_values[position_id] = current_price
        
        # Calculate trailing stop level
        trailing_stop_level = self.peak_values[position_id] * (1 - self.stop_loss)
        
        return trailing_stop_level
    
    def calculate_var(self, returns: np.ndarray,
                     confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Historical returns
            confidence_level: Confidence level for VaR
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0
        
        # Historical VaR
        var_index = int((1 - confidence_level) * len(returns))
        sorted_returns = np.sort(returns)
        
        if var_index < len(sorted_returns):
            return abs(sorted_returns[var_index])
        else:
            return abs(sorted_returns[0])
    
    def check_risk_limits(self, portfolio_value: float,
                         peak_value: float,
                         returns: np.ndarray) -> Dict[str, bool]:
        """
        Check if any risk limits are breached.
        
        Args:
            portfolio_value: Current portfolio value
            peak_value: Peak portfolio value
            returns: Recent returns
            
        Returns:
            Dictionary of risk checks
        """
        checks = {}
        
        # Check maximum drawdown
        current_drawdown = (peak_value - portfolio_value) / peak_value
        checks['max_drawdown_breached'] = current_drawdown >= self.max_drawdown
        
        # Check VaR limit
        current_var = self.calculate_var(returns)
        checks['var_limit_breached'] = current_var >= self.var_limit
        
        return checks