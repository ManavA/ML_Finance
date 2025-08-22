#!/usr/bin/env python3
"""
Advanced backtesting framework with comprehensive metrics and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class AdvancedMetrics:
    # Basic metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int
    value_at_risk_95: float
    conditional_value_at_risk_95: float
    downside_deviation: float
    
    # Trading metrics
    n_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    best_trade: float
    worst_trade: float
    
    # Statistical metrics
    skewness: float
    kurtosis: float
    information_ratio: float
    treynor_ratio: float
    omega_ratio: float
    
    # Regime-specific metrics
    bull_market_return: float
    bear_market_return: float
    high_vol_return: float
    low_vol_return: float
    
    # Portfolio metrics
    beta: float
    alpha: float
    correlation_to_benchmark: float
    tracking_error: float
    
    # Additional analysis
    monthly_returns: List[float] = field(default_factory=list)
    rolling_sharpe: List[float] = field(default_factory=list)
    underwater_curve: List[float] = field(default_factory=list)


class MarketRegimeDetector:
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        
    def identify_regimes(self, data: pd.DataFrame) -> pd.DataFrame:

        regimes = pd.DataFrame(index=data.index)
        
        # Price-based regimes
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        
        regimes['trend'] = np.where(
            sma_20 > sma_50, 'bull', 'bear'
        )
        
        # Volatility regimes
        returns = data['close'].pct_change()
        vol = returns.rolling(self.lookback_period).std()
        vol_median = vol.median()
        
        regimes['volatility'] = np.where(
            vol > vol_median * 1.5, 'high_vol', 'low_vol'
        )
        
        # Volume regimes
        volume_ma = data['volume'].rolling(20).mean()
        volume_median = volume_ma.median()
        
        regimes['volume'] = np.where(
            volume_ma > volume_median * 1.2, 'high_volume', 'low_volume'
        )
        
        # Market stress indicator
        regimes['stress'] = self._calculate_stress_indicator(data)
        
        return regimes
    
    def _calculate_stress_indicator(self, data: pd.DataFrame) -> pd.Series:
        returns = data['close'].pct_change()
        
        # Components of stress
        volatility = returns.rolling(20).std()
        negative_returns = returns.rolling(20).apply(lambda x: (x < 0).sum() / len(x))
        volume_spike = data['volume'] / data['volume'].rolling(20).mean()
        
        # Normalize and combine
        vol_z = (volatility - volatility.mean()) / volatility.std()
        neg_z = (negative_returns - negative_returns.mean()) / negative_returns.std()
        volume_z = (volume_spike - volume_spike.mean()) / volume_spike.std()
        
        stress = (vol_z + neg_z + volume_z) / 3
        
        return pd.cut(stress, bins=[-np.inf, -0.5, 0.5, np.inf], 
                     labels=['low_stress', 'normal', 'high_stress'])


class PortfolioOptimizer:
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def optimize_weights(
        self, 
        returns: pd.DataFrame, 
        method: str = 'sharpe',
        constraints: Optional[Dict] = None
    ) -> np.ndarray:

        n_assets = len(returns.columns)
        
        if method == 'equal_weight':
            return np.array([1/n_assets] * n_assets)
        
        # Calculate statistics
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        if method == 'sharpe':
            return self._maximize_sharpe(mean_returns, cov_matrix, constraints)
        elif method == 'min_vol':
            return self._minimize_volatility(cov_matrix, constraints)
        elif method == 'risk_parity':
            return self._risk_parity(cov_matrix)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _maximize_sharpe(
        self, 
        mean_returns: pd.Series, 
        cov_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        n = len(mean_returns)
        
        def neg_sharpe(weights):
            portfolio_return = np.sum(mean_returns * weights) * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # Constraints
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(n))
        if constraints:
            if 'max_weight' in constraints:
                bounds = tuple((0, constraints['max_weight']) for _ in range(n))
        
        # Initial guess
        x0 = np.array([1/n] * n)
        
        # Optimize
        result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        return result.x
    
    def _minimize_volatility(
        self, 
        cov_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        n = len(cov_matrix)
        
        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        x0 = np.array([1/n] * n)
        
        result = minimize(portfolio_vol, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        return result.x
    
    def _risk_parity(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        n = len(cov_matrix)
        
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_vol
            
            # Target equal risk contribution
            target = portfolio_vol / n
            return np.sum((contrib - target) ** 2)
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.01, 1) for _ in range(n))
        x0 = np.array([1/n] * n)
        
        result = minimize(risk_contribution, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        return result.x


class AdvancedBacktester:

    def __init__(
        self,
        commission: float = 0.001,
        slippage: float = 0.0005,
        initial_capital: float = 100000,
        risk_free_rate: float = 0.02,
        benchmark: Optional[pd.Series] = None
    ):

        self.commission = commission
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.benchmark = benchmark
        
        self.regime_detector = MarketRegimeDetector()
        self.portfolio_optimizer = PortfolioOptimizer(risk_free_rate)
        
    def backtest(
        self,
        data: pd.DataFrame,
        signals: Union[np.ndarray, pd.Series, pd.DataFrame],
        symbol: str = None,
        analyze_regimes: bool = True,
        calculate_rolling_metrics: bool = True
    ) -> AdvancedMetrics:

        # Prepare data
        prices = data['close'].values
        returns = pd.Series(prices).pct_change().fillna(0)
        
        # Handle different signal types
        if isinstance(signals, pd.DataFrame):
            # Multi-asset signals
            portfolio_returns = self._calculate_portfolio_returns(data, signals)
        else:
            # Single asset signals
            if isinstance(signals, pd.Series):
                signals = signals.values
            portfolio_returns = self._calculate_returns_with_costs(returns, signals)
        
        # Calculate basic metrics
        metrics = self._calculate_basic_metrics(portfolio_returns)
        
        # Calculate advanced metrics
        metrics.update(self._calculate_risk_metrics(portfolio_returns))
        metrics.update(self._calculate_trading_metrics(portfolio_returns, signals))
        metrics.update(self._calculate_statistical_metrics(portfolio_returns))
        
        # Regime analysis
        if analyze_regimes:
            regimes = self.regime_detector.identify_regimes(data)
            regime_metrics = self._analyze_by_regime(portfolio_returns, regimes)
            metrics.update(regime_metrics)
        else:
            metrics.update({
                'bull_market_return': 0,
                'bear_market_return': 0,
                'high_vol_return': 0,
                'low_vol_return': 0
            })
        
        # Benchmark relative metrics
        if self.benchmark is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(portfolio_returns, self.benchmark)
            metrics.update(benchmark_metrics)
        else:
            metrics.update({
                'beta': 0,
                'alpha': 0,
                'correlation_to_benchmark': 0,
                'tracking_error': 0
            })
        
        # Rolling metrics
        if calculate_rolling_metrics:
            metrics['monthly_returns'] = self._calculate_monthly_returns(portfolio_returns)
            metrics['rolling_sharpe'] = self._calculate_rolling_sharpe(portfolio_returns)
            metrics['underwater_curve'] = self._calculate_underwater_curve(portfolio_returns)
        
        return AdvancedMetrics(**metrics)
    
    def _calculate_returns_with_costs(
        self, 
        returns: pd.Series, 
        signals: np.ndarray
    ) -> pd.Series:
        # Align signals with returns
        if len(signals) == len(returns) - 1:
            signals = np.concatenate([[0], signals])
        
        # Calculate positions
        positions = pd.Series(signals)
        
        # Calculate trades (position changes)
        trades = positions.diff().fillna(0)
        
        # Calculate costs
        costs = np.abs(trades) * (self.commission + self.slippage)
        
        # Calculate strategy returns
        strategy_returns = positions.shift(1) * returns - costs
        strategy_returns = strategy_returns.fillna(0)
        
        return strategy_returns
    
    def _calculate_portfolio_returns(
        self, 
        data: pd.DataFrame, 
        weights: pd.DataFrame
    ) -> pd.Series:
        # Implementation for multi-asset portfolios
        portfolio_returns = pd.Series(index=data.index, dtype=float)
        
        # Calculate weighted returns
        for date in weights.index:
            if date in data.index:
                daily_weights = weights.loc[date]
                # Calculate portfolio return for this period
                # This is simplified - real implementation would be more complex
                portfolio_returns.loc[date] = 0  # Placeholder
        
        return portfolio_returns
    
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict:
        total_return = (1 + returns).prod() - 1
        days = len(returns)
        years = days / 252
        
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'total_return': total_return * 100,
            'annualized_return': annualized_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'downside_deviation': downside_deviation * 100
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        cumulative = (1 + returns).cumprod()
        
        # Drawdown analysis
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
        
        # Value at Risk and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Calmar ratio
        calmar_ratio = (returns.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'max_drawdown': max_drawdown * 100,
            'max_drawdown_duration': max_duration,
            'value_at_risk_95': var_95 * 100,
            'conditional_value_at_risk_95': cvar_95 * 100,
            'calmar_ratio': calmar_ratio
        }
    
    def _calculate_trading_metrics(self, returns: pd.Series, signals: np.ndarray) -> Dict:
        # Identify trades
        if len(signals.shape) > 1:
            # Multi-asset - simplified for now
            trades = pd.Series([1] * len(returns))
        else:
            positions = pd.Series(signals)
            trades = positions.diff().fillna(0)
            trades = trades[trades != 0]
        
        n_trades = len(trades)
        
        # Calculate trade returns
        trade_returns = []
        current_position = 0
        entry_price = 0
        
        for i, signal in enumerate(signals[:-1]):  # Skip last signal
            if signal != current_position:
                if current_position != 0:
                    # Exit trade
                    exit_return = returns.iloc[i]
                    trade_returns.append(exit_return)
                current_position = signal
        
        if len(trade_returns) == 0:
            trade_returns = [0]
        
        trade_returns = pd.Series(trade_returns)
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]
        
        win_rate = len(winning_trades) / len(trade_returns) if len(trade_returns) > 0 else 0
        
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() != 0 else 0
        
        return {
            'n_trades': n_trades,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'average_win': avg_win * 100,
            'average_loss': avg_loss * 100,
            'best_trade': trade_returns.max() * 100 if len(trade_returns) > 0 else 0,
            'worst_trade': trade_returns.min() * 100 if len(trade_returns) > 0 else 0
        }
    
    def _calculate_statistical_metrics(self, returns: pd.Series) -> Dict:
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Information ratio (if benchmark available)
        if self.benchmark is not None:
            active_returns = returns - self.benchmark
            information_ratio = (active_returns.mean() * 252) / (active_returns.std() * np.sqrt(252))
        else:
            information_ratio = 0
        
        # Treynor ratio (requires beta)
        treynor_ratio = 0  # Calculated with benchmark metrics
        
        # Omega ratio
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        omega_ratio = gains / losses if losses != 0 else 0
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'omega_ratio': omega_ratio
        }
    
    def _analyze_by_regime(self, returns: pd.Series, regimes: pd.DataFrame) -> Dict:
        regime_returns = {}
        
        # Trend regimes
        if 'trend' in regimes.columns:
            bull_returns = returns[regimes['trend'] == 'bull']
            bear_returns = returns[regimes['trend'] == 'bear']
            
            regime_returns['bull_market_return'] = bull_returns.mean() * 252 * 100 if len(bull_returns) > 0 else 0
            regime_returns['bear_market_return'] = bear_returns.mean() * 252 * 100 if len(bear_returns) > 0 else 0
        
        # Volatility regimes
        if 'volatility' in regimes.columns:
            high_vol_returns = returns[regimes['volatility'] == 'high_vol']
            low_vol_returns = returns[regimes['volatility'] == 'low_vol']
            
            regime_returns['high_vol_return'] = high_vol_returns.mean() * 252 * 100 if len(high_vol_returns) > 0 else 0
            regime_returns['low_vol_return'] = low_vol_returns.mean() * 252 * 100 if len(low_vol_returns) > 0 else 0
        
        return regime_returns
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, benchmark: pd.Series) -> Dict:
        # Align indices
        aligned = pd.DataFrame({'returns': returns, 'benchmark': benchmark}).dropna()
        
        if len(aligned) == 0:
            return {
                'beta': 0,
                'alpha': 0,
                'correlation_to_benchmark': 0,
                'tracking_error': 0
            }
        
        # Beta calculation
        covariance = aligned.cov()
        beta = covariance.loc['returns', 'benchmark'] / aligned['benchmark'].var()
        
        # Alpha calculation
        strategy_return = aligned['returns'].mean() * 252
        benchmark_return = aligned['benchmark'].mean() * 252
        alpha = strategy_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        # Correlation
        correlation = aligned.corr().loc['returns', 'benchmark']
        
        # Tracking error
        tracking_error = (aligned['returns'] - aligned['benchmark']).std() * np.sqrt(252)
        
        return {
            'beta': beta,
            'alpha': alpha * 100,
            'correlation_to_benchmark': correlation,
            'tracking_error': tracking_error * 100
        }
    
    def _calculate_monthly_returns(self, returns: pd.Series) -> List[float]:
        if not isinstance(returns.index, pd.DatetimeIndex):
            # Create datetime index if not present
            returns.index = pd.date_range(end=datetime.now(), periods=len(returns), freq='D')
        
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        return (monthly * 100).tolist()
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 60) -> List[float]:
        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mean - self.risk_free_rate) / rolling_std
        
        return rolling_sharpe.fillna(0).tolist()
    
    def _calculate_underwater_curve(self, returns: pd.Series) -> List[float]:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        underwater = ((cumulative - running_max) / running_max * 100)
        
        return underwater.tolist()
    
    def compare_strategies(
        self,
        data: pd.DataFrame,
        strategies: Dict[str, np.ndarray],
        create_report: bool = True
    ) -> pd.DataFrame:

        results = []
        
        for name, signals in strategies.items():
            metrics = self.backtest(data, signals, symbol=name)
            
            results.append({
                'Strategy': name,
                'Total Return': metrics.total_return,
                'Sharpe Ratio': metrics.sharpe_ratio,
                'Max Drawdown': metrics.max_drawdown,
                'Win Rate': metrics.win_rate,
                'Number of Trades': metrics.n_trades,
                'Calmar Ratio': metrics.calmar_ratio
            })
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('Sharpe Ratio', ascending=False)
        
        if create_report:
            self._create_comparison_report(comparison_df)
        
        return comparison_df
    
    def _create_comparison_report(self, comparison_df: pd.DataFrame):
        print("\n" + "="*60)
        print("STRATEGY COMPARISON REPORT")
        print("="*60)
        print(comparison_df.to_string())
        print("\n" + "="*60)
        
        # Best strategy by metric
        print("\nBest Strategies by Metric:")
        for col in comparison_df.columns[1:]:
            best_idx = comparison_df[col].idxmax() if 'Drawdown' not in col else comparison_df[col].idxmin()
            best_strategy = comparison_df.loc[best_idx, 'Strategy']
            best_value = comparison_df.loc[best_idx, col]
            print(f"  {col}: {best_strategy} ({best_value:.2f})")


def main():
    print("Advanced Backtester Module Loaded")
    
if __name__ == "__main__":
    main()