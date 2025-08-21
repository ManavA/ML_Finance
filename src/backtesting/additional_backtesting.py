# src/backtesting/advanced_backtesting.py
"""
Advanced Backtesting Framework with Statistical Validation
Addresses common pitfalls and ensures robust testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 10000
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.001    # 0.1% slippage
    min_trade_size: float = 10  # Minimum trade size in USD
    max_position_size: float = 1.0  # Maximum position as fraction of capital
    risk_free_rate: float = 0.02  # Annual risk-free rate
    confidence_level: float = 0.95  # For VaR and CVaR
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    use_fractional_shares: bool = True
    short_selling_allowed: bool = True
    margin_requirement: float = 0.5  # For short selling
    

@dataclass
class BacktestResults:
    """Complete backtesting results with all metrics"""
    # Core metrics
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    downside_deviation: float
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    
    # Statistical metrics
    skewness: float
    kurtosis: float
    information_ratio: float
    treynor_ratio: float
    
    # Advanced metrics
    omega_ratio: float
    gain_to_pain_ratio: float
    ulcer_index: float
    serenity_ratio: float
    lake_ratio: float
    burke_ratio: float
    
    # Time series
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    drawdown_series: pd.Series
    rolling_sharpe: pd.Series
    
    # Statistical tests
    normality_test: Dict
    autocorrelation_test: Dict
    stationarity_test: Dict


class AdvancedBacktester:
    """
    Advanced backtesting engine with robust validation
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.results_cache = {}
        
    def backtest(self, 
                data: pd.DataFrame,
                signals: pd.Series,
                benchmark: Optional[pd.Series] = None) -> BacktestResults:
        """
        Run comprehensive backtest with all validations
        
        Args:
            data: OHLCV data
            signals: Trading signals (-1, 0, 1)
            benchmark: Benchmark returns for comparison
            
        Returns:
            Complete backtest results
        """
        
        # Validate inputs
        self._validate_inputs(data, signals)
        
        # Align data and signals
        data, signals = self._align_data(data, signals)
        
        # Calculate positions with proper position sizing
        positions = self._calculate_positions(signals, data)
        
        # Calculate returns accounting for costs
        returns = self._calculate_returns(data, positions)
        
        # Build equity curve
        equity_curve = self._build_equity_curve(returns)
        
        # Calculate all metrics
        metrics = self._calculate_all_metrics(
            returns, equity_curve, positions, benchmark
        )
        
        # Run statistical tests
        statistical_tests = self._run_statistical_tests(returns)
        
        # Create results object
        results = BacktestResults(
            **metrics,
            equity_curve=equity_curve,
            returns=returns,
            positions=positions,
            **statistical_tests
        )
        
        return results
    
    def _validate_inputs(self, data: pd.DataFrame, signals: pd.Series):
        """Validate input data integrity"""
        
        # Check for missing values
        if data.isnull().any().any():
            raise ValueError("Data contains missing values")
        
        if signals.isnull().any():
            raise ValueError("Signals contain missing values")
        
        # Check signal values
        valid_signals = {-1, 0, 1}
        if not set(signals.unique()).issubset(valid_signals):
            raise ValueError(f"Signals must be in {valid_signals}")
        
        # Check for look-ahead bias
        self._check_lookahead_bias(data, signals)
    
    def _check_lookahead_bias(self, data: pd.DataFrame, signals: pd.Series):
        """Check for potential look-ahead bias"""
        
        # Signals should not depend on future data
        # This is a simple check - more sophisticated tests needed for complex strategies
        for i in range(1, len(signals)):
            if signals.index[i] < data.index[i]:
                warnings.warn("Potential look-ahead bias detected")
                break
    
    def _align_data(self, data: pd.DataFrame, signals: pd.Series):
        """Align data and signals to same index"""
        
        common_index = data.index.intersection(signals.index)
        return data.loc[common_index], signals.loc[common_index]
    
    def _calculate_positions(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        Calculate actual positions with risk management
        
        Implements:
        - Position sizing based on volatility
        - Maximum position limits
        - Minimum trade size requirements
        """
        
        positions = signals.copy()
        
        # Calculate rolling volatility for position sizing
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        
        # Volatility-based position sizing (inverse volatility)
        target_vol = 0.02  # 2% daily volatility target
        position_size = target_vol / (volatility + 1e-8)
        position_size = position_size.clip(0, self.config.max_position_size)
        
        # Apply position sizing
        positions = positions * position_size
        
        # Apply minimum trade size filter
        trade_value = positions.abs() * data['close'] * self.config.initial_capital
        positions[trade_value < self.config.min_trade_size] = 0
        
        return positions
    
    def _calculate_returns(self, data: pd.DataFrame, positions: pd.Series) -> pd.Series:
        """
        Calculate returns with realistic transaction costs
        
        Includes:
        - Commissions
        - Slippage
        - Market impact (for large positions)
        - Borrowing costs (for short positions)
        """
        
        # Base returns
        price_returns = data['close'].pct_change()
        
        # Position changes (for transaction costs)
        position_changes = positions.diff().abs()
        
        # Transaction costs
        commission_costs = position_changes * self.config.commission
        slippage_costs = position_changes * self.config.slippage
        
        # Market impact (square root model)
        market_impact = position_changes.pow(0.5) * 0.0001
        
        # Borrowing costs for short positions
        short_costs = (positions < 0).astype(float) * 0.0001  # 1bp per day
        
        # Total costs
        total_costs = commission_costs + slippage_costs + market_impact + short_costs
        
        # Net returns
        strategy_returns = positions.shift(1) * price_returns - total_costs
        
        return strategy_returns
    
    def _build_equity_curve(self, returns: pd.Series) -> pd.Series:
        """Build equity curve from returns"""
        
        equity_curve = (1 + returns).cumprod() * self.config.initial_capital
        equity_curve.iloc[0] = self.config.initial_capital
        
        return equity_curve
    
    def _calculate_all_metrics(self, 
                              returns: pd.Series,
                              equity_curve: pd.Series,
                              positions: pd.Series,
                              benchmark: Optional[pd.Series] = None) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Clean returns
        returns_clean = returns.dropna()
        
        # Annualization factor
        periods_per_year = 252  # Daily data
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annual_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
        volatility = returns_clean.std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio (with proper risk-free rate)
        excess_returns = returns_clean - self.config.risk_free_rate / periods_per_year
        sharpe_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / returns_clean.std() \
                      if returns_clean.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_clean[returns_clean < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(periods_per_year)
        sortino_ratio = (annual_return - self.config.risk_free_rate) / downside_deviation \
                       if downside_deviation > 0 else 0
        
        # Drawdown metrics
        drawdown_series = self._calculate_drawdown_series(equity_curve)
        max_drawdown = drawdown_series.min()
        max_dd_duration = self._calculate_max_drawdown_duration(drawdown_series)
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR and CVaR
        var_95 = returns_clean.quantile(1 - self.config.confidence_level)
        cvar_95 = returns_clean[returns_clean <= var_95].mean()
        
        # Trading metrics
        trades = self._identify_trades(positions)
        trade_returns = self._calculate_trade_returns(trades, returns)
        
        winning_trades = (trade_returns > 0).sum()
        losing_trades = (trade_returns < 0).sum()
        total_trades = len(trade_returns)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = trade_returns[trade_returns > 0].mean() if winning_trades > 0 else 0
        avg_loss = trade_returns[trade_returns < 0].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(trade_returns[trade_returns > 0].sum() / 
                           trade_returns[trade_returns < 0].sum()) \
                       if losing_trades > 0 else float('inf')
        
        expectancy = trade_returns.mean() if total_trades > 0 else 0
        
        # Statistical metrics
        skewness = stats.skew(returns_clean)
        kurtosis = stats.kurtosis(returns_clean)
        
        # Information ratio
        if benchmark is not None:
            tracking_error = (returns_clean - benchmark).std() * np.sqrt(periods_per_year)
            information_ratio = (annual_return - benchmark.mean() * periods_per_year) / tracking_error \
                              if tracking_error > 0 else 0
        else:
            information_ratio = 0
        
        # Treynor ratio (using market beta)
        if benchmark is not None:
            beta = np.cov(returns_clean, benchmark)[0, 1] / np.var(benchmark)
            treynor_ratio = (annual_return - self.config.risk_free_rate) / beta if beta != 0 else 0
        else:
            treynor_ratio = 0
        
        # Advanced metrics
        omega_ratio = self._calculate_omega_ratio(returns_clean)
        gain_to_pain_ratio = self._calculate_gain_to_pain_ratio(returns_clean)
        ulcer_index = self._calculate_ulcer_index(drawdown_series)
        serenity_ratio = self._calculate_serenity_ratio(returns_clean, ulcer_index)
        lake_ratio = self._calculate_lake_ratio(drawdown_series)
        burke_ratio = self._calculate_burke_ratio(returns_clean, drawdown_series)
        
        # Rolling metrics
        rolling_sharpe = self._calculate_rolling_sharpe(returns_clean)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'downside_deviation': downside_deviation,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'omega_ratio': omega_ratio,
            'gain_to_pain_ratio': gain_to_pain_ratio,
            'ulcer_index': ulcer_index,
            'serenity_ratio': serenity_ratio,
            'lake_ratio': lake_ratio,
            'burke_ratio': burke_ratio,
            'drawdown_series': drawdown_series,
            'rolling_sharpe': rolling_sharpe,
        }
    }
    
    return pd.DataFrame(metrics).T
    def _calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown
    
    def _calculate_max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        underwater = drawdown_series < 0
        runs = underwater.ne(underwater.shift()).cumsum()
        run_lengths = underwater.groupby(runs).sum()
        return int(run_lengths.max()) if len(run_lengths) > 0 else 0
    
    def _identify_trades(self, positions: pd.Series) -> List[Dict]:
        """Identify individual trades from position series"""
        trades = []
        position_changes = positions.diff()
        
        for i in range(1, len(positions)):
            if position_changes.iloc[i] != 0:
                trades.append({
                    'entry_date': positions.index[i],
                    'entry_position': positions.iloc[i],
                    'exit_date': None,
                    'exit_position': None
                })
        
        return trades
    
    def _calculate_trade_returns(self, trades: List[Dict], returns: pd.Series) -> pd.Series:
        """Calculate returns for each trade"""
        trade_returns = []
        
        for i in range(len(trades) - 1):
            entry_date = trades[i]['entry_date']
            exit_date = trades[i + 1]['entry_date']
            
            trade_return = returns[entry_date:exit_date].sum()
            trade_returns.append(trade_return)
        
        return pd.Series(trade_returns)
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        excess = returns - threshold
        wins = excess[excess > 0].sum()
        losses = -excess[excess < 0].sum()
        
        return wins / losses if losses != 0 else float('inf')
    
    def _calculate_gain_to_pain_ratio(self, returns: pd.Series) -> float:
        """Calculate Gain-to-Pain ratio"""
        gains = returns[returns > 0].sum()
        pains = abs(returns[returns < 0].sum())
        
        return gains / pains if pains != 0 else float('inf')
    
    def _calculate_ulcer_index(self, drawdown_series: pd.Series) -> float:
        """Calculate Ulcer Index"""
        return np.sqrt(np.mean(drawdown_series**2)) * 100
    
    def _calculate_serenity_ratio(self, returns: pd.Series, ulcer_index: float) -> float:
        """Calculate Serenity ratio"""
        annual_return = returns.mean() * 252
        return annual_return / ulcer_index if ulcer_index != 0 else 0
    
    def _calculate_lake_ratio(self, drawdown_series: pd.Series) -> float:
        """Calculate Lake ratio"""
        underwater_area = abs(drawdown_series[drawdown_series < 0].sum())
        total_periods = len(drawdown_series)
        
        return underwater_area / total_periods if total_periods > 0 else 0
    
    def _calculate_burke_ratio(self, returns: pd.Series, drawdown_series: pd.Series) -> float:
        """Calculate Burke ratio"""
        annual_return = returns.mean() * 252
        burke_deviation = np.sqrt(np.sum(drawdown_series**2))
        
        return annual_return / burke_deviation if burke_deviation != 0 else 0
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        
        rolling_sharpe = np.sqrt(252) * rolling_mean / rolling_std
        return rolling_sharpe
    
    def _run_statistical_tests(self, returns: pd.Series) -> Dict:
        """Run comprehensive statistical tests"""
        
        # Normality test (Jarque-Bera)
        jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
        normality_test = {
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'is_normal': jb_pvalue > 0.05
        }
        
        # Autocorrelation test (Ljung-Box)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(returns.dropna(), lags=10, return_df=True)
        autocorrelation_test = {
            'ljung_box_pvalue': lb_result['lb_pvalue'].iloc[-1],
            'has_autocorrelation': lb_result['lb_pvalue'].iloc[-1] < 0.05
        }
        
        # Stationarity test (ADF)
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(returns.dropna())
        stationarity_test = {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'is_stationary': adf_result[1] < 0.05
        }
        
        return {
            'normality_test': normality_test,
            'autocorrelation_test': autocorrelation_test,
            'stationarity_test': stationarity_test
        }


class WalkForwardValidator:
    """
    Walk-forward analysis for strategy validation
    """
    
    def __init__(self, 
                n_splits: int = 5,
                train_period: int = 252,
                test_period: int = 63):
        """
        Initialize walk-forward validator
        
        Args:
            n_splits: Number of walk-forward windows
            train_period: Training period length (days)
            test_period: Testing period length (days)
        """
        self.n_splits = n_splits
        self.train_period = train_period
        self.test_period = test_period
    
    def validate(self,
                data: pd.DataFrame,
                strategy_func: callable,
                backtester: AdvancedBacktester) -> Dict:
        """
        Run walk-forward validation
        
        Args:
            data: Full dataset
            strategy_func: Function that generates signals given data
            backtester: Backtester instance
            
        Returns:
            Validation results with statistics
        """
        
        results = []
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
            # Split data
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Train strategy on training data
            strategy_params = self._optimize_strategy(train_data, strategy_func)
            
            # Generate signals on test data
            test_signals = strategy_func(test_data, **strategy_params)
            
            # Backtest on test data
            fold_results = backtester.backtest(test_data, test_signals)
            
            results.append({
                'fold': fold,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'sharpe_ratio': fold_results.sharpe_ratio,
                'total_return': fold_results.total_return,
                'max_drawdown': fold_results.max_drawdown,
            })
        
        # Aggregate results
        results_df = pd.DataFrame(results)
        
        return {
            'detailed_results': results_df,
            'mean_sharpe': results_df['sharpe_ratio'].mean(),
            'std_sharpe': results_df['sharpe_ratio'].std(),
            'mean_return': results_df['total_return'].mean(),
            'consistency': (results_df['sharpe_ratio'] > 0).mean(),
            'worst_fold': results_df['sharpe_ratio'].min(),
            'best_fold': results_df['sharpe_ratio'].max(),
        }
    
    def _optimize_strategy(self, data: pd.DataFrame, strategy_func: callable) -> Dict:
        """Optimize strategy parameters on training data"""
        # Simplified - in practice would use proper optimization
        return {}


def calculate_performance_metrics(results: BacktestResults) -> pd.DataFrame:
    """
    Create comprehensive performance report
    """
    
    metrics = {
        'Performance': {
            'Total Return': f"{results.total_return*100:.2f}%",
            'Annual Return': f"{results.annual_return*100:.2f}%",
            'Volatility': f"{results.volatility*100:.2f}%",
            'Sharpe Ratio': f"{results.sharpe_ratio:.3f}",
            'Sortino Ratio': f"{results.sortino_ratio:.3f}",
            'Calmar Ratio': f"{results.calmar_ratio:.3f}",
        },
        'Risk': {
            'Max Drawdown': f"{results.max_drawdown*100:.2f}%",
            'DD Duration': f"{results.max_drawdown_duration} days",
            'VaR (95%)': f"{results.var_95*100:.2f}%",
            'CVaR (95%)': f"{results.cvar_95*100:.2f}%",
            'Downside Dev': f"{results.downside_deviation*100:.2f}%",
            'Ulcer Index': f"{results.ulcer_index:.3f}",
        },
        'Trading': {
            'Total Trades': results.total_trades,
            'Win Rate': f"{results.win_rate*100:.1f}%",
            'Avg Win': f"{results.avg_win*100:.2f}%",
            'Avg Loss': f"{results.avg_loss*100:.2f}%",
            'Profit Factor': f"{results.profit_factor:.2f}",
            'Expectancy': f"{results.expectancy*100:.3f}%",
        },
        'Advanced': {
            'Omega Ratio': f"{results.omega_ratio:.3f}",
            'Gain/Pain': f"{results.gain_to_pain_ratio:.3f}",
            'Serenity': f"{results.serenity_ratio:.3f}",
            'Lake Ratio': f"{results.lake_ratio:.3f}",
            'Burke Ratio': f"{results.burke_ratio:.3f}",
            'Info Ratio': f"{results.information_ratio:.3f}",
        }
    