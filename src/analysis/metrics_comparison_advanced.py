# src/analysis/metrics_comparison.py
"""
Comprehensive Metrics Framework for Model Comparison
Answers: "Are we looking at the right metrics?"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import scipy.stats as stats
from sklearn.metrics import matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# METRICS CATEGORIES
# ============================================================================

@dataclass
class MetricCategory:
    """Categories of metrics for different evaluation purposes"""
    
    # Return-based metrics (most common but can be misleading)
    RETURN_METRICS = [
        'total_return',
        'annual_return', 
        'cumulative_return',
        'compound_annual_growth_rate',
        'time_weighted_return',
        'money_weighted_return'
    ]
    
    # Risk-adjusted metrics (better for comparison)
    RISK_ADJUSTED_METRICS = [
        'sharpe_ratio',          # Return per unit of total risk
        'sortino_ratio',          # Return per unit of downside risk
        'calmar_ratio',           # Return per unit of max drawdown
        'sterling_ratio',         # Similar to Calmar with average DD
        'burke_ratio',            # Uses drawdown deviation
        'omega_ratio',            # Probability-weighted ratio
        'kappa_three_ratio',      # Higher moment consideration
        'gain_to_pain_ratio',     # Upside vs downside
        'information_ratio',      # Active return per tracking error
        'treynor_ratio',          # Return per unit of systematic risk
        'modified_sharpe_ratio',  # Adjusted for non-normal distributions
        'adjusted_sharpe_ratio',  # Corrected for serial correlation
        'probabilistic_sharpe',   # Probability of positive Sharpe
        'deflated_sharpe',        # Adjusted for multiple testing
    ]
    
    # Drawdown metrics (critical for crypto's volatility)
    DRAWDOWN_METRICS = [
        'max_drawdown',
        'average_drawdown',
        'max_drawdown_duration',
        'recovery_time',
        'underwater_time',
        'ulcer_index',            # RMS of drawdowns
        'pain_index',             # Average drawdown
        'martin_ratio',           # Uses Ulcer index
        'pain_ratio',             # Return over pain index
        'lake_ratio',             # Underwater area
        'burke_ratio',            # Drawdown deviation based
    ]
    
    # Risk metrics (for position sizing)
    RISK_METRICS = [
        'volatility',
        'downside_deviation',
        'semi_variance',
        'value_at_risk_95',
        'conditional_value_at_risk_95',
        'expected_shortfall',
        'maximum_loss',
        'tail_ratio',
        'var_breach_frequency',
        'risk_of_ruin',
        'kelly_fraction',
    ]
    
    # Statistical metrics (for robustness)
    STATISTICAL_METRICS = [
        'skewness',
        'kurtosis',
        'jarque_bera_stat',
        'anderson_darling_stat',
        'hurst_exponent',
        'correlation_with_market',
        'beta',
        'alpha',
        'r_squared',
        'tracking_error',
        'active_share',
    ]
    
    # Trading efficiency metrics (for execution quality)
    TRADING_METRICS = [
        'win_rate',
        'profit_factor',
        'expectancy',
        'payoff_ratio',
        'average_win',
        'average_loss',
        'largest_win',
        'largest_loss',
        'consecutive_wins',
        'consecutive_losses',
        'recovery_factor',
        'system_quality_number',
        'common_sense_ratio',
        'pessimistic_return_ratio',
        'profit_and_loss_ratio',
    ]
    
    # Machine Learning specific metrics
    ML_METRICS = [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'matthews_correlation',
        'directional_accuracy',
        'hit_rate',
        'false_discovery_rate',
        'mean_absolute_error',
        'root_mean_squared_error',
        'mean_absolute_percentage_error',
        'symmetric_mape',
        'log_loss',
        'brier_score',
    ]
    
    # Stability metrics (for consistency)
    STABILITY_METRICS = [
        'rolling_sharpe_stability',
        'rolling_return_stability',
        'parameter_stability',
        'regime_consistency',
        'cross_validation_score',
        'out_of_sample_decay',
        'forward_testing_correlation',
        'backtest_overfitting_score',
    ]


class ComprehensiveMetrics:
    """
    Calculate all relevant metrics for strategy comparison
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = 252  # Daily data
        
    def calculate_all_metrics(self,
                             returns: pd.Series,
                             predictions: Optional[np.ndarray] = None,
                             actual: Optional[np.ndarray] = None,
                             benchmark: Optional[pd.Series] = None) -> Dict:
        """
        Calculate comprehensive metrics suite
        
        Args:
            returns: Strategy returns
            predictions: Model predictions (for ML metrics)
            actual: Actual values (for ML metrics)
            benchmark: Benchmark returns for comparison
            
        Returns:
            Dictionary of all calculated metrics
        """
        
        metrics = {}
        
        # Return metrics
        metrics.update(self._calculate_return_metrics(returns))
        
        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns))
        
        # Drawdown metrics
        metrics.update(self._calculate_drawdown_metrics(returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))
        
        # Statistical metrics
        metrics.update(self._calculate_statistical_metrics(returns, benchmark))
        
        # Trading metrics
        metrics.update(self._calculate_trading_metrics(returns))
        
        # ML metrics (if predictions provided)
        if predictions is not None and actual is not None:
            metrics.update(self._calculate_ml_metrics(predictions, actual))
        
        # Stability metrics
        metrics.update(self._calculate_stability_metrics(returns))
        
        # Crypto-specific metrics
        metrics.update(self._calculate_crypto_specific_metrics(returns))
        
        return metrics
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict:
        """Calculate return-based metrics"""
        
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        annual_factor = self.annualization_factor / n_periods
        
        return {
            'total_return': total_return,
            'annual_return': (1 + total_return) ** annual_factor - 1,
            'cumulative_return': (1 + returns).cumprod().iloc[-1] - 1,
            'compound_annual_growth_rate': ((1 + total_return) ** annual_factor - 1),
            'average_daily_return': returns.mean(),
            'median_daily_return': returns.median(),
            'geometric_mean_return': stats.gmean(1 + returns[returns > -1]) - 1,
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk-adjusted performance metrics"""
        
        # Basic statistics
        mean_return = returns.mean()
        std_return = returns.std()
        downside_returns = returns[returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0
        
        # Sharpe ratio and variants
        sharpe = np.sqrt(self.annualization_factor) * mean_return / std_return if std_return > 0 else 0
        
        # Sortino ratio
        sortino = np.sqrt(self.annualization_factor) * mean_return / downside_std if downside_std > 0 else 0
        
        # Calmar ratio (using max drawdown)
        equity_curve = (1 + returns).cumprod()
        drawdowns = (equity_curve / equity_curve.cummax() - 1)
        max_dd = drawdowns.min()
        calmar = (mean_return * self.annualization_factor) / abs(max_dd) if max_dd != 0 else 0
        
        # Omega ratio
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        omega = gains / losses if losses > 0 else float('inf')
        
        # Gain to Pain ratio
        sum_gains = returns[returns > 0].sum()
        sum_losses = abs(returns[returns < 0].sum())
        gain_to_pain = sum_gains / sum_losses if sum_losses > 0 else float('inf')
        
        # Modified Sharpe (adjusted for skewness and kurtosis)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        modified_sharpe = sharpe * (1 + (skew/6) * sharpe - ((kurt-3)/24) * sharpe**2)
        
        # Probabilistic Sharpe Ratio (PSR)
        # Probability that Sharpe is truly positive
        if len(returns) > 2:
            sr_std = np.sqrt((1 + 0.5 * sharpe**2) / (len(returns) - 1))
            psr = stats.norm.cdf(sharpe / sr_std)
        else:
            psr = 0.5
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'omega_ratio': omega,
            'gain_to_pain_ratio': gain_to_pain,
            'modified_sharpe_ratio': modified_sharpe,
            'probabilistic_sharpe_ratio': psr,
            'sterling_ratio': (mean_return * self.annualization_factor) / abs(drawdowns.mean()) if drawdowns.mean() != 0 else 0,
            'kappa_three_ratio': mean_return / (np.mean(np.abs(returns - threshold)**3)**(1/3)) if returns.std() > 0 else 0,
        }
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict:
        """Calculate drawdown-related metrics"""
        
        equity_curve = (1 + returns).cumprod()
        peak = equity_curve.cummax()
        drawdowns = (equity_curve - peak) / peak
        
        # Underwater periods
        underwater = drawdowns < 0
        underwater_periods = underwater.astype(int).groupby((~underwater).cumsum()).sum()
        
        return {
            'max_drawdown': drawdowns.min(),
            'average_drawdown': drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0,
            'max_drawdown_duration': underwater_periods.max() if len(underwater_periods) > 0 else 0,
            'recovery_time': self._calculate_recovery_time(drawdowns),
            'underwater_time': underwater.mean(),  # Percentage of time underwater
            'ulcer_index': np.sqrt(np.mean(drawdowns**2)) * 100,
            'pain_index': abs(drawdowns[drawdowns < 0].mean()) if (drawdowns < 0).any() else 0,
            'lake_ratio': abs(drawdowns.sum()) / len(drawdowns),
            'burke_ratio': (returns.mean() * self.annualization_factor) / np.sqrt(np.sum(drawdowns**2)),
        }
    
    def _calculate_recovery_time(self, drawdowns: pd.Series) -> float:
        """Calculate average recovery time from drawdowns"""
        
        # Find drawdown periods
        in_drawdown = drawdowns < 0
        drawdown_starts = (~in_drawdown).shift(1) & in_drawdown
        drawdown_ends = in_drawdown.shift(1) & (~in_drawdown)
        
        recovery_times = []
        start_indices = drawdown_starts[drawdown_starts].index
        end_indices = drawdown_ends[drawdown_ends].index
        
        for start, end in zip(start_indices, end_indices[:len(start_indices)]):
            recovery_times.append((end - start).days if hasattr(end - start, 'days') else end - start)
        
        return np.mean(recovery_times) if recovery_times else 0
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk metrics"""
        
        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Tail ratio
        right_tail = returns.quantile(0.95)
        left_tail = abs(returns.quantile(0.05))
        tail_ratio = right_tail / left_tail if left_tail > 0 else float('inf')
        
        # Kelly Criterion
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0
        
        return {
            'volatility': returns.std() * np.sqrt(self.annualization_factor),
            'downside_deviation': np.sqrt(np.mean(np.minimum(returns, 0)**2)) * np.sqrt(self.annualization_factor),
            'semi_variance': returns[returns < returns.mean()].var() if len(returns[returns < returns.mean()]) > 1 else 0,
            'value_at_risk_95': var_95,
            'conditional_value_at_risk_95': cvar_95,
            'expected_shortfall': cvar_95,  # Same as CVaR
            'maximum_loss': returns.min(),
            'tail_ratio': tail_ratio,
            'kelly_fraction': max(0, min(1, kelly)),  # Capped Kelly
            'risk_of_ruin': self._calculate_risk_of_ruin(returns),
        }
    
    def _calculate_risk_of_ruin(self, returns: pd.Series, ruin_threshold: float = -0.5) -> float:
        """Calculate probability of ruin (losing 50% of capital)"""
        
        # Simplified calculation using normal approximation
        mean = returns.mean()
        std = returns.std()
        
        if std == 0:
            return 0 if mean >= 0 else 1
        
        # Probability of reaching ruin threshold
        z_score = (ruin_threshold - mean * len(returns)) / (std * np.sqrt(len(returns)))
        prob_ruin = stats.norm.cdf(z_score)
        
        return prob_ruin
    
    def _calculate_statistical_metrics(self, returns: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict:
        """Calculate statistical metrics"""
        
        metrics = {
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'jarque_bera_stat': stats.jarque_bera(returns)[0],
            'jarque_bera_pvalue': stats.jarque_bera(returns)[1],
            'hurst_exponent': self._calculate_hurst_exponent(returns),
        }
        
        if benchmark is not None and len(benchmark) == len(returns):
            # Calculate beta and alpha
            covariance = np.cov(returns, benchmark)[0, 1]
            benchmark_variance = benchmark.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            alpha = returns.mean() - beta * benchmark.mean()
            
            # R-squared
            correlation = returns.corr(benchmark)
            r_squared = correlation ** 2
            
            # Tracking error
            tracking_error = (returns - benchmark).std() * np.sqrt(self.annualization_factor)
            
            # Information ratio
            active_return = (returns.mean() - benchmark.mean()) * self.annualization_factor
            information_ratio = active_return / tracking_error if tracking_error > 0 else 0
            
            metrics.update({
                'beta': beta,
                'alpha': alpha * self.annualization_factor,
                'r_squared': r_squared,
                'correlation_with_market': correlation,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'active_share': np.abs(returns - benchmark).mean(),
            })
        
        return metrics
    
    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent for time series memory"""
        
        # Simplified R/S analysis
        lags = range(2, min(100, len(returns) // 2))
        tau = []
        
        for lag in lags:
            # Calculate R/S for this lag
            chunks = [returns.iloc[i:i+lag] for i in range(0, len(returns), lag)]
            rs_values = []
            
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                    
                mean = chunk.mean()
                std = chunk.std()
                
                if std == 0:
                    continue
                    
                cumsum = (chunk - mean).cumsum()
                R = cumsum.max() - cumsum.min()
                S = std
                
                rs_values.append(R / S if S > 0 else 0)
            
            if rs_values:
                tau.append(np.mean(rs_values))
        
        if len(tau) > 2:
            # Fit log(R/S) = H * log(n) + c
            log_lags = np.log(list(lags[:len(tau)]))
            log_tau = np.log(tau)
            
            # Remove any inf or nan values
            mask = np.isfinite(log_tau)
            if mask.sum() > 2:
                H = np.polyfit(log_lags[mask], log_tau[mask], 1)[0]
                return H
        
        return 0.5  # Random walk
    
    def _calculate_trading_metrics(self, returns: pd.Series) -> Dict:
        """Calculate trading efficiency metrics"""
        
        # Identify trades (non-zero returns)
        trades = returns[returns != 0]
        winning_trades = trades[trades > 0]
        losing_trades = trades[trades < 0]
        
        # Win/loss statistics
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        
        # Consecutive wins/losses
        is_win = returns > 0
        consecutive_wins = self._max_consecutive(is_win, True)
        consecutive_losses = self._max_consecutive(is_win, False)
        
        # System Quality Number (SQN)
        if len(trades) > 1:
            sqn = np.sqrt(len(trades)) * trades.mean() / trades.std()
        else:
            sqn = 0
        
        # Recovery factor
        equity_curve = (1 + returns).cumprod()
        max_drawdown = ((equity_curve / equity_curve.cummax()) - 1).min()
        recovery_factor = (equity_curve.iloc[-1] - 1) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'payoff_ratio': avg_win / avg_loss if avg_loss > 0 else float('inf'),
            'average_win': avg_win,
            'average_loss': avg_loss,
            'largest_win': winning_trades.max() if len(winning_trades) > 0 else 0,
            'largest_loss': abs(losing_trades.min()) if len(losing_trades) > 0 else 0,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'recovery_factor': recovery_factor,
            'system_quality_number': sqn,
            'common_sense_ratio': (profit_factor * win_rate) / (1 + abs(max_drawdown) * 10),
        }
    
    def _max_consecutive(self, series: pd.Series, value: bool) -> int:
        """Find maximum consecutive occurrences of value"""
        
        groups = (series != value).cumsum()[series == value]
        if len(groups) == 0:
            return 0
        return groups.value_counts().max()
    
    def _calculate_ml_metrics(self, predictions: np.ndarray, actual: np.ndarray) -> Dict:
        """Calculate machine learning specific metrics"""
        
        # Convert to binary classification if needed
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actual)
        
        # Basic classification metrics
        tp = ((pred_direction == 1) & (actual_direction == 1)).sum()
        tn = ((pred_direction == -1) & (actual_direction == -1)).sum()
        fp = ((pred_direction == 1) & (actual_direction == -1)).sum()
        fn = ((pred_direction == -1) & (actual_direction == 1)).sum()
        
        total = tp + tn + fp + fn
        
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(actual_direction, pred_direction) if len(np.unique(actual_direction)) > 1 else 0
        
        # Regression metrics
        mae = np.mean(np.abs(predictions - actual))
        rmse = np.sqrt(np.mean((predictions - actual)**2))
        mape = np.mean(np.abs((actual - predictions) / (actual + 1e-10))) * 100
        
        # Directional accuracy
        directional_accuracy = ((np.sign(predictions) == np.sign(actual)).mean())
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'matthews_correlation': mcc,
            'directional_accuracy': directional_accuracy,
            'hit_rate': directional_accuracy,  # Same as directional accuracy
            'false_discovery_rate': fp / (tp + fp) if (tp + fp) > 0 else 0,
            'mean_absolute_error': mae,
            'root_mean_squared_error': rmse,
            'mean_absolute_percentage_error': mape,
        }
    
    def _calculate_stability_metrics(self, returns: pd.Series) -> Dict:
        """Calculate stability and consistency metrics"""
        
        # Rolling window metrics
        window = min(252, len(returns) // 4)
        
        if window > 20:
            rolling_sharpe = returns.rolling(window).apply(
                lambda x: np.sqrt(self.annualization_factor) * x.mean() / x.std() if x.std() > 0 else 0
            )
            
            rolling_returns = returns.rolling(window).mean()
            
            stability_metrics = {
                'rolling_sharpe_stability': 1 / (rolling_sharpe.std() + 1e-10),
                'rolling_return_stability': 1 / (rolling_returns.std() + 1e-10),
                'return_consistency': (returns > 0).rolling(window).mean().mean(),
                'sharpe_consistency': (rolling_sharpe > 0).mean() if len(rolling_sharpe) > 0 else 0,
            }
        else:
            stability_metrics = {
                'rolling_sharpe_stability': 0,
                'rolling_return_stability': 0,
                'return_consistency': (returns > 0).mean(),
                'sharpe_consistency': 0,
            }
        
        # Regime analysis
        volatility_regimes = pd.qcut(returns.rolling(20).std(), q=3, labels=['low', 'medium', 'high'])
        regime_returns = returns.groupby(volatility_regimes).mean()
        regime_consistency = 1 / (regime_returns.std() + 1e-10) if len(regime_returns) > 1 else 0
        
        stability_metrics['regime_consistency'] = regime_consistency
        
        return stability_metrics
    
    def _calculate_crypto_specific_metrics(self, returns: pd.Series) -> Dict:
        """Calculate metrics specific to cryptocurrency trading"""
        
        # 24/7 trading adjustments
        hourly_factor = 24 * 365  # If using hourly data
        
        # Extreme event frequency
        extreme_threshold = returns.std() * 3
        extreme_events = (np.abs(returns) > extreme_threshold).sum()
        extreme_frequency = extreme_events / len(returns)
        
        # Weekend effect (if timestamps available)
        # This is simplified - would need actual timestamps
        weekend_returns = returns.iloc[::7]  # Simplified assumption
        weekday_returns = returns.drop(weekend_returns.index)
        
        weekend_effect = weekend_returns.mean() - weekday_returns.mean() if len(weekend_returns) > 0 else 0
        
        # Volatility regime persistence
        high_vol_threshold = returns.rolling(20).std().quantile(0.75)
        high_vol_periods = returns.rolling(20).std() > high_vol_threshold
        vol_persistence = high_vol_periods.rolling(10).mean().max() if len(returns) > 30 else 0
        
        return {
            'extreme_event_frequency': extreme_frequency,
            'weekend_effect': weekend_effect,
            'volatility_persistence': vol_persistence,
            'crypto_sharpe_24_7': np.sqrt(hourly_factor) * returns.mean() / returns.std() if returns.std() > 0 else 0,
            'flash_crash_resistance': 1 / (extreme_frequency + 0.01),  # Inverse of extreme frequency
        }


class MetricsValidator:
    """
    Validate and test metrics for significance
    """
    
    def __init__(self):
        self.min_samples = 30  # Minimum samples for statistical tests
        
    def validate_metrics(self, 
                        strategy_metrics: Dict,
                        baseline_metrics: Dict,
                        n_samples: int) -> Dict:
        """
        Validate if strategy metrics are statistically better than baseline
        
        Args:
            strategy_metrics: Metrics from strategy
            baseline_metrics: Metrics from baseline (e.g., buy-and-hold)
            n_samples: Number of samples used
            
        Returns:
            Validation results with statistical tests
        """
        
        validation_results = {}
        
        # Check if we have enough samples
        if n_samples < self.min_samples:
            warnings.warn(f"Only {n_samples} samples - results may not be statistically significant")
        
        # Compare key metrics
        key_metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'omega_ratio']
        
        for metric in key_metrics:
            if metric in strategy_metrics and metric in baseline_metrics:
                improvement = strategy_metrics[metric] - baseline_metrics[metric]
                
                # Calculate significance (simplified - would need bootstrap in practice)
                is_significant = self._test_significance(
                    strategy_metrics[metric],
                    baseline_metrics[metric],
                    n_samples
                )
                
                validation_results[f'{metric}_improvement'] = improvement
                validation_results[f'{metric}_significant'] = is_significant
        
        # Overall recommendation
        significant_improvements = sum(
            1 for k, v in validation_results.items() 
            if k.endswith('_significant') and v
        )
        
        validation_results['recommendation'] = self._get_recommendation(
            significant_improvements,
            len(key_metrics)
        )
        
        return validation_results
    
    def _test_significance(self, 
                          strategy_value: float,
                          baseline_value: float,
                          n_samples: int) -> bool:
        """
        Test if difference is statistically significant
        
        Simplified test - in practice would use bootstrap or more sophisticated methods
        """
        
        # Calculate standard error (simplified)
        se = np.sqrt(2 / n_samples)  # Simplified assumption
        
        # Z-score
        z_score = (strategy_value - baseline_value) / se if se > 0 else 0
        
        # Two-tailed test at 95% confidence
        return abs(z_score) > 1.96
    
    def _get_recommendation(self, significant_improvements: int, total_metrics: int) -> str:
        """Get recommendation based on validation results"""
        
        ratio = significant_improvements / total_metrics
        
        if ratio >= 0.75:
            return "STRONG BUY - Strategy significantly outperforms"
        elif ratio >= 0.5:
            return "BUY - Strategy shows meaningful improvement"
        elif ratio >= 0.25:
            return "HOLD - Mixed results, further testing needed"
        else:
            return "AVOID - No significant improvement over baseline"


def select_optimal_metrics(strategy_type: str, market_type: str = 'crypto') -> List[str]:
    """
    Select the most relevant metrics for evaluation based on strategy and market
    
    Args:
        strategy_type: Type of strategy ('ml', 'momentum', 'mean_reversion', 'arbitrage')
        market_type: Market type ('crypto', 'equity', 'forex')
        
    Returns:
        List of recommended metrics to focus on
    """
    
    # Base metrics for all strategies
    base_metrics = [
        'sharpe_ratio',
        'max_drawdown',
        'total_return',
        'win_rate'
    ]
    
    # Strategy-specific metrics
    if strategy_type == 'ml':
        specific_metrics = [
            'directional_accuracy',
            'information_ratio',
            'stability_metrics',
            'out_of_sample_decay',
            'matthews_correlation',
            'probabilistic_sharpe_ratio',
            'deflated_sharpe',  # Accounts for multiple testing
        ]
    elif strategy_type == 'momentum':
        specific_metrics = [
            'profit_factor',
            'consecutive_wins',
            'payoff_ratio',
            'recovery_factor',
        ]
    elif strategy_type == 'mean_reversion':
        specific_metrics = [
            'sortino_ratio',
            'omega_ratio',
            'win_rate',
            'average_win',
        ]
    elif strategy_type == 'arbitrage':
        specific_metrics = [
            'sharpe_ratio',  # Should be very high
            'win_rate',      # Should be very high
            'max_drawdown',  # Should be very low
            'tail_ratio',
        ]
    else:
        specific_metrics = []
    
    # Market-specific adjustments
    if market_type == 'crypto':
        market_metrics = [
            'calmar_ratio',  # Important due to high drawdowns
            'ulcer_index',   # Measures pain of holding
            'extreme_event_frequency',
            'crypto_sharpe_24_7',
            'regime_consistency',
        ]
    elif market_type == 'equity':
        market_metrics = [
            'information_ratio',
            'treynor_ratio',
            'beta',
            'tracking_error',
        ]
    else:
        market_metrics = []
    
    return base_metrics + specific_metrics + market_metrics


class AdvancedMetrics(ComprehensiveMetrics):
    """
    Extended metrics class with additional advanced calculations
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        super().__init__(risk_free_rate)
        self.crypto_annualization = 365 * 24  # For hourly crypto data
        
    def calculate_deflated_sharpe_ratio(self, 
                                       sharpe: float, 
                                       n_samples: int,
                                       n_strategies_tested: int = 1) -> float:
        """
        Calculate Deflated Sharpe Ratio (DSR) accounting for multiple testing
        Bailey & Lopez de Prado (2014)
        """
        
        # Estimate probability of observing this Sharpe by chance
        if n_samples < 2:
            return 0
            
        # Standard error of Sharpe ratio
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / (n_samples - 1))
        
        # Account for multiple testing
        import scipy.special as sp
        euler_mascheroni = 0.5772156649
        adjustment = (1 - euler_mascheroni) * stats.norm.ppf(1 - 1/n_strategies_tested) + \
                     euler_mascheroni * stats.norm.ppf(1 - 1/(n_strategies_tested * np.e))
        
        # Deflated Sharpe Ratio
        dsr = stats.norm.cdf((sharpe - adjustment * se_sharpe) / se_sharpe)
        
        return dsr
    
    def calculate_probabilistic_sharpe_ratio(self,
                                            observed_sharpe: float,
                                            benchmark_sharpe: float,
                                            n_samples: int) -> float:
        """
        Probability that strategy Sharpe exceeds benchmark Sharpe
        Marcos Lopez de Prado (2018)
        """
        
        if n_samples < 2:
            return 0.5
            
        # Standard deviation of Sharpe ratio difference
        sharpe_diff = observed_sharpe - benchmark_sharpe
        
        # Approximate standard error
        se = np.sqrt((1 + 0.5 * observed_sharpe**2) / (n_samples - 1) + 
                     (1 + 0.5 * benchmark_sharpe**2) / (n_samples - 1))
        
        # Probability of outperformance
        psr = stats.norm.cdf(sharpe_diff / se) if se > 0 else 0.5
        
        return psr
    
    def calculate_t_statistic_sharpe(self, 
                                    sharpe: float,
                                    n_samples: int) -> Tuple[float, float]:
        """
        Calculate t-statistic and p-value for Sharpe ratio
        """
        
        if n_samples < 2:
            return 0, 1.0
            
        # T-statistic for Sharpe ratio
        t_stat = sharpe * np.sqrt(n_samples - 1) / np.sqrt(1 + 0.5 * sharpe**2)
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_samples-1))
        
        return t_stat, p_value
    
    def calculate_minimum_track_record_length(self,
                                             observed_sharpe: float,
                                             target_sharpe: float = 1.0,
                                             confidence: float = 0.95) -> int:
        """
        Calculate minimum track record length needed for statistical significance
        """
        
        if observed_sharpe <= 0:
            return float('inf')
            
        # Z-score for confidence level
        z_score = stats.norm.ppf(confidence)
        
        # Minimum sample size
        min_n = 1 + (1 + 0.5 * observed_sharpe**2) * (z_score / (observed_sharpe - target_sharpe))**2
        
        return max(2, int(np.ceil(min_n)))
    
    def calculate_break_even_probability(self, returns: pd.Series) -> float:
        """
        Probability of breaking even (positive return) using empirical distribution
        """
        
        # Fit distribution to returns
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 1.0 if mean_return > 0 else 0.0
            
        # Probability of positive cumulative return
        n_periods = len(returns)
        cumulative_mean = mean_return * n_periods
        cumulative_std = std_return * np.sqrt(n_periods)
        
        break_even_prob = 1 - stats.norm.cdf(0, loc=cumulative_mean, scale=cumulative_std)
        
        return break_even_prob
    
    def calculate_rachev_ratio(self, returns: pd.Series, 
                              alpha: float = 0.05,
                              beta: float = 0.05) -> float:
        """
        Rachev Ratio: Ratio of expected tail gain to expected tail loss
        """
        
        # Calculate expected tail gain (ETG) and expected tail loss (ETL)
        etg = returns.quantile(1 - alpha)
        etl = abs(returns.quantile(beta))
        
        rachev = etg / etl if etl > 0 else float('inf')
        
        return rachev
    
    def calculate_generalized_sharpe_ratio(self, returns: pd.Series, 
                                          order: int = 3) -> float:
        """
        Generalized Sharpe Ratio using higher moments
        Zakamouline & Koekebakker (2009)
        """
        
        mean_return = returns.mean()
        
        # Calculate lower partial moment
        threshold = 0
        lpm = np.mean(np.maximum(threshold - returns, 0) ** order) ** (1/order)
        
        gsr = mean_return / lpm if lpm > 0 else float('inf')
        
        return gsr * np.sqrt(self.annualization_factor)
    
    def calculate_bias_ratio(self, returns: pd.Series) -> float:
        """
        Bias Ratio: Measures smoothness of returns (potential manipulation indicator)
        """
        
        # Count consecutive positive returns
        positive_streaks = []
        current_streak = 0
        
        for r in returns:
            if r > 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    positive_streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            positive_streaks.append(current_streak)
        
        if not positive_streaks:
            return 0
            
        # Expected streak length under randomness
        win_rate = (returns > 0).mean()
        expected_streak = 1 / (1 - win_rate) if win_rate < 1 else float('inf')
        
        # Bias ratio
        actual_streak = np.mean(positive_streaks)
        bias_ratio = actual_streak / expected_streak if expected_streak > 0 else 0
        
        return bias_ratio
    
    def calculate_effective_number_of_bets(self, returns: pd.Series) -> float:
        """
        Effective number of independent bets (accounting for correlation)
        """
        
        n = len(returns)
        if n < 2:
            return n
            
        # Calculate autocorrelation
        autocorr = returns.autocorr(lag=1) if len(returns) > 1 else 0
        
        # Effective number of bets
        if abs(autocorr) < 1:
            effective_n = n * (1 - autocorr) / (1 + autocorr)
        else:
            effective_n = 1
            
        return max(1, effective_n)
    
    def calculate_active_premium(self, 
                                strategy_returns: pd.Series,
                                benchmark_returns: pd.Series) -> float:
        """
        Active Premium: Excess return over benchmark
        """
        
        if len(strategy_returns) != len(benchmark_returns):
            return np.nan
            
        active_returns = strategy_returns - benchmark_returns
        active_premium = active_returns.mean() * self.annualization_factor
        
        return active_premium
    
    def calculate_m_squared(self,
                          strategy_returns: pd.Series,
                          benchmark_returns: pd.Series,
                          market_returns: pd.Series) -> float:
        """
        M-squared: Risk-adjusted performance relative to market
        """
        
        strategy_sharpe = self._calculate_risk_adjusted_metrics(strategy_returns)['sharpe_ratio']
        market_vol = market_returns.std() * np.sqrt(self.annualization_factor)
        
        # Adjust strategy to market volatility
        m_squared = self.risk_free_rate + strategy_sharpe * market_vol
        
        return m_squared
    
    def calculate_upside_potential_ratio(self, returns: pd.Series,
                                        threshold: float = 0) -> float:
        """
        Upside Potential Ratio: Upside potential vs downside risk
        """
        
        upside_returns = returns[returns > threshold]
        downside_returns = returns[returns < threshold]
        
        if len(downside_returns) == 0:
            return float('inf')
            
        upside_potential = upside_returns.mean() if len(upside_returns) > 0 else 0
        downside_risk = np.sqrt(np.mean(downside_returns**2))
        
        upr = upside_potential / downside_risk if downside_risk > 0 else float('inf')
        
        return upr
    
    def calculate_capture_ratios(self,
                                strategy_returns: pd.Series,
                                benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate upside and downside capture ratios
        """
        
        if len(strategy_returns) != len(benchmark_returns):
            return {'upside_capture': np.nan, 'downside_capture': np.nan}
            
        # Upside periods
        upside_mask = benchmark_returns > 0
        if upside_mask.any():
            upside_capture = (1 + strategy_returns[upside_mask]).prod() / \
                           (1 + benchmark_returns[upside_mask]).prod()
        else:
            upside_capture = 1.0
            
        # Downside periods
        downside_mask = benchmark_returns < 0
        if downside_mask.any():
            downside_capture = (1 + strategy_returns[downside_mask]).prod() / \
                             (1 + benchmark_returns[downside_mask]).prod()
        else:
            downside_capture = 1.0
            
        return {
            'upside_capture': upside_capture,
            'downside_capture': downside_capture,
            'capture_ratio': upside_capture / downside_capture if downside_capture != 0 else float('inf')
        }


class CryptoSpecificMetrics:
    """
    Cryptocurrency-specific metrics and calculations
    """
    
    def __init__(self):
        self.hourly_periods = 24
        self.daily_periods = 365
        self.annual_periods = 365 * 24
        
    def calculate_24h_volatility(self, hourly_returns: pd.Series) -> float:
        """Calculate 24-hour rolling volatility"""
        
        if len(hourly_returns) < 24:
            return hourly_returns.std() * np.sqrt(24)
            
        return hourly_returns.rolling(24).std().mean() * np.sqrt(self.annual_periods)
    
    def calculate_funding_rate_impact(self,
                                    returns: pd.Series,
                                    funding_rates: Optional[pd.Series] = None) -> float:
        """
        Calculate impact of funding rates on returns (for perpetual futures)
        """
        
        if funding_rates is None:
            # Estimate funding impact (typically 0.01% every 8 hours)
            funding_cost = 0.0001 * 3  # Daily funding cost
            annual_funding = funding_cost * 365
            return -annual_funding
            
        # Calculate actual funding impact
        funding_impact = -funding_rates.sum()  # Negative because we pay funding
        return funding_impact
    
    def calculate_exchange_risk_score(self,
                                     returns: pd.Series,
                                     exchange_volumes: Optional[Dict] = None) -> float:
        """
        Calculate risk score based on exchange concentration
        """
        
        if exchange_volumes is None:
            # Default assumption: single exchange
            return 1.0  # Maximum risk
            
        # Calculate Herfindahl index for concentration
        total_volume = sum(exchange_volumes.values())
        market_shares = [v/total_volume for v in exchange_volumes.values()]
        herfindahl = sum(s**2 for s in market_shares)
        
        # Risk score (0 = perfectly distributed, 1 = single exchange)
        return herfindahl
    
    def calculate_defi_yield_adjusted_return(self,
                                            returns: pd.Series,
                                            staking_apy: float = 0.05) -> pd.Series:
        """
        Adjust returns for opportunity cost of DeFi yields
        """
        
        # Daily staking return
        daily_staking = (1 + staking_apy) ** (1/365) - 1
        
        # Adjust returns for opportunity cost
        adjusted_returns = returns - daily_staking
        
        return adjusted_returns
    
    def calculate_gas_adjusted_returns(self,
                                      returns: pd.Series,
                                      trade_sizes: pd.Series,
                                      gas_prices: Optional[pd.Series] = None) -> pd.Series:
        """
        Adjust returns for Ethereum gas costs
        """
        
        if gas_prices is None:
            # Estimate gas cost as percentage of trade
            gas_cost_pct = 0.001  # 0.1% per trade
        else:
            # Calculate actual gas costs
            gas_cost_pct = gas_prices / trade_sizes
            
        # Adjust returns
        gas_adjusted = returns - gas_cost_pct
        
        return gas_adjusted
    
    def calculate_impermanent_loss(self,
                                  price_start: float,
                                  price_end: float) -> float:
        """
        Calculate impermanent loss for liquidity providers
        """
        
        price_ratio = price_end / price_start
        
        # IL formula for 50/50 pool
        il = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
        
        return il
    
    def calculate_flash_crash_score(self, returns: pd.Series,
                                   window: int = 60) -> float:
        """
        Calculate susceptibility to flash crashes (1-minute to 1-hour recovery)
        """
        
        if len(returns) < window:
            return 0
            
        # Identify potential flash crashes (>5% drop)
        flash_threshold = -0.05
        
        flash_events = 0
        for i in range(len(returns) - window):
            min_return = returns.iloc[i:i+window].min()
            recovery = returns.iloc[i+window] - returns.iloc[i]
            
            if min_return < flash_threshold and recovery > abs(min_return) * 0.5:
                flash_events += 1
                
        # Normalize by number of periods
        flash_score = flash_events / (len(returns) / window)
        
        return flash_score


def compare_strategies_comprehensive(strategies: Dict[str, pd.Series],
                                    benchmark: pd.Series,
                                    strategy_types: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Comprehensive comparison of multiple strategies
    
    Args:
        strategies: Dictionary of strategy names to return series
        benchmark: Benchmark return series
        strategy_types: Optional dictionary of strategy types
        
    Returns:
        DataFrame with comprehensive metrics for all strategies
    """
    
    # Initialize metrics calculator
    metrics_calc = AdvancedMetrics()
    crypto_metrics = CryptoSpecificMetrics()
    validator = MetricsValidator()
    
    results = {}
    
    for name, returns in strategies.items():
        # Calculate all metrics
        strategy_metrics = metrics_calc.calculate_all_metrics(
            returns=returns,
            benchmark=benchmark
        )
        
        # Add advanced metrics
        n_samples = len(returns)
        if 'sharpe_ratio' in strategy_metrics:
            sharpe = strategy_metrics['sharpe_ratio']
            
            # Deflated Sharpe
            strategy_metrics['deflated_sharpe'] = metrics_calc.calculate_deflated_sharpe_ratio(
                sharpe, n_samples, len(strategies)
            )
            
            # Probabilistic Sharpe
            benchmark_metrics = metrics_calc.calculate_all_metrics(benchmark)
            benchmark_sharpe = benchmark_metrics.get('sharpe_ratio', 0)
            strategy_metrics['probabilistic_sharpe'] = metrics_calc.calculate_probabilistic_sharpe_ratio(
                sharpe, benchmark_sharpe, n_samples
            )
            
            # Statistical significance
            t_stat, p_value = metrics_calc.calculate_t_statistic_sharpe(sharpe, n_samples)
            strategy_metrics['sharpe_t_stat'] = t_stat
            strategy_metrics['sharpe_p_value'] = p_value
            
            # Minimum track record
            strategy_metrics['min_track_record'] = metrics_calc.calculate_minimum_track_record_length(sharpe)
        
        # Additional advanced metrics
        strategy_metrics['break_even_probability'] = metrics_calc.calculate_break_even_probability(returns)
        strategy_metrics['rachev_ratio'] = metrics_calc.calculate_rachev_ratio(returns)
        strategy_metrics['generalized_sharpe'] = metrics_calc.calculate_generalized_sharpe_ratio(returns)
        strategy_metrics['bias_ratio'] = metrics_calc.calculate_bias_ratio(returns)
        strategy_metrics['effective_bets'] = metrics_calc.calculate_effective_number_of_bets(returns)
        
        # Relative metrics
        if len(returns) == len(benchmark):
            strategy_metrics['active_premium'] = metrics_calc.calculate_active_premium(returns, benchmark)
            capture_ratios = metrics_calc.calculate_capture_ratios(returns, benchmark)
            strategy_metrics.update(capture_ratios)
        
        # Crypto-specific metrics
        strategy_metrics['24h_volatility'] = crypto_metrics.calculate_24h_volatility(returns)
        strategy_metrics['flash_crash_score'] = crypto_metrics.calculate_flash_crash_score(returns)
        strategy_metrics['funding_impact'] = crypto_metrics.calculate_funding_rate_impact(returns)
        
        results[name] = strategy_metrics
    
    # Create comparison DataFrame
    df = pd.DataFrame(results).T
    
    # Add rankings
    ranking_metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 
                      'omega_ratio', 'deflated_sharpe', 'probabilistic_sharpe']
    
    for metric in ranking_metrics:
        if metric in df.columns:
            df[f'{metric}_rank'] = df[metric].rank(ascending=False)
    
    # Overall score (average rank)
    rank_cols = [col for col in df.columns if col.endswith('_rank')]
    if rank_cols:
        df['overall_rank'] = df[rank_cols].mean(axis=1).rank()
    
    return df.sort_values('overall_rank') if 'overall_rank' in df.columns else df