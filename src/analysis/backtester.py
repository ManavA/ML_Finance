# src/analysis/backtester.py
"""Backtesting framework for strategy analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestResults:
    """Container for backtest results."""
    strategy_name: str
    returns: pd.Series
    cumulative_returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    equity_curve: pd.Series
    drawdown: pd.Series


class Backtester:
    """Simple backtesting engine for research."""
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 commission: float = 0.001,
                 slippage: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
    def backtest(self, 
                data: pd.DataFrame, 
                signals: pd.Series,
                strategy_name: str = "Strategy") -> BacktestResults:
        """Run backtest on signals."""
        # Align data and signals
        common_index = data.index.intersection(signals.index)
        data = data.loc[common_index]
        signals = signals.loc[common_index]
        
        # Calculate positions
        positions = self._calculate_positions(signals)
        
        # Calculate returns
        price_returns = data['close'].pct_change()
        
        # Strategy returns (before costs)
        strategy_returns = positions.shift(1) * price_returns
        
        # Apply transaction costs
        trades = self._identify_trades(positions)
        transaction_costs = trades * (self.commission + self.slippage)
        
        # Net returns
        net_returns = strategy_returns - transaction_costs
        net_returns = net_returns.fillna(0)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + net_returns).cumprod()
        
        # Calculate equity curve
        equity_curve = self.initial_capital * cumulative_returns
        
        # Calculate drawdown
        drawdown = self._calculate_drawdown(equity_curve)
        
        # Calculate metrics
        metrics = self._calculate_metrics(net_returns, equity_curve, positions, trades)
        
        # Compile trade log
        trade_log = self._compile_trade_log(data, positions, trades)
        
        return BacktestResults(
            strategy_name=strategy_name,
            returns=net_returns,
            cumulative_returns=cumulative_returns,
            positions=positions,
            trades=trade_log,
            metrics=metrics,
            equity_curve=equity_curve,
            drawdown=drawdown
        )
    
    def _calculate_positions(self, signals: pd.Series) -> pd.Series:
        """Convert signals to positions."""
        positions = pd.Series(index=signals.index, data=0.0)
        
        position = 0
        for i in range(len(signals)):
            if signals.iloc[i] == 1:  # Buy signal
                position = 1
            elif signals.iloc[i] == -1:  # Sell signal
                position = 0
            positions.iloc[i] = position
            
        return positions
    
    def _identify_trades(self, positions: pd.Series) -> pd.Series:
        """Identify when trades occur."""
        trades = positions.diff().abs()
        trades = trades.fillna(0)
        return trades
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown
    
    def _calculate_metrics(self, 
                          returns: pd.Series,
                          equity_curve: pd.Series,
                          positions: pd.Series,
                          trades: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1
        
        # Annualized metrics (assuming daily data)
        n_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns / downside_std if downside_std > 0 else 0
        
        # Drawdown metrics
        max_drawdown = self._calculate_drawdown(equity_curve).min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        n_trades = trades.sum() / 2  # Divide by 2 because we count both entry and exit
        
        # Win rate
        trade_returns = []
        position = 0
        entry_price = 0
        
        for i in range(len(positions)):
            if positions.iloc[i] == 1 and position == 0:
                # Entry
                entry_price = equity_curve.iloc[i]
                position = 1
            elif positions.iloc[i] == 0 and position == 1:
                # Exit
                exit_price = equity_curve.iloc[i]
                trade_return = (exit_price - entry_price) / entry_price
                trade_returns.append(trade_return)
                position = 0
        
        if trade_returns:
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
            avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
            avg_loss = np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0
            profit_factor = abs(sum(r for r in trade_returns if r > 0) / sum(r for r in trade_returns if r < 0)) if any(r < 0 for r in trade_returns) else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_equity': equity_curve.iloc[-1]
        }
    
    def _compile_trade_log(self, 
                          data: pd.DataFrame,
                          positions: pd.Series,
                          trades: pd.Series) -> pd.DataFrame:
        """Compile detailed trade log."""
        trade_list = []
        position = 0
        entry_date = None
        entry_price = None
        
        for i in range(len(positions)):
            if positions.iloc[i] == 1 and position == 0:
                # Entry
                entry_date = positions.index[i]
                entry_price = data.loc[entry_date, 'close']
                position = 1
                
            elif positions.iloc[i] == 0 and position == 1:
                # Exit
                exit_date = positions.index[i]
                exit_price = data.loc[exit_date, 'close']
                
                trade_list.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': (exit_price - entry_price) / entry_price,
                    'duration': (exit_date - entry_date).days
                })
                position = 0
        
        if trade_list:
            return pd.DataFrame(trade_list)
        else:
            return pd.DataFrame()
    
    def compare_strategies(self, 
                          results_list: List[BacktestResults]) -> pd.DataFrame:
        """Compare multiple strategy results."""
        comparison = []
        
        for result in results_list:
            metrics = result.metrics.copy()
            metrics['strategy'] = result.strategy_name
            comparison.append(metrics)
        
        df = pd.DataFrame(comparison)
        
        # Reorder columns
        cols = ['strategy'] + [col for col in df.columns if col != 'strategy']
        df = df[cols]
        
        # Sort by Sharpe ratio
        df = df.sort_values('sharpe_ratio', ascending=False)
        
        return df