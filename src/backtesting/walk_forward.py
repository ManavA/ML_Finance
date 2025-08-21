#!/usr/bin/env python3
"""
Walk-forward backtesting framework for ML comparison research
Implements chronological cross-validation to prevent lookahead bias
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardSplit:
    """Configuration for a single walk-forward split"""
    fold: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    
    def to_dict(self) -> Dict:
        return {
            'fold': self.fold,
            'train': (self.train_start, self.train_end),
            'val': (self.val_start, self.val_end),
            'test': (self.test_start, self.test_end)
        }

@dataclass
class BacktestResults:
    """Results from a single backtest"""
    fold: int
    symbol: str
    model_name: str
    returns: np.ndarray
    predictions: np.ndarray
    positions: np.ndarray
    equity_curve: np.ndarray
    metrics: Dict[str, float]
    timestamps: pd.DatetimeIndex
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        # Basic returns metrics
        total_return = (self.equity_curve[-1] / self.equity_curve[0] - 1) * 100
        
        # Annualized metrics
        n_periods = len(self.returns)
        n_years = n_periods / (252 * 24)  # Assuming hourly data, 252 trading days
        annualized_return = (self.equity_curve[-1] / self.equity_curve[0]) ** (1/n_years) - 1
        
        # Risk metrics
        returns_clean = self.returns[~np.isnan(self.returns)]
        volatility = np.std(returns_clean) * np.sqrt(252 * 24)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Downside metrics
        downside_returns = returns_clean[returns_clean < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252 * 24) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0
        
        # Drawdown metrics
        cumulative = pd.Series(self.equity_curve)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Trading metrics
        n_trades = np.sum(np.abs(np.diff(self.positions)) > 0)
        win_rate = np.mean(self.returns[self.returns != 0] > 0) if np.any(self.returns != 0) else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'n_trades': n_trades,
            'win_rate': win_rate * 100
        }

class WalkForwardBacktester:
    """
    Walk-forward backtesting framework for comparing ML models vs traditional strategies
    """
    
    def __init__(self, 
                 train_months: int = 12,
                 val_months: int = 3,
                 test_months: int = 3,
                 step_months: int = 3,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        """
        Initialize walk-forward backtester
        
        Args:
            train_months: Training period length
            val_months: Validation period length
            test_months: Test period length
            step_months: Step size for rolling window
            commission: Trading commission (0.1% default)
            slippage: Slippage factor (0.05% default)
        """
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.step_months = step_months
        self.commission = commission
        self.slippage = slippage
        self.results = []
        
    def create_walk_forward_splits(self, 
                                 start_date: str,
                                 end_date: str) -> List[WalkForwardSplit]:
        """
        Create walk-forward splits for chronological cross-validation
        
        Args:
            start_date: Start date of data (YYYY-MM-DD)
            end_date: End date of data (YYYY-MM-DD)
            
        Returns:
            List of WalkForwardSplit objects
        """
        splits = []
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        current_date = start
        fold = 1
        
        total_months = self.train_months + self.val_months + self.test_months
        
        while current_date + pd.DateOffset(months=total_months) <= end:
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=self.train_months)
            val_start = train_end
            val_end = val_start + pd.DateOffset(months=self.val_months)
            test_start = val_end
            test_end = test_start + pd.DateOffset(months=self.test_months)
            
            split = WalkForwardSplit(
                fold=fold,
                train_start=train_start.strftime('%Y-%m-%d'),
                train_end=train_end.strftime('%Y-%m-%d'),
                val_start=val_start.strftime('%Y-%m-%d'),
                val_end=val_end.strftime('%Y-%m-%d'),
                test_start=test_start.strftime('%Y-%m-%d'),
                test_end=test_end.strftime('%Y-%m-%d')
            )
            splits.append(split)
            
            # Move window forward
            current_date += pd.DateOffset(months=self.step_months)
            fold += 1
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def split_data(self, 
                  data: pd.DataFrame,
                  split: WalkForwardSplit) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data according to walk-forward split
        
        Args:
            data: Full dataset
            split: WalkForwardSplit configuration
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Ensure timestamp column exists and is datetime
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            time_col = 'timestamp'
        else:
            time_col = data.index.name or 'index'
        
        # Create masks for each period
        train_mask = (data[time_col] >= split.train_start) & (data[time_col] < split.train_end)
        val_mask = (data[time_col] >= split.val_start) & (data[time_col] < split.val_end)
        test_mask = (data[time_col] >= split.test_start) & (data[time_col] < split.test_end)
        
        train_data = data[train_mask].copy()
        val_data = data[val_mask].copy()
        test_data = data[test_mask].copy()
        
        logger.info(f"Fold {split.fold} - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def backtest_strategy(self,
                         data: pd.DataFrame,
                         predictions: np.ndarray,
                         symbol: str,
                         model_name: str,
                         fold: int,
                         position_sizing: str = 'fixed') -> BacktestResults:
        """
        Backtest a strategy based on predictions
        
        Args:
            data: Test data with prices
            predictions: Model predictions (can be returns, directions, or probabilities)
            symbol: Asset symbol
            model_name: Name of the model/strategy
            fold: Fold number
            position_sizing: Position sizing method ('fixed', 'kelly', 'risk_parity')
            
        Returns:
            BacktestResults object
        """
        # Ensure we have price data
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        prices = data['close'].values
        timestamps = pd.to_datetime(data['timestamp']) if 'timestamp' in data.columns else data.index
        
        # Convert predictions to positions
        positions = self.predictions_to_positions(predictions, position_sizing)
        
        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]
        
        # Align positions with returns (positions are for entering at time t, returns realized at t+1)
        if len(positions) > len(price_returns):
            positions = positions[:-1]
        elif len(positions) < len(price_returns):
            price_returns = price_returns[:len(positions)]
        
        # Calculate strategy returns accounting for transaction costs
        strategy_returns = []
        prev_position = 0
        
        for i, (pos, ret) in enumerate(zip(positions, price_returns)):
            # Transaction cost if position changed
            trade_cost = 0
            if pos != prev_position:
                trade_cost = abs(pos - prev_position) * (self.commission + self.slippage)
            
            # Strategy return
            strat_return = pos * ret - trade_cost
            strategy_returns.append(strat_return)
            prev_position = pos
        
        strategy_returns = np.array(strategy_returns)
        
        # Calculate equity curve
        equity_curve = np.cumprod(1 + strategy_returns)
        equity_curve = np.concatenate([[1.0], equity_curve])
        
        # Create results object
        results = BacktestResults(
            fold=fold,
            symbol=symbol,
            model_name=model_name,
            returns=strategy_returns,
            predictions=predictions[:len(positions)],
            positions=positions,
            equity_curve=equity_curve,
            metrics={},
            timestamps=timestamps[:len(positions)]
        )
        
        # Calculate metrics
        results.metrics = results.calculate_metrics()
        
        return results
    
    def predictions_to_positions(self, 
                                predictions: np.ndarray,
                                method: str = 'fixed') -> np.ndarray:
        """
        Convert model predictions to trading positions
        
        Args:
            predictions: Model predictions
            method: Position sizing method
            
        Returns:
            Array of positions (-1, 0, 1 for short, flat, long)
        """
        if method == 'fixed':
            # Simple threshold-based positions
            if predictions.ndim == 1:
                # Assume predictions are returns or directions
                if np.all((predictions >= 0) & (predictions <= 1)):
                    # Probabilities
                    positions = np.where(predictions > 0.5, 1, -1)
                else:
                    # Returns
                    positions = np.sign(predictions)
            else:
                # Multi-class or multi-output
                if predictions.shape[1] == 2:
                    # Binary classification probabilities
                    positions = np.where(predictions[:, 1] > 0.5, 1, -1)
                else:
                    # Multi-class - map to positions
                    positions = np.argmax(predictions, axis=1) - 1
        
        elif method == 'kelly':
            # Kelly criterion position sizing
            # Simplified Kelly: f = (p * b - q) / b
            # where p = win probability, q = loss probability, b = win/loss ratio
            if predictions.ndim == 1 and np.all((predictions >= 0) & (predictions <= 1)):
                p = predictions
                q = 1 - p
                b = 1.5  # Assumed win/loss ratio
                kelly = (p * b - q) / b
                # Cap at 0.25 for safety
                positions = np.clip(kelly, -0.25, 0.25) * 4  # Scale to -1, 1 range
            else:
                positions = self.predictions_to_positions(predictions, 'fixed')
        
        elif method == 'risk_parity':
            # Risk parity position sizing (simplified)
            # Position inversely proportional to predicted volatility
            positions = self.predictions_to_positions(predictions, 'fixed')
            # Would need volatility predictions to implement properly
        
        else:
            positions = self.predictions_to_positions(predictions, 'fixed')
        
        return positions
    
    def run_walk_forward_backtest(self,
                                 data: pd.DataFrame,
                                 model_fn: Callable,
                                 symbol: str,
                                 model_name: str,
                                 splits: Optional[List[WalkForwardSplit]] = None) -> pd.DataFrame:
        """
        Run complete walk-forward backtest for a model
        
        Args:
            data: Full dataset with features
            model_fn: Function that trains model and returns predictions
                     Signature: model_fn(train_data, val_data, test_data) -> predictions
            symbol: Asset symbol
            model_name: Model name
            splits: Optional pre-defined splits
            
        Returns:
            DataFrame with aggregated results
        """
        if splits is None:
            # Create default splits
            start_date = data['timestamp'].min() if 'timestamp' in data.columns else data.index.min()
            end_date = data['timestamp'].max() if 'timestamp' in data.columns else data.index.max()
            splits = self.create_walk_forward_splits(str(start_date)[:10], str(end_date)[:10])
        
        fold_results = []
        
        for split in tqdm(splits, desc=f"Walk-forward {model_name}"):
            try:
                # Split data
                train_data, val_data, test_data = self.split_data(data, split)
                
                if len(train_data) < 100 or len(test_data) < 20:
                    logger.warning(f"Skipping fold {split.fold} - insufficient data")
                    continue
                
                # Train model and get predictions
                predictions = model_fn(train_data, val_data, test_data)
                
                # Run backtest
                results = self.backtest_strategy(
                    test_data,
                    predictions,
                    symbol,
                    model_name,
                    split.fold
                )
                
                fold_results.append(results)
                
            except Exception as e:
                logger.error(f"Error in fold {split.fold}: {e}")
                continue
        
        # Aggregate results
        if fold_results:
            aggregated = self.aggregate_results(fold_results)
            self.results.extend(fold_results)
            return aggregated
        else:
            return pd.DataFrame()
    
    def aggregate_results(self, results: List[BacktestResults]) -> pd.DataFrame:
        """
        Aggregate results across folds
        
        Args:
            results: List of BacktestResults
            
        Returns:
            DataFrame with aggregated metrics
        """
        metrics_list = []
        
        for result in results:
            metrics = result.metrics.copy()
            metrics['fold'] = result.fold
            metrics['symbol'] = result.symbol
            metrics['model'] = result.model_name
            metrics_list.append(metrics)
        
        df = pd.DataFrame(metrics_list)
        
        # Calculate aggregate statistics
        agg_stats = df.groupby(['symbol', 'model']).agg({
            'total_return': ['mean', 'std', 'min', 'max'],
            'sharpe_ratio': ['mean', 'std'],
            'max_drawdown': ['mean', 'min'],
            'win_rate': 'mean',
            'n_trades': 'sum'
        }).round(2)
        
        return agg_stats
    
    def compare_models(self, models: Dict[str, Callable],
                       data: pd.DataFrame,
                       symbol: str) -> pd.DataFrame:
        """
        Compare multiple models using walk-forward validation
        
        Args:
            models: Dictionary of model_name -> model_function
            data: Full dataset
            symbol: Asset symbol
            
        Returns:
            DataFrame comparing all models
        """
        all_results = []
        
        # Create splits once for consistency
        start_date = data['timestamp'].min() if 'timestamp' in data.columns else data.index.min()
        end_date = data['timestamp'].max() if 'timestamp' in data.columns else data.index.max()
        splits = self.create_walk_forward_splits(str(start_date)[:10], str(end_date)[:10])
        
        for model_name, model_fn in models.items():
            logger.info(f"Testing {model_name} on {symbol}")
            results = self.run_walk_forward_backtest(
                data, model_fn, symbol, model_name, splits
            )
            if not results.empty:
                all_results.append(results)
        
        if all_results:
            comparison_df = pd.concat(all_results)
            return comparison_df
        else:
            return pd.DataFrame()
    
    def save_results(self, filepath: str):
        """Save backtest results to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load backtest results from file"""
        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)
        logger.info(f"Loaded {len(self.results)} results from {filepath}")