# src/strategies/ensemble.py


import numpy as np
import pandas as pd
from typing import List, Dict, Any
from .baseline_strategies import BaseStrategy


class AdaptiveEnsemble(BaseStrategy):

    def __init__(self, strategies: List[BaseStrategy], lookback: int = 30):
        super().__init__("Adaptive Ensemble")
        self.strategies = strategies
        self.lookback = lookback
        self.weights = np.ones(len(strategies)) / len(strategies)
        self.performance_history = []
    
    def update_weights(self, recent_performance: np.ndarray):

        if len(recent_performance) == 0:
            return
        
        # Calculate exponentially weighted performance
        decay_factor = 0.9
        weighted_performance = np.zeros(len(self.strategies))
        
        for i, perf in enumerate(recent_performance):
            weight = decay_factor ** (len(recent_performance) - i - 1)
            weighted_performance += weight * perf
        
        # Normalize to get weights (softmax for stability)
        exp_perf = np.exp(weighted_performance - np.max(weighted_performance))
        self.weights = exp_perf / np.sum(exp_perf)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:

        if not self.strategies:
            return pd.Series(index=data.index, data=0)
        
        # Get signals from all strategies
        strategy_signals = []
        for strategy in self.strategies:
            try:
                signals = strategy.generate_signals(data)
                strategy_signals.append(signals)
            except Exception as e:
                print(f"Strategy {strategy.name} failed: {e}")
                # Use neutral signals if strategy fails
                strategy_signals.append(pd.Series(index=data.index, data=0))
        
        # Combine signals using weights
        combined_signals = pd.Series(index=data.index, data=0.0)
        
        for i, signals in enumerate(strategy_signals):
            combined_signals += self.weights[i] * signals
        
        # Convert to discrete signals (-1, 0, 1)
        discrete_signals = pd.Series(index=data.index, data=0)
        discrete_signals[combined_signals > 0.3] = 1    # Buy threshold
        discrete_signals[combined_signals < -0.3] = -1  # Sell threshold
        
        return discrete_signals


class VotingEnsemble(BaseStrategy):

    def __init__(self, strategies: List[BaseStrategy], min_votes: int = None):
        super().__init__("Voting Ensemble")
        self.strategies = strategies
        self.min_votes = min_votes or (len(strategies) // 2 + 1)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:

        if not self.strategies:
            return pd.Series(index=data.index, data=0)
        
        # Get signals from all strategies
        all_signals = []
        for strategy in self.strategies:
            try:
                signals = strategy.generate_signals(data)
                all_signals.append(signals)
            except Exception as e:
                print(f"Strategy {strategy.name} failed: {e}")
                continue
        
        if not all_signals:
            return pd.Series(index=data.index, data=0)
        
        # Count votes for each signal type
        vote_matrix = pd.DataFrame(all_signals).T
        
        # Count buy votes (signal = 1)
        buy_votes = (vote_matrix == 1).sum(axis=1)
        
        # Count sell votes (signal = -1)
        sell_votes = (vote_matrix == -1).sum(axis=1)
        
        # Generate ensemble signals
        ensemble_signals = pd.Series(index=data.index, data=0)
        ensemble_signals[buy_votes >= self.min_votes] = 1
        ensemble_signals[sell_votes >= self.min_votes] = -1
        
        return ensemble_signals


class StackedEnsemble(BaseStrategy):

    
    def __init__(self, strategies: List[BaseStrategy], meta_model=None):
        super().__init__("Stacked Ensemble")
        self.strategies = strategies
        self.meta_model = meta_model
        self.is_trained = False
    
    def train_meta_model(self, data: pd.DataFrame, target_returns: pd.Series):

        # Get strategy signals as features
        strategy_features = []
        feature_names = []
        
        for strategy in self.strategies:
            try:
                signals = strategy.generate_signals(data)
                strategy_features.append(signals)
                feature_names.append(strategy.name)
            except Exception as e:
                print(f"Strategy {strategy.name} failed: {e}")
                continue
        
        if not strategy_features:
            return
        
        # Create feature matrix
        feature_matrix = pd.DataFrame(strategy_features).T
        feature_matrix.columns = feature_names
        
        # Initialize default meta-model if none provided
        if self.meta_model is None:
            from sklearn.linear_model import Ridge
            self.meta_model = Ridge(alpha=1.0)
        
        # Align features and targets
        aligned_features = feature_matrix.dropna()
        aligned_targets = target_returns.loc[aligned_features.index]
        
        if len(aligned_features) > 0 and len(aligned_targets) > 0:
            # Train meta-model
            self.meta_model.fit(aligned_features, aligned_targets)
            self.is_trained = True
            print(f"Meta-model trained on {len(aligned_features)} samples")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:

        if not self.strategies:
            return pd.Series(index=data.index, data=0)
        
        # Get strategy signals as features
        strategy_features = []
        
        for strategy in self.strategies:
            try:
                signals = strategy.generate_signals(data)
                strategy_features.append(signals)
            except Exception as e:
                print(f"Strategy {strategy.name} failed: {e}")
                # Use neutral signals if strategy fails
                strategy_features.append(pd.Series(index=data.index, data=0))
        
        # Create feature matrix
        feature_matrix = pd.DataFrame(strategy_features).T
        
        # Generate signals using meta-model
        if self.meta_model is not None and self.is_trained:
            try:
                # Get predictions from meta-model
                predictions = self.meta_model.predict(feature_matrix.fillna(0))
                
                # Convert predictions to signals
                signals = pd.Series(index=data.index, data=0)
                signals[predictions > 0.01] = 1   # Buy if predicted return > 1%
                signals[predictions < -0.01] = -1  # Sell if predicted return < -1%
                
                return signals
            except Exception as e:
                print(f"Meta-model prediction failed: {e}")
        
        # Fallback to simple averaging if meta-model fails
        return self._simple_average(strategy_features, data.index)
    
    def _simple_average(self, strategy_features: List[pd.Series], index: pd.Index) -> pd.Series:

        if not strategy_features:
            return pd.Series(index=index, data=0)
        
        # Average the signals
        avg_signals = pd.DataFrame(strategy_features).T.mean(axis=1)
        
        # Convert to discrete signals
        signals = pd.Series(index=index, data=0)
        signals[avg_signals > 0.3] = 1
        signals[avg_signals < -0.3] = -1
        
        return signals


class RiskParityEnsemble(BaseStrategy):

    
    def __init__(self, strategies: List[BaseStrategy], lookback: int = 60):
        super().__init__("Risk Parity Ensemble")
        self.strategies = strategies
        self.lookback = lookback
        self.strategy_returns = {}
        self.weights = np.ones(len(strategies)) / len(strategies)
    
    def update_strategy_returns(self, data: pd.DataFrame, returns: pd.Series):

        for i, strategy in enumerate(self.strategies):
            strategy_name = strategy.name
            
            if strategy_name not in self.strategy_returns:
                self.strategy_returns[strategy_name] = []
            
            try:
                # Get strategy signals and calculate implied returns
                signals = strategy.generate_signals(data)
                positions = strategy.get_positions(signals)
                
                # Calculate strategy returns (simplified)
                strategy_returns = positions.shift(1) * returns
                self.strategy_returns[strategy_name].extend(strategy_returns.dropna().tolist())
                
                # Keep only recent returns
                if len(self.strategy_returns[strategy_name]) > self.lookback:
                    self.strategy_returns[strategy_name] = \
                        self.strategy_returns[strategy_name][-self.lookback:]
                        
            except Exception as e:
                print(f"Failed to update returns for {strategy_name}: {e}")
    
    def calculate_risk_parity_weights(self):

        volatilities = []
        
        for strategy in self.strategies:
            strategy_name = strategy.name
            
            if strategy_name in self.strategy_returns and self.strategy_returns[strategy_name]:
                vol = np.std(self.strategy_returns[strategy_name])
                volatilities.append(vol if vol > 0 else 1e-6)  # Avoid division by zero
            else:
                volatilities.append(1.0)  # Default volatility
        
        # Calculate inverse volatility weights
        inv_vol_weights = 1.0 / np.array(volatilities)
        self.weights = inv_vol_weights / np.sum(inv_vol_weights)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
  
        if not self.strategies:
            return pd.Series(index=data.index, data=0)
        
        # Update weights based on recent performance
        self.calculate_risk_parity_weights()
        
        # Get weighted signals
        strategy_signals = []
        for strategy in self.strategies:
            try:
                signals = strategy.generate_signals(data)
                strategy_signals.append(signals)
            except Exception as e:
                print(f"Strategy {strategy.name} failed: {e}")
                strategy_signals.append(pd.Series(index=data.index, data=0))
        
        # Combine using risk parity weights
        combined_signals = pd.Series(index=data.index, data=0.0)
        
        for i, signals in enumerate(strategy_signals):
            combined_signals += self.weights[i] * signals
        
        # Convert to discrete signals
        discrete_signals = pd.Series(index=data.index, data=0)
        discrete_signals[combined_signals > 0.3] = 1
        discrete_signals[combined_signals < -0.3] = -1
        
        return discrete_signals