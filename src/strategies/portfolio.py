
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PortfolioManager:
    
    def __init__(self, config: Dict[str, Any]):

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