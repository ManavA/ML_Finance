
class RiskManager:
    
    def __init__(self, config: Dict[str, Any]):

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
        entry_price = position['entry_price']
        
        if position['type'] == 'long':
            loss = (entry_price - current_price) / entry_price
            return loss >= self.stop_loss
        else:  # short position
            loss = (current_price - entry_price) / entry_price
            return loss >= self.stop_loss
    
    def check_take_profit(self, position: Dict[str, Any],
                         current_price: float) -> bool:
        entry_price = position['entry_price']
        
        if position['type'] == 'long':
            profit = (current_price - entry_price) / entry_price
            return profit >= self.take_profit
        else:  # short position
            profit = (entry_price - current_price) / entry_price
            return profit >= self.take_profit
    
    def update_trailing_stop(self, position_id: str,
                            current_price: float) -> Optional[float]:
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

        checks = {}
        
        # Check maximum drawdown
        current_drawdown = (peak_value - portfolio_value) / peak_value
        checks['max_drawdown_breached'] = current_drawdown >= self.max_drawdown
        
        # Check VaR limit
        current_var = self.calculate_var(returns)
        checks['var_limit_breached'] = current_var >= self.var_limit
        
        return checks