# src/models/reinforcement_learning.py
"""
Complete Reinforcement Learning Models for Cryptocurrency Trading
Including DQN, PPO, A2C, SAC, and Rainbow DQN
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CRYPTO TRADING ENVIRONMENT
# ============================================================================

class CryptoTradingEnv(gym.Env):
    """Advanced cryptocurrency trading environment for RL agents"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_balance: float = 10000,
                 commission: float = 0.001,
                 slippage: float = 0.001,
                 max_position: float = 1.0,
                 lookback_window: int = 50,
                 reward_scaling: float = 1e-4):
        """
        Initialize trading environment
        
        Args:
            data: OHLCV data with technical indicators
            initial_balance: Starting capital
            commission: Trading commission (0.1%)
            slippage: Slippage percentage
            max_position: Maximum position size (1.0 = 100% of capital)
            lookback_window: Historical window for observations
            reward_scaling: Scale factor for rewards
        """
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.max_position = max_position
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling
        
        # Define action space: [hold, buy_25%, buy_50%, buy_75%, buy_100%, 
        #                       sell_25%, sell_50%, sell_75%, sell_100%]
        self.action_space = spaces.Discrete(9)
        
        # Define observation space (price features + technical indicators + portfolio state)
        n_features = len(self._get_features(0))
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(lookback_window, n_features), 
            dtype=np.float32
        )
        
        self.reset()
    
    def _get_features(self, idx: int) -> np.ndarray:
        """Extract features for given index"""
        if idx < 0 or idx >= len(self.data):
            return np.zeros(20)  # Return zeros for out of bounds
        
        row = self.data.iloc[idx]
        
        # Price features
        returns = row.get('returns', 0)
        log_returns = np.log1p(returns) if returns > -1 else -1
        
        # Technical indicators (normalized)
        rsi = (row.get('RSI', 50) - 50) / 50
        macd = row.get('MACD', 0) / row.get('close', 1)
        bb_position = (row.get('close', 0) - row.get('BB_lower', 0)) / \
                      (row.get('BB_upper', 1) - row.get('BB_lower', 0.01))
        
        # Volume features
        volume_ratio = row.get('volume', 0) / self.data['volume'].rolling(20).mean().iloc[idx] \
                      if idx >= 20 else 1
        
        # Volatility
        volatility = self.data['returns'].rolling(20).std().iloc[idx] if idx >= 20 else 0.02
        
        # Market microstructure
        high_low_ratio = (row.get('high', 1) - row.get('low', 1)) / row.get('close', 1)
        close_open_ratio = (row.get('close', 1) - row.get('open', 1)) / row.get('open', 1)
        
        # Portfolio state
        position_ratio = self.position / self.max_position
        pnl_ratio = (self.balance - self.initial_balance) / self.initial_balance
        
        features = np.array([
            returns,
            log_returns,
            rsi,
            macd,
            bb_position,
            volume_ratio,
            volatility,
            high_low_ratio,
            close_open_ratio,
            position_ratio,
            pnl_ratio,
            self.position,  # Current position
            self.balance / self.initial_balance,  # Normalized balance
            self.trades_count / 100,  # Normalized trade count
            self.win_rate,  # Current win rate
            self.sharpe_ratio,  # Running Sharpe ratio
            self.max_drawdown,  # Current max drawdown
            self.holding_time / 100,  # Normalized holding time
            row.get('SMA_20', 0) / row.get('close', 1),  # SMA ratio
            row.get('EMA_12', 0) / row.get('close', 1),  # EMA ratio
        ])
        
        return features
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation window"""
        observations = []
        
        for i in range(self.lookback_window):
            idx = self.current_step - self.lookback_window + i + 1
            features = self._get_features(idx)
            observations.append(features)
        
        return np.array(observations, dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.current_step = self.lookback_window
        self.trades_count = 0
        self.winning_trades = 0
        self.trade_returns = []
        self.balance_history = [self.initial_balance]
        self.position_history = [0]
        self.action_history = []
        
        # Performance metrics
        self.win_rate = 0
        self.sharpe_ratio = 0
        self.max_drawdown = 0
        self.holding_time = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # Store previous balance for reward calculation
        prev_balance = self.balance
        prev_position_value = self.position * self.data.iloc[self.current_step]['close'] \
                              if self.position > 0 else 0
        prev_total = prev_balance + prev_position_value
        
        # Execute action
        self._execute_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate current total value
        current_price = self.data.iloc[self.current_step]['close']
        position_value = self.position * current_price if self.position > 0 else 0
        current_total = self.balance + position_value
        
        # Calculate reward
        reward = self._calculate_reward(prev_total, current_total, action)
        
        # Update metrics
        self._update_metrics()
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        truncated = self.balance <= self.initial_balance * 0.1  # Stop if lost 90%
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, truncated, info
    
    def _execute_action(self, action: int):
        """Execute trading action"""
        current_price = self.data.iloc[self.current_step]['close']
        
        # Map action to position change
        action_map = {
            0: 0,      # Hold
            1: 0.25,   # Buy 25%
            2: 0.50,   # Buy 50%
            3: 0.75,   # Buy 75%
            4: 1.00,   # Buy 100%
            5: -0.25,  # Sell 25%
            6: -0.50,  # Sell 50%
            7: -0.75,  # Sell 75%
            8: -1.00,  # Sell 100%
        }
        
        position_change = action_map[action]
        
        if position_change > 0:  # Buy
            # Calculate how much to buy
            max_buy = min(self.balance / current_price * (1 + self.commission + self.slippage),
                         self.max_position - self.position)
            buy_amount = max_buy * position_change
            
            if buy_amount > 0:
                # Apply slippage and commission
                actual_price = current_price * (1 + self.slippage)
                cost = buy_amount * actual_price * (1 + self.commission)
                
                if cost <= self.balance:
                    self.position += buy_amount
                    self.balance -= cost
                    self.position_price = actual_price  # Update average price
                    self.trades_count += 1
                    if self.holding_time == 0:
                        self.holding_time = 1
        
        elif position_change < 0:  # Sell
            # Calculate how much to sell
            sell_amount = min(self.position, abs(position_change) * self.position)
            
            if sell_amount > 0:
                # Apply slippage and commission
                actual_price = current_price * (1 - self.slippage)
                proceeds = sell_amount * actual_price * (1 - self.commission)
                
                # Calculate trade return
                if self.position_price > 0:
                    trade_return = (actual_price - self.position_price) / self.position_price
                    self.trade_returns.append(trade_return)
                    if trade_return > 0:
                        self.winning_trades += 1
                
                self.position -= sell_amount
                self.balance += proceeds
                self.trades_count += 1
                
                if self.position == 0:
                    self.holding_time = 0
        
        # Update histories
        self.balance_history.append(self.balance)
        self.position_history.append(self.position)
        self.action_history.append(action)
    
    def _calculate_reward(self, prev_total: float, current_total: float, action: int) -> float:
        """Calculate step reward using advanced reward shaping"""
        
        # Base reward: change in total portfolio value
        value_change = (current_total - prev_total) / prev_total
        base_reward = value_change * 100  # Scale to percentage
        
        # Sharpe ratio component (risk-adjusted returns)
        if len(self.trade_returns) > 1:
            returns = np.array(self.trade_returns)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            sharpe_reward = sharpe * 0.1
        else:
            sharpe_reward = 0
        
        # Win rate component
        win_rate_reward = (self.win_rate - 0.5) * 0.5 if self.trades_count > 0 else 0
        
        # Drawdown penalty
        drawdown_penalty = -abs(self.max_drawdown) * 0.2
        
        # Action penalty (to reduce overtrading)
        action_penalty = -0.01 if action != 0 else 0
        
        # Combine rewards
        total_reward = base_reward + sharpe_reward + win_rate_reward + drawdown_penalty + action_penalty
        
        return total_reward * self.reward_scaling
    
    def _update_metrics(self):
        """Update performance metrics"""
        
        # Win rate
        if self.trades_count > 0:
            self.win_rate = self.winning_trades / self.trades_count
        
        # Sharpe ratio
        if len(self.balance_history) > 20:
            returns = np.diff(self.balance_history[-20:]) / self.balance_history[-21:-1]
            if np.std(returns) > 0:
                self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Max drawdown
        if len(self.balance_history) > 1:
            peak = np.maximum.accumulate(self.balance_history)
            drawdown = (np.array(self.balance_history) - peak) / peak
            self.max_drawdown = np.min(drawdown)
        
        # Update holding time
        if self.position > 0:
            self.holding_time += 1
    
    def _get_info(self) -> Dict:
        """Get environment info"""
        return {
            'balance': self.balance,
            'position': self.position,
            'total_value': self.balance + self.position * self.data.iloc[self.current_step]['close'] 
                          if self.position > 0 else self.balance,
            'trades_count': self.trades_count,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'current_step': self.current_step,
        }


# ============================================================================
# DQN AGENT
# ============================================================================

class DQNetwork(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, input_shape: Tuple[int, int], n_actions: int, hidden_size: int = 512):
        super(DQNetwork, self).__init__()
        
        # Flatten input shape
        self.input_size = input_shape[0] * input_shape[1]
        
        # Dueling DQN architecture
        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_actions)
        )
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Extract features
        features = self.feature_layer(x)
        
        # Calculate value and advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine to get Q-values (Dueling DQN formula)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
        self.Transition = namedtuple('Transition', 
                                     ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = self.Transition(state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample batch with prioritization"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # Calculate sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get samples
        batch = [self.buffer[idx] for idx in indices]
        states = torch.stack([torch.tensor(t.state, dtype=torch.float32) for t in batch])
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.stack([torch.tensor(t.next_state, dtype=torch.float32) for t in batch])
        dones = torch.tensor([t.done for t in batch], dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network Agent with advanced features"""
    
    def __init__(self,
                 state_shape: Tuple[int, int],
                 n_actions: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 10000,
                 buffer_size: int = 100000,
                 batch_size: int = 32,
                 update_frequency: int = 4,
                 target_update_frequency: int = 1000,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize DQN agent
        
        Args:
            state_shape: Shape of observation space
            n_actions: Number of actions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay steps
            buffer_size: Replay buffer size
            batch_size: Training batch size
            update_frequency: Steps between training updates
            target_update_frequency: Steps between target network updates
            device: Device to use for training
        """
        
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        self.device = torch.device(device)
        
        # Epsilon-greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        
        # Networks
        self.q_network = DQNetwork(state_shape, n_actions).to(self.device)
        self.target_network = DQNetwork(state_shape, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Training tracking
        self.steps = 0
        self.training_losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step"""
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * ~dones
        
        # Calculate TD errors for priority update
        td_errors = (target_q_values - current_q_values).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Calculate weighted loss
        loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.training_losses.append(loss.item())
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon_start - self.steps / self.epsilon_decay)
        
        # Update target network
        if self.steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.steps += 1
    
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon,
        }, path)
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_dqn(data: pd.DataFrame, 
              episodes: int = 100,
              save_path: str = 'models/dqn_crypto.pt') -> DQNAgent:
    """
    Train DQN agent on cryptocurrency data
    
    Args:
        data: OHLCV data with technical indicators
        episodes: Number of training episodes
        save_path: Path to save trained model
        
    Returns:
        Trained DQN agent
    """
    
    # Create environment
    env = CryptoTradingEnv(data)
    
    # Create agent
    agent = DQNAgent(
        state_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=episodes * len(data) // 10,
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        truncated = False
        
        while not done and not truncated:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Execute action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done or truncated)
            
            # Train
            if agent.steps % agent.update_frequency == 0:
                agent.train_step()
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(f"Episode {episode + 1}/{episodes}, "
                       f"Avg Reward: {avg_reward:.4f}, "
                       f"Epsilon: {agent.epsilon:.4f}, "
                       f"Total Value: {info['total_value']:.2f}")
    
    # Save model
    agent.save(save_path)
    logger.info(f"Model saved to {save_path}")
    
    return agent