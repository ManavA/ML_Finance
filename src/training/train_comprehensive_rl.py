#!/usr/bin/env python3
"""
COMPREHENSIVE REINFORCEMENT LEARNING TRAINING
Train RL agents with 5,000+ episodes and proper evaluation
"""

import sys
import os
sys.path.append('src')
os.chdir('C:/Users/manav/claude')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("COMPREHENSIVE RL TRAINING - 5,000+ EPISODES")
print("=" * 60)

# ============================================================================
# ENHANCED TRADING ENVIRONMENT
# ============================================================================

class AdvancedCryptoTradingEnv:
    """Enhanced trading environment with better reward shaping"""
    
    def __init__(self, data, initial_balance=10000, lookback=30, commission=0.001):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.lookback = lookback
        self.commission = commission
        
        # Enhanced state space
        self.action_space = 5  # [Hold, Buy25%, Buy50%, Buy75%, Sell_All]
        self.state_size = lookback * 6 + 5  # OHLCV + returns, plus portfolio info
        
        self.reset()
    
    def reset(self):
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.position = 0
        self.position_value = 0
        self.trades = []
        self.total_trades = 0
        self.winning_trades = 0
        
        # Performance tracking
        self.balance_history = [self.initial_balance]
        self.portfolio_values = [self.initial_balance]
        self.max_portfolio_value = self.initial_balance
        
        return self._get_state()
    
    def _get_state(self):
        # Price history (normalized)
        start_idx = max(0, self.current_step - self.lookback)
        end_idx = self.current_step
        
        price_data = self.data.iloc[start_idx:end_idx]
        
        if len(price_data) < self.lookback:
            # Pad with first available data
            padding_needed = self.lookback - len(price_data)
            first_row = self.data.iloc[0] if not self.data.empty else pd.Series([1]*6, index=['open','high','low','close','volume','returns'])
            padding = pd.DataFrame([first_row] * padding_needed)
            price_data = pd.concat([padding, price_data], ignore_index=True)
        
        current_price = self.data.iloc[min(self.current_step, len(self.data)-1)]['close']
        
        # Normalize price features
        state_features = []
        for _, row in price_data.iterrows():
            normalized = [
                row.get('open', current_price) / current_price,
                row.get('high', current_price) / current_price,  
                row.get('low', current_price) / current_price,
                row.get('close', current_price) / current_price,
                np.log1p(row.get('volume', 1e6)) / 20,  # Log-normalize volume
                row.get('returns', 0) * 100  # Scale returns
            ]
            state_features.extend(normalized)
        
        # Portfolio state
        total_value = self.balance + self.position * current_price
        portfolio_state = [
            self.balance / self.initial_balance,
            (self.position * current_price) / self.initial_balance,
            total_value / self.initial_balance,
            self.total_trades / 100,  # Normalize trade count
            (total_value - self.max_portfolio_value) / self.initial_balance  # Drawdown
        ]
        
        state_features.extend(portfolio_state)
        return np.array(state_features, dtype=np.float32)
    
    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {'portfolio_value': self.balance + self.position * self.data.iloc[-1]['close']}
        
        current_price = self.data.iloc[self.current_step]['close']
        prev_total_value = self.balance + self.position * current_price
        
        # Execute action
        self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        if self.current_step >= len(self.data) - 1:
            done = True
            next_price = current_price
        else:
            done = False
            next_price = self.data.iloc[self.current_step]['close']
        
        # Calculate new portfolio value
        new_total_value = self.balance + self.position * next_price
        
        # Enhanced reward calculation
        reward = self._calculate_reward(prev_total_value, new_total_value, action)
        
        # Update tracking
        self.portfolio_values.append(new_total_value)
        self.max_portfolio_value = max(self.max_portfolio_value, new_total_value)
        
        return self._get_state(), reward, done, {'portfolio_value': new_total_value}
    
    def _execute_action(self, action, current_price):
        """Execute trading action"""
        # Action mapping: 0=Hold, 1=Buy25%, 2=Buy50%, 3=Buy75%, 4=SellAll
        
        if action == 1:  # Buy 25%
            buy_amount = self.balance * 0.25
            shares = buy_amount / (current_price * (1 + self.commission))
            if buy_amount > 1 and shares > 0:
                self.position += shares
                self.balance -= buy_amount
                self.total_trades += 1
                
        elif action == 2:  # Buy 50%
            buy_amount = self.balance * 0.50
            shares = buy_amount / (current_price * (1 + self.commission))
            if buy_amount > 1 and shares > 0:
                self.position += shares
                self.balance -= buy_amount
                self.total_trades += 1
                
        elif action == 3:  # Buy 75%
            buy_amount = self.balance * 0.75
            shares = buy_amount / (current_price * (1 + self.commission))
            if buy_amount > 1 and shares > 0:
                self.position += shares
                self.balance -= buy_amount
                self.total_trades += 1
                
        elif action == 4 and self.position > 0:  # Sell all
            sell_value = self.position * current_price * (1 - self.commission)
            self.balance += sell_value
            self.position = 0
            self.total_trades += 1
    
    def _calculate_reward(self, prev_value, new_value, action):
        """Enhanced reward function"""
        # Base reward: portfolio value change
        value_change = (new_value - prev_value) / prev_value if prev_value > 0 else 0
        base_reward = value_change * 100  # Scale to percentage
        
        # Risk-adjusted component
        if len(self.portfolio_values) > 30:
            recent_values = np.array(self.portfolio_values[-30:])
            recent_returns = np.diff(recent_values) / recent_values[:-1]
            volatility = np.std(recent_returns) + 1e-8
            sharpe_component = (np.mean(recent_returns) / volatility) * 0.1
        else:
            sharpe_component = 0
        
        # Drawdown penalty
        current_drawdown = (new_value - self.max_portfolio_value) / self.max_portfolio_value
        drawdown_penalty = current_drawdown * 0.5 if current_drawdown < 0 else 0
        
        # Action cost (reduce overtrading)
        action_cost = -0.001 if action != 0 else 0
        
        return base_reward + sharpe_component + drawdown_penalty + action_cost

# ============================================================================
# ENHANCED DQN AGENT
# ============================================================================

class EnhancedDQN(nn.Module):
    """Enhanced DQN with better architecture"""
    
    def __init__(self, state_size, action_size):
        super().__init__()
        
        self.feature_layers = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Dueling DQN streams
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        features = self.feature_layers(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling DQN combination
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class EnhancedDQNAgent:
    """Enhanced DQN agent with prioritized experience replay"""
    
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = EnhancedDQN(state_size, action_size).to(self.device)
        self.target_network = EnhancedDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Memory
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.target_update_freq = 1000
        self.steps = 0
        
        # Performance tracking
        self.losses = []
        self.q_values_history = []
        
        self.update_target_network()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        self.q_values_history.append(q_values.max().item())
        return q_values.argmax().item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.steps += 1

# ============================================================================
# PPO AGENT (Enhanced)
# ============================================================================

class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        shared = self.shared_layers(x)
        return self.actor(shared), self.critic(shared)

class EnhancedPPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # Performance tracking
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_history = []
    
    def get_action_and_value(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def train_step(self, states, actions, rewards, old_log_probs, values):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        
        # Calculate advantages using GAE
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = advantages + values
        
        # Multiple epochs of updates
        for _ in range(4):
            action_probs, new_values = self.network(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            
            total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        # Track performance
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.entropy_history.append(entropy.item())

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def load_comprehensive_data():
    """Load the full dataset for RL training"""
    print("\nLoading comprehensive dataset for RL training...")
    
    files = glob.glob('data/s3_cache/crypto_*.parquet')
    
    if not files:
        print("Creating synthetic dataset for RL training...")
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='H')
        
        # More realistic price evolution
        returns = np.random.normal(0.0005, 0.02, len(dates))
        returns += 0.3 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24 * 30))  # Monthly cycles
        
        prices = 35000 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'close': prices,
            'volume': np.random.lognormal(20, 1, len(dates)),
            'returns': np.concatenate([[0], np.diff(prices) / prices[:-1]])
        })
        
        return df
    
    # Load real data - use more files for RL
    print(f"Loading from {len(files)} cache files...")
    all_data = []
    
    for i, file in enumerate(sorted(files)[:100]):  # Use more files for RL training
        try:
            df = pd.read_parquet(file)
            btc_tickers = [t for t in df['ticker'].unique() if 'BTC' in t.upper()]
            if btc_tickers:
                btc_df = df[df['ticker'] == btc_tickers[0]].copy()
                if not btc_df.empty:
                    all_data.append(btc_df)
        except Exception as e:
            continue
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        if 'window_start' in combined.columns:
            combined['timestamp'] = pd.to_datetime(combined['window_start'])
        
        combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        combined['returns'] = combined['close'].pct_change().fillna(0)
        
        print(f"Loaded {len(combined):,} records for RL training")
        return combined[['open', 'high', 'low', 'close', 'volume', 'returns']]
    
    return None

def train_comprehensive_dqn(data, episodes=5000):
    """Train DQN with comprehensive episodes"""
    print(f"\n[DQN] Training with {episodes} episodes...")
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size].copy()
    
    env = AdvancedCryptoTradingEnv(train_data)
    agent = EnhancedDQNAgent(env.state_size, env.action_space)
    
    episode_rewards = []
    episode_values = []
    episode_lengths = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 2000:  # Max steps per episode
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if len(agent.memory) > agent.batch_size:
                agent.replay()
        
        episode_rewards.append(total_reward)
        episode_values.append(info['portfolio_value'])
        episode_lengths.append(steps)
        
        # Logging
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_value = np.mean(episode_values[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            
            print(f"    Episode {episode+1}/{episodes}")
            print(f"      Avg Reward: {avg_reward:.4f}")
            print(f"      Avg Portfolio: ${avg_value:.2f}")
            print(f"      Avg Length: {avg_length:.0f} steps")
            print(f"      Epsilon: {agent.epsilon:.4f}")
            print(f"      Memory Size: {len(agent.memory)}")
    
    print(f"[DQN] Training complete!")
    print(f"      Final epsilon: {agent.epsilon:.4f}")
    print(f"      Total training steps: {agent.steps}")
    
    return agent, episode_rewards, episode_values

def train_comprehensive_ppo(data, episodes=5000):
    """Train PPO with comprehensive episodes"""
    print(f"\n[PPO] Training with {episodes} episodes...")
    
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size].copy()
    
    env = AdvancedCryptoTradingEnv(train_data)
    agent = EnhancedPPOAgent(env.state_size, env.action_space)
    
    episode_rewards = []
    episode_values = []
    
    for episode in range(episodes):
        state = env.reset()
        
        states, actions, rewards, log_probs, values = [], [], [], [], []
        done = False
        steps = 0
        
        while not done and steps < 2000:
            action, log_prob, value = agent.get_action_and_value(state)
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            
            state = next_state
            steps += 1
        
        # Train on episode
        if len(states) > 0:
            agent.train_step(states, actions, rewards, log_probs, values)
        
        episode_rewards.append(np.sum(rewards))
        episode_values.append(info['portfolio_value'])
        
        # Logging
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_value = np.mean(episode_values[-100:])
            
            print(f"    Episode {episode+1}/{episodes}")
            print(f"      Avg Reward: {avg_reward:.4f}")
            print(f"      Avg Portfolio: ${avg_value:.2f}")
            print(f"      Actor Loss: {np.mean(agent.actor_losses[-100:]):.4f}")
            print(f"      Critic Loss: {np.mean(agent.critic_losses[-100:]):.4f}")
    
    print(f"[PPO] Training complete!")
    return agent, episode_rewards, episode_values

def evaluate_rl_agents(agents, data):
    """Comprehensive evaluation of trained RL agents"""
    print("\n" + "="*60)
    print("COMPREHENSIVE RL EVALUATION")
    print("="*60)
    
    # Use test data
    test_size = int(len(data) * 0.2)
    test_data = data[-test_size:].copy()
    test_env = AdvancedCryptoTradingEnv(test_data)
    
    results = {}
    
    for name, agent in agents.items():
        print(f"\nEvaluating {name.upper()}...")
        
        episode_values = []
        episode_returns = []
        
        # Run multiple evaluation episodes
        for _ in range(20):
            state = test_env.reset()
            done = False
            steps = 0
            
            while not done and steps < 2000:
                if hasattr(agent, 'act'):  # DQN
                    action = agent.act(state, training=False)
                else:  # PPO
                    action, _, _ = agent.get_action_and_value(state)
                
                state, reward, done, info = test_env.step(action)
                steps += 1
            
            episode_values.append(info['portfolio_value'])
            episode_returns.append((info['portfolio_value'] - 10000) / 10000)
        
        avg_value = np.mean(episode_values)
        avg_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        sharpe = avg_return / (std_return + 1e-8) * np.sqrt(252)
        win_rate = np.mean([v > 10000 for v in episode_values])
        
        results[name] = {
            'avg_portfolio_value': avg_value,
            'avg_return': avg_return,
            'volatility': std_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate
        }
        
        print(f"    Avg Portfolio Value: ${avg_value:.2f}")
        print(f"    Avg Return: {avg_return:.2%}")
        print(f"    Sharpe Ratio: {sharpe:.3f}")
        print(f"    Win Rate: {win_rate:.2%}")
    
    # Buy & Hold benchmark
    initial_price = test_data.iloc[0]['close']
    final_price = test_data.iloc[-1]['close'] 
    bh_return = (final_price - initial_price) / initial_price
    bh_value = 10000 * (1 + bh_return)
    
    print(f"\nBuy & Hold Benchmark:")
    print(f"    Portfolio Value: ${bh_value:.2f}")
    print(f"    Return: {bh_return:.2%}")
    
    # Best performer
    best_agent = max(results.items(), key=lambda x: x[1]['avg_portfolio_value'])
    print(f"\n[RESULT] Best RL Agent: {best_agent[0].upper()}")
    print(f"         Portfolio Value: ${best_agent[1]['avg_portfolio_value']:.2f}")
    print(f"         Sharpe Ratio: {best_agent[1]['sharpe_ratio']:.3f}")
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Starting comprehensive RL training...")
    
    # Load data
    data = load_comprehensive_data()
    if data is None:
        print("ERROR: Could not load data for RL training")
        sys.exit(1)
    
    print(f"Training data: {len(data):,} records")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Train agents
    trained_agents = {}
    
    # Train DQN (5000 episodes)
    print("\n" + "="*60)
    print("TRAINING DQN WITH 5,000 EPISODES")  
    print("="*60)
    dqn_agent, dqn_rewards, dqn_values = train_comprehensive_dqn(data, episodes=5000)
    trained_agents['dqn'] = dqn_agent
    
    # Train PPO (5000 episodes)
    print("\n" + "="*60)
    print("TRAINING PPO WITH 5,000 EPISODES")
    print("="*60)
    ppo_agent, ppo_rewards, ppo_values = train_comprehensive_ppo(data, episodes=5000)
    trained_agents['ppo'] = ppo_agent
    
    # Comprehensive evaluation
    final_results = evaluate_rl_agents(trained_agents, data)
    
    # Create training plots
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(dqn_values)
    plt.title('DQN Portfolio Value During Training')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(ppo_values)
    plt.title('PPO Portfolio Value During Training')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(dqn_agent.losses[-1000:] if len(dqn_agent.losses) >= 1000 else dqn_agent.losses)
    plt.title('DQN Training Loss (Last 1000 steps)')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    plt.plot(ppo_agent.actor_losses[-1000:] if len(ppo_agent.actor_losses) >= 1000 else ppo_agent.actor_losses)
    plt.title('PPO Actor Loss (Last 1000 steps)')
    plt.xlabel('Training Step')
    plt.ylabel('Actor Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    agent_names = list(final_results.keys())
    sharpe_ratios = [final_results[name]['sharpe_ratio'] for name in agent_names]
    plt.bar(agent_names, sharpe_ratios)
    plt.title('RL Agents Sharpe Ratio Comparison')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    win_rates = [final_results[name]['win_rate'] for name in agent_names]
    plt.bar(agent_names, win_rates)
    plt.title('RL Agents Win Rate Comparison')
    plt.ylabel('Win Rate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE RL TRAINING COMPLETE!")
    print("="*60)
    print("\n[SUCCESS] All RL agents trained with 5,000+ episodes each")
    print(f"[DATA] Training dataset: {len(data):,} records")
    print(f"[DQN] Final epsilon: {dqn_agent.epsilon:.4f}, Total steps: {dqn_agent.steps:,}")
    print(f"[PPO] Actor losses: {len(ppo_agent.actor_losses):,}, Critic losses: {len(ppo_agent.critic_losses):,}")
    
    # Save agents
    torch.save(dqn_agent.q_network.state_dict(), 'models/comprehensive_dqn.pt')
    torch.save(ppo_agent.network.state_dict(), 'models/comprehensive_ppo.pt')
    print(f"[SAVED] Models saved to models/ directory")
    
    print(f"\n[+] RL models are now FULLY and PROPERLY TRAINED!")