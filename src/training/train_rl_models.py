#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

print("Loading RL libraries...")

# ============================================================================
# TRADING ENVIRONMENT
# ============================================================================

class CryptoTradingEnvironment:
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000, 
                 lookback: int = 20, commission: float = 0.001):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.lookback = lookback
        self.commission = commission
        
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # BTC holdings
        self.trades = []
        
        self.action_space = 3
        
        self.state_size = lookback * 4 + 3
        
    def reset(self):
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        return self._get_state()
    
    def _get_state(self):
        start = self.current_step - self.lookback
        price_data = self.data.iloc[start:self.current_step]
        
        current_price = self.data.iloc[self.current_step]['close']
        norm_prices = []
        
        for _, row in price_data.iterrows():
            norm_prices.extend([
                row['open'] / current_price,
                row['high'] / current_price,
                row['low'] / current_price,
                row['close'] / current_price
            ])
        
        # Add position information
        total_value = self.balance + self.position * current_price
        position_info = [
            self.balance / self.initial_balance,
            self.position * current_price / self.initial_balance,
            total_value / self.initial_balance
        ]
        
        return np.array(norm_prices + position_info, dtype=np.float32)
    
    def step(self, action: int):
        current_price = self.data.iloc[self.current_step]['close']
        prev_value = self.balance + self.position * current_price
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > 0:
                buy_amount = self.balance * 0.95
                self.position += buy_amount / current_price * (1 - self.commission)
                self.balance -= buy_amount
                self.trades.append(('BUY', current_price, self.current_step))
                
        elif action == 2:
            if self.position > 0:
                sell_value = self.position * current_price * (1 - self.commission)
                self.balance += sell_value
                self.position = 0
                self.trades.append(('SELL', current_price, self.current_step))
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Calculate reward
        new_price = self.data.iloc[self.current_step]['close'] if not done else current_price
        new_value = self.balance + self.position * new_price
        
        # Reward = portfolio value change
        reward = (new_value - prev_value) / self.initial_balance
        
        next_state = self._get_state() if not done else None
        
        return next_state, reward, done, {'value': new_value}

class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.learning_rate = lr
        
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.update_target_network()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.detach().numpy())
    
    def replay(self, batch_size: int = 32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] if e[3] is not None else np.zeros(self.state_size) for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn_agent(env, episodes: int = 100):
    agent = DQNAgent(env.state_size, env.action_space)
    
    scores = []
    epsilons = []
    
    print("\nTraining DQN Agent...")
    print("-" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > 32:
                agent.replay(32)
        
        scores.append(info['value'])
        epsilons.append(agent.epsilon)
        
        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_network()
        
        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
            print(f"Episode {episode}/{episodes}, Avg Value: ${avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, scores


class PolicyNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        
        return action_probs, state_value

class SimplePPO:
    def __init__(self, state_size: int, action_size: int, lr: float = 0.0003):
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.policy(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def train_step(self, states, actions, rewards, log_probs):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        old_log_probs = torch.stack(log_probs)
        
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Get current policy outputs
        action_probs, values = self.policy(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        
        # Policy loss (simplified)
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        policy_loss = -torch.min(ratio * rewards, 
                                 torch.clamp(ratio, 0.8, 1.2) * rewards).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), rewards)
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def load_data():
    print("Loading data...")
    
    # Try to load real data
    try:
        import glob
        files = glob.glob('data/s3_cache/crypto_*.parquet')
        if files:
            df = pd.read_parquet(files[0])
            # Filter for BTC
            btc_tickers = [t for t in df['ticker'].unique() if 'BTC' in t]
            if btc_tickers:
                btc_data = df[df['ticker'] == btc_tickers[0]].copy()
                btc_data = btc_data[['open', 'high', 'low', 'close', 'volume']]
                print(f"Loaded {len(btc_data)} BTC records")
                return btc_data[:5000]  # Use subset for faster training
    except:
        pass
    
    # Generate synthetic data if real data not available
    print("Using synthetic data...")
    np.random.seed(42)
    n_points = 5000
    returns = np.random.normal(0.001, 0.02, n_points)
    prices = 30000 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'close': prices,
        'volume': np.random.lognormal(20, 1, n_points)
    })
    
    return data

def evaluate_agent(agent, env, episodes: int = 10):
    total_rewards = []
    final_values = []
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if hasattr(agent, 'act'):  # DQN
                action = agent.act(state, training=False)
            else:  # PPO
                action, _ = agent.get_action(state)
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        final_values.append(info['value'])
    
    return np.mean(total_rewards), np.mean(final_values)

if __name__ == "__main__":
    print("="*60)
    print("REINFORCEMENT LEARNING TRADING AGENTS")
    print("="*60)
    
    # Load data
    data = load_data()
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size].copy()
    test_data = data[train_size:].copy()
    
    print(f"\nData split: Train={len(train_data)}, Test={len(test_data)}")
    
    # Create environments
    train_env = CryptoTradingEnvironment(train_data)
    test_env = CryptoTradingEnvironment(test_data)
    
    # Train DQN
    print("\n" + "="*60)
    print("TRAINING DQN AGENT")
    print("="*60)
    
    dqn_agent, dqn_scores = train_dqn_agent(train_env, episodes=50)
    
    # Evaluate DQN
    dqn_reward, dqn_value = evaluate_agent(dqn_agent, test_env)
    print(f"\nDQN Test Performance:")
    print(f"  Average Reward: {dqn_reward:.4f}")
    print(f"  Final Portfolio Value: ${dqn_value:.2f}")
    
    # Simple PPO Training
    print("\n" + "="*60)
    print("TRAINING PPO AGENT")
    print("="*60)
    
    ppo_agent = SimplePPO(train_env.state_size, train_env.action_space)
    
    for episode in range(50):
        state = train_env.reset()
        states, actions, rewards, log_probs = [], [], [], []
        done = False
        
        while not done:
            action, log_prob = ppo_agent.get_action(state)
            next_state, reward, done, _ = train_env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = next_state
        
        if len(states) > 0:
            ppo_agent.train_step(states, actions, rewards, log_probs)
        
        if episode % 10 == 0:
            print(f"PPO Episode {episode}/50")
    
    # Evaluate PPO
    ppo_reward, ppo_value = evaluate_agent(ppo_agent, test_env)
    print(f"\nPPO Test Performance:")
    print(f"  Average Reward: {ppo_reward:.4f}")
    print(f"  Final Portfolio Value: ${ppo_value:.2f}")
    
    # Buy & Hold Baseline
    initial_value = test_env.initial_balance
    final_price = test_data.iloc[-1]['close']
    initial_price = test_data.iloc[0]['close']
    bh_value = initial_value * (final_price / initial_price)
    
    print(f"\nBuy & Hold Baseline:")
    print(f"  Final Portfolio Value: ${bh_value:.2f}")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    results = {
        'DQN': {'reward': dqn_reward, 'value': dqn_value},
        'PPO': {'reward': ppo_reward, 'value': ppo_value},
        'Buy & Hold': {'reward': 0, 'value': bh_value}
    }
    
    best_agent = max(results.items(), key=lambda x: x[1]['value'])
    
    print(f"\nBest Performing Strategy: {best_agent[0]}")
    print(f"Portfolio Value: ${best_agent[1]['value']:.2f}")
    
    # Plot training progress
    if len(dqn_scores) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(dqn_scores)
        plt.title('DQN Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Portfolio Value')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    print("\n[+] RL Training Complete!")
