#!/usr/bin/env python3
"""
FOCUSED RL TRAINING
"""

import sys
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import glob
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("FOCUSED RL TRAINING - 1000 EPISODES")
print("=" * 60)

# Simplified but effective RL implementation
class SimpleTradingEnv:
    def __init__(self, data, initial_balance=10000):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.action_space = 3  # Hold, Buy, Sell
        self.state_size = 23  # Last 20 price ratios + 3 portfolio info
        self.reset()
    
    def reset(self):
        self.current_step = 20
        self.balance = self.initial_balance
        self.position = 0
        self.history = [self.initial_balance]
        return self._get_state()
    
    def _get_state(self):
        # Price ratios for last 20 steps
        end_idx = min(self.current_step, len(self.data) - 1)
        start_idx = max(0, end_idx - 19)
        
        prices = self.data.iloc[start_idx:end_idx+1]['close'].values
        if len(prices) < 20:
            prices = np.pad(prices, (20-len(prices), 0), 'edge')
        
        # Normalize prices to current price
        current_price = prices[-1]
        ratios = prices / current_price
        
        # Add portfolio info
        portfolio_value = self.balance + self.position * current_price
        portfolio_ratio = portfolio_value / self.initial_balance
        position_ratio = (self.position * current_price) / self.initial_balance
        balance_ratio = self.balance / self.initial_balance
        
        state = np.concatenate([ratios, [portfolio_ratio, position_ratio, balance_ratio]])
        
        # Ensure exactly 20 + 3 = 23 features
        if len(state) != 23:
            state = np.pad(state, (0, 23 - len(state)), 'constant', constant_values=1.0)[:23]
        
        return state.astype(np.float32)
    
    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            current_price = self.data.iloc[-1]['close']
            portfolio_value = self.balance + self.position * current_price
            return self._get_state(), 0, True, {'portfolio_value': portfolio_value}
        
        current_price = self.data.iloc[self.current_step]['close']
        prev_value = self.balance + self.position * current_price
        
        # Execute action
        if action == 1 and self.balance > 100:  # Buy
            shares = self.balance * 0.95 / current_price
            self.position += shares
            self.balance = self.balance * 0.05
        elif action == 2 and self.position > 0:  # Sell
            self.balance += self.position * current_price * 0.999  # Small commission
            self.position = 0
        
        # Move forward
        self.current_step += 1
        new_price = self.data.iloc[min(self.current_step, len(self.data)-1)]['close']
        new_value = self.balance + self.position * new_price
        
        # Simple reward: portfolio value change
        reward = (new_value - prev_value) / prev_value if prev_value > 0 else 0
        
        self.history.append(new_value)
        done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, done, {'portfolio_value': new_value}

class SimpleDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class SimpleDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = SimpleDQN(state_size, action_size).to(self.device)
        self.target_network = SimpleDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        self.update_target_network()
        self.losses = []
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def load_sample_data():
    files = glob.glob('data/s3_cache/crypto_*.parquet')
    
    if files:
        # Load a reasonable sample
        all_data = []
        for file in sorted(files)[:20]:  # Use 20 files
            try:
                df = pd.read_parquet(file)
                btc_tickers = [t for t in df['ticker'].unique() if 'BTC' in t.upper()]
                if btc_tickers:
                    btc_df = df[df['ticker'] == btc_tickers[0]].copy()
                    if not btc_df.empty:
                        all_data.append(btc_df)
            except:
                continue
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            if 'window_start' in combined.columns:
                combined['timestamp'] = pd.to_datetime(combined['window_start'])
            
            combined = combined.sort_values('timestamp').drop_duplicates()
            return combined[['open', 'high', 'low', 'close', 'volume']].head(5000)
    
    # Fallback synthetic data
    print("Using synthetic data...")
    np.random.seed(42)
    n_points = 5000
    returns = np.random.normal(0.001, 0.02, n_points)
    prices = 50000 * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'close': prices,
        'volume': np.random.lognormal(20, 1, n_points)
    })

def train_focused_dqn(data, episodes=1000):
    print(f"\nTraining DQN for {episodes} episodes...")
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size].copy()
    
    env = SimpleTradingEnv(train_data)
    agent = SimpleDQNAgent(env.state_size, env.action_space)
    
    episode_rewards = []
    episode_values = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 1000:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            if len(agent.memory) > 32:
                agent.replay(32)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_values.append(info['portfolio_value'])
        
        # Update target network
        if episode % 100 == 0:
            agent.update_target_network()
        
        # Progress
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_value = np.mean(episode_values[-50:])
            print(f"    Episode {episode+1}: Avg Reward={avg_reward:.4f}, Avg Value=${avg_value:.2f}, ε={agent.epsilon:.3f}")
    
    return agent, episode_rewards, episode_values

def evaluate_agents(agents, data):
    print("\nEvaluating trained agents...")
    
    test_size = int(len(data) * 0.2)
    test_data = data[-test_size:].copy()
    
    results = {}
    
    for name, agent in agents.items():
        test_env = SimpleTradingEnv(test_data)
        
        # Run 10 evaluation episodes
        values = []
        for _ in range(10):
            state = test_env.reset()
            done = False
            
            while not done:
                action = agent.act(state, training=False)
                state, reward, done, info = test_env.step(action)
            
            values.append(info['portfolio_value'])
        
        avg_value = np.mean(values)
        results[name] = {
            'avg_portfolio_value': avg_value,
            'return_pct': (avg_value - 10000) / 10000 * 100,
            'win_rate': np.mean([v > 10000 for v in values])
        }
        
        print(f"  {name.upper()}: ${avg_value:.2f} ({results[name]['return_pct']:.1f}% return)")
    
    # Buy & Hold benchmark
    initial_price = test_data.iloc[0]['close']
    final_price = test_data.iloc[-1]['close']
    bh_return = (final_price - initial_price) / initial_price
    bh_value = 10000 * (1 + bh_return)
    
    print(f"  BUY & HOLD: ${bh_value:.2f} ({bh_return*100:.1f}% return)")
    
    return results

if __name__ == "__main__":
    # Load data
    data = load_sample_data()
    print(f"Loaded {len(data):,} records")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Train agents
    agents = {}
    
    # Train DQN
    print("\n" + "="*50)
    print("TRAINING DQN (1000 episodes)")
    print("="*50)
    dqn_agent, dqn_rewards, dqn_values = train_focused_dqn(data, episodes=1000)
    agents['dqn'] = dqn_agent
    
    print(f"\nDQN Training Complete!")
    print(f"  Final ε: {dqn_agent.epsilon:.4f}")
    print(f"  Memory size: {len(dqn_agent.memory):,}")
    print(f"  Training losses: {len(dqn_agent.losses):,}")
    
    # Evaluate
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    results = evaluate_agents(agents, data)
    
    # Create plots
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(dqn_values)
    plt.title('DQN Portfolio Value During Training')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(dqn_rewards)
    plt.title('DQN Rewards During Training')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    if len(dqn_agent.losses) > 0:
        plt.plot(dqn_agent.losses[-500:])
        plt.title('DQN Training Loss (Last 500 steps)')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(data['close'])
    plt.title('BTC Price During Training Period')
    plt.xlabel('Time Steps')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('focused_rl_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*50)
    print("FOCUSED RL TRAINING COMPLETE!")
    print("="*50)
    print(f"\n[SUCCESS] DQN trained with {len(dqn_values)} episodes")
    print(f"[PERFORMANCE] Best portfolio value: ${max(dqn_values):.2f}")
    print(f"[LEARNING] Final epsilon: {dqn_agent.epsilon:.4f}")
    print(f"[CONVERGENCE] Training losses tracked: {len(dqn_agent.losses):,}")
    
    # Save model
    torch.save(dqn_agent.q_network.state_dict(), 'models/focused_dqn.pt')
    print(f"[SAVED] Model saved to models/focused_dqn.pt")
    
    print(f"\n[+] RL model now properly trained with sufficient episodes!") 