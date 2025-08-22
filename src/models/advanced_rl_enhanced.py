import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any
import random
import copy


class ReplayBuffer:
    
    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)
        self.Experience = namedtuple('Experience', 
            ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, *args):
        self.buffer.append(self.Experience(*args))
    
    def sample(self, batch_size: int):
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.FloatTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class NoisyLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor):
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 n_atoms: int = 51,
                 v_min: float = -10,
                 v_max: float = 10,
                 hidden_dim: int = 512,
                 n_layers: int = 3):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        self.support = torch.linspace(v_min, v_max, n_atoms)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        layers = []
        current_dim = state_dim
        
        for i in range(n_layers - 1):
            layers.append(NoisyLinear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        self.advantage_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.advantage_out = NoisyLinear(hidden_dim, action_dim * n_atoms)
        
        self.value_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.value_out = NoisyLinear(hidden_dim, n_atoms)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(x)
        
        advantage = F.relu(self.advantage_hidden(features))
        advantage = self.advantage_out(advantage).view(-1, self.action_dim, self.n_atoms)
        
        value = F.relu(self.value_hidden(features))
        value = self.value_out(value).view(-1, 1, self.n_atoms)
        
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_dist, dim=-1)
        
        return q_dist
    
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def act(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            q_dist = self(state)
            q_values = (q_dist * self.support.to(state.device)).sum(dim=-1)
            return q_values.argmax(dim=-1).item()


class Actor(nn.Module):
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super().__init__()
        
        self.max_action = max_action
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)


class Critic(nn.Module):
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
    
    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class TD3Agent:
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float = 1.0,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_freq: int = 2,
                 device: str = 'cuda'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.total_it = 0
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.max_action * 0.1, size=action.shape)
            action = (action + noise).clip(-self.max_action, self.max_action)
        
        return action
    
    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        self.total_it += 1
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q
        
        current_Q1, current_Q2 = self.critic(states, actions)
        
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class SACAgent:
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float = 1.0,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 automatic_entropy_tuning: bool = True,
                 device: str = 'cuda'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.actor = GaussianActor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        
        self.automatic_entropy_tuning = automatic_entropy_tuning
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        
        return action.detach().cpu().numpy()[0]
    
    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_states)
            qf1_next_target, qf2_next_target = self.critic_target(next_states, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + (1 - dones) * self.gamma * min_qf_next_target
        
        qf1, qf2 = self.critic(states, actions)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        critic_loss = qf1_loss + qf2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        pi, log_pi, _ = self.actor.sample(states)
        
        qf1_pi, qf2_pi = self.critic(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            self.alpha = self.log_alpha.exp()
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class GaussianActor(nn.Module):
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.max_action
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * self.max_action
        
        return action, log_prob, mean


class PPOMemory:
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, logprob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def get_batches(self, batch_size: int):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]
        
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        logprobs = torch.FloatTensor(np.array(self.logprobs))
        rewards = torch.FloatTensor(np.array(self.rewards))
        values = torch.FloatTensor(np.array(self.values))
        dones = torch.FloatTensor(np.array(self.dones))
        
        return states, actions, logprobs, rewards, values, dones, batches


class PPOActorCritic(nn.Module):
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_net(state)
        value = self.critic(features)
        return value
    
    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared_net(state)
        
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        value = self.critic(features)
        
        return action, action_logprob, value
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared_net(state)
        
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        action_logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        value = self.critic(features)
        
        return action_logprob, value.squeeze(-1), entropy


class PPOAgent:
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 eps_clip: float = 0.2,
                 c1: float = 0.5,
                 c2: float = 0.01,
                 epochs: int = 10,
                 batch_size: int = 64,
                 device: str = 'cuda'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.policy = PPOActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.c1 = c1
        self.c2 = c2
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.memory = PPOMemory()
    
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            action, logprob, value = self.policy.act(state)
        
        return action.cpu().numpy()[0], logprob.cpu().item(), value.cpu().item()
    
    def train(self):
        states, actions, logprobs, rewards, values, dones, batches = self.memory.get_batches(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        logprobs = logprobs.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        
        advantages = torch.zeros_like(rewards).to(self.device)
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            if t == len(rewards) - 1:
                advantages[t] = delta
            else:
                advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * advantages[t + 1]
        
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.epochs):
            for batch in batches:
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_logprobs = logprobs[batch]
                batch_returns = returns[batch]
                batch_advantages = advantages[batch]
                
                new_logprobs, new_values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                ratio = torch.exp(new_logprobs - batch_logprobs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_values, batch_returns)
                entropy_loss = -entropy.mean()
                
                loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        self.memory.clear()


def create_rl_agent(
    agent_type: str,
    state_dim: int,
    action_dim: int,
    max_action: float = 1.0,
    device: str = 'cuda'
) -> Any:
    
    if agent_type.lower() == 'td3':
        return TD3Agent(state_dim, action_dim, max_action, device=device)
    elif agent_type.lower() == 'sac':
        return SACAgent(state_dim, action_dim, max_action, device=device)
    elif agent_type.lower() == 'ppo':
        return PPOAgent(state_dim, action_dim, device=device)
    elif agent_type.lower() == 'rainbow':
        return RainbowDQN(state_dim, action_dim)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")