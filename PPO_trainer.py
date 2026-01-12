"""
Proximal Policy Optimization (PPO) implementation for drone navigation.
Includes LSTM for temporal dependencies and curriculum learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PPOTrainer:
    """Main PPO trainer class with LSTM and curriculum learning."""
    
    def __init__(self, config: dict, device: str = 'cuda'):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # Network architecture
        self.hidden_size = config.get('hidden_size', 256)
        self.lstm_layers = config.get('lstm_layers', 2)
        
        # Training buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []
        
        # Curriculum learning
        self.curriculum_stages = [
            {'helicopter_speed': 15.0, 'weather': 'dry'},
            {'helicopter_speed': 25.0, 'weather': 'dry'},
            {'helicopter_speed': 15.0, 'weather': 'rain', 'rain_intensity': 8.0},
            {'helicopter_speed': 25.0, 'weather': 'rain', 'rain_intensity': 8.0}
        ]
        self.current_stage = 0
        self.stage_threshold = 0.8  # 80% success rate to advance
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
        self.best_success_rate = 0.0
        
        # Initialize networks
        self._init_networks()
        
        logger.info(f"PPO trainer initialized on {self.device}")
    
    def _init_networks(self):
        """Initialize actor and critic networks."""
        # Actor network (policy)
        self.actor = LSTMActor(
            state_dim=10,
            action_dim=4,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers
        ).to(self.device)
        
        # Critic network (value function)
        self.critic = LSTMCritic(
            state_dim=10,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=self.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=self.learning_rate
        )
    
    def collect_trajectory(self, env, num_steps: int = 2048):
        """Collect experience by interacting with environment."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.log_probs.clear()
        
        state = env.reset()
        actor_hidden = None
        critic_hidden = None
        
        for step in range(num_steps):
            # Convert state to tensor
            state_tensor = self._state_to_tensor(state)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, actor_hidden = self.actor(
                    state_tensor, 
                    actor_hidden
                )
                value, critic_hidden = self.critic(
                    state_tensor, 
                    critic_hidden
                )
            
            # Take action in environment
            next_state, reward, done, info = env.step(action.cpu().numpy())
            
            # Store transition
            self.states.append(state)
            self.actions.append(action.cpu().numpy())
            self.rewards.append(reward)
            self.values.append(value.cpu().numpy())
            self.dones.append(done)
            self.log_probs.append(log_prob.cpu().numpy())
            
            # Update state
            state = next_state
            
            # Reset hidden states if episode done
            if done:
                state = env.reset()
                actor_hidden = None
                critic_hidden = None
        
        # Calculate advantages
        advantages = self._compute_advantages()
        
        return advantages
    
    def update_policy(self, advantages: np.ndarray):
        """Update policy networks using PPO loss."""
        # Convert data to tensors
        states_tensor = self._batch_to_tensor(self.states)
        actions_tensor = torch.tensor(
            np.array(self.actions), 
            device=self.device, 
            dtype=torch.float32
        )
        old_log_probs = torch.tensor(
            np.array(self.log_probs), 
            device=self.device, 
            dtype=torch.float32
        )
        advantages_tensor = torch.tensor(
            advantages, 
            device=self.device, 
            dtype=torch.float32
        )
        returns_tensor = advantages_tensor + torch.tensor(
            np.array(self.values), 
            device=self.device, 
            dtype=torch.float32
        ).squeeze()
        
        # PPO epochs
        num_epochs = self.config.get('ppo_epochs', 10)
        batch_size = self.config.get('batch_size', 64)
        
        for epoch in range(num_epochs):
            # Shuffle indices for mini-batch training
            indices = np.random.permutation(len(self.states))
            
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                
                # Get batch data
                batch_states = states_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]
                
                # Forward pass
                new_action_dist, _ = self.actor(batch_states)
                new_log_probs = new_action_dist.log_prob(batch_actions).sum(-1)
                entropy = new_action_dist.entropy().mean()
                
                # Calculate ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.clip_epsilon, 
                    1 + self.clip_epsilon
                ) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values, _ = self.critic(batch_states)
                value_loss = nn.MSELoss()(
                    values.squeeze(), 
                    batch_returns
                )
                
                # Total loss
                total_loss = (
                    actor_loss + 
                    self.value_coef * value_loss - 
                    self.entropy_coef * entropy
                )
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 
                    max_norm=0.5
                )
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), 
                    max_norm=0.5
                )
                self.critic_optimizer.step()
        
        # Clear buffers after update
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.log_probs.clear()
    
    def _compute_advantages(self) -> np.ndarray:
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = []
        gae = 0.0
        
        # Convert to numpy arrays
        rewards = np.array(self.rewards)
        values = np.array(self.values).squeeze()
        dones = np.array(self.dones)
        
        # Calculate advantages in reverse
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_nonterminal = 1.0 - dones[t]
                next_value = values[t] if dones[t] else 0.0
            else:
                next_nonterminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            advantages.insert(0, gae)
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def _state_to_tensor(self, state: dict) -> torch.Tensor:
        """Convert state dictionary to tensor."""
        # Use state vector for now (can be extended with camera/radar)
        state_vector = torch.tensor(
            state['state'], 
            device=self.device, 
            dtype=torch.float32
        ).unsqueeze(0)
        return state_vector
    
    def _batch_to_tensor(self, states: list) -> torch.Tensor:
        """Convert batch of states to tensor."""
        state_vectors = [s['state'] for s in states]
        return torch.tensor(
            np.array(state_vectors), 
            device=self.device, 
            dtype=torch.float32
        )
    
    def update_curriculum(self, success_rate: float):
        """Update curriculum stage based on performance."""
        if (success_rate >= self.stage_threshold and 
            self.current_stage < len(self.curriculum_stages) - 1):
            
            self.current_stage += 1
            logger.info(f"Advancing to curriculum stage {self.current_stage + 1}")
            
            return self.curriculum_stages[self.current_stage]
        
        return None
    
    def save_checkpoint(self, path: str, episode: int):
        """Save model checkpoint."""
        checkpoint = {
            'episode': episode,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'current_stage': self.current_stage,
            'best_success_rate': self.best_success_rate,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.current_stage = checkpoint['current_stage']
        self.best_success_rate = checkpoint['best_success_rate']
        
        logger.info(f"Checkpoint loaded from {path}")
        logger.info(f"Resuming from stage {self.current_stage + 1}")

class LSTMActor(nn.Module):
    """Actor network with LSTM for temporal dependencies."""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_size: int = 256, lstm_layers: int = 2):
        super().__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Policy head
        self.policy_mean = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Log standard deviation (learnable parameter)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Hidden state storage
        self.hidden_state = None
    
    def forward(self, x, hidden_state=None):
        """Forward pass through actor network."""
        batch_size = x.size(0)
        
        # Reshape for LSTM if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        
        # Use last timestep output
        lstm_features = lstm_out[:, -1, :]
        
        # Policy mean
        mean = self.policy_mean(lstm_features)
        
        # Create distribution
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        
        # Sample action
        action = dist.rsample()  # reparameterized sample
        
        # Calculate log probability
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        # Clip action to valid range
        action = torch.tanh(action)
        
        return action, log_prob, new_hidden

class LSTMCritic(nn.Module):
    """Critic network with LSTM for value estimation."""
    
    def __init__(self, state_dim: int, hidden_size: int = 256, 
                 lstm_layers: int = 2):
        super().__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, hidden_state=None):
        """Forward pass through critic network."""
        batch_size = x.size(0)
        
        # Reshape for LSTM if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        
        # Use last timestep output
        lstm_features = lstm_out[:, -1, :]
        
        # Value estimate
        value = self.value_head(lstm_features)
        
        return value, new_hidden