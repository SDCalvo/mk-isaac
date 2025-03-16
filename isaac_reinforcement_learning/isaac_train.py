#!/usr/bin/env python
"""
Isaac Reinforcement Learning Training Script

This script trains a reinforcement learning agent to play The Binding of Isaac
using the IsaacEnv environment wrapper.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from collections import deque
import random
from isaac_gym_env import IsaacEnv

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQNNetwork(nn.Module):
    """
    DQN network for learning to play Isaac
    
    The network takes in state observations and outputs Q-values for each action.
    """
    def __init__(self, observation_space, action_space):
        super(DQNNetwork, self).__init__()
        
        # Calculate input size from observation space
        self.input_size = (
            1 +  # health
            1 +  # max_health
            1 +  # soul_hearts
            1 +  # damage
            1 +  # fire_rate
            1 +  # speed
            1 +  # stage
            1 +  # room_cleared
            1 +  # enemies
            20   # enemy_positions (10 enemies * 2 coordinates)
        )
        
        # Number of actions
        self.action_size = action_space.n
        
        # Network layers
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.action_size)
        
        # Activation functions
        self.relu = nn.ReLU()
    
    def forward(self, state_dict):
        """
        Forward pass through the network
        
        Args:
            state_dict: Dictionary of state observations
            
        Returns:
            Q-values for each action
        """
        # Convert dictionary of observations to flat tensor
        x = torch.cat([
            state_dict['health'],
            state_dict['max_health'],
            state_dict['soul_hearts'],
            state_dict['damage'],
            state_dict['fire_rate'],
            state_dict['speed'],
            state_dict['stage'].float(),
            torch.tensor([[float(state_dict['room_cleared'])]], device=device),
            state_dict['enemies'].float(),
            state_dict['enemy_positions'].flatten().unsqueeze(0)
        ], dim=1)
        
        # Forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        q_values = self.fc4(x)
        
        return q_values

class ReplayBuffer:
    """
    Replay buffer for storing experiences
    
    The buffer stores transitions (state, action, reward, next_state, done)
    and allows for random sampling for training.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        # Create empty batch for each observation
        states = {
            'health': [],
            'max_health': [],
            'soul_hearts': [],
            'damage': [],
            'fire_rate': [],
            'speed': [],
            'stage': [],
            'room_cleared': [],
            'enemies': [],
            'enemy_positions': []
        }
        
        actions = []
        rewards = []
        
        next_states = {
            'health': [],
            'max_health': [],
            'soul_hearts': [],
            'damage': [],
            'fire_rate': [],
            'speed': [],
            'stage': [],
            'room_cleared': [],
            'enemies': [],
            'enemy_positions': []
        }
        
        dones = []
        
        # Fill batches
        for experience in batch:
            state, action, reward, next_state, done = experience
            
            # Add state observations to batch
            for key in states:
                states[key].append(state[key])
            
            actions.append(action)
            rewards.append(reward)
            
            # Add next_state observations to batch
            for key in next_states:
                next_states[key].append(next_state[key])
                
            dones.append(done)
        
        # Convert to tensors
        for key in states:
            # Special case for room_cleared which is a discrete space
            if key == 'room_cleared':
                states[key] = torch.tensor(states[key], device=device).float().unsqueeze(1)
                next_states[key] = torch.tensor(next_states[key], device=device).float().unsqueeze(1)
            else:
                states[key] = torch.tensor(np.array(states[key]), device=device).float()
                next_states[key] = torch.tensor(np.array(next_states[key]), device=device).float()
        
        actions = torch.tensor(actions, device=device).long().unsqueeze(1)
        rewards = torch.tensor(rewards, device=device).float().unsqueeze(1)
        dones = torch.tensor(dones, device=device).float().unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return buffer size"""
        return len(self.buffer)

class DQNAgent:
    """
    DQN Agent for playing Isaac
    
    The agent implements the Deep Q-Network algorithm for reinforcement learning.
    """
    def __init__(self, env, model_dir="models", log_dir="logs"):
        self.env = env
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Initialize networks
        self.policy_net = DQNNetwork(env.observation_space, env.action_space).to(device)
        self.target_net = DQNNetwork(env.observation_space, env.action_space).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.learning_starts = 1000
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state observation
            training: Whether we're in training mode (use epsilon-greedy)
            
        Returns:
            Selected action
        """
        if training and (random.random() < self.epsilon or len(self.replay_buffer) < self.learning_starts):
            # Exploration: select random action
            return self.env.action_space.sample()
        else:
            # Exploitation: select best action
            with torch.no_grad():
                # Convert state to tensors
                state_tensors = {}
                for key in state:
                    if key == 'room_cleared':
                        state_tensors[key] = torch.tensor([[float(state[key])]], device=device)
                    else:
                        state_tensors[key] = torch.tensor(state[key], device=device).float()
                
                # Get Q-values
                q_values = self.policy_net(state_tensors)
                
                # Select action with highest Q-value
                return q_values.argmax().item()
    
    def train_step(self):
        """
        Perform one step of training
        
        Returns:
            Loss value for this step
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Compute Q-values
        q_values = self.policy_net(states).gather(1, actions)
        
        # Compute target Q-values
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.functional.smooth_l1_loss(q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
                
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes, max_episode_steps=1000, save_interval=100, evaluation_episodes=5):
        """
        Train the agent
        
        Args:
            num_episodes: Number of episodes to train for
            max_episode_steps: Maximum steps per episode
            save_interval: Interval for saving the model
            evaluation_episodes: Number of episodes to evaluate after each training episode
        """
        # Track statistics
        episode_rewards = []
        episode_losses = []
        episode_lengths = []
        best_eval_reward = float('-inf')
        
        for episode in range(1, num_episodes + 1):
            # Reset environment
            state, _ = self.env.reset()
            
            episode_reward = 0
            episode_loss = 0
            episode_steps = 0
            
            for step in range(max_episode_steps):
                # Select action
                action = self.select_action(state)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Update statistics
                episode_reward += reward
                episode_steps += 1
                
                # Add experience to replay buffer
                self.replay_buffer.add(state, action, reward, next_state, terminated or truncated)
                
                # Update state
                state = next_state
                
                # Train the model
                if len(self.replay_buffer) >= self.batch_size:
                    loss = self.train_step()
                    if loss is not None:
                        episode_loss += loss
                
                # Check if episode is done
                if terminated or truncated:
                    break
            
            # Update epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Calculate average loss
            avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0
            
            # Log statistics
            self.writer.add_scalar('Train/Reward', episode_reward, episode)
            self.writer.add_scalar('Train/Loss', avg_loss, episode)
            self.writer.add_scalar('Train/Epsilon', self.epsilon, episode)
            self.writer.add_scalar('Train/Steps', episode_steps, episode)
            
            # Store statistics
            episode_rewards.append(episode_reward)
            episode_losses.append(avg_loss)
            episode_lengths.append(episode_steps)
            
            # Print statistics
            print(f"Episode {episode}/{num_episodes} - Reward: {episode_reward:.2f} - Loss: {avg_loss:.4f} - Steps: {episode_steps} - Epsilon: {self.epsilon:.4f}")
            
            # Save model
            if episode % save_interval == 0:
                self.save_model(f"{self.model_dir}/isaac_dqn_episode_{episode}.pt")
                
                # Evaluate model
                eval_reward = self.evaluate(evaluation_episodes)
                self.writer.add_scalar('Eval/Reward', eval_reward, episode)
                
                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.save_model(f"{self.model_dir}/isaac_dqn_best.pt")
                    print(f"New best model saved with evaluation reward: {eval_reward:.2f}")
        
        # Save final model
        self.save_model(f"{self.model_dir}/isaac_dqn_final.pt")
        
        # Close tensorboard writer
        self.writer.close()
        
        return episode_rewards, episode_losses, episode_lengths
    
    def evaluate(self, num_episodes, max_episode_steps=1000):
        """
        Evaluate the agent
        
        Args:
            num_episodes: Number of episodes to evaluate for
            max_episode_steps: Maximum steps per episode
            
        Returns:
            Average reward over evaluation episodes
        """
        total_reward = 0
        
        for episode in range(num_episodes):
            # Reset environment
            state, _ = self.env.reset()
            
            episode_reward = 0
            
            for step in range(max_episode_steps):
                # Select action
                action = self.select_action(state, training=False)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Update statistics
                episode_reward += reward
                
                # Update state
                state = next_state
                
                # Check if episode is done
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
            print(f"Evaluation Episode {episode+1}/{num_episodes} - Reward: {episode_reward:.2f}")
        
        average_reward = total_reward / num_episodes
        print(f"Evaluation complete - Average Reward: {average_reward:.2f}")
        
        return average_reward
    
    def save_model(self, path):
        """Save model to file"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}")

def main():
    parser = argparse.ArgumentParser(description="Train Isaac DQN Agent")
    parser.add_argument('--game-dir', type=str, default="E:\\Steam\\steamapps\\common\\The Binding of Isaac Rebirth",
                       help="Path to The Binding of Isaac Rebirth game directory")
    parser.add_argument('--episodes', type=int, default=1000,
                       help="Number of episodes to train for")
    parser.add_argument('--max-steps', type=int, default=2000,
                       help="Maximum steps per episode")
    parser.add_argument('--save-interval', type=int, default=50,
                       help="Interval for saving the model")
    parser.add_argument('--model-dir', type=str, default="models",
                       help="Directory to save models")
    parser.add_argument('--log-dir', type=str, default="logs",
                       help="Directory to save logs")
    parser.add_argument('--load-model', type=str, default=None,
                       help="Path to load model from")
    
    args = parser.parse_args()
    
    # Create environment
    env = IsaacEnv(game_dir=args.game_dir)
    
    # Create agent
    agent = DQNAgent(env, model_dir=args.model_dir, log_dir=args.log_dir)
    
    # Load model if specified
    if args.load_model:
        agent.load_model(args.load_model)
    
    # Train agent
    print("Starting training...")
    agent.train(num_episodes=args.episodes, max_episode_steps=args.max_steps, save_interval=args.save_interval)
    
    # Close environment
    env.close()
    print("Training complete.")

if __name__ == "__main__":
    main() 