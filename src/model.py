import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    """
    Deep Q-Network for The Binding of Isaac
    """
    def __init__(self, input_shape, n_actions):
        """
        Initialize the DQN model.
        
        Args:
            input_shape (tuple): Shape of input observations (frame stack size, height, width)
            n_actions (int): Number of possible actions
        """
        super(DQN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the conv output
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    def _get_conv_output_size(self, shape):
        """Calculate the size of convolutional layer output"""
        o = self._forward_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
        
    def _forward_conv(self, x):
        """Forward pass through convolutional layers"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, frames, height, width)
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        # Pass through convolutional layers
        conv_out = self._forward_conv(x)
        
        # Flatten the output
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(conv_out))
        x = self.fc2(x)
        
        return x

class ReplayBuffer:
    """
    Experience replay buffer for DQN training
    """
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size (int): Size of the batch to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
        
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)

class Agent:
    """
    DQN Agent for playing The Binding of Isaac
    """
    def __init__(self, state_shape, n_actions, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the DQN agent.
        
        Args:
            state_shape (tuple): Shape of state observations
            n_actions (int): Number of possible actions
            device (str): Device to run the model on ("cuda" or "cpu")
        """
        self.device = device
        self.n_actions = n_actions
        
        # Create policy and target networks
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-4
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Create replay buffer
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        
    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state (numpy.ndarray): Current state
            training (bool): Whether in training mode
            
        Returns:
            int: Selected action index
        """
        if training and np.random.rand() < self.epsilon:
            # Explore: select a random action
            return np.random.randint(0, self.n_actions)
        
        # Exploit: select the best action according to the policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
            
    def update_epsilon(self):
        """Update the exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def learn(self):
        """
        Update the policy network using experiences from the replay buffer.
        
        Returns:
            float: Loss value
        """
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()
        
    def update_target_network(self):
        """Update the target network with the policy network's weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, path):
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load(self, path):
        """
        Load the model from a file.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
