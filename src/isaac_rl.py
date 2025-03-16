"""
Reinforcement Learning for The Binding of Isaac

This script implements a basic reinforcement learning setup for playing 
The Binding of Isaac using the simplified game state detection.
"""

import os
import time
import numpy as np
import keyboard
import random
from collections import deque
import json

from game_capture import GameCapture

# Define actions (keyboard keys for controlling Isaac)
ACTIONS = {
    0: 'w',  # Up
    1: 's',  # Down
    2: 'a',  # Left
    3: 'd',  # Right
    4: None  # No action
}

class IsaacRL:
    """
    A simple reinforcement learning agent for playing The Binding of Isaac.
    Uses Q-learning with the simplified game state detection.
    """
    
    def __init__(self, epsilon=0.1, learning_rate=0.1, discount_factor=0.95):
        # Initialize game capture
        self.game_capture = GameCapture()
        
        # Initialize Q-learning parameters
        self.epsilon = epsilon  # Exploration rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Initialize Q-table (state -> action values)
        self.q_table = {}
        
        # Tracking information
        self.floor = 1
        self.total_rooms_explored = 0
        self.game_started = False
        self.episode_steps = 0
        self.max_steps_per_episode = 1000
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=2000)
        
        # Create output directory
        os.makedirs("rl_output", exist_ok=True)
        
        # Load existing Q-table if available
        self.load_q_table()
    
    def load_q_table(self):
        """Load Q-table from disk if available"""
        try:
            with open('rl_output/q_table.json', 'r') as f:
                self.q_table = json.load(f)
                
                # Convert string keys back to tuples
                for key in list(self.q_table.keys()):
                    state = eval(key)
                    self.q_table[state] = self.q_table.pop(key)
                
                print(f"Loaded Q-table with {len(self.q_table)} states")
        except (FileNotFoundError, json.JSONDecodeError):
            print("No valid Q-table found, starting fresh")
    
    def save_q_table(self):
        """Save Q-table to disk"""
        # Convert tuple keys to strings for JSON serialization
        serializable_q_table = {str(key): value for key, value in self.q_table.items()}
        
        with open('rl_output/q_table.json', 'w') as f:
            json.dump(serializable_q_table, f, indent=2)
        
        print(f"Saved Q-table with {len(self.q_table)} states")
    
    def get_state_representation(self, game_state):
        """
        Convert the raw game state into a simplified state representation.
        Returns a tuple of discrete values that can be used as Q-table key.
        """
        # Extract relevant information
        health = game_state['health']
        floor = game_state['current_floor']
        is_unexplored = game_state['is_unexplored_room']
        is_game_over = game_state['is_game_over']
        
        # Discretize health (0-3, 4-6, 7-9, 10+)
        if health <= 0:
            health_bin = 0
        elif health <= 3:
            health_bin = 1
        elif health <= 6:
            health_bin = 2
        elif health <= 9:
            health_bin = 3
        else:
            health_bin = 4
        
        # Combine into a state tuple
        state = (health_bin, floor, is_unexplored, is_game_over)
        return state
    
    def get_action(self, state):
        """Select an action using epsilon-greedy policy"""
        # Explore: choose a random action
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTIONS) - 1)
        
        # Exploit: choose the best action from Q-table
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(ACTIONS)
        
        return np.argmax(self.q_table[state])
    
    def calculate_reward(self, prev_state, current_state, action):
        """
        Calculate reward based on the previous state, current state, and action.
        Rewards:
        - +10 for exploring a new room
        - +25 for moving down a floor
        - -5 for taking damage
        - -50 for game over
        """
        # Extract state information
        prev_health_bin, prev_floor, prev_unexplored, prev_game_over = prev_state
        current_health_bin, current_floor, current_unexplored, current_game_over = current_state
        
        reward = 0
        
        # Reward for exploring new rooms
        if current_unexplored:
            reward += 10
            self.total_rooms_explored += 1
        
        # Reward for moving down floors
        if current_floor > prev_floor:
            reward += 25
            self.floor = current_floor
        
        # Penalty for taking damage
        if current_health_bin < prev_health_bin:
            reward -= 5
        
        # Big penalty for game over
        if current_game_over:
            reward -= 50
        
        # Small penalty for doing nothing
        if action == 4:  # No action
            reward -= 0.1
        
        return reward
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule"""
        # Initialize Q-values if states don't exist
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(ACTIONS)
        
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * len(ACTIONS)
        
        # Get current Q-value
        current_q = self.q_table[state][action]
        
        # Get max Q-value for next state
        max_next_q = max(self.q_table[next_state])
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[state][action] = new_q
        
        # Add to experience replay buffer
        self.replay_buffer.append((state, action, reward, next_state))
    
    def experience_replay(self, batch_size=32):
        """Learn from past experiences"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample random batch from replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        
        for state, action, reward, next_state in batch:
            # Initialize Q-values if states don't exist
            if state not in self.q_table:
                self.q_table[state] = [0.0] * len(ACTIONS)
            
            if next_state not in self.q_table:
                self.q_table[next_state] = [0.0] * len(ACTIONS)
            
            # Get current Q-value
            current_q = self.q_table[state][action]
            
            # Get max Q-value for next state
            max_next_q = max(self.q_table[next_state])
            
            # Q-learning update rule
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
            
            # Update Q-table
            self.q_table[state][action] = new_q
    
    def perform_action(self, action_id):
        """Execute the selected action using keyboard inputs"""
        key = ACTIONS[action_id]
        if key is not None:
            keyboard.press(key)
            time.sleep(0.1)  # Press for a short duration
            keyboard.release(key)
            time.sleep(0.05)  # Small delay between actions
    
    def train(self, num_episodes=10):
        """Train the agent for the specified number of episodes"""
        print(f"Starting training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode+1}/{num_episodes}")
            
            # Reset episode variables
            self.episode_steps = 0
            self.game_started = False
            
            # Make sure game is in focus
            frame = self.game_capture.capture_game_window()
            if frame is None:
                print("Cannot focus game window. Make sure the game is running.")
                return
            
            # Get initial state
            game_state = self.game_capture.get_game_state(frame)
            current_state = self.get_state_representation(game_state)
            
            # Main training loop
            game_over = False
            while not game_over and self.episode_steps < self.max_steps_per_episode:
                # Select action
                action = self.get_action(current_state)
                
                # Execute action
                self.perform_action(action)
                
                # Allow time for action to have an effect
                time.sleep(0.2)
                
                # Get new state
                frame = self.game_capture.capture_game_window()
                if frame is None:
                    print("Lost game window focus. Ending episode.")
                    break
                    
                new_game_state = self.game_capture.get_game_state(frame)
                new_state = self.get_state_representation(new_game_state)
                
                # Calculate reward
                reward = self.calculate_reward(current_state, new_state, action)
                
                # Update Q-table
                self.update_q_table(current_state, action, reward, new_state)
                
                # Periodically perform experience replay
                if self.episode_steps % 10 == 0:
                    self.experience_replay()
                
                # Log progress
                print(f"\rStep {self.episode_steps}: Action={ACTIONS[action]}, Reward={reward:.1f}, " +
                      f"Health={new_game_state['health']}, New Room={new_game_state['is_new_room']}, " +
                      f"Floor={new_game_state['current_floor']}", end="")
                
                # Update state
                current_state = new_state
                
                # Check if game over
                game_over = new_game_state['is_game_over']
                
                # Increment step counter
                self.episode_steps += 1
            
            # End of episode
            print(f"\nEpisode {episode+1} ended after {self.episode_steps} steps")
            print(f"Total rooms explored: {self.total_rooms_explored}")
            print(f"Current floor: {self.floor}")
            
            # Save Q-table every episode
            self.save_q_table()
            
            # Wait a bit before starting next episode
            print("Waiting 5 seconds before next episode...")
            time.sleep(5)
        
        print("\nTraining complete!")

def main():
    """Main function to start the reinforcement learning process"""
    # Create RL agent
    agent = IsaacRL(epsilon=0.2)  # Higher exploration rate for learning
    
    print("Isaac Reinforcement Learning Agent")
    print("----------------------------------")
    print("This agent will play The Binding of Isaac using reinforcement learning.")
    print("Make sure the game is running in windowed mode.")
    print("The agent will take control of the game, so be ready!")
    print("Press Ctrl+C at any time to stop the training.")
    
    print("\nStarting in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    try:
        # Start training
        agent.train(num_episodes=5)  # Start with a small number of episodes
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Save Q-table before exiting
        agent.save_q_table()
        print("Q-table saved. Training complete.")

if __name__ == "__main__":
    main() 