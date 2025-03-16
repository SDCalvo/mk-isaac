#!/usr/bin/env python
"""
Isaac Gym Environment

This module provides a gym-compatible environment for The Binding of Isaac: Rebirth
using the IsaacGameStateReader mod to get game state and control the game.
"""

import os
import sys
import time
from typing import Dict, Tuple, Any, Union, Optional

import numpy as np

# Add the project root to the path to access the isaac_communication package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import gymnasium first, fall back to gym if not available
try:
    import gymnasium as gym
    from gymnasium import spaces
    USING_GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    USING_GYMNASIUM = False
    print("Using legacy gym package instead of gymnasium")

from isaac_communication.isaac_game_state_reader import IsaacGameStateReader


# Using type ignore to fix the base class issue
class IsaacEnv(gym.Env):  # type: ignore
    """
    A gym environment for The Binding of Isaac: Rebirth.
    
    This environment uses the IsaacGameStateReader mod to get game state and control the game.
    The observation space is a dictionary representing the game state.
    The action space is discrete, representing the different commands the agent can send.
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 15}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Initialize the game state reader
        self.client = IsaacGameStateReader()
        print("Connecting to Isaac mod...")
        self.connected = False
        self.is_game_over = False
        
        try:
            # Check connection
            if self.client._check_connection():
                self.connected = True
                print("Successfully connected to Isaac mod!")
            else:
                print("Failed to connect to the Isaac mod. Make sure the game is running with --luadebug option.")
        except Exception as e:
            print(f"Failed to connect to the Isaac mod: {e}")
            print("Make sure the game is running with --luadebug option.")
        
        # Define action space
        # 0: no-op, 1: move_up, 2: move_down, 3: move_left, 4: move_right, 
        # 5: shoot_up, 6: shoot_down, 7: shoot_left, 8: shoot_right, 9: use_item, 10: place_bomb
        self.action_space = spaces.Discrete(11)
        
        # Define observation space
        # We use a Dict space for the observation, with low and high values for each component
        self.observation_space = spaces.Dict({
            # Player health
            'health': spaces.Box(low=0, high=24, shape=(1,), dtype=np.float32),
            
            # Player position
            'position': spaces.Box(low=0, high=1000, shape=(2,), dtype=np.float32),
            
            # Floor/stage information
            'stage': spaces.Box(low=1, high=13, shape=(1,), dtype=np.int32),
            
            # Room information
            'room_clear': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            
            # Enemy count
            'enemy_count': spaces.Box(low=0, high=30, shape=(1,), dtype=np.int32),
        })
        
        # Store the render mode
        self.render_mode = render_mode
        
        # Initialize state variables
        self.prev_state = None
        self.current_state = None
        self.episode_rewards = 0
        self.steps_since_last_room = 0
    
    def _get_obs(self):
        """Get observation from current game state"""
        # Get game state
        try:
            # Send status command and wait for response
            self.client.send_command("status")
            time.sleep(0.1)  # Give the mod time to respond
            response = self.client.read_response()
            
            if response:
                # Use the game state parser from the client
                game_state = self.client._parse_status_text(response)
            else:
                raise Exception("No response received from mod")
        except Exception as e:
            print(f"Failed to get game state: {e}")
            return {
                'health': np.array([0], dtype=np.float32),
                'position': np.zeros((2,), dtype=np.float32),
                'stage': np.array([0], dtype=np.int32),
                'room_clear': np.array([0], dtype=np.int32),
                'enemy_count': np.array([0], dtype=np.int32),
            }
        
        # Extract values from game state
        player = game_state.get('player', {})
        health = player.get('health', {}).get('current', 0)
        
        position = player.get('position', {})
        position_x = position.get('x', 0)
        position_y = position.get('y', 0)
        
        floor = game_state.get('floor', {})
        stage = floor.get('stage', 1)
        
        room = game_state.get('room', {})
        room_clear = room.get('is_clear', True)
        enemy_count = len(game_state.get('enemies', []))
        
        # Create observation dictionary
        obs = {
            'health': np.array([health], dtype=np.float32),
            'position': np.array([position_x, position_y], dtype=np.float32),
            'stage': np.array([stage], dtype=np.int32),
            'room_clear': np.array([int(room_clear)], dtype=np.int32),
            'enemy_count': np.array([enemy_count], dtype=np.int32),
        }
        
        # Check for game over
        self.is_game_over = health <= 0
        
        # Update state
        self.current_state = {
            'obs': obs,
            'is_game_over': self.is_game_over,
            'room_clear': int(room_clear),
            'enemy_count': enemy_count,
            'health': health,
            'floor': stage,
        }
        
        return obs
    
    def _get_info(self):
        """Return additional info for debugging"""
        return {
            'episode_rewards': self.episode_rewards,
            'steps_since_last_room': self.steps_since_last_room,
        }
    
    def _calculate_reward(self, prev_state):
        """Calculate reward based on state changes"""
        if prev_state is None or self.current_state is None:
            return 0
        
        reward = 0
        
        # Reward for entering a new room (exploration)
        if prev_state['room_clear'] == 0 and self.current_state['room_clear'] == 1:
            reward += 5
            self.steps_since_last_room = 0
        
        # Reward for clearing a room
        if prev_state['room_clear'] == 1 and self.current_state['room_clear'] == 0:
            reward += 10
            self.steps_since_last_room = 0
        
        # Reward for killing enemies
        if prev_state['enemy_count'] > self.current_state['enemy_count'] and self.current_state['room_clear'] == 1:
            reward += 1
            self.steps_since_last_room = 0
        
        # Reward for advancing floors
        if self.current_state['floor'] > prev_state['floor']:
            reward += 50
            self.steps_since_last_room = 0
        
        # Penalty for taking damage
        health_change = self.current_state['health'] - prev_state['health']
        if health_change < 0:
            reward += health_change * 5  # Negative reward for damage
        
        # Large penalty for game over
        if self.current_state['is_game_over']:
            reward -= 100
        
        # Small penalty for each step to encourage faster progress
        reward -= 0.1
        
        # Penalize staying in the same state for too long
        self.steps_since_last_room += 1
        if self.steps_since_last_room > 300:
            reward -= 10
            self.steps_since_last_room = 0
        
        self.episode_rewards += reward
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        # Handle differences between gym and gymnasium interfaces
        try:
            super().reset(seed=seed)
        except TypeError:
            # Old gym API doesn't have options parameter
            if hasattr(super(), 'reset'):
                super().reset(seed=seed)
        
        # Wait for game to be running
        attempts = 0
        connected = False
        
        while not connected and attempts < 30:
            try:
                connected = self.client._check_connection()
                if connected:
                    print("Connected to the game!")
                    break
                
                print(f"Waiting for Isaac game to start... (attempt {attempts+1}/30)")
                time.sleep(1)
                attempts += 1
            except Exception as e:
                print(f"Error checking connection: {e}")
                time.sleep(1)
                attempts += 1
        
        if not connected:
            print("WARNING: Could not connect to Isaac game. Make sure the game is running with the mod installed.")
            print("Continuing with default values, but the environment may not work correctly.")
        
        # Reset environment state
        self.episode_rewards = 0
        self.steps_since_last_room = 0
        
        # Get initial observation
        observation = self._get_obs()
        info = self._get_info()
        
        # Return appropriate values based on which gym API we're using
        if USING_GYMNASIUM:
            return observation, info
        else:
            return observation
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action (int): Action to take
                0: no-op
                1: move_up
                2: move_down
                3: move_left
                4: move_right
                5: shoot_up
                6: shoot_down
                7: shoot_left
                8: shoot_right
                9: use_item
                10: place_bomb
        
        Returns:
            observation: Current observation
            reward: Reward for this step
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated (gymnasium only)
            info: Additional information
        """
        prev_state = self.current_state
        
        # Map action to command
        action_to_command = {
            1: "move_up",
            2: "move_down",
            3: "move_left",
            4: "move_right",
            5: "shoot_up",
            6: "shoot_down",
            7: "shoot_left",
            8: "shoot_right",
            9: "use_item",
            10: "place_bomb"
        }
        
        # Send command if not no-op
        if action in action_to_command:
            command = action_to_command[action]
            self.client.send_command(command)
        
        # Wait for action to take effect
        time.sleep(0.1)
        
        # Get new observation
        observation = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward(prev_state)
        
        # Check if episode is done
        terminated = self.is_game_over
        truncated = False
        
        # Get info
        info = self._get_info()
        
        # Return appropriate values based on which gym API we're using
        if USING_GYMNASIUM:
            return observation, reward, terminated, truncated, info
        else:
            return observation, reward, terminated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            # Nothing to do here, the game already renders itself
            pass
        
        return None
    
    def close(self):
        """Close the environment"""
        # Nothing to do here, we don't control the game process
        pass

def main():
    """Test the environment with random actions"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Isaac Gym Environment")
    parser.add_argument('--steps', type=int, default=100,
                       help="Number of steps to run")
    
    args = parser.parse_args()
    
    # Create environment
    env = IsaacEnv(render_mode="human")
    
    # Reset environment - avoid tuple unpacking to handle both gym versions
    reset_result = env.reset()
    
    observation = None
    info = {}
    
    # Safely handle different return types for gymnasium vs gym
    if USING_GYMNASIUM and isinstance(reset_result, tuple) and len(reset_result) >= 2:
        observation = reset_result[0]
        info = reset_result[1]
    else:
        observation = reset_result
    
    print("Environment reset")
    print("Observation:", observation)
    print("Info:", info)
    
    # Take random actions
    print("\nTaking random actions...")
    for i in range(args.steps):
        action = env.action_space.sample()
        
        # Call step but avoid tuple unpacking to handle both gym versions
        step_result = env.step(action)
        
        # Default values
        observation = None
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        # Safely handle different return types for gymnasium vs gym
        if isinstance(step_result, tuple):
            if len(step_result) >= 1:
                observation = step_result[0]
            if len(step_result) >= 2:
                reward = step_result[1] 
            if len(step_result) >= 3:
                terminated = step_result[2]
            if len(step_result) >= 4 and USING_GYMNASIUM:
                truncated = step_result[3]
                if len(step_result) >= 5:
                    info = step_result[4]
            elif len(step_result) >= 4 and not USING_GYMNASIUM:
                info = step_result[3]
        
        # Use dict.get safely with proper type checking
        episode_rewards = 0
        steps_since_room = 0
        if isinstance(info, dict):
            episode_rewards = info.get('episode_rewards', 0)
            steps_since_room = info.get('steps_since_last_room', 0)
        
        print(f"Step {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Total Reward: {episode_rewards}")
        print(f"  Steps Since Last Room: {steps_since_room}")
        
        if terminated or truncated:
            print("Episode finished")
            break
        
        time.sleep(0.2)
    
    env.close()
    print("Environment closed")

if __name__ == "__main__":
    main() 