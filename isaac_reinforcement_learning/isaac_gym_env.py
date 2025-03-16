#!/usr/bin/env python
"""
Isaac Gym Environment

This module provides a gym-compatible environment for The Binding of Isaac: Rebirth
using the IsaacGameStateReader mod to get game state and control the game.
"""

import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from isaac_client import IsaacClient

class IsaacEnv(gym.Env):
    """
    A gym environment for The Binding of Isaac: Rebirth.
    
    This environment uses the IsaacGameStateReader mod to get game state and control the game.
    The observation space is a dictionary representing the game state.
    The action space is discrete, representing the different commands the agent can send.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 15}
    
    def __init__(self, game_dir, render_mode=None):
        super(IsaacEnv, self).__init__()
        
        # Initialize the client
        self.client = IsaacClient(game_dir)
        
        # Define action space
        # 0: no-op, 1: move_up, 2: move_down, 3: move_left, 4: move_right, 5: use_item, 6: bomb
        self.action_space = spaces.Discrete(7)
        
        # Define observation space
        # We use a Dict space for the observation, with low and high values for each component
        self.observation_space = spaces.Dict({
            # Player health
            'health': spaces.Box(low=0, high=24, shape=(1,), dtype=np.float32),
            'max_health': spaces.Box(low=0, high=24, shape=(1,), dtype=np.float32),
            'soul_hearts': spaces.Box(low=0, high=24, shape=(1,), dtype=np.float32),
            
            # Player stats
            'damage': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'fire_rate': spaces.Box(low=0, high=30, shape=(1,), dtype=np.float32),
            'speed': spaces.Box(low=0, high=2, shape=(1,), dtype=np.float32),
            
            # Floor/stage information
            'stage': spaces.Box(low=1, high=13, shape=(1,), dtype=np.int32),
            
            # Room information
            'room_cleared': spaces.Discrete(2),
            'enemies': spaces.Box(low=0, high=50, shape=(1,), dtype=np.int32),
            
            # Optional: enemy positions (for advanced agents)
            'enemy_positions': spaces.Box(low=0, high=1000, shape=(10, 2), dtype=np.float32),
        })
        
        # Initialize state
        self.state = None
        self.steps_since_last_reward = 0
        self.max_steps_without_reward = 300
        self.rooms_visited = set()
        self.current_floor = None
        self.total_reward = 0
        self.render_mode = render_mode
    
    def _get_obs(self):
        """Get observation from the game state"""
        game_state = self.client.get_game_state()
        
        if game_state is None:
            # If we can't get the game state, return a zeroed observation
            return {
                'health': np.array([0], dtype=np.float32),
                'max_health': np.array([0], dtype=np.float32),
                'soul_hearts': np.array([0], dtype=np.float32),
                'damage': np.array([0], dtype=np.float32),
                'fire_rate': np.array([0], dtype=np.float32),
                'speed': np.array([0], dtype=np.float32),
                'stage': np.array([0], dtype=np.int32),
                'room_cleared': 0,
                'enemies': np.array([0], dtype=np.int32),
                'enemy_positions': np.zeros((10, 2), dtype=np.float32),
            }
        
        # Extract values from game state
        player = game_state['player']
        health = player['health']
        stats = player['stats']
        floor = game_state['floor']
        room = game_state['room']
        
        # Track the current room
        room_id = f"{floor['stage']}-{room['type']}-{player['position']['x']:.0f}-{player['position']['y']:.0f}"
        
        # Track floor changes
        if self.current_floor != floor['stage']:
            self.current_floor = floor['stage']
            # Reset rooms visited when we change floors
            self.rooms_visited = set()
        
        # Add this room to visited rooms if not already there
        is_new_room = room_id not in self.rooms_visited
        if is_new_room:
            self.rooms_visited.add(room_id)
        
        # Process enemy positions (max 10 enemies)
        enemy_positions = np.zeros((10, 2), dtype=np.float32)
        for i, enemy in enumerate(room['enemies']):
            if i < 10:  # Only take up to 10 enemies
                enemy_positions[i, 0] = enemy['position']['x']
                enemy_positions[i, 1] = enemy['position']['y']
        
        # Create observation dictionary
        obs = {
            'health': np.array([health['hearts']], dtype=np.float32),
            'max_health': np.array([health['maxHearts']], dtype=np.float32),
            'soul_hearts': np.array([health['soulHearts']], dtype=np.float32),
            'damage': np.array([stats['damage']], dtype=np.float32),
            'fire_rate': np.array([stats['firerate']], dtype=np.float32),
            'speed': np.array([stats['speed']], dtype=np.float32),
            'stage': np.array([floor['stage']], dtype=np.int32),
            'room_cleared': int(room['cleared']),
            'enemies': np.array([room['enemyCount']], dtype=np.int32),
            'enemy_positions': enemy_positions,
        }
        
        # Check for game over
        self.is_game_over = health['hearts'] <= 0 and health['soulHearts'] <= 0
        
        # Update state
        self.state = {
            'obs': obs,
            'is_new_room': is_new_room,
            'is_game_over': self.is_game_over,
            'room_cleared': room['cleared'],
            'enemy_count': room['enemyCount'],
            'health': health['hearts'],
            'max_health': health['maxHearts'],
            'soul_hearts': health['soulHearts'],
            'floor': floor['stage'],
        }
        
        return obs
    
    def _get_info(self):
        """Return additional info for debugging"""
        return {
            'rooms_visited': len(self.rooms_visited),
            'current_floor': self.current_floor,
            'total_reward': self.total_reward,
            'steps_without_reward': self.steps_since_last_reward,
        }
    
    def _calculate_reward(self, prev_state):
        """Calculate reward based on state changes"""
        if prev_state is None or self.state is None:
            return 0
        
        reward = 0
        
        # Reward for entering a new room (exploration)
        if self.state['is_new_room']:
            reward += 5
            self.steps_since_last_reward = 0
        
        # Reward for clearing a room
        if not prev_state['room_cleared'] and self.state['room_cleared']:
            reward += 10
            self.steps_since_last_reward = 0
        
        # Reward for killing enemies
        if prev_state['enemy_count'] > self.state['enemy_count'] and not self.state['room_cleared']:
            reward += 1
            self.steps_since_last_reward = 0
        
        # Reward for advancing floors
        if self.state['floor'] > prev_state['floor']:
            reward += 50
            self.steps_since_last_reward = 0
        
        # Penalty for taking damage
        health_change = (self.state['health'] + self.state['soul_hearts']) - (prev_state['health'] + prev_state['soul_hearts'])
        if health_change < 0:
            reward += health_change * 5  # Negative reward for damage
        
        # Large penalty for game over
        if self.state['is_game_over']:
            reward -= 100
        
        # Small penalty for each step to encourage faster progress
        reward -= 0.1
        
        # Penalize staying in the same state for too long
        self.steps_since_last_reward += 1
        if self.steps_since_last_reward > self.max_steps_without_reward:
            reward -= 10
            self.steps_since_last_reward = 0
        
        self.total_reward += reward
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Wait for game to be running
        game_state = None
        attempts = 0
        while game_state is None and attempts < 10:
            game_state = self.client.get_game_state()
            if game_state is None:
                print("Waiting for Isaac game to start...")
                time.sleep(1)
                attempts += 1
        
        if game_state is None:
            raise RuntimeError("Could not connect to Isaac game. Make sure the game is running with the mod installed.")
        
        # Reset environment state
        self.steps_since_last_reward = 0
        self.total_reward = 0
        
        # Get initial observation
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
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
                5: use_item
                6: bomb
        
        Returns:
            observation: Current observation
            reward: Reward for this step
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        prev_state = self.state
        
        # Map action to command
        action_to_command = {
            1: "move_up",
            2: "move_down",
            3: "move_left",
            4: "move_right",
            5: "use_item",
            6: "bomb"
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
        
        return observation, reward, terminated, truncated, info
    
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
    parser.add_argument('--game-dir', type=str, default="E:\\Steam\\steamapps\\common\\The Binding of Isaac Rebirth",
                       help="Path to The Binding of Isaac Rebirth game directory")
    
    args = parser.parse_args()
    
    # Create environment
    env = IsaacEnv(game_dir=args.game_dir, render_mode="human")
    
    # Reset environment
    observation, info = env.reset()
    print("Environment reset")
    print("Observation:", observation)
    print("Info:", info)
    
    # Take 100 random actions
    print("\nTaking random actions...")
    for i in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Total Reward: {info['total_reward']}")
        print(f"  Rooms Visited: {info['rooms_visited']}")
        print(f"  Current Floor: {info['current_floor']}")
        
        if terminated or truncated:
            print("Episode finished")
            break
        
        time.sleep(0.2)
    
    env.close()
    print("Environment closed")

if __name__ == "__main__":
    main() 