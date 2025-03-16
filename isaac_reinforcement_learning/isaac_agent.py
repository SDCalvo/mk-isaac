"""
Isaac Reinforcement Learning Example

This module shows how to use the IsaacGameStateReader for reinforcement learning.
It implements a simple AI agent that can navigate and shoot at enemies.

Usage:
    python isaac_agent.py

Requirements:
    - The Binding of Isaac: Rebirth with the IsaacGameStateReader mod installed
    - The game must be launched with the --luadebug option
"""

import time
import random
import numpy as np
import sys
import os

# Add the communication directory to the path so we can import the game state reader
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
communication_dir = os.path.join(project_root, "isaac_communication")
sys.path.append(communication_dir)

from isaac_game_state_reader import IsaacGameStateReader

class SimpleIsaacAgent:
    """A simple agent that can play The Binding of Isaac"""
    
    def __init__(self):
        # Initialize the game state reader
        self.reader = IsaacGameStateReader()
        
        # Set up action space
        self.actions = {
            'move_up': 0,
            'move_down': 1,
            'move_left': 2,
            'move_right': 3,
            'shoot_up': 4,
            'shoot_down': 5,
            'shoot_left': 6,
            'shoot_right': 7,
            'use_item': 8,
            'place_bomb': 9,
            'no_op': 10
        }
        
        # Initialize exploration/exploitation parameters
        self.epsilon = 0.2  # Probability of taking a random action
        
        # Simple Q-learning parameters (normally these would be learned)
        self.Q = {}  # Q-table
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        
        # Tracking variables
        self.last_state = None
        self.last_action = None
        self.last_health = 0
        self.last_enemy_count = 0
        self.last_room_clear = False
        self.frame_count = 0
    
    def connect(self):
        """Connect to the game"""
        if not self.reader.is_connected():
            print("Failed to connect to the game. Make sure it's running with --luadebug.")
            return False
        
        print("Connected to the game!")
        return True
    
    def state_to_features(self, game_state):
        """Convert the game state to a feature vector"""
        if not game_state:
            return None
        
        player = game_state.get('player', {})
        position = player.get('position', {})
        health = player.get('health', {})
        room = game_state.get('room', {})
        
        # Get player position
        player_x = position.get('x', 0)
        player_y = position.get('y', 0)
        
        # Get room size
        room_width = room.get('width', 0)
        room_height = room.get('height', 0)
        
        # Get enemies
        enemies = game_state.get('enemies', [])
        enemy_features = []
        
        # Get the 3 nearest enemies
        nearest_enemies = sorted(enemies, 
                                key=lambda e: ((e.get('position', {}).get('x', 0) - player_x)**2 + 
                                             (e.get('position', {}).get('y', 0) - player_y)**2))[:3]
        
        for enemy in nearest_enemies:
            enemy_pos = enemy.get('position', {})
            enemy_x = enemy_pos.get('x', 0)
            enemy_y = enemy_pos.get('y', 0)
            
            # Calculate relative position
            rel_x = (enemy_x - player_x) / max(room_width, 1)
            rel_y = (enemy_y - player_y) / max(room_height, 1)
            
            enemy_features.extend([rel_x, rel_y])
        
        # Pad with zeros if we don't have 3 enemies
        enemy_features.extend([0, 0] * (3 - len(nearest_enemies)))
        
        # Get pickups
        pickups = game_state.get('pickups', [])
        pickup_features = []
        
        # Get the nearest pickup
        nearest_pickup = self.reader.get_nearest_pickup()
        if nearest_pickup:
            pickup_pos = nearest_pickup.get('position', {})
            pickup_x = pickup_pos.get('x', 0)
            pickup_y = pickup_pos.get('y', 0)
            
            # Calculate relative position
            rel_x = (pickup_x - player_x) / max(room_width, 1)
            rel_y = (pickup_y - player_y) / max(room_height, 1)
            
            pickup_features.extend([rel_x, rel_y])
        else:
            pickup_features.extend([0, 0])
        
        # Get door features
        doors = room.get('doors', [])
        door_features = []
        
        # Find the nearest unexplored door
        nearest_door = None
        min_dist = float('inf')
        
        for door in doors:
            door_pos = door.get('position', {})
            door_x = door_pos.get('x', 0)
            door_y = door_pos.get('y', 0)
            
            dist = ((door_x - player_x)**2 + (door_y - player_y)**2)**0.5
            
            if dist < min_dist:
                min_dist = dist
                nearest_door = door
        
        if nearest_door:
            door_pos = nearest_door.get('position', {})
            door_x = door_pos.get('x', 0)
            door_y = door_pos.get('y', 0)
            
            # Calculate relative position
            rel_x = (door_x - player_x) / max(room_width, 1)
            rel_y = (door_y - player_y) / max(room_height, 1)
            
            door_features.extend([rel_x, rel_y])
        else:
            door_features.extend([0, 0])
        
        # Combine all features
        features = [
            health.get('current', 0) / max(health.get('max', 6), 1),  # Normalized health
            len(enemies),  # Number of enemies
            room.get('is_clear', False),  # Room cleared
        ]
        
        features.extend(enemy_features)
        features.extend(pickup_features)
        features.extend(door_features)
        
        # Convert to tuple for Q-table lookup
        return tuple(np.round(features, 2))
    
    def choose_action(self, state):
        """Choose an action based on the current state"""
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.choice(list(self.actions.keys()))
        
        # Get Q-values for this state
        if state not in self.Q:
            self.Q[state] = {action: 0 for action in self.actions.keys()}
        
        # Choose the action with the highest Q-value
        return max(self.Q[state], key=self.Q[state].get)
    
    def take_action(self, action):
        """Execute an action in the game"""
        if action.startswith('move_'):
            direction = action.replace('move_', '')
            self.reader.move_player(direction)
        elif action.startswith('shoot_'):
            direction = action.replace('shoot_', '')
            self.reader.shoot(direction)
        elif action == 'use_item':
            self.reader.use_item()
        elif action == 'place_bomb':
            self.reader.place_bomb()
        elif action == 'no_op':
            pass  # Do nothing
    
    def calculate_reward(self, state, game_state):
        """Calculate the reward for the current state"""
        reward = 0
        
        # Get current stats
        player = game_state.get('player', {})
        health = player.get('health', {})
        current_health = health.get('current', 0)
        room = game_state.get('room', {})
        is_room_clear = room.get('is_clear', False)
        enemy_count = len(game_state.get('enemies', []))
        
        # Health loss penalty
        if self.last_health > current_health:
            reward -= 10 * (self.last_health - current_health)
        
        # Enemy defeat reward
        if self.last_enemy_count > enemy_count:
            reward += 5 * (self.last_enemy_count - enemy_count)
        
        # Room clear reward
        if is_room_clear and not self.last_room_clear:
            reward += 20
        
        # Collect pickup reward
        if 'pickups' in game_state:
            # This would need to track which pickups were collected
            pass
        
        # Update tracking variables
        self.last_health = current_health
        self.last_enemy_count = enemy_count
        self.last_room_clear = is_room_clear
        
        return reward
    
    def update_q_value(self, state, action, reward, next_state):
        """Update the Q-value for the given state-action pair"""
        if state not in self.Q:
            self.Q[state] = {a: 0 for a in self.actions.keys()}
        
        if next_state not in self.Q:
            self.Q[next_state] = {a: 0 for a in self.actions.keys()}
        
        # Q-learning update rule
        best_next_value = max(self.Q[next_state].values())
        self.Q[state][action] += self.learning_rate * (reward + self.discount_factor * best_next_value - self.Q[state][action])
    
    def run(self):
        """Run the agent"""
        if not self.connect():
            return
        
        print("Starting agent...")
        
        # Initialize tracking variables
        game_state = self.reader.get_game_state()
        if game_state:
            player = game_state.get('player', {})
            health = player.get('health', {})
            self.last_health = health.get('current', 0)
            self.last_enemy_count = len(game_state.get('enemies', []))
            self.last_room_clear = game_state.get('room', {}).get('is_clear', False)
        
        try:
            while True:
                # Get the current game state
                game_state = self.reader.get_game_state()
                if not game_state:
                    print("Failed to get game state")
                    time.sleep(0.5)
                    continue
                
                # Update frame count
                self.frame_count += 1
                
                # Only take an action every 10 frames
                if self.frame_count % 10 == 0:
                    # Convert game state to features
                    state = self.state_to_features(game_state)
                    if not state:
                        continue
                    
                    # Choose and take an action
                    action = self.choose_action(state)
                    self.take_action(action)
                    
                    # Calculate reward
                    reward = self.calculate_reward(state, game_state)
                    
                    # Print current status
                    print(f"\n--- Frame {self.frame_count} ---")
                    print(f"State features: {state}")
                    print(f"Action: {action}")
                    print(f"Reward: {reward}")
                    
                    # Update Q-value
                    if self.last_state is not None and self.last_action is not None:
                        self.update_q_value(self.last_state, self.last_action, reward, state)
                    
                    # Save for next update
                    self.last_state = state
                    self.last_action = action
                
                # Sleep to avoid overloading the game
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\nStopping agent...")
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function"""
    agent = SimpleIsaacAgent()
    agent.run()

if __name__ == "__main__":
    main() 