try:
    # Try absolute import first (for running as a module)
    from src.game_capture import GameCapture
    from src.input_controller import Action, InputController
except ImportError:
    # Fall back to relative import (for running within the package)
    from .game_capture import GameCapture
    from .input_controller import InputController, Action

import time

import numpy as np


class IsaacEnvironment:
    """
    Reinforcement learning environment for The Binding of Isaac
    """

    def __init__(self, capture_region=None, frame_skip=4):
        """
        Initialize the game environment.

        Args:
            capture_region (tuple): Region of screen to capture (left, top, width, height)
            frame_skip (int): Number of frames to skip between actions (for performance)
        """
        self.capture = GameCapture()
        self.controller = InputController()
        self.capture_region = capture_region
        self.frame_skip = frame_skip

        # State variables
        self.current_frame = None
        self.frames_buffer = []  # For frame stacking
        self.max_frames_in_buffer = 4  # Store last 4 frames

        # Reward tracking
        self.previous_health = None
        self.previous_score = None
        self.previous_enemies = []
        self.previous_items = []
        self.previous_room_doors = {}
        self.previous_player_position = None

        # Room tracking
        self.rooms_visited = set()
        self.current_room_cleared = False

        # Reward weights
        self.reward_weights = {
            "health_loss": -5.0,  # Penalty for losing health
            "health_gain": 2.0,  # Reward for gaining health
            "enemy_killed": 1.0,  # Reward for killing an enemy
            "item_collected": 3.0,  # Reward for collecting an item
            "room_cleared": 5.0,  # Reward for clearing a room
            "room_explored": 2.0,  # Reward for exploring a new room
            "door_approach": 0.5,  # Small reward for approaching a door
            "movement": 0.01,  # Small reward for moving (to encourage exploration)
            "staying_still": -0.01,  # Small penalty for not moving
        }

    def reset(self):
        """
        Reset the environment to initial state.
        This doesn't actually reset the game, just resets our state tracking.

        Returns:
            numpy.ndarray: Initial state observation
        """
        # Reset input controller to ensure no keys are stuck
        self.controller.reset_inputs()

        # Clear frame buffer
        self.frames_buffer = []

        # Reset tracking variables
        self.previous_health = None
        self.previous_score = None
        self.previous_enemies = []
        self.previous_items = []
        self.previous_room_doors = {}
        self.previous_player_position = None
        self.rooms_visited = set()
        self.current_room_cleared = False

        # Capture initial frame
        for _ in range(self.max_frames_in_buffer):
            self._capture_frame()

        # Get the initial game state to initialize tracking variables
        game_state = self.capture.detect_game_state(self.current_frame)
        self.previous_health = game_state.get("player_health")
        self.previous_enemies = game_state.get("enemies", [])
        self.previous_items = game_state.get("items", [])
        self.previous_room_doors = game_state.get("doors", {})
        self.previous_player_position = game_state.get("player_position")

        # Return the initial state
        return self._get_state()

    def step(self, action):
        """
        Take a step in the environment by performing an action.

        Args:
            action (int): Index of the action to perform

        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Map action index to actual action
        game_action = self._map_action(action)

        # Perform action
        self.controller.perform_action(game_action)

        # Skip frames for performance
        for _ in range(self.frame_skip):
            self._capture_frame()

        # Get current state
        next_state = self._get_state()

        # Detect game state for reward calculation
        game_state = self.capture.detect_game_state(self.current_frame)

        # Calculate reward
        reward = self._calculate_reward(game_state)

        # Check if game is done
        done = self._is_done(game_state)

        # Additional info
        info = {
            "health": game_state.get("player_health"),
            "player_position": game_state.get("player_position"),
            "enemies": game_state.get("enemies"),
            "items": game_state.get("items"),
            "doors": game_state.get("doors"),
        }

        # Save debug frame occasionally
        if np.random.random() < 0.01:  # 1% chance each step
            self.capture.save_debug_frame(
                self.current_frame, game_state, f"debug_frame_{time.time()}.jpg"
            )

        return next_state, reward, done, info

    def _capture_frame(self):
        """Capture a frame and add it to the buffer"""
        self.current_frame = self.capture.capture_game_window(self.capture_region)
        processed_frame = self.capture.process_frame(self.current_frame)

        # Add to frame buffer and maintain max size
        self.frames_buffer.append(processed_frame)
        if len(self.frames_buffer) > self.max_frames_in_buffer:
            self.frames_buffer.pop(0)

    def _get_state(self):
        """
        Get the current state representation (stacked frames).

        Returns:
            numpy.ndarray: Stacked frames representing the current state
        """
        # Stack frames along a new axis
        return np.stack(self.frames_buffer, axis=0)

    def _map_action(self, action_idx):
        """
        Map an action index to an Action enum value.

        Args:
            action_idx (int): Index of the action

        Returns:
            Action: The corresponding Action enum value
        """
        # Define mapping from indices to actions
        actions = list(Action)
        if 0 <= action_idx < len(actions):
            return actions[action_idx]
        return Action.NONE

    def _calculate_reward(self, game_state):
        """
        Calculate the reward based on the game state.

        Args:
            game_state (dict): Current game state information

        Returns:
            float: The calculated reward
        """
        reward = 0.0

        # Health change reward
        current_health = game_state.get("player_health")
        if current_health is not None and self.previous_health is not None:
            health_change = current_health - self.previous_health
            if health_change < 0:
                # Penalty for losing health
                reward += health_change * self.reward_weights["health_loss"]
            elif health_change > 0:
                # Reward for gaining health
                reward += health_change * self.reward_weights["health_gain"]

        # Enemy defeated reward
        current_enemies = game_state.get("enemies", [])
        if len(current_enemies) < len(self.previous_enemies):
            # Reward for each enemy killed
            enemies_killed = len(self.previous_enemies) - len(current_enemies)
            reward += enemies_killed * self.reward_weights["enemy_killed"]

        # Item collection reward
        current_items = game_state.get("items", [])
        if len(current_items) < len(self.previous_items):
            # Reward for each item collected (fewer items visible means they were picked up)
            items_collected = len(self.previous_items) - len(current_items)
            reward += items_collected * self.reward_weights["item_collected"]

        # Room cleared reward
        if len(current_enemies) == 0 and len(self.previous_enemies) > 0:
            # All enemies cleared from the room
            if not self.current_room_cleared:
                reward += self.reward_weights["room_cleared"]
                self.current_room_cleared = True

        # Room exploration reward
        current_doors = game_state.get("doors", {})
        if current_doors != self.previous_room_doors:
            # Room changed
            room_signature = str(sorted(current_doors.items()))
            if room_signature not in self.rooms_visited:
                reward += self.reward_weights["room_explored"]
                self.rooms_visited.add(room_signature)
            self.current_room_cleared = False  # Reset for new room

        # Movement reward
        current_position = game_state.get("player_position")
        if current_position and self.previous_player_position:
            # Calculate distance moved
            x1, y1 = current_position
            x2, y2 = self.previous_player_position
            distance_moved = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            if distance_moved > 5:  # Threshold to avoid noise
                reward += self.reward_weights["movement"]
            else:
                # Small penalty for staying still to encourage exploration
                reward += self.reward_weights["staying_still"]

            # Door approach reward
            for direction, exists in current_doors.items():
                if exists:
                    # Check if player is approaching a door
                    door_approached = False
                    if direction == "north" and y1 < y2 and y1 < self.current_frame.shape[0] * 0.2:
                        door_approached = True
                    elif (
                        direction == "south" and y1 > y2 and y1 > self.current_frame.shape[0] * 0.8
                    ):
                        door_approached = True
                    elif direction == "east" and x1 > x2 and x1 > self.current_frame.shape[1] * 0.8:
                        door_approached = True
                    elif direction == "west" and x1 < x2 and x1 < self.current_frame.shape[1] * 0.2:
                        door_approached = True

                    if door_approached:
                        reward += self.reward_weights["door_approach"]

        # Update previous state values
        self.previous_health = current_health
        self.previous_enemies = current_enemies
        self.previous_items = current_items
        self.previous_room_doors = current_doors
        self.previous_player_position = current_position

        return reward

    def _is_done(self, game_state):
        """
        Check if the episode is done (game over or won).

        Args:
            game_state (dict): Current game state information

        Returns:
            bool: True if the episode is done, False otherwise
        """
        # Check if health is zero
        health = game_state.get("player_health")
        if health is not None and health <= 0:
            return True

        # Check if player is not detected (might be in a transition screen or game over)
        if game_state.get("player_position") is None:
            # This could be a false positive, so we might want to add more checks
            # For example, check if player hasn't been detected for several consecutive frames
            return False

        return False

    def close(self):
        """Clean up resources"""
        self.controller.reset_inputs()
