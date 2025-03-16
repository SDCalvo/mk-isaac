from game_capture import GameCapture
from input_controller import InputController, Action
import numpy as np
import time

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
        
        # Capture initial frame
        for _ in range(self.max_frames_in_buffer):
            self._capture_frame()
            
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
            "score": game_state.get("score"),
            "items": game_state.get("items")
        }
        
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
        
        # TODO: Implement proper reward function based on:
        # - Health changes (negative for damage)
        # - Enemies defeated
        # - Items collected
        # - Rooms cleared
        # - Progress in the game
        
        # Placeholder for health-based reward
        current_health = game_state.get("player_health")
        if current_health is not None and self.previous_health is not None:
            # Penalize for losing health
            health_change = current_health - self.previous_health
            reward += health_change * 5.0  # Higher weight for health
        
        # Update previous health
        self.previous_health = current_health
        
        return reward
    
    def _is_done(self, game_state):
        """
        Check if the episode is done (game over or won).
        
        Args:
            game_state (dict): Current game state information
            
        Returns:
            bool: True if the episode is done, False otherwise
        """
        # TODO: Implement proper done condition
        # Check if health is zero or game over screen is detected
        health = game_state.get("player_health")
        if health is not None and health <= 0:
            return True
            
        # Check for game over screen
        # This would require image recognition to detect game over screen
        
        return False
    
    def close(self):
        """Clean up resources"""
        self.controller.reset_inputs()
