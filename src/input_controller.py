import keyboard
import pyautogui
import time
from enum import Enum

class Action(Enum):
    """Possible actions in the game"""
    UP = 'w'
    DOWN = 's'
    LEFT = 'a'
    RIGHT = 'd'
    SHOOT_UP = 'up'
    SHOOT_DOWN = 'down'
    SHOOT_LEFT = 'left'
    SHOOT_RIGHT = 'right'
    BOMB = 'e'
    ITEM = 'q'
    NONE = 'none'

class InputController:
    def __init__(self):
        """Initialize the input controller with safety delay"""
        # Add a small delay between actions to prevent overwhelming the game
        self.action_delay = 0.05
        # Disable pyautogui's failsafe
        pyautogui.FAILSAFE = False
        
    def perform_action(self, action: Action):
        """
        Perform a game action by simulating keyboard input
        Args:
            action (Action): The action to perform
        """
        if action == Action.NONE:
            return
            
        # Press and release the corresponding key
        keyboard.press(action.value)
        time.sleep(self.action_delay)
        keyboard.release(action.value)
        
    def perform_actions(self, actions):
        """
        Perform multiple actions in sequence
        Args:
            actions (list[Action]): List of actions to perform
        """
        for action in actions:
            self.perform_action(action)
            
    def reset_inputs(self):
        """Reset all inputs to ensure no keys are stuck"""
        for action in Action:
            keyboard.release(action.value)

if __name__ == "__main__":
    # Test the input controller
    controller = InputController()
    
    print("Testing basic movements...")
    # Test basic movement
    test_sequence = [
        Action.UP,
        Action.RIGHT,
        Action.DOWN,
        Action.LEFT
    ]
    
    # Perform test sequence
    print("Starting in 3 seconds...")
    time.sleep(3)
    
    for action in test_sequence:
        print(f"Performing action: {action.name}")
        controller.perform_action(action)
        time.sleep(0.5)
    
    # Reset all inputs
    controller.reset_inputs()
    print("Test complete!")
