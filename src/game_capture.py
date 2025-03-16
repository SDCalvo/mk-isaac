import numpy as np
import cv2
from mss import mss
from PIL import Image

class GameCapture:
    def __init__(self):
        """Initialize the game capture with MSS screen capture."""
        self.sct = mss()
        self.monitor = self.sct.monitors[1]  # Primary monitor
        
    def capture_game_window(self, region=None):
        """
        Capture a specific region of the screen.
        Args:
            region (tuple): (left, top, width, height) of capture region
        Returns:
            numpy.ndarray: Captured frame in RGB format
        """
        if region is None:
            # Default to full monitor
            region = {
                "left": self.monitor["left"],
                "top": self.monitor["top"],
                "width": self.monitor["width"],
                "height": self.monitor["height"]
            }
        else:
            left, top, width, height = region
            region = {
                "left": left,
                "top": top,
                "width": width,
                "height": height
            }
            
        screenshot = self.sct.grab(region)
        # Convert to PIL Image and then to RGB numpy array
        frame = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        return np.array(frame)
    
    def process_frame(self, frame):
        """
        Process the captured frame for AI input.
        Args:
            frame (numpy.ndarray): Raw captured frame
        Returns:
            numpy.ndarray: Processed frame ready for the AI model
        """
        # Resize to a standard size (e.g., 84x84 as commonly used in DQN)
        processed = cv2.resize(frame, (84, 84))
        # Convert to grayscale
        processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        # Normalize pixel values
        processed = processed / 255.0
        return processed
    
    def detect_game_state(self, frame):
        """
        Detect the current state of the game (room layout, enemies, items, etc.)
        Args:
            frame (numpy.ndarray): Processed frame
        Returns:
            dict: Game state information
        """
        # TODO: Implement game state detection
        # This will include:
        # - Player position and health
        # - Enemy positions and types
        # - Item locations
        # - Room layout
        return {
            "player_health": None,
            "enemies": [],
            "items": [],
            "room_layout": None
        }

if __name__ == "__main__":
    # Test the capture functionality
    import time
    
    capture = GameCapture()
    
    # Capture 10 frames as a test
    for _ in range(10):
        frame = capture.capture_game_window()
        processed = capture.process_frame(frame)
        print(f"Frame shape: {frame.shape}, Processed shape: {processed.shape}")
        time.sleep(0.1)  # Small delay between captures
