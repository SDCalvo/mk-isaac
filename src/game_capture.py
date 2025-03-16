import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

# Type information for cv2 methods is provided in cv2_stubs.py
# pyright: reportMissingImports=false
# pyright: reportArgumentType=false
import cv2
import numpy as np
from mss import mss
from PIL import Image


# Define TypedDict for ROI dictionaries
class RoiDict(TypedDict):
    left: int
    top: int
    width: int
    height: int


class GameCapture:
    def __init__(self):
        """Initialize the game capture with MSS screen capture."""
        self.sct = mss()
        self.monitor = self.sct.monitors[1]  # Primary monitor

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("GameCapture")

        # Define regions of interest (ROIs) for different game elements
        # These will need to be calibrated based on your specific game window size
        self.roi: Dict[str, Optional[RoiDict]] = {
            "health_bar": None,  # Will be set during calibration
            "minimap": None,  # Region for the minimap/floor layout
            "item_display": None,  # Region for currently held items
        }

        # Reference images for template matching
        self.templates = {
            "health_full": None,
            "health_half": None,
            "health_empty": None,
            "door_closed": None,
            "door_open": None,
        }

        # Color ranges for object detection
        self.color_ranges = {
            "player": {
                "lower": np.array([200, 0, 0], dtype=np.uint8),
                "upper": np.array([255, 100, 100], dtype=np.uint8),
            },  # Red for player
            "enemy": {
                "lower": np.array([0, 0, 100], dtype=np.uint8),
                "upper": np.array([100, 100, 255], dtype=np.uint8),
            },  # Blue for enemies
            "pickup": {
                "lower": np.array([200, 200, 0], dtype=np.uint8),
                "upper": np.array([255, 255, 100], dtype=np.uint8),
            },  # Yellow for pickups
        }

        # Flag to indicate if calibration has been performed
        self.is_calibrated = False

    def calibrate(self, frame: np.ndarray) -> None:
        """
        Calibrate the detection regions based on a reference frame.
        This needs to be called once at the start with a clear view of the game UI.

        Args:
            frame (numpy.ndarray): A reference frame from the game
        """
        # Get frame dimensions
        height, width = frame.shape[:2]

        # Set ROIs based on frame size
        # These values are approximate and should be adjusted based on testing
        # For The Binding of Isaac, the health display is typically in the top-right corner
        self.roi["health_bar"] = {
            "left": int(width * 0.8),  # Right side of screen
            "top": int(height * 0.05),  # Near the top
            "width": int(width * 0.15),
            "height": int(height * 0.1),
        }

        # Minimap is usually in the top-left corner
        self.roi["minimap"] = {
            "left": int(width * 0.05),
            "top": int(height * 0.05),
            "width": int(width * 0.2),
            "height": int(height * 0.2),
        }

        # Item display is often at the bottom-right
        self.roi["item_display"] = {
            "left": int(width * 0.8),
            "top": int(height * 0.8),
            "width": int(width * 0.15),
            "height": int(height * 0.15),
        }

        self.logger.info("Calibration complete with frame size: %dx%d", width, height)
        self.is_calibrated = True

        # TODO: Load templates for health, doors, etc. if using template matching

    def capture_game_window(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Capture a specific region of the screen.
        Args:
            region (tuple): (left, top, width, height) of capture region
        Returns:
            numpy.ndarray: Captured frame in RGB format
        """
        if region is None:
            # Default to full monitor
            region_dict = {
                "left": self.monitor["left"],
                "top": self.monitor["top"],
                "width": self.monitor["width"],
                "height": self.monitor["height"],
            }
        else:
            left, top, width, height = region
            region_dict = {"left": left, "top": top, "width": width, "height": height}

        screenshot = self.sct.grab(region_dict)
        # Convert to PIL Image and then to RGB numpy array
        frame = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        return np.array(frame)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
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

    def detect_health(self, frame: np.ndarray) -> int:
        """
        Detect player's health from the frame.
        This uses color detection to identify red hearts in The Binding of Isaac.

        Args:
            frame (numpy.ndarray): Full game frame

        Returns:
            float: Estimated health value
        """
        if not self.is_calibrated:
            self.calibrate(frame)

        health_bar = self.roi["health_bar"]
        if health_bar is None:
            # If we couldn't calibrate the health bar region, return a default value
            self.logger.warning("Health bar region not calibrated")
            return 3  # Default to middle health value

        # Type checking to help linter
        assert health_bar is not None

        # Extract health bar region
        health_region = frame[
            health_bar["top"] : health_bar["top"] + health_bar["height"],
            health_bar["left"] : health_bar["left"] + health_bar["width"],
        ]

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(health_region, cv2.COLOR_RGB2HSV)

        # Define red heart color range in HSV
        # Isaac uses red hearts, so we're looking for red color
        lower_red1 = np.array([0, 100, 100], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([160, 100, 100], dtype=np.uint8)  # Red wraps in HSV, so we need two ranges
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        # Create masks for each red range
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Count non-zero pixels in the mask (red pixels)
        red_pixel_count = cv2.countNonZero(mask)

        # Calculate health based on red pixel count
        # This is very approximate and needs calibration
        # A more robust approach would be template matching heart icons
        max_health_pixels = health_bar["width"] * health_bar["height"] * 0.5
        health_ratio = min(1.0, red_pixel_count / max_health_pixels)

        # Convert to health value (assuming max health is 6 - 3 hearts)
        # In Isaac, health is measured in half-hearts, so 6 = 3 full hearts
        estimated_health = round(health_ratio * 6)

        self.logger.debug(
            f"Health detection: red pixels: {red_pixel_count}, estimated health: {estimated_health}"
        )

        return estimated_health

    def detect_enemies(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect enemies in the frame.

        Args:
            frame (numpy.ndarray): Full game frame

        Returns:
            list: List of dictionaries containing enemy positions and estimated types
        """
        # Convert to HSV for better color-based detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Define a color range for enemies (this would need refinement)
        # Many enemies in Isaac appear as darker entities against the background
        lower_bound = np.array([0, 0, 0], dtype=np.uint8)  # Very dark colors
        upper_bound = np.array([180, 150, 100], dtype=np.uint8)  # Not too saturated and not too bright

        # Create a mask for enemy detection
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        enemies = []
        # Filter contours by size to identify potential enemies
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter out very small or very large contours
            if 100 < area < 5000:  # Adjust these thresholds based on testing
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                enemies.append({"position": (center_x, center_y), "size": max(w, h), "area": area})

        self.logger.debug(f"Detected {len(enemies)} potential enemies")
        return enemies

    def detect_doors(self, frame: np.ndarray) -> Dict[str, bool]:
        """
        Detect doors in the room.

        Args:
            frame (numpy.ndarray): Full game frame

        Returns:
            dict: Dictionary with door directions and states (open/closed)
        """
        # For simplicity, we'll just detect if there are doors in cardinal directions
        # A more robust approach would use template matching

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Check for doors in cardinal directions by examining edges of the frame
        # These are very approximate and would need calibration
        door_regions = {
            "north": frame[
                int(height * 0.05) : int(height * 0.15), int(width * 0.45) : int(width * 0.55)
            ],
            "east": frame[
                int(height * 0.45) : int(height * 0.55), int(width * 0.85) : int(width * 0.95)
            ],
            "south": frame[
                int(height * 0.85) : int(height * 0.95), int(width * 0.45) : int(width * 0.55)
            ],
            "west": frame[
                int(height * 0.45) : int(height * 0.55), int(width * 0.05) : int(width * 0.15)
            ],
        }

        doors = {}
        for direction, region in door_regions.items():
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

            # Check if the variance of the region is high (indicating a door)
            variance = np.var(gray)

            # If variance is high, there might be a door
            doors[direction] = variance > 200  # Threshold would need calibration

        self.logger.debug(f"Detected doors: {doors}")
        return doors

    def detect_player(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect the player character in the frame.

        Args:
            frame (numpy.ndarray): Full game frame

        Returns:
            tuple: (x, y) position of the player, or None if not detected
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Define color range for Isaac (typically pale/flesh colored)
        lower_bound = np.array([0, 20, 150], dtype=np.uint8)  # Light skin tone in HSV
        upper_bound = np.array([30, 150, 255], dtype=np.uint8)

        # Create a mask for player detection
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest contour which is likely to be the player
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # If the contour is too small, it's probably not the player
        if area < 100:
            return None

        # Get the center of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        self.logger.debug(f"Detected player at position: ({cx}, {cy})")
        return (cx, cy)

    def detect_items(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect collectible items in the room.

        Args:
            frame (numpy.ndarray): Full game frame

        Returns:
            list: List of detected item positions
        """
        # This is a simplified implementation
        # A more robust approach would use object detection or template matching

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Define color range for items (typically bright and colorful)
        lower_bound = np.array([20, 100, 200], dtype=np.uint8)  # Bright, saturated colors
        upper_bound = np.array([140, 255, 255], dtype=np.uint8)

        # Create a mask for item detection
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        items = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by size to exclude noise and large objects
            if 20 < area < 500:  # Adjust thresholds based on testing
                x, y, w, h = cv2.boundingRect(contour)
                items.append((x + w // 2, y + h // 2))

        self.logger.debug(f"Detected {len(items)} potential items")
        return items

    def detect_game_state(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect the current state of the game (room layout, enemies, items, etc.)
        Args:
            frame (numpy.ndarray): Raw frame (not the processed 84x84 one)
        Returns:
            dict: Game state information
        """
        if not self.is_calibrated:
            self.calibrate(frame)

        game_state = {
            "player_health": self.detect_health(frame),
            "player_position": self.detect_player(frame),
            "enemies": self.detect_enemies(frame),
            "doors": self.detect_doors(frame),
            "items": self.detect_items(frame),
            "room_layout": None,  # This would require more complex analysis
        }

        return game_state

    def save_debug_frame(
        self, frame: np.ndarray, game_state: Dict[str, Any], filename: str = "debug_frame.jpg"
    ) -> None:
        """
        Save a debug frame with annotations showing what was detected.
        Useful for debugging and visualizing the detection algorithms.

        Args:
            frame (numpy.ndarray): Original game frame
            game_state (dict): Detected game state information
            filename (str): Output filename
        """
        # Make a copy of the frame to draw on
        debug_frame = frame.copy()

        # Draw player position
        if game_state["player_position"]:
            x, y = game_state["player_position"]
            cv2.circle(debug_frame, (x, y), 15, (0, 255, 0), 2)
            cv2.putText(
                debug_frame, "Player", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        # Draw enemies
        for i, enemy in enumerate(game_state["enemies"]):
            x, y = enemy["position"]
            cv2.circle(debug_frame, (x, y), 10, (0, 0, 255), 2)
            cv2.putText(
                debug_frame, f"Enemy {i}", (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )

        # Draw items
        for i, item_pos in enumerate(game_state["items"]):
            x, y = item_pos
            cv2.circle(debug_frame, (x, y), 5, (255, 255, 0), 2)
            cv2.putText(
                debug_frame,
                f"Item {i}",
                (x + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        # Draw health info
        health_text = f"Health: {game_state['player_health']}"
        cv2.putText(debug_frame, health_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Draw door info
        door_text = "Doors: " + ", ".join(
            [d for d, exists in game_state["doors"].items() if exists]
        )
        cv2.putText(
            debug_frame, door_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        # Save the image
        cv2.imwrite(filename, cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
        self.logger.info(f"Saved debug frame to {filename}")


if __name__ == "__main__":
    # Test the capture functionality and game state detection
    import time

    capture = GameCapture()

    print("Starting capture and detection test...")
    print("Please have The Binding of Isaac game window visible.")
    print("Capturing in 3 seconds...")
    time.sleep(3)

    # Capture frame and detect game state
    frame = capture.capture_game_window()
    game_state = capture.detect_game_state(frame)

    # Save debug frame
    capture.save_debug_frame(frame, game_state)

    # Print detected state
    print("\nDetected Game State:")
    print(f"Player Health: {game_state['player_health']}")
    print(f"Player Position: {game_state['player_position']}")
    print(f"Number of Enemies: {len(game_state['enemies'])}")
    print(f"Doors: {[d for d, exists in game_state['doors'].items() if exists]}")
    print(f"Number of Items: {len(game_state['items'])}")

    print("\nSaved debug frame to debug_frame.jpg")
    print("Review this image to adjust detection parameters as needed.")
