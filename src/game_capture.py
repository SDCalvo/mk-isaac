"""
Capture the game window and detect game state.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict

# Type information for cv2 methods is provided in cv2_stubs.py
# pyright: reportMissingImports=false
# pyright: reportArgumentType=false
import cv2
import numpy as np
from mss import mss
from PIL import Image
import win32gui
import win32con
import time
import win32api
from ctypes import Structure, c_long, c_ulong, sizeof, POINTER, pointer
from ctypes import windll


# Define TypedDict for ROI dictionaries
class RoiDict(TypedDict):
    """
    Dictionary for regions of interest.
    """
    left: int
    top: int
    width: int
    height: int


class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

class MOUSEINPUT(Structure):
    _fields_ = [("dx", c_long),
                ("dy", c_long),
                ("mouseData", c_ulong),
                ("dwFlags", c_ulong),
                ("time", c_ulong),
                ("dwExtraInfo", POINTER(c_ulong))]

class INPUT(Structure):
    _fields_ = [("type", c_ulong),
                ("mi", MOUSEINPUT)]

def unpause_game(hwnd):
    """Send Escape key to unpause the game"""
    # Send Escape key to unpause
    windll.user32.PostMessageW(hwnd, win32con.WM_KEYDOWN, win32con.VK_ESCAPE, 0)
    time.sleep(0.1)  # Small delay between key down and up
    windll.user32.PostMessageW(hwnd, win32con.WM_KEYUP, win32con.VK_ESCAPE, 0)
    time.sleep(0.2)  # Wait for unpause to take effect

def focus_and_unpause(hwnd):
    """Focus the window and unpause the game"""
    # Restore window if minimized
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    time.sleep(0.1)  # Wait for window to restore
    
    # Set foreground window
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.2)  # Wait for focus
    
    # Unpause the game
    unpause_game(hwnd)

def click_window(hwnd):
    """Click on the window to properly focus it and unpause the game"""
    # Get window rect
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    # Click in the middle of the window
    x = (left + right) // 2
    y = (top + bottom) // 2
    
    # Store current mouse position
    point = POINT()
    windll.user32.GetCursorPos(pointer(point))
    old_x, old_y = point.x, point.y
    
    # Move mouse and click
    windll.user32.SetCursorPos(x, y)
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
    time.sleep(0.5)  # Wait longer for focus
    
    # Send Escape key message directly to the window
    windll.user32.PostMessageW(hwnd, win32con.WM_KEYDOWN, win32con.VK_ESCAPE, 0)
    time.sleep(0.1)
    windll.user32.PostMessageW(hwnd, win32con.WM_KEYUP, win32con.VK_ESCAPE, 0)
    time.sleep(0.5)  # Wait longer for unpause
    
    # Restore mouse position
    windll.user32.SetCursorPos(old_x, old_y)

def log_focused_window():
    """Log information about the currently focused window"""
    try:
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        return f"Focused window: '{title}' (hwnd: {hwnd})"
    except Exception as e:
        return f"Error getting focused window: {e}"

def is_window_active(hwnd):
    """Check if the given window is the active window"""
    return hwnd == win32gui.GetForegroundWindow()

class GameCapture:
    """
    Capture the game window and detect game state.
    """
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

    def calibrate(self, __frame: np.ndarray) -> None:
        """
        Calibrate the detection regions based on a reference frame.
        This needs to be called once at the start with a clear view of the game UI.

        Args:
            _frame (numpy.ndarray): A reference frame from the game
        """
        # Get frame dimensions
        height, width = __frame.shape[:2]

        # Set ROIs based on frame size
        # Health bar is in the top-left corner (smaller than minimap)
        self.roi["health_bar"] = {
            "left": int(width * 0.05),  # Left side of screen
            "top": int(height * 0.05),  # Near the top
            "width": int(width * 0.15),  # Smaller width for health bar
            "height": int(height * 0.1),  # Smaller height for health bar
        }

        # Minimap is in the top-right corner (larger than health bar)
        self.roi["minimap"] = {
            "left": int(width * 0.8),  # Right side of screen
            "top": int(height * 0.05),  # Near the top
            "width": int(width * 0.2),  # Larger width for minimap
            "height": int(height * 0.2),  # Larger height for minimap
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

    def capture_game_window(self, region: Optional[Tuple[int, int, int, int]] = None, no_unpause: bool = False) -> np.ndarray:
        """
        Capture The Binding of Isaac game window.
        Args:
            region (tuple): (left, top, width, height) of capture region
            no_unpause (bool): If True, don't try to focus or unpause the window
        Returns:
            numpy.ndarray: Captured frame in RGB format
        """
        try:
            # Try to find The Binding of Isaac window
            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if "isaac" in title.lower():  # Case insensitive search for "isaac" in window title
                        windows.append(hwnd)
                return True

            windows = []
            win32gui.EnumWindows(callback, windows)

            self.logger.info("Before any window operations: " + log_focused_window())

            if windows:
                # Get the first Isaac window found
                hwnd = windows[0]
                title = win32gui.GetWindowText(hwnd)
                self.logger.info(f"Found Isaac window: '{title}' (hwnd: {hwnd})")
                
                # Only focus and unpause if not explicitly disabled
                if not no_unpause:
                    # Make sure window is in normal state (not minimized)
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    time.sleep(0.1)
                    
                    # Get window rect
                    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                    
                    # Try to activate the window
                    win32gui.BringWindowToTop(hwnd)
                    win32gui.SetForegroundWindow(hwnd)
                    time.sleep(0.2)  # Give it more time to activate
                    
                    # Check if we successfully got focus
                    if is_window_active(hwnd):
                        self.logger.info("Successfully activated Isaac window")
                        # Send Escape key only if we have focus
                        windll.user32.PostMessageW(hwnd, win32con.WM_KEYDOWN, win32con.VK_ESCAPE, 0)
                        time.sleep(0.05)
                        windll.user32.PostMessageW(hwnd, win32con.WM_KEYUP, win32con.VK_ESCAPE, 0)
                        
                        # Add a MUCH longer delay to allow the pause menu to fully transition out
                        self.logger.info("Waiting for pause menu to fully transition out...")
                        time.sleep(1.0)  # Wait long enough for transition animation to complete
                    else:
                        self.logger.warning("Failed to activate Isaac window!")
                else:
                    self.logger.info("Skipping window focus and unpause as requested")
                
                # Get window rect
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                region_dict = {
                    "left": left,
                    "top": top,
                    "width": right - left,
                    "height": bottom - top,
                }
            else:
                self.logger.warning("No Isaac window found!")
                # If no Isaac window found, use provided region or default to monitor
                if region is None:
                    region_dict = {
                        "left": self.monitor["left"],
                        "top": self.monitor["top"],
                        "width": self.monitor["width"],
                        "height": self.monitor["height"],
                    }
                else:
                    left, top, width, height = region
                    region_dict = {"left": left, "top": top, "width": width, "height": height}

            self.logger.info("Before taking screenshot: " + log_focused_window())
            # Take screenshot
            screenshot = self.sct.grab(region_dict)
            self.logger.info("After taking screenshot: " + log_focused_window())
            
            # Convert to PIL Image and then to RGB numpy array
            frame = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            return np.array(frame)
        except Exception as e:
            self.logger.error(f"Error capturing game window: {e}")
            # Return a black frame as fallback
            return np.zeros((self.monitor["height"], self.monitor["width"], 3), dtype=np.uint8)

    def process__frame(self, __frame: np.ndarray) -> np.ndarray:
        """
        Process the captured _frame for AI input.
        Args:
            _frame (numpy.ndarray): Raw captured _frame
        Returns:
            numpy.ndarray: Processed _frame ready for the AI model
        """
        # Resize to a standard size (e.g., 84x84 as commonly used in DQN)
        processed = cv2.resize(__frame, (84, 84))
        # Convert to grayscale
        processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        # Normalize pixel values
        processed = processed / 255.0
        return processed

    def detect_health(self, __frame: np.ndarray) -> int:
        """
        Detect player's health from the _frame.
        This uses color detection to identify red hearts in The Binding of Isaac.

        Args:
            _frame (numpy.ndarray): Full game _frame

        Returns:
            float: Estimated health value
        """
        if not self.is_calibrated:
            self.calibrate(__frame)

        health_bar = self.roi["health_bar"]
        if health_bar is None:
            # If we couldn't calibrate the health bar region, return a default value
            self.logger.warning("Health bar region not calibrated")
            return 3  # Default to middle health value

        # Type checking to help linter
        assert health_bar is not None

        # Extract health bar region
        health_region = __frame[
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

    def detect_enemies(self, _frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect enemies in the _frame.

        Args:
            _frame (numpy.ndarray): Full game _frame

        Returns:
            list: List of dictionaries containing enemy positions and estimated types
        """
        # Convert to HSV for better color-based detection
        hsv = cv2.cvtColor(_frame, cv2.COLOR_RGB2HSV)

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

    def detect_doors(self, _frame: np.ndarray) -> Dict[str, bool]:
        """
        Detect doors in the room.

        Args:
            _frame (numpy.ndarray): Full game _frame

        Returns:
            dict: Dictionary with door directions and states (open/closed)
        """
        # For simplicity, we'll just detect if there are doors in cardinal directions
        # A more robust approach would use template matching

        # Get _frame dimensions
        height, width = _frame.shape[:2]

        # Check for doors in cardinal directions by examining edges of the _frame
        # These are very approximate and would need calibration
        door_regions = {
            "north": _frame[
                int(height * 0.05) : int(height * 0.15), int(width * 0.45) : int(width * 0.55)
            ],
            "east": _frame[
                int(height * 0.45) : int(height * 0.55), int(width * 0.85) : int(width * 0.95)
            ],
            "south": _frame[
                int(height * 0.85) : int(height * 0.95), int(width * 0.45) : int(width * 0.55)
            ],
            "west": _frame[
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

    def detect_player(self, _frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect the player character in the _frame.

        Args:
            _frame (numpy.ndarray): Full game _frame

        Returns:
            tuple: (x, y) position of the player, or None if not detected
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(_frame, cv2.COLOR_RGB2HSV)

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

    def detect_items(self, _frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect collectible items in the room.

        Args:
            _frame (numpy.ndarray): Full game _frame

        Returns:
            list: List of detected item positions
        """
        # This is a simplified implementation
        # A more robust approach would use object detection or template matching

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(_frame, cv2.COLOR_RGB2HSV)

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

    def detect_game_state(self, _frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect the current state of the game (room layout, enemies, items, etc.)
        Args:
            _frame (numpy.ndarray): Raw _frame (not the processed 84x84 one)
        Returns:
            dict: Game state information
        """
        if not self.is_calibrated:
            self.calibrate(_frame)

        game_state = {
            "player_health": self.detect_health(_frame),
            "player_position": self.detect_player(_frame),
            "enemies": self.detect_enemies(_frame),
            "doors": self.detect_doors(_frame),
            "items": self.detect_items(_frame),
            "room_layout": None,  # This would require more complex analysis
        }

        return game_state

    def save_debug__frame(
        self, _frame: np.ndarray, game_state: Dict[str, Any], filename: str = "debug__frame.jpg"
    ) -> None:
        """
        Save a debug _frame with annotations showing what was detected.
        Useful for debugging and visualizing the detection algorithms.

        Args:
            _frame (numpy.ndarray): Original game _frame
            game_state (dict): Detected game state information
            filename (str): Output filename
        """
        # Make a copy of the _frame to draw on
        debug__frame = _frame.copy()

        # Draw player position
        if game_state["player_position"]:
            x, y = game_state["player_position"]
            cv2.circle(debug__frame, (x, y), 15, (0, 255, 0), 2)
            cv2.putText(
                debug__frame, "Player", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        # Draw enemies
        for i, enemy in enumerate(game_state["enemies"]):
            x, y = enemy["position"]
            cv2.circle(debug__frame, (x, y), 10, (0, 0, 255), 2)
            cv2.putText(
                debug__frame, f"Enemy {i}", (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )

        # Draw items
        for i, item_pos in enumerate(game_state["items"]):
            x, y = item_pos
            cv2.circle(debug__frame, (x, y), 5, (255, 255, 0), 2)
            cv2.putText(
                debug__frame,
                f"Item {i}",
                (x + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        # Draw health info
        health_text = f"Health: {game_state['player_health']}"
        cv2.putText(debug__frame, health_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Draw door info
        door_text = "Doors: " + ", ".join(
            [d for d, exists in game_state["doors"].items() if exists]
        )
        cv2.putText(
            debug__frame, door_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        # Save the image with IMWRITE_JPEG_QUALITY flag to ensure overwrite
        cv2.imwrite(filename, cv2.cvtColor(debug__frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.logger.info(f"Saved debug _frame to {filename}")


if __name__ == "__main__":
    # Test the capture functionality and game state detection
    import time

    capture = GameCapture()

    print("Starting capture and detection test...")
    print("Please have The Binding of Isaac game window visible.")
    print("Capturing in 3 seconds...")
    time.sleep(3)

    # Capture _frame and detect game state
    _frame = capture.capture_game_window()
    game_state = capture.detect_game_state(_frame)

    # Save debug _frame
    capture.save_debug__frame(_frame, game_state)

    # Print detected state
    print("\nDetected Game State:")
    print(f"Player Health: {game_state['player_health']}")
    print(f"Player Position: {game_state['player_position']}")
    print(f"Number of Enemies: {len(game_state['enemies'])}")
    print(f"Doors: {[d for d, exists in game_state['doors'].items() if exists]}")
    print(f"Number of Items: {len(game_state['items'])}")

    print("\nSaved debug _frame to debug__frame.jpg")
    print("Review this image to adjust detection parameters as needed.")
