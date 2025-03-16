"""
Capture the game window and detect game state.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import json
import os

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
import win32ui
import hashlib
from collections import defaultdict


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
    A simplified class to capture game state from The Binding of Isaac.
    Focused on essential elements for reinforcement learning:
    - Health
    - Floor number
    - Room exploration tracking
    - Game over detection
    """
    
    def __init__(self):
        """Initialize the GameCapture object."""
        self.roi_data = self._load_roi_data()
        self.game_window_handle = None
        self.game_window_rect = None
        self.explored_rooms = set()  # Store hashes of explored rooms
        self.debug = True  # Enable debug image saving
        
        # Create debug folder if debug is enabled
        if self.debug:
            os.makedirs("debug_images", exist_ok=True)
    
    def _load_roi_data(self):
        """Load ROI data from json file."""
        try:
            with open("roi_coordinates.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Return default ROIs if file not found or invalid
            print("ROI data file not found or invalid. Using default values.")
            return {
                "health": {"x": 0.05, "y": 0.05, "width": 0.15, "height": 0.05},
                "minimap": {"x": 0.85, "y": 0.1, "width": 0.12, "height": 0.12},
                "game_area": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8},
                "game_window": {"width": 1920, "height": 1080}
            }
    
    def find_game_window(self):
        """Find the Binding of Isaac game window."""
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if "isaac" in window_text.lower():
                    windows.append((hwnd, window_text))
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        if not windows:
            return None, None
        
        # Use the first Isaac window found
        hwnd, window_name = windows[0]
        try:
            rect = win32gui.GetWindowRect(hwnd)
            return hwnd, rect
        except:
            return None, None
    
    def capture_game_window(self):
        """Capture a frame from the game window."""
        # Find game window if not already found
        if not self.game_window_handle:
            self.game_window_handle, self.game_window_rect = self.find_game_window()
            if not self.game_window_handle:
                print("Game window not found")
                return None
            
            # Set focus to the game window
            try:
                # First focus the window
                win32gui.ShowWindow(self.game_window_handle, win32con.SW_RESTORE)
                time.sleep(0.1)
                win32gui.SetForegroundWindow(self.game_window_handle)
                time.sleep(0.2)
                
                # Unpause the game with Escape key
                windll.user32.PostMessageW(self.game_window_handle, win32con.WM_KEYDOWN, win32con.VK_ESCAPE, 0)
                time.sleep(0.05)
                windll.user32.PostMessageW(self.game_window_handle, win32con.WM_KEYUP, win32con.VK_ESCAPE, 0)
                time.sleep(0.5)  # Wait for pause menu to disappear
                
                # Update window rectangle after focusing
                self.game_window_rect = win32gui.GetWindowRect(self.game_window_handle)
            except Exception as e:
                print(f"Could not set focus or unpause game window: {e}")
                return None  # Return None if we can't focus or unpause
        
        # Check if we have a valid window rectangle
        if not self.game_window_rect:
            print("Invalid window rectangle")
            self.game_window_handle = None  # Reset handle to force re-finding
            return None
        
        try:
            # Get window size
            left, top, right, bottom = self.game_window_rect
            width = right - left
            height = bottom - top
            
            # Create device context
            hwndDC = win32gui.GetWindowDC(self.game_window_handle)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # Create bitmap
            saveBitmap = win32ui.CreateBitmap()
            saveBitmap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitmap)
            
            # Copy window content to bitmap
            saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
            
            # Convert bitmap to image
            bmpinfo = saveBitmap.GetInfo()
            bmpstr = saveBitmap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype='uint8')
            img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
            
            # Clean up
            win32gui.DeleteObject(saveBitmap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.game_window_handle, hwndDC)
            
            # Convert to BGR format for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Save debug capture if needed
            if self.debug:
                cv2.imwrite(f"debug_images/capture_{time.time()}.jpg", img)
            
            return img
            
        except Exception as e:
            print(f"Error capturing game window: {e}")
            # Reset window handle to force re-finding it next time
            self.game_window_handle = None
            self.game_window_rect = None
            return None
    
    def get_game_state(self, frame):
        """
        Extract the current game state from a frame.
        
        Returns:
            dict: A dictionary with game state information:
                - health (float): Current health value
                - current_floor (int): Current floor number
                - is_game_over (bool): Whether the game is over
                - is_new_room (bool): Whether player has entered a new room
                - is_unexplored_room (bool): Whether this room is unexplored
        """
        if frame is None:
            return {
                "health": 0.0,
                "current_floor": 1,
                "is_game_over": False,
                "is_new_room": False,
                "is_unexplored_room": False
            }
        
        # Get ROIs from the frame
        health_roi = self._get_roi(frame, "health")
        minimap_roi = self._get_roi(frame, "minimap")
        game_area_roi = self._get_roi(frame, "game_area")
        
        # Detect health
        health = self._detect_health(health_roi)
        
        # Detect current floor from minimap
        floor = self._detect_floor(minimap_roi)
        
        # Check if game over
        is_game_over = self._detect_game_over(frame)
        
        # Generate unique hash for the current room based on game area
        current_room_hash = self._get_room_hash(game_area_roi)
        
        # Check if this is a new room the player has entered
        is_new_room = self._is_new_room(current_room_hash)
        
        # Check if this room has not been explored before
        is_unexplored_room = current_room_hash not in self.explored_rooms
        
        # Add to explored rooms set
        if is_new_room and is_unexplored_room:
            self.explored_rooms.add(current_room_hash)
        
        # Create game state dictionary
        game_state = {
            "health": health,
            "current_floor": floor,
            "is_game_over": is_game_over,
            "is_new_room": is_new_room,
            "is_unexplored_room": is_unexplored_room,
        }
        
        # Save debug image if debug is enabled
        if self.debug:
            self._save_debug_image(frame, game_state)
        
        return game_state
    
    def _get_roi(self, frame, roi_name):
        """Extract a region of interest from frame."""
        if frame is None or roi_name not in self.roi_data:
            return None
        
        roi = self.roi_data[roi_name]
        height, width = frame.shape[:2]
        
        # Use raw values if available, otherwise calculate from percentages
        if "x_raw" in roi and "y_raw" in roi and "width_raw" in roi and "height_raw" in roi:
            x = int(roi["x_raw"])
            y = int(roi["y_raw"])
            w = int(roi["width_raw"])
            h = int(roi["height_raw"])
        else:
            x = int(width * roi["x"])
            y = int(height * roi["y"])
            w = int(width * roi["width"])
            h = int(height * roi["height"])
        
        # Ensure within frame bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        
        return frame[y:y+h, x:x+w]
    
    def _detect_health(self, health_roi):
        """
        Detect player health from health ROI.
        Uses color detection to identify red hearts (filled) vs empty heart containers.
        """
        if health_roi is None:
            return 0.0
        
        # Save debug image if debug is enabled
        if self.debug:
            timestamp = time.time()
            debug_path = f"debug_images/health_roi_{timestamp}.jpg"
            cv2.imwrite(debug_path, health_roi)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(health_roi, cv2.COLOR_BGR2HSV)
        
        # Color masks for different heart types
        
        # Red hearts (filled) - these are the most important for health
        # Red in HSV wraps around 0/180
        lower_red1 = np.array([0, 150, 150])   # Increased saturation and value for more specific detection
        upper_red1 = np.array([15, 255, 255])  # Wider hue range to catch more red variants
        lower_red2 = np.array([160, 150, 150]) # Upper red hue range
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        
        # Save debug mask if debug is enabled
        if self.debug:
            cv2.imwrite(f"debug_images/health_mask_{timestamp}.jpg", clean_mask)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug image
        if self.debug:
            debug_img = health_roi.copy()
            # Draw all contours in blue first
            cv2.drawContours(debug_img, contours, -1, (255, 0, 0), 1)
        
        # Analyze contours to count hearts
        hearts = 0
        min_heart_area = 25  # Minimum area to be considered a heart
        max_heart_area = 500  # Maximum area for a heart
        
        # Look for contours that could be heart pieces (each half heart = 1 health point)
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_heart_area or area > max_heart_area:
                continue
            
            # Get bounding box for aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Hearts typically have aspect ratio close to 1 (roughly square)
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # This is likely a heart or part of a heart
            # The area will determine if it's a full or partial heart
            if area > 100:
                # Likely a full heart (2 health points)
                health_points = 2
                color = (0, 255, 0)  # Green for full heart
            else:
                # Likely a half heart (1 health point)
                health_points = 1
                color = (0, 255, 255)  # Yellow for half heart
            
            hearts += health_points
            
            # Draw for debug
            if self.debug:
                cv2.drawContours(debug_img, [contour], 0, color, 2)
                cv2.putText(
                    debug_img,
                    f"{health_points}hp",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
        
        # Cap health at a reasonable maximum
        max_health = 12  # 6 full hearts
        hearts = min(hearts, max_health)
        
        # Save debug visualization
        if self.debug:
            cv2.putText(
                debug_img,
                f"Total health: {hearts}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imwrite(f"debug_images/health_debug_{timestamp}.jpg", debug_img)
            print(f"Detected health: {hearts}")
        
        return hearts
    
    def _detect_floor(self, minimap_roi):
        """
        Track the current floor based on transition events.
        Since we can't directly detect the floor from visuals reliably,
        we'll need to track when the player moves between floors.
        """
        # This is a simple placeholder that returns our tracked floor
        # In a real implementation, we would detect floor transitions
        
        # Initialize floor number if we haven't yet
        if not hasattr(self, '_current_floor'):
            self._current_floor = 1
        
        # In an actual implementation, we would detect:
        # 1. Trapdoor usage
        # 2. Floor transition screens 
        # 3. Other visual cues indicating floor changes
        
        # For now, we'll just return our tracked floor
        # We can manually adjust this when testing
        
        # Debug info
        if self.debug and minimap_roi is not None:
            timestamp = time.time()
            cv2.imwrite(f"debug_images/minimap_{timestamp}.jpg", minimap_roi)
            
            # Create debug visualization
            debug_img = minimap_roi.copy()
            cv2.putText(
                debug_img, 
                f"Floor (tracked): {self._current_floor}", 
                (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                1
            )
            cv2.imwrite(f"debug_images/floor_debug_{timestamp}.jpg", debug_img)
        
        return self._current_floor
    
    def _detect_game_over(self, frame):
        """
        Detect if the game is over.
        Looks for the game over screen which has specific colors and text.
        """
        if frame is None:
            return False
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check if mostly dark with some bright text (game over screen)
        dark_ratio = np.sum(gray < 30) / (gray.shape[0] * gray.shape[1])
        bright_points = np.sum(gray > 200)
        
        # Debug visualization
        if self.debug:
            timestamp = time.time()
            debug_img = frame.copy()
            cv2.putText(debug_img, f"Dark ratio: {dark_ratio:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(debug_img, f"Bright points: {bright_points}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(f"debug_images/game_over_debug_{timestamp}.jpg", debug_img)
        
        # If mostly dark (>90%) but has some bright spots, likely game over screen
        return dark_ratio > 0.9 and bright_points > 1000
    
    def _get_room_hash(self, game_area_roi):
        """
        Generate a hash for the current room to track exploration.
        Uses downsampled, blurred game area to get a stable room identifier.
        """
        if game_area_roi is None:
            return "none"
            
        # Resize to small size to remove noise and details
        small = cv2.resize(game_area_roi, (32, 24))
        
        # Blur to further remove noise
        blurred = cv2.GaussianBlur(small, (3, 3), 0)
        
        # Convert to grayscale
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean hash of image
        # First, resize to 8x8
        resized = cv2.resize(gray, (8, 8))
        # Get average pixel value
        avg_pixel = resized.mean()
        # Create hash: 1 if pixel > avg, 0 otherwise
        hash_array = (resized > avg_pixel).flatten()
        # Convert to hash string
        hash_str = ''.join(['1' if x else '0' for x in hash_array])
        
        # Debug visualization
        if self.debug:
            timestamp = time.time()
            # Create visualization of room hash process
            debug_img = np.hstack([
                cv2.resize(game_area_roi, (320, 240)),
                cv2.resize(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB), (320, 240)),
                cv2.resize(cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR), (320, 240))
            ])
            cv2.putText(debug_img, f"Room hash: {hash_str[:16]}...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(f"debug_images/room_hash_debug_{timestamp}.jpg", debug_img)
        
        return hash_str
    
    def _is_new_room(self, current_hash):
        """
        Check if the current room is new (player just entered it).
        Uses a simple hash comparison approach.
        """
        # Check if we have a last room hash
        if not hasattr(self, '_last_room_hash'):
            self._last_room_hash = None
        
        # If same as last hash, not a new room
        if self._last_room_hash == current_hash:
            return False
        
        # Store current hash for next comparison
        is_new = True
        self._last_room_hash = current_hash
        
        return is_new
    
    def _save_debug_image(self, frame, game_state):
        """Save a debug image with game state visualization."""
        if frame is None:
            return
            
        debug_img = frame.copy()
        timestamp = time.time()
        
        # Draw game state info on the frame
        cv2.putText(debug_img, f"Health: {game_state['health']}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Floor: {game_state['current_floor']}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, f"New room: {game_state['is_new_room']}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Unexplored: {game_state['is_unexplored_room']}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Game over: {game_state['is_game_over']}", (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw ROI rectangles
        frame_h, frame_w = frame.shape[:2]
        
        for roi_name in ["health", "minimap", "game_area"]:
            if roi_name in self.roi_data:
                roi = self.roi_data[roi_name]
                
                # Use raw values if available, otherwise calculate from percentages
                if "x_raw" in roi and "y_raw" in roi and "width_raw" in roi and "height_raw" in roi:
                    x = int(roi["x_raw"])
                    y = int(roi["y_raw"])
                    w = int(roi["width_raw"])
                    h = int(roi["height_raw"])
                else:
                    x = int(frame_w * roi["x"])
                    y = int(frame_h * roi["y"])
                    w = int(frame_w * roi["width"])
                    h = int(frame_h * roi["height"])
                
                color = (0, 255, 0) if roi_name == "game_area" else (0, 0, 255)
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(debug_img, roi_name, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save the image
        cv2.imwrite(f"debug_images/game_state_{timestamp}.jpg", debug_img)

    def save_debug_frame(self, frame, game_state, filename):
        """Save a debug visualization of the game state to a file"""
        if frame is None:
            print("Cannot save debug frame - frame is None")
            return
        
        # Create a copy of the frame to draw on
        debug_img = frame.copy()
        
        # Draw game state info on the frame
        cv2.putText(debug_img, f"Health: {game_state['health']}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Floor: {game_state['current_floor']}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, f"New room: {game_state['is_new_room']}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Unexplored: {game_state['is_unexplored_room']}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Game over: {game_state['is_game_over']}", (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw ROI rectangles
        frame_h, frame_w = frame.shape[:2]
        
        for roi_name in ["health", "minimap", "game_area"]:
            if roi_name in self.roi_data:
                roi = self.roi_data[roi_name]
                
                # Use raw values if available, otherwise calculate from percentages
                if "x_raw" in roi and "y_raw" in roi and "width_raw" in roi and "height_raw" in roi:
                    x = int(roi["x_raw"])
                    y = int(roi["y_raw"])
                    w = int(roi["width_raw"])
                    h = int(roi["height_raw"])
                else:
                    x = int(frame_w * roi["x"])
                    y = int(frame_h * roi["y"])
                    w = int(frame_w * roi["width"])
                    h = int(frame_h * roi["height"])
                
                color = (0, 255, 0) if roi_name == "game_area" else (0, 0, 255)
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(debug_img, roi_name, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Create directories if needed
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        
        # Save the image
        cv2.imwrite(filename, debug_img)
        print(f"Saved debug visualization to {filename}")


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
    game_state = capture.get_game_state(_frame)

    # Save debug _frame
    capture.save_debug_frame(_frame, game_state, "debug_frame.jpg")

    # Print detected state
    print("\nDetected Game State:")
    print(f"Player Health: {game_state['health']}")
    print(f"Game Over: {game_state['is_game_over']}")
    print(f"New Room: {game_state['is_new_room']}")
    print(f"Unexplored Room: {game_state['is_unexplored_room']}")
    print(f"Current Floor: {game_state['current_floor']}")

    print("\nSaved debug _frame to debug_frame.jpg")
    print("Review this image to adjust detection parameters as needed.")
