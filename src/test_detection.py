"""
Test script for The Binding of Isaac game state detection.
This script will capture frames from the game and display the detected game state.
"""

import os
import time
import argparse
import cv2
import numpy as np
import win32gui
import win32con

try:
    from src.game_capture import GameCapture
except ImportError:
    from game_capture import GameCapture


def find_game_window():
    """Find the Isaac window and return its handle"""
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "isaac" in title.lower():
                windows.append((hwnd, title))
        return True

    windows = []
    win32gui.EnumWindows(callback, windows)
    
    if not windows:
        return None
        
    # Return the first Isaac window found
    return windows[0][0]


def focus_game_window(hwnd):
    """Focus the game window"""
    if not hwnd:
        return False
        
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.2)  # Small delay to ensure window is focused
        return True
    except:
        return False


def test_state_detection(num_frames=1, delay=1.0):
    """
    Test state detection by capturing frames and saving debug visualizations.

    Args:
        num_frames (int): Number of frames to capture
        delay (float): Delay between frames in seconds
    """
    # Create game capture instance
    capture = GameCapture()
    
    print("Starting simplified state detection test...")
    print("Please make sure The Binding of Isaac is running and visible.")
    
    # Find and focus the Isaac window
    print("Finding and focusing the game window...")
    isaac_hwnd = find_game_window()
    if isaac_hwnd:
        print(f"Found Isaac window. Focusing...")
        focus_game_window(isaac_hwnd)
    else:
        print("Could not find the Isaac window. Please focus it manually.")
    
    # Give time for the game to be properly focused
    time.sleep(1)
    
    print(f"Starting capture of {num_frames} frames...")
    for i in range(num_frames):
        print(f"\nCapturing frame {i+1}/{num_frames}...")
        
        # Capture the game window
        frame = capture.capture_game_window()
        
        if frame is None:
            print("Failed to capture game window. Make sure the game is running.")
            continue
            
        print(f"Frame captured, size: {frame.shape[1]}x{frame.shape[0]}")
            
        # Get game state
        game_state = capture.get_game_state(frame)
        
        # Print detected game state
        print("\nDetected game state:")
        print(f"Player Health: {game_state['health']}")
        print(f"Floor: {game_state['current_floor']}")
        print(f"New Room: {game_state['is_new_room']}")
        print(f"Unexplored Room: {game_state['is_unexplored_room']}")
        print(f"Game Over: {game_state['is_game_over']}")
        
        # Wait before next capture if there are more frames
        if i < num_frames - 1:
            print(f"Waiting {delay} seconds before next capture...")
            time.sleep(delay)
    
    print("\nState detection test complete.")
    print("Check the debug_images directory for visualizations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test The Binding of Isaac game state detection")
    parser.add_argument(
        "--test",
        choices=["state"],
        default="state",
        help="Which test to run",
    )
    parser.add_argument(
        "--frames", type=int, default=1, help="Number of frames to capture for state detection test"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay between frames in seconds"
    )

    args = parser.parse_args()
    
    # Create debug images directory
    os.makedirs("debug_images", exist_ok=True)
    
    # Run state detection test
    test_state_detection(args.frames, args.delay)
