"""
Test script for simplified game state detection.
This script captures frames from the Binding of Isaac and detects basic game state.
"""

import cv2
import time
import os
from game_capture import GameCapture

def main():
    """Run a test to verify our simplified game state detection"""
    print("Starting simplified game state test...")
    print("Please make sure The Binding of Isaac is running and visible.")
    
    # Create output directory for debug images
    os.makedirs("debug_images", exist_ok=True)
    
    # Create game capture
    game_capture = GameCapture()
    
    # Track state changes
    last_health = None
    last_room_hash = None
    rooms_explored = 0
    
    try:
        print("\nRunning test loop - press Ctrl+C to stop")
        print("Move around in the game to test room detection")
        
        # Run a test loop capturing state every second
        while True:
            # Capture game window
            frame = game_capture.capture_game_window()
            
            if frame is None:
                print("Game window not found. Make sure the game is running.")
                time.sleep(2)
                continue
            
            # Get game state
            game_state = game_capture.get_game_state(frame)
            
            # Handle state changes
            if game_state["is_new_room"]:
                if game_state["is_unexplored_room"]:
                    rooms_explored += 1
                    print(f"\n🆕 New room discovered! (Total: {rooms_explored})")
                else:
                    print("\n↪️ Returned to previously explored room")
                    
            # Check health changes
            if last_health is not None and game_state["health"] != last_health:
                if game_state["health"] < last_health:
                    print(f"\n❤️ Health decreased: {last_health} -> {game_state['health']}")
                else:
                    print(f"\n💖 Health increased: {last_health} -> {game_state['health']}")
            
            # Print current state
            print(f"\rHealth: {game_state['health']} | " +
                  f"Floor: {game_state['current_floor']} | " +
                  f"Rooms Explored: {rooms_explored} | " +
                  f"Game Over: {game_state['is_game_over']}", end="")
            
            # Update tracking
            last_health = game_state["health"]
            
            # Wait a bit before next capture
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\nTest stopped by user.")
    finally:
        print("\nTest completed. Check debug_images folder for visualization.")

if __name__ == "__main__":
    main() 