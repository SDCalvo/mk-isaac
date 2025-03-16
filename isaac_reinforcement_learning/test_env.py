#!/usr/bin/env python
"""
Test script for the Isaac Gym Environment

This script tests if the environment can connect to the game.
"""

import os
import sys
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the IsaacGameStateReader directly to test connection
from isaac_communication.isaac_game_state_reader import IsaacGameStateReader

def main():
    """Test basic connection to the game"""
    print("Initializing Isaac Game State Reader...")
    
    # Create the reader
    reader = IsaacGameStateReader()
    
    # Check if we can get a response
    print("\nSending status command...")
    reader.send_command("status")
    time.sleep(1)
    
    response = reader.read_response()
    if response:
        print(f"\nSuccessfully received response from game:")
        print(response[:200] + "..." if len(response) > 200 else response)
        
        # Parse the response
        print("\nParsing response...")
        try:
            game_state = reader._parse_status_text(response)
            print("\nParsed game state:")
            print(f"- Time: {game_state.get('time')}")
            print(f"- Frame: {game_state.get('frame')}")
            
            # Player info
            player = game_state.get('player', {})
            health = player.get('health', {})
            position = player.get('position', {})
            print(f"- Health: {health.get('current', 0)}/{health.get('max', 0)}")
            print(f"- Position: ({position.get('x', 0)}, {position.get('y', 0)})")
            
            # Floor info
            floor = game_state.get('floor', {})
            print(f"- Floor: {floor.get('name', 'Unknown')}")
            
            # Room info
            room = game_state.get('room', {})
            print(f"- Room cleared: {room.get('is_clear', False)}")
            print(f"- Enemy count: {len(game_state.get('enemies', []))}")
            
            print("\nConnection test SUCCESSFUL!")
            return True
        except Exception as e:
            print(f"\nError parsing response: {e}")
            print("Raw response:")
            print(response)
    else:
        print("\nNo response received from the game.")
        print("Make sure Isaac is running with the --luadebug option and the mod is installed correctly.")
    
    print("\nConnection test FAILED!")
    return False

if __name__ == "__main__":
    main() 