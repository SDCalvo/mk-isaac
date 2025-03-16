"""
Isaac Game State Reader

This module connects to The Binding of Isaac game via file-based IPC to 
read game state information and send commands.

Usage:
    python isaac_game_state_reader.py

Requirements:
    - The Binding of Isaac: Rebirth with the IsaacGameStateReader mod installed
    - The game must be launched with the --luadebug option
"""

import os
import sys
import time
import json
import tempfile
from datetime import datetime

class IsaacGameStateReader:
    """Class for interacting with Isaac game state via files"""
    
    def __init__(self):
        # Temp directory for communication files
        self.temp_dir = tempfile.gettempdir()
        self.input_pipe = os.path.join(self.temp_dir, "isaac_input_pipe.txt")
        self.output_pipe = os.path.join(self.temp_dir, "isaac_output_pipe.txt")
        
        # Mod directory
        self.mod_dir = "E:/Steam/steamapps/common/The Binding of Isaac Rebirth/mods/IsaacGameStateReader"
        self.output_dir = os.path.join(self.mod_dir, "output")
        self.pipe_info_path = os.path.join(self.output_dir, "pipe_info.txt")
        
        # Game state
        self.last_game_state = {}
        self.connected = False
        
        # Initialize communication
        self._initialize()
    
    def _initialize(self):
        """Initialize communication with the game"""
        print(f"Initializing Isaac Game State Reader at {datetime.now()}")
        
        # Create pipes (using regular files for Windows compatibility)
        with open(self.input_pipe, 'w') as f:
            f.write('')
        print(f"Created input file: {self.input_pipe}")
        
        with open(self.output_pipe, 'w') as f:
            f.write('')
        print(f"Created output file: {self.output_pipe}")
        
        # Write the pipe locations to a file that the mod can read
        try:
            with open(self.pipe_info_path, 'w') as f:
                f.write(f"INPUT_PIPE={self.input_pipe}\nOUTPUT_PIPE={self.output_pipe}")
            print(f"Wrote pipe info to: {self.pipe_info_path}")
        except Exception as e:
            print(f"Error writing pipe info: {e}")
        
        # Wait for the mod to connect
        print("Waiting for mod to connect...")
        for i in range(60):  # Try for 60 seconds
            print(f"Connection attempt {i+1}/60...")
            if self._check_connection():
                self.connected = True
                print("Connected to the mod!")
                return
            time.sleep(1)
        
        print("Failed to connect to the mod. Make sure the game is running with the --luadebug option.")
        print("\nTROUBLESHOOTING:")
        print("1. Check that the game is running with the --luadebug option")
        print("2. Check that the mod is enabled in the game")
        print("3. Check the debug log for errors:")
        debug_log = os.path.join(self.output_dir, "debug_log.txt")
        if os.path.exists(debug_log):
            print("\nDebug log contents:")
            try:
                with open(debug_log, 'r') as f:
                    print(f.read())
            except Exception as e:
                print(f"Error reading debug log: {e}")
        else:
            print("Debug log not found")
    
    def _check_connection(self):
        """Check if the mod is connected by sending a status command"""
        try:
            print("Sending status command...")
            self.send_command("status")
            print("Waiting for response...")
            time.sleep(1)  # Give the mod more time to respond
            
            if os.path.exists(self.output_pipe):
                size = os.path.getsize(self.output_pipe)
                print(f"Output pipe exists, size: {size} bytes")
                if size > 0:
                    # Try to read the content
                    try:
                        with open(self.output_pipe, 'r') as f:
                            content = f.read()
                        print(f"Received response: {content[:100]}...")
                        return True
                    except Exception as e:
                        print(f"Error reading response: {e}")
                else:
                    print("Output pipe is empty")
            else:
                print("Output pipe does not exist")
        except Exception as e:
            print(f"Error in check_connection: {e}")
        return False
    
    def send_command(self, command):
        """Send a command to the game"""
        if not self.connected and command != "status":
            print("Not connected to the game")
            return False
        
        try:
            with open(self.input_pipe, 'w') as f:
                f.write(command)
            print(f"Sent command: {command}")
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def read_response(self):
        """Read response from the game"""
        if not self.connected:
            return None
        
        try:
            if os.path.exists(self.output_pipe) and os.path.getsize(self.output_pipe) > 0:
                with open(self.output_pipe, 'r') as f:
                    response = f.read()
                
                # Clear the output pipe
                with open(self.output_pipe, 'w') as f:
                    f.write('')
                
                return response
            return None
        except Exception as e:
            print(f"Error reading response: {e}")
            return None
    
    def get_game_state(self):
        """Get the current game state"""
        self.send_command("status")
        time.sleep(0.1)  # Give the mod time to respond
        
        response = self.read_response()
        if response:
            # Parse the response into a structured game state
            game_state = self._parse_status_response(response)
            self.last_game_state = game_state
            return game_state
        
        return self.last_game_state
    
    def _parse_status_response(self, response):
        """Parse the status response into a structured game state"""
        try:
            # Check if the response starts with "Error:"
            if response.strip().startswith("Error:"):
                print(f"Error from mod: {response.strip()}")
                return {
                    'error': response.strip(),
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'frame': None,
                    'player': {
                        'health': {'current': 0, 'max': 0},
                        'position': {'x': 0, 'y': 0}
                    },
                    'enemies': [],
                    'room': {'type': 0, 'type_name': 'ERROR', 'is_clear': False}
                }
            
            # Debug the raw response
            print(f"Raw response first 100 chars: {response[:100]}...")
            
            # First try to parse as "Status Report:" text format
            if response.strip().startswith("Status Report:"):
                return self._parse_status_text(response)
            
            # If that fails, try to parse as JSON
            try:
                parsed_state = json.loads(response)
                
                # Debug the enemy and room data
                enemies = parsed_state.get('enemies', [])
                print(f"Found {len(enemies)} enemies in the JSON data")
                
                if len(enemies) > 0:
                    print("First enemy data sample:")
                    print(f"  Type: {enemies[0].get('type', 'Unknown')}")
                    print(f"  Name: {enemies[0].get('type_name', 'Unknown')}")
                    print(f"  HP: {enemies[0].get('hp', 0)}/{enemies[0].get('max_hp', 0)}")
                
                # Debug room information
                room = parsed_state.get('room', {})
                print(f"Room type: {room.get('type', 'Unknown')}, Name: {room.get('type_name', 'Unknown')}")
                
                return parsed_state
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                # Fall back to the text-based parsing
                return self._parse_status_text(response)
        except Exception as e:
            print(f"Error parsing status response: {e}")
            print(f"Raw response: {response[:200]}...")
            return {"error": str(e), "raw_response": response}
    
    def _parse_status_text(self, response):
        """Parse the text-based status format"""
        game_state = {
            'time': None,
            'frame': None,
            'player': {
                'health': {
                    'current': 0,
                    'max': 0
                },
                'position': {
                    'x': 0,
                    'y': 0
                }
            },
            'floor': {
                'name': None,
                'stage': None
            },
            'enemies': [],
            'room': {
                'type': 0,
                'type_name': 'UNKNOWN',
                'is_clear': False
            },
            'raw_response': response
        }
        
        try:
            lines = response.strip().split('\n')
            
            for line in lines:
                if line.startswith('Time:'):
                    game_state['time'] = line.replace('Time:', '').strip()
                elif line.startswith('Frame:'):
                    try:
                        game_state['frame'] = int(line.replace('Frame:', '').strip())
                    except:
                        game_state['frame'] = 0
                elif line.startswith('Player Health:'):
                    health_parts = line.replace('Player Health:', '').strip().split('/')
                    if len(health_parts) == 2:
                        try:
                            game_state['player']['health']['current'] = int(health_parts[0])
                            game_state['player']['health']['max'] = int(health_parts[1])
                        except:
                            pass
                elif line.startswith('Player Position:'):
                    pos_parts = line.replace('Player Position:', '').strip().split(',')
                    if len(pos_parts) == 2:
                        try:
                            game_state['player']['position']['x'] = float(pos_parts[0])
                            game_state['player']['position']['y'] = float(pos_parts[1])
                        except:
                            pass
                elif line.startswith('Floor:'):
                    floor_info = line.replace('Floor:', '').strip()
                    if '(Stage' in floor_info:
                        name, stage = floor_info.split('(Stage')
                        game_state['floor']['name'] = name.strip()
                        try:
                            game_state['floor']['stage'] = int(stage.replace(')', '').strip())
                        except:
                            game_state['floor']['stage'] = 1
                    else:
                        game_state['floor']['name'] = floor_info
                elif line.startswith('Room Type:'):
                    try:
                        game_state['room']['type'] = int(line.replace('Room Type:', '').strip())
                    except:
                        pass
                elif line.startswith('Room Clear:'):
                    game_state['room']['is_clear'] = line.replace('Room Clear:', '').strip().lower() == 'true'
                elif line.startswith('Enemy Count:'):
                    try:
                        enemy_count = int(line.replace('Enemy Count:', '').strip())
                        # We'll parse individual enemies below
                    except:
                        pass
                elif line.startswith('Enemy '):
                    # Parse enemy information (format: "Enemy N: Type=X, HP=Y, Pos=(Z,W)")
                    try:
                        # Extract the type, hp, and position
                        type_part = line.split('Type=')[1].split(',')[0].strip()
                        hp_part = line.split('HP=')[1].split(',')[0].strip()
                        pos_part = line.split('Pos=(')[1].split(')')[0].strip()
                        pos_x, pos_y = pos_part.split(',')
                        
                        enemy = {
                            'type': int(type_part),
                            'hp': float(hp_part),
                            'position': {
                                'x': float(pos_x),
                                'y': float(pos_y)
                            }
                        }
                        game_state['enemies'].append(enemy)
                    except:
                        pass
        except Exception as e:
            print(f"Error parsing status text response: {e}")
        
        return game_state
    
    def move_player(self, direction):
        """Move the player in the specified direction"""
        if direction not in ['up', 'down', 'left', 'right']:
            print(f"Invalid direction: {direction}")
            return False
        
        return self.send_command(f"move_{direction}")
    
    def shoot(self, direction):
        """Shoot tears in the specified direction"""
        if direction not in ['up', 'down', 'left', 'right']:
            print(f"Invalid direction: {direction}")
            return False
        
        return self.send_command(f"shoot_{direction}")
    
    def use_item(self):
        """Use the player's active item"""
        return self.send_command("use_item")
    
    def place_bomb(self):
        """Place a bomb"""
        return self.send_command("bomb")
    
    def is_connected(self):
        """Check if we're connected to the game"""
        return self.connected
    
    def get_enemies(self):
        """Get a list of enemies in the current room"""
        game_state = self.get_game_state()
        return game_state.get('enemies', [])
    
    def get_nearest_enemy(self):
        """Get the enemy nearest to the player"""
        game_state = self.get_game_state()
        enemies = game_state.get('enemies', [])
        
        if not enemies:
            return None
            
        player_pos = game_state.get('player', {}).get('position', {})
        if not player_pos.get('x') or not player_pos.get('y'):
            return None
        
        nearest = None
        min_dist = float('inf')
        
        for enemy in enemies:
            enemy_pos = enemy.get('position', {})
            if not enemy_pos.get('x') or not enemy_pos.get('y'):
                continue
                
            dist = ((enemy_pos.get('x') - player_pos.get('x'))**2 + 
                    (enemy_pos.get('y') - player_pos.get('y'))**2)**0.5
                    
            if dist < min_dist:
                min_dist = dist
                nearest = enemy
        
        return nearest
    
    def get_pickup_info(self):
        """Get a list of pickups in the current room"""
        game_state = self.get_game_state()
        return game_state.get('pickups', [])
    
    def get_nearest_pickup(self):
        """Get the pickup nearest to the player"""
        game_state = self.get_game_state()
        pickups = game_state.get('pickups', [])
        
        if not pickups:
            return None
            
        player_pos = game_state.get('player', {}).get('position', {})
        if not player_pos.get('x') or not player_pos.get('y'):
            return None
        
        nearest = None
        min_dist = float('inf')
        
        for pickup in pickups:
            pickup_pos = pickup.get('position', {})
            if not pickup_pos.get('x') or not pickup_pos.get('y'):
                continue
                
            dist = ((pickup_pos.get('x') - player_pos.get('x'))**2 + 
                    (pickup_pos.get('y') - player_pos.get('y'))**2)**0.5
                    
            if dist < min_dist:
                min_dist = dist
                nearest = pickup
        
        return nearest
    
    def get_door_info(self):
        """Get a list of doors in the current room"""
        game_state = self.get_game_state()
        return game_state.get('room', {}).get('doors', [])
    
    def get_player_stats(self):
        """Get detailed player stats"""
        game_state = self.get_game_state()
        return game_state.get('player', {}).get('stats', {})
    
    def get_room_info(self):
        """Get detailed room information"""
        game_state = self.get_game_state()
        return game_state.get('room', {})
    
    def get_full_health_info(self):
        """Get detailed player health information"""
        game_state = self.get_game_state()
        return game_state.get('player', {}).get('health', {})

def main():
    """Main function to test the Isaac Game State Reader"""
    reader = IsaacGameStateReader()
    
    if not reader.is_connected():
        print("Failed to connect to the game. Exiting.")
        return
    
    try:
        print("\nStarting fixed game state monitoring. Press Ctrl+C to exit.")
        while True:
            # Get and display game state
            game_state = reader.get_game_state()
            if game_state:
                # Display a comprehensive version of the game state
                print("\n--- Game State Summary ---")
                
                # Check for errors
                if 'error' in game_state:
                    print(f"Error in game state: {game_state.get('error')}")
                    time.sleep(1)
                    continue
                
                # Basic info
                print(f"Frame: {game_state.get('frame')}")
                
                # Player info
                player = game_state.get('player', {})
                health = player.get('health', {})
                position = player.get('position', {})
                
                print("\n--- Player Info ---")
                current_health = health.get('current', 0)
                max_health = health.get('max', 0)
                print(f"Hearts: {current_health}/{max_health}")
                
                x_pos = position.get('x', 0)
                y_pos = position.get('y', 0)
                print(f"Position: ({x_pos:.1f}, {y_pos:.1f})")
                
                # Room info
                room = game_state.get('room', {})
                print("\n--- Room Info ---")
                print(f"Type: {room.get('type_name', 'Unknown')} (ID: {room.get('type', 0)})")
                print(f"Cleared: {room.get('is_clear', False)}")
                
                # Enemy info
                enemies = game_state.get('enemies', [])
                print(f"\n--- Enemies ({len(enemies)}) ---")
                
                if not enemies:
                    print("No enemies detected")
                
                for i, enemy in enumerate(enemies[:3]):  # Show first 3 enemies
                    enemy_pos = enemy.get('position', {})
                    enemy_type = enemy.get('type', 0)
                    enemy_hp = enemy.get('hp', 0)
                    print(f"{i+1}. Type: {enemy_type} - " + 
                          f"HP: {enemy_hp:.1f} - " + 
                          f"Pos: ({enemy_pos.get('x', 0):.1f}, {enemy_pos.get('y', 0):.1f})")
                
                if len(enemies) > 3:
                    print(f"... and {len(enemies) - 3} more enemies")
                
                # Floor info
                floor = game_state.get('floor', {})
                print("\n--- Floor Info ---")
                print(f"Name: {floor.get('name', 'Unknown')} (Stage {floor.get('stage', 0)})")
            
            # Wait before checking again
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 