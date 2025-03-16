# The Binding of Isaac: Game State Reader

A tool that enables reinforcement learning and AI experimentation with The Binding of Isaac: Rebirth.

## Project Structure

The project is organized into three main components:

```
isaac_mod/                 # The Binding of Isaac Lua mod files
  ├── main.lua             # Main mod file (Lua)
  └── metadata.xml         # Mod metadata

isaac_communication/       # Communication tools for interfacing with the game
  ├── isaac_game_state_reader.py   # Main Python API for reading game state
  ├── install_mod.ps1              # Script to install the mod to Isaac
  ├── restart_isaac_with_reader.ps1 # Script to restart Isaac and the reader
  └── pipe_server.py               # Low-level pipe server utility

isaac_reinforcement_learning/  # AI and reinforcement learning components
  ├── isaac_agent.py           # Simple Q-learning agent example
  ├── isaac_gym_env.py         # OpenAI Gym environment wrapper
  └── isaac_train.py           # Training script for RL algorithms
```

## Overview

This project provides a communication bridge between The Binding of Isaac: Rebirth and external Python applications, allowing you to:

1. Read detailed game state information in real-time
2. Send commands to control the game
3. Train AI agents to play the game
4. Experiment with reinforcement learning algorithms

## Quick Start

### 1. Install the mod

Run the installation script:

```powershell
cd isaac_communication
.\install_mod.ps1
```

This will copy the mod files to your Isaac mods directory.

### 2. Launch the game with the provided script

```powershell
cd isaac_communication
.\restart_isaac_with_reader.ps1
```

This script will:

- Stop any running Isaac processes
- Clean up temporary files
- Start Isaac with the `--luadebug` option
- Launch the game state reader script

### 3. Try the reinforcement learning agent

```powershell
cd isaac_reinforcement_learning
python isaac_agent.py
```

This will start a simple Q-learning agent that will learn to play the game.

## Features

### Game State Reader

The game state reader provides access to:

- **Player Information:** Health, position, velocity, stats, items
- **Enemy Information:** Type, health, position, boss status
- **Room Information:** Type, size, doors, completion status
- **Floor Information:** Name, stage, curses

### Control Options

You can control the game with these commands:

- Movement (up, down, left, right)
- Shooting (up, down, left, right)
- Using active items
- Placing bombs

## Requirements

- The Binding of Isaac: Rebirth
- Python 3.7+
- Required Python packages: `numpy`

## Troubleshooting

If you encounter issues:

1. Make sure the mod is installed correctly
2. Check that you're launching the game with the `--luadebug` option
3. Look for error messages in the console output
4. Try restarting both the game and the reader script

## License

MIT License - Feel free to use, modify, and distribute this code.
