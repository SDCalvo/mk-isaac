# The Binding of Isaac: Reinforcement Learning

This directory contains the reinforcement learning components for The Binding of Isaac.

## Project Structure

- `isaac_agent.py`: A simple agent that can play the game without using the Gym environment
- `isaac_gym_env.py`: A Gym-compatible environment wrapper for the game
- `isaac_train.py`: Training script for DQN-based reinforcement learning
- `test_env.py`: Test script to verify communication with the game

## Prerequisites

Before you can start training a model, you need to:

1. Have The Binding of Isaac: Rebirth installed
2. Install the IsaacGameStateReader mod (located in the `isaac_mod` directory)
3. Launch the game with the `--luadebug` option
4. Have the required Python packages installed:
   ```
   pipenv install gymnasium torch
   ```

## Getting Started

1. **Launch the game with the mod**:

   ```
   cd isaac_communication
   .\restart_isaac_with_reader.ps1
   ```

   This script will:

   - Stop any running Isaac processes
   - Install the mod if needed
   - Start Isaac with the `--luadebug` option

2. **Test the connection**:

   ```
   cd isaac_reinforcement_learning
   pipenv run python test_env.py
   ```

   This will check if the connection to the game is working.

3. **Run the simple agent**:

   ```
   cd isaac_reinforcement_learning
   pipenv run python isaac_agent.py
   ```

   This will start a simple agent that plays the game.

4. **Train a model**:

   ```
   cd isaac_reinforcement_learning
   pipenv run python isaac_train.py
   ```

   This will start training a DQN model to play the game.

## Project Organization

This project contains two different approaches to interacting with The Binding of Isaac:

1. **File-based communication** (current approach):

   - Located in `isaac_communication`, `isaac_mod`, and `isaac_reinforcement_learning`
   - Uses a Lua mod to communicate directly with the game
   - Requires the `--luadebug` option to enable file I/O
   - More reliable and provides more detailed game state information

2. **Screen capture-based approach** (legacy):
   - Located in the `src` directory
   - Uses screen capture to detect game state
   - Does not require modifying the game
   - Less reliable and provides limited information

## Important Notes

- Always launch the game before running any scripts that interact with it
- The DQN training can take a long time to converge
- You may need to adjust the reward function to get better results
- Model checkpoints are saved to the `models` directory
- Training logs are saved to the `logs` directory
