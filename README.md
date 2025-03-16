# Binding of Isaac AI Player

An AI-powered bot that learns to play The Binding of Isaac using deep reinforcement learning.

## Project Structure

- `src/`: Source code directory
  - `game_capture.py`: Screen capture and image processing
  - `input_controller.py`: Game input simulation
  - `model.py`: Deep learning model architecture
  - `environment.py`: Game environment wrapper
  - `train.py`: Training script
  - `play.py`: Script to play the game with a trained model

## Requirements

- Python 3.12
- The Binding of Isaac game installed
- Pipenv for dependency management

## Setup

1. Install Pipenv if you don't have it already:

```bash
pip install pipenv
```

2. Install dependencies and the project in development mode:

```bash
pipenv install
pipenv install -e .
```

3. Activate virtual environment:

```bash
pipenv shell
```

## Usage

### Training the AI

Train the AI model with:

```bash
python -m src.train --episodes 100 --capture_region 0 0 1920 1080
```

Parameters:

- `--episodes`: Number of episodes to train
- `--max_steps`: Maximum steps per episode
- `--target_update`: Episodes between target network updates
- `--save_interval`: Episodes between model saves
- `--load_model`: Path to load existing model
- `--capture_region`: Screen region to capture (left top width height)
- `--render`: Enable rendering for debugging

### Playing with a trained model

Use a trained model to play:

```bash
python -m src.play --model models/isaac_dqn_final.pt --visualize
```

Parameters:

- `--model`: Path to the trained model
- `--max_steps`: Maximum steps to play
- `--delay`: Delay between actions
- `--capture_region`: Screen region to capture
- `--visualize`: Show what the agent sees
- `--record`: Record gameplay to a video file

## How it Works

The AI uses computer vision to understand the game state and deep reinforcement learning to make decisions about which actions to take. It learns by playing the game repeatedly and improving its strategy based on the outcomes.

1. **Game Capture**: The system captures frames from the game screen
2. **Image Processing**: Processes these frames to extract relevant information
3. **Deep Q-Learning**: Uses a convolutional neural network to learn game strategies
4. **Action Execution**: Controls the game using simulated keyboard inputs

## Future Improvements

- Implement advanced reward functions based on game events
- Add object detection for enemies, items, and hazards
- Implement curiosity-driven exploration
- Support for different character types and game modes
