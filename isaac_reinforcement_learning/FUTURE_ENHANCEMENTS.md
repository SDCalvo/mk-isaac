# Future Enhancements: Screenshot Integration

This document outlines the plan for integrating screenshot-based information into the reinforcement learning agent after the code cleanup.

## Motivation

While the mod-based communication provides detailed information about the game state, adding screenshot-based visual data could benefit the agent in several ways:

1. **Richer observations**: The agent can learn from visual patterns that aren't captured in the text-based state
2. **Redundancy**: If the mod communication fails, the agent can still operate based on visual information
3. **More realistic learning**: Human players rely heavily on visual information

## Implementation Plan

### 1. Screenshot Capture Module

Create a new module `isaac_reinforcement_learning/screen_capture.py` that:

- Captures game screenshots at regular intervals
- Processes screenshots to extract relevant visual features
- Provides an API to access the latest screenshot data

```python
class IsaacScreenCapture:
    def __init__(self, window_title="The Binding of Isaac: Rebirth"):
        self.window_title = window_title
        # Initialize screen capture

    def capture(self):
        """Capture a screenshot of the game window"""
        # Implementation

    def get_game_area(self):
        """Extract just the game area (excluding UI elements)"""
        # Implementation

    def get_downsampled_image(self, size=(84, 84)):
        """Return a downsampled version of the game area for CNN input"""
        # Implementation
```

### 2. Enhanced Gym Environment

Modify the `isaac_gym_env.py` to include visual observations:

```python
class IsaacEnv(gym.Env):
    def __init__(self, render_mode=None, use_screenshots=True):
        super().__init__()

        # Initialize game state reader
        self.client = IsaacGameStateReader()

        # Initialize screen capture if enabled
        self.use_screenshots = use_screenshots
        if use_screenshots:
            self.screen_capture = IsaacScreenCapture()

        # Define observation space that includes both text and visual data
        self.observation_space = spaces.Dict({
            # Existing text-based observations
            'health': spaces.Box(low=0, high=24, shape=(1,), dtype=np.float32),
            'position': spaces.Box(low=0, high=1000, shape=(2,), dtype=np.float32),
            'stage': spaces.Box(low=1, high=13, shape=(1,), dtype=np.int32),
            'room_clear': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            'enemy_count': spaces.Box(low=0, high=30, shape=(1,), dtype=np.int32),

            # New visual observation
            'screen': spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        })
```

### 3. CNN-Based Neural Network

Update the DQN network in `isaac_train.py` to process visual data:

```python
class CNNDQNNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CNNDQNNetwork, self).__init__()

        # CNN for processing visual input
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_visual = nn.Linear(64 * 7 * 7, 512)

        # MLP for processing text-based input
        self.text_input_size = (
            1 +  # health
            2 +  # position (x, y)
            1 +  # stage
            1 +  # room_clear
            1    # enemy_count
        )
        self.fc_text = nn.Linear(self.text_input_size, 128)

        # Combined layers
        self.fc_combined = nn.Linear(512 + 128, 512)
        self.fc_out = nn.Linear(512, action_space.n)

    def forward(self, state_dict):
        # Process visual input
        screen = state_dict['screen'].permute(0, 3, 1, 2) / 255.0
        x_visual = F.relu(self.conv1(screen))
        x_visual = F.relu(self.conv2(x_visual))
        x_visual = F.relu(self.conv3(x_visual))
        x_visual = x_visual.flatten(1)
        x_visual = F.relu(self.fc_visual(x_visual))

        # Process text-based input
        x_text = torch.cat([
            state_dict['health'],
            state_dict['position'],
            state_dict['stage'],
            state_dict['room_clear'],
            state_dict['enemy_count']
        ], dim=1)
        x_text = F.relu(self.fc_text(x_text))

        # Combine features
        x_combined = torch.cat([x_visual, x_text], dim=1)
        x_combined = F.relu(self.fc_combined(x_combined))
        q_values = self.fc_out(x_combined)

        return q_values
```

## Timeline

1. **Phase 1**: Complete code cleanup and ensure the mod-based approach works reliably
2. **Phase 2**: Implement the screen capture module
3. **Phase 3**: Enhance the gym environment to include visual observations
4. **Phase 4**: Develop and train the CNN-based network
5. **Phase 5**: Compare performance of text-only vs. combined approach

## Requirements

- OpenCV (`cv2`) for screenshot processing
- PyTorch for CNN implementation
- Additional GPU memory for training the larger network

## Challenges to Address

- **Performance**: Screen capture and processing can be resource-intensive
- **Synchronization**: Ensuring text data and visual data are in sync
- **Training time**: CNN-based networks take longer to train
- **Window detection**: Reliable method to find and capture the game window

## Success Metrics

- **Improved win rate**: Does the agent perform better with visual data?
- **Generalization**: Can the agent handle situations not seen during training?
- **Robustness**: Can the agent continue to function if some data is missing?
