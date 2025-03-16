"""
A safer test script for game capture functionality that works even when the game isn't running.
This script will either use sample images if available or create a simple mock game frame.
"""

import logging
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Import the GameCapture class
from game_capture import GameCapture

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestDetection")

# Create debug directory if it doesn't exist
debug_dir = Path("debug")
debug_dir.mkdir(exist_ok=True)


def create_mock_game_frame():
    """
    Create a mock game frame that simulates The Binding of Isaac.
    This is used when no sample images are available.
    """
    # Create a blank frame (black background)
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    # Draw a room background (gray floor)
    cv2.rectangle(frame, (50, 50), (750, 550), (100, 100, 100), -1)

    # Draw player character (light skin tone color)
    cv2.circle(frame, (400, 300), 15, (200, 150, 150), -1)

    # Draw some enemies (darker colors)
    cv2.circle(frame, (200, 200), 10, (50, 50, 100), -1)
    cv2.circle(frame, (600, 400), 12, (30, 30, 80), -1)
    cv2.circle(frame, (300, 450), 8, (40, 40, 90), -1)

    # Draw item (bright yellow)
    cv2.circle(frame, (500, 250), 5, (0, 255, 255), -1)

    # Draw health hearts (red) in the top-right corner
    for i in range(3):
        cv2.circle(frame, (700 + i * 20, 30), 8, (0, 0, 200), -1)

    # Draw doors
    # North door
    cv2.rectangle(frame, (375, 50), (425, 70), (150, 150, 150), -1)
    # East door
    cv2.rectangle(frame, (730, 275), (750, 325), (150, 150, 150), -1)
    # South door
    cv2.rectangle(frame, (375, 530), (425, 550), (150, 150, 150), -1)
    # West door
    cv2.rectangle(frame, (50, 275), (70, 325), (150, 150, 150), -1)

    # Convert to RGB (from BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


def test_detection_with_mock():
    """Test detection with a mock game frame."""
    logger.info("Creating a mock game frame for testing...")

    # Create mock frame
    mock_frame = create_mock_game_frame()

    # Initialize GameCapture
    capture = GameCapture()

    # Detect game state
    game_state = capture.detect_game_state(mock_frame)

    # Save debug visualization
    mock_debug_path = debug_dir / "mock_debug.jpg"
    capture.save_debug_frame(mock_frame, game_state, str(mock_debug_path))

    # Print detected state
    logger.info("\nDetected Game State (Mock Frame):")
    logger.info(f"Player Health: {game_state['player_health']}")
    logger.info(f"Player Position: {game_state['player_position']}")
    logger.info(f"Number of Enemies: {len(game_state['enemies'])}")
    logger.info(f"Doors: {[d for d, exists in game_state['doors'].items() if exists]}")
    logger.info(f"Number of Items: {len(game_state['items'])}")

    logger.info(f"Saved debug frame to {mock_debug_path}")

    # Show the mock frame with matplotlib if available
    try:
        plt.figure(figsize=(10, 8))
        plt.imshow(mock_frame)
        plt.title("Mock Game Frame")
        plt.savefig(str(debug_dir / "mock_frame.jpg"))
        logger.info(f"Saved mock frame to {debug_dir / 'mock_frame.jpg'}")
    except Exception as e:
        logger.warning(f"Could not save visualization: {e}")


def test_calibration():
    """Test calibration with a mock frame."""
    logger.info("Testing calibration...")

    # Create mock frame
    mock_frame = create_mock_game_frame()

    # Initialize GameCapture
    capture = GameCapture()

    # Calibrate with mock frame
    capture.calibrate(mock_frame)

    # Draw calibration visualization
    calib_vis = mock_frame.copy()

    # Draw ROI rectangles for visualization
    for roi_name, roi in capture.roi.items():
        if roi:
            # Draw rectangle for each ROI
            color = (
                (0, 255, 0)
                if roi_name == "health_bar"
                else (255, 0, 0) if roi_name == "minimap" else (0, 0, 255)
            )
            cv2.rectangle(
                calib_vis,
                (roi["left"], roi["top"]),
                (roi["left"] + roi["width"], roi["top"] + roi["height"]),
                color,
                2,
            )
            # Add label
            cv2.putText(
                calib_vis,
                roi_name,
                (roi["left"], roi["top"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    # Save calibration visualization
    calib_path = debug_dir / "calibration.jpg"
    cv2.imwrite(str(calib_path), cv2.cvtColor(calib_vis, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved calibration visualization to {calib_path}")


def test_with_real_game_if_available():
    """Attempt to test with the real game if it's available."""
    logger.info("Checking if real game is available...")

    # Initialize GameCapture
    capture = GameCapture()

    try:
        # Try to capture the screen
        logger.info("Attempting to capture screen...")
        logger.info("If you have The Binding of Isaac open, this will use the actual game.")
        logger.info("Otherwise, this part of the test will be skipped.")
        time.sleep(2)  # Short delay to let user read the message

        # Capture frame
        frame = capture.capture_game_window()

        # Check if the frame looks like a game (this is a very simple heuristic)
        # For a real game, we'd expect a colorful image with variation
        frame_std = np.std(frame)

        if frame_std < 50:  # If image has low variation, probably not a game
            logger.warning("Captured frame doesn't look like a game (low color variation).")
            logger.warning("Skipping real game detection test.")
            return False

        # Detect game state
        game_state = capture.detect_game_state(frame)

        # Save debug frame
        real_debug_path = debug_dir / "real_game_debug.jpg"
        capture.save_debug_frame(frame, game_state, str(real_debug_path))

        # Print detected state
        logger.info("\nDetected Game State (Real Game):")
        logger.info(f"Player Health: {game_state['player_health']}")
        logger.info(f"Player Position: {game_state['player_position']}")
        logger.info(f"Number of Enemies: {len(game_state['enemies'])}")
        logger.info(f"Doors: {[d for d, exists in game_state['doors'].items() if exists]}")
        logger.info(f"Number of Items: {len(game_state['items'])}")

        logger.info(f"Saved debug frame to {real_debug_path}")
        return True

    except Exception as e:
        logger.error(f"Error during real game detection: {e}")
        return False


def main():
    """Run all the tests."""
    logger.info("Starting the detection test suite...")

    # Test calibration
    test_calibration()

    # Test detection with mock frame
    test_detection_with_mock()

    # Try real game if available
    if test_with_real_game_if_available():
        logger.info("Real game detection test completed successfully.")
    else:
        logger.info("Real game detection test was skipped or failed.")
        logger.info("You can still review the mock frame detection results.")

    logger.info("\nDetection tests completed.")
    logger.info(f"Check the '{debug_dir}' directory for debug visualizations.")


if __name__ == "__main__":
    main()
