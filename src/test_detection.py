"""
Test script for The Binding of Isaac game state detection.
This script will capture frames from the game and display the detected game state.
"""

import os
import time

import cv2
import numpy as np

try:
    from src.game_capture import GameCapture
except ImportError:
    from game_capture import GameCapture


def test_calibration():
    """Test the calibration by capturing frames and visualizing ROIs"""
    os.makedirs("debug", exist_ok=True)

    capture = GameCapture()

    print("Starting calibration test...")
    print("Please have The Binding of Isaac game window visible.")
    print("Capturing in 3 seconds...")
    time.sleep(3)

    # Capture frame
    frame = capture.capture_game_window()

    # Calibrate with this frame
    capture.calibrate(frame)

    # Draw ROIs on frame
    debug_frame = frame.copy()
    roi_colors = {
        "health_bar": (255, 0, 0),  # Red
        "minimap": (0, 255, 0),  # Green
        "item_display": (0, 0, 255),  # Blue
    }

    for roi_name, color in roi_colors.items():
        if roi_name in capture.roi and capture.roi[roi_name]:
            roi = capture.roi[roi_name]
            cv2.rectangle(
                debug_frame,
                (roi["left"], roi["top"]),
                (roi["left"] + roi["width"], roi["top"] + roi["height"]),
                color,
                2,
            )
            cv2.putText(
                debug_frame,
                roi_name,
                (roi["left"], roi["top"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    # Save the debug frame
    cv2.imwrite("debug/calibration.jpg", cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
    print("Saved calibration visualization to debug/calibration.jpg")


def test_state_detection(num_frames=5, delay=1.0):
    """
    Test state detection by capturing multiple frames and saving debug visualizations.

    Args:
        num_frames (int): Number of frames to capture
        delay (float): Delay between frames in seconds
    """
    os.makedirs("debug", exist_ok=True)

    capture = GameCapture()

    print(f"Starting state detection test for {num_frames} frames...")
    print("Please have The Binding of Isaac game window visible.")
    print("Capturing will begin in 3 seconds...")
    time.sleep(3)

    for i in range(num_frames):
        print(f"Capturing frame {i+1}/{num_frames}...")

        # Capture frame
        frame = capture.capture_game_window()

        # Detect game state
        game_state = capture.detect_game_state(frame)

        # Save debug frame
        filename = f"debug/frame_{i+1}.jpg"
        capture.save_debug_frame(frame, game_state, filename)

        print(f"Detected game state for frame {i+1}:")
        print(f"  Player Health: {game_state['player_health']}")
        print(f"  Player Position: {game_state['player_position']}")
        print(f"  Number of Enemies: {len(game_state['enemies'])}")
        print(f"  Doors: {[d for d, exists in game_state['doors'].items() if exists]}")
        print(f"  Number of Items: {len(game_state['items'])}")
        print(f"Saved debug visualization to {filename}")

        # Delay before next capture
        if i < num_frames - 1:
            print(f"Waiting {delay} seconds before next capture...")
            time.sleep(delay)

    print("\nState detection test complete.")
    print(f"Saved {num_frames} debug frames to the debug directory.")


def test_component_detection():
    """Test each component of the detection system separately"""
    os.makedirs("debug", exist_ok=True)

    capture = GameCapture()

    print("Starting component detection test...")
    print("Please have The Binding of Isaac game window visible.")
    print("Capturing in 3 seconds...")
    time.sleep(3)

    # Capture frame
    frame = capture.capture_game_window()

    # Test each detection component
    components = {
        "health": capture.detect_health(frame),
        "player": capture.detect_player(frame),
        "enemies": capture.detect_enemies(frame),
        "doors": capture.detect_doors(frame),
        "items": capture.detect_items(frame),
    }

    # Create debug visualizations for each component
    debug_frames = {}

    # Health detection
    health_frame = frame.copy()
    cv2.putText(
        health_frame,
        f"Health: {components['health']}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    debug_frames["health"] = health_frame

    # Player detection
    player_frame = frame.copy()
    if components["player"]:
        x, y = components["player"]
        cv2.circle(player_frame, (x, y), 15, (0, 255, 0), 2)
        cv2.putText(
            player_frame, "Player", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
    debug_frames["player"] = player_frame

    # Enemy detection
    enemy_frame = frame.copy()
    for i, enemy in enumerate(components["enemies"]):
        x, y = enemy["position"]
        cv2.circle(enemy_frame, (x, y), 10, (0, 0, 255), 2)
        cv2.putText(
            enemy_frame, f"Enemy {i}", (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
    debug_frames["enemies"] = enemy_frame

    # Door detection
    door_frame = frame.copy()
    height, width = frame.shape[:2]
    door_positions = {
        "north": (width // 2, height // 10),
        "east": (width * 9 // 10, height // 2),
        "south": (width // 2, height * 9 // 10),
        "west": (width // 10, height // 2),
    }
    for direction, exists in components["doors"].items():
        color = (0, 255, 0) if exists else (0, 0, 255)  # Green if door exists, red if not
        cv2.putText(
            door_frame,
            f"{direction}: {'yes' if exists else 'no'}",
            door_positions[direction],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    debug_frames["doors"] = door_frame

    # Item detection
    item_frame = frame.copy()
    for i, item_pos in enumerate(components["items"]):
        x, y = item_pos
        cv2.circle(item_frame, (x, y), 5, (255, 255, 0), 2)
        cv2.putText(
            item_frame, f"Item {i}", (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
        )
    debug_frames["items"] = item_frame

    # Save all debug frames
    for component, debug_frame in debug_frames.items():
        filename = f"debug/{component}_detection.jpg"
        cv2.imwrite(filename, cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
        print(f"Saved {component} detection visualization to {filename}")

    print("\nComponent detection test complete.")
    print("Check the debug directory for visualization images.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test The Binding of Isaac game state detection")
    parser.add_argument(
        "--test",
        choices=["calibration", "state", "components", "all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "--frames", type=int, default=5, help="Number of frames to capture for state detection test"
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between frames in seconds")

    args = parser.parse_args()

    if args.test == "calibration" or args.test == "all":
        test_calibration()
        if args.test != "all":
            exit()

    if args.test == "components" or args.test == "all":
        test_component_detection()
        if args.test != "all":
            exit()

    if args.test == "state" or args.test == "all":
        test_state_detection(args.frames, args.delay)
