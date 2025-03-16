import os
import time
import torch
import numpy as np
import cv2
from environment import IsaacEnvironment
from model import Agent
from input_controller import Action
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("play.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Play The Binding of Isaac using a trained agent")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum steps to play")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between actions (seconds)")
    parser.add_argument("--capture_region", nargs=4, type=int, default=None, help="Screen capture region (left top width height)")
    parser.add_argument("--visualize", action="store_true", help="Visualize agent's perspective")
    parser.add_argument("--record", action="store_true", help="Record gameplay to video")
    return parser.parse_args()

def visualize_state(state, action, q_values=None, reward=None):
    """
    Visualize the agent's current state, action, and Q-values.
    
    Args:
        state (numpy.ndarray): Current state (stacked frames)
        action (int): Action index
        q_values (numpy.ndarray): Q-values for all actions
        reward (float): Last reward received
    
    Returns:
        numpy.ndarray: Visualization image
    """
    # Get the latest frame from the state (last of stacked frames)
    frame = state[-1]  # Shape: (84, 84)
    
    # Convert to RGB for visualization
    rgb_frame = np.stack([frame, frame, frame], axis=2)
    rgb_frame = (rgb_frame * 255).astype(np.uint8)
    
    # Resize for better visualization
    vis_frame = cv2.resize(rgb_frame, (336, 336))
    
    # Map action index to action name
    action_names = [a.name for a in Action]
    action_name = action_names[action] if 0 <= action < len(action_names) else "UNKNOWN"
    
    # Add text for action
    cv2.putText(
        vis_frame, f"Action: {action_name}", (10, 20), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
    )
    
    # Add reward information if available
    if reward is not None:
        cv2.putText(
            vis_frame, f"Reward: {reward:.2f}", (10, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
    
    # Add Q-values if available
    if q_values is not None:
        for i, q in enumerate(q_values):
            if i < len(action_names):
                y_pos = 60 + i * 20
                cv2.putText(
                    vis_frame, f"{action_names[i]}: {q:.2f}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
                )
    
    return vis_frame

def play(args):
    """
    Play the game using a trained agent.
    
    Args:
        args: Command line arguments
    """
    # Create environment
    capture_region = None
    if args.capture_region:
        capture_region = tuple(args.capture_region)
    env = IsaacEnvironment(capture_region=capture_region)
    
    # Get the number of actions
    n_actions = len(list(Action))
    
    # Create agent
    state_shape = (4, 84, 84)  # 4 stacked frames of 84x84 pixels
    agent = Agent(state_shape, n_actions)
    
    # Load model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    logging.info(f"Loading model from {args.model}")
    agent.load(args.model)
    
    # Set up video recording if enabled
    video_writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_path = f"gameplay_{timestamp}.avi"
        video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (336, 336))
        logging.info(f"Recording video to {video_path}")
    
    # Reset environment and get initial state
    state = env.reset()
    
    # Play loop
    total_reward = 0
    step = 0
    last_reward = 0
    
    logging.info("Starting gameplay...")
    
    try:
        for step in range(1, args.max_steps + 1):
            # Get Q-values for visualization
            q_values = None
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor).cpu().numpy()[0]
            
            # Select action (no exploration)
            action = agent.select_action(state, training=False)
            
            # Visualize before taking action
            if args.visualize or args.record:
                vis_frame = visualize_state(state, action, q_values, last_reward)
                
                if args.visualize:
                    cv2.imshow("Agent's View", vis_frame)
                    cv2.waitKey(1)
                
                if args.record and video_writer:
                    video_writer.write(vis_frame)
            
            # Perform action in environment
            next_state, reward, done, info = env.step(action)
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            last_reward = reward
            
            # Add delay between actions for better visualization
            time.sleep(args.delay)
            
            # Log every 100 steps
            if step % 100 == 0:
                logging.info(f"Step {step}, Total Reward: {total_reward:.2f}")
            
            # Check if episode is done
            if done:
                logging.info(f"Game over at step {step}")
                break
    
    except KeyboardInterrupt:
        logging.info("Gameplay interrupted by user")
    except Exception as e:
        logging.error(f"Error during gameplay: {str(e)}")
        raise
    finally:
        # Clean up
        if args.visualize:
            cv2.destroyAllWindows()
        
        if args.record and video_writer:
            video_writer.release()
        
        env.close()
    
    logging.info(f"Gameplay ended after {step} steps with total reward: {total_reward:.2f}")

def main():
    """Main function"""
    print("=" * 50)
    print("The Binding of Isaac - AI Player")
    print("=" * 50)
    
    # Parse arguments
    args = parse_args()
    
    # Start playing
    play(args)

if __name__ == "__main__":
    main() 