import argparse
import logging
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from environment import IsaacEnvironment
from input_controller import Action
from model import Agent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train an agent to play The Binding of Isaac")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum steps per episode")
    parser.add_argument(
        "--target_update",
        type=int,
        default=10,
        help="Number of episodes between target network updates",
    )
    parser.add_argument(
        "--save_interval", type=int, default=50, help="Number of episodes between model saves"
    )
    parser.add_argument("--load_model", type=str, default=None, help="Path to load a saved model")
    parser.add_argument(
        "--capture_region",
        nargs=4,
        type=int,
        default=None,
        help="Screen capture region (left top width height)",
    )
    parser.add_argument("--render", action="store_true", help="Render training episodes")
    return parser.parse_args()


def train(args):
    """
    Train the agent to play The Binding of Isaac

    Args:
        args: Command line arguments
    """
    # Create output directory
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

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

    # Load model if specified
    if args.load_model and os.path.exists(args.load_model):
        logging.info(f"Loading model from {args.load_model}")
        agent.load(args.load_model)

    # Training stats
    episode_rewards = []
    episode_losses = []
    time_start = time.time()

    # Main training loop
    for episode in range(1, args.episodes + 1):
        # Reset environment and get initial state
        state = env.reset()
        episode_reward = 0
        episode_loss = []

        # Episode loop
        for step in range(1, args.max_steps + 1):
            # Select action
            action = agent.select_action(state)

            # Perform action in environment
            next_state, reward, done, info = env.step(action)

            # Store experience in replay buffer
            agent.memory.push(state, action, reward, next_state, done)

            # Update state and episode reward
            state = next_state
            episode_reward += reward

            # Train the agent
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)

            # Render if enabled
            if args.render:
                # This would be used for visual debugging
                # For example, displaying the game frame with model predictions
                pass

            # Check if episode is done
            if done:
                break

        # Update target network periodically
        if episode % args.target_update == 0:
            agent.update_target_network()
            logging.info(f"Updated target network at episode {episode}")

        # Update exploration rate
        agent.update_epsilon()

        # Save model periodically
        if episode % args.save_interval == 0:
            model_path = f"models/isaac_dqn_ep{episode}.pt"
            agent.save(model_path)
            logging.info(f"Saved model to {model_path}")

        # Log episode stats
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        duration = time.time() - time_start
        episode_rewards.append(episode_reward)
        episode_losses.append(avg_loss)

        logging.info(
            f"Episode {episode} - Reward: {episode_reward:.2f}, Steps: {step}, "
            f"Loss: {avg_loss:.6f}, Epsilon: {agent.epsilon:.4f}, "
            f"Duration: {duration:.2f}s"
        )

        # Plot progress every 10 episodes
        if episode % 10 == 0:
            plot_progress(episode_rewards, episode_losses, episode)

    # Save final model
    final_model_path = f"models/isaac_dqn_final.pt"
    agent.save(final_model_path)
    logging.info(f"Training completed. Final model saved to {final_model_path}")

    # Clean up
    env.close()


def plot_progress(rewards, losses, episode):
    """
    Plot training progress.

    Args:
        rewards (list): Episode rewards
        losses (list): Episode losses
        episode (int): Current episode number
    """
    plt.figure(figsize=(12, 5))

    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title(f"Episode Rewards (Episode {episode})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    # Plot losses
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title(f"Training Loss (Episode {episode})")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"plots/progress_ep{episode}.png")
    plt.close()


def main():
    """Main function"""
    print("=" * 50)
    print("The Binding of Isaac - DQN Training")
    print("=" * 50)

    # Parse arguments
    args = parse_args()

    # Start training
    try:
        train(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
