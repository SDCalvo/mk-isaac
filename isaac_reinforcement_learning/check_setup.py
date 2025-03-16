#!/usr/bin/env python
"""
Check Setup Script for Isaac RL

This script checks if the environment is properly set up for training,
without requiring the game to be running.
"""

import importlib.util
import os
import sys


def print_header(text):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)


def check_package(package_name):
    """Check if a Python package is installed"""
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        print(f"✅ {package_name} is installed")
        return True
    else:
        print(f"❌ {package_name} is NOT installed")
        return False


def check_directory(path, description):
    """Check if a directory exists"""
    # Adjust path based on current directory
    adjusted_path = path
    if os.path.basename(os.getcwd()) == "isaac_reinforcement_learning":
        adjusted_path = os.path.join("..", path)

    if os.path.exists(adjusted_path) and os.path.isdir(adjusted_path):
        print(f"✅ {description} directory found at {adjusted_path}")
        return True
    else:
        print(f"❌ {description} directory NOT found at {adjusted_path}")
        return False


def check_file(path, description):
    """Check if a file exists"""
    # Adjust path based on current directory
    adjusted_path = path
    if os.path.basename(os.getcwd()) == "isaac_reinforcement_learning":
        adjusted_path = os.path.join("..", path)

    if os.path.exists(adjusted_path) and os.path.isfile(adjusted_path):
        print(f"✅ {description} file found at {adjusted_path}")
        return True
    else:
        print(f"❌ {description} file NOT found at {adjusted_path}")
        return False


def main():
    """Main function to check the setup"""
    print_header("ISAAC RL SETUP CHECK")
    print(f"Current directory: {os.getcwd()}")

    # Check Python packages
    print_header("CHECKING REQUIRED PACKAGES")
    packages_ok = all(
        [
            check_package("numpy"),
            check_package("torch"),
            check_package("gymnasium") or check_package("gym"),
        ]
    )

    # Check project structure
    print_header("CHECKING PROJECT STRUCTURE")
    structure_ok = all(
        [
            check_directory("isaac_mod", "Isaac mod"),
            check_directory("isaac_communication", "Communication"),
            check_directory("isaac_reinforcement_learning", "Reinforcement learning"),
            check_directory("models", "Models (for saving checkpoints)"),
            check_directory("logs", "Logs (for training logs)"),
        ]
    )

    # Check key files
    print_header("CHECKING KEY FILES")
    files_ok = all(
        [
            check_file("isaac_mod/main.lua", "Isaac mod main script"),
            check_file("isaac_mod/metadata.xml", "Isaac mod metadata"),
            check_file("isaac_communication/isaac_game_state_reader.py", "Game state reader"),
            check_file("isaac_communication/restart_isaac_with_reader.ps1", "Game restart script"),
            check_file("isaac_reinforcement_learning/isaac_gym_env.py", "Gym environment"),
            check_file("isaac_reinforcement_learning/isaac_train.py", "Training script"),
        ]
    )

    # Check mod installation
    print_header("CHECKING MOD INSTALLATION")
    isaac_path = "E:/Steam/steamapps/common/The Binding of Isaac Rebirth/mods/IsaacGameStateReader"
    mod_installed = check_directory(isaac_path, "Isaac mod installation")

    if not mod_installed:
        print("\n⚠️ The mod doesn't appear to be installed in the default location.")
        print("You may need to run the installation script before training:")
        print("    cd isaac_communication")
        print("    .\\install_mod.ps1")

    # Overall status
    print_header("SETUP STATUS")
    all_ok = packages_ok and structure_ok and files_ok and mod_installed

    if all_ok:
        print("✅ Your environment is ready for training!")
        print("\nTo start training, you need to:")
        print("1. Launch the game with the mod:")
        print("    cd isaac_communication")
        print("    .\\restart_isaac_with_reader.ps1")
        print("\n2. Start the training script:")
        print("    cd isaac_reinforcement_learning")
        print("    pipenv run python isaac_train.py")
    else:
        print("❌ There are issues with your setup that need to be resolved.")
        print("\nPlease fix the issues mentioned above before starting training.")

    return all_ok


if __name__ == "__main__":
    main()
