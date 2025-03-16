#!/usr/bin/env python
"""
Cleanup script for the Isaac RL project

This script:
1. Removes the src folder (old screen capture approach)
2. Ensures proper imports in remaining files
3. Creates necessary directories for training

Run this script from the project root:
    pipenv run python cleanup.py
"""

import os
import re
import shutil
import sys


def print_header(text):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)


def remove_directory(path):
    """Remove a directory safely"""
    if os.path.exists(path):
        print(f"Removing directory: {path}")
        try:
            shutil.rmtree(path)
            print(f"✅ Successfully removed {path}")
            return True
        except Exception as e:
            print(f"❌ Failed to remove {path}: {e}")
            return False
    else:
        print(f"⚠️ Directory does not exist: {path}")
        return True


def create_directory(path):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(path):
        print(f"Creating directory: {path}")
        try:
            os.makedirs(path)
            print(f"✅ Successfully created {path}")
            return True
        except Exception as e:
            print(f"❌ Failed to create {path}: {e}")
            return False
    else:
        print(f"✓ Directory already exists: {path}")
        return True


def fix_imports_in_file(file_path):
    """Fix imports in a Python file"""
    print(f"Fixing imports in: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix imports: src.* -> isaac_*
    content = re.sub(r"from src\.", "from isaac_", content)
    content = re.sub(r"import src\.", "import isaac_", content)

    # Remove any path manipulation for src
    content = re.sub(r"sys\.path\.append\(.*src.*\)", "", content)

    # Write changes back to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Fixed imports in {file_path}")
    return True


def main():
    """Main cleanup function"""
    print_header("ISAAC RL PROJECT CLEANUP")

    # 1. Remove src folder (old screen capture approach)
    print_header("REMOVING OLD CODE")
    remove_directory("src")

    # 2. Create necessary directories for training
    print_header("CREATING NECESSARY DIRECTORIES")
    create_directory("models")
    create_directory("logs")

    # 3. Fix imports in remaining files
    print_header("FIXING IMPORTS")
    python_files = []

    # Find all Python files in the project
    for root, _, files in os.walk("."):
        if ".git" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    # Fix imports in each file
    for file in python_files:
        fix_imports_in_file(file)

    print_header("CLEANUP COMPLETE")
    print("The project has been cleaned up successfully.")
    print("Old screen capture approach has been removed.")
    print("Imports have been fixed.")
    print("You can now continue with the reinforcement learning work.")


if __name__ == "__main__":
    main()
