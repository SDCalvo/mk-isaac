import json
import os
import sys
import tempfile
import time
from datetime import datetime

# Create a temporary directory for our pipe
PIPE_DIR = tempfile.gettempdir()
INPUT_PIPE = os.path.join(PIPE_DIR, "isaac_input_pipe.txt")
OUTPUT_PIPE = os.path.join(PIPE_DIR, "isaac_output_pipe.txt")


def main():
    print(f"Starting Isaac pipe server at {datetime.now()}")
    print(f"Input pipe: {INPUT_PIPE}")
    print(f"Output pipe: {OUTPUT_PIPE}")

    # Create pipes (using regular files for Windows compatibility)
    with open(INPUT_PIPE, "w") as f:
        f.write("")
    print(f"Created input file: {INPUT_PIPE}")

    with open(OUTPUT_PIPE, "w") as f:
        f.write("")
    print(f"Created output file: {OUTPUT_PIPE}")

    # Write the pipe locations to a file that the mod can read
    pipe_info_path = "E:/Steam/steamapps/common/The Binding of Isaac Rebirth/mods/IsaacGameStateReader/output/pipe_info.txt"
    try:
        with open(pipe_info_path, "w") as f:
            f.write(f"INPUT_PIPE={INPUT_PIPE}\nOUTPUT_PIPE={OUTPUT_PIPE}")
        print(f"Wrote pipe info to: {pipe_info_path}")
    except Exception as e:
        print(f"Error writing pipe info: {e}")

    # Main loop
    try:
        while True:
            # Send a command to Isaac
            command = "status"
            try:
                with open(INPUT_PIPE, "w") as f:
                    f.write(command)
                print(f"Sent command: {command}")
            except Exception as e:
                print(f"Error writing to input pipe: {e}")

            # Wait for response
            print("Waiting for response...")
            try:
                time.sleep(1)  # Give Isaac time to respond
                if os.path.exists(OUTPUT_PIPE) and os.path.getsize(OUTPUT_PIPE) > 0:
                    with open(OUTPUT_PIPE, "r") as f:
                        response = f.read()
                    if response:
                        print(f"Received response: {response}")
                        # Clear the response file
                        with open(OUTPUT_PIPE, "w") as f:
                            f.write("")
                    else:
                        print("No response received.")
                else:
                    print("Output file is empty or doesn't exist.")
            except Exception as e:
                print(f"Error reading from output pipe: {e}")

            # Wait before sending the next command
            time.sleep(5)
    except KeyboardInterrupt:
        print("Server stopped by user.")
    except Exception as e:
        print(f"Error in main loop: {e}")


if __name__ == "__main__":
    main()
