import re
import sys

def update_roi_in_file(file_path, roi_name, x, y, width, height, use_percentages=True):
    """
    Update ROI coordinates in the game_capture.py file
    
    Args:
        file_path (str): Path to the game_capture.py file
        roi_name (str): Name of the ROI to update (health, minimap, resources, or game_area)
        x (float): X coordinate (percentage or absolute)
        y (float): Y coordinate (percentage or absolute)
        width (float): Width (percentage or absolute)
        height (float): Height (percentage or absolute)
        use_percentages (bool): Whether to use percentages or absolute values
    """
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Function name pattern based on ROI name
    function_name = f"get_{roi_name.lower()}_roi"
    
    # Find the function in the file
    pattern = rf"def {function_name}\(self, frame\):\s+\"\"\".*?\"\"\"\s+height, width = frame\.shape\[:2\]\s+(.*?)\s+return \(x, y, w, h\)"
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print(f"Could not find {function_name} function in {file_path}")
        return False
    
    # Current code inside the function
    current_code = match.group(1)
    
    # Create new code with updated coordinates
    if use_percentages:
        new_code = f"""        # {roi_name} ROI with updated coordinates
        x = int(width * {x/100:.4f})
        y = int(height * {y/100:.4f})
        w = int(width * {width/100:.4f})
        h = int(height * {height/100:.4f})"""
    else:
        new_code = f"""        # {roi_name} ROI with updated coordinates
        x = {x}
        y = {y}
        w = {width}
        h = {height}"""
    
    # Replace the old code with the new code
    updated_content = content.replace(match.group(0), match.group(0).replace(current_code, new_code))
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(updated_content)
    
    print(f"Updated {roi_name} ROI in {file_path}")
    return True

def main():
    """Main function to update ROIs from command line arguments"""
    if len(sys.argv) < 7:
        print("Usage: python update_roi.py <roi_name> <x> <y> <width> <height> <use_percentages>")
        print("Example: python update_roi.py health 5 5 15 8 1")
        return
    
    roi_name = sys.argv[1].lower()
    x = float(sys.argv[2])
    y = float(sys.argv[3])
    width = float(sys.argv[4])
    height = float(sys.argv[5])
    use_percentages = bool(int(sys.argv[6]))
    
    file_path = "src/game_capture.py"
    
    if roi_name not in ["health", "minimap", "resources", "game_area"]:
        print(f"Invalid ROI name: {roi_name}. Must be: health, minimap, resources, or game_area")
        return
    
    update_roi_in_file(file_path, roi_name, x, y, width, height, use_percentages)

if __name__ == "__main__":
    main() 