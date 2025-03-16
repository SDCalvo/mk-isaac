import tkinter as tk
from tkinter import messagebox
import win32gui
import win32con
import time
import sys
import json
import os
from ctypes import windll
import keyboard
import threading

def find_isaac_window():
    """Find The Binding of Isaac window"""
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "isaac" in title.lower():
                windows.append(hwnd)
        return True

    windows = []
    win32gui.EnumWindows(callback, windows)
    if windows:
        return windows[0]
    return None

def get_screen_size():
    """Get the screen dimensions"""
    return windll.user32.GetSystemMetrics(0), windll.user32.GetSystemMetrics(1)

class CoordinateFinder:
    def __init__(self, root):
        self.root = root
        self.root.title("Screen Coordinate Finder")
        self.root.geometry("600x450")
        self.root.attributes("-topmost", True)
        
        # JSON file path
        self.json_file = "roi_coordinates.json"
        
        # ROI data dict for JSON export
        self.roi_data = {}
        
        # Load existing data if file exists
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r') as f:
                    self.roi_data = json.load(f)
                print(f"Loaded existing ROI data from {self.json_file}")
            except Exception as e:
                print(f"Error loading ROI data: {e}")
                self.roi_data = {}
        
        # Screen dimensions
        self.screen_width, self.screen_height = get_screen_size()
        
        # Find Isaac window
        self.isaac_hwnd = find_isaac_window()
        if self.isaac_hwnd:
            self.left, self.top, self.right, self.bottom = win32gui.GetWindowRect(self.isaac_hwnd)
            self.game_width = self.right - self.left
            self.game_height = self.bottom - self.top
        else:
            self.left, self.top = 0, 0
            self.game_width = self.screen_width
            self.game_height = self.screen_height
        
        # ROI tracking
        self.roi_points = {
            'health': [],
            'minimap': [],
            'resources': [],
            'game_area': []
        }
        
        # Create widgets
        self.create_widgets()
        
        # Start tracking mouse position
        self.root.after(50, self.update_position)
        
        # Set up keyboard hooks
        self.setup_keyboard_hooks()
    
    def setup_keyboard_hooks(self):
        """Set up keyboard shortcuts for capturing coordinates"""
        # Create a separate thread for keyboard listening
        self.keyboard_thread = threading.Thread(target=self.listen_keyboard, daemon=True)
        self.keyboard_thread.start()
    
    def listen_keyboard(self):
        """Listen for keyboard shortcuts"""
        # Map keys to ROI names
        key_map = {
            '1': 'health',
            '2': 'minimap',
            '3': 'resources',
            '4': 'game_area',
        }
        
        # Save key
        save_key = '5'
        
        while True:
            # Check if save key is pressed
            if keyboard.is_pressed(save_key):
                self.save_to_json()
                time.sleep(0.5)  # Prevent multiple saves
            
            # Check ROI capture keys
            for key, roi in key_map.items():
                if keyboard.is_pressed(key):
                    # Get current mouse position
                    x, y = win32gui.GetCursorPos()
                    # Calculate relative position to game window
                    rel_x = x - self.left
                    rel_y = y - self.top
                    # Save the position
                    self.save_position(roi, rel_x, rel_y)
                    # Wait to avoid multiple detections
                    time.sleep(0.3)
            
            # Small delay to reduce CPU usage
            time.sleep(0.05)
    
    def save_position(self, roi_name, x, y):
        """Save a position for an ROI"""
        if len(self.roi_points[roi_name]) < 2:
            self.roi_points[roi_name].append((x, y))
            
            # Calculate percentage
            pct_x = round(x / self.game_width * 100, 2)
            pct_y = round(y / self.game_height * 100, 2)
            
            # Add to results
            point_num = len(self.roi_points[roi_name])
            if point_num == 1:
                self.results_text.insert(tk.END, f"{roi_name} ROI - Top-Left: ({x}, {y}) - ({pct_x}%, {pct_y}%)\n")
                self.status_label.config(text=f"Captured top-left of {roi_name}. Press {roi_name}'s key again to capture bottom-right.")
            else:
                # We have both points, calculate width and height
                x1, y1 = self.roi_points[roi_name][0]
                width = x - x1
                height = y - y1
                pct_width = round(width / self.game_width * 100, 2)
                pct_height = round(height / self.game_height * 100, 2)
                
                self.results_text.insert(tk.END, f"{roi_name} ROI - Bottom-Right: ({x}, {y}) - ({pct_x}%, {pct_y}%)\n")
                self.results_text.insert(tk.END, f"{roi_name} ROI - Width: {width}, Height: {height} - ({pct_width}%, {pct_height}%)\n")
                self.results_text.insert(tk.END, f"Command: python -m src.update_roi {roi_name} {pct_x-pct_width} {pct_y-pct_height} {pct_width} {pct_height} 1\n\n")
                
                # Save to ROI data
                top_left_x = pct_x - pct_width
                top_left_y = pct_y - pct_height
                
                self.roi_data[roi_name] = {
                    "x": top_left_x,
                    "y": top_left_y,
                    "width": pct_width,
                    "height": pct_height,
                    "x_raw": x1,
                    "y_raw": y1,
                    "width_raw": width,
                    "height_raw": height
                }
                
                self.status_label.config(text=f"Captured both points for {roi_name}. Press 5 to save all ROIs to JSON.")
                
            self.results_text.see(tk.END)
    
    def save_to_json(self):
        """Save ROI data to JSON file"""
        try:
            # Add game window info
            self.roi_data["game_window"] = {
                "left": self.left,
                "top": self.top,
                "right": self.right,
                "bottom": self.bottom,
                "width": self.game_width,
                "height": self.game_height
            }
            
            # Save to file
            with open(self.json_file, 'w') as f:
                json.dump(self.roi_data, f, indent=4)
            
            self.results_text.insert(tk.END, f"Saved ROI data to {self.json_file}\n\n")
            self.results_text.see(tk.END)
            self.status_label.config(text=f"ROI data saved to {self.json_file}")
            
            # Show popup
            tk.messagebox.showinfo("Success", f"ROI data saved to {self.json_file}")
            
        except Exception as e:
            self.results_text.insert(tk.END, f"Error saving to JSON: {e}\n\n")
            self.results_text.see(tk.END)
            self.status_label.config(text=f"Error saving to JSON: {e}")
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Isaac ROI Coordinate Finder", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Instructions
        instructions = (
            "Instructions:\n"
            "1. Position your mouse over a region in the game window\n"
            "2. Press the number key for that region to capture position\n"
            "3. Press the same key again at the bottom-right corner\n"
            "4. Press 5 to save all ROIs to JSON file\n\n"
            "Keyboard shortcuts:\n"
            "1 - Health ROI\n"
            "2 - Minimap ROI\n"
            "3 - Resources ROI\n"
            "4 - Game Area ROI\n"
            "5 - Save to JSON"
        )
        instructions_label = tk.Label(self.root, text=instructions, justify=tk.LEFT)
        instructions_label.pack(pady=5)
        
        # Mouse position frame
        pos_frame = tk.Frame(self.root)
        pos_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Mouse position label
        tk.Label(pos_frame, text="Mouse Position:").grid(row=0, column=0, padx=5)
        self.pos_label = tk.Label(pos_frame, text="(0, 0)", width=15)
        self.pos_label.grid(row=0, column=1, padx=5)
        
        # Relative position
        tk.Label(pos_frame, text="Game Position:").grid(row=0, column=2, padx=5)
        self.rel_pos_label = tk.Label(pos_frame, text="(0, 0)", width=15)
        self.rel_pos_label.grid(row=0, column=3, padx=5)
        
        # Percentage
        tk.Label(pos_frame, text="Percentage:").grid(row=0, column=4, padx=5)
        self.pct_label = tk.Label(pos_frame, text="(0%, 0%)", width=15)
        self.pct_label.grid(row=0, column=5, padx=5)
        
        # Save button
        save_button = tk.Button(self.root, text="Save to JSON", command=self.save_to_json)
        save_button.pack(pady=5)
        
        # Status
        self.status_label = tk.Label(self.root, text="Move your mouse to the desired position and press the corresponding key")
        self.status_label.pack(pady=5)
        
        # Game window info
        if self.isaac_hwnd:
            title = win32gui.GetWindowText(self.isaac_hwnd)
            window_info = f"Game Window: {title} | Position: ({self.left}, {self.top}) | Size: {self.game_width}x{self.game_height}"
        else:
            window_info = "No Isaac window found! Using full screen dimensions."
        
        window_label = tk.Label(self.root, text=window_info)
        window_label.pack(pady=5)
        
        # Results
        results_frame = tk.Frame(self.root)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        results_label = tk.Label(results_frame, text="Results:")
        results_label.pack(anchor=tk.W)
        
        self.results_text = tk.Text(results_frame, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
        
        self.results_text.insert(tk.END, "Captured coordinates will appear here...\n\n")
    
    def update_position(self):
        try:
            # Get current mouse position
            x, y = win32gui.GetCursorPos()
            
            # Update position label
            self.pos_label.config(text=f"({x}, {y})")
            
            # Calculate relative position to game window
            rel_x = x - self.left
            rel_y = y - self.top
            self.rel_pos_label.config(text=f"({rel_x}, {rel_y})")
            
            # Calculate percentage
            pct_x = round(rel_x / self.game_width * 100, 2)
            pct_y = round(rel_y / self.game_height * 100, 2)
            self.pct_label.config(text=f"({pct_x}%, {pct_y}%)")
            
            # Update 20 times per second
            self.root.after(50, self.update_position)
        except Exception as e:
            print(f"Error updating position: {e}")
            self.root.after(50, self.update_position)

def main():
    """Main function for the coordinate finder"""
    root = tk.Tk()
    app = CoordinateFinder(root)
    root.mainloop()

if __name__ == "__main__":
    main() 