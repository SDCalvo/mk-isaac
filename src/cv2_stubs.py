"""
This file provides stub definitions for OpenCV functions and constants
to help the linter recognize them. This file is not meant to be imported
or used directly in the code, but rather to provide type information.
"""
import numpy as np
from typing import Any, Tuple, List, Union, Optional

# Common OpenCV constants
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 4
COLOR_RGB2GRAY = 7
COLOR_BGR2GRAY = 6
COLOR_RGB2HSV = 40
COLOR_BGR2HSV = 41

# Constants for morphological operations
MORPH_OPEN = 2
MORPH_CLOSE = 3
MORPH_DILATE = 1
MORPH_ERODE = 0

# Contour retrieval modes
RETR_EXTERNAL = 0
RETR_LIST = 1
RETR_CCOMP = 2
RETR_TREE = 3
RETR_FLOODFILL = 4

# Contour approximation methods
CHAIN_APPROX_NONE = 1
CHAIN_APPROX_SIMPLE = 2
CHAIN_APPROX_TC89_L1 = 3
CHAIN_APPROX_TC89_KCOS = 4

# Font options
FONT_HERSHEY_SIMPLEX = 0
FONT_HERSHEY_PLAIN = 1
FONT_HERSHEY_DUPLEX = 2
FONT_HERSHEY_COMPLEX = 3
FONT_HERSHEY_TRIPLEX = 4
FONT_HERSHEY_COMPLEX_SMALL = 5
FONT_HERSHEY_SCRIPT_SIMPLEX = 6
FONT_HERSHEY_SCRIPT_COMPLEX = 7

# Common OpenCV functions with type hints

def resize(img: np.ndarray, dsize: Tuple[int, int], fx: Optional[float] = None, fy: Optional[float] = None, interpolation: int = 0) -> np.ndarray:
    """Resize an image"""
    pass

def cvtColor(img: np.ndarray, code: int) -> np.ndarray:
    """Convert an image from one color space to another"""
    pass

def inRange(src: np.ndarray, lowerb: np.ndarray, upperb: np.ndarray) -> np.ndarray:
    """Check if array elements lie between the elements of two other arrays"""
    pass

def morphologyEx(src: np.ndarray, op: int, kernel: np.ndarray) -> np.ndarray:
    """Perform advanced morphological transformations"""
    pass

def findContours(image: np.ndarray, mode: int, method: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Find contours in a binary image"""
    pass

def contourArea(contour: np.ndarray) -> float:
    """Calculate the area of a contour"""
    pass

def boundingRect(points: np.ndarray) -> Tuple[int, int, int, int]:
    """Calculate the bounding rectangle of a point set"""
    pass

def moments(array: np.ndarray) -> dict:
    """Calculate moments of a contour"""
    pass

def circle(img: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int], thickness: int = 1) -> np.ndarray:
    """Draw a circle"""
    pass

def putText(img: np.ndarray, text: str, org: Tuple[int, int], fontFace: int, fontScale: float, color: Tuple[int, int, int], thickness: int = 1) -> np.ndarray:
    """Draw text on an image"""
    pass

def imwrite(filename: str, img: np.ndarray) -> bool:
    """Save an image to a file"""
    pass

def countNonZero(src: np.ndarray) -> int:
    """Count non-zero array elements"""
    pass 