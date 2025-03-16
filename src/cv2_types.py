"""
Type hints for OpenCV (cv2) to help the IDE's type checker.
This file isn't imported at runtime but helps with development.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Define Mat type for OpenCV
Mat = Union[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float64]]


# Basic OpenCV functions
def resize(
    img: Mat,
    dsize: Tuple[int, int],
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    interpolation: int = 0,
) -> Mat: ...


def cvtColor(src: Mat, code: int, dst: Optional[Mat] = None, dstCn: int = 0) -> Mat: ...


def inRange(
    src: Mat,
    lowerb: Union[Mat, Tuple[int, ...], NDArray[Any]],
    upperb: Union[Mat, Tuple[int, ...], NDArray[Any]],
) -> Mat: ...


def countNonZero(src: Mat) -> int: ...


def morphologyEx(src: Mat, op: int, kernel: Mat) -> Mat: ...


def findContours(
    image: Mat, mode: int, method: int
) -> Tuple[List[NDArray[np.int32]], Optional[Mat]]: ...


def contourArea(contour: NDArray[np.int32]) -> float: ...


def boundingRect(contour: NDArray[np.int32]) -> Tuple[int, int, int, int]: ...


def moments(contour: NDArray[np.int32]) -> Dict[str, float]: ...


def circle(
    img: Mat, center: Tuple[int, int], radius: int, color: Tuple[int, int, int], thickness: int = 1
) -> Mat: ...


def putText(
    img: Mat,
    text: str,
    org: Tuple[int, int],
    fontFace: int,
    fontScale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> Mat: ...


def rectangle(
    img: Mat,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> Mat: ...


def imwrite(filename: str, img: Mat) -> bool: ...


# Common constants
COLOR_RGB2GRAY: int = 6
COLOR_RGB2HSV: int = 40
COLOR_RGB2BGR: int = 4

FONT_HERSHEY_SIMPLEX: int = 0
FONT_HERSHEY_PLAIN: int = 1
FONT_HERSHEY_DUPLEX: int = 2

MORPH_OPEN: int = 2
MORPH_CLOSE: int = 3

RETR_EXTERNAL: int = 0
CHAIN_APPROX_SIMPLE: int = 1
