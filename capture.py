"""
capture.py - Video / Webcam / Image capture module.

Provides VideoCapture for webcam and video files, and ImageCapture for static
image files. Both classes support context-manager and iterator protocols.
"""

import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_image(source) -> bool:
    """Return True when source is a supported image file path."""
    if isinstance(source, int):
        return False
    _, ext = os.path.splitext(str(source))
    return ext.lower() in _IMAGE_EXTENSIONS


class ImageCapture:
    """
    Static image reader with the same basic interface as VideoCapture.

    Parameters
    ----------
    source : str
        Path to an image file.
    """

    def __init__(self, source: str):
        self._source = source
        self._frame: Optional[np.ndarray] = None
        self._consumed = False

    def open(self) -> "ImageCapture":
        """Load the image from disk."""
        self._frame = cv2.imread(self._source)
        if self._frame is None:
            raise IOError(f"Cannot read image: {self._source}")
        self._consumed = False
        return self

    def close(self) -> None:
        """Release image data."""
        self._frame = None

    def __enter__(self) -> "ImageCapture":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def is_opened(self) -> bool:
        return self._frame is not None

    @property
    def fps(self) -> float:
        return 0.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        if self._frame is None:
            return (0, 0)
        h, w = self._frame.shape[:2]
        return (w, h)

    @property
    def frame_count(self) -> int:
        return 1 if self._frame is not None else 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the loaded image once."""
        if self._frame is not None and not self._consumed:
            self._consumed = True
            return True, self._frame.copy()
        return False, None

    def __iter__(self):
        if not self.is_opened:
            self.open()
        self._consumed = False
        return self

    def __next__(self) -> np.ndarray:
        ret, frame = self.read()
        if not ret:
            raise StopIteration
        return frame


class VideoCapture:
    """
    Wrapper around cv2.VideoCapture with convenience helpers.

    Parameters
    ----------
    source : int | str
        Camera index, for example 0, or video file path.
    width : int, optional
        Requested webcam frame width.
    height : int, optional
        Requested webcam frame height.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        self._source = source
        self._cap: Optional[cv2.VideoCapture] = None
        self._width = width
        self._height = height

    def open(self) -> "VideoCapture":
        """Open the video source and configure webcam resolution if requested."""
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {self._source}")

        if isinstance(self._source, int):
            if self._width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            if self._height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        return self

    def close(self) -> None:
        """Release the capture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoCapture":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    @property
    def frame_count(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read one BGR frame."""
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    def __iter__(self):
        if not self.is_opened:
            self.open()
        return self

    def __next__(self) -> np.ndarray:
        ret, frame = self.read()
        if not ret:
            raise StopIteration
        return frame
