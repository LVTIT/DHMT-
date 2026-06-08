"""
capture.py - Video / Webcam / Image capture module.

Provides VideoCapture for webcam and video files, and ImageCapture
for static image files (.jpg, .png, etc.).
Both support context manager and iterator protocol.
"""

import os
import cv2
from typing import Union, Optional, Tuple
import numpy as np

# Supported image extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_image(source) -> bool:
    """Check if source path points to an image file."""
    if isinstance(source, int):
        return False
    _, ext = os.path.splitext(str(source))
    return ext.lower() in _IMAGE_EXTENSIONS


class ImageCapture:
    """
    Static image reader with the same interface as VideoCapture.

    Reads a single image file and yields it once via the iterator protocol.
    Useful for skeleton detection on still images.

    Parameters
    ----------
    source : str
        Path to an image file (.jpg, .png, etc.).
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
        """Return the image (only once)."""
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
        Camera index (e.g. 0) hoặc đường dẫn file video.
    width : int, optional
        Chiều rộng frame mong muốn (chỉ áp dụng cho webcam).
    height : int, optional
        Chiều cao frame mong muốn (chỉ áp dụng cho webcam).
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

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def open(self) -> "VideoCapture":
        """Mở nguồn video và cấu hình resolution (nếu cần)."""
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise IOError(f"Không thể mở nguồn video: {self._source}")

        # Chỉ set resolution khi nguồn là webcam (int)
        if isinstance(self._source, int):
            if self._width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            if self._height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        return self

    def close(self) -> None:
        """Giải phóng tài nguyên capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoCapture":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_opened(self) -> bool:
        """Trả về True nếu nguồn video đang mở."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        """FPS của nguồn video (0 nếu chưa mở)."""
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        """(width, height) của frame hiện tại."""
        if self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    @property
    def frame_count(self) -> int:
        """Tổng số frame (0 nếu là webcam)."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Đọc một frame.

        Returns
        -------
        success : bool
            True nếu đọc thành công.
        frame : np.ndarray | None
            BGR frame hoặc None nếu hết / lỗi.
        """
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------
    def __iter__(self):
        """Cho phép dùng `for frame in capture: ...`"""
        if not self.is_opened:
            self.open()
        return self

    def __next__(self) -> np.ndarray:
        ret, frame = self.read()
        if not ret:
            raise StopIteration
        return frame
