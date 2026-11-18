"""Contract between model-specific code and the shared detection pipeline."""
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray


@dataclass(frozen=True)
class TrainingContext:
    output: Path
    train: Sequence[Mapping]
    val: Sequence[Mapping]
    classes: tuple[str, ...]
    calibration_images: tuple[Path, ...]
    size: int
    epochs: int
    batch: int
    seed: int
    device: str


@dataclass(frozen=True)
class Artifact:
    path: Path
    predictor: Callable[["ndarray"], "ndarray"]
    format: str
    preprocessing: str
    postprocessing: str


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: Path
    initialization: str
    artifacts: Mapping[str, Artifact]


def prepare_image(path, size):
    """Load square RGB input and return the original image's bounds within it."""
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    ratio = min(size / width, size / height)
    new_width, new_height = max(1, round(width * ratio)), max(1, round(height * ratio))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    # Pad the image to a square so it keeps its original proportions.
    frame = np.full((size, size, 3), 114, dtype=np.uint8)
    frame[top:top + new_height, left:left + new_width] = cv2.resize(
        rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    return frame, (left, top, new_width, new_height)


def ground_truth_boxes(labels, bounds):
    """Convert YOLO center coordinates to corners in the prepared RGB image."""
    left, top, width, height = bounds
    return [
        [left + max(0, x - w / 2) * width, top + max(0, y - h / 2) * height,
         left + min(1, x + w / 2) * width, top + min(1, y + h / 2) * height]
        for _, x, y, w, h in labels
    ]
