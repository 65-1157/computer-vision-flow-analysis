"""
Image-preprocessing functions for the computer-vision pipeline.

These functions prepare raw frames for segmentation and tracking.
They are intentionally simple and reusable across notebooks.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


ROI = Tuple[int, int, int, int]


def read_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """
    Read an image from disk.

    Parameters
    ----------
    path : str or Path
        Image file path.
    grayscale : bool
        Whether to load the image as grayscale.

    Returns
    -------
    np.ndarray
        Loaded image.
    """
    path = Path(path)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR

    image = cv2.imread(str(path), flag)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.

    Parameters
    ----------
    image : np.ndarray
        Input image. Can be grayscale or BGR.

    Returns
    -------
    np.ndarray
        Grayscale image.
    """
    if image.ndim == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def crop_roi(image: np.ndarray, roi: Optional[ROI] = None) -> np.ndarray:
    """
    Crop an image using a region of interest.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    roi : tuple or None
        Region of interest in the format (x, y, width, height).
        If None, the original image is returned.

    Returns
    -------
    np.ndarray
        Cropped image.
    """
    if roi is None:
        return image

    x, y, width, height = roi

    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive.")

    return image[y:y + height, x:x + width]


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0
) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    kernel_size : int
        Gaussian kernel size. Must be odd.
    sigma : float
        Gaussian sigma.

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median blur to reduce salt-and-pepper noise.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    kernel_size : int
        Median kernel size. Must be odd.

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    return cv2.medianBlur(image, kernel_size)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Improve contrast using histogram equalization.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image.

    Returns
    -------
    np.ndarray
        Contrast-enhanced image.
    """
    gray = convert_to_grayscale(image)
    return cv2.equalizeHist(gray)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Improve local contrast using CLAHE.

    CLAHE means Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale or BGR image.
    clip_limit : float
        Contrast limiting threshold.
    tile_grid_size : tuple
        Tile grid size.

    Returns
    -------
    np.ndarray
        Contrast-enhanced grayscale image.
    """
    gray = convert_to_grayscale(image)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(gray)


def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 31,
    c_value: int = 5,
    invert: bool = False
) -> np.ndarray:
    """
    Apply adaptive thresholding.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale or BGR image.
    block_size : int
        Size of neighborhood area. Must be odd and greater than 1.
    c_value : int
        Constant subtracted from the local mean.
    invert : bool
        Whether to invert the binary output.

    Returns
    -------
    np.ndarray
        Binary image.
    """
    gray = convert_to_grayscale(image)

    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError("block_size must be odd and greater than 1.")

    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=threshold_type,
        blockSize=block_size,
        C=c_value
    )


def otsu_threshold(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Apply Otsu global thresholding.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale or BGR image.
    invert : bool
        Whether to invert the binary output.

    Returns
    -------
    np.ndarray
        Binary image.
    """
    gray = convert_to_grayscale(image)

    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        threshold_type + cv2.THRESH_OTSU
    )

    return binary


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image to the uint8 range [0, 255].

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Normalized uint8 image.
    """
    normalized = cv2.normalize(
        image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    return normalized.astype(np.uint8)


def preprocess_frame(
    frame: np.ndarray,
    roi: Optional[ROI] = None,
    grayscale: bool = True,
    denoise: bool = True,
    enhance_contrast: bool = True,
    blur_kernel_size: int = 5
) -> np.ndarray:
    """
    Apply a standard preprocessing sequence to one frame.

    Parameters
    ----------
    frame : np.ndarray
        Input image frame.
    roi : tuple or None
        Optional region of interest in the format (x, y, width, height).
    grayscale : bool
        Whether to convert the frame to grayscale.
    denoise : bool
        Whether to apply Gaussian blur.
    enhance_contrast : bool
        Whether to apply CLAHE contrast enhancement.
    blur_kernel_size : int
        Kernel size used for denoising.

    Returns
    -------
    np.ndarray
        Preprocessed frame.
    """
    output = crop_roi(frame, roi)

    if grayscale:
        output = convert_to_grayscale(output)

    if denoise:
        output = apply_gaussian_blur(output, kernel_size=blur_kernel_size)

    if enhance_contrast:
        output = apply_clahe(output)

    return output


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """
    Save an image to disk.

    Parameters
    ----------
    path : str or Path
        Output image path.
    image : np.ndarray
        Image to save.

    Returns
    -------
    Path
        Saved image path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)

    if not success:
        raise IOError(f"Could not save image: {path}")

    return path
