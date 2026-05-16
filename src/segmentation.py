"""
Segmentation functions for the computer-vision pipeline.

These functions detect objects or regions of interest in preprocessed frames.
The current baseline uses classical computer-vision methods.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd


BBox = Tuple[int, int, int, int]


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a mask to a binary uint8 image with values 0 and 255.

    Parameters
    ----------
    mask : np.ndarray
        Input mask.

    Returns
    -------
    np.ndarray
        Binary mask.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    return binary


def clean_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    open_iterations: int = 1,
    close_iterations: int = 1
) -> np.ndarray:
    """
    Clean a binary mask using morphological opening and closing.

    Opening removes small noise.
    Closing fills small gaps.

    Parameters
    ----------
    mask : np.ndarray
        Input binary mask.
    kernel_size : int
        Morphological kernel size.
    open_iterations : int
        Number of opening iterations.
    close_iterations : int
        Number of closing iterations.

    Returns
    -------
    np.ndarray
        Cleaned binary mask.
    """
    binary = ensure_binary_mask(mask)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    if open_iterations > 0:
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=open_iterations
        )

    if close_iterations > 0:
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iterations
        )

    return binary


def find_external_contours(mask: np.ndarray) -> list[np.ndarray]:
    """
    Find external contours in a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Input binary mask.

    Returns
    -------
    list[np.ndarray]
        List of contours.
    """
    binary = ensure_binary_mask(mask)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def contour_centroid(contour: np.ndarray) -> Optional[tuple[float, float]]:
    """
    Compute the centroid of a contour.

    Parameters
    ----------
    contour : np.ndarray
        Input contour.

    Returns
    -------
    tuple or None
        Centroid coordinates (cx, cy), or None if the contour is invalid.
    """
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    return float(cx), float(cy)


def contour_circularity(contour: np.ndarray) -> float:
    """
    Compute contour circularity.

    Circularity is close to 1 for circular objects.

    Parameters
    ----------
    contour : np.ndarray
        Input contour.

    Returns
    -------
    float
        Circularity value.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)

    if perimeter == 0:
        return 0.0

    return float(4 * np.pi * area / (perimeter ** 2))


def contour_features(
    contour: np.ndarray,
    frame_id: Optional[int] = None,
    object_id: Optional[int] = None
) -> dict:
    """
    Extract basic geometric features from one contour.

    Parameters
    ----------
    contour : np.ndarray
        Input contour.
    frame_id : int or None
        Optional frame identifier.
    object_id : int or None
        Optional object identifier inside the frame.

    Returns
    -------
    dict
        Contour feature dictionary.
    """
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))
    x, y, width, height = cv2.boundingRect(contour)

    centroid = contour_centroid(contour)
    cx, cy = centroid if centroid is not None else (np.nan, np.nan)

    equivalent_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0.0

    return {
        "frame_id": frame_id,
        "object_id": object_id,
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
        "cx": float(cx),
        "cy": float(cy),
        "area_px": area,
        "perimeter_px": perimeter,
        "equivalent_diameter_px": float(equivalent_diameter),
        "circularity": contour_circularity(contour),
    }


def filter_contours(
    contours: list[np.ndarray],
    min_area_px: float = 20,
    max_area_px: Optional[float] = None,
    min_circularity: Optional[float] = None
) -> list[np.ndarray]:
    """
    Filter contours by area and optional circularity.

    Parameters
    ----------
    contours : list[np.ndarray]
        Input contours.
    min_area_px : float
        Minimum accepted area.
    max_area_px : float or None
        Maximum accepted area.
    min_circularity : float or None
        Minimum accepted circularity.

    Returns
    -------
    list[np.ndarray]
        Filtered contours.
    """
    selected = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area_px:
            continue

        if max_area_px is not None and area > max_area_px:
            continue

        if min_circularity is not None:
            circ = contour_circularity(contour)
            if circ < min_circularity:
                continue

        selected.append(contour)

    return selected


def segment_from_mask(
    mask: np.ndarray,
    frame_id: Optional[int] = None,
    min_area_px: float = 20,
    max_area_px: Optional[float] = None,
    min_circularity: Optional[float] = None,
    clean: bool = True,
    kernel_size: int = 3
) -> tuple[pd.DataFrame, list[np.ndarray], np.ndarray]:
    """
    Segment objects from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.
    frame_id : int or None
        Optional frame identifier.
    min_area_px : float
        Minimum accepted object area.
    max_area_px : float or None
        Maximum accepted object area.
    min_circularity : float or None
        Minimum accepted circularity.
    clean : bool
        Whether to apply mask cleaning before contour detection.
    kernel_size : int
        Morphological kernel size used for cleaning.

    Returns
    -------
    tuple
        detections_df, contours, cleaned_mask
    """
    processed_mask = clean_mask(mask, kernel_size=kernel_size) if clean else ensure_binary_mask(mask)

    contours = find_external_contours(processed_mask)

    contours = filter_contours(
        contours,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
        min_circularity=min_circularity
    )

    rows = [
        contour_features(contour, frame_id=frame_id, object_id=i)
        for i, contour in enumerate(contours)
    ]

    detections_df = pd.DataFrame(rows)

    return detections_df, contours, processed_mask


def draw_detections(
    image: np.ndarray,
    detections: pd.DataFrame,
    draw_centroids: bool = True,
    draw_boxes: bool = True,
    draw_ids: bool = True
) -> np.ndarray:
    """
    Draw detections over an image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    detections : pd.DataFrame
        Detection table with x, y, width, height, cx, cy, and object_id.
    draw_centroids : bool
        Whether to draw centroid points.
    draw_boxes : bool
        Whether to draw bounding boxes.
    draw_ids : bool
        Whether to draw object IDs.

    Returns
    -------
    np.ndarray
        Image with detections drawn.
    """
    output = image.copy()

    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    for _, row in detections.iterrows():
        x = int(row["x"])
        y = int(row["y"])
        width = int(row["width"])
        height = int(row["height"])
        cx = int(row["cx"])
        cy = int(row["cy"])

        if draw_boxes:
            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

        if draw_centroids:
            cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)

        if draw_ids and "object_id" in detections.columns:
            cv2.putText(
                output,
                str(int(row["object_id"])),
                (x, max(y - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

    return output
