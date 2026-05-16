"""
Basic smoke tests for the computer-vision pipeline.

These tests do not validate the full scientific accuracy of the project.
They only check whether the main reusable functions run without breaking.
"""

import numpy as np
import pandas as pd

from src.preprocessing import (
    convert_to_grayscale,
    crop_roi,
    preprocess_frame,
)

from src.segmentation import (
    ensure_binary_mask,
    clean_mask,
    segment_from_mask,
)

from src.tracking import (
    build_tracks,
    summarize_tracks,
)

from src.metrics import (
    add_physical_tracking_metrics,
    campaign_summary,
)


def test_preprocessing_basic_frame():
    """
    Check whether a synthetic image can be preprocessed.
    """
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[30:60, 30:60] = 255

    gray = convert_to_grayscale(frame)
    assert gray.shape == (100, 100)

    cropped = crop_roi(gray, roi=(20, 20, 50, 50))
    assert cropped.shape == (50, 50)

    processed = preprocess_frame(frame, roi=(20, 20, 50, 50))
    assert processed.shape == (50, 50)
    assert processed.dtype == np.uint8


def test_segmentation_from_synthetic_mask():
    """
    Check whether a simple binary object can be segmented.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:60, 30:60] = 255

    binary = ensure_binary_mask(mask)
    assert binary.max() == 255
    assert binary.min() == 0

    cleaned = clean_mask(binary)
    assert cleaned.shape == mask.shape

    detections, contours, cleaned_mask = segment_from_mask(
        cleaned,
        frame_id=0,
        min_area_px=20,
    )

    assert len(contours) >= 1
    assert not detections.empty
    assert "cx" in detections.columns
    assert "cy" in detections.columns
    assert cleaned_mask.shape == mask.shape


def test_tracking_from_synthetic_detections():
    """
    Check whether simple detections can be connected into tracks.
    """
    detections = pd.DataFrame(
        {
            "frame_id": [0, 1, 2, 0, 1, 2],
            "cx": [10, 12, 14, 80, 82, 84],
            "cy": [10, 12, 14, 80, 82, 84],
            "area_px": [100, 100, 100, 120, 120, 120],
        }
    )

    tracks = build_tracks(
        detections,
        max_distance_px=10,
        max_missing_frames=1,
        min_track_length=2,
    )

    assert not tracks.empty
    assert "track_id" in tracks.columns
    assert "step_displacement_px" in tracks.columns
    assert tracks["track_id"].nunique() == 2

    summary = summarize_tracks(tracks)
    assert not summary.empty
    assert "duration_frames" in summary.columns


def test_metrics_from_synthetic_tracks():
    """
    Check whether physical metrics can be computed from tracks.
    """
    tracks = pd.DataFrame(
        {
            "track_id": [0, 0, 0],
            "frame_id": [0, 1, 2],
            "cx": [10, 12, 14],
            "cy": [10, 12, 14],
            "step_displacement_px": [np.nan, 2.8, 2.8],
            "frame_gap": [np.nan, 1, 1],
        }
    )

    metrics = add_physical_tracking_metrics(
        tracks,
        pixel_to_mm=0.5,
        frame_rate_fps=30,
    )

    assert "step_displacement_mm" in metrics.columns
    assert "velocity_mm_s" in metrics.columns

    summary = campaign_summary(tracks=metrics)

    assert "n_tracks" in summary
    assert summary["n_tracks"] == 1
