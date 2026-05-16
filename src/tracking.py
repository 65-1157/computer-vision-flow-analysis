"""
Tracking functions for the computer-vision pipeline.

The current baseline uses a simple nearest-neighbor tracker.
This is intentionally explainable and suitable as a first tracking baseline.
"""

from typing import Optional

import numpy as np
import pandas as pd


def euclidean_distance(
    point_a: tuple[float, float],
    point_b: tuple[float, float]
) -> float:
    """
    Compute Euclidean distance between two 2D points.

    Parameters
    ----------
    point_a : tuple
        First point as (x, y).
    point_b : tuple
        Second point as (x, y).

    Returns
    -------
    float
        Euclidean distance.
    """
    ax, ay = point_a
    bx, by = point_b

    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def prepare_detections(
    detections: pd.DataFrame,
    frame_col: str = "frame_id",
    x_col: str = "cx",
    y_col: str = "cy"
) -> pd.DataFrame:
    """
    Prepare detection table for tracking.

    Parameters
    ----------
    detections : pd.DataFrame
        Detection table.
    frame_col : str
        Name of the frame column.
    x_col : str
        Name of the x-coordinate column.
    y_col : str
        Name of the y-coordinate column.

    Returns
    -------
    pd.DataFrame
        Cleaned and sorted detection table.
    """
    required_cols = [frame_col, x_col, y_col]

    missing_cols = [col for col in required_cols if col not in detections.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    output = detections.copy()

    output = output.dropna(subset=required_cols)
    output = output.sort_values([frame_col, x_col, y_col]).reset_index(drop=True)

    return output


def nearest_neighbor_tracking(
    detections: pd.DataFrame,
    frame_col: str = "frame_id",
    x_col: str = "cx",
    y_col: str = "cy",
    max_distance_px: float = 30.0,
    max_missing_frames: int = 5
) -> pd.DataFrame:
    """
    Track detections across frames using a nearest-neighbor rule.

    The tracker assigns each detection to the closest active track when the
    distance is below `max_distance_px`. If no valid track is found, a new
    track is created.

    Parameters
    ----------
    detections : pd.DataFrame
        Detection table with frame and centroid columns.
    frame_col : str
        Name of the frame column.
    x_col : str
        Name of the x-coordinate column.
    y_col : str
        Name of the y-coordinate column.
    max_distance_px : float
        Maximum distance allowed to associate a detection with an active track.
    max_missing_frames : int
        Maximum number of frames a track can remain unmatched before being
        considered inactive.

    Returns
    -------
    pd.DataFrame
        Detection table with an added `track_id` column.
    """
    detections = prepare_detections(
        detections,
        frame_col=frame_col,
        x_col=x_col,
        y_col=y_col
    )

    if detections.empty:
        output = detections.copy()
        output["track_id"] = pd.Series(dtype="int")
        return output

    next_track_id = 0

    active_tracks = {}
    tracked_rows = []

    frame_ids = sorted(detections[frame_col].unique())

    for frame_id in frame_ids:
        frame_data = detections[detections[frame_col] == frame_id].copy()

        assigned_tracks = set()

        for _, detection in frame_data.iterrows():
            detection_point = (float(detection[x_col]), float(detection[y_col]))

            best_track_id = None
            best_distance = np.inf

            for track_id, track_info in active_tracks.items():
                if track_id in assigned_tracks:
                    continue

                missing_frames = frame_id - track_info["last_frame"]

                if missing_frames > max_missing_frames:
                    continue

                track_point = track_info["last_point"]

                distance = euclidean_distance(detection_point, track_point)

                if distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is not None and best_distance <= max_distance_px:
                track_id = best_track_id
            else:
                track_id = next_track_id
                next_track_id += 1

            active_tracks[track_id] = {
                "last_point": detection_point,
                "last_frame": frame_id
            }

            assigned_tracks.add(track_id)

            row = detection.to_dict()
            row["track_id"] = int(track_id)
            row["tracking_distance_px"] = (
                float(best_distance)
                if np.isfinite(best_distance) and best_track_id is not None
                else np.nan
            )

            tracked_rows.append(row)

        active_tracks = {
            track_id: track_info
            for track_id, track_info in active_tracks.items()
            if frame_id - track_info["last_frame"] <= max_missing_frames
        }

    return pd.DataFrame(tracked_rows)


def add_track_statistics(
    tracks: pd.DataFrame,
    frame_col: str = "frame_id",
    track_col: str = "track_id",
    x_col: str = "cx",
    y_col: str = "cy"
) -> pd.DataFrame:
    """
    Add frame-to-frame displacement information to tracked detections.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracking table.
    frame_col : str
        Name of the frame column.
    track_col : str
        Name of the track ID column.
    x_col : str
        Name of the x-coordinate column.
    y_col : str
        Name of the y-coordinate column.

    Returns
    -------
    pd.DataFrame
        Tracking table with displacement columns.
    """
    required_cols = [frame_col, track_col, x_col, y_col]

    missing_cols = [col for col in required_cols if col not in tracks.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    output = tracks.copy()
    output = output.sort_values([track_col, frame_col]).reset_index(drop=True)

    output["dx_px"] = output.groupby(track_col)[x_col].diff()
    output["dy_px"] = output.groupby(track_col)[y_col].diff()

    output["step_displacement_px"] = np.sqrt(
        output["dx_px"] ** 2 + output["dy_px"] ** 2
    )

    output["frame_gap"] = output.groupby(track_col)[frame_col].diff()

    output["step_displacement_per_frame_px"] = (
        output["step_displacement_px"] / output["frame_gap"]
    )

    return output


def summarize_tracks(
    tracks: pd.DataFrame,
    frame_col: str = "frame_id",
    track_col: str = "track_id",
    x_col: str = "cx",
    y_col: str = "cy",
    min_track_length: int = 1
) -> pd.DataFrame:
    """
    Summarize each track into one row.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracking table.
    frame_col : str
        Name of the frame column.
    track_col : str
        Name of the track ID column.
    x_col : str
        Name of the x-coordinate column.
    y_col : str
        Name of the y-coordinate column.
    min_track_length : int
        Minimum number of detections required to keep a track.

    Returns
    -------
    pd.DataFrame
        Track-level summary table.
    """
    required_cols = [frame_col, track_col, x_col, y_col]

    missing_cols = [col for col in required_cols if col not in tracks.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    rows = []

    for track_id, group in tracks.groupby(track_col):
        group = group.sort_values(frame_col)

        n_detections = len(group)

        if n_detections < min_track_length:
            continue

        first_frame = group[frame_col].min()
        last_frame = group[frame_col].max()

        start_x = group.iloc[0][x_col]
        start_y = group.iloc[0][y_col]
        end_x = group.iloc[-1][x_col]
        end_y = group.iloc[-1][y_col]

        total_displacement_px = euclidean_distance(
            (start_x, start_y),
            (end_x, end_y)
        )

        duration_frames = last_frame - first_frame + 1

        rows.append(
            {
                "track_id": int(track_id),
                "n_detections": int(n_detections),
                "first_frame": int(first_frame),
                "last_frame": int(last_frame),
                "duration_frames": int(duration_frames),
                "start_x": float(start_x),
                "start_y": float(start_y),
                "end_x": float(end_x),
                "end_y": float(end_y),
                "total_displacement_px": float(total_displacement_px),
                "mean_x": float(group[x_col].mean()),
                "mean_y": float(group[y_col].mean()),
            }
        )

    return pd.DataFrame(rows)


def filter_short_tracks(
    tracks: pd.DataFrame,
    min_track_length: int = 3,
    track_col: str = "track_id"
) -> pd.DataFrame:
    """
    Remove tracks with fewer detections than the minimum length.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracking table.
    min_track_length : int
        Minimum number of detections required.
    track_col : str
        Name of the track ID column.

    Returns
    -------
    pd.DataFrame
        Filtered tracking table.
    """
    if track_col not in tracks.columns:
        raise ValueError(f"Missing required column: {track_col}")

    track_sizes = tracks.groupby(track_col).size()

    valid_track_ids = track_sizes[track_sizes >= min_track_length].index

    return tracks[tracks[track_col].isin(valid_track_ids)].reset_index(drop=True)


def build_tracks(
    detections: pd.DataFrame,
    frame_col: str = "frame_id",
    x_col: str = "cx",
    y_col: str = "cy",
    max_distance_px: float = 30.0,
    max_missing_frames: int = 5,
    min_track_length: int = 3
) -> pd.DataFrame:
    """
    Build object tracks from frame-level detections.

    This is the main high-level function used by the notebooks.

    Parameters
    ----------
    detections : pd.DataFrame
        Detection table.
    frame_col : str
        Name of the frame column.
    x_col : str
        Name of the x-coordinate column.
    y_col : str
        Name of the y-coordinate column.
    max_distance_px : float
        Maximum distance used for nearest-neighbor association.
    max_missing_frames : int
        Maximum number of missing frames allowed for an active track.
    min_track_length : int
        Minimum number of detections required to keep a track.

    Returns
    -------
    pd.DataFrame
        Tracking table with track IDs and displacement information.
    """
    tracks = nearest_neighbor_tracking(
        detections=detections,
        frame_col=frame_col,
        x_col=x_col,
        y_col=y_col,
        max_distance_px=max_distance_px,
        max_missing_frames=max_missing_frames
    )

    tracks = filter_short_tracks(
        tracks,
        min_track_length=min_track_length,
        track_col="track_id"
    )

    tracks = add_track_statistics(
        tracks,
        frame_col=frame_col,
        track_col="track_id",
        x_col=x_col,
        y_col=y_col
    )

    return tracks
