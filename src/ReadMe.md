# Source Code Modules

This folder contains reusable Python code used by the notebooks.

The notebooks explain the workflow step by step. The `src/` modules contain functions that can be reused, tested, and improved without duplicating code across notebooks.

---

## Module Overview

| Module | Purpose |
|---|---|
| `utils.py` | General helper functions for paths, configuration files, JSON handling, folder creation, and safe file saving. |
| `preprocessing.py` | Image-preprocessing routines such as cropping, grayscale conversion, denoising, contrast adjustment, and threshold preparation. |
| `segmentation.py` | Functions for detecting objects or regions of interest, including masks, contours, centroids, bounding boxes, and shape features. |
| `tracking.py` | Functions for connecting detections across frames, assigning track IDs, handling lost tracks, and estimating object trajectories. |
| `metrics.py` | Functions for calculating physical and statistical indicators such as displacement, velocity, object size, and campaign-level summaries. |

---

## Development Principle

The project follows a simple rule:

> Notebooks are used for explanation and validation.  
> Reusable logic should live in `src/`.

This avoids repeated code and makes the project easier to maintain.

---

## Recommended Usage

From a notebook, functions should be imported from `src/` instead of being rewritten.

Example:

```python
from src.preprocessing import preprocess_frame
from src.segmentation import segment_objects
from src.tracking import build_tracks
from src.metrics import compute_flow_metrics
