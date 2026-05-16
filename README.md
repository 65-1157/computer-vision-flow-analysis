# Computer Vision Flow Analysis

A computer-vision project for processing video frames, segmenting visual objects, tracking them over time, and estimating flow-related indicators.

## Project goal

This repository demonstrates an end-to-end image-processing pipeline suitable for industrial or scientific video analysis. The current version uses a deterministic computer-vision baseline. The planned evolution is to compare this baseline with neural-network-based segmentation or detection.

## Pipeline

1. **Calibration** — define scale, region of interest, and physical conversion parameters.
2. **Preprocessing** — clean frames, crop ROI, normalize contrast, denoise, and prepare chunks.
3. **Segmentation** — detect objects/components from processed frames.
4. **Tracking** — preserve object identity across frames and estimate trajectories.
5. **Temporal analysis** — smooth trajectories and compute temporal consistency.
6. **Flow characterisation** — estimate velocity, diameter, regime, and flow indicators.
7. **Campaign statistics** — generate final summary tables, figures, and quality checks.

## Repository structure

```text
computer-vision-flow-analysis/
├── notebooks/          # Executable workflow notebooks
├── src/                # Reusable Python functions extracted from notebooks
├── reports/figures/    # Selected outputs for presentation/interview
├── data/               # Local data folders, ignored by Git except .gitkeep
├── config.example.json # Example configuration
├── requirements.txt    # Python dependencies
└── README.md
```

## Current status

- Classical computer-vision baseline is available in notebooks.
- Neural-network segmentation/detection is under development.
- The repository is structured to support gradual refactoring from notebooks into reusable modules.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in numerical order.

## Main technologies

Python, OpenCV, NumPy, Pandas, Matplotlib, SciPy, Scikit-learn, Jupyter.

## Next development steps

- Move repeated notebook functions into `src/`.
- Add tests for calibration, segmentation metrics, and tracking assignment.
- Add a neural-network segmentation baseline, such as U-Net.
- Add a compact model/result comparison table.

