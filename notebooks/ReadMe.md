Notebooks
This folder contains the executable workflow of the computer-vision pipeline.
The notebooks should be executed in the order shown below. Each notebook receives outputs from previous stages and generates intermediate or final artifacts for the next stage.
---
Execution Order
Order	Notebook	Main Goal	Main Output
00	`00\_calibration.ipynb`	Define calibration parameters, physical scale, and region of interest.	Calibration configuration and ROI parameters.
00b	`00\_pre\_reference\_frame.ipynb`	Prepare the reference frame used by later processing steps.	Reference frame and related metadata.
01	`01\_preprocessing.ipynb`	Clean and standardize raw frames before segmentation.	Preprocessed frames.
02	`02\_segmentation.ipynb`	Detect objects or relevant regions in each frame.	Masks, contours, centroids, and detection tables.
03	`03\_tracking.ipynb`	Connect detections across frames and create trajectories.	Track IDs, trajectories, and tracking tables.
04	`04\_temporal\_analysis.ipynb`	Analyze object behavior through time.	Temporal features and consistency checks.
05	`05\_flow\_characterisation.ipynb`	Convert trajectories into flow-related indicators.	Velocity, displacement, size, and flow summaries.
06	`06\_campaign\_statistics.ipynb`	Consolidate the full campaign results.	Final tables, figures, and summary statistics.
---
Recommended Notebook Header
Each notebook should begin with a short markdown cell using the structure below:
```markdown
# Notebook XX — Title

## Goal
Describe what this notebook does.

## Input
Describe the files, folders, or tables expected by this notebook.

## Process
Summarize the main processing steps.

## Output
Describe the files, figures, or tables generated.

## Role in the pipeline
Explain how this notebook connects to the previous and next stages.
```
---
Development Rule
The notebooks are used to explain and validate the workflow.
Reusable logic should gradually move to the `src/` folder, for example:
repeated file-handling functions to `src/utils.py`;
image-preprocessing routines to `src/preprocessing.py`;
segmentation routines to `src/segmentation.py`;
tracking routines to `src/tracking.py`;
metric calculations to `src/metrics.py`.
This keeps the project readable for interviews and easier to maintain as software.
