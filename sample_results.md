# Sample Results

This file summarizes the expected outputs of the computer-vision flow-analysis pipeline.

The project is still under development. Therefore, this document should be updated whenever a new experiment, campaign, or processing run is completed.

---

## 1. Input

The pipeline receives video frames or extracted image sequences from an experimental campaign.

Typical input data:

- raw video file or frame sequence;
- calibration reference;
- region of interest;
- frame rate;
- pixel-to-physical-unit conversion factor.

Large raw files should not be committed to GitHub.

---

## 2. Processing Summary

The current baseline follows this sequence:

```text
Calibration
→ Reference frame preparation
→ Preprocessing
→ Segmentation
→ Tracking
→ Temporal analysis
→ Flow characterisation
→ Campaign statistics
