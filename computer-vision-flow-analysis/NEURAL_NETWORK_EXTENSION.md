# Neural-Network Extension Plan

This document describes how neural networks have been added to the current computer-vision pipeline.

The current project version uses classical computer-vision methods as an explainable baseline. This is useful because every processing stage can be inspected and validated.

The neural-network extension do not replace the whole pipeline at once. It should start by improving the stage where neural networks add the most value: segmentation.

---

## 1. Current Baseline

The current pipeline follows this structure:

```text
Raw frames
→ Preprocessing
→ Classical segmentation
→ Tracking
→ Temporal analysis
→ Flow metrics
→ Campaign statistics
