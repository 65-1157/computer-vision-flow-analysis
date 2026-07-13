# ── BATCH FRAME HARVESTER ────────────────────────────────────────────────────
# Extracts the N clearest frames from a LIST of .mp4 files for annotation.
# Enforces temporal and cross-video diversity.
# Saves images to annotation_pool/ and prints a CVAT-ready summary.
#
# Upgrade from single-video version:
#   - Loops over VIDEO_LIST automatically
#   - Per-video quota + global dedup prevents one video dominating the pool
#   - Exports a harvest_manifest.json for full traceability
#   - Skips already-harvested frames (safe to re-run)
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# ── GOOGLE DRIVE MOUNT ────────────────────────────────────────────────────────
import sys, subprocess
try:
    from google.colab import drive; IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
    from google.colab import drive
    try:
        drive.mount('/content/drive', force_remount=False)
        print('Drive mounted at /content/drive')
    except Exception as _e:
        if 'already contain files' in str(_e) or 'symlink' in str(_e):
            subprocess.run(['umount', '/content/drive'], capture_output=True)
            subprocess.run(['rm', '-rf', '/content/drive'], capture_output=True)
            drive.mount('/content/drive', force_remount=False)
            print('Drive remounted successfully.')
        else:
            raise
else:
    print('Not in Colab — Drive mount skipped.')

# ─────────────────────────────────────────────────────────────────────────────
# ── EDIT THESE ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

DRIVE_BASE = "/content/drive/MyDrive/SIGNAL_NN_2026/HIDRO_2026/campaigns/setup_A"

VIDEO_LIST = [
    # (run_label,  full path to .mp4)
    ("run_001_Q10", f"{DRIVE_BASE}/run_001_Q10/VID_20260413_124627.mp4"),
    ("run_002_Q20", f"{DRIVE_BASE}/run_002_Q20/VID_20260429_124526.mp4"),
    ("run_003_Q30", f"{DRIVE_BASE}/run_003_Q30/VID_20260429_125119.mp4"),
    ("run_004_Q40", f"{DRIVE_BASE}/run_004_Q40/VID_20260429_125717.mp4"),
    ("run_005_Q50", f"{DRIVE_BASE}/run_005_Q50/VID_20260429_123807.mp4"),
    # Add a 6th video here when available:
    # ("run_006_Q60", f"{DRIVE_BASE}/run_006_Q60/VID_XXXXXXXX_XXXXXX.mp4"),
]

OUTPUT_DIR      = "/content/drive/MyDrive/SIGNAL_NN_2026/HIDRO_2026/anot_pool/images_batch"
FRAMES_PER_VIDEO = 25          # frames to extract per video  →  6×25 = 150 total
ROI             = [0, 450, 2160, 3400]   # [x0, y0, x1, y1] from config.json
MIN_BUBBLES     = 3            # reject frames with fewer bright blobs
MAX_BUBBLES     = 8            # tighter than before: avoids crowded frames
TARGET_BUBBLES  = 5            # scorer peaks here (quality sweet spot)
SAMPLE_EVERY_N  = 5            # consider every Nth frame (avoids near-duplicates)
MIN_FRAME_GAP   = 30           # minimum frame distance between selected frames
                               # (30 frames @ 50fps = 0.6 s apart — ensures diversity)
JPEG_QUALITY    = 95

# ─────────────────────────────────────────────────────────────────────────────


def score_frame(gray, n_bubbles, target=TARGET_BUBBLES):
    """
    Score = Laplacian variance (sharpness) × density penalty.
    Peaks at target bubble count, falls off for extremes.
    """
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    density_penalty = 1.0 - abs(n_bubbles - target) / float(max(target, 1))
    density_penalty = max(0.0, density_penalty)
    return sharpness * density_penalty


def count_valid_blobs(gray, roi_area):
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh)
    min_area = 50
    max_area = int(roi_area * 0.15)
    valid = [
        s for s in stats[1:]
        if min_area < s[cv2.CC_STAT_AREA] < max_area
    ]
    return len(valid)


def harvest_video(run_label, video_path, out_dir, frames_to_save,
                  roi, min_bubbles, max_bubbles, sample_every_n, min_gap):
    """
    Scan one video and return the best `frames_to_save` frames.
    Returns list of (frame_idx, score, frame_image, filename).
    """
    x0, y0, x1, y1 = roi
    roi_area = (x1 - x0) * (y1 - y0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n  Video : {Path(video_path).name}")
    print(f"  Frames: {total}  |  FPS: {fps:.2f}")

    candidates = []
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip startup / shutdown frames
        if frame_idx < 10 or frame_idx > total - 10:
            frame_idx += 1
            continue

        # Temporal subsampling
        if frame_idx % sample_every_n != 0:
            frame_idx += 1
            continue

        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        n_bubbles = count_valid_blobs(gray, roi_area)
        if n_bubbles < min_bubbles or n_bubbles > max_bubbles:
            frame_idx += 1
            continue

        sc = score_frame(gray, n_bubbles)
        candidates.append((frame_idx, sc, frame.copy()))
        frame_idx += 1

    cap.release()
    print(f"  Candidates found: {len(candidates)}")

    # Sort by score descending, enforce minimum temporal gap
    candidates.sort(key=lambda x: -x[1])
    selected    = []
    used_indices = set()

    for fidx, sc, frm in candidates:
        if any(abs(fidx - u) < min_gap for u in used_indices):
            continue
        video_stem = Path(video_path).stem
        fname = f"{video_stem}_frame{fidx:05d}.jpg"
        selected.append((fidx, sc, frm, fname))
        used_indices.add(fidx)
        if len(selected) >= frames_to_save:
            break

    print(f"  Selected : {len(selected)} frames  "
          f"(target={frames_to_save}, gap≥{min_gap})")
    return selected


# ── MAIN ─────────────────────────────────────────────────────────────────────

out = Path(OUTPUT_DIR)
out.mkdir(parents=True, exist_ok=True)

manifest = {
    "created_at":      datetime.now().isoformat(),
    "output_dir":      str(out),
    "frames_per_video": FRAMES_PER_VIDEO,
    "roi":             ROI,
    "min_bubbles":     MIN_BUBBLES,
    "max_bubbles":     MAX_BUBBLES,
    "target_bubbles":  TARGET_BUBBLES,
    "sample_every_n":  SAMPLE_EVERY_N,
    "min_frame_gap":   MIN_FRAME_GAP,
    "videos":          [],
}

total_saved  = 0
all_filenames = []

print("=" * 70)
print("BATCH FRAME HARVESTER")
print("=" * 70)

for run_label, video_path in VIDEO_LIST:

    if not Path(video_path).exists():
        print(f"\n[SKIP] {run_label}: file not found → {video_path}")
        manifest["videos"].append({
            "run_label":   run_label,
            "video_path":  video_path,
            "status":      "file_not_found",
            "n_saved":     0,
            "frames":      [],
        })
        continue

    print(f"\n[{run_label}]")
    selected = harvest_video(
        run_label    = run_label,
        video_path   = video_path,
        out_dir      = out,
        frames_to_save = FRAMES_PER_VIDEO,
        roi          = ROI,
        min_bubbles  = MIN_BUBBLES,
        max_bubbles  = MAX_BUBBLES,
        sample_every_n = SAMPLE_EVERY_N,
        min_gap      = MIN_FRAME_GAP,
    )

    video_record = {
        "run_label":  run_label,
        "video_path": video_path,
        "status":     "ok",
        "n_saved":    0,
        "frames":     [],
    }

    for fidx, score, frm, fname in sorted(selected, key=lambda x: x[0]):
        fpath = out / fname
        if fpath.exists():
            print(f"    [SKIP — already exists] {fname}")
        else:
            cv2.imwrite(str(fpath), frm, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            print(f"    frame {fidx:5d}  score={score:8.2f}  → {fname}")

        video_record["frames"].append({
            "frame_idx": fidx,
            "score":     round(score, 4),
            "filename":  fname,
        })
        all_filenames.append(fname)
        total_saved += 1

    video_record["n_saved"] = len(selected)
    manifest["videos"].append(video_record)

# ── Save manifest ─────────────────────────────────────────────────────────────
manifest["total_saved"] = total_saved
manifest_path = out.parent / "harvest_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("HARVEST COMPLETE")
print("=" * 70)
print(f"  Total frames saved : {total_saved}")
print(f"  Output directory   : {out}")
print(f"  Manifest           : {manifest_path}")
print()
print("Per-video breakdown:")
for v in manifest["videos"]:
    status = "✓" if v["status"] == "ok" else "✗"
    print(f"  {status} {v['run_label']:<18}  {v['n_saved']:>3} frames")
print()
print("Next steps:")
print("  1. Upload the entire output folder to CVAT as a new task")
print("  2. Draw rectangles around each bubble (4–6 per image target)")
print("  3. Export as COCO 1.0 → save as anot_pool/i_def_XX.json")
print("  4. Run Cell 7B (bbox→ellipse converter) before retraining")
print("  5. Set FORCE_RETRAIN=True in NB-02 Cell 7 and rerun Cell 8")
