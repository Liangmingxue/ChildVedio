"""Video frame extraction and loading."""
from __future__ import annotations

import glob
import os
from typing import List

import numpy as np


def _target_indices(total: int, n: int) -> List[int]:
    if total <= 0:
        return []
    seg = total / float(n)
    return sorted({min(total - 1, int(seg * i + seg / 2)) for i in range(n)})


def _resize(frame, max_side: int):
    import cv2

    h, w = frame.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return frame


def extract_frames(video_path: str, out_dir: str, n_frames: int = 8, max_side: int = 448) -> int:
    """Uniformly sample ``n_frames`` (segment centers) and save them as JPEGs. Returns #saved."""
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:  # some containers do not report the frame count
        total = 0
        while cap.grab():
            total += 1
        cap.release()
        cap = cv2.VideoCapture(video_path)
    targets = _target_indices(total, n_frames)
    if not targets:
        cap.release()
        return 0
    target_set = set(targets)
    last = targets[-1]
    saved, idx = 0, 0
    while idx <= last:
        if not cap.grab():
            break
        if idx in target_set:
            ok, frame = cap.retrieve()
            if ok:
                frame = _resize(frame, max_side)
                cv2.imwrite(os.path.join(out_dir, f"{saved:02d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved += 1
        idx += 1
    cap.release()
    return saved


def load_frame_paths(frames_dir: str, video_id: str, n_frames: int) -> List[str]:
    paths = sorted(glob.glob(os.path.join(frames_dir, str(video_id), "*.jpg")))
    if len(paths) <= n_frames:
        return paths
    idx = np.linspace(0, len(paths) - 1, n_frames).round().astype(int)
    return [paths[i] for i in idx]
