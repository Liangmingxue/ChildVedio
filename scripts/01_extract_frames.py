"""Extract uniformly sampled frames for every video in videos.jsonl (CPU, multiprocessing)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

from cvr.data import extract_frames
from cvr.utils import add_common_args, load_config, read_jsonl


def _job(row, frames_dir, n, max_side):
    out = os.path.join(frames_dir, row["video_id"])
    if os.path.isdir(out) and len(os.listdir(out)) >= n:
        return row["video_id"], n
    return row["video_id"], extract_frames(row["path"], out, n, max_side)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--split", default=None, help="extract frames only for this split")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n_frames", type=int, default=None, help="frames stored on disk (default 2x frames.n_frames)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    n = args.n_frames or 2 * cfg.frames.n_frames
    rows = read_jsonl(os.path.join(cfg.paths.data_dir, "videos.jsonl"))
    if args.split is not None:
        rows = [row for row in rows if row["split"] == args.split]
        if not rows:
            ap.error(f"No videos in split {args.split!r}")
    fn = partial(_job, frames_dir=cfg.paths.frames_dir, n=n, max_side=cfg.frames.store_side)
    bad = []
    with Pool(args.workers) as pool:
        for vid, saved in tqdm(pool.imap_unordered(fn, rows, chunksize=8), total=len(rows), desc="frames"):
            if saved == 0:
                bad.append(vid)
    print(f"done; {len(bad)} videos failed" + (f", e.g. {bad[:5]}" if bad else ""))


if __name__ == "__main__":
    main()
