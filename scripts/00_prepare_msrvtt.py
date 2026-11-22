"""Convert MSR-VTT (CLIP4Clip-style annotation files) into the unified CVR format.

Outputs under paths.data_dir:
  videos.jsonl         {"video_id", "path", "split"}
  captions.json        {video_id: [caption, ...]}   (all captions, used for fragile labels)
  queries_train.jsonl  {"qid", "text", "video_id"}  (N random captions per 9k-train video)
  queries_test.jsonl   {"qid", "text", "video_id"}  (1k-A test pairs)

Other datasets (DiDeMo, ActivityNet, ...) only need converters that write the same files.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import random

from cvr.utils import add_common_args, load_config, read_json, write_json, write_jsonl


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--anno", required=True, help="MSRVTT_data.json")
    ap.add_argument("--train_csv", required=True, help="MSRVTT_train.9k.csv")
    ap.add_argument("--test_csv", required=True, help="MSRVTT_JSFUSION_test.csv")
    ap.add_argument("--video_dir", required=True, help="folder with video*.mp4")
    ap.add_argument("--caps_per_video", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    rng = random.Random(cfg.seed)

    anno = read_json(args.anno)
    captions = {}
    for s in anno["sentences"]:
        captions.setdefault(s["video_id"], []).append(s["caption"].strip())

    with open(args.train_csv, newline="", encoding="utf-8") as f:
        train_ids = [r["video_id"] for r in csv.DictReader(f)]
    with open(args.test_csv, newline="", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    test_ids = [r["video_id"] for r in test_rows]

    videos = []
    for split, ids in (("train", train_ids), ("test", test_ids)):
        for vid in ids:
            videos.append({"video_id": vid, "path": os.path.join(args.video_dir, f"{vid}.mp4"), "split": split})

    train_q = []
    for vid in train_ids:
        caps = captions.get(vid, [])
        for j, c in enumerate(rng.sample(caps, min(args.caps_per_video, len(caps)))):
            train_q.append({"qid": f"train_{vid}_{j}", "text": c, "video_id": vid})
    test_q = [{"qid": f"test_{i}", "text": r["sentence"].strip(), "video_id": r["video_id"]} for i, r in enumerate(test_rows)]

    d = cfg.paths.data_dir
    write_jsonl(os.path.join(d, "videos.jsonl"), videos)
    write_json(os.path.join(d, "captions.json"), captions)
    write_jsonl(os.path.join(d, "queries_train.jsonl"), train_q)
    write_jsonl(os.path.join(d, "queries_test.jsonl"), test_q)
    print(f"videos={len(videos)} train_queries={len(train_q)} test_queries={len(test_q)}")


if __name__ == "__main__":
    main()
