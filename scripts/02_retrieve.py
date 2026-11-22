"""First-stage retrieval: writes outputs/retrieval/{split}.jsonl with top-K candidates.

Default: CLIP frame mean-pooling. To use a stronger retriever (CLIP4Clip / InternVideo2),
export a (n_queries x n_gallery) similarity matrix aligned with queries_{split}.jsonl and
the split's video order in videos.jsonl, then pass --sim_npy.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cvr.retriever import ClipRetriever, rank_topk
from cvr.utils import add_common_args, load_config, read_jsonl, work_path, write_jsonl


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--split", default="test")
    ap.add_argument("--sim_npy", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    d = cfg.paths.data_dir
    queries = read_jsonl(os.path.join(d, f"queries_{args.split}.jsonl"))
    gallery = [v["video_id"] for v in read_jsonl(os.path.join(d, "videos.jsonl")) if v["split"] == args.split]

    if args.sim_npy:
        sim = np.load(args.sim_npy)
        assert sim.shape == (len(queries), len(gallery)), f"sim shape {sim.shape} != {(len(queries), len(gallery))}"
    else:
        r = ClipRetriever(cfg.retriever.model, device=args.device, batch_size=cfg.retriever.batch_size)
        emb_path = work_path(cfg, "retrieval", f"video_emb_{args.split}.npy")
        if os.path.exists(emb_path):
            v = np.load(emb_path)
        else:
            v = r.encode_videos(gallery, cfg.paths.frames_dir, cfg.retriever.n_frames)
            np.save(emb_path, v)
        t = r.encode_texts([q["text"] for q in queries])
        sim = t @ v.T

    rows = rank_topk(sim, queries, gallery, cfg.retriever.topk)
    out = work_path(cfg, "retrieval", f"{args.split}.jsonl")
    write_jsonl(out, rows)
    ranks = np.array([r["gt_rank"] for r in rows])
    print(f"{args.split}: n={len(rows)} R@1={100*(ranks<=1).mean():.2f} R@5={100*(ranks<=5).mean():.2f} "
          f"R@10={100*(ranks<=10).mean():.2f} MdR={np.median(ranks):.1f} -> {out}")


if __name__ == "__main__":
    main()
