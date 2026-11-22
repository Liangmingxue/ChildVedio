"""Export CLIP4Clip similarities in ChildVedio query and gallery order."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/msrvtt"
SOURCE = DATA / "source"
UPSTREAM = SOURCE / "clip4clip"
sys.path[:0] = [str(UPSTREAM), str(SOURCE / "python-deps")]

from dataloaders.rawvideo_util import RawVideoExtractor
from modules.modeling import CLIP4Clip
from modules.module_cross import CrossConfig
from modules.tokenization_clip import SimpleTokenizer


MAX_WORDS = 32
MAX_FRAMES = 12


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class Gallery(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.extractor = RawVideoExtractor(framerate=1, size=224)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path = self.rows[idx]["path"]
        frames = self.extractor.get_video_data(path)["video"]
        if frames.ndim != 4:
            raise ValueError(f"No CLIP4Clip frames decoded from {path}")
        if len(frames) > MAX_FRAMES:
            frames = frames[np.linspace(0, len(frames) - 1, MAX_FRAMES, dtype=int)]
        count = len(frames)
        padded = torch.zeros((MAX_FRAMES, 3, 224, 224), dtype=frames.dtype)
        padded[:count] = frames
        return padded, torch.arange(MAX_FRAMES) < count


def load_model(device):
    weights = torch.load(SOURCE / "qbnorm_clip4clip_msrvtt_7k.bin", weights_only=True, map_location="cpu")
    clip_weights = {key.removeprefix("clip."): value for key, value in weights.items() if key.startswith("clip.")}
    config = CrossConfig(str(UPSTREAM / "modules/cross-base/cross_config.json"))
    task = SimpleNamespace(
        max_words=MAX_WORDS, max_frames=MAX_FRAMES, local_rank=0,
        loose_type=True, sim_header="meanP", linear_patch="2d",
    )
    model = CLIP4Clip(config, clip_weights, task)
    model.load_state_dict(weights, strict=True)
    return model.eval().to(device)


@torch.inference_mode()
def similarities(queries, gallery, workers, batch_size):
    device = "cuda:0"
    cv2.setNumThreads(1)
    model = load_model(device)
    tokenizer = SimpleTokenizer()
    input_ids = torch.zeros((len(queries), MAX_WORDS), dtype=torch.long)
    for i, query in enumerate(queries):
        ids = [tokenizer.encoder["<|startoftext|>"]]
        ids.extend(tokenizer.encode(query["text"])[:MAX_WORDS - 2])
        ids.append(tokenizer.encoder["<|endoftext|>"])
        input_ids[i, :len(ids)] = torch.tensor(ids)
    text_features = torch.cat([model.clip.encode_text(batch.to(device)).float() for batch in input_ids.split(64)])
    loader = DataLoader(Gallery(gallery), batch_size=batch_size, num_workers=workers, pin_memory=True)
    batches = []
    for frames, mask in tqdm(loader, desc="CLIP4Clip encode gallery"):
        mask = mask.to(device, non_blocking=True)
        frames = frames.to(device, non_blocking=True).flatten(0, 1)
        visual = model.get_visual_output(frames, mask, shaped=True, video_frame=MAX_FRAMES)
        logits, _ = model.get_similarity_logits(text_features.unsqueeze(1), visual, input_ids, mask, loose_type=True)
        batches.append(logits.cpu().numpy())
    matrix = np.concatenate(batches, axis=1)
    assert matrix.shape == (len(queries), len(gallery)) and np.isfinite(matrix).all()
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    args = parser.parse_args()
    queries = read_jsonl(DATA / f"queries_{args.split}.jsonl")
    gallery = [row for row in read_jsonl(DATA / "videos.jsonl") if row["split"] == args.split]
    expected_size = {"train": 9000, "test": 1000}[args.split]
    if len(queries) != expected_size or len({row["video_id"] for row in gallery}) != expected_size:
        raise ValueError(f"Expected {expected_size} MSR-VTT {args.split} queries and videos")
    missing = [row["path"] for row in gallery if not Path(row["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {args.split} videos missing, e.g. {missing[:3]}")
    matrix = similarities(queries, gallery, args.workers, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, matrix)
    print(f"CLIP4Clip similarities {matrix.shape} -> {args.output}")


if __name__ == "__main__":
    main()
