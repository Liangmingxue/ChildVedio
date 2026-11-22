"""Exercise stage 0 with generated MP4s and a local in-process VLM response stub."""

import base64
import csv
import io
import json
import re
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
N_TEST = 18


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def main():
    with tempfile.TemporaryDirectory(prefix="stage0-smoke-", dir=ROOT / "temp") as folder:
        work = Path(folder)
        ids = [f"video{i}" for i in range(N_TEST + 1)]
        captions = [f"Test query {i} then walks" for i in range(len(ids))]
        for i, video_id in enumerate(ids):
            writer = cv2.VideoWriter(
                str(work / f"{video_id}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 64)
            )
            assert writer.isOpened()
            frame = np.full((64, 64, 3), 12 + 12 * i, dtype=np.uint8)
            for _ in range(18):
                writer.write(frame)
            writer.release()

        anno = work / "anno.json"
        anno.write_text(json.dumps({"sentences": [
            {"video_id": video_id, "caption": caption} for video_id, caption in zip(ids, captions)
        ]}))
        train_csv, test_csv = work / "train.csv", work / "test.csv"
        write_csv(train_csv, ["video_id"], [(ids[-1],)])
        write_csv(test_csv, ["video_id", "sentence"], zip(ids[:N_TEST], captions[:N_TEST]))
        overrides = [
            "--set", f"paths.data_dir={work / 'data'}", f"paths.frames_dir={work / 'frames'}",
            f"paths.work_dir={work / 'outputs'}", "retriever.topk=16", "vlm.yes_no_token_ids=[1,2]",
        ]

        def run(script, *args):
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / script), *args, *overrides],
                           cwd=ROOT, check=True)

        run("00_prepare_msrvtt.py", "--anno", str(anno), "--train_csv", str(train_csv),
            "--test_csv", str(test_csv), "--video_dir", str(work))
        run("01_extract_frames.py", "--workers", "2")
        assert all(len(list((work / "frames" / video_id).glob("*.jpg"))) == 16 for video_id in ids)

        scores = np.tile(np.linspace(0.7, 0.0, N_TEST), (N_TEST, 1))
        np.fill_diagonal(scores, 0.9)
        scores[1, 0] = 0.95
        scores[2, 0] = 0.9
        matrix = work / "sim.npy"
        np.save(matrix, scores)
        run("02_retrieve.py", "--split", "test", "--sim_npy", str(matrix))
        retrieval_path = work / "outputs/retrieval/test.jsonl"
        retrieval = [json.loads(line) for line in retrieval_path.read_text().splitlines()]
        assert all(len(row["cands"]) == 16 for row in retrieval)
        assert retrieval[2]["cands"][:2] == ["video0", "video2"]
        assert retrieval[2]["gt_rank"] == 2, "A tie must use the same order for candidates and rank"

        seen = set()

        async def reply(**body):
            assert body["max_tokens"] == 1 and body["logprobs"]
            assert body["extra_body"]["logprob_token_ids"] == [1, 2]
            content = body["messages"][0]["content"]
            assert len(content) == 9 and all(part["type"] == "image_url" for part in content[:-1])
            frame_bytes = base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1])
            with Image.open(io.BytesIO(frame_bytes)) as frame_image:
                video = round((frame_image.getpixel((0, 0))[0] - 12) / 12)
            query = int(re.search(r"Test query (\d+)", content[-1]["text"]).group(1))
            seen.add((query, video))
            preferred = {0: 1, 2: 0}.get(query, query)
            answer = "Yes" if video == preferred else "No"
            other = "No" if answer == "Yes" else "Yes"
            other_logprob = -50.0 if video == preferred else -4.0
            if (query, video) == (0, 0):
                answer, other, other_logprob = "Yes", "No", -40.0
            logits = [SimpleNamespace(token=answer, logprob=-0.1), SimpleNamespace(token=other, logprob=other_logprob)]
            content = [SimpleNamespace(top_logprobs=logits)]
            return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=content))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=reply)))
        pilot = ROOT / "scripts/03_pilot_yesno.py"
        pilot_args = [str(pilot), "--k", "16", "--beta", "1", *overrides]
        with patch("openai.AsyncOpenAI", return_value=client), patch.object(sys, "argv", pilot_args):
            runpy.run_path(str(pilot), run_name="__main__")

        metrics = json.loads((work / "outputs/pilot/test_yesno_k16_metrics.json").read_text())
        assert len(seen) == N_TEST * 16
        assert metrics["base"]["n"] == N_TEST and metrics["vlm_calls_per_query"] == 16
        assert metrics["flips"]["harmful_flips"] == metrics["flips"]["helpful_flips"] == 1
        assert metrics["flips"]["HFR"] == 6.25 and metrics["flips"]["HelpR"] == 50
        assert metrics["by_tag"]["temporal"]["n"] == N_TEST
        print(f"Stage 0 smoke test passed ({len(seen)} stubbed requests, 8 frames per request)")


if __name__ == "__main__":
    main()
