"""Complete the MSR-VTT archive with byte-range downloads and extract the 9K split."""

import concurrent.futures
import json
import re
import shutil
import time
import zipfile
from pathlib import PurePosixPath

import requests
from tqdm import tqdm

from stage0_msrvtt_test_videos import DATA, FULL, PART, URL, fetch


def video_entries(archive):
    entries = {}
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        if not re.fullmatch(r"video[0-9]+\.mp4", name.name):
            continue
        if name.stem in entries:
            raise ValueError(f"Duplicate video in MSR-VTT archive: {name.stem}")
        entries[name.stem] = info
    return entries


def complete_archive():
    if FULL.exists():
        return
    with requests.get(URL, headers={"Range": "bytes=0-0"}, timeout=(10, 30), stream=True) as response:
        response.raise_for_status()
        if response.status_code != 206:
            raise OSError("MSR-VTT archive server did not accept byte ranges")
        size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
    start = PART.stat().st_size
    chunks = DATA / "source/MSRVTT.byte_ranges"
    chunks.mkdir(exist_ok=True)
    chunk_size = 16 * 1024 * 1024

    def download(offset):
        end = min(size, offset + chunk_size)
        path = chunks / f"{offset}-{end}"
        if path.exists() and path.stat().st_size == end - offset:
            return path
        for attempt in range(5):
            try:
                data = fetch(offset, end - 1)
            except (OSError, requests.RequestException):
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)
            else:
                path.write_bytes(data)
                return path

    offsets = range(start, size, chunk_size)
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        jobs = {pool.submit(download, offset): offset for offset in offsets}
        paths = {jobs[task]: task.result() for task in tqdm(concurrent.futures.as_completed(jobs), total=len(jobs),
                                                          desc="MSR-VTT archive ranges", mininterval=3)}
    with PART.open("ab") as out:
        for offset in offsets:
            with paths[offset].open("rb") as src:
                shutil.copyfileobj(src, out)
    if PART.stat().st_size != size:
        raise OSError("MSR-VTT archive size mismatch")
    with (DATA / "videos.jsonl").open() as manifest:
        expected = {row["video_id"] for row in map(json.loads, manifest)}
    with zipfile.ZipFile(PART) as archive:
        if len(expected) != 10000 or video_entries(archive).keys() != expected:
            raise ValueError("MSR-VTT archive video IDs do not match the 10,000-video manifest")
    PART.replace(FULL)
    for path in paths.values():
        path.unlink()
    if not any(chunks.iterdir()):
        chunks.rmdir()


def extract_training_videos():
    with (DATA / "videos.jsonl").open() as manifest:
        videos = [row for row in map(json.loads, manifest) if row["split"] == "train"]
    if len(videos) != 9000:
        raise ValueError("Expected the MSR-VTT 9K training split")
    with zipfile.ZipFile(FULL) as archive:
        entries = video_entries(archive)
        for row in tqdm(videos, desc="MSR-VTT training videos", mininterval=3):
            destination = DATA / "videos" / f"{row['video_id']}.mp4"
            info = entries[row["video_id"]]
            if destination.is_file() and destination.stat().st_size == info.file_size:
                continue
            partial = destination.with_suffix(".mp4.part")
            with archive.open(info) as src, partial.open("wb") as out:
                shutil.copyfileobj(src, out)
            partial.replace(destination)


if __name__ == "__main__":
    complete_archive()
    extract_training_videos()
