"""Extract only the MSR-VTT 1K-A videos, using ZIP byte ranges until the full archive is local."""

import concurrent.futures
import json
import os
import shutil
import threading
import zipfile
from pathlib import Path, PurePosixPath

import requests
from tqdm import tqdm


DATA = Path(__file__).resolve().parents[1] / "data/msrvtt"
URL = "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip"
FULL = DATA / "source/MSRVTT.zip"
PART = FULL.with_suffix(".zip.part")
THREAD = threading.local()


def fetch(start, end):
    if not hasattr(THREAD, "session"):
        THREAD.session = requests.Session()
    response = THREAD.session.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=(10, 60), stream=True)
    with response:
        expected = f"bytes {start}-{end}/"
        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
            raise OSError(f"Unexpected HTTP range response for {start}-{end}: {response.status_code}")
        data = response.content
        if len(data) != end - start + 1:
            raise OSError(f"Short HTTP range response for {start}-{end}")
        return data


class ArchiveReader:
    def __init__(self, size):
        self.size = size
        self.offset = 0
        self.cache = threading.local()

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=os.SEEK_SET):
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self.offset, os.SEEK_END: self.size}[whence]
        self.offset = base + offset
        if self.offset < 0:
            raise OSError("Seek before start of ZIP")
        return self.offset

    def prefetch(self, start, end):
        end = min(end, self.size)
        for path in (FULL, PART):
            try:
                with path.open("rb") as source:
                    if os.fstat(source.fileno()).st_size >= end:
                        source.seek(start)
                        self.cache.block = (start, source.read(end - start))
                        return
            except FileNotFoundError:
                pass
        self.cache.block = (start, fetch(start, end - 1))

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.offset + size, self.size)
        if self.offset >= end:
            return b""
        block_start, block = getattr(self.cache, "block", (0, b""))
        if self.offset < block_start or end > block_start + len(block):
            self.prefetch(self.offset, end)
            block_start, block = self.cache.block
        start = self.offset - block_start
        self.offset, count = end, end - self.offset
        return block[start:start + count]


def main():
    gallery = {row["video_id"]: Path(row["path"]) for row in map(json.loads, (DATA / "videos.jsonl").open())
               if row["split"] == "test"}
    assert len(gallery) == 1000
    with requests.get(URL, headers={"Range": "bytes=0-0"}, timeout=(10, 15), stream=True) as response:
        if response.status_code != 206:
            raise OSError(f"Server did not accept byte ranges: {response.status_code}")
        size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
    reader = ArchiveReader(size)
    index = zipfile.ZipFile(reader)
    entries = {}
    for info in index.infolist():
        name = PurePosixPath(info.filename)
        if name.suffix == ".mp4" and name.stem in gallery:
            if name.stem in entries:
                raise ValueError(f"Duplicate video in archive: {name.stem}")
            entries[name.stem] = info
    missing = gallery.keys() - entries.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} MSR-VTT test videos: {sorted(missing)[:5]}")

    def extract(video_id):
        destination = gallery[video_id]
        info = entries[video_id]
        if destination.is_file() and destination.stat().st_size == info.file_size:
            return
        for attempt in range(3):
            try:
                start = info.header_offset
                reader.prefetch(start, start + info.compress_size + 256)
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_suffix(".mp4.part")
                with index.open(info) as source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output)
                partial.replace(destination)
                return
            except (requests.RequestException, OSError):
                if attempt == 2:
                    raise

    ordered = sorted(entries, key=lambda vid: entries[vid].header_offset)
    with index, concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for _ in tqdm(pool.map(extract, ordered), total=len(ordered), desc="MSR-VTT 1K-A videos"):
            pass


if __name__ == "__main__":
    main()
