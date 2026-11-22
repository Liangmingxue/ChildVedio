"""Frozen VLM verifier served by vLLM through its OpenAI-compatible API.

Two scoring modes (both read first-token log-probs, no free-text generation):

* ``yes_prob``      positive verification: p(Yes | video, query)
* ``forced_choice`` counterfactual forced choice over {query, h_1..h_m, none}; returns
                    p(query | video) from a restricted softmax over the option letters,
                    averaged over cyclic option permutations to remove position bias.

Frames are sent *before* the text so that vLLM prefix caching re-uses the video KV cache
when the same video is scored with different option sets (important for RL rewards).
"""
from __future__ import annotations

import asyncio
import base64
import io
import math
import random
from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np

from .data import load_frame_paths
from .utils import logsumexp, softmax_np, stable_hash

LETTERS = "ABCDEFGHIJKL"
NONE_OPTION = "None of the other descriptions matches the video."
FLOOR = -50.0

YESNO_PROMPT = (
    "The images are {n} frames sampled uniformly from one video, in temporal order.\n"
    'Description: "{query}"\n'
    "Does the video match the description exactly, including the actions, their temporal order, "
    "negations, counts, attributes and interactions? Answer with Yes or No only."
)

FC_PROMPT = (
    "The images are {n} frames sampled uniformly from one video, in temporal order.\n"
    "Which description matches the video? Check the actions, their temporal order, negations, "
    "counts, attributes and interactions carefully.\n{options}\n"
    "Answer with the option letter only."
)


class VLMVerifier:
    def __init__(
        self,
        base_url: str,
        model: str,
        frames_dir: str,
        n_frames: int = 8,
        max_side: int = 336,
        concurrency: int = 32,
        option_temp: float = 1.0,
        api_key: str = "EMPTY",
        timeout: float = 600.0,
        cache_size: int = 4096,
        yes_no_token_ids: Sequence[int] | None = None,
        option_token_ids: Dict[str, Sequence[int]] | None = None,
    ):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.frames_dir = frames_dir
        self.n_frames = n_frames
        self.max_side = max_side
        self.concurrency = concurrency
        self.option_temp = option_temp
        self.cache_size = cache_size
        self.yes_no_token_ids = yes_no_token_ids
        self.option_token_ids = option_token_ids
        self._sem: asyncio.Semaphore | None = None
        self._img_cache: "OrderedDict[str, list]" = OrderedDict()
        self.n_calls = 0

    # ------------------------------------------------------------------ helpers
    def _semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.concurrency)
        return self._sem

    def _images(self, video_id: str) -> list:
        if video_id in self._img_cache:
            self._img_cache.move_to_end(video_id)
            return self._img_cache[video_id]
        from PIL import Image

        items = []
        for p in load_frame_paths(self.frames_dir, video_id, self.n_frames):
            im = Image.open(p).convert("RGB")
            im.thumbnail((self.max_side, self.max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            items.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        if not items:
            raise FileNotFoundError(f"No frames for video {video_id} under {self.frames_dir}")
        self._img_cache[video_id] = items
        if len(self._img_cache) > self.cache_size:
            self._img_cache.popitem(last=False)
        return items

    async def _first_token_logprobs(self, video_id: str, text: str, retries: int = 3,
                                    token_ids: Sequence[int] | None = None) -> Dict[str, float]:
        content = list(self._images(video_id)) + [{"type": "text", "text": text}]
        messages = [{"role": "user", "content": content}]
        logprob_options = {"extra_body": {"logprob_token_ids": list(token_ids)}} if token_ids else {"top_logprobs": 20}
        last_err = None
        for attempt in range(retries):
            try:
                async with self._semaphore():
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=1,
                        temperature=0.0,
                        logprobs=True,
                        **logprob_options,
                    )
                break
            except Exception as err:  # network hiccups / server overload
                last_err = err
                await asyncio.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"VLM request failed for {video_id}: {last_err}")
        self.n_calls += 1
        grouped: Dict[str, List[float]] = {}
        lp = resp.choices[0].logprobs
        if lp is not None and lp.content:
            for t in lp.content[0].top_logprobs:
                grouped.setdefault(t.token.strip(), []).append(t.logprob)
        return {k: logsumexp(v) for k, v in grouped.items()}

    @staticmethod
    def _get(lps: Dict[str, float], keys: Sequence[str]) -> float:
        vals = [lps[k] for k in keys if k in lps]
        return logsumexp(vals) if vals else FLOOR

    # ------------------------------------------------------------------ scoring
    async def yes_log_odds(self, query: str, video_id: str) -> float:
        lps = await self._first_token_logprobs(
            video_id, YESNO_PROMPT.format(n=self.n_frames, query=query), token_ids=self.yes_no_token_ids
        )
        yes, no = ("Yes", "yes", "YES"), ("No", "no", "NO")
        if self.yes_no_token_ids and not (any(k in lps for k in yes) and any(k in lps for k in no)):
            raise RuntimeError(f"VLM did not return both Yes and No logprobs for {video_id}")
        return self._get(lps, yes) - self._get(lps, no)

    async def yes_prob(self, query: str, video_id: str) -> float:
        log_odds = await self.yes_log_odds(query, video_id)
        return 1.0 / (1.0 + math.exp(-max(min(log_odds, 60.0), -60.0)))

    @staticmethod
    def _orders(n_options: int, n_perm: int, key: str) -> List[List[int]]:
        rng = random.Random(stable_hash(key))
        base = list(range(n_options))
        rng.shuffle(base)
        n_perm = max(1, n_perm)
        orders = []
        for k in range(n_perm):
            shift = (k * n_options) // n_perm
            orders.append(base[shift:] + base[:shift])
        return orders

    async def _fc_once(self, options: Sequence[str], order: Sequence[int], video_id: str) -> float:
        letters = LETTERS[: len(options)]
        lines = "\n".join(f"{letters[i]}. {options[j]}" for i, j in enumerate(order))
        token_ids = [token for letter in letters for token in self.option_token_ids[letter]] if self.option_token_ids else None
        lps = await self._first_token_logprobs(video_id, FC_PROMPT.format(n=self.n_frames, options=lines), token_ids=token_ids)
        if token_ids and any(letter not in lps for letter in letters):
            raise RuntimeError(f"VLM did not return every option logprob for {video_id}")
        logits = np.array([self._get(lps, (l,)) for l in letters]) / self.option_temp
        probs = softmax_np(logits)
        return float(probs[list(order).index(0)])  # option 0 is the original query

    async def forced_choice(
        self, query: str, hyps: Sequence[str], video_id: str, n_perm: int = 1, seed: int = 0
    ) -> float:
        hyps = list(hyps)[: len(LETTERS) - 2]
        options = [query] + hyps + [NONE_OPTION]
        orders = self._orders(len(options), n_perm, f"{seed}|{query}")
        ps = await asyncio.gather(*[self._fc_once(options, o, video_id) for o in orders])
        return float(np.mean(ps))

    async def independent_delta(self, query: str, hyps: Sequence[str], video_id: str) -> float:
        """Ablation: S(Q,v) - max_j S(h_j,v) with independent yes/no scoring."""
        sq = await self.yes_prob(query, video_id)
        if not hyps:
            return sq
        sh = await asyncio.gather(*[self.yes_prob(h, video_id) for h in hyps])
        return float(sq - max(sh))
