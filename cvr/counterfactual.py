"""Counterfactual hypothesis generation, NLI validity filtering and fragile-type labelling."""
from __future__ import annotations

import asyncio
import json
import re
from typing import List, Optional, Sequence

import numpy as np

FRAGILE_TYPES = ["action", "temporal", "negation", "count", "attribute", "interaction", "none"]

TYPE_DESC = {
    "action": "replace the main action/verb with a visually similar but different action (e.g. put down -> pick up)",
    "temporal": "change the order or timing of events (e.g. 'A then B' -> 'B then A')",
    "negation": "add, remove or flip a negation or an absence (e.g. 'with' -> 'without')",
    "count": "change a number or quantity (e.g. 'two people' -> 'one person')",
    "attribute": "change an attribute such as color, size, gender, clothing or place",
    "interaction": "change who does what to whom, or the object of the interaction",
    "none": "change whichever single detail is most likely to be confused with the query",
    "auto": "change whichever single detail is most likely to be confused with the query",
}

_LETTERS = "ABCDEFG"


# ----------------------------------------------------------------------------- prompts
def build_cf_messages(query: str, fragile: str, m: int) -> List[dict]:
    desc = TYPE_DESC.get(fragile, TYPE_DESC["auto"])
    user = (
        f'Query: "{query}"\n\n'
        f"Write {m} counterfactual descriptions of a video. Each description must:\n"
        "1. edit the query minimally and keep most of its words;\n"
        "2. contradict the query, so that a video satisfying the query cannot satisfy it;\n"
        f"3. focus on this kind of edit: {desc};\n"
        "4. still describe a plausible video that a retrieval system could confuse with the query.\n"
        f"Return only a JSON list of {m} strings."
    )
    return [
        {"role": "system", "content": "You write minimal counterfactual descriptions for video retrieval."},
        {"role": "user", "content": user},
    ]


def build_fragile_messages(query: str, neg_caption: str) -> List[dict]:
    opts = "\n".join(f"{_LETTERS[i]}. {t}" for i, t in enumerate(FRAGILE_TYPES))
    user = (
        f'Query: "{query}"\n'
        f'Caption of a different (wrong) video that a retrieval model ranked highly: "{neg_caption}"\n\n'
        "Which single aspect best distinguishes the query from the wrong caption?\n"
        f"{opts}\nAnswer with the option letter only."
    )
    return [{"role": "user", "content": user}]


def parse_fragile(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"\b([A-G])\b", text)
    if m:
        return FRAGILE_TYPES[_LETTERS.index(m.group(1))]
    low = text.lower()
    for t in FRAGILE_TYPES:
        if t in low:
            return t
    return "none"


def _norm(text: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()


def parse_hypotheses(text: str, max_n: int = 5, query: Optional[str] = None) -> List[str]:
    """Parse a JSON list (or a bullet list as fallback); dedupe and drop copies of the query."""
    text = text or ""
    cands: List[str] = []
    m = re.search(r"\[.*\]", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                cands = [str(x) for x in obj if isinstance(x, (str, int, float))]
        except json.JSONDecodeError:
            cands = []
    if not cands:
        for line in text.splitlines():
            line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)]|[A-Za-z][.)])\s*", "", line)
            line = line.strip().strip(",").strip().strip('"').strip()
            if len(line.split()) >= 3:
                cands.append(line)
    out, seen = [], set()
    qn = _norm(query) if query else None
    for c in cands:
        c = c.strip().strip('"').strip()
        n = _norm(c)
        if not n or n in seen or n == qn:
            continue
        seen.add(n)
        out.append(c)
        if len(out) >= max_n:
            break
    return out


def rule_fallback(query: str) -> List[str]:
    """Tiny rule-based counterfactuals used only when the generator returns nothing usable."""
    q = query.strip()
    low = q.lower()
    out = []
    if " then " in low:
        i = low.index(" then ")
        out.append(f"{q[i + 6:]} then {q[:i]}")
    if " before " in low:
        out.append(re.sub(r"\bbefore\b", "after", q, flags=re.I))
    elif " after " in low:
        out.append(re.sub(r"\bafter\b", "before", q, flags=re.I))
    if " without " in low:
        out.append(re.sub(r"\bwithout\b", "with", q, flags=re.I))
    elif " with " in low:
        out.append(re.sub(r"\bwith\b", "without", q, count=1, flags=re.I))
    return [o for o in out if _norm(o) != _norm(q)]


# ----------------------------------------------------------------------------- clients
class TextLLMClient:
    """Text LLM served by vLLM (OpenAI-compatible). Used for CF generation and labelling."""

    def __init__(self, base_url: str, model: str, concurrency: int = 32, temperature: float = 0.7,
                 max_tokens: int = 256, api_key: str = "EMPTY", timeout: float = 600.0):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.concurrency = concurrency
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._sem: asyncio.Semaphore | None = None
        self.n_calls = 0

    async def chat(self, messages: List[dict], temperature: float | None = None,
                   max_tokens: int | None = None, retries: int = 3) -> str:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.concurrency)
        last_err = None
        for attempt in range(retries):
            try:
                async with self._sem:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature if temperature is None else temperature,
                        max_tokens=max_tokens or self.max_tokens,
                    )
                self.n_calls += 1
                return resp.choices[0].message.content or ""
            except Exception as err:
                last_err = err
                await asyncio.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM request failed: {last_err}")

    async def counterfactuals(self, query: str, fragile: str, m: int) -> List[str]:
        text = await self.chat(build_cf_messages(query, fragile, m))
        hyps = parse_hypotheses(text, m, query)
        return hyps or rule_fallback(query)

    async def fragile_label(self, query: str, neg_caption: str) -> str:
        text = await self.chat(build_fragile_messages(query, neg_caption), temperature=0.0, max_tokens=4)
        return parse_fragile(text)


class NLIFilter:
    """Keeps hypotheses that the NLI model judges as contradicting the query."""

    def __init__(self, model_name: str, device: str = "cuda:0", thr: float = 0.5, batch_size: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.thr = thr
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        contra = [i for i, l in id2label.items() if "contra" in l]
        if not contra:
            raise ValueError(f"Cannot find a contradiction label in {id2label}")
        self.contra_idx = contra[0]

    def contradiction_probs(self, premises: Sequence[str], hyps: Sequence[str]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(premises), self.batch_size):
                enc = self.tok(list(premises[i:i + self.batch_size]), list(hyps[i:i + self.batch_size]),
                               padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
                probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, self.contra_idx]
                out.append(probs.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def filter_many(self, queries: Sequence[str], hyp_lists: Sequence[Sequence[str]]) -> List[List[str]]:
        prem, hyp, owner = [], [], []
        for i, (q, hs) in enumerate(zip(queries, hyp_lists)):
            for h in hs:
                prem.append(q)
                hyp.append(h)
                owner.append(i)
        kept: List[List[str]] = [[] for _ in queries]
        if not prem:
            return kept
        probs = self.contradiction_probs(prem, hyp)
        for i, h, p in zip(owner, hyp, probs):
            if p >= self.thr:
                kept[i].append(h)
        return kept
