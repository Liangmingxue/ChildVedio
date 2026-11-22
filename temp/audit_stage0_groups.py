"""Audit stage-0 query types and action contrasts using the saved pilot output."""

import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "outputs/stage0_clip4clip_qwen3vl8b/pilot"
BASE = "http://127.0.0.1:8000/v1"

QUERY_RULES = """Label each query independently of any retrieval result.
Temporal: the text explicitly states the order or coexistence of two distinct actions or events.
Examples: 'washes hands and then cleans a surface'; 'drives a car while talking'.
Exclude spatial 'next to', viewpoint 'first person', idioms 'look after', 'start doing X' alone,
future plans with no second observed action, and bare context like 'a goal during a soccer game'.
An ordinary 'and' that lists actions without specifying their relation does not by itself qualify.
Negation: an explicit denied, absent or intentionally avoided action, sound, participant or state.
Include 'no sound', 'doesn t move', 'tries not to cry' and an explicitly 'empty hallway'.
Return only the qids eligible for each of the two groups. A qid can be in both or neither."""

ACTION_RULES = """These are a video retrieval query and a different candidate video's human captions.
Decide whether the text documents a concrete fine-grained action contrast.
Action: the query and candidate describe a comparable situation involving similar participants,
objects or activity, but the captions give affirmative evidence for a meaningfully different
action, action stage, or action direction on those participants or objects (e.g. placing vs picking
up a cup, passing vs shooting a basketball, racing vs lining up to race).
Other: same action is compatible; or the observable difference concerns a different scene,
unrelated activity, identity, topic, object or attribute instead of a fine-grained action contrast.
Unclear: the captions are too vague or contradictory to determine whether the action is different.
Do not treat a caption failing to mention an action as proof that it did not occur.
For 'action', quote an exact supporting fragment of at most 10 words from a candidate caption.
Leave the quote empty for other labels. Give a reason of at most 15 words.
No retrieval outcome is provided or should be inferred."""


def schema(name, properties):
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": {
            "type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False,
        }},
    }


def load_rows():
    with (PILOT / "test_yesno_k16.jsonl").open() as source:
        return [json.loads(line) for line in source]


async def main():
    rows = load_rows()
    captions = json.loads((ROOT / "data/msrvtt/captions.json").read_text())
    facts = {r["qid"]: r for r in rows}
    pair_input = []
    for r in rows:
        if r["gt_rank"] != 1:
            continue
        rival = max(
            ((v, s) for v, s in zip(r["cands"], r["yes_log_odds"]) if v != r["gt"]),
            key=lambda item: item[1],
        )[0]
        pair_input.append({"qid": r["qid"], "rival": rival, "query": r["text"],
                           "candidate_captions": captions[rival]})
    assert len(rows) == 1000 and len(pair_input) == 396
    response_action = schema("action_contrast", {
        "label": {"type": "string", "enum": ["action", "other", "unclear"]},
        "reason": {"type": "string", "maxLength": 180},
        "caption_quote": {"type": "string", "maxLength": 100},
    })
    limiter = asyncio.Semaphore(24)
    async with AsyncOpenAI(base_url=BASE, api_key="EMPTY", timeout=90, max_retries=2) as client:
        model = (await client.models.list()).data[0].id

        async def ask(instructions, payload, response_format, max_tokens):
            async with limiter:
                response = await client.chat.completions.create(
                    model=model, messages=[{"role": "system", "content": instructions},
                                           {"role": "user", "content": json.dumps(payload)}],
                    response_format=response_format, temperature=0.0, max_tokens=max_tokens,
                )
            return json.loads(response.choices[0].message.content)

        async def tag_batch(batch):
            data = {item["qid"]: item["text"] for item in batch}
            qids = {"type": "array", "maxItems": len(data), "items": {"type": "string", "enum": list(data)}}
            answer = await ask(QUERY_RULES, data, schema("query_types", {"temporal": qids, "negation": qids}), 600)
            assert set(answer["temporal"] + answer["negation"]) <= data.keys()
            return answer

        async def tag_pair(example):
            data = {k: example[k] for k in ("query", "candidate_captions")}
            try:
                result = await ask(ACTION_RULES, data, response_action, 240)
            except json.JSONDecodeError:
                result = await ask(ACTION_RULES, data, response_action, 640)
            return {"qid": example["qid"], "rival": example["rival"], **result}

        qgroups = await asyncio.gather(*(tag_batch(rows[i:i + 25]) for i in range(0, len(rows), 25)))
        print("Tagged all query texts", flush=True)
        pairs = []
        for i in range(0, len(pair_input), 64):
            pairs.extend(await asyncio.gather(*(tag_pair(example) for example in pair_input[i:i + 64])))
            print(f"Tagged {len(pairs)}/{len(pair_input)} pairs", flush=True)
    groups = {tag: sorted({qid for batch in qgroups for qid in batch[tag]}, key=lambda qid: int(qid.split("_")[1]))
              for tag in ("temporal", "negation")}
    assert all(qid in facts for ids in groups.values() for qid in ids)
    data = {"model": model, "query_labels": groups, "pairs": pairs}
    raw_output = Path(__file__).with_name("group_audit_model_labels.json")
    raw_output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
