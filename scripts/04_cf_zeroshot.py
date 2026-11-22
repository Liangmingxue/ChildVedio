"""Compare yes/no, independent delta and forced choice on the same candidates."""
import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cvr.factory import make_llm, make_nli, make_verifier
from cvr.metrics import breakdown_by_tag, summarize_ranks
from cvr.rerank import new_gt_rank, rerank_topk
from cvr.utils import add_common_args, gather_progress, load_config, read_json, read_jsonl, work_path, write_json, write_jsonl


async def gen_hyps(cfg, rows):
    llm = make_llm(cfg, "llm")
    async with llm.client:
        return await gather_progress([llm.counterfactuals(r["text"], "auto", cfg.cf.m) for r in rows], "counterfactuals")


def cached_yes_scores(cfg, path, rows, k):
    if path is None:
        return [None] * len(rows)
    metrics = read_json(os.path.splitext(path)[0] + "_metrics.json")
    expected = {"vlm_model": cfg.vlm.model, "n_frames": cfg.frames.n_frames,
                "yes_no_token_ids": cfg.vlm.get("yes_no_token_ids")}
    if any(metrics.get(key) != value for key, value in expected.items()):
        raise ValueError("Cached yes/no scores were produced with a different verifier configuration")
    saved = {r["qid"]: r for r in read_jsonl(path)}
    result = []
    for row in rows:
        old = saved.get(row["qid"])
        candidates = row["cands"][:k]
        if not old or old["text"] != row["text"] or old["gt"] != row["gt"] or old["cands"][:k] != candidates:
            raise ValueError(f"Cached yes/no candidates do not match {row['qid']}")
        values = old["yes"][:k]
        if len(values) != len(candidates) or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError(f"Invalid cached yes/no scores for {row['qid']}")
        result.append(values)
    return result


async def score_batches(cfg, rows, hypotheses, cached_yes, modes, k, completed, progress_path):
    verifier = make_verifier(cfg)

    async def score_candidate(row, hyps, video, yes):
        if yes is None and (not hyps or "yesno" in modes or "delta" in modes):
            yes = await verifier.yes_prob(row["text"], video)
        scores = {mode: yes for mode in modes}
        if hyps and "delta" in modes:
            alternatives = await asyncio.gather(*[verifier.yes_prob(h, video) for h in hyps])
            scores["delta"] = yes - max(alternatives)
        if hyps and "fc" in modes:
            scores["fc"] = await verifier.forced_choice(
                row["text"], hyps, video, n_perm=cfg.vlm.n_perm, seed=cfg.seed
            )
        return scores

    async with verifier.client:
        for start in range(len(completed), len(rows), 32):
            stop = min(start + 32, len(rows))
            jobs = sorted((video, i, j) for i in range(start, stop) for j, video in enumerate(rows[i]["cands"][:k]))
            values = await gather_progress([
                score_candidate(rows[i], hypotheses[i]["hyps"], video, None if cached_yes[i] is None else cached_yes[i][j])
                for video, i, j in jobs
            ], f"queries {start + 1}–{stop}/{len(rows)}")
            by_position = {(i, j): result for (_, i, j), result in zip(jobs, values)}
            for i in range(start, stop):
                row = rows[i]
                record = {**row, **hypotheses[i]}
                for mode in modes:
                    scores = [by_position[i, j][mode] for j in range(len(row["cands"][:k]))]
                    if not all(math.isfinite(s) for s in scores):
                        raise ValueError(f"Non-finite {mode} score for {row['qid']}")
                    rank = new_gt_rank(row, rerank_topk(row["cands"], row["scores"], scores, cfg.rerank.beta))
                    record[f"{mode}_scores"], record[f"{mode}_rank"] = scores, rank
                completed.append(record)
            write_jsonl(progress_path + ".part", completed)
            os.replace(progress_path + ".part", progress_path)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=None, help="default: rerank.k_deep")
    ap.add_argument("--n", type=int, default=500, help="number of queries (0 = all)")
    ap.add_argument("--modes", nargs="+", choices=["yesno", "delta", "fc"], default=["yesno", "delta", "fc"])
    ap.add_argument("--yesno-jsonl", help="reuse verified yes/no probabilities for the same query and candidates")
    ap.add_argument("--resume", action="store_true", help="reuse hypotheses and completed query batches for this run")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    k = cfg.rerank.k_deep if args.k is None else args.k
    if k < 1 or args.n < 0 or cfg.vlm.n_perm < 1:
        ap.error("k and vlm.n_perm must be positive, and n cannot be negative")
    modes = list(dict.fromkeys(args.modes))
    rows = read_jsonl(work_path(cfg, "retrieval", f"{args.split}.jsonl"))
    if args.n:
        rows = random.Random(cfg.seed).sample(rows, min(args.n, len(rows)))
    if not rows or any(not row["cands"] for row in rows):
        raise ValueError("Stage 1 requires nonempty retrieval candidates")
    cached_yes = cached_yes_scores(cfg, args.yesno_jsonl, rows, k)

    prefix = work_path(cfg, "cf_zeroshot", f"{args.split}_k{k}")
    settings = {"config": cfg, "k": k, "modes": modes, "yesno_source": args.yesno_jsonl}
    signature = hashlib.sha256(json.dumps([settings, rows, cached_yes], sort_keys=True).encode()).hexdigest()
    manifest_path, hyp_path, progress_path = (prefix + suffix for suffix in ("_run.json", "_hypotheses.jsonl", "_progress.jsonl"))
    if args.resume and os.path.exists(manifest_path):
        if read_json(manifest_path)["signature"] != signature:
            raise ValueError("Run settings or inputs differ from the checkpoint")
    else:
        for path in (hyp_path, progress_path):
            if os.path.exists(path):
                os.remove(path)
        write_json(manifest_path, {**settings, "signature": signature, "qid_order": [r["qid"] for r in rows]})

    if os.path.exists(hyp_path):
        hypotheses = read_jsonl(hyp_path)
    else:
        generated = asyncio.run(gen_hyps(cfg, rows))
        nli = make_nli(cfg)
        accepted = nli.filter_many([r["text"] for r in rows], generated)
        del nli
        hypotheses = [dict(qid=r["qid"], generated_hyps=g, hyps=h) for r, g, h in zip(rows, generated, accepted)]
        write_jsonl(hyp_path + ".part", hypotheses)
        os.replace(hyp_path + ".part", hyp_path)
    if [h["qid"] for h in hypotheses] != [r["qid"] for r in rows]:
        raise ValueError("Hypotheses do not match the queries")
    covered = sum(bool(h["hyps"]) for h in hypotheses)
    print(f"Queries with a valid counterfactual: {covered}/{len(rows)}", flush=True)

    per_query = read_jsonl(progress_path) if os.path.exists(progress_path) else []
    if [r["qid"] for r in per_query] != [r["qid"] for r in rows[:len(per_query)]]:
        raise ValueError("Completed rows do not match the requested queries")
    if len(per_query) < len(rows):
        asyncio.run(score_batches(cfg, rows, hypotheses, cached_yes, modes, k, per_query, progress_path))

    base = [r["gt_rank"] for r in per_query]
    metrics = {"k": k, "n": len(rows), "beta": cfg.rerank.beta, "n_perm": cfg.vlm.n_perm,
               "n_with_counterfactuals": covered, "nli_threshold": cfg.cf.nli_thr,
               "n_generated_hypotheses": sum(len(h["generated_hyps"]) for h in hypotheses),
               "n_accepted_hypotheses": sum(len(h["hyps"]) for h in hypotheses)}
    for mode in modes:
        new = [r[f"{mode}_rank"] for r in per_query]
        calls = 0
        for row in per_query:
            attempts = 1
            if mode == "delta":
                attempts += len(row["hyps"])
            elif mode == "fc" and row["hyps"]:
                attempts = cfg.vlm.n_perm
            calls += attempts * len(row["cands"][:k])
        metrics[mode] = {**summarize_ranks(base, new), "vlm_calls_per_query": calls / len(rows),
                         "by_tag": breakdown_by_tag([r["text"] for r in per_query], base, new)}
        if mode != "yesno" and "yesno" in modes:
            metrics[mode]["yesno_harms_recovered"] = sum(
                r["gt_rank"] == 1 and r["yesno_rank"] != 1 and r[f"{mode}_rank"] == 1 for r in per_query
            )
            metrics[mode]["yesno_helps_retained"] = sum(
                r["gt_rank"] != 1 and r["yesno_rank"] == 1 and r[f"{mode}_rank"] == 1 for r in per_query
            )
        f = metrics[mode]["flips"]
        print(f"[{mode}] R@1 {metrics[mode]['base']['R@1']:.2f}->{metrics[mode]['new']['R@1']:.2f} "
              f"HFR={f['HFR']:.2f}% HelpR={f['HelpR']:.2f}% calls/q={calls / len(rows):.1f}")
    write_jsonl(prefix + ".jsonl", per_query)
    write_json(prefix + "_metrics.json", metrics)


if __name__ == "__main__":
    main()
