"""Train gallery-disjoint D3, matched frozen controls and a visual LoRA ranker."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cvr.router import load_router
from cvr.utils import set_seed
from fanganC.common import atomic_json, digest, legacy, read_json, read_jsonl, snapshot_config
from fanganC.vendor.pointer import PointerHead
from fanganD.evidence_jev import ResidualPointer, evidence
from fanganD.lib import BUDGET, ListwiseFusion, build, fill_budget

SOURCE = Path('fanganC/runs/aligned-full-20261122')
SEED = 42


def batches(ids):
    return (ids[start:start + 8] for start in range(0, len(ids), 8))


def load_galleries():
    rows = read_jsonl(SOURCE/'snapshot/router_data.jsonl')
    groups = read_json(SOURCE/'alignment/galleries.json')
    mapping = {vid: i for i, group in enumerate(groups) for vid in group}
    if len(groups) != 9 or len(mapping) != 9000 or len({r['gt'] for r in rows}) != 9000 or len(rows) != 9000:
        raise ValueError('Expected nine disjoint galleries of 1000 distinct target videos')
    gallery = np.array([mapping[r['gt']] for r in rows])
    for r, g in zip(rows, gallery, strict=True):
        if len(r['cands']) != 16 or any(mapping[v] != g for v in r['cands']):
            raise ValueError(f"Candidate crossed gallery boundary for {r['qid']}")
    if not np.all(np.bincount(gallery) == 1000):
        raise ValueError('Unexpected gallery sizes')
    return rows, gallery


def routes(priority, costs, gallery):
    chosen = np.zeros(len(gallery), dtype=bool)
    for g in np.unique(gallery):
        ids = np.flatnonzero(gallery == g)
        chosen[ids], _ = fill_budget(-priority[ids], costs[ids, 2], int(costs[ids, 1].sum()),
                                     round(BUDGET * len(ids) / 1000))
    return chosen


class CandidateRanker(nn.Module):
    def __init__(self, numeric, hidden, clip_dim):
        super().__init__()
        self.numeric = copy.deepcopy(numeric)
        self.query = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, clip_dim))
        self.pointer = PointerHead(clip_dim, 128, similarity='cosine')
        self.scale = nn.Parameter(torch.zeros(2))

    def forward(self, h, x, initial, videos):
        scores, gate = self.numeric(h, x, initial)
        semantic = self.pointer(videos, self.query(h.float()))
        return scores + self.scale[None, :, None] * semantic[:, None, :], gate


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    set_seed(SEED)
    rows, gallery = load_galleries()
    _, cfg = snapshot_config(SOURCE)
    train = np.flatnonzero(gallery < 4)
    dev = np.flatnonzero(gallery == 6)
    atomic_json(out/'protocol.json', dict(train=[0,1,2,3], decision_fit=[4,5], development=6,
                calibration=7, evaluation=8, seed=SEED, alpha=cfg.router.alpha, budget=BUDGET,
                epochs=2, microbatch=8, d0_seed=1, ranker_selection='fixed-route dev R@1, then lower dev CE',
                checkpoint_selection='minimum development CE', source=str(SOURCE)))
    costs = legacy().Replay.from_rows(rows, cfg).calls
    archive = SOURCE/'features/clip.npz'
    if digest(archive) != read_json(SOURCE/'features/clip_manifest.json')['sha256']:
        raise ValueError('CLIP archive checksum mismatch')
    with np.load(archive, allow_pickle=False) as z:
        mapping = {v: i for i, v in enumerate(z['video_ids'].tolist())}
        videos = torch.from_numpy(z['video']).cuda()
    candidate = torch.tensor([[mapping[v] for v in r['cands']] for r in rows], device='cuda')
    features, scores, models = [], [], {}
    for i in range(2):
        data = build(rows, True, bool(i))
        m = ListwiseFusion(seed=1).fit([data[j] for j in train])
        raw = np.stack([x for x, _ in data])
        features.append((raw - m.mu) / m.sd)
        scores.append(m.score(raw))
        models.update({f'{i}_{k}': getattr(m,k) for k in ('mu','sd','W1','w')})
    np.savez(out/'d0.npz', **models)
    gt = torch.tensor([g for _, g in data], device='cuda')
    x = torch.tensor(np.stack(features, 1), device='cuda', dtype=torch.float32)
    initial = torch.tensor(np.stack(scores, 1), device='cuda', dtype=torch.float32)
    prob = initial[:, 0].softmax(-1).topk(2).values
    fixed = routes((prob[:, 1]-prob[:, 0]).cpu().numpy(), costs, gallery)
    np.savez(out/'inputs.npz', gallery=gallery, qids=[r['qid'] for r in rows], gt=gt.cpu().numpy(),
             original=[r['gt_rank'] for r in rows], costs=costs, fixed=fixed)
    wins = (initial.argmax(-1) == gt[:, None]).float()
    utility = wins[:, 1] - wins[:, 0]
    tok, model = load_router(cfg.router.base_model, device='cuda', train=True, lora_r=8, lora_alpha=16)
    model.config.use_cache = False
    tokens = tok([evidence(r) for r in rows], truncation=True, max_length=512)['input_ids']

    def hidden(ids):
        batch = tok.pad({'input_ids': [tokens[i] for i in ids]}, return_tensors='pt').to('cuda')
        h = model.get_base_model().model(**batch, use_cache=False).last_hidden_state
        positions = torch.arange(h.shape[1], device='cuda').expand_as(batch.attention_mask)
        return h[torch.arange(len(ids), device='cuda'), positions.masked_fill(batch.attention_mask == 0, -1).max(1).values]

    params = {k:p for k,p in model.named_parameters() if p.requires_grad}
    pristine = {k:p.detach().cpu().clone() for k,p in params.items()}
    model.eval()
    with torch.no_grad():
        frozen = torch.cat([hidden(ids) for ids in batches(np.arange(len(rows)))])
    history = {}

    def train_phase(name, head, joint, original_gate):
        set_seed(SEED)
        with torch.no_grad():
            for k,p in params.items():
                p.copy_(pristine[k])
        groups = [dict(params=list(head.parameters()), lr=1e-4)]
        if joint:
            groups.append(dict(params=list(params.values()), lr=5e-6))
        opt = torch.optim.AdamW(groups)
        rng = np.random.default_rng(SEED)
        history[name] = []
        best = float('inf')

        def forward(ids):
            states = hidden(ids) if joint else frozen[ids]
            if isinstance(head, CandidateRanker):
                return head(states, x[ids], initial[ids], videos[candidate[ids]])
            return head(states, x[ids], initial[ids])

        for epoch in range(2):
            model.train(joint)
            for step, ids in enumerate(batches(rng.permutation(train)), 1):
                scores, gate = forward(ids)
                truth = gt[ids]
                valid = truth >= 0
                loss = (F.cross_entropy(scores[valid].flatten(0,1), truth[valid].repeat_interleave(2))
                        if valid.any() else scores.sum() * 0)
                if original_gate:
                    loss = loss + F.mse_loss(gate, utility[ids])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 1., error_if_nonfinite=True)
                opt.step()
                if step not in (250, 500):
                    continue
                model.eval()
                with torch.no_grad():
                    logits = torch.cat([forward(j)[0] for j in batches(dev)])
                    truth = gt[dev]
                    valid = truth >= 0
                    routed = logits[torch.arange(len(dev)), fixed[dev].astype(int)]
                    if original_gate:
                        ce = F.cross_entropy(logits[valid].flatten(0,1), truth[valid].repeat_interleave(2))
                    else:
                        ce = F.cross_entropy(routed[valid], truth[valid])
                    value = float(ce)
                    hits = int((routed.argmax(1) == truth).sum())
                history[name].append(dict(epoch=epoch+1, step=step, dev_ce=value, fixed_hits=hits))
                if value < best:
                    best = value
                    best_head = copy.deepcopy(head.state_dict())
                    best_adapter = {k:p.detach().cpu().clone() for k,p in params.items()} if joint else {}
                atomic_json(out/'history.json', history)
                print(name, history[name][-1], flush=True)
                model.train(joint)
        head.load_state_dict(best_head)
        model.eval()
        with torch.no_grad():
            for k,v in best_adapter.items():
                params[k].copy_(v)
            torch.save(head.state_dict(), out/f'{name}_head.pt')
            if joint:
                model.save_pretrained(out/f'{name}_adapter')
            parts = [forward(ids) for ids in batches(np.arange(len(rows)))]
            np.savez(out/f'{name}.npz', scores=torch.cat([v[0] for v in parts]).cpu().numpy(),
                     gate=torch.cat([v[1] for v in parts]).cpu().numpy())
        return head

    warm = train_phase('warm', ResidualPointer(frozen.shape[1], x.shape[-1]).cuda(), False, True)
    train_phase('D3', copy.deepcopy(warm), True, True)
    from fanganD.gallery_decisions import crc_experiment
    crc_experiment(out, rows)
    train_phase('A', copy.deepcopy(warm), False, False)
    set_seed(SEED)
    visual = CandidateRanker(warm, frozen.shape[1], videos.shape[1]).cuda()
    train_phase('B', copy.deepcopy(visual), False, False)
    train_phase('C', copy.deepcopy(visual), True, False)
    from fanganD.gallery_decisions import finish
    finish(out, rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
