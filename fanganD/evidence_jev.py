"""Offline D0, frozen-Qwen D2 and LoRA D3 with post-light routing."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cvr.calibration import crc_threshold
from cvr.router import load_router
from cvr.utils import set_seed
from fanganC.common import atomic_json, legacy, read_jsonl, resolve, snapshot_config
from fanganD.lib import BUDGET, ListwiseFusion, build, fill_budget, load_run, metrics, paired_bootstrap


def evidence(row):
    coarse = ' '.join(f'{v:.2f}' for v in row['scores'])
    light = ' '.join(f'{v:.3f}' for v in row['light_scores'])
    return f"Query: {row['text']}\nCandidates in retrieval order: {coarse}\nYes probabilities for the first five: {light}\nDecide whether to escalate and which candidate to trust."


class ResidualPointer(nn.Module):
    def __init__(self, hidden, dims):
        super().__init__()
        self.context = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 32), nn.Tanh())
        self.pointer = nn.Linear(32, 2 * dims)
        self.upgrade = nn.Linear(32, 1)
        for layer in (self.pointer, self.upgrade):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, states, x, initial):
        h = self.context(states.float())
        offsets = self.pointer(h).reshape(len(h), 2, -1)
        scores = initial + (x * offsets[:, :, None]).sum(-1)
        gap = scores[:, 0].softmax(-1).topk(2).values
        return scores, self.upgrade(h).flatten() + gap[:, 1] - gap[:, 0]


def main(source, out):
    out.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    cfg, stage = snapshot_config(source)
    rows, test, _, _ = load_run(source)
    rows += [dict(r, split='test') for r in test]
    split = {s: np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ('train','dev','calib','test')}
    tr, dev = split['train'], split['dev']
    data = [build(rows, True, deep) for deep in (False, True)]
    gt = np.array([g for _,g in data[0]])
    original = np.array([r['gt_rank'] for r in rows])
    costs = legacy().Replay.from_rows(rows, stage).calls
    labels = {r['qid']:r['fragile'] for r in read_jsonl(resolve(stage.paths.work_dir)/'routes/test.jsonl')}

    def routed(ids, scores, priority):
        budget = round(BUDGET*len(ids)/1000)
        chosen, spent = fill_budget(-priority, costs[ids,2], int(costs[ids,1].sum()), budget)
        final = scores[np.arange(len(ids)), chosen.astype(int)]
        order = np.argsort(-final, kind='stable')
        rank = np.where(gt[ids]>=0, (order == gt[ids,None]).argmax(1)+1, original[ids])
        p = torch.tensor(final).softmax(1).topk(2).values.numpy()
        return rank, p[:,0]-p[:,1], spent, chosen, order[:,0]

    def gaps(scores):
        p = torch.tensor(scores[:,0]).softmax(1).topk(2).values.numpy()
        return p[:,1]-p[:,0]

    def listwise_loss(scores, ids):
        valid = targets[ids] >= 0
        labels = targets[ids][valid].repeat_interleave(2)
        return F.cross_entropy(scores[valid].flatten(0, 1), labels) if labels.numel() else scores.sum() * 0

    targets = torch.tensor(gt, device='cuda')
    best = None
    trials = []
    for seed in (0,1,2):
        models = [ListwiseFusion(seed=seed).fit([d[i] for i in tr]) for d in data]
        raw = np.stack([[m.score(x) for x,_ in d] for m,d in zip(models,data)],1)
        ranks = routed(dev,raw[dev],gaps(raw[dev]))[0]
        value = float(listwise_loss(torch.as_tensor(raw[dev], device='cuda'), dev))
        trials.append(dict(seed=seed, dev_R1=metrics(ranks)['R@1'], dev_listwise_loss=value))
        if best is None or value < best[0]:
            best = value,seed,models,raw
        print('D0', trials[-1],flush=True)
    _,seed,models,raw = best
    np.savez(out/'d0.npz', **{f'{i}_{k}':getattr(m,k) for i,m in enumerate(models) for k in ('mu','sd','W1','w')})
    atomic_json(out/'d0_selection.json',dict(selected_seed=seed,trials=trials))
    standardized = np.stack([[(x-m.mu)/m.sd for x,_ in d] for m,d in zip(models,data)],1)
    x = torch.tensor(standardized,dtype=torch.float32,device='cuda')
    initial = torch.tensor(raw,dtype=torch.float32,device='cuda')
    tok, model = load_router(str(resolve(stage.router.base_model)),device='cuda',train=True,lora_r=8,lora_alpha=16)
    model.config.use_cache=False
    tokens=tok([evidence(r) for r in rows],truncation=True,max_length=512)['input_ids']
    def hidden(ids):
        batch=tok.pad({'input_ids':[tokens[i] for i in ids]},return_tensors='pt').to('cuda')
        states=model.get_base_model().model(**batch,use_cache=False).last_hidden_state
        pos=torch.arange(states.shape[1],device='cuda').expand_as(batch.attention_mask)
        return states[torch.arange(len(ids),device='cuda'),pos.masked_fill(batch.attention_mask==0,-1).max(1).values]

    model.eval()
    with torch.no_grad():
        frozen=torch.cat([hidden(np.arange(s,min(s+8,len(rows)))).detach() for s in range(0,len(rows),8)])
    head=ResidualPointer(frozen.shape[1],x.shape[-1]).cuda()
    predictions={'D0':(raw,gaps(raw))}
    # Fixed D0 action outcomes supervise the expected gain from upgrading.
    winner=raw.argmax(-1)==gt[:,None]
    utility=torch.tensor(winner[:,1].astype(float)-winner[:,0],device='cuda',dtype=torch.float32)
    def loss_fn(scores,gate,ids):
        return listwise_loss(scores, ids) + F.mse_loss(gate,utility[ids])

    history = {}
    for phase in ('D2','D3'):
        history[phase] = []
        joint=phase=='D3'
        groups=[dict(params=head.parameters(),lr=1e-4)]
        if joint:
            groups.append(dict(params=[p for p in model.parameters() if p.requires_grad],lr=5e-6))
        opt=torch.optim.AdamW(groups)
        best_loss=float('inf')
        rng=np.random.default_rng(42)
        def forward(ids):
            return head(hidden(ids) if joint else frozen[ids],x[ids],initial[ids])
        for epoch in range(2):
            model.train(joint)
            batches=np.array_split(rng.permutation(tr),int(np.ceil(len(tr)/8)))
            for step,ids in enumerate(batches,1):
                opt.zero_grad(set_to_none=True)
                loss=loss_fn(*forward(ids),ids)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],1.,error_if_nonfinite=True)
                opt.step()
                if step not in ((len(batches)+1)//2,len(batches)):
                    continue
                model.eval()
                with torch.no_grad():
                    scores,gates=zip(*(forward(dev[s:s+8]) for s in range(0,len(dev),8)))
                    dev_scores = torch.cat(scores)
                    value = float(listwise_loss(dev_scores, dev))
                    dev_ranks = routed(dev, dev_scores.cpu().numpy(), torch.cat(gates).cpu().numpy())[0]
                    history[phase].append(dict(epoch=epoch+1, step=step, dev_listwise_loss=value,
                                               dev_R1=metrics(dev_ranks)['R@1']))
                    atomic_json(out/'checkpoint_history.json', history)
                if value<best_loss:
                    best_loss=value
                    state=copy.deepcopy(head.state_dict())
                    adapter={k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad} if joint else {}
                    torch.save(state,out/f'{phase}_head.pt')
                    if joint:model.save_pretrained(out/'D3_adapter')
                print(phase,epoch,step,'dev_listwise_loss',value,flush=True)
                model.train(joint)
        head.load_state_dict(state)
        with torch.no_grad():
            for k,p in model.named_parameters():
                if k in adapter:p.copy_(adapter[k])
            model.eval()
            eval_ids = np.concatenate((dev, split['calib'], split['test']))
            scores, gates = zip(*(forward(eval_ids[s:s+8]) for s in range(0, len(eval_ids), 8)))
            phase_scores = np.full(raw.shape, np.nan, dtype=np.float32)
            phase_gates = np.full(len(rows), np.nan, dtype=np.float32)
            phase_scores[eval_ids] = torch.cat(scores).cpu().numpy()
            phase_gates[eval_ids] = torch.cat(gates).cpu().numpy()
            predictions[phase] = phase_scores, phase_gates

    report={}
    hits={}
    raw_hits = {}
    for phase,(scores,gates) in predictions.items():
        cal=split['calib']
        ranks,conf,*_=routed(cal,scores[cal],gates[cal])
        good=original[cal]==1
        tau=crc_threshold(conf[good],ranks[good]!=1,stage.router.alpha)
        ids=split['test']
        ranks,conf,spent,chosen,pointer=routed(ids,scores[ids],gates[ids])
        final=np.where(conf>=tau,ranks,original[ids])
        hits[phase]=final==1
        raw_hits[phase] = ranks == 1
        types=np.array([labels[rows[i]['qid']] for i in ids])
        report[phase]=dict(dev_raw=metrics(routed(dev, scores[dev], gates[dev])[0]), raw=metrics(ranks),crc=metrics(final),threshold=tau,calls=spent,upgrades=int(chosen.sum()),
                           harmful=int(((original[ids]==1)&(final!=1)).sum()),by_predicted_fragile={
                           k:dict(n=int((types==k).sum()),**metrics(final[types==k])) for k in sorted(set(types))})
        decisions=[dict(qid=rows[i]['qid'],upgrade=bool(u),pointer=int(p if confidence>=tau else 0),rank=int(r))
                   for i,u,p,confidence,r in zip(ids,chosen,pointer,conf,final)]
        atomic_json(out/f'{phase}_decisions.json',decisions)
    report['D3_vs_D0_raw'] = paired_bootstrap(raw_hits['D3'], raw_hits['D0'])
    report['D3_vs_D2_raw'] = paired_bootstrap(raw_hits['D3'], raw_hits['D2'])
    report['D3_vs_D0_crc']=paired_bootstrap(hits['D3'],hits['D0'])
    atomic_json(out/'results.json',report)


if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('source',type=Path)
    args.add_argument('output',type=Path)
    paths=args.parse_args()
    main(paths.source,paths.output)
