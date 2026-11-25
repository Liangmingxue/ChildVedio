"""Full-coverage GRPO and budget-matched D3 replay."""
import argparse
import asyncio
from itertools import count
import time
import random
from pathlib import Path

import numpy as np
import torch
from cvr.calibration import crc_threshold
from cvr.counterfactual import parse_hypotheses
from cvr.factory import make_nli, make_verifier
from cvr.grpo import CFGRPOTrainer, build_grpo_rows
from cvr.router import load_router
from cvr.utils import set_seed
from fanganC.common import atomic_json, read_json, read_jsonl, snapshot_config
from fanganD.evidence_jev import ResidualPointer, evidence
from fanganD.lib import ListwiseFusion, build, fill_budget, metrics, paired_bootstrap

SOURCE = Path('fanganC/runs/aligned-full-20261122')
D3 = Path('fanganD/runs/jev-20261122-153820')


class Replay:
    def __init__(self, rows, cfg):
        self.rows = rows
        self.models = []
        with np.load(D3/'d0.npz') as archive:
            for i in range(2):
                m = ListwiseFusion()
                for key in ('mu','sd','W1','w'):
                    setattr(m,key,archive[f'{i}_{key}'])
                self.models.append(m)
        tok, model = load_router(cfg.router.base_model, adapter=str(D3/'D3_adapter'), device='cuda')
        head = ResidualPointer(model.config.hidden_size,23).cuda().eval()
        head.load_state_dict(torch.load(D3/'D3_head.pt',weights_only=True))
        self.offsets, self.bonuses = [], []
        with torch.inference_mode():
            for start in range(0,len(rows),8):
                batch = tok([evidence(r) for r in rows[start:start+8]],padding=True,truncation=True,
                            max_length=512,return_tensors='pt').to('cuda')
                state = model.get_base_model().model(**batch,use_cache=False).last_hidden_state
                pos = torch.arange(state.shape[1],device='cuda').expand_as(batch.attention_mask)
                h = head.context(state[torch.arange(len(state),device='cuda'),pos.masked_fill(batch.attention_mask==0,-1).max(1).values].float())
                self.offsets.extend(head.pointer(h).reshape(-1,2,23).cpu().numpy())
                self.bonuses.extend(head.upgrade(h).flatten().cpu().tolist())
        self.offsets = torch.tensor(np.array(self.offsets))
        self.bonuses = np.array(self.bonuses)
        self.ids = {s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ('dev','calib','test')}

    def route(self, split, rows, fixed=None):
        ids = self.ids[split]
        scores = []
        for i,m in enumerate(self.models):
            x = np.stack([v for v,_ in build(rows,True,bool(i))])
            initial = torch.tensor(m.score(x),dtype=torch.float32)
            features = torch.tensor((x-m.mu)/m.sd,dtype=torch.float32)
            scores.append(initial+(features*self.offsets[ids,i,None,:]).sum(-1))
        gap = scores[0].softmax(1).topk(2).values
        priority = self.bonuses[ids] + (gap[:,1]-gap[:,0]).numpy()
        costs = np.array([len(r['cands'])*(2 if r['hyps'] else 1) for r in rows])
        sel,spent = fill_budget(-priority,costs,sum(len(r['light_scores']) for r in rows),round(27824*len(rows)/1000))
        if fixed is not None:
            sel=fixed
            spent=int(sum(len(r["light_scores"]) for r in rows)+costs[sel].sum())
        final = torch.where(torch.tensor(sel[:,None]),scores[1],scores[0])
        order = final.argsort(descending=True,stable=True).numpy()
        ranks = np.array([list(np.array(r['cands'])[o]).index(r['gt'])+1 if r['gt'] in r['cands'] else r['gt_rank']
                          for o,r in zip(order,rows,strict=True)])
        p = final.softmax(1).topk(2).values
        return sel,spent,ranks,(p[:,0]-p[:,1]).numpy()


@torch.inference_mode()
def generate(trainer,rows):
    t=trainer
    t.model.eval()
    prompts=[t._prompt(dict(text=r['text'],fragile='auto')) for r in rows]
    enc=t.tok(prompts,return_tensors='pt',padding=True,add_special_tokens=False).to(t.device)
    ids=t.model.generate(**enc,do_sample=False,max_new_tokens=128,use_cache=True,pad_token_id=t.tok.pad_token_id)
    texts=t.tok.batch_decode(ids[:,enc.input_ids.shape[1]:],skip_special_tokens=True)
    queries=[r['text'] for r in rows]
    return t.nli.filter_many(queries,[parse_hypotheses(s,3,q) for s,q in zip(texts,queries,strict=True)])


def evaluate(replay,trainer,split,tag,out):
    rows=[dict(replay.rows[i]) for i in replay.ids[split]]
    old,_,_,_=replay.route(split,rows)
    if tag!='A':
        for start in range(0,len(rows),8):
            batch=rows[start:start+8]
            for row,h in zip(batch,generate(trainer,batch),strict=True):
                row['hyps']=h
        chosen,*_=replay.route(split,rows)
        async def score(r):
            v=trainer.verifier
            return await asyncio.gather(*(v.forced_choice(r['text'],r['hyps'],c,n_perm=2,seed=42)
                         if r['hyps'] else v.yes_prob(r['text'],c) for c in r['cands']))
        targets=np.flatnonzero(chosen|old)
        for start in range(0,len(targets),8):
            ids=targets[start:start+8]
            async def batch_score():
                return await asyncio.gather(*(score(rows[i]) for i in ids))
            for i,s in zip(ids,trainer.loop.run_until_complete(batch_score()),strict=True):
                rows[i]['deep_scores']=s
        # Compare both fixed original upgrade assignments and cost-adjusted routing.
    sel,spent,ranks,conf=replay.route(split,rows)
    original=np.array([r['gt_rank'] for r in rows])
    report=dict(n=len(rows),metrics=metrics(ranks),calls=spent,upgrades=int(sel.sum()),
                harmful=int(((original==1)&(ranks!=1)).sum()),empty_hypotheses=sum(not r['hyps'] for r in rows))
    _,fixed_cost,fixed_rank,_=replay.route(split,rows,fixed=old)
    report["fixed_original_routes"]=dict(calls=fixed_cost,metrics=metrics(fixed_rank))
    records=[dict(qid=r['qid'],rank=int(rank),upgrade=bool(up),confidence=float(p),hyps=r['hyps'],
                  deep_scores=r['deep_scores'] if up or prior else None)
             for r,rank,up,prior,p in zip(rows,ranks,sel,old,conf,strict=True)]
    atomic_json(out/f'{tag}_{split}.json',dict(report=report,queries=records))
    return report,ranks,conf,original


def main(out):
    set_seed(42)
    _,cfg=snapshot_config(SOURCE)
    cfg.grpo.policy_model='data/models/qwen2.5-3b-instruct'
    cfg.grpo.lr=float(cfg.grpo.lr)
    cfg.grpo.m_max=3
    cfg.grpo.r_invalid=-2.
    cfg.grpo.lam_count=0.
    cfg.grpo.updates_per_rollout=2
    cfg.grpo.micro_batch=2
    cfg.vlm.base_url='http://127.0.0.1:8026/v1'
    data=read_jsonl(SOURCE/'snapshot/router_data.jsonl')
    train=build_grpo_rows(data,3,16)
    test=[dict(r,split='test') for r in read_jsonl(SOURCE/'snapshot/test.jsonl')]
    replay=Replay([r for r in data if r['split']!='train']+test,cfg)
    torch.cuda.empty_cache()
    summary,A,_,_=evaluate(replay,None,'test','A',out)
    if (len(A),sum(A==1),summary['calls'],summary['upgrades'])!=(1000,531,27800,717):
        raise ValueError(f'D3 replay mismatch: {summary}')
    print('D3 replay verified',summary,flush=True)
    assert len(train)==7407 and len(set(r['qid'] for r in train))==len(train)
    model_dir=Path(cfg.grpo.policy_model)
    shards=set(read_json(model_dir/"model.safetensors.index.json")["weight_map"].values())
    print("Waiting for complete 3B policy shards",flush=True)
    for attempt in range(1440):
        if all((model_dir/name).is_file() for name in shards):break
        time.sleep(10)
    else:raise FileNotFoundError("Policy download incomplete")
    trainer=CFGRPOTrainer(cfg,make_verifier(cfg),make_nli(cfg),str(out))
    trainer.verifier.cache_size=256
    best,*_=evaluate(replay,trainer,'dev','B',out)
    key=lambda r:(r['metrics']['R@1'],-r['harmful'])
    # Keep the best trained adapter separately from the untrained control.
    best_rl=None
    history=[]
    step=stale=0
    rng=random.Random(cfg.seed)
    print("Early stopping: uncapped epochs, patience=3 dev checks", flush=True)
    for epoch in count():
        rng.shuffle(train)
        for start in range(0,len(train),4):
            step+=1
            stats=trainer.step(train[start:start+4],step)
            trainer.log(dict(epoch=epoch+1,**stats))
            if step%25==0:print('epoch',epoch+1,'seen',min(start+4,len(train)),flush=True)
            if start+4<len(train) and start!=3700:continue
            report,*_=evaluate(replay,trainer,'dev',f'step_{step}',out)
            history.append(dict(step=step,epoch=epoch+1,report=report))
            if best_rl is None or key(report)>key(best_rl):
                best_rl=report
                best_step=step
                trainer.save('best')
                stale=0
            else:stale+=1
            atomic_json(out/'selection.json',dict(best_step=best_step,history=history,untrained_dev=best,
                patience=3,checks_without_improvement=stale,stopped_early=stale>=3))
            if stale>=3:break
        if stale>=3:break
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(trainer.model,load_file(str(out/'best/adapter_model.safetensors')))
    results={'A':summary}
    from contextlib import nullcontext
    for tag in ('B','C'):
        with trainer.model.disable_adapter() if tag=='B' else nullcontext():
            _,cal,confidence,orig=evaluate(replay,trainer,'calib',tag,out)
            tau=crc_threshold(confidence[orig==1],cal[orig==1]!=1,cfg.router.alpha)
            report,ranks,conf,orig=evaluate(replay,trainer,'test',tag,out)
        report['crc']=metrics(np.where(conf>=tau,ranks,orig))
        report['crc_threshold']=tau
        report['vs_A']=paired_bootstrap(ranks==1,A==1)
        if tag=='B':B=ranks
        else:report['vs_B']=paired_bootstrap(ranks==1,B==1)
        results[tag]=report
        atomic_json(out/'results.json',results)
    print(results,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('out',type=Path)
    main(parser.parse_args().out)
