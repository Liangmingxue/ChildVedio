"""Fine-tune a frozen-run candidate checkpoint for equal-cost routing swaps."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cvr.router import JevFormatter, load_router
from cvr.utils import set_seed
from fanganC.common import atomic_json, legacy, read_json, resolve
from fanganC.features import FeatureStore, encode
from fanganC.training import (checkpoint_hashes, experiment_signature, new_head, optimizer_groups,
                              predict, publish_best, replay, training_logits)


def exchanges(reference, costs, scores, fraction, threshold):
    pairs = []
    for cost in np.unique(costs):
        donors = np.flatnonzero((costs == cost) & (reference == 2))
        receivers = np.flatnonzero((costs == cost) & (reference == 0))
        donors = donors[np.argsort(scores[donors], kind="stable")]
        receivers = receivers[np.argsort(-scores[receivers], kind="stable")]
        for a, b in zip(donors, receivers):
            gap = scores[b] - scores[a]
            if gap <= threshold:
                break
            pairs.append((float(gap), int(a), int(b)))
    pairs.sort(key=lambda x: -x[0])
    return pairs[:round(len(reference) * fraction)]


def measure(data, ref, scores, fraction, threshold):
    actions = ref.copy()
    pairs = exchanges(ref, data.calls[:, 2], scores, fraction, threshold)
    outcomes = []
    for _, a, b in pairs:
        actions[a], actions[b] = 0, 2
        outcomes.append(sum(int(data.ranks[i, actions[i]] == 1) - int(data.ranks[i, ref[i]] == 1)
                            for i in (a, b)))
    metrics = data.metrics(actions)
    assert metrics["vlm_calls"] == data.metrics(ref)["vlm_calls"]
    return dict(fraction=fraction, threshold=threshold, pairs=len(pairs), wins=sum(x > 0 for x in outcomes),
                losses=sum(x < 0 for x in outcomes), neutral=outcomes.count(0), metrics=metrics)


def run(source, out):
    device = 'cuda:0'
    store = FeatureStore(source)
    cfg = store.config
    set_seed(cfg.seed)
    folder = source / 'candidate'
    old = read_json(folder / 'training_complete.json')
    if (old['signature'] != experiment_signature(store)
            or checkpoint_hashes(folder / 'best', 'joint') != old['checkpoint_sha256']):
        raise ValueError("Prior checkpoint provenance mismatch")
    head = new_head(store, 'candidate').to(device)
    base_model = str(resolve(store.cfg.router.base_model))

    def load_weights(path, train):
        head.load_state_dict(torch.load(path / 'head.pt', map_location=device, weights_only=True))
        tok, model = load_router(base_model, adapter=str(path / 'adapter'), device=device, train=train)
        model.config.use_cache = False
        return JevFormatter(tok, max_len=store.cfg.router.max_len), model

    formatter, model = load_weights(folder / 'best', True)
    encoded = encode(store.rows, store.cfg, formatter)

    def score(indices):
        logits, _ = predict(store, head, indices, device, model, formatter, encoded)
        prob = head.probabilities(torch.as_tensor(logits)).numpy()
        return prob[:, 1, 1] - prob[:, 1, 2]

    dev = replay(store, store.dev_indices)
    ref = dev.margin_actions(cfg.deep_fraction)
    baseline = dev.metrics(ref)

    def validate():
        scores = score(store.dev_indices)
        trials = [measure(dev, ref, scores, f, t) for f in (0., .01, .025, .05)
                  for t in ((0.,) if not f else (0., .1, .2))]
        best = max(trials, key=lambda x: legacy().selection_key(x['metrics'], baseline))
        return best, trials

    best, trials = validate()
    atomic_json(out / 'initial_dev.json', dict(baseline=baseline, policies=trials))
    publish_best(out, head, model)
    best_step = 0
    stale = step = 0
    train = replay(store, store.train_indices)
    train_ref = train.margin_actions(cfg.deep_fraction)
    optimizer = torch.optim.AdamW(optimizer_groups(head, model, cfg, 'joint'))
    rng = np.random.default_rng(cfg.seed)
    history = []
    for epoch in range(cfg.joint_epochs):
        blocks = np.array_split(rng.permutation(len(train_ref)), int(np.ceil(len(train_ref)/cfg.effective_batch)))
        for index, block in enumerate(blocks, 1):
            head.train()
            model.train()
            ids = np.asarray(store.train_indices)[block]
            logits = torch.cat([training_logits(head, store, part, device, model, formatter, encoded,
                                                checkpointed=True) for part in
                                np.array_split(ids, int(np.ceil(len(ids)/cfg.microbatch)))])
            prob = head.probabilities(logits)
            gain = prob[:, 1, 1] - prob[:, 1, 2]
            correct = store.correct[ids].astype(int)
            target = correct[:, 2] - correct[:, 0]
            diff = target[:, None] - target
            costs = train.calls[block, 2]
            valid = ((diff > 0) & (costs[:, None] == costs)
                     & (train_ref[block, None] != train_ref[block]))
            i, j = np.where(valid)
            loss = head.loss(logits, torch.as_tensor(store.correct[ids], device=device))
            if len(i):
                loss = loss + F.softplus(-(gain[i] - gain[j])).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']], 1.,
                                           error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if index not in ((len(blocks)+1)//2, len(blocks)):
                continue
            current, trials = validate()
            history.append(dict(step=step, policies=trials))
            if legacy().selection_key(current['metrics'], baseline) > legacy().selection_key(best['metrics'], baseline):
                best, best_step, stale = current, step, 0
                publish_best(out, head, model)
            else:
                stale += 1
            atomic_json(out / 'development.json', dict(best=best, best_step=best_step, history=history))
            if stale >= 3:
                break
        if stale >= 3:
            break
    del optimizer, model, logits, prob, gain, loss
    torch.cuda.empty_cache()
    formatter, model = load_weights(out / 'best', False)
    test = replay(store, store.test_indices)
    budget = read_json(source / 'snapshot/budget.json')
    quotas = {int(k): v for k, v in budget['deep_query_count_by_vlm_calls'].items()}
    ref = test.margin_actions(cfg.deep_fraction, quotas)
    result = measure(test, ref, score(store.test_indices), best['fraction'], best['threshold'])
    atomic_json(out / 'results.json', dict(best_step=best_step, dev=best, test=result, baseline=test.metrics(ref)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
