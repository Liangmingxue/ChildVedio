"""Fit post-ranking flip and pre-deep upgrade decisions on out-of-ranker-fit galleries."""

import numpy as np
import torch
from torch.nn import functional as F

from cvr.calibration import crc_threshold
from fanganC.common import atomic_json, read_json
from fanganD.lib import metrics, paired_bootstrap
from fanganD.gallery_rankers import routes


def outcome_score(features, labels, fit, path):
    """Ridge multinomial regression for harm/neutral/fix, returning expected gain in [0,1]."""
    z = torch.as_tensor(features, dtype=torch.float64)
    mean = z[fit].mean(0)
    scale = z[fit].std(0, unbiased=False).clamp_min(1e-6)
    z = (z - mean) / scale
    train_x = z[fit]
    train_y = torch.as_tensor(labels[fit] + 1, dtype=torch.long)
    weight = torch.zeros((z.shape[1], 3), dtype=z.dtype, requires_grad=True)
    bias = torch.zeros(3, dtype=z.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(train_x @ weight + bias, train_y) + 0.005 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        prob = (z @ weight + bias).softmax(1).numpy()
        np.savez(path, mean=mean.numpy(), scale=scale.numpy(), weight=weight.numpy(), bias=bias.numpy())
    return (1 + prob[:, 2] - prob[:, 0]) / 2


def confidence(scores):
    p = torch.from_numpy(scores).softmax(-1).topk(2).values.numpy()
    return p[:, 0] - p[:, 1]


def raw_ranks(scores, gt, original):
    order = np.argsort(-scores, axis=1, kind='stable')
    return np.where(gt >= 0, (order == gt[:, None]).argmax(1) + 1, original)


def summarize(rank, original, chosen, costs, mask):
    r, orig = rank[mask], original[mask]
    hits = r == 1
    harms = int(((orig == 1) & ~hits).sum())
    return dict(n=len(r), hits=int(hits.sum()), **metrics(r),
                fixes=int(((orig != 1) & hits).sum()), harms=harms,
                harmful_fraction=harms/int((orig == 1).sum()),
                calls=int(costs[mask, 1].sum() + costs[mask & chosen, 2].sum()),
                upgrades=int(chosen[mask].sum()))


def evidence_arrays(rows):
    return (np.stack([r['scores'] for r in rows]), np.stack([r['light_scores'] for r in rows]),
            np.stack([r['deep_scores'] for r in rows]))


def flip_features(rows, logits, chosen):
    coarse, light, deep = evidence_arrays(rows)
    top = logits.argmax(1)
    ix = np.arange(len(rows))
    has_light = top < light.shape[1]
    ir = np.minimum(top, light.shape[1] - 1)
    return np.column_stack([
        confidence(logits), logits[ix, top] - logits[:, 0], coarse[:, 0] - coarse[ix, top],
        coarse[:, 0] - coarse[:, 1], top, has_light, np.where(has_light, light[ix, ir] - light[:, 0], 0),
        light.argmax(1) == top, chosen, np.where(chosen, deep[ix, top] - deep[:, 0], 0),
        chosen & (deep.argmax(1) == top),
    ])


def fit_crc(out, rows, inputs, scores, chosen, prefix):
    gallery, costs, original = (inputs[k] for k in ('gallery','costs','original'))
    logits = scores[np.arange(len(rows)), chosen.astype(int)]
    ranks = raw_ranks(logits, inputs['gt'], original)
    changed = logits.argmax(1) != 0
    labels = (ranks == 1).astype(int) - (original == 1)
    fit = np.isin(gallery, [4, 5]) & changed
    good_cal = (gallery == 7) & (original == 1)
    gain = outcome_score(flip_features(rows, logits, chosen), labels, fit, out/f'{prefix}_flip_policy.npz')
    alpha = read_json(out/'protocol.json')['alpha']
    result = dict(fit_changed=int(fit.sum()), class_counts=np.bincount(labels[fit]+1, minlength=3).tolist())
    gated = {'raw': ranks}
    for name, signal in [('gap', confidence(logits)), ('learned', gain)]:
        tau = crc_threshold(signal[good_cal], ranks[good_cal] != 1, alpha)
        accept = signal >= tau
        if name == 'learned':
            accept |= ~changed
        gated[name] = np.where(accept, ranks, original)
        result[name] = dict(threshold=tau, calibration_bound=float(
            (1 + ((gated[name] != 1) & good_cal).sum()) / (1 + good_cal.sum())))
    result['splits'] = {}
    for g, split in [(6, 'development'), (7, 'calibration')]:
        result['splits'][split] = {k:summarize(r, original, chosen, costs, gallery == g) for k,r in gated.items()}
    np.savez(out/f'{prefix}_ranks.npz', chosen=chosen, **gated)
    atomic_json(out/f'{prefix}_selection.json', result)
    return gated


def crc_experiment(out, rows):
    inputs = np.load(out/'inputs.npz')
    preds = np.load(out/'D3.npz')
    chosen = routes(preds['gate'], inputs['costs'], inputs['gallery'])
    fit_crc(out, rows, inputs, preds['scores'], chosen, 'crc')
    print('CRC calibrated on gallery 7', flush=True)


def upgrade_features(rows, light_logits):
    # Strictly pre-deep inputs. The cost ledger is used only by the common budget allocator.
    coarse = np.stack([r['scores'] for r in rows])
    light = np.stack([r['light_scores'] for r in rows])
    top = light_logits.argmax(1)
    p = torch.from_numpy(light_logits).softmax(1).numpy()
    return np.column_stack([confidence(light_logits), p[:,0], p.max(1), top,
                            coarse[:,0]-coarse[:,1], coarse[:,0]-coarse[:,-1], light[:,0], light.max(1),
                            light.max(1)-light[:,0], light.argmax(1) == top])


def finish(out, rows):
    inp = np.load(out/'inputs.npz')
    gt, gallery, costs, original, fixed = (inp[k] for k in ('gt','gallery','costs','original','fixed'))
    dev = gallery == 6
    hold = gallery == 8
    ix = np.arange(len(rows))
    histories = read_json(out/'history.json')
    scores = {k:np.load(out/f'{k}.npz')['scores'] for k in ('warm','A','B','C')}
    fixed_ranks = {k:raw_ranks(v[ix, fixed.astype(int)], gt, original) for k,v in scores.items()}
    # Choose before opening held-out metrics. Break exact ties by development CE then simpler arm.
    selected = max(('A','B','C'), key=lambda k: (
        (fixed_ranks[k][dev] == 1).sum(), -min(e['dev_ce'] for e in histories[k])))
    atomic_json(out/'selected_ranker.json', dict(selected=selected, development={
        k:summarize(r, original, fixed, costs, dev) for k,r in fixed_ranks.items()}))
    logits = scores[selected]
    wins = logits.argmax(-1) == gt[:, None]
    target = wins[:, 1].astype(int) - wins[:, 0]
    policy_fit = np.isin(gallery, [4, 5])
    features = upgrade_features(rows, logits[:, 0])
    priority = outcome_score(features, target, policy_fit, out/'upgrade_policy.npz')
    policies = {'fixed_D0':fixed, 'selected_light_gap':routes(-confidence(logits[:, 0]), costs, gallery),
                'learned':routes(priority, costs, gallery)}
    routed = {k:raw_ranks(logits[ix, chosen.astype(int)], gt, original) for k,chosen in policies.items()}
    # The flip model also needs routes out of sample from the learned upgrade model.
    stacked_routes = policies['learned'].copy()
    for g in (4, 5):
        eval_fold = gallery == g
        cross_priority = outcome_score(features, target, policy_fit & ~eval_fold, out/f'upgrade_excluding_{g}.npz')
        stacked_routes[eval_fold] = routes(cross_priority[eval_fold], costs[eval_fold], gallery[eval_fold])
    end_to_end = fit_crc(out, rows, inp, logits, stacked_routes, 'combined_crc')
    crc = np.load(out/'crc_ranks.npz')
    report = dict(selected=selected, fixed_route={}, upgrade={}, crc={}, paired={},
                  upgrade_train_labels=np.bincount(target[policy_fit]+1, minlength=3).tolist())
    for k,r in fixed_ranks.items():
        report['fixed_route'][k] = summarize(r, original, fixed, costs, hold)
    for k,r in routed.items():
        report['upgrade'][k] = summarize(r, original, policies[k], costs, hold)
    for k in ('raw','gap','learned'):
        report['crc'][k] = summarize(crc[k], original, crc['chosen'], costs, hold)
    report['combined'] = {k:summarize(r, original, stacked_routes, costs, hold) for k,r in end_to_end.items()}
    for name, new, old in [('combined_vs_original_crc',end_to_end['learned'],crc['gap']),
                           ('B_vs_A',fixed_ranks['B'],fixed_ranks['A']),
                           ('C_vs_B',fixed_ranks['C'],fixed_ranks['B']),
                           ('upgrade_vs_fixed',routed['learned'],routed['fixed_D0']),
                           ('upgrade_vs_light_gap',routed['learned'],routed['selected_light_gap']),
                           ('crc_vs_gap',crc['learned'],crc['gap'])]:
        report['paired'][name] = paired_bootstrap(new[hold] == 1, old[hold] == 1)
    for section in ('fixed_route','upgrade','crc','combined'):
        if any(r['calls'] > read_json(out/'protocol.json')['budget'] for r in report[section].values()):
            raise ValueError('Exceeded evaluation budget')
    np.savez(out/'holdout_predictions.npz', qids=inp['qids'][hold], original=original[hold],
             **{f'fixed_{k}':v[hold] for k,v in fixed_ranks.items()},
             **{f'upgrade_{k}':v[hold] for k,v in routed.items()},
             **{f'combined_{k}':r[hold] for k,r in end_to_end.items()},
             **{f'crc_{k}':crc[k][hold] for k in ('raw','gap','learned')})
    atomic_json(out/'results.json', report)
    print('Completed gallery 8 evaluation', out/'results.json', flush=True)
