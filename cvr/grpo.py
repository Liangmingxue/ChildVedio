"""GRPO / DAPO-style RL for the counterfactual generator (method 2).

Policy  : small text LLM + LoRA (default Qwen2.5-3B-Instruct) on GPU 0.
Reward  : frozen VLM (vLLM server on GPU 1) runs counterfactual forced choice on the
          positive video and its hardest retrieved negatives;
          R = [p(Q|v+) - max_j p(Q|v-_j)] - lam * #valid_hypotheses, gated by NLI validity.
Tricks  : group-relative advantages, DAPO dynamic sampling (drop zero-variance groups),
          clip-higher, token-level loss aggregation, optional k3 KL to the frozen base.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import List, Sequence

import numpy as np

from .counterfactual import FRAGILE_TYPES, build_cf_messages, parse_hypotheses
from .router import LORA_TARGETS


def build_grpo_rows(router_rows: Sequence[dict], n_neg: int, max_gt_rank: int) -> List[dict]:
    rows = []
    for r in router_rows:
        if r.get("split", "train") != "train" or r["gt"] not in r["cands"][:max_gt_rank]:
            continue
        negs = [v for v in r["cands"] if v != r["gt"]][:n_neg]
        if not negs:
            continue
        fragile = r.get("label_fragile", "none")
        rows.append({
            "qid": r["qid"], "text": r["text"], "videos": [r["gt"]] + negs,
            "fragile": fragile if fragile in FRAGILE_TYPES and fragile != "none" else "auto",
        })
    return rows


class CFGRPOTrainer:
    def __init__(self, cfg, verifier, nli, out_dir: str):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup

        self.torch = torch
        self.cfg, self.g = cfg, cfg.grpo
        g = self.g
        self.verifier, self.nli, self.out_dir = verifier, nli, out_dir
        self.device = torch.device(g.device)
        self.tok = AutoTokenizer.from_pretrained(g.policy_model)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(g.policy_model, torch_dtype=torch.bfloat16).to(self.device)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        base.config.use_cache = False
        self.model = get_peft_model(base, LoraConfig(
            r=g.lora_r, lora_alpha=g.lora_alpha, lora_dropout=0.0,
            target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
        ))
        self.model.print_trainable_parameters()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=g.lr, betas=(0.9, 0.99), weight_decay=0.0)
        self.sched = get_constant_schedule_with_warmup(self.opt, g.warmup_steps)
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        eos = [e for e in eos + [self.tok.eos_token_id] if e is not None]
        self.eos_ids = torch.tensor(sorted(set(eos)), device=self.device)
        self.loop = asyncio.new_event_loop()
        os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ rollout
    def _prompt(self, row: dict) -> str:
        return self.tok.apply_chat_template(
            build_cf_messages(row["text"], row["fragile"], self.g.m), tokenize=False, add_generation_prompt=True
        )

    def rollout(self, rows: Sequence[dict]):
        torch = self.torch
        g = self.g
        enc = self.tok([self._prompt(r) for r in rows], return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                **enc, do_sample=True, temperature=g.temperature, top_p=g.top_p,
                max_new_tokens=g.max_new_tokens, num_return_sequences=g.group_size,
                pad_token_id=self.tok.pad_token_id, use_cache=True,
            )
        self.model.train()
        P = enc["input_ids"].shape[1]
        prompt_ids = enc["input_ids"].repeat_interleave(g.group_size, 0)
        prompt_mask = enc["attention_mask"].repeat_interleave(g.group_size, 0)
        comp = out[:, P:]
        is_eos = torch.isin(comp, self.eos_ids)
        has_eos = is_eos.any(1)
        first = torch.where(has_eos, is_eos.int().argmax(1), torch.full_like(has_eos, comp.size(1) - 1, dtype=torch.long))
        comp_mask = (torch.arange(comp.size(1), device=comp.device)[None, :] <= first[:, None]).long()
        texts = self.tok.batch_decode(comp * comp_mask + self.tok.pad_token_id * (1 - comp_mask), skip_special_tokens=True)
        return prompt_ids, prompt_mask, comp, comp_mask, texts

    # ------------------------------------------------------------------ reward
    def compute_rewards(self, rows: Sequence[dict], texts: Sequence[str], step: int):
        g = self.g
        G = g.group_size
        queries = [rows[i // G]["text"] for i in range(len(texts))]
        hyps = [parse_hypotheses(t, g.m_max, q) for t, q in zip(texts, queries)]
        valid = self.nli.filter_many(queries, hyps)

        async def score(i: int):
            if not valid[i]:
                return None
            row = rows[i // G]
            ps = await asyncio.gather(*[
                self.verifier.forced_choice(row["text"], valid[i], v, n_perm=1, seed=step) for v in row["videos"]
            ])
            return float(ps[0] - max(ps[1:]))

        async def run_all():
            return await asyncio.gather(*[score(i) for i in range(len(texts))])

        margins = self.loop.run_until_complete(run_all())
        rewards = [g.r_invalid if m is None else m - g.lam_count * len(v) for m, v in zip(margins, valid)]
        info = {
            "valid_rate": float(np.mean([len(v) > 0 for v in valid])),
            "mean_valid_hyps": float(np.mean([len(v) for v in valid])),
            "mean_margin": float(np.mean([m for m in margins if m is not None])) if any(m is not None for m in margins) else 0.0,
        }
        return np.asarray(rewards, dtype=np.float64), info

    # ------------------------------------------------------------------ update
    def _logps(self, prompt_ids, prompt_mask, comp, comp_mask):
        torch = self.torch
        ids = torch.cat([prompt_ids, comp], 1)
        att = torch.cat([prompt_mask, comp_mask], 1)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        logits = self.model(input_ids=ids, attention_mask=att, position_ids=pos).logits
        P = prompt_ids.size(1)
        logits = logits[:, P - 1:-1].float()
        return torch.log_softmax(logits, -1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)

    def step(self, rows: Sequence[dict], step: int) -> dict:
        torch = self.torch
        g = self.g
        prompt_ids, prompt_mask, comp, comp_mask, texts = self.rollout(rows)
        rewards, info = self.compute_rewards(rows, texts, step)
        R = rewards.reshape(len(rows), g.group_size)
        mean, std = R.mean(1, keepdims=True), R.std(1, keepdims=True)
        keep_group = std[:, 0] > 1e-6  # DAPO dynamic sampling
        adv = ((R - mean) / (std + 1e-4)).reshape(-1)
        sel = [i for i in range(len(texts)) if keep_group[i // g.group_size]]
        stats = {"step": step, "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std()),
                 "kept_groups": int(keep_group.sum()), **info}
        if not sel:
            stats["skipped"] = True
            return stats
        total_tokens = float(comp_mask[sel].sum().item())
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        updates = g.get("updates_per_rollout", 1)
        old_logps = {}
        clipped_tokens = 0.
        for update in range(updates):
            self.opt.zero_grad(set_to_none=True)
            loss_sum, kl_sum = 0.0, 0.0
            for j in range(0, len(sel), g.micro_batch):
                mb = torch.tensor(sel[j:j + g.micro_batch], device=self.device)
                mask = comp_mask[mb].float()
                logp = self._logps(prompt_ids[mb], prompt_mask[mb], comp[mb], comp_mask[mb])
                if update == 0:
                    old_logps[j] = logp.detach()
                old = old_logps[j]
                ratio = torch.exp(logp - old)
                clipped_tokens += float((((ratio < 1 - g.eps_low) | (ratio > 1 + g.eps_high)) * mask).sum().detach())
                a = adv_t[mb][:, None]
                pg = -torch.min(ratio * a, ratio.clamp(1 - g.eps_low, 1 + g.eps_high) * a)
                if g.beta_kl > 0:
                    with torch.no_grad(), self.model.disable_adapter():
                        ref = self._logps(prompt_ids[mb], prompt_mask[mb], comp[mb], comp_mask[mb])
                    d = ref - logp
                    kl = torch.exp(d) - d - 1
                    pg = pg + g.beta_kl * kl
                    kl_sum += float((kl * mask).sum().item())
                loss = (pg * mask).sum() / total_tokens
                loss.backward()
                loss_sum += float(loss.item())
            torch.nn.utils.clip_grad_norm_(self.params, g.max_grad_norm)
            self.opt.step()
            self.sched.step()
        stats["clip_fraction"] = clipped_tokens / (total_tokens * updates)
        stats["optimizer_updates"] = updates
        stats.update({"loss": loss_sum, "kl": kl_sum / max(total_tokens, 1.0), "tokens": total_tokens})
        return stats

    def save(self, name: str) -> str:
        path = os.path.join(self.out_dir, name)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
        return path

    def log(self, stats: dict) -> None:
        with open(os.path.join(self.out_dir, "log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
