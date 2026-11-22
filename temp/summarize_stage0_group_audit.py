"""Recount the stage-0 group metrics using reviewed query and candidate annotations."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "outputs/stage0_clip4clip_qwen3vl8b/pilot"
sys.path.insert(0, str(ROOT))

from cvr.utils import tag_query


# Original temporal cues and an expanded cue screen were reviewed in context.
TEMPORAL = [
    21, 37, 72, 83, 102, 105, 125, 133, 165, 181, 198, 202, 203, 225,
    272, 281, 285, 287, 295, 296, 304, 305, 318, 339, 362, 391, 428, 445,
    500, 517, 523, 564, 594, 647, 678, 687, 732, 737, 783, 794, 809, 822,
    832, 838, 840, 841, 850, 904, 921, 922, 959, 986, 991,
]
NEGATION = [196, 279, 369, 542, 809, 840]

# Include only an explicitly supported contrast in a comparable activity or setting.
# The caption screen covered the strongest non-ground-truth rival for each eligible query.
ACTION_CONTRASTS = {
    55: ("记者在靶场提问", "射手装填弹药"),
    145: ("卡通角色准备骑车", "卡通角色已经骑行，随后唱歌"),
    182: ("婴儿玩猫尾巴", "猫舔婴儿的脚"),
    296: ("车辆仍在燃烧时旁人谈话", "用灭火器灭火后，两人对镜头说话"),
    678: ("运动员从准备过渡到跑步", "运动员主要在起跑线前准备"),
    695: ("视角随赛车在赛道上行驶", "赛车游戏开场展示车辆和发车准备"),
    829: ("体操运动员练习翻滚", "运动员练习在杠上支撑和平衡"),
    866: ("踩冲浪板随浪滑行", "在水中翻转倒扣的风翼"),
    872: ("士兵向队伍讲话", "士兵之间近距离交谈"),
    904: ("一人说唱、另一人演奏吉他", "人声演唱者与吉他手一同表演"),
}


def main():
    with (PILOT / "test_yesno_k16.jsonl").open() as source:
        rows = {int(r["qid"].split("_")[1]): r for r in map(json.loads, source)}
    all_ids = set(rows)
    eligible = {i for i, row in rows.items() if row["gt_rank"] == 1}
    harmful = {i for i in eligible if rows[i]["new_rank"] != 1}
    helpful = {i for i in all_ids - eligible if rows[i]["new_rank"] == 1}
    assert (len(rows), len(eligible), len(harmful), len(helpful)) == (1000, 396, 101, 188)

    def stats(ids):
        correct = len(ids & eligible)
        flips = len(ids & harmful)
        return {"queries": len(ids), "base_top1_correct": correct, "harmful_flips": flips,
                "helpful_flips": len(ids & helpful), "hfr_pct": 100 * flips / correct if correct else None}

    original = {tag: {i for i, row in rows.items() if tag in tag_query(row["text"])}
                for tag in ("temporal", "negation", "count", "long", "plain")}
    temporal, negation, action = set(TEMPORAL), set(NEGATION), set(ACTION_CONTRASTS)
    assert temporal <= all_ids and negation <= all_ids and action <= eligible
    assert len(TEMPORAL) == len(temporal) and len(NEGATION) == len(negation)
    contrast_cases = []
    for i, (target_action, rival_action) in ACTION_CONTRASTS.items():
        row = rows[i]
        rival, _ = max(((v, s) for v, s in zip(row["cands"], row["yes_log_odds"]) if v != row["gt"]),
                       key=lambda item: item[1])
        contrast_cases.append({"qid": row["qid"], "query": row["text"], "reference": row["gt"], "rival": rival,
                               "target_action": target_action, "rival_action": rival_action, "harmful_flip": i in harmful,
                               "evidence": "captions and sampled frames" if i != 904 else "captions"})

    data = {
        "overall": stats(all_ids),
        "original_groups": {tag: stats(ids) for tag, ids in original.items()},
        "reviewed_query_groups": {tag: {**stats(ids), "qids": [rows[i]["qid"] for i in sorted(ids)]}
                                  for tag, ids in (("temporal", temporal), ("negation", negation))},
        "without_reviewed_temporal_cue": stats(all_ids - temporal),
        "action": {"confirmed_contrast": stats(action), "remaining_or_unconfirmed": stats(eligible - action),
                   "cases": contrast_cases},
        "eligible_with_any_confirmed_label": stats(eligible & (temporal | negation | action)),
        "method": {
            "temporal": "Explicit time cues linking distinct events or states; original and expanded cue matches reviewed in context.",
            "negation": "Explicit absence, denial or attempted avoidance; normalized spaced contractions were included.",
            "action": "Screened the top non-reference candidate by saved yes/no score for all 396 baseline-correct queries. "
                      "Confirmed concrete contrasts with dataset captions and inspected sampled frames for borderline cases. "
                      "Other cases include insufficient evidence; caption and frame evidence does not identify the model's causal failure mode.",
            "overlap": "Temporal, negation and action labels can overlap.",
        },
    }
    destination = PILOT / "group_audit.json"
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
