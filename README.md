# ChildVedio · 风险可控的反事实验证文本-视频检索智能体

> Trust or Falsify: Risk-Controlled Intervention Routing and RL-Optimized Counterfactual Verification for Agentic Text-to-Video Retrieval

## 问题：智能体检索中的“不可靠干预”

VLM 智能体在粗排结果上做干预（重排）时有两种失效：

1. **该信时不信**：粗排 Top-1 本来正确，被 VLM 翻错（有害翻转率 HFR），而且白白花了算力。
2. **该疑时被骗**：遇到组合式近似负样本（动作替换、时序颠倒、否定、计数等），正向 yes/no 验证会误判为“匹配”。

## 方法

| 模块 | 作用 | 代码 |
|---|---|---|
| 方法一：Jev 风格校准路由 | 一次前向同时输出 `action ∈ {accept, light, deep}` 和 `fragile` 约束类型；温度校准；用 Conformal Risk Control 选采纳阈值 τ | `cvr/router.py`, `cvr/calibration.py` |
| 方法二：反事实强制多选验证 | 沿 fragile 方向生成最小编辑反事实 → NLI 过滤 → 冻结 VLM 在 {Q, h⁻…, 都不是} 上做受限 softmax，取 p(Q\|v) | `cvr/counterfactual.py`, `cvr/vlm.py` |
| 方法二的 RL | GRPO + DAPO（dynamic sampling、clip-higher、token 级损失），只训练 3B 文本生成器，VLM 冻结作为奖励 | `cvr/grpo.py` |

## 硬件规划（2 × RTX 4090 24GB）

| 阶段 | GPU0 | GPU1 | 估计显存 |
|---|---|---|---|
| 01–02 抽帧 / CLIP 粗排 | CLIP-L | — | < 6 GB |
| 03–05 预实验 / 构造路由数据 | 文本 LLM 服务 Qwen2.5-7B + NLI | VLM 服务 Qwen2.5-VL-7B | 约 20 GB / 约 22 GB |
| 06–08 路由训练 / 校准 / 推理 | Qwen3-4B + LoRA（梯度检查点） | 空闲 | 约 14–18 GB |
| 09 GRPO | 策略 Qwen2.5-3B + LoRA + NLI | VLM 服务 | 约 16–20 GB / 约 22 GB |
| 10 端到端评测 | cfgen 文本服务 + NLI | VLM 服务 | 同 03–05 |

不在本方案内（4090 不现实）：全参数微调 VLM、在 7B 视频 VLM 上做 Vision-GRPO。

## 运行顺序

```bash
pip install -r requirements.txt

# 0. 数据（MSR-VTT，CLIP4Clip 格式标注）
python scripts/00_prepare_msrvtt.py --anno MSRVTT_data.json \
  --train_csv MSRVTT_train.9k.csv --test_csv MSRVTT_JSFUSION_test.csv --video_dir /path/to/videos
python scripts/01_extract_frames.py --workers 16
python scripts/02_retrieve.py --split test
python scripts/02_retrieve.py --split train

# 1. 启动服务（两个终端）
bash scripts/serve_vlm.sh          # GPU1, :8000
bash scripts/serve_llm.sh          # GPU0, :8001

# 2. 预实验（决定是否继续）+ 免训练对比
python scripts/03_pilot_yesno.py --split test --k 10
python scripts/04_cf_zeroshot.py --split test --n 500

# 3. 路由：构造数据 → 停掉 GPU0 的文本服务 → 训练 / 校准 / 推理
python scripts/05_build_router_data.py
python scripts/06_train_router.py
python scripts/07_calibrate_router.py
python scripts/08_route.py --split test

# 4. RL 训练反事实生成器（GPU0 训练，GPU1 保持 VLM 服务）
python scripts/09_train_cf_grpo.py

# 5. 端到端评测
bash scripts/serve_llm.sh                                    # prompt-only 生成器
python scripts/10_run_pipeline.py --mode retriever
python scripts/10_run_pipeline.py --mode light
python scripts/10_run_pipeline.py --mode deep --tag prompt
python scripts/10_run_pipeline.py --mode router --tag prompt
MODEL=Qwen/Qwen2.5-3B-Instruct LORA=outputs/cf_grpo/final bash scripts/serve_llm.sh
python scripts/10_run_pipeline.py --mode router --tag grpo --set cfgen.model=cfgen

# 6. 画图
python scripts/11_plot.py pareto
python scripts/11_plot.py reliability
```

所有超参数在 `configs/default.yaml`，可用 `--set key.sub=value` 覆盖。

## 数据格式（接入 DiDeMo / ActivityNet 只需写同格式转换脚本）

- `videos.jsonl`: `{"video_id", "path", "split"}`
- `captions.json`: `{video_id: [caption, ...]}`（训练集用于 fragile 标签）
- `queries_{split}.jsonl`: `{"qid", "text", "video_id"}`
- 更强的粗排模型：导出 `(n_query, n_gallery)` 相似度矩阵，用 `02_retrieve.py --sim_npy` 接入。

## 关键指标

R@1/5/10、MdR、MnR；HFR（有害翻转率）、HelpR（有益翻转率）；每查询 VLM 调用次数和耗时；路由 ECE / Brier / 可靠性图；按查询类型拆分。

## 说明

代码未经运行测试，只做过 Python 语法检查。第一次跑请先用 `--n 50` 之类的小规模参数检查路径和服务是否正常。
