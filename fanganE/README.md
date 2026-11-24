# fanganE：JEV-style 软分布蒸馏与候选证据排序

## 1. 目标

fanganE 在 fanganD 的 light→deep 级联和候选证据融合基础上，引入 JEV 风格的概率分布学习。它不是对 train-jev 的直接移植，也不使用 27B 模型或强化学习。

本方案复用当前仓库已有的：
- Qwen3-4B 路由主干与 LoRA 加载器：cvr/router.py
- 列表级特征融合、成本分配与评测工具：fanganD/lib.py
- 候选视频 PointerHead：fanganC/vendor/pointer.py

不会修改 fanganC 或 fanganD 的既有实现。所有运行产物写入 fanganE/runs/，该目录已加入忽略规则。

## 2. 模型结构

输入由三部分组成：

1. 文本证据：Query、Top-16 粗排分数、Top-5 light 验证分数。
2. 候选证据：冻结 CLIP4Clip 视频向量，以及 light-only / light+deep 两组数值特征。
3. 教师分布：缓存的 VLM light_scores 和 deep_scores。

共享 Qwen3-4B 主干输出最后一个有效 token 的隐状态。隐状态进入两个任务分支：

- 双列表排序头：分别产生 light-only 和 light+deep 的 Top-16 候选 logits；通过数值残差与 PointerHead 融合候选视频向量。
- 三类干预结果头：预测 harm、unchanged、fix。该头只接收文本、粗排和 light 证据，不读取 deep 分数，因此在推理时可以先做升级决策，再调用 deep。

最终升级优先级为：

    priority = P(fix) - P(harm)

在每个图库内部，先支付所有查询的 light 成本，再按 priority 从高到低分配 deep 增量成本，严格匹配预算。

## 3. JEV 风格蒸馏目标

### 3.1 排序分布蒸馏

对每条查询的实际验证候选，将 VLM 分数按 teacher_temperature 做 softmax：

    P_teacher = softmax(VLM_scores / T_teacher)

light 分支只在实际验证的 Top-5 上计算 KL；deep 分支只在实际验证的候选上计算 KL（当前对齐快照为 Top-16）。未验证候选不会被伪造为教师标签。

### 3.2 干预结果软目标

令 p_light(gt) 和 p_deep(gt) 为两个教师分布分配给 GT 候选的概率质量，定义：

    delta  = p_deep(gt) - p_light(gt)
    p_harm = max(-delta, 0)
    p_fix  = max( delta, 0)
    p_same = 1 - abs(delta)

三个分量构成和为 1 的软概率目标。另保留缓存 rank 产生的 harm / unchanged / fix 硬标签作为辅助监督。

### 3.3 总损失

    L = λ_rank_KL · KL_rank
      + λ_rank_CE · CE_listwise
      + λ_route_KL · KL_outcome
      + λ_route_CE · CE_outcome

其中 KL_rank 学习 VLM 对候选的相对概率，CE_listwise 保留 GT 排序锚点；KL_outcome 学习干预结果软分布，CE_outcome 约束实际 fix / harm 结果。

这不是直接复制 JEV 的 24 槽位决策头。fanganE 将其概率决策思想改造成适合 Top-16 视频排序与 light→deep 升级的双任务输出。

## 4. 数据划分与防泄漏

输入源默认使用：

    fanganC/runs/aligned-full-20261122

代码会校验 snapshot/manifest.json 中冻结文件的 SHA-256，并校验 CLIP 特征归档的哈希。

九个互不重叠的 1,000-query gallery 按如下方式使用：

| Gallery | 用途 | fanganE 是否使用 |
|---|---|---|
| 0–3 | 模型训练 | 是 |
| 4–5 | 决策拟合预留 | 否，保持未使用 |
| 6 | 开发集选 checkpoint | 是 |
| 7 | 路由温度校准 | 是 |
| 8 | 最终 holdout | 仅训练结束后评估 |

模型选择只依据 gallery 6 的复合蒸馏损失，不按 R@1 选 checkpoint。路由温度只在 gallery 7 拟合。gallery 8 只在最终阶段评估一次。

本轮不读取 snapshot/test.jsonl，不对原始 JSFusion 测试集调参，也不发起新的 VLM 请求。gallery 8 是训练侧独立图库 holdout；正式论文测试仍应在模型和超参数冻结后单独执行一次。

## 5. 双 RTX 4090 训练设置

建议先运行 frozen，再单独运行 LoRA：

| 模式 | Qwen 主干 | 训练参数 | 用途 |
|---|---|---|---|
| frozen | 冻结 Qwen3-4B | 只训练排序头和结果头 | 首要基线 |
| lora | Qwen3-4B + LoRA r=8 | 头部 + LoRA，低学习率 | 后续消融 |

默认设置：
- 物理 GPU 0 用于 Qwen 训练；GPU 1 保留给现有 VLM 服务。
- BF16，最大序列长度 512。
- micro-batch 2，梯度累积 4。
- LoRA rank 8，LoRA learning rate 5e-6。
- 排序 / 决策头 learning rate 1e-4。
- 先冻结主干，确认收益后再运行 LoRA 版本。
- 训练前脚本要求所选 GPU 至少有 16 GiB 空闲显存。该阈值是启动保护，不代表已实测峰值显存。

两张 24GB 卡的显存不会自动合并为一个 48GB 显存池。本实现默认单卡训练，避免引入不必要的 DDP/FSDP 复杂度。

## 6. 运行

从仓库根目录执行。首先运行冻结主干版本：

    MODE=frozen bash fanganE/run_train.sh

完成后，再以独立输出目录运行 LoRA 消融：

    MODE=lora bash fanganE/run_train.sh

如 Python 环境不在默认位置，可指定：

    PYTHON_BIN=/path/to/python MODE=frozen bash fanganE/run_train.sh

运行脚本默认使用物理 GPU 0，并将其映射为进程内的 cuda:0。GPU_INDEX 可覆盖物理卡编号；如果 GPU 1 正在运行 VLM，不要将训练任务切换到 GPU 1。

也可以直接运行模块，但需要先创建输出父目录：

    mkdir -p fanganE/runs
    python -m fanganE --config fanganE/config.yaml --mode frozen --device cuda:0 --out fanganE/runs/frozen-manual

输出包括：
- protocol.json：数据划分、配置、代码哈希、GPU 与耗时信息。
- history.json / train.log：逐轮训练和开发集记录。
- best_head.pt：最佳任务头。
- best_adapter/：LoRA 模式下的最佳 adapter。
- predictions.npz：holdout 排序分数、路由概率和预测。
- results.json：gallery 8 与 D0 light-confidence cascade 的同预算比较。
- d0_models.npz：用于复现的初始列表融合器参数。

## 7. 验证

不加载大模型的单元测试：

    python -m unittest discover -s fanganE/tests -v

语法检查：

    python -m compileall -q fanganE

首次正式训练前，先完成单元测试，并确认本机存在对齐快照、CLIP 特征归档和对应的 Qwen3-4B 本地模型。

## 8. 结果解释边界

fanganE 是新的训练实现，不预设它一定超过 fanganD。核心比较是同一 gallery 8、相同调用预算下，fanganE 是否优于 D0 light-confidence cascade；同时检查 R@1、R@5、R@10、fixes、harms、VLM calls 和校准质量。

如果 frozen 有收益而 LoRA 没有收益，应保留 frozen 版本，不应仅因为 LoRA 更复杂就选用它。

## 9. 冻结主干的排序蒸馏对照（2026-11-24）

三组使用相同的 Qwen3-4B 冻结主干、初始化种子、两轮训练和路由监督，按开发集复合损失均选中第 1 轮。在固定 D0 升级名单下比较排序，并单独报告各自使用学习升级头后的端到端结果。开发和留出图库各有 1000 条查询。

| 排序 KL | 开发 R@1（固定路由） | 留出 R@1（固定路由） | 留出纠正 / 误伤（相对粗排，固定路由） | 留出端到端 R@1 |
|---|---:|---:|---:|---:|
| 全量 | 53.2% | 52.9% | 128 / 105 | 51.2% |
| 关闭 | 59.1% | 57.8% | 127 / 55 | 57.4% |
| 仅教师首选为 GT 的路径 | 52.4% | 51.9% | 132 / 119 | 50.8% |

同一留出图库的 D0 基线为 57.8%。关闭排序 KL 相对全量蒸馏提高 4.9 个百分点，逐查询配对 bootstrap 95% 区间为 [+2.8, +7.1]。本轮结果支持暂停当前排序蒸馏方案。路由软标签的独立贡献仍待验证。

复现实验时使用 `MODE=frozen bash fanganE/run_train.sh --epochs 2`。关闭排序 KL 时追加 `--rank-kd-weight 0`。选择性蒸馏保留原排序 KL 权重，在配置副本中设置 `rank_kd_correct_only: true`，并用 `--config` 指向该副本。
