# fanganC 候选证据路由实验

独立实现方案 C：保留 B 的条件收益与风险定义，引入冻结 CLIP4Clip 候选内容，先训练预测头，再联合训练 Qwen3-4B LoRA。所有新增文件和运行产物都保存在本目录。

## 全量对齐实验

- 读取完整的 9,000 条训练侧标注来保留 train/dev/calib 划分。用训练视频组成 9 个互不重叠的 1,000 视频图库，在各自图库内重算排名，并用测试侧的 `auto` 协议生成反事实和深验证标签。仅复用查询、视频和验证提示一致的旧评分。
- 候选版保留冻结 CLIP4Clip 的有序帧特征，用 GRU 编码帧序及其和查询的相关度，与现有池化候选分支一起预测收益和风险。
- 在原条件 BCE 上加预算分配损失。每个有效训练批次使用分差路由的预算，两次调用同一个精确分配器构造损失，分别使用真实收益和加上真实收益差的预测分数。测试仍使用原测试快照的分差预算。
- 联合训练每约半轮检查一次开发集。新最佳按原风险约束和 R@1 规则保存匹配的 LoRA 与预测头。连续 3 次验证无改善结束训练，联合训练上限为 8 轮，LoRA 学习率为 `2e-5`。

```bash
nohup bash fanganC/run_aligned_background.sh fanganC/runs/aligned-full \
  > /tmp/fanganC-aligned.log 2>&1 < /dev/null &
```

后台脚本在物理 GPU 0 顺序运行反事实生成、VLM 打分、特征提取和训练。对齐打分按已完成批次续跑，复用同一个运行目录即可。`alignment/` 保存新图库与标签，`training.json` 中 `validations` 保存每次开发集检查。

## 首轮设置

- 原 3000 条三路标注加当前已完成的扩充结果，启动时保存固定快照。新增结果只加入 train，保留原 dev/calib。
- 保留现有图库、候选和反事实标签；不重新调用 VLM，不改变后台标注流程。原训练与测试生成协议差异仍存在，本轮只检验输入与优化方法。
- 顺序比较 `text_only` 与 `candidate`，二者使用相同数值分支、优化方式和开发集选择规则。`text_only` 是新实现的无候选对照，不是原 B 的逐行复刻。
- 头预热 1 轮，学习率 3e-4；联合训练 3 轮，LoRA 2e-5、头 1e-4。有效 batch 16、microbatch 4，种子 42，关闭 fragile 辅助损失。
- dev 选择 checkpoint 与风险权重 1 或 2；两个版本均完成选择后才测试。calib 只报告概率质量，不调参。
- 测试为现有结果的离线回放，精确匹配分差基线 27824 次 VLM 调用。缓存中的路径成本也用于回放分配，并不是未知验证成本的在线部署实现。

## 使用

在项目根目录使用已有环境，无需安装或修改全局依赖：

```bash
PYTHONPATH=data/models/router-python-deps:data/msrvtt/source/python-deps \
  /home/omnisky/miniconda3/envs/chart-vllm/bin/python -B -m unittest discover -s fanganC/tests -v

PYTHONPATH=data/models/router-python-deps:data/msrvtt/source/python-deps \
  /home/omnisky/miniconda3/envs/chart-vllm/bin/python -B -m fanganC smoke \
  --run-dir fanganC/runs/smoke --device cuda:0

nohup bash fanganC/run_background.sh fanganC/runs/first-run \
  > /tmp/fanganC-launch.log 2>&1 < /dev/null &
```

后台脚本固定物理 GPU 0，启动前检查占用并使用本实验的互斥锁。GPU 1 的标注服务不受影响。

`python -m fanganC --help` 提供 `snapshot`、`features`、`train`、`evaluate`、`run`、`smoke`。所有命令需要 `--run-dir`；`snapshot` 和 `run` 可用 `--config` 指定实验配置；训练和评估支持 `--variant`。运行目录必须位于 `fanganC/` 内。

分步示例：

```bash
python -m fanganC snapshot --run-dir fanganC/runs/example
python -m fanganC features --run-dir fanganC/runs/example --device cuda:0
python -m fanganC train --run-dir fanganC/runs/example --device cuda:0
python -m fanganC evaluate --run-dir fanganC/runs/example --device cuda:0
```

首个快照确定后，后台新增标注不会进入当前实验。使用新的运行目录才能扩充数据。原文件哈希、split、配置与代码指纹保存在快照中；改动实现或输入后应创建新实验。

失败后可以重新运行同一命令：已完成的特征块保留并复用，完整训练版本会被跳过；中断的训练版本从初始化重新开始，不从半轮优化器状态恢复。不要同时在同一运行目录执行两个命令。

## 结果位置

- `snapshot/manifest.json`：固定数据来源、哈希和 split 数量。
- `features/*_manifest.json`：特征指纹、耗时与检索分数重建误差；视频和文本块可重用。
- 每个版本的 `training.json`、`training_complete.json`：开发集选择过程、训练耗时和显存。
- 每个版本的 `evaluation.json`、`test_predictions.jsonl`：指标、概率质量、逐查询动作与相对分差基线的变化。
- `results.json`：两个版本的最终报告；`status.json`、`workflow.log`、`workflow.pid`、`exit_code`：运行状态。

路由耗时不包含离线视频、文本特征提取，提取耗时单独记录。相同 VLM 调用不代表总计算成本相同。是否优于基线由实际报告决定，不保证提升。

## 复用来源

项目原有 JEV 构建、B 的条件收益头、动态规划、指标和数据校验通过 `common.py` 适配层复用。视频加载与编码复用 `temp/stage0_clip4clip.py` 及其 CLIP4Clip 依赖；复杂解码和模型结构没有重写。

`vendor/pointer.py` 摘取自本地 VideoJev 的 `timejev/model.py`，仅移除了无关主干依赖。许可证和 NOTICE 随文件保留。CLIP4Clip 的 MIT 许可证也保存在 `vendor/`。

功能验证完成后，独立审查结论记录在 `REVIEW.md`。
