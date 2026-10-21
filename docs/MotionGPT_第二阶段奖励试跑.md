# MotionGPT 第二阶段：小规模奖励微调试跑

本次先完成五条留出测试条件的 MotionCritic 基线，再启动一轮功能试跑。原独立样例 `004822` 保留，另四条从已保存的 320 条生成结果中按 seed=1234 无放回抽取；选择在评分前确定，不按分数筛选。五条合计不是 320 条的总体感知均分。

`tools/build_motioncritic_panel.py` 在五条都评分成功、转换协议一致时才生成 `results/motiongpt_stage1/critic_panel_five.json`。该文件固定文本、样本 ID、原生成结果与参考动作路径，供后续微调前后对照。它不含可用于训练的样本。训练代码排除这些 ID、镜像 ID，并排除整个测试集和验证集对应的原动作 ID。

用户已选择运行小规模试跑，配置为 `configs/motiongpt_stage2/pilot.yaml`，入口为：

```powershell
.\work\miniconda3\envs\motiongpt\python.exe tools/build_motioncritic_panel.py
.\work\miniconda3\envs\motiongpt\python.exe tools/run_motiongpt_reward_pilot.py
```

训练选择一个 train split 的文本条件，从冻结的 MotionGPT 采样两段可评分候选。保留完整语言 token 序列与采样策略的旧对数概率；采样使用 temperature=1、top_k=0、top_p=1。过短或无效候选不送入拟合，拒绝记录公开保存。这个“可评分候选”限制意味着试跑不代表全部输出分布。

两段候选沿用第一阶段 XYZ → SMPL Adam 拟合与 60 帧相对位移窗口协议，由冻结的 MotionCritic 给出动作级奖励。组内减去两候选的均值，得到各自优势，优势共享给该动作的有效语言 token。优化目标是逐 token 的截断重要性比率策略损失，加上相对于冻结原模型的精确类别 KL；先对每条动作的 token 取均值，再对两动作取均值。没有学习价值网络，不将它称为完整 PPO 复现。

这个机制通过动作 token 的对数概率更新生成策略，不对不可微的 SMPL 拟合过程反向传播。奖励来自训练动作，五条测试分数不进入目标、超参选择或训练停止条件。

为适配 6GB 显存，仅在 T5 decoder 自注意力的 q/v 线性层加入 rank=4、alpha=8 的 LoRA；原始 T5、VQ-VAE 和 MotionCritic 权重冻结。B 矩阵从零开始，因此注入适配器时输出保持不变。使用两个 AdamW 更新，学习率 1e-4，策略比率截断范围 0.2，KL 权重 0.02，梯度范数上限 1。保持 dropout 关闭，以便旧策略和当前策略概率可比较。

同时训练普通 SFT 对照：使用同一个训练文本对应的真实动作 VQ tokens，保持相同初始 LoRA、学习率和更新步数，不使用 MotionCritic 奖励。分别保存两个适配器。检查适配器变化量大于零、梯度有限非零、所有冻结参数的完整 SHA256 在更新前后一致。

训练后给五个留出条件设置共同的逐条件随机种子，分别生成原模型、普通 SFT、奖励微调三组动作，保存为可视化诊断。它们没有被送回优化器，也尚未完成新的常用指标/感知评分。

结果应以 `results/motiongpt_stage2/pilot/pilot_result.json` 为准。两步更新、一个训练条件只能验证“采样 → 奖励 → 梯度 → 权重保存”是否可用，不能据此声称质量提高。正式实验仍需要扩大训练条件、明确计算预算，并在固定测试条件上重新比较三组常用指标和感知分数。

本次已实际完成试跑：训练集条件为 `012738`，文本为 “the person had his arms out straight and put them down.”。两个候选均为 196 帧，奖励分别为 **0.43342591 / 0.69736882**，组内中心化优势为 **-0.13197145 / +0.13197145**。共训练 147,456 个 LoRA 参数；普通 SFT 与奖励分支均完成两次更新，适配器变化 L2 分别为 **0.05788048 / 0.05464074**。两分支梯度有限且非零，原模型冻结权重的完整哈希前后一致。采样、拟合、训练与预览合计约 932 秒，峰值 PyTorch 已分配显存约 1,188 MiB（不含驱动或其他进程）。

五个留出条件的三组预览全部有效，共 15 条，已加入离线网页（累计 336 条记录）。**本次两步更新后，两分支在共同种子下的五条生成 token 均与原模型相同**：参数确实更新，但尚未观察到这些离散生成结果变化，不能报告质量提升。奖励适配器在新进程中重新加载后，seed=5321 的留出动作 tokens 和骨架均与保存预览一致，结果在 `adapter_reload_check.json`。微调后的常用指标与新的感知评分尚未计算。

中断后可使用 `python tools/run_motiongpt_reward_pilot.py --resume`，复用完整采样 token、旧策略概率和已完成拟合。恢复前检查配置、五条留出面板、权重哈希、数据来源哈希与上游源码版本一致。只允许训练同一批数据；完成的试跑不会追加更新。两分支保持训练条件和步数相同，但没有匹配训练 token 总量或算力预算，不构成正式公平对比。

后续可通过标准评测入口加载任一适配器，例如：

```powershell
python tools/run_motiongpt_stage1.py --config configs/motiongpt_stage1/config.yaml evaluate --max-samples 320 --repetitions 1 --adapter results/motiongpt_stage2/pilot/critic_reward_adapter.pt
```

该命令不在本次“两步功能试跑”中自动运行；它仍需要微调后感知评分和重复实验配合，才能判断质量变化。

实现思路参考 [PPO 原论文](https://arxiv.org/abs/1707.06347) 的截断策略目标和 [LoRA 原论文](https://arxiv.org/abs/2106.09685) 的冻结权重与低秩增量。这里的分组优势、HumanML3D → SMPL 迁移评分及试跑规模是本项目的实验选择，不是 MotionGPT 或 MotionCritic 论文原始训练协议。
