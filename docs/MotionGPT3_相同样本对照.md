# MotionGPT3：与 MotionGPT 使用相同测试条件

使用 OpenMotionLab 官方 [MotionGPT3](https://github.com/OpenMotionLab/MotionGPT3)，源码固定为 `cf5bab7daa568ce472e7201cc2bf8b83d8c1b732`。本次复现预训练模型的文本生成动作任务，不从头训练，也不做奖励微调。

## 环境与权重

单独创建项目内 Conda 环境 `motiongpt3`，Python 3.11；不修改原 `motiongpt` 环境。CUDA PyTorch 使用 2.1.2+cu118，Transformers 使用上游指定的 4.47.1。上游说明的 PyTorch 2.0 与其 SDPA 代码兼容范围存在差异，采用 Windows 上支持该接口的 2.1.2。依赖定义在 `configs/motiongpt3/`。仅安装推理依赖，不安装 Blender、WebUI 或训练服务。

首次创建可执行：

```powershell
.\work\miniconda3\Scripts\conda.exe create -n motiongpt3 python=3.11 pip --override-channels -c conda-forge -y
.\work\miniconda3\envs\motiongpt3\python.exe -m pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
.\work\miniconda3\envs\motiongpt3\python.exe -m pip install -r configs/motiongpt3/requirements.txt
```

```powershell
. .\work\miniconda3\shell\condabin\conda-hook.ps1
conda activate motiongpt3
```

官方 Hugging Face 权重页面需要访问授权；本次使用官方仓库 `prepare/download_pretrained_motiongpt3_model.sh` 指定的公开 Google Drive 文件 `1Wvx5PGJjVKPRvjcl8firChw1UVjUj36l`，保存到 `checkpoints/motiongpt3/motiongpt3.ckpt`。完整 checkpoint 已包含语言分支、动作分支、扩散头和动作 VAE，无需重新下载基础 GPT2 权重或单独 VAE。

`tools/prepare_motiongpt3.py` 只下载 GPT2 配置和分词器，记录 Hugging Face revision 与逐文件 SHA256。上游构造函数要求读取初始权重文件，因此创建空的 `model_state_dict.pth` 作为构造阶段占位；随后必须严格加载正式 checkpoint 的全部 `lm.*` 和 `vae.*` 权重。任何缺失、额外或形状不匹配都会中止，空初始化模型不能进入生成。未使用的 checkpoint 张量仅允许属于原评测器或训练损失。

## 相同样本的具体定义

原 MotionGPT 的 320 条来自 `results/motiongpt_stage1/baseline_20261019_230154/rep_00/samples.jsonl`。本次逐条复用该清单中的文字、ID、顺序和保存的真实参考动作，不重新随机抽样或裁剪参考动作。每个条件生成一个候选，另生成原独立样例 004822，共 321 条条件。生成种子为 `1234 + 清单位置`，支持复用已完成生成。

使用官方 `configs/test.yaml` 的架构与引导设置，包括 100 步扩散采样。动作长度遵循官方验证方法：MotionGPT3 VAE 按保存的参考长度解码；原 MotionGPT 先生成离散 token，再按生成长度与参考长度取较小值。此长度条件差异必须保留在报告中，不能声称两种模型的所有输入条件完全一致。不拉伸任何动作以强行匹配长度。

```powershell
python tools/run_motiongpt3_comparison.py --smoke
python tools/run_motiongpt3_comparison.py
```

运行配置与权重 SHA256 保存到 `results/motiongpt3_comparison/protocol.json`。`load_report.json` 记录严格加载范围；`generation_result.json` 记录实际解释器、GPU、输出数和无效生成数。

完整流水线入口为 `python tools/run_motiongpt3_pipeline.py`，会按顺序调用两个环境：新环境生成、原评测环境计算指标和拟合、生成离线网页。模型已完成的生成文件可继续复用；SMPL 拟合的恢复沿用原评测入口的 `--resume`，不重算已完成记录。

## 常用指标与感知评分

使用原 `motiongpt` 环境中的同一冻结 T2M 评测器，避免切换 Torch/编码器实现造成额外差异。生成器在新环境运行，评测与 SMPL 拟合复用已验证环境，两个环境相互独立。

```powershell
.\work\miniconda3\envs\motiongpt\python.exe tools/evaluate_motiongpt3_comparison.py
.\work\miniconda3\envs\motiongpt\python.exe tools/run_motiongpt_stage1.py --config configs/motiongpt_stage1/config.yaml score --manifest results/motiongpt3_comparison/critic_panel_samples.jsonl --max-samples 5
.\work\miniconda3\envs\motiongpt\python.exe tools/render_motiongpt3_comparison.py
```

常用指标使用相同的 320 条条件、32 条一组的 padding、冻结编码器、真实动作 embedding、文字 embedding。原 MotionGPT 生成 embedding 直接复用；两模型都重设 seed=1234 后计算检索分组和 Diversity 对，从而控制随机分组差异。旧网页的历史 R-Precision、MM-Dist 和 Diversity 可能与本次统一重算不同；FID 不受随机分组影响。历史结果保留，不覆盖。

MotionCritic 使用第一阶段固定的五条条件：004822、007644、M000597、M006420、M008828。按完整文件路径匹配，避免同一 ID 的其他文字或裁剪片段被误用。保持 XYZ→SMPL、Adam 100 次/帧、20 FPS、60 帧首帧相对平移窗口协议，公开失败、过短与覆盖率。新动作需要重新拟合，不能复用 MotionGPT 的 SMPL 参数或分数。

离线网页为 `results/motiongpt3_comparison/experiment_viewer.html`：同一条件下并排播放 MotionGPT、MotionGPT3、真实参考。全部动作与指标内嵌，支持搜索、逐帧、调速、旋转、缩放和跟随根关节，不请求外部网络。较短动作播放结束后停在末帧。

本次为固定 320 条、一次重复和五条感知评分的子集对照，不是完整测试集的多次重复论文复现；不依据小样本宣称模型优劣。

## 已完成的常用指标

实际已在独立环境生成 321/321 条有效动作，其中前 320 条进入常用指标。严格加载 649 个生成器/VAE 张量，其余 96 个张量仅属于旧训练损失和评测器。`pip check`、实际 CUDA 推理与网页离线交互检查均通过。

统一重算结果（MotionGPT → MotionGPT3）：

- R-Precision Top-1：0.4750 → 0.55625；Top-2：0.634375 → 0.759375；Top-3：0.721875 → 0.85625。
- FID：0.31280563 → 0.26241968。
- MM-Dist：3.70073178 → 2.72318757。
- Diversity：8.92177773 → 9.46833992；真实参考为 8.99437809，本次 MotionGPT 的该数值更接近真实参考，不将更高直接解释为更好。

原 MotionGPT 历史 Diversity=9.31168079，本次变为 8.92177773 的原因是固定统一重算的随机动作对不同，生成动作本身没有变化；旧结果继续保留。检索指标在这次重算中恰好与历史结果一致。完整数值来自 `results/motiongpt3_comparison/paired_metrics.json`，以该文件和长度条件说明为准。

## 已完成的五条感知评分

五条全部评分成功，覆盖率 100%，无失败或过短记录。MotionGPT3 的动作等权均值为 **-1.73737330**，动作间总体标准差为 **2.21742912**；原 MotionGPT 五条均值为 **-2.07968047**。逐条 MotionGPT3 分数：004822 **-0.43716876**、007644 **-5.03349113**、M000597 **1.42706436**、M006420 **-3.13822737**、M008828 **-1.50504362**。

本次五条中，MotionGPT3 三条分数更高、两条更低；不将平均分提高解释为每条动作都更好。生成时长仍遵循各模型既定机制，例如 M008828 原生成 72 帧、MotionGPT3 按参考长度解码 196 帧，该差异在汇总和回放中明确保留。

完整汇总为 `results/motiongpt3_comparison/comparison_result.json`，包含常用指标、逐条感知分数、拟合误差、时长和协议。`tools/finalize_motiongpt3_comparison.py` 在五条成功、文字条件一致、评分协议一致、原始分数与汇总均值一致时才写出完成状态。原始逐窗口结果与缓存拟合保留在同目录的 `critic/` 下。
