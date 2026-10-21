# MotionGPT 第一阶段：HumanML3D 基线与 MotionCritic 补充评测

本阶段使用 OpenMotionLab 的 NeurIPS 2023 MotionGPT，任务为文本生成动作。冻结生成器、动作 VQ-VAE、T2M 评测器和 MotionCritic；不进行奖励微调。阶段二再比较普通继续训练与感知奖励微调。

本地指标与动作回放页面：`results/motiongpt_stage1/experiment_viewer.html`，双击即可离线打开。基础数据包含 320 条子集生成结果和独立感知样例，共 321 条记录；第二阶段试跑完成后再加入三组各五条留出条件的动作。支持按文本/ID 搜索、生成与真实骨架并排播放、逐帧、调速、旋转、缩放和根节点跟随。两侧不做时间拉伸或姿态对齐，较短序列结束后停留在末帧。页面明确区分 320 条常用指标、五条 MotionCritic 基线和未重新评分的微调预览。

生成页面使用已保存的 `.npz` 和指标报告，不调用模型：`python tools/render_motiongpt_report.py`。可通过 `--baseline`、`--single` 和 `--output` 指定报告目录和输出路径。HTML 内嵌压缩数据，约 18 MB，建议使用近期版本 Edge / Chrome / Firefox。

最初按用户的额度限制，只执行 1 条 HumanML3D 测试样本的生成和感知评分。入口为 `python tools/run_motiongpt_single_test.py`；结果在 `results/motiongpt_stage1/single_test_*/single_test_result.json`。随后扩大到下述子集；这仍不等于完成完整测试集基准。

随后按用户要求补算了固定随机子集的常用指标：320 条测试样本、seed=1234、1 次重复、检索候选组大小 32、Diversity 300 对。结果位于 `results/motiongpt_stage1/baseline_20261019_230154/baseline_metrics.json`：R-Precision Top-1/2/3 为 0.4750 / 0.634375 / 0.721875，FID 为 0.31280563，Diversity 为 9.31168079，MM-Dist 为 3.70073178，无无效生成。该子集结果不是完整测试集的 20 次重复结果，未估计置信区间。表格 `实验指标汇总_320条评测.xlsx` 同时保留独立样本 004822 的 MotionCritic 分数；该分数不是这 320 条的感知均值。

已完成测试样本 `004822`：文本为 “a person is walking in place at a slow pace.”，生成 196 帧、20 FPS；MotionCritic 四个窗口均值为 **-0.49802348**。SMPL 拟合平均 RMSE 为 1.3883 厘米，最大 1.8433 厘米；拟合与评分耗时约 433 秒，无失败。报告位于 `results/motiongpt_stage1/single_test_20261019_224114/single_test_result.json`。分数是未校准的学习评分，不是概率；单条结果不能代表整体质量。

后续按用户要求，从 320 条生成结果中以 seed=1234 随机抽取四条评分，与原独立样例组成五条留出对照。五条全部成功，动作等权均值 **-2.07968047**，覆盖率 **100%**。原始记录和来源见 `results/motiongpt_stage1/critic_panel_five.json`，逐条分数为：004822 -0.49802348、007644 -4.08437991、M000597 -0.28317105、M006420 -1.94818121、M008828 -3.58464670。这五条不是完整 320 条的感知均分。第二阶段的训练隔离与试跑说明见 [奖励微调试跑](MotionGPT_第二阶段奖励试跑.md)。

**安装或样例评分通过不等于复现完成。** 完成条件是同一个真实 HumanML3D 测试样本库同时具有 `baseline_metrics.json` 和 `critic_summary.json`，并公开评分覆盖率、失败数和输入转换协议。实际状态以 `results/motiongpt_stage1/` 的报告为准。

## 固定来源

- [MotionGPT 官方代码](https://github.com/OpenMotionLab/MotionGPT)：`001aaca8d0ee218fc17f8265d11ac124044fe42f`。
- [MotionCritic 官方代码](https://github.com/ou524u/MotionCritic)：`71e9f52e67df4d9924607afd6ac4e0d4dc0dd27e`。
- [MotionGPT-base 官方权重](https://huggingface.co/OpenMotionLab/MotionGPT-base)：`motiongpt_s3_h3d.tar`。
- [MotionCritic 官方权重](https://huggingface.co/littlekoyo/MotionCritic)：`motioncritic_pre.pth`。
- T2M 评测器、归一化文件和 SMPL 资源来自 MotionGPT 官方 `prepare/` 脚本列出的公开文件。
- [HumanML3D 官方元数据与预处理说明](https://github.com/EricGuo5513/HumanML3D)。官方完整动作数据需从 AMASS 预处理。
- XYZ 数据来自[公开社区镜像](https://huggingface.co/datasets/yonful/gem-analysis-humanml3d)；263 维特征直接来自[特征镜像](https://huggingface.co/datasets/lavinal712/HumanML3D/tree/430eae544e2d71b8e582123c1ea01cdb24eef725)，不是官方发行。准备脚本固定 revision、校验 LFS SHA256，并核验官方划分、统计文件、公开特征样例、动作维度与完整性。记录在 `datasets/HumanML3D/source_manifest.json`。原始数据条款仍适用；元数据和单个样例一致不代表全部动作逐字节一致。

源码在 `work/`，权重在 `checkpoints/`，数据在 `datasets/`，结果在 `results/`，均不进入 Git。

## 目录与 Conda 环境

现有 Human3.6M 数据已移至 `datasets/Human3.6M/`；HumanML3D 使用 `datasets/HumanML3D/`。相关旧阶段 PowerShell 启动脚本已更新数据路径。

按用户要求使用 **Miniconda + 独立 Conda 环境 `motiongpt`**：

```text
work/miniconda3/                         Miniconda 安装位置
work/miniconda3/envs/motiongpt/           本实验 Conda 环境
configs/motiongpt_stage1/environment.yml  Conda 基础环境定义
configs/motiongpt_stage1/requirements.txt CUDA PyTorch 与 Python 依赖
configs/motiongpt_stage1/conda-win64-explicit.txt  实际 Conda 包精确锁定
configs/motiongpt_stage1/requirements-win64.lock.txt 实际 Python 包精确锁定
```

Miniconda 安装器使用官方固定文件 `Miniconda3-py310_26.9.3-0-Windows-x86_64.exe`，SHA256 为 `4716fd947314668123e9b1a4aa011a6bff629bf1ec0614fb72df50997608a1e3`。安装脚本验证哈希，安装到项目内，不修改全局 PATH 或默认 Python。

在项目根目录运行：

```powershell
.\tools\setup_motiongpt_stage1.ps1 -DownloadAssets
```

Python 使用 3.10，PyTorch 使用 `2.1.2+cu118`，Transformers 使用 `4.35.2`。这是针对本地 Windows/CUDA 的兼容环境，官方原始测试环境为 Python 3.10.6/PyTorch 2.0.0；不能仅凭依赖安装成功声称复现论文数值。

本机实际创建的 Conda Python 为 3.10.22。安装脚本优先使用两份 Windows 锁文件；`environment.yml` 和 `requirements.txt` 保留可读的环境定义。Conda 环境与 CUDA 已实际验证，`critic_example.json` 记录解释器路径和 `is_conda_environment=true`。

当前 PowerShell 会话激活环境：

```powershell
. .\work\miniconda3\shell\condabin\conda-hook.ps1
conda activate motiongpt
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

也可以直接使用 `work/miniconda3/envs/motiongpt/python.exe`。此前用于前置探测的 `work/motiongpt_env/` 是临时环境，最终运行入口和文档统一使用 Conda 环境。

数据下载与核验：

```powershell
python tools/prepare_humanml3d.py
python tools/run_motiongpt_stage1.py --config configs/motiongpt_stage1/config.yaml check
```

`check` 返回 2 表示存在缺失条件，返回 0 仅表示资源可用，不表示完成正式评测。资源包下载后可单独执行：

```powershell
python tools/run_motiongpt_stage1.py --config configs/motiongpt_stage1/config.yaml extract
```

## 先做小样本验证

官方 MotionCritic 样例验证：

```powershell
python tools/run_motiongpt_stage1.py --config configs/motiongpt_stage1/config.yaml score-example
```

该入口使用官方 `into_critic` 处理作者提供的 SMPL 样例，严格加载全部 critic 参数并冻结，检查 CPU/GPU 分数一致性。报告在 `results/motiongpt_stage1/critic_example.json`。这不涉及 HumanML3D 生成结果。

MotionGPT 生成验证：

```powershell
python tools/run_motiongpt_stage1.py --config configs/motiongpt_stage1/config.yaml demo --captions configs/motiongpt_stage1/smoke_prompts.txt
```

每个样本保存原始尺度的 263 维特征、22 关节 XYZ、动作 tokens、文本与帧数。`demo.json` 明确标记 `humanml3d_benchmark=false`。从完整官方 checkpoint 加载 T5 与 VQ-VAE；使用原始 T5 配置和 tokenizer 构造相同结构，不重复下载已经被完整 checkpoint 覆盖的基础模型权重。

## 正式基线评测

```powershell
python tools/run_motiongpt_stage1.py --config configs/motiongpt_stage1/config.yaml evaluate
```

默认完整测试集、20 次重复，输出均值及 95% 置信区间。保留每次重复的样本清单、特征与 embedding，便于后续微调前后比较。主要指标：R-Precision Top-1/2/3、FID、Diversity、MM-Dist，同时报告真实动作对应的检索和多样性指标。

小规模链路检查可以运行：

```powershell
python tools/run_motiongpt_stage1.py --config configs/motiongpt_stage1/config.yaml evaluate --max-samples 320 --repetitions 1
```

该结果必须标记为子集验证；不能替代完整 20 次评测。默认 Diversity 使用 300 对，因此样本数必须大于 300；R-Precision 固定每组 32 个候选，不能为了适配显存减少候选数。

实现复用固定版本的官方数据集、运动/文本编码器与 FID/Diversity 数学函数。正式评测保持评测 batch size=32；生成时可用较小的微批次。保留官方随机文本选择、动作裁剪、生成长度与参考长度取较小值、两套不同归一化统计、按文本长度排序与检索前随机分组的处理。

特别注意：固定版本的官方 motion GRU 并没有按长度打包，padding 会影响 embedding。这里保留其行为，不能随意将评测 batch size 改成 1；也不修改其重复的长度整除操作来“优化”基线。

每个样本记录无效 token 输出；采用与官方相同的零特征回退，同时保留失败标志，不将失败样本悄悄从指标分母中移除。

## MotionCritic 补充评分

针对某次评测保存的同一批动作运行，例如：

```powershell
python tools/run_motiongpt_stage1.py --config configs/motiongpt_stage1/config.yaml score --manifest results/motiongpt_stage1/baseline_YYYYMMDD_HHMMSS/rep_00/samples.jsonl
```

配置 `critic.sample_count=1` 保留最初的低成本默认值；本次补充四条时显式使用 `--max-samples 4`。从同一生成样本库按 seed=1234 无放回随机抽取，避免只评测按长度排序的最短样本。`critic/selection.json` 保存样本 ID、文本、生成文件、清单哈希和拟合协议，阶段二使用相同条件对照。报告明确标记子集；`tools/build_motioncritic_panel.py` 合并原独立样例和四条新评分。

评分保存到样本库的 `critic/` 子目录和 `critic_summary.json`。默认拒绝覆盖已有目录；中断后以完全相同的参数追加 `--resume`，继续处理尚未完成的样本，不重复已完成的拟合。恢复时检查样本清单、随机选择及协议一致性。

默认输入转换协议 `smplify_adam_xyz_20_to_20fps_60frame_relative_root_windows_v2`：

1. 从 MotionGPT 的原始尺度特征恢复 HumanML3D 的 XYZ，单位米、Y 向上。
2. 使用 MotionGPT 官方 SMPLify3D 的 Adam 分支，对前 22 个 SMPL 关节拟合 24 关节轴角姿态与平移；每帧默认 100 次身体拟合更新，使用前一帧初始化，保存拟合参数和每帧 RMSE。默认任一帧 RMSE > 0.10 米则拒绝该序列。优化器与迭代次数写入报告；不会把 Adam 分支称为原始默认 LBFGS 配置。
3. 默认保持 HumanML3D 和上游 MDM 生成代码使用的 20 FPS；作者渲染器写视频的 24 FPS 不视为训练采样率。若以后修改采样率，旋转使用 SLERP、平移使用线性插值，保持时间长度，不把任意动作拉伸成 60 帧。
4. 按 60 帧窗口评分，步长 60 帧；覆盖尾部时允许最后一个窗口重叠。每个窗口减去首帧平移，与上游 HumanAct12 数据加载器的相对 pelvis 位移约定一致。不到 60 帧的动作标记无法按此协议评分，不补零、不丢失统计记录。
5. critic 输入为 `[B,60,25,3]`，前 24 项为 SMPL 轴角，最后一项为平移。先对每条动作的窗口分数取均值，再对动作取均值，避免长动作权重过大。

HumanML3D 的 IK 旋转并不等于 SMPL 姿态参数，不能直接追加两个关节或 reshape 后送入 MotionCritic。该转换包含拟合开销，也可能引入平滑和误差，因此这里的分数是一个明确声明的迁移评测协议，不能直接等同于论文的全部原始设置。

必须同时查看评分覆盖率、拟合误差、过短序列数、无效生成数和失败列表。MotionCritic 分数不表示人类满意概率；训练模型直接优化该分数属于阶段二，仍需独立可视化/人工评价来检查是否迎合评测器。

## 验证与当前边界

- `tests/test_motiongpt_stage1.py` 验证时间保持、根速度、旋转跨越 ±π 的插值、窗口尾部覆盖、世界平移不变性、非法输入、归一化文件歧义及官方检索/FID 的独立语义。
- 已在 RTX 3060 Laptop 6GB 上对官方 4 段 SMPL 样例进行真实 MotionCritic 推理，CPU/CUDA 分数一致性通过，详见 `critic_example.json`。
- 对真实 HumanML3D 的正式指标、SMPL 拟合与评分结果，需要以实际生成的报告为准。未生成报告的步骤一律不视为通过。

XYZ 包提供分目录存储的 `new_joints`，准备脚本统一 ID 布局。正式评测使用另行下载的现成 `new_joint_vecs.zip`，不使用本地 XYZ 重建特征。准备时覆盖前置探索生成的特征，以压缩包 CRC、SHA256 和数组维度/有限值/官方 ID 完整性检查为准。官方公开样例 `012314.npy` 采用 rtol=1e-4、atol=2e-5 核验，实际最大差异记录在来源清单中，不声称逐字节一致。所有生成条件使用官方原始 `texts.zip`，不采用镜像为其他任务扩展的文本。
