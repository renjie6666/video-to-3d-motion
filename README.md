# 三阶段方案设计 V3

| 文档 | 状态 |
| --- | --- |
| [01 阶段一：三维人体姿态估计方案设计](docs/01_阶段一_三维人体姿态估计方案设计.md) | V3.16设计稿 |
| 阶段二方案 | 待编写 |
| 阶段三方案 | 待编写 |

阶段一时间基准统一采用Human3.6M的50 FPS。S1-01支持视频文件和连续图片序列两种输入，二者统一输出`FrameTask`并写入有界FIFO队列。

仓库包含源代码、测试、模型配置、依赖锁文件及设计和实验分析文档。数据集、模型权重、实验原始输出、虚拟环境和`work/`临时文件不随仓库发布。克隆后需按下文安装环境，并自行准备`datasets/`中的数据及`checkpoints/`中的权重。

详细设计见：[S1-01 视频与连续图片帧输入详细设计](docs/S1-01_视频解码与帧任务队列详细设计.md)。

## S1-01验证

在项目根目录执行。以下命令使用本地Human3.6M连续图片序列，生成运行摘要和每帧任务元数据：

```powershell
.\.venv\Scripts\decode-images.exe `
  "human3.6mtoolbox处理后的\images\s_01_act_02_subact_01_ca_01" `
  --config configs\media_decode.yaml `
  --tasks-output results\image_frame_tasks.jsonl `
  --summary-output results\decode_images_result.json
```

输入目录、`--config`、`--tasks-output`和`--summary-output`均为必填项。缺少任一项时，命令会列出缺少的参数并终止；运行期间终端显示进度条，成功后显示`SUCCESS`。

验证汇总结果：

```powershell
Get-Content results\decode_images_result.json
```

查看前5个`FrameTask`元数据：

```powershell
Get-Content results\image_frame_tasks.jsonl -TotalCount 5
```

图片目录必须只包含一个序列，文件名末尾使用六位数字帧号，并从`000001`开始；缺帧返回`frame_missing`，起始帧不符合要求返回`frame_start_invalid`。

当前命令是S1-01验证命令。它内置诊断接收循环，持续从队列取出任务以生成汇总和JSONL，尚未接入S1-02二维姿态处理。

## S1-02开发链路验证

S1-02真实RTMDet＋RTMPose链路要求NVIDIA CUDA环境，可在本地Windows或AutoDL等Linux环境运行。程序固定使用第一张可见GPU，即`cuda:0`；CUDA不可用时立即失败，不回退CPU。

建议在AutoDL选择PyTorch 2.1.2、Python 3.10、CUDA 11.8镜像。代码放在`/root/video-to-3d-motion`，模型权重、结果和叠加图放在`/root/autodl-tmp`。安装顺序是：先确认镜像自带的CUDA版PyTorch可用，再安装匹配的MMCV、MMDetection 3.3.0和MMPose 1.3.2，最后安装本项目。不要在安装OpenMMLab组件时覆盖镜像中的PyTorch。

启动前检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import mmcv, mmdet, mmpose; print(mmcv.__version__, mmdet.__version__, mmpose.__version__)"
```

连续图片输入：

```bash
estimate-pose2d-images /root/autodl-tmp/input/sequence --media-config configs/media_decode.yaml --pose2d-config configs/pose2d.yaml --frames-output /root/autodl-tmp/results/pose2d_frames.jsonl --summary-output /root/autodl-tmp/results/pose2d_summary.json
```

视频输入：

```bash
estimate-pose2d-video /root/autodl-tmp/input/input.mp4 --media-config configs/media_decode.yaml --pose2d-config configs/pose2d.yaml --frames-output /root/autodl-tmp/results/pose2d_frames.jsonl --summary-output /root/autodl-tmp/results/pose2d_summary.json
```

默认配置中的逐帧叠加图写入项目根目录的`results/visualizations/pose2d/<video_id>/`。AutoDL部署时可将权重和叠加图输出路径改为`/root/autodl-tmp`下的绝对路径。运行前必须准备匹配的PyTorch、MMCV、MMDetection、MMPose环境，以及配置中指定的本地权重。

`configs/pose2d.yaml`中的相对路径统一以该YAML所在目录为基准，模型配置、权重和叠加图输出均不随启动目录改变。例如`../checkpoints/rtmpose-m_256x192.pth`指向项目根目录的`checkpoints/rtmpose-m_256x192.pth`；绝对路径仍按原值使用。命令行输入与结果文件参数仍以启动目录为基准。

当前`configs/pose2d.yaml`默认使用Kineo公开的Human3.6M微调RTMPose-l作为`baseline`，输出协议为`h36m17`。网络输入为宽288、高384，由`configs/project/h36m17/rtmpose-l_8xb256-420e_h36m-384x288.py`统一定义；人体框仍由现有RTMDet检测并交给MMPose裁剪。`baseline`允许`coco17`或`h36m17`，以所选模型的`joint_schema`为准；`full`仍要求`h36m17`。

从[作者模型页面](https://huggingface.co/charlesjvt/rtmpose-l-h36m-384x288/tree/main)下载的两个文件保留原文件名：`rtmpose_h36m.pth`放在`checkpoints/`，`rtmpose-l_8xb256-420e_h36m-384x288.py`放在`configs/project/h36m17/`。模型配置的基础运行配置引用改为项目内路径，并先关闭翻转测试。作者声明权重限研究和评估使用；checkpoint中的`dataset_meta`与`CLASSES.keypoints`对左右腿的顺序描述存在差异，需通过标注对比确认实际通道顺序。

当前保留三个配置项：`rtmpose_m`、`rtmpose_l`和`rtmpose_l_h36m`。三者网络输入均为高384、宽288，翻转测试均关闭。通用M和L使用官方AIC+COCO权重、输出COCO17，H36M微调版输出H36M17。M已补齐[384×288官方权重](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-384x288-a62a0b32_20230228.pth)；L使用[对应官方权重](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth)。权重保留原文件名放在`checkpoints/`，匹配的官方MMPose v1.3.2配置在`configs/models/mmpose/body_2d_keypoint/rtmpose/coco/`。本地将M和L的`flip_test`改为`false`以匹配H36M模型，来源及差异见`configs/models/README.md`。

将`configs/pose2d.yaml`的`active_model`依次设为上述三个键，分别运行同一序列；不传`-MaxFrames`表示处理全部1384张图片。保持检测器、阈值和图片选择一致，简易表优先比较`common10_mean_error_px`，因为COCO17与H36M17的关节定义不同。以前M使用256×192、COCO权重且开启翻转测试，该配置和权重仍保留用于追溯旧结果；新的同分辨率M实验使用另一份384×288配置和权重，不能把变化全部归因于增加图片数量。完整模式的项目自训练权重仍是待实现项。

## 本地 GPU 环境检测

### 本地图片序列：一次运行解码、姿态估计和标注对比

`decode-images`只验证S1-01。完整链路使用`estimate-pose2d-images`：同一个进程内启动S1-01生产者和S1-02消费者，通过有界队列传递带原图像素的`FrameTask`，随后通过另一个队列输出`Pose2DFrame`。不需要先运行解码命令，也不需要把解码JSONL再读回来。

为当前放在`datasets/`的序列和训练标注提供了启动脚本。在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_pose2d_h36m.ps1
```

先验证前30张图片，可使用独立结果目录，避免覆盖完整运行的结果：

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_pose2d_h36m.ps1 -MaxFrames 30 -OutputDirectory results\h36m_smoke
```

脚本默认使用项目的`.venv`、`configs/pose2d.yaml`、`datasets/s_01_act_02_subact_01_ca_01/`和`datasets/h36m_train.pkl`。每次在`results/<active_model>_<北京时间>/`新建独立目录，避免覆盖。可以通过`-Sequence`、`-Annotations`、`-OutputDirectory`、`-Pose2DConfig`指定其他路径；`-OutputDirectory`现在表示实验根目录，其下每次新建实验子目录并维护两张总表。`-MaxFrames`默认为0，处理所有图片。`-ComparisonEveryNFrames 10`只减少对照图片的数量，所有推理帧仍参与表格对比。

对于已经抽样的稀疏图片序列，在媒体配置中设置`image_sequence.require_contiguous: false`，通过通用命令的`--media-config`指定该配置。此模式对每张图片仅处理一次，保留源帧号和源时间戳，不填补缺帧，也不按目标FPS重采样；它适合逐帧二维效果对比，接入下游时序模型前还需处理时间间隔。

H36M专用脚本默认`-PersonSelection highest_score`，针对已知单人序列，在超过检测阈值的候选框中选置信度最高者。当前第318、319帧把头部误检为额外的小人体框，严格模式会停止；此选项使单人序列能继续处理。选框规则保存到推理汇总、每帧JSONL和对比CSV，不使用标注框替代检测框。可用`-PersonSelection strict`要求多框时立即失败。直接使用通用命令时不指定`--person-selection`，仍遵循YAML原有设置。

默认输出包括：

- `results/experiment_summary.csv`：完整累计表，每个实验一行。
- `results/experiment_summary_simple.csv`：六列简易累计表，包含实验编号、模型、平均像素误差、共同10关节平均误差、有效置信度点平均误差和置信度覆盖率。跨COCO17与H36M17比较时优先看`common10_mean_error_px`；`confidence_coverage=1`表示100%的可见标注对应点通过阈值，不代表100%准确。字段说明见方案设计文档S1-02输出格式。
- 实验目录中的`<experiment_id>_summary.csv`：本次完整汇总；`<experiment_id>_effective_config.yaml`和`<experiment_id>_media_config.yaml`保存配置；`<experiment_id>_run.log`保存运行日志。
- `<experiment_id>_pose2d_frames.jsonl`：原图像素坐标、置信度、有效标记、人体框和帧信息，供S1-03读取；标记`coordinate_space=original_image_pixels`。
- `<experiment_id>_joint_comparison.csv`和`<experiment_id>_joint_summary.csv`：每帧每关节的明细及逐关节统计。
- 实验目录中的`pose_overlays/<video_id>/`和`comparison_overlays/<video_id>/`：原尺寸预测叠加图及标注对照图，绿色为预测，红色为低置信度预测，黄色为H36M标注，灰线为对应点偏差。

已有实验可重新生成完整与简易累计表，无需重新推理：

```powershell
.\.venv\Scripts\python.exe -m video_to_3d_motion.pose2d.experiments --output-root results --rebuild-reports
```

只生成简易表并保留完整表的当前内容时，将`--rebuild-reports`改为`--rebuild-simple`。

选择COCO17模型时，只比较12个可对应的四肢关节。肩、肘、腕、髋、膝为10个直接对应点；COCO踝与H36M足为2个近似对应点，单独汇总。鼻、眼、耳没有直接对应的H36M标注，不参与误差计算，不合成骨盆、脊柱或头部关节来充当真实预测。选择H36M17模型时，对17个关节直接比较。读取标注时假定关节顺序为标准H36M17；首次使用其他来源的PKL或权重应核对其生成工具和叠加图。

当前1384张图片中，训练标注匹配1383张，最后一张没有标注；仍保留推理结果，对比行标记`missing_gt`，误差留空并排除出统计。`gt_invisible`表示标注不可见，`unmapped_joint`表示无对应关节。`low_confidence`的误差仍纳入`all_visible_pairs`，另外报告置信度有效点的统计与覆盖率，避免只筛选好点而夸大精度。没有指定“足够准确”的阈值，因此程序不会自动宣称通过准确度验证。当前S1序列属于训练集，用于效果检查，不能代表验证集成绩。

大型PKL按帧记录批次读取，保留共享相机信息并释放已处理帧。只允许NumPy数据重建所需类型，不执行任意自定义类；不支持跨帧复用关节数组等复杂对象引用。匹配结果缓存到标注所在目录的`.annotation_cache/`，标注文件大小、修改时间或图片名单改变后缓存失效。当前数据、缓存、权重和结果都被Git忽略。

需要查看全部参数时：

```powershell
.\.venv\Scripts\estimate-pose2d-images.exe --help
```

已有推理结果时，可以只重新做标注对比，无需重新加载模型或使用GPU：

```powershell
.\.venv\Scripts\python.exe tools\run_h36m_comparison.py `
  datasets\s_01_act_02_subact_01_ca_01 `
  --frames-output results\pose2d_h36m\pose2d_frames.jsonl `
  --annotations datasets\h36m_train.pkl `
  --output results\pose2d_h36m\comparison
```

JSONL表示“一行一帧的JSON”。程序可以逐帧写入而不把所有结果攒在内存中；中断后已写入的完整行仍可检查；嵌套的17×2坐标、17个分数和元数据也能保留结构。它是结果落盘与回放格式，运行时模块之间仍使用队列。目前命令行消费者负责保存S1-02结果，S1-03三维处理器尚未实现；将来可直接消费姿态队列，也可离线读取JSONL。给人看的逐关节表格使用CSV。

### 环境安装与检查

项目统一使用 `.venv` 环境，包含Windows本地S1-02所需的GPU依赖。安装脚本要求 `uv` 已在 PATH 中，下载 Python 3.10 和依赖，并执行 GPU 检测：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_local_gpu.ps1
```

版本组合为 Python 3.10、PyTorch 2.1.2 / torchvision 0.16.2（CUDA 11.8版）、NumPy 1.26.4、MMEngine 0.10.7、MMCV 2.1.0、MMDetection 3.3.0、MMPose 1.3.2。脚本显式使用 PyTorch 官方 CUDA 源和 Windows MMCV CUDA 预编译包，不需要源码编译 CUDA 算子。代码运行、测试和推理均使用这一个环境。

S1-02 重依赖在 `pyproject.toml` 的 `pose2d` 可选依赖中声明，额外兼容性约束见 `configs/gpu-windows-constraints.txt`。脚本先准备旧依赖所需的构建工具，再按 `uv.lock` 同步完整环境。请使用安装脚本建立此环境；仅运行 `pip install .[pose2d]` 不能保证获得正确的 CUDA 安装包。手动使用 `uv sync` 更新环境时需要加上 `--extra pose2d`，以包含GPU推理依赖。

安装后使用专用解释器检查或运行测试：

```powershell
.\.venv\Scripts\python.exe tools\check_local_gpu.py --require-pose2d
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

使用计划运行 S1-02 的 Python 环境执行，检测程序本身只依赖 Python 标准库，不需要先安装本项目：

```powershell
python tools\check_local_gpu.py
```

若 `python` 指向不可用的环境，请使用已有 Conda/虚拟环境中 Python 的完整路径，例如：

```powershell
& "C:\path\to\env\python.exe" tools\check_local_gpu.py
```

程序检查 NVIDIA 驱动、PyTorch CUDA、第一张可见 GPU 的矩阵运算、基础依赖、OpenMMLab API 导入和 MMCV CUDA NMS 算子。每项依赖检测在独立子进程中运行，默认超时为120秒，可通过 `--timeout` 修改；不会安装依赖或下载模型。

结果默认写入项目根目录的 `results/local_gpu_check.json`，不受启动目录影响，包含解释器路径、系统信息、依赖版本和失败原因；可以使用 `--output` 指定其他文件：

- `gpu_usable=true`：当前解释器能在 `cuda:0` 上实际执行运算。
- `pose2d_dependencies_ready=true`：基础依赖、OpenMMLab API 和 CUDA NMS 检测也通过；仍需准备模型配置、权重并完成真实推理验收。
- 默认退出码以 GPU 检测为准；添加 `--require-pose2d` 后，必须依赖检测也通过才返回0。

```powershell
python tools\check_local_gpu.py --require-pose2d
```

当前 `configs/pose2d.yaml` 已使用相对路径，可供Windows和Linux共用。本地真实推理前仍需准备匹配的RTMPose、RTMDet权重。

## 自动化测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 开发与提交规范

项目采用结构化 Git 提交模板。提交标题遵循 Conventional Commits，正文记录变更动机、验证结果以及实验与数据影响。首次克隆后的配置方法和完整示例见 [CONTRIBUTING.md](CONTRIBUTING.md)。
