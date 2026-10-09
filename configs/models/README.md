# 本地 OpenMMLab 模型配置

配置与许可证保存在各自的`mmpose/`和`mmdet/`目录。YAML中的相对路径以`configs/pose2d.yaml`所在目录为基准；模型Python配置的`_base_`则以该Python文件所在目录为基准。

## RTMPose-l：同分辨率对比

- 配置键：`rtmpose_l`，与恢复的`rtmpose_m`及`rtmpose_l_h36m`并存。
- 官方配置：[MMPose v1.3.2 RTMPose-l AIC+COCO 384×288](https://github.com/open-mmlab/mmpose/blob/v1.3.2/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-384x288.py)。本地文件：`mmpose/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-384x288.py`。
- [官方模型清单](https://github.com/open-mmlab/mmpose/blob/v1.3.2/configs/body_2d_keypoint/rtmpose/coco/rtmpose_coco.yml)中的权重：[下载PTH](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth)。保存到项目根目录`checkpoints/`，保留原文件名。
- 输入高384、宽288；MMPose代码以宽、高顺序写为`input_size=(288, 384)`。它是人体框裁剪后的网络输入尺寸；原图和输出叠加图仍保留原始尺寸。
- 本地配置相对官方版本仅将`model.test_cfg.flip_test`由`True`改为`False`，与当前`rtmpose_l_h36m`一致；不调整网络结构、预处理或关节头。该文件引用的`../../../_base_/default_runtime.py`已在本地存在。
- [H36M模型作者说明](https://huggingface.co/charlesjvt/rtmpose-l-h36m-384x288)将上述官方权重列为微调初始化模型。H36M微调权重已在`checkpoints/rtmpose_h36m.pth`，使用现有`rtmpose_l_h36m`配置项。

比较两个模型时，使用同一序列、帧数、RTMDet检测器及阈值，并看简易累计表的`common10_mean_error_px`。两者分别输出COCO17和H36M17，全部关节的平均误差统计范围不同。

## RTMPose-m：同分辨率三组实验

- 配置键：`rtmpose_m`，与两个L模型并存。
- [MMPose v1.3.2官方配置](https://github.com/open-mmlab/mmpose/blob/v1.3.2/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_aic-coco-384x288.py)：本地同名文件在`mmpose/body_2d_keypoint/rtmpose/coco/`。
- [下载官方权重](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-384x288-a62a0b32_20230228.pth)，保留文件名放在项目根目录`checkpoints/`。
- 与通用L同为AIC+COCO训练，宽288、高384输入、COCO17输出；本地仅把官方`flip_test=True`改为`False`，三组统一推理设置。
- 旧的256×192 M配置和权重不删除，仍可追溯旧实验。当前M模型键指向384×288版本，检查完整汇总中的`model_config`、`checkpoint`和`checkpoint_sha256`可区分两种M实验。

已有RTMPose-m和RTMDet配置继续保留，便于追溯以前实验的配置快照。模型权重不存放在此目录，`checkpoints/`与`results/`仍被Git忽略。
