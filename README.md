# 三阶段方案设计 V3

| 文档 | 状态 |
| --- | --- |
| [01 阶段一：三维人体姿态估计方案设计](docs/01_阶段一_三维人体姿态估计方案设计.md) | V3.14设计稿 |
| 阶段二方案 | 待编写 |
| 阶段三方案 | 待编写 |

阶段一时间基准统一采用Human3.6M的50 FPS。S1-01支持视频文件和连续图片序列两种输入，二者统一输出`FrameTask`并写入有界FIFO队列。

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

## 自动化测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```