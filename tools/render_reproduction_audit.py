"""Render the completed single-action reproduction audit as a local HTML page."""

from pathlib import Path
from html import escape
import json

root=Path(__file__).resolve().parents[1]/'results/reproduction_audit_2026-10-10'
evaluation=json.loads((root/'evaluation_audit.json').read_text())
original_path=root/'original_data_audit.json'
original=json.loads(original_path.read_text()) if original_path.exists() else None
official={row['case']:row for row in original['cached_runs']} if original else {}
page='''<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PoseMamba 单动作复现核查</title>
<style>body{background:#111820;color:#e5eaf0;font:15px system-ui,"Microsoft YaHei";max-width:1240px;margin:auto;padding:28px}h1{font-size:27px}h2{font-size:20px;margin-top:30px}p,li{line-height:1.9}a{color:#7ec8ff}table{border-collapse:collapse;width:100%;margin:18px 0}th,td{text-align:left;padding:13px 10px;border-bottom:1px solid #344757}th{color:#9eafbf}strong{color:#b6e5c1}.note{padding:14px;background:#20303e;border-left:4px solid #6eb5e2}.scroll{overflow-x:auto}.small{color:#a4b3bf;font-size:13px}</style>
<h1>PoseMamba 单动作复现核查</h1><p>S1 Directions：323 帧；S9 Directions_1：472 帧；相机 54138969。图片链路为检测器 → RTMPose → 243 帧 PoseMamba-L → 重叠平均，无平滑，启用翻转测试。</p>
<p class="note">S1 属于训练主体，只能用于排查；S9 只选一个动作、一个相机。这里的数值不能当作论文完整 H36M 测试集成绩。毫米指标越小越好。</p>
<h2>同帧输入对照</h2><div class="scroll"><table><thead><tr><th>输入 / 评估方式</th><th>帧数</th><th>相机 MPJPE · GT 根深度辅助</th><th>N-MPJPE · 相机 GT</th><th>PA-MPJPE · 相机 GT</th><th>官方 2.5D P1 · 同批稀疏帧</th><th>查看</th></tr></thead><tbody>'''
for row in evaluation['rows']:
    key=row['case'].replace(' + official XY passthrough',' + XY passthrough (distorted H5 input)')
    p1=official.get(key,{}).get('mpjpe_official_25d_mm')
    viewer=Path(row['viewer']).relative_to(root.resolve()).as_posix()
    page+=f'<tr><td>{escape(row["case"])}</td><td>{row["frames"]}</td><td>{row["mpjpe_camera_gt_root_depth_mm"]:.2f}</td><td>{row["n_mpjpe_camera_gt_mm"]:.2f}</td><td>{row["pa_mpjpe_camera_gt_mm"]:.2f}</td><td>{f"{p1:.2f}" if p1 is not None else "训练 split 无官方评估字段"}</td><td><a href="{viewer}">骨架</a></td></tr>'
page+='''</tbody></table></div><p class="small">相机 MPJPE：用已核验内参和 GT 根节点 XY/深度恢复相机毫米坐标，无最优尺度、旋转拟合。N 与 PA 是旧的相机 GT 诊断口径；它们与论文 2.5D 口径不同。GT XY 覆盖是官方 gt_2d 评估分支中的步骤。当前发布权重的配置为 gt_2d=false，输入替换不能等同于论文 GT-2D 专门训练实验。</p>
<h2>为什么之前 MPJPE 不可用</h2><p>模型输出是归一化的图像 3D，并非毫米相机 XYZ。H5 没有论文需要的 2.5d_factor 和 joints_2.5d_image，所以旧页面留空。现在已取得原版预处理数据，S9 可以计算并显示官方 2.5D P1；原版训练 split 不包含这两项，S1 仍使用明确标注的 GT 根深度辅助相机 MPJPE。</p><h2>逐项结论</h2><ol>
<li><strong>模型结构通过。</strong>主模型类与固定版本官方代码一致，权重严格加载。20 个空间块 + 20 个时间块；全参数 9,450,499，叶子模块计数 6,713,859，差别来自统计口径。</li>
<li><strong>关节与相机标注通过。</strong>H5 关节转换后，3D 标签投影到 2D 标签的最大误差不足 0.00006 像素。</li>
<li><strong>扫描误差不足以解释差距。</strong>所有 40 块、原图和翻转两次推理均抽样核对独立 float64 串行递推。在每块注入参考通道后，输出最大变化约 0.0015 mm；这不是原 CUDA 内核直接对照。</li>
<li><strong>输入协议有差异。</strong>原模型用 SH 二维输入，我们用 RTMPose；H5 part 是含畸变投影，官方 GT 输入来自 joint3d_image 的针孔投影。官方 gt_2d 还在评估时覆盖预测 XY。</li>
<li><strong>评估坐标有差异。</strong>旧页面用相机 XYZ 标注直接拟合归一化预测；论文用 joints_2.5d_image 和 2.5d_factor，属于缩放后的 2.5D 评估。不能直接横向比较。</li>
<li><strong>采样差异已经验证。</strong>目前图片每隔 5 个源帧一张，实际 10 FPS；243 个样本覆盖 24.2 秒。论文测试步长 243，我们重叠步长 81；论文按各动作均值再平均，并排除指定异常序列。</li>
</ol>'''
if original:
    page+='<h2>原版数据核查与采样对照</h2><p>以下直接使用官方预处理文件中的 factor / 2.5D 标签。输入仍是同一份发布权重；仅测试一个 S9 Directions 原版窗口，不测试完整数据集。</p>'
    page+='<table><thead><tr><th>条件</th><th>共同帧数</th><th>P1 MPJPE · mm</th><th>P2 PA-MPJPE · mm</th></tr></thead><tbody>'
    for row in original['temporal_control']:
        page+=f'<tr><td>{escape(row["case"])}</td><td>{row["frames"]}</td><td>{row["mpjpe_official_25d_mm"]:.2f}</td><td>{row["pa_mpjpe_official_25d_mm"]:.2f}</td></tr>'
    page+='</tbody></table><p class="small">native 使用真实原版连续帧，sparse 使用每隔 5 帧采样；两组只比较相同的真实标注帧，不使用插值。采样对照与整段稀疏评估的帧数不同。</p>'
    page+='<p><a href="original_data_audit.json">原版核查 JSON</a> · <a href="original_data_cached.csv">官方口径整段 CSV</a> · <a href="original_temporal_control.csv">采样对照 CSV</a></p>'
page+='''<p><a href="s1_input_comparison/comparison/comparison.html">S1 输入对比页</a> · <a href="evaluation_audit.csv">相机诊断指标 CSV</a> · <a href="structure_audit.json">结构核查</a> · <a href="all_blocks_scan_audit.json">数值核查</a></p>
<p class="small">依据：<a href="https://github.com/nankingjing/PoseMamba/blob/df38d599212c058259f473464baec66d6a6487e0/train.py">PoseMamba 官方评估代码</a>；<a href="https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/gendb.py">原始 LCN 标注预处理</a>。记录的是实验结果，未新增设计或说明文档。</p></html>'''
(root/'audit.html').write_text(page,encoding='utf-8')
