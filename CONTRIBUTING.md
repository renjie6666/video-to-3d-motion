# 贡献与提交规范

## 提交原则

每个提交只表达一个完整意图，并保持代码、测试和文档同步。数据集、模型权重、实验输出、日志、密钥与本机环境文件不得提交。

在项目根目录直接运行 `git commit` 时，Git 会载入仓库中的 `.gitmessage`。所有模板字段都应填写；不适用的字段写“无”，以便后续复现实验和定位回归。

新克隆仓库后，执行以下命令启用模板：

```powershell
git config --local commit.template .gitmessage
git config --local commit.cleanup strip
git config --local core.hooksPath .githooks
```

第三条命令启用仓库内的 `commit-msg` 强制校验。校验失败时提交会被拒绝，并列出缺失或不合规的字段。普通提交不得使用 `--no-verify` 绕过校验；只有修复钩子自身故障时才可临时使用，并应在后续提交中说明原因。

强制规则包括：

- 标题必须符合 `<type>(<scope>): <summary>`，且不超过 72 个字符。
- `type` 必须来自下方允许列表，`scope` 必须使用小写稳定标识。
- 五个正文区段必须存在、顺序正确且不得重复。
- 验证命令、验证结果、数据与指标影响等字段必须填写；不适用时明确写“无”。
- Git 自动生成的 merge、revert、fixup 和 squash 消息允许通过。

## 标题格式

```text
<type>(<scope>): <summary>
```

允许的 `type`：

| 类型 | 用途 |
| --- | --- |
| `feat` | 新功能或新研究能力 |
| `fix` | 缺陷修复 |
| `refactor` | 不改变外部行为的重构 |
| `perf` | 性能优化 |
| `test` | 测试新增或调整 |
| `docs` | 文档变更 |
| `build` | 构建或依赖变更 |
| `ci` | 持续集成变更 |
| `chore` | 日常维护 |
| `data` | 数据处理或数据接口变更 |
| `exp` | 实验配置或实验方案变更 |

推荐的 `scope`：`decode`、`pose2d`、`pose3d`、`completion`、`prediction`、`generation`、`eval`、`config`、`docs`。如果现有范围不适用，可以添加简短、稳定的新范围。

## 完整示例

```text
feat(decode): support ordered image sequence input

背景与目的：
- 为 Human3.6M 图片序列建立与视频输入一致的 FrameTask 接口。

改动内容：
- 增加六位数字帧号校验和缺帧检测。
- 将解码结果写入有界 FIFO 队列。

验证方式与结果：
- 命令：python -m unittest discover -s tests -v
- 结果：全部测试通过。

实验与数据影响：
- 数据集/配置：Human3.6M，configs/media_decode.yaml
- 指标或结果：50 FPS 时间基准保持不变。
- 可复现性影响：新增输入文件命名约束。

关联事项：
- Issue/任务：S1-01
- BREAKING CHANGE：无
```

## 提交前检查

- 变更内容与提交标题一致。
- 测试或验证命令已经实际执行，并如实记录结果。
- 配置、随机种子、数据版本和指标变化已记录。
- 未提交数据集、权重、输出文件、日志或凭据。
- 若存在不兼容变更，已在 `BREAKING CHANGE` 中说明迁移方法。
