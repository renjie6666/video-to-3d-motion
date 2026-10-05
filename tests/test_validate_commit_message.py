from __future__ import annotations

import unittest

from tools.validate_commit_message import validate_message


VALID_MESSAGE = """feat(decode): support ordered image sequence input

背景与目的：
- 统一视频与图片序列的输入接口。

改动内容：
- 增加图片序列解码器。

验证方式与结果：
- 命令：python -m unittest discover -s tests -v
- 结果：全部测试通过。

实验与数据影响：
- 数据集/配置：Human3.6M
- 指标或结果：50 FPS 基准不变。
- 可复现性影响：新增文件命名约束。

关联事项：
- Issue/任务：S1-01
- BREAKING CHANGE：无
"""


class ValidateCommitMessageTests(unittest.TestCase):
    def test_accepts_complete_message(self) -> None:
        self.assertEqual(validate_message(VALID_MESSAGE), [])

    def test_rejects_short_message(self) -> None:
        errors = validate_message("feat(decode): add decoder\n")
        self.assertTrue(any("required section" in error for error in errors))

    def test_rejects_invalid_subject(self) -> None:
        message = VALID_MESSAGE.replace(
            "feat(decode): support ordered image sequence input",
            "added decoder",
        )
        errors = validate_message(message)
        self.assertTrue(any("subject must match" in error for error in errors))

    def test_rejects_empty_required_field(self) -> None:
        message = VALID_MESSAGE.replace("- 结果：全部测试通过。", "- 结果：")
        errors = validate_message(message)
        self.assertIn("required field has no value: 结果：", errors)

    def test_allows_git_generated_messages(self) -> None:
        self.assertEqual(validate_message('Merge branch "main"\n'), [])
        self.assertEqual(validate_message('Revert "feat: change"\n'), [])
        self.assertEqual(validate_message("fixup! feat(decode): change\n"), [])


if __name__ == "__main__":
    unittest.main()
