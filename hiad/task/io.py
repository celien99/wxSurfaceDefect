from __future__ import annotations

import json
import os
from collections.abc import Sequence

from .contracts import TaskDefinition
from .task import validate_tasks


def save_tasks(
    tasks: Sequence[TaskDefinition],
    save_path: str | os.PathLike[str],
) -> None:
    """按稳定 JSON 契约持久化任务定义。

    Args:
        tasks (Sequence[TaskDefinition]): 已生成并验证的任务序列。
        save_path (str | os.PathLike[str]): 输出 JSON 文件路径。

    Raises:
        OSError: 文件无法创建或写入。
        TypeError: 任务包含不能 JSON 序列化的值。
    """
    with open(save_path, "w", encoding="utf-8") as stream:
        json.dump(tasks, stream, indent=2)
        stream.write("\n")


def load_tasks(load_path: str | os.PathLike[str]) -> list[TaskDefinition]:
    """读取并校验任务定义，避免把未验证 JSON 伪装成静态契约。

    Args:
        load_path (str | os.PathLike[str]): UTF-8 编码的任务 JSON 路径。

    Returns:
        list[TaskDefinition]: 深拷贝且通过三任务生产约束校验的定义。

    Raises:
        OSError: 文件不存在或不可读。
        json.JSONDecodeError: 文件不是合法 JSON。
        TypeError: 任务条目的顶层类型错误。
        ValueError: 任务数量、类型或字段值不符合生产约束。
    """
    with open(load_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return validate_tasks(payload)
