from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar


ItemT = TypeVar("ItemT")


def round_robin_partition(items: Iterable[ItemT], group_count: int) -> list[list[ItemT]]:
    """按输入顺序轮询分组，使任务在设备之间保持确定性分配。

    Args:
        items (Iterable[ItemT]): 需要分配的任意可迭代对象。
        group_count (int): 分组数量，必须为正整数且不能是布尔值。

    Returns:
        list[list[ItemT]]: 固定包含 ``group_count`` 个列表；第 ``i`` 个输入放入
        ``i % group_count`` 组。

    Raises:
        TypeError: ``group_count`` 不是整数或是布尔值。
        ValueError: ``group_count`` 不为正数。
    """
    if isinstance(group_count, bool) or not isinstance(group_count, int):
        raise TypeError("group_count must be an integer")
    if group_count <= 0:
        raise ValueError("group_count must be positive")

    groups: list[list[ItemT]] = [[] for _ in range(group_count)]
    for index, item in enumerate(items):
        groups[index % group_count].append(item)
    return groups
