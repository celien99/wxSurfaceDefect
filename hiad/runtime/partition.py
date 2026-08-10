def round_robin_partition(items, group_count: int):
    if isinstance(group_count, bool) or not isinstance(group_count, int):
        raise TypeError("group_count must be an integer")
    if group_count <= 0:
        raise ValueError("group_count must be positive")

    groups = [[] for _ in range(group_count)]
    for index, item in enumerate(items):
        groups[index % group_count].append(item)
    return groups
