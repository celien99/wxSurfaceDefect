def validate_gpu_ids(gpu_ids) -> list[int]:
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise ValueError("gpu_ids must be a non-empty list")
    if any(
        isinstance(gpu_id, bool)
        or not isinstance(gpu_id, int)
        or gpu_id < 0
        for gpu_id in gpu_ids
    ):
        raise ValueError("gpu_ids must contain non-negative integers")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must not contain duplicates")
    return list(gpu_ids)
