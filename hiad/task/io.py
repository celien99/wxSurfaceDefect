import json

from hiad.checkpoints import atomic_write_json

from .task import validate_tasks


def save_tasks(tasks, save_path: str):
    atomic_write_json({"tasks": validate_tasks(tasks)}, save_path)


def load_tasks(load_path: str):
    with open(load_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or set(payload) != {"tasks"}:
        raise ValueError("Task configuration has an invalid schema")
    return validate_tasks(payload["tasks"])
