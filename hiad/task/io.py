import json

from .task import validate_tasks


def save_tasks(tasks, save_path: str):
    with open(save_path, "w", encoding="utf-8") as stream:
        json.dump(validate_tasks(tasks), stream, indent=2)
        stream.write("\n")


def load_tasks(load_path: str):
    with open(load_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list):
        raise ValueError("Task configuration has an invalid schema")
    return validate_tasks(payload)
