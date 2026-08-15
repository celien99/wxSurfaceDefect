import json


def save_tasks(tasks, save_path: str):
    with open(save_path, "w", encoding="utf-8") as stream:
        json.dump(tasks, stream, indent=2)
        stream.write("\n")


def load_tasks(load_path: str):
    with open(load_path, "r", encoding="utf-8") as stream:
        return json.load(stream)
