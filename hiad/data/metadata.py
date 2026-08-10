import json
from collections.abc import Mapping
from pathlib import Path


def read_jsonl_records(path) -> list[dict]:
    metadata_path = Path(path)
    records = []
    with metadata_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(
                    f"Blank metadata record at {metadata_path}:{line_number}"
                )
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"Metadata record at {metadata_path}:{line_number} "
                    "must be a mapping"
                )
            records.append(dict(record))
    return records
