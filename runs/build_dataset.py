import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a unified anomaly-detection dataset from an explicit JSONL manifest"
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def _read_manifest(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"Manifest record {line_number} must be a mapping")
            records.append(record)
    if not records:
        raise ValueError("Manifest must contain at least one record")
    return records


def _source_path(source_dir: Path, value, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative to source_dir")
    path = (source_dir / relative).resolve()
    if source_dir not in path.parents or not path.is_file():
        raise FileNotFoundError(f"{field} not found under source_dir: {value}")
    return path


def _validate_roi(value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("roi must be [x, y, width, height]")
    x, y, width, height = value
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("roi values must be integers")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("roi must contain non-negative x/y and positive width/height")
    return x, y, width, height


def _validate_category(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every manifest record must contain clsname")
    value = value.strip()
    if value in {".", ".."} or any(char in value for char in "/\\"):
        raise ValueError("clsname must be a single path component")
    return value


def _copy_image(source: Path, destination: Path, roi, *, is_mask=False) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if roi is None:
        shutil.copy2(source, destination)
        with Image.open(source) as image:
            return image.size

    with Image.open(source) as image:
        x, y, width, height = roi
        if x + width > image.width or y + height > image.height:
            raise ValueError(f"ROI {roi} exceeds image bounds {image.size} for {source}")
        cropped = image.crop((x, y, x + width, y + height))
        if is_mask:
            cropped = cropped.point(lambda value: 255 if value else 0)
        cropped.save(destination)
        return cropped.size


def _output_relative(record: dict, source: Path, source_dir: Path) -> Path:
    split = record.get("split")
    if split not in {"train", "test"}:
        raise ValueError("Every manifest record must use split 'train' or 'test'")
    clsname = _validate_category(record.get("clsname"))
    return Path("images") / split / clsname / source.relative_to(source_dir)


def build_dataset(source_dir, manifest, output_root):
    source_dir = Path(source_dir).resolve()
    manifest = Path(manifest).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")

    source_records = _read_manifest(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    output_records = {"train": [], "test": []}
    sizes = set()

    for record in source_records:
        source = _source_path(source_dir, record.get("filename"), "filename")
        roi = _validate_roi(record.get("roi"))
        relative = _output_relative(record, source, source_dir)
        label = record.get("label")
        if isinstance(label, bool) or (label is not None and label not in {0, 1}):
            raise ValueError("label must be 0, 1, or omitted")
        if record["split"] == "train" and (label != 0 or record.get("mask")):
            raise ValueError("Training records must be normal and mask-free")
        sizes.add(_copy_image(source, output_root / relative, roi))

        output_record = {
            "filename": relative.as_posix(),
            "foreground": None,
            "mask": None,
            "clsname": _validate_category(record["clsname"]),
            "label": label,
            "label_name": record.get("label_name"),
        }
        for field, is_mask in (("foreground", True), ("mask", True)):
            if record.get(field):
                auxiliary = _source_path(source_dir, record[field], field)
                auxiliary_relative = Path("annotations") / field / relative
                auxiliary_relative = auxiliary_relative.with_suffix(auxiliary.suffix)
                _copy_image(
                    auxiliary,
                    output_root / auxiliary_relative,
                    roi,
                    is_mask=is_mask,
                )
                output_record[field] = auxiliary_relative.as_posix()
        output_records[record["split"]].append(output_record)

    for split, records in output_records.items():
        path = output_root / f"{split}_uni.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary = {
        "source_dir": str(source_dir),
        "manifest": str(manifest),
        "train": len(output_records["train"]),
        "test": len(output_records["test"]),
        "categories": sorted({record["clsname"] for record in source_records}),
        "image_sizes": [list(size) for size in sorted(sizes)],
    }
    with (output_root / "dataset_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    return summary


if __name__ == "__main__":
    args = parse_args()
    result = build_dataset(args.source_dir, args.manifest, args.output_root)
    print(json.dumps(result, indent=2, ensure_ascii=True))
