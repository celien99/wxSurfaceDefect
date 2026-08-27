from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from hiad.utils.fusion import CarbonFiberStripeFusion, FusionConfig


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.ndim != 2:
        raise ValueError(f"Expected a single-channel image: {path}")
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse 8 programmed stripe images into one map")
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_images = [_read_gray(args.input_dir / f"x{index}.png") for index in range(4)]
    y_images = [_read_gray(args.input_dir / f"y{index}.png") for index in range(4)]
    fusion = CarbonFiberStripeFusion(FusionConfig())
    result = fusion.analyze(x_images, y_images)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fusion.png": result.fused,
        "shape.png": np.clip(result.shape * 255.0, 0, 255).astype(np.uint8),
        "scratch.png": np.clip(result.scratch * 255.0, 0, 255).astype(np.uint8),
        "dark.png": np.clip(result.dark * 255.0, 0, 255).astype(np.uint8),
        "texture.png": np.clip(result.texture * 255.0, 0, 255).astype(np.uint8),
        "confidence.png": np.clip(result.phase_confidence * 255.0, 0, 255).astype(np.uint8),
    }
    for name, image in outputs.items():
        output_path = args.output_dir / name
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Failed to write image: {output_path}")
    print(f"8 → 1 fusion written to {args.output_dir / 'fusion.png'}")


if __name__ == "__main__":
    main()
