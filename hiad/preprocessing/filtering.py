from __future__ import annotations

from collections.abc import Sequence

from hiad.data import HRSample
from hiad.preprocessing.masks import MaskRejected


def filter_registerable_samples(
    samples: Sequence[HRSample],
    preprocessors,
    *,
    logger=None,
) -> list[HRSample]:
    """Keep samples whose foreground registration succeeds; drop failures.

    Processes one image at a time and discards pixels immediately so peak
    memory stays bounded. Raises if nothing remains.
    """
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("samples must be a non-empty sequence")
    if preprocessors is None:
        raise ValueError("preprocessors must not be None")

    kept: list[HRSample] = []
    rejected = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, HRSample):
            raise TypeError(f"Sample index {index} must be an HRSample")
        path = sample.image.image_path
        try:
            processed = preprocessors.get(sample.clsname).process_file(
                path,
                sample.clsname,
            )
            del processed
            kept.append(sample)
        except MaskRejected as error:
            rejected += 1
            if logger is not None:
                logger.warning(
                    "Dropping training sample after registration failure: "
                    f"index={index}, path={path}, clsname={sample.clsname}, "
                    f"reason={error}"
                )

    if not kept:
        raise ValueError(
            "No training samples passed foreground registration "
            f"(rejected={rejected})"
        )
    if logger is not None:
        logger.info(
            "Registration filter: "
            f"kept={len(kept)}, rejected={rejected}, input={len(samples)}"
        )
    return kept
