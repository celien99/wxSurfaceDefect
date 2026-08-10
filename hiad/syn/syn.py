import copy
import random
from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import List

from hiad.data import LRPatch


class _PatchAttemptsExhausted(RuntimeError):
    pass


class BaseAnomalySynthesizer(ABC):

    @abstractmethod
    def anomaly_synthesize(self, sample: LRPatch, **kwargs) -> LRPatch:
        raise NotImplementedError

    def copy_low_resolution_images(self, dst_sample: LRPatch, src_sample: LRPatch):
        if src_sample.low_resolution_images is None:
            return dst_sample
        for low_resolution_image, low_resolution_index in zip(
            src_sample.low_resolution_images,
            src_sample.low_resolution_indexes,
        ):
            new_low_resolution_image = low_resolution_image.copy()
            y_start = low_resolution_index.y
            y_end = y_start + low_resolution_index.height
            x_start = low_resolution_index.x
            x_end = x_start + low_resolution_index.width
            new_low_resolution_image[y_start:y_end, x_start:x_end, :] = cv2.resize(
                dst_sample.image,
                (low_resolution_index.width, low_resolution_index.height),
            )

            if dst_sample.low_resolution_indexes is None:
                dst_sample.low_resolution_indexes = []
            if dst_sample.low_resolution_images is None:
                dst_sample.low_resolution_images = []

            dst_sample.low_resolution_images.append(new_low_resolution_image)
            dst_sample.low_resolution_indexes.append(copy.copy(low_resolution_index))
        return dst_sample


class RandomeBoxSynthesizer(BaseAnomalySynthesizer):

    def __init__(self,
                 p: float,
                 max_patch_num: int = None,
                 anomaly_sizes: List = None,
                 diff_threshold: List = None,
                 input_scale: float = 255.0,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):

        if (
            isinstance(p, bool)
            or not isinstance(p, (int, float, np.integer, np.floating))
            or not np.isfinite(p)
            or p < 0
            or p > 1
        ):
            raise ValueError("p must be finite and in [0, 1]")
        self.p = float(p)

        self.max_patch_num = 5 if max_patch_num is None else max_patch_num
        if (
            isinstance(self.max_patch_num, bool)
            or not isinstance(self.max_patch_num, int)
            or self.max_patch_num <= 0
        ):
            raise ValueError("max_patch_num must be a positive integer")

        anomaly_sizes = (
            [[16, 40], [80, 100]] if anomaly_sizes is None else anomaly_sizes
        )
        anomaly_size_array = np.asarray(anomaly_sizes, dtype=np.float64)
        if anomaly_size_array.ndim == 1:
            anomaly_size_array = anomaly_size_array.reshape(1, -1)
        if (
            anomaly_size_array.ndim != 2
            or anomaly_size_array.shape[1] != 2
            or not np.isfinite(anomaly_size_array).all()
            or np.any(anomaly_size_array <= 0)
            or np.any(anomaly_size_array[:, 1] < anomaly_size_array[:, 0])
        ):
            raise ValueError("anomaly_sizes must contain positive finite [min, max] pairs")
        self.anomaly_sizes = anomaly_size_array.tolist()

        threshold_array = np.asarray(
            [20, 60] if diff_threshold is None else diff_threshold,
            dtype=np.float64,
        )
        if (
            threshold_array.shape != (2,)
            or not np.isfinite(threshold_array).all()
            or np.any(threshold_array < 0)
            or threshold_array[1] < threshold_array[0]
        ):
            raise ValueError("diff_threshold must be a nonnegative finite [min, max] pair")
        self.diff_threshold = threshold_array.tolist()

        if (
            isinstance(input_scale, bool)
            or not isinstance(input_scale, (int, float, np.integer, np.floating))
            or not np.isfinite(input_scale)
            or input_scale <= 0
        ):
            raise ValueError("input_scale must be finite and positive")
        self.input_scale = float(input_scale)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (3,) or not np.isfinite(self.mean).all():
            raise ValueError("mean must contain three finite values")
        if (
            self.std.shape != (3,)
            or not np.isfinite(self.std).all()
            or np.any(self.std <= 0)
        ):
            raise ValueError("std must contain three finite positive values")

    def anomaly_synthesize(self, sample: LRPatch, **kwargs) -> LRPatch:
        if sample.label is not None and sample.label != 0:
            raise ValueError("RandomeBoxSynthesizer accepts only normal samples")
        if (
            not isinstance(sample.image, np.ndarray)
            or sample.image.dtype != np.float32
            or sample.image.ndim != 3
            or sample.image.shape[0] <= 0
            or sample.image.shape[1] <= 0
            or sample.image.shape[2] != 3
            or not np.isfinite(sample.image).all()
        ):
            raise ValueError("sample.image must be finite non-empty HWC float32 RGB")

        if random.random() < self.p:
            patchex = sample.image.copy()
            mask = np.zeros(sample.image.shape[:2], dtype=np.uint8)
            valid_source_hw = sample.valid_source_hw or sample.image.shape[:2]
            valid_height, valid_width = valid_source_hw
            if (
                valid_height <= 0
                or valid_width <= 0
                or valid_height > sample.image.shape[0]
                or valid_width > sample.image.shape[1]
            ):
                raise ValueError("valid_source_hw must fit within sample.image")
            synthesized_regions = 0

            for i in range(self.max_patch_num):
                if i == 0 or np.random.randint(2) > 0:  # at least one patch
                    try:
                        patchex, (
                            (y_start, y_end),
                            (x_start, x_end),
                        ), patch_mask = self._patch_ex(
                            patchex,
                            valid_source_hw,
                        )
                    except _PatchAttemptsExhausted:
                        continue
                    mask[y_start:y_end, x_start:x_end] = patch_mask[..., 0]
                    synthesized_regions += 1

            if synthesized_regions == 0:
                normal_sample = copy.copy(sample)
                if normal_sample.label is None:
                    normal_sample.label = 0
                return normal_sample
            if patchex.dtype != np.float32 or not np.isfinite(patchex).all():
                raise ValueError("Synthesized image must remain finite float32")

            anomaly_sample = LRPatch(image=patchex,
                                     mask=mask,
                                     label=1,
                                     label_name=sample.label_name,
                                     clsname=sample.clsname,
                                     main_index=sample.main_index,
                                     valid_source_hw=sample.valid_source_hw)

            return self.copy_low_resolution_images(anomaly_sample, sample)
        normal_sample = copy.copy(sample)
        if normal_sample.label is None:
            normal_sample.label = 0
        return normal_sample


    def _patch_ex(self, ima_dest, valid_source_hw=None):
        anomaly_sizes = random.choice(self.anomaly_sizes)
        valid_height, valid_width = valid_source_hw or ima_dest.shape[:2]
        height = min(
            max(1, int(random.uniform(*anomaly_sizes))),
            valid_height,
        )
        width = min(
            max(1, int(random.uniform(*anomaly_sizes))),
            valid_width,
        )

        raw_color_max = max(0, int(self.input_scale))
        for _ in range(50):
            raw_color = random.randint(0, raw_color_max)
            normalized_color = (
                (raw_color / self.input_scale - self.mean) / self.std
            ).astype(np.float32)
            y_start = np.random.randint(0, valid_height - height + 1)
            x_start = np.random.randint(0, valid_width - width + 1)
            y_end = y_start + height
            x_end = x_start + width
            region = ima_dest[y_start:y_end, x_start:x_end]
            difference = float(
                np.mean(
                    np.abs(region - normalized_color)
                    * self.std
                    * self.input_scale
                )
            )
            if self.diff_threshold[0] <= difference <= self.diff_threshold[1]:
                ima_dest[y_start:y_end, x_start:x_end] = normalized_color
                patch_mask = np.ones((height, width, 1), dtype=np.uint8)
                return ima_dest, ((y_start, y_end), (x_start, x_end)), patch_mask
        raise _PatchAttemptsExhausted("no candidate color passed diff_threshold")
