from typing import List

import numpy
import torch
from torch.utils.data import Dataset

from hiad.data import LRPatch


class PatchDataset(Dataset):

    def __init__(
        self,
        patches: List[LRPatch],
        training: bool,
        task_name: str,
    ):

        super().__init__()
        self.patches = patches
        self.training = training

    @staticmethod
    def _image_to_tensor(image, name):
        if (
            not isinstance(image, numpy.ndarray)
            or image.dtype != numpy.float32
            or image.ndim != 3
            or image.shape[0] <= 0
            or image.shape[1] <= 0
            or image.shape[2] != 3
            or not numpy.isfinite(image).all()
        ):
            raise ValueError(f"{name} must be finite non-empty HWC float32 RGB")
        return torch.from_numpy(image).permute(2, 0, 1)

    def __getitem__(self, idx):

        patch = self.patches[idx]
        image = self._image_to_tensor(patch.image, "patch.image")

        if patch.mask is not None:
            mask_array = numpy.asarray(patch.mask)
            if mask_array.ndim != 2 or mask_array.shape != tuple(image.shape[1:]):
                raise ValueError("patch.mask must match the patch image height and width")
            mask = torch.from_numpy(mask_array).ne(0).to(torch.float32)
        else:
            mask = torch.zeros(image.shape[1:], dtype=torch.float32)

        item = {
            "image": image,
            "mask": mask,
        }

        if patch.clsname is not None:
            item.update({"clsname": patch.clsname})

        if patch.label is not None:
            item.update({"label": patch.label})

        if patch.low_resolution_images is not None:
            for i, (low_resolution_image, low_resolution_index) in enumerate(
                zip(patch.low_resolution_images, patch.low_resolution_indexes)
            ):
                item[f'low_resolution_image_{i}'] = self._image_to_tensor(
                    low_resolution_image,
                    f"patch.low_resolution_images[{i}]",
                )
                item[f'low_resolution_index_{i}'] = str(low_resolution_index)
        return item

    def __len__(self):
        return len(self.patches)
