from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from hiad.data import LRPatch


class PatchDataset(Dataset):
    """Convert raw RGB patches to normalized model tensors."""

    def __init__(self, patches: List[LRPatch], training: bool, task_name: str):
        super().__init__()
        self.patches = patches
        self.training = training
        self.task_name = task_name
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def _image_to_tensor(self, image, name):
        image = np.asarray(image)
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] <= 0
            or image.shape[1] <= 0
            or image.dtype not in (np.uint8, np.float32)
            or not np.isfinite(image).all()
        ):
            raise ValueError(f"{name} must be a finite HWC RGB image")
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        if image.dtype == np.uint8:
            tensor = tensor / 255.0
        return (tensor - self.mean) / self.std

    def __getitem__(self, idx):
        patch = self.patches[idx]
        image = self._image_to_tensor(patch.image, "patch.image")

        if patch.mask is not None:
            mask_array = np.asarray(patch.mask)
            if mask_array.ndim != 2 or mask_array.shape != tuple(image.shape[1:]):
                raise ValueError("patch.mask must match the patch image height and width")
            mask = torch.from_numpy(mask_array).ne(0).to(torch.float32)
        else:
            mask = torch.zeros(image.shape[1:], dtype=torch.float32)

        item = {"image": image, "mask": mask}
        if patch.clsname is not None:
            item["clsname"] = patch.clsname
        if patch.label is not None:
            item["label"] = patch.label
        if patch.low_resolution_images is not None:
            for i, (low_image, low_index) in enumerate(
                zip(patch.low_resolution_images, patch.low_resolution_indexes)
            ):
                item[f"low_resolution_image_{i}"] = self._image_to_tensor(
                    low_image,
                    f"patch.low_resolution_images[{i}]",
                )
                item[f"low_resolution_index_{i}"] = str(low_index)
        return item

    def __len__(self):
        return len(self.patches)
