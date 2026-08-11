from abc import ABC, abstractmethod
from typing import List
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from hiad.data import HRImageIndex


class BaseDetector(ABC):

    def __init__(self, patch_size, device, fusion_weights=None, logger=None, seed=0, **kwargs):
        r"""
           Base class for detectors. New detectors can be created by inheriting from this class.
           Args:
               patch_size (int or list): Resolution of image patches.
               device (torch.Device): Model load device.
               fusion_weights (list): Fusion weights for feature fusion. If None, a set of equal fusion weights will be used.
               logger (logging.Logger): Logger object
               seed (int): random seed.
           """

        if isinstance(patch_size, int):
            self.patch_size = [patch_size, patch_size]
        else:
            self.patch_size = patch_size
        self.seed = seed
        self.logger = logger
        self.device = device
        self.fusion_weights = fusion_weights

    @abstractmethod
    def embedding(self, input_tensor: torch.Tensor ) -> List[torch.Tensor]:
        r"""
           This method encodes image patches into feature representations (feature extraction).

           Args
               input_tensor (torch.Tensor): Input image patch tensor. Shape: (B,3,H,W)
           return:
                Returns a list of extracted features (multi-scale features). Shape: ([B,C1,H1,W1], [B,C2,H2,W2], ..., [B,Cn,Hn,Wn])
        """
        raise NotImplementedError

    @abstractmethod
    def to_device(self, device):
        raise NotImplementedError

    @abstractmethod
    def train_step(self,
                   train_dataloader: DataLoader,
                   task_name: str,
                   ) -> None:
        r"""
           This method defines the training procedure of the model.

           Args
               train_dataloader (torch.utils.data.DataLoader): DataLoader used for training.
               task_name (str): Task Name.
           return:
                None. The trainer saves the final task checkpoint after training.
        """
        raise NotImplementedError


    @abstractmethod
    def inference_step(
        self,
        test_dataloader: DataLoader,
        task_name: str,
    ) -> list:
        r"""
           Produce one pixel anomaly map for each detector input.

           Args
               test_dataloader (torch.utils.data.DataLoader): DataLoader used for testing.
               task_name (str): Task Name.

           return:
                One two-dimensional numpy anomaly map per input sample.
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self,
                        checkpoint_path: str
                        ):
        r"""
            save checkpoint
            Args
                checkpoint_path (str): Path of checkpoint
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        r"""
            load checkpoint
            Args
                checkpoint_path (str): Path of checkpoint
        """
        raise NotImplementedError

    @torch.no_grad()
    def get_multi_resolution_fusion_embeddings(self, data) -> List[torch.Tensor]:
        r"""
            This method can obtain multi-resolution fusion features for patches. It is only available when `embedding` method is defined.
            Args:
                data: The data produced by iterating over the `DataLoader`.
            return:
                Returns a list of fusion features (multi-scale features). Shape: ([B,C1,H1,W1], [B,C2,H2,W2], ..., [B,Cn,Hn,Wn]) same as 'embedding' method

        """
        image = data['image'].to(self.device, non_blocking=True)
        low_resolution_image_keys = [key for key in data if key.startswith('low_resolution_image')]

        if len(low_resolution_image_keys) == 0:
            return self.embedding(image)

        if self.fusion_weights is not None:
            if len(self.fusion_weights) != len(low_resolution_image_keys) + 1:
                raise ValueError(
                    "fusion_weights must have one value for the main image and "
                    "each low-resolution image"
                )
            fusion_weights = [weight / sum(self.fusion_weights) for weight in self.fusion_weights]
        else:
            fusion_weights = [1 / (len(low_resolution_image_keys) + 1)] * (len(low_resolution_image_keys)+1)

        low_resolution_image_keys.sort(key=lambda item: int(item.split('_')[-1]))

        # Collect all images into a single batch for one encoder forward pass
        all_images = [image]
        for key in low_resolution_image_keys:
            all_images.append(data[key].to(self.device, non_blocking=True))
        all_images = torch.cat(all_images, dim=0)
        all_embeddings = self.embedding(all_images)

        B = image.shape[0]
        main_embeddings = [feat[:B] for feat in all_embeddings]
        embeddings = [[embedding * fusion_weights[0]] for embedding in main_embeddings]

        for rs_index, low_resolution_image_key in enumerate(low_resolution_image_keys):
            low_resolution_index = data[low_resolution_image_key.replace('image', 'index')]
            start = (rs_index + 1) * B
            low_resolution_embeddings = [feat[start:start + B] for feat in all_embeddings]

            for i, low_resolution_embedding in enumerate(low_resolution_embeddings):
                downsampling_embedding = []
                for feature, index in zip(low_resolution_embedding, low_resolution_index):
                    feature_stride_H, feature_stride_W = self.patch_size[1] / feature.shape[1], self.patch_size[0] / feature.shape[2]
                    index = HRImageIndex.from_str(index)
                    x_start = index.x / feature_stride_W
                    y_start = index.y / feature_stride_H
                    x_end = x_start + index.width / feature_stride_W
                    y_end = y_start + index.height / feature_stride_H
                    downsampling_embedding.append(feature[:, int(y_start) : int(y_end), int(x_start) : int(x_end)])
                try:
                    downsampling_embedding = torch.stack(downsampling_embedding)
                except RuntimeError:
                    first_embedding = downsampling_embedding[0]
                    downsampling_embedding = [F.interpolate(feat.unsqueeze(0), size=(first_embedding.shape[-2], first_embedding.shape[-1]), mode='bilinear').squeeze(0)
                                              for feat in downsampling_embedding[1:]]
                    downsampling_embedding = [first_embedding] + downsampling_embedding
                    downsampling_embedding = torch.stack(downsampling_embedding)

                embeddings[i].append(fusion_weights[rs_index+1] * F.interpolate(
                        downsampling_embedding,
                        size=(embeddings[i][-1].shape[-2], embeddings[i][-1].shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                        ))

        embeddings = [torch.sum(torch.stack(embedding), dim=0, keepdim=False) for embedding in embeddings]
        return embeddings
