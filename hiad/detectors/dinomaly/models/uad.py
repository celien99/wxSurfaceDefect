import torch
import torch.nn as nn


class ViTill(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
    ) -> None:
        super(ViTill, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder

    def encoder_image(self, x):
        return self.encoder(x)


    def distillation(self, en_feats):
        if not en_feats:
            raise ValueError("DINOv3 encoder returned no intermediate features")

        feat_size = en_feats[0].shape[-2:]
        for idx, en in enumerate(en_feats):
            B, C, H, W = en.shape
            if (H, W) != feat_size:
                raise ValueError(
                    "Dinomaly requires DINOv3 intermediate features with a shared spatial size, "
                    f"got {feat_size} and {(H, W)}"
                )
            en_feats[idx] = en.reshape((B, C, H*W)).permute(0, 2, 1).contiguous()

        x = self.fuse_feature(en_feats)

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_feats[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        for idx, e in enumerate(en):
            en[idx] = e.permute(0, 2, 1).reshape([x.shape[0], -1, *feat_size]).contiguous()

        for idx, d in enumerate(de):
            de[idx] = d.permute(0, 2, 1).reshape([x.shape[0], -1, *feat_size]).contiguous()

        return en, de


    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)
