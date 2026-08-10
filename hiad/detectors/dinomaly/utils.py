import torch
import numpy as np
from functools import partial


def modify_grad(grad, inds, factor=0.):
    inds = inds.expand_as(grad)
    return torch.where(inds, grad * factor, grad)


def global_cosine_hm_percent(a, b, p=0.9, factor=0.):
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if not 0 <= factor <= 1:
        raise ValueError(f"factor must be in [0, 1], got {factor}")
    if len(a) != len(b):
        raise ValueError(f"Feature list lengths must match, got {len(a)} and {len(b)}")
    if not a:
        raise ValueError("Feature lists must not be empty")

    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        easy_count = int(point_dist.numel() * p)
        if easy_count > 0:
            flat_mask = torch.zeros(point_dist.numel(), dtype=torch.bool, device=point_dist.device)
            if easy_count == point_dist.numel():
                flat_mask.fill_(True)
            else:
                easy_indices = torch.argsort(point_dist.reshape(-1), stable=True)[:easy_count]
                flat_mask[easy_indices] = True
            easy_mask = flat_mask.reshape_as(point_dist)
            partial_func = partial(modify_grad, inds=easy_mask, factor=factor)
            b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


from torch.optim.lr_scheduler import _LRScheduler


class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]
