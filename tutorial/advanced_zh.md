# 原图分辨率高分辨率设置

HiAD 固定使用 DINOv3 Dinomaly。源图按照原始宽高加载，只有局部模型窗口是固定尺寸。

## 训练与全局校准

`HRTrainer.train` 接收完整的 `train_samples` 和 `gpu_ids`。所有正常训练样本都参与 Dinomaly 权重更新，训练结束后同一完整集合建立一个全局正常 ECDF 校准。当前不固定划出验证集；独立验证数据后续补充即可。

## 动态切块

```python
from hiad.task import DynamicTaskGenerator

tasks = DynamicTaskGenerator(
    patch_size=512,
    stride=384,
    ds_factors=[0, 1],
).create_tasks(thumbnail_size=512)
```

由于 DINOv3 patch embedding 为 16 像素，`patch_size=512` 对应 32x32 token 网格。`stride=384` 产生重叠窗口，重建每张原图尺寸异常图时会对重叠位置取逐像素最大值，避免稀释小缺陷分数。小于窗口的图像只在模型输入侧填充，聚合后裁回原始尺寸。

## 多尺度特征融合

`ds_factors=[0, 1]` 将保留细节的 512x512 主视图与 1024x1024 上下文视图结合。上下文视图缩放到局部模型窗口，经 DINOv3 编码并与主视图空间对齐后完成特征融合。每张源图都会独立生成这些窗口。

## 动态 Task

局部切块 checkpoint 由唯一的 `dynamic_patch` task 管理。task 只保存切块策略，不保存任何固定整图坐标网格，因此同一 checkpoint 可以处理不同宽高的图像。

## 缩略图分支

可选 thumbnail 分支继续使用相同的 DINOv3 Dinomaly 实现并保留独立 checkpoint。向 `create_tasks()` 传入 `thumbnail_size` 即启用该任务；不传则只运行动态切块分支。thumbnail 提供全局上下文，动态切块分支负责保留原始缺陷像素。
