# Streaming Train + No-SAM Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 彻底重写训练/推理数据管线：去掉 SAM，DINO 配准后 warp 参考 mask，按需流式读图，禁止全量内存驻留。

**Architecture:** 预处理只保留冻结 DINO 配准与 mask warp；新增 `StreamingTaskDataset` 在 `__getitem__` 内读图→预处理→切块；`HRTrainer`/`HRInferencer` 主进程只持有元数据；generation 产物去掉全部 SAM 字段。不兼容旧 checkpoint。

**Tech Stack:** Python 3.10+、PyTorch、timm DINOv3、OpenCV、NumPy、pytest

**Spec:** [`docs/specs/2026-08-10-stream-train-no-sam-design.md`](../specs/2026-08-10-stream-train-no-sam-design.md)

## Global Constraints

- 禁止训练/校准/推理路径上的全量 `share_memory_()` 像素驻留
- 禁止 SAM2 / `transformers` Sam2 依赖
- 配准失败必须抛错，禁止 `full_image` fallback
- `preprocessing.schema_version` bump 到 `3`；旧 generation 加载失败即可
- 每个 `clsname` 仍只需 `foreground_references.yaml` 中一张参考图 + 一张二值 mask
- `working_longest_edge` 改为 preprocessing 配置项（不再从 SAM processor 推导）；默认 `1024`
- 计划文件路径：`docs/plans/`（`docs/superpowers/` 被仓库 gitignore，正确）
- 默认 DataLoader `num_workers=0`

---

## File Map

| 路径 | 职责 |
|------|------|
| `hiad/preprocessing/sam.py` | **删除** |
| `hiad/preprocessing/registration.py` | 只输出 `warped_mask` + DINO metrics |
| `hiad/preprocessing/masks.py` | 仅校验/清理 warped mask（无 SAM IoU） |
| `hiad/preprocessing/config.py` / `constants.py` | 去 SAM 键；加 `working_longest_edge`；schema=3 |
| `hiad/preprocessing/calibration.py` | 校准不加载 SAM；用 config 的 `working_longest_edge` |
| `hiad/preprocessing/runtime.py` | 单图：DINO warp → 归一化；无 SAM |
| `hiad/preprocessing/artifacts.py` | manifest 无 sam2 块 |
| `hiad/data/preparation.py` | 删除全量 `prepared_source_session`；保留纯几何切块工具或迁出 |
| `hiad/datasets/streaming_dataset.py` | **新建** 流式 task Dataset |
| `hiad/trainer/worker.py` | 流式 Dataset 训练 |
| `hiad/trainer/trainer.py` | 主进程不打开像素；流式校准 |
| `hiad/trainer/checkpoint_evidence.py` | 流式收集 evidence |
| `hiad/inferencer/inferencer.py` | 按图流式预处理 |
| `hiad/runtime/evidence.py` | 适配流式输入（若接口变化） |
| `configs/dinomaly.yaml` | 去 sam2；加 `working_longest_edge` |
| `pyproject.toml` | 移除 `transformers`（仅被 SAM 使用） |
| `hiad/README.md` | 同步文档 |
| `tests/preprocessing/...` / `tests/data/...` | 新建测试 |

---

### Task 1: 去 SAM 的配准与 mask 校验（纯函数层）

**Files:**
- Modify: `hiad/preprocessing/registration.py`
- Modify: `hiad/preprocessing/masks.py`
- Delete: `hiad/preprocessing/sam.py`
- Create: `tests/preprocessing/test_registration_warp.py`
- Create: `tests/preprocessing/test_masks_warped.py`

**Interfaces:**
- Produces:
  - `register_and_warp_mask(rgb, *, encoder, prototypes, template, reference_mask, config, device) -> tuple[np.ndarray, dict[str, Any]]`  
    返回 `(warped_mask: bool HxW, metrics)`；失败抛 `MaskRejected`
  - `validate_warped_mask(warped_mask, reference_mask, config) -> tuple[np.ndarray, dict[str, float]]`  
    对 warped mask 做孔洞填充/边界膨胀（复用现有形态学逻辑，输入仅 warped，不再接收 sam_mask）；校验 `min_reference_coverage`（相对参考 mask 面积比）与 `max_area_ratio_deviation`；失败抛 `MaskRejected`
- Consumes: 现有 `extract_dino_grid`、`MaskRejected`、`template`/`prototypes` 结构

- [ ] **Step 1: 写失败测试（warp 几何）**

```python
# tests/preprocessing/test_registration_warp.py
import numpy as np
import pytest
from hiad.preprocessing.masks import MaskRejected
from hiad.preprocessing.registration import register_and_warp_mask

def test_register_and_warp_mask_rejects_empty_matches(monkeypatch):
    # 构造最小 stub：encoder/prototypes/template 使 match_count < min_dino_matches
    # 断言抛 MaskRejected，且 str(exc) 含 insufficient 或 reason
    ...
```

- [ ] **Step 2: 写失败测试（warped mask 校验）**

```python
# tests/preprocessing/test_masks_warped.py
import numpy as np
import pytest
from hiad.preprocessing.masks import validate_warped_mask, MaskRejected

def test_validate_warped_mask_rejects_empty():
    empty = np.zeros((32, 32), dtype=bool)
    ref = np.ones((32, 32), dtype=bool)
    config = {
        "boundary_expand_ratio": 0.0,
        "min_reference_coverage": 1.0,
        "max_area_ratio_deviation": 0.35,
    }
    with pytest.raises(MaskRejected):
        validate_warped_mask(empty, ref, config)
```

- [ ] **Step 3: 运行确认失败**

Run: `pytest tests/preprocessing/test_masks_warped.py tests/preprocessing/test_registration_warp.py -v`  
Expected: FAIL（符号不存在或旧 API）

- [ ] **Step 4: 实现**

1. 将 `generate_registration_prompts` 重写/重命名为 `register_and_warp_mask`：保留互匹配 + `estimateAffinePartial2D` + `cv2.warpAffine`；**删除** box / positive_points / `MAX_SAM_POSITIVE_POINTS` 逻辑；成功返回 `(warped_prior.astype(bool), metrics)`。
2. 用 `validate_warped_mask` 替换 `validate_and_clean_mask` / `clean_foreground_mask` 的双输入 SAM 路径：只清理 warped mask；删除 `min_sam_prior_iou` / `sam_prior_iou` 指标。
3. 删除 `hiad/preprocessing/sam.py`。
4. 全库 grep：`generate_registration_prompts|validate_and_clean_mask|sam2|from .sam|from hiad.preprocessing.sam` → 清零。

- [ ] **Step 5: 跑测试通过**

Run: `pytest tests/preprocessing/ -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add hiad/preprocessing/registration.py hiad/preprocessing/masks.py tests/preprocessing/
git rm hiad/preprocessing/sam.py
git commit -m "refactor: replace SAM prompts with DINO mask warp"
```

---

### Task 2: Preprocessing 配置、校准与 runtime（无 SAM）

**Files:**
- Modify: `hiad/preprocessing/constants.py`
- Modify: `hiad/preprocessing/config.py`
- Modify: `hiad/preprocessing/calibration.py`
- Modify: `hiad/preprocessing/runtime.py`
- Modify: `hiad/preprocessing/artifacts.py`
- Modify: `configs/dinomaly.yaml`
- Modify: `pyproject.toml`（移除 `transformers`）
- Create: `tests/preprocessing/test_config_schema_v3.py`

**Interfaces:**
- Produces:
  - `PREPROCESSING_SCHEMA_VERSION = 3`
  - `canonicalize_preprocessing_config` 要求键含 `working_longest_edge: int > 0`，且为 DINO patch size 的倍数；**不含**任何 `sam2_*` / `min_sam_prior_iou`
  - `ForegroundPreprocessor._effective_mask` → `register_and_warp_mask` + `validate_warped_mask`；失败向上抛 `MaskRejected`（**无** full_image fallback）
  - `calibrate_preprocessing_registry` 只用 DINO；`working_longest_edge` 来自 config；manifest 无 `sam2` 键
- Consumes: Task 1 的 `register_and_warp_mask` / `validate_warped_mask`

- [ ] **Step 1: 写配置测试**

```python
def test_canonicalize_rejects_sam_keys():
    raw = {..., "sam2_model_id": "x"}  # 完整合法 v3 字段 + 多余 sam 键
    with pytest.raises(ValueError):
        canonicalize_preprocessing_config(raw)

def test_canonicalize_requires_working_longest_edge():
    raw = {完整 v3 字段但不含 working_longest_edge}
    with pytest.raises((KeyError, ValueError)):
        canonicalize_preprocessing_config(raw)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/preprocessing/test_config_schema_v3.py -v`

- [ ] **Step 3: 实现配置与产物**

1. `CONFIG_KEYS` 去掉所有 sam 相关；加入 `working_longest_edge`。
2. `configs/dinomaly.yaml` 同步；`schema_version: 3`；`working_longest_edge: 1024`。
3. `calibration.py`：删除 `load_sam2_components`；用 `canonical_config["working_longest_edge"]`；manifest 只写 `dino` + reference hashes。
4. `artifacts.py`：校验 manifest **不得**含 `sam2`；schema 3。
5. `runtime.py`：删除 `_ensure_sam2` 与 transformers 导入；`_effective_mask` 用 Task 1 API；去掉 fallback 分支（`except MaskRejected` 不再返回全 1 mask，直接 raise）。
6. `pyproject.toml` 删除 `transformers` 依赖行。

- [ ] **Step 4: 跑测试**

Run: `pytest tests/preprocessing/ -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hiad/preprocessing/ configs/dinomaly.yaml pyproject.toml tests/preprocessing/
git commit -m "feat: preprocessing schema v3 without SAM"
```

---

### Task 3: StreamingTaskDataset

**Files:**
- Create: `hiad/datasets/streaming_dataset.py`
- Modify: `hiad/datasets/__init__.py`（若无则创建）
- Modify: `hiad/data/preparation.py` — 删除 `prepare_source_samples` / `prepared_source_session` / `close_prepared_sources`；将 `build_task_inputs_from_open_samples` 改为仅测试辅助或删除
- Create: `tests/data/test_streaming_dataset.py`

**Interfaces:**
- Produces:

```python
class StreamingTaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list[HRSample],
        task: dict,
        preprocessor_registry: ForegroundPreprocessorRegistry,
        *,
        training: bool,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict:  # 与 PatchDataset 相同的 tensor 字典约定
        ...
```

- 索引构建：对每个 sample 只用 `PIL.Image.open(path).size`（或等价）取宽高，**不**跑预处理；用现有 `split_multiresolution_regions` / thumbnail 规则生成 `(sample_index, region_or_None)` 列表。
- `__getitem__`：`registry.get(clsname).process_file(path)` → 得到归一化全图 ndarray → 按 index 切 patch / 缩略图 → 转成与现 `PatchDataset.__getitem__` 相同字段（`image`, `mask`, 可选 `low_resolution_image_*`）。处理完不把全图缓存在 Dataset 实例上。
- Consumes: Task 2 的 `ForegroundPreprocessorRegistry` / `process_file`

- [ ] **Step 1: 写测试（假 preprocessor）**

```python
class FakeRegistry:
    def get(self, clsname):
        return self
    def process_file(self, path, category=None):
        # 返回固定 float32 HxWx3 图
        ...

def test_streaming_dataset_len_matches_regions(tmp_path):
    # 写两张小 PNG；task dynamic_patch patch_size=16 stride=16
    # FakeRegistry 返回已知尺寸
    # assert len(ds) == 期望 patch 数
    # item = ds[0]; assert item["image"].shape[0] == 3
```

- [ ] **Step 2: 跑测试确认失败 → 实现 → 再跑通过**

- [ ] **Step 3: 删除全量驻留 API**

从 `preparation.py` 与所有 import 处移除 `prepared_source_session` 等；`hiad` 内 grep 清零。

- [ ] **Step 4: Commit**

```bash
git add hiad/datasets/streaming_dataset.py hiad/data/preparation.py tests/data/
git commit -m "feat: add StreamingTaskDataset and drop full-image residency"
```

---

### Task 4: Trainer / Worker 流式训练

**Files:**
- Modify: `hiad/trainer/worker.py`
- Modify: `hiad/trainer/trainer.py`
- Modify: `hiad/detectors/base.py`（若 `create_dataset` 仍假定预切 patches — 改为可直接吃 StreamingTaskDataset，或 worker 绕过 `create_dataset`）
- Create: `tests/trainer/test_worker_uses_streaming.py`（可用 mock detector，不必真 GPU）

**Interfaces:**
- Produces:
  - `train_tasks_in_device(..., train_samples, ..., generation_root, ...)`：在 worker 内  
    `ForegroundPreprocessorRegistry.from_checkpoint(generation_root, device, ...)` → 对每个 task 建 `StreamingTaskDataset` → `DataLoader(..., num_workers=0, shuffle=True, batch_size=...)` → `detector.train_step` → `save_checkpoint` → `release` preprocessor GPU
  - `HRTrainer.train`：**不再**调用 `prepared_source_session`；校准前也不在主进程打开图像
- Consumes: Task 2 registry、Task 3 dataset

- [ ] **Step 1: 改 `worker.py`**

伪代码目标：

```python
preprocessors = ForegroundPreprocessorRegistry.from_checkpoint(...)
try:
    for task in tasks:
        dataset = StreamingTaskDataset(train_samples, task, preprocessors, training=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, ...)
        detector = detector_class(...)
        detector.train_step(loader, task_name)
        detector.save_checkpoint(...)
        ...
finally:
    preprocessors.close()
```

注意：传给子进程的 `train_samples` 必须是**未打开**的 `HRSample`（仅路径元数据），保证可 pickle。

- [ ] **Step 2: 改 `trainer.py`**

去掉：

```python
with prepared_source_session(training_samples, preprocessors):
```

改为：先 `calibrate_preprocessing_registry` → 多进程 `train_tasks_in_device` → 再流式 `collect_checkpoint_evidence`（Task 5）→ `publish_generation`。主进程校准预处理后可 `release_gpu`，不必持有像素。

- [ ] **Step 3: 单元测试 mock**

断言 worker 构造了 `StreamingTaskDataset`（patch `StreamingTaskDataset` / DataLoader），且未调用 `share_memory_`。

- [ ] **Step 4: Commit**

```bash
git add hiad/trainer/ hiad/detectors/base.py tests/trainer/
git commit -m "feat: stream training data per GPU worker"
```

---

### Task 5: 流式校准 evidence + Inferencer

**Files:**
- Modify: `hiad/trainer/checkpoint_evidence.py`
- Modify: `hiad/runtime/evidence.py`（按需）
- Modify: `hiad/inferencer/inferencer.py`
- Create: `tests/inferencer/test_inference_no_full_residency.py`

**Interfaces:**
- Produces:
  - `collect_checkpoint_evidence`：各 worker 内用 `StreamingTaskDataset(training=False)` + 现有 detector predict 路径收集 evidence；**不**依赖已 open 的样本
  - `HRInferencer`：删除 `_prepare_samples` 中整集 open + `share_memory_`；改为按 batch 对路径调用 `preprocessor.process_file`，推理后 `close` 该样本图像
- Consumes: Task 3 dataset、Task 2 preprocessor

- [ ] **Step 1: 重写 evidence 收集**

`_collect_checkpoint_evidence_in_device` 内：

```python
preprocessors = ForegroundPreprocessorRegistry.from_checkpoint(...)
task_inputs 不再从 open samples 构建；
对每个 task:
    dataset = StreamingTaskDataset(samples, task, preprocessors, training=False)
    # 需要 records 元数据（source_xywh 等）时：在 StreamingTaskDataset 暴露
    #   .records: list[PreparedInputRecord] 与 patches 对齐，索引阶段即可生成（只需 size，无需预处理）
```

将 `PreparedInputRecord` 列表在 Dataset 索引阶段填好（只依赖文件尺寸），evidence 合并逻辑继续按 `image_path` 聚合。

- [ ] **Step 2: 重写 Inferencer 准备路径**

按 GPU 分片样本列表；每个样本：`process_file` → 构建该样本的 task patches（可抽 `build_task_inputs_for_single_sample`）→ predict → 立即释放图像。禁止循环末尾仍持有全部 `share_memory_` 张量。

- [ ] **Step 3: 测试**

- mock 下调用 inferencer 准备逻辑，断言不存在 `share_memory_` 调用（monkeypatch `HRImage.share_memory_` 计数为 0）。
- evidence 收集对 2 张假图返回可合并结构。

- [ ] **Step 4: Commit**

```bash
git add hiad/trainer/checkpoint_evidence.py hiad/runtime/evidence.py hiad/inferencer/inferencer.py tests/
git commit -m "feat: stream calibration and inference without shared residency"
```

---

### Task 6: 产物清单、文档与清理

**Files:**
- Modify: `hiad/trainer/trainer.py`（`required_files` 确认与 spec 一致，无 SAM 文件）
- Modify: `hiad/README.md`
- Modify: `tutorial/advanced_zh.md`（若提及 SAM）
- Grep 全库清理：`sam2|SAM2|prepared_source_session|share_memory_|full_image|transformers`

**Interfaces:**
- Produces: README 描述与 spec 一致的训练流与产物树；`required_files` 仅：

```text
tasks.json
preprocessing_registry.json
multirisk_calibration.json
{dynamic_patch,thumbnail}_weight.pt
preprocessing/<clsname>/{preprocessing.yaml,preprocessing_manifest.json,
  foreground_prototypes.pt,reference_feature_template.pt,reference_foreground.rle}
```

- [ ] **Step 1: 更新 README「准备前景参考 / 训练流程 / 重要约束」**，删除 SAM2 安装与 full_image fallback 描述

- [ ] **Step 2: 全库 grep 清零**

```bash
rg -n "sam2|SAM2|prepared_source_session|from transformers|generate_registration_prompts|validate_and_clean_mask|min_sam_prior_iou" .
```

Expected: 无业务代码命中（测试里若故意断言拒绝 sam 键除外）

- [ ] **Step 3: Commit**

```bash
git add hiad/README.md tutorial/ hiad/trainer/trainer.py
git commit -m "docs: align README with streaming no-SAM training"
```

---

### Task 7: 端到端冒烟（真实小数据）

**Files:**
- 无新代码文件；使用现有 `dataSets/MVTec-2K` 中极少样本或临时裁剪 jsonl

- [ ] **Step 1: 准备最小 `train_uni` 子集**（例如 2–4 张 bottle good）与已有 `foreground_references.yaml`

- [ ] **Step 2: 跑短训**

```bash
python runs/train.py --data-root dataSets/MVTec-2K --config configs/dinomaly.yaml --gpus 0 --batch-size 2
```

（若 `total_iters` 过大，可临时在本地 config 副本把 `total_iters` 改为 `20` 做冒烟，勿提交该临时文件。）

Expected:
- 不 OOM
- generation 含 Task 6 清单且无 sam 字段
- 日志无 `foreground_mask_rejected` fallback

- [ ] **Step 3: 跑推理冒烟**

```bash
python runs/inference.py --data-root dataSets/MVTec-2K --config configs/dinomaly.yaml --gpus 0 --batch-size 2
```

Expected: 正常出分或评估日志；无 SAM 加载

- [ ] **Step 4: 若有需修的 bug，修完后追加 commit；否则记录冒烟通过于 PR/对话**

---

## Spec Coverage Checklist

| Spec 要求 | Task |
|-----------|------|
| 流式驻留 / 禁全量 share_memory | 3, 4, 5 |
| 去 SAM，DINO warp mask | 1, 2 |
| 彻底删除旧路径 / 无兼容 | 1–6 |
| 精简 generation（无 SAM） | 2, 6 |
| 配准失败显式报错 | 1, 2 |
| `clsname` 一张参考 mask | 不变，2/6 文档 |
| 推理同步 | 5 |
| 内存硬约束 | 3, 4, 5, 7 |
| 测试要点 | 1–5, 7 |

## Plan Self-Review Notes

- `working_longest_edge` 原依赖 SAM processor，已在 Global Constraints 与 Task 2 显式改为配置项。
- `StreamingTaskDataset` 索引阶段只读尺寸，避免「先预处理才知道 patch 数」的死锁。
- `transformers` 仅 SAM 使用，随 Task 2 移除。
- 无 TBD 占位；Task 7 允许本地临时短 config，不提交。
