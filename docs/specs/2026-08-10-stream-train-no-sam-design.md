# 流式训练 + 去 SAM 预处理重构设计

日期：2026-08-10  
状态：已确认待实现

## 背景与问题

当前训练在 `prepared_source_session` 中对**全部**训练图做前景预处理，并 `share_memory_()` 驻留，单品类/多品类都会随样本量线性吃满内存，易 OOM。

预处理对每张图走冻结 DINO 配准 + SAM2 分割。产品语义是多视野（多机位）：每个 `clsname` 只需一张固定参考 mask。SAM 对固定机位场景过重，且拖累依赖与产物。

项目处于开发阶段，**不保留**旧 SAM generation、全量共享内存路径的兼容。

## 目标

1. **流式驻留**：训练/校准/推理均按需读图，禁止全量像素驻留。
2. **去 SAM**：保留 DINO 几何配准，将参考 mask warp 到当前图后做前景归一化。
3. **彻底重写**数据与预处理管线，删除尾大不掉的兼容与死路径。
4. **精简产物**：generation 自洽可用，不含 SAM 相关字段与文件依赖。

## 非目标

- 不兼容旧含 SAM 的 checkpoint generation。
- 不引入磁盘预处理缓存作为默认路径。
- 不在配准失败时静默 fallback 全图（开发期显式失败）。

## 架构与数据流

| 角色 | 职责 |
|------|------|
| `clsname` | 一个视野/机位；绑定一张参考图 + 一张固定 mask |
| 预处理 | 冻结 DINO 配准 → warp 参考 mask → 前景归一化；无 SAM |
| 训练数据 | Dataset 按需读盘、预处理、切块；禁止全量加载 + `share_memory_` |
| Worker | 每个 GPU 进程只处理分到的 task；样本生命周期限在 batch 内 |
| 产物 | 重新定义 generation 清单；删除 SAM 与仅为旧流水线服务的逻辑 |

### 训练数据流

```text
train_uni.jsonl
  → 仅解析元数据（路径/clsname），不读像素
  → 启动时：按 clsname 校准一次参考产物（DINO 模板/原型 + 参考 mask）
  → 每个 GPU worker：
        Dataset[i] → 读一张图
                 → DINO 配准 + warp 参考 mask
                 → 归一化
                 → 按 task 切出 patch / thumbnail
                 → 进 batch 训练
                 → 释放该样本
  → 全部 task 训完后：再流式跑正常样本做全局校准
  → 发布精简 generation
```

### 推理数据流

与训练同一套「按图即时预处理」；不驻留全量测试集。推理输入仍只有原图 + `clsname`。

## 组件边界

| 模块 | 做什么 | 不做什么 |
|------|--------|----------|
| `preprocessing/` | 按 `clsname` 校准参考；单图 DINO→warp→归一化 | 不加载 SAM；不批量驻留图像；不生成 SAM prompt |
| `data/` | `HRSample` 元数据；新流式 Dataset（按需读图/预处理/切块） | 不再提供全量 `prepared_source_session` / `share_memory_` 训练路径 |
| `trainer/` | 编排校准 → 多 GPU task 训练 → 流式校准 evidence → 发布 | 不在主进程打开全部训练图 |
| `inferencer/` | 按图流式预处理 + 推理 | 不把整份测试集先预处理进共享内存 |
| `scoring/` | 多风险分数与全局校准 | 不感知 SAM |
| `checkpoints/` | generation 原子发布（清单按新产物） | 不校验旧 SAM generation |

### 删除 / 废弃

- `hiad/preprocessing/sam.py` 及一切 `sam2_*` 配置
- `generate_registration_prompts` 中仅服务 SAM 的 box/positive points；改为直接输出 `warped_mask` + DINO 指标
- `prepared_source_session`、训练路径上的 `share_memory_()`
- 旧产物兼容逻辑与 `min_sam_prior_iou` 等 SAM 阈值
- 配准失败时的 `full_image` fallback

## Generation 产物

```text
generations/<id>/
├── tasks.json
├── dynamic_patch_weight.pt
├── thumbnail_weight.pt
├── multirisk_calibration.json
├── preprocessing_registry.json
└── preprocessing/<clsname>/
    ├── preprocessing.yaml
    ├── preprocessing_manifest.json
    ├── foreground_prototypes.pt
    ├── reference_feature_template.pt
    └── reference_foreground.rle
```

说明：配准所需参考产物仍打进 generation，避免推理依赖训练机上的 `foreground_references.yaml` 绝对路径；仅去掉 SAM 相关内容。

根目录 `current.json` 继续指向当前 generation（既有机制保留）。

## 配置变化

`configs/dinomaly.yaml` 的 `preprocessing`：

**保留**

- `schema_version`（实现时 bump，与旧配置显式不兼容）
- `array_color_space` / `input_scale` / `mean` / `std`
- `reference_manifest`
- `dino_backbone_name` / `dino_feature_layer`
- `boundary_expand_ratio`（若仍用于 warp mask 清理）
- `min_dino_matches` / `min_dino_inlier_ratio` / `max_dino_reprojection_ratio`
- `max_area_ratio_deviation` / `min_reference_coverage`（若仍用于 warp mask 校验）

**删除**

- `sam2_model_id` / `sam2_dtype` / `sam2_batch_size`
- `min_sam_prior_iou`
- 其它仅服务 SAM 的字段

训练集契约不变：`train_uni.jsonl` 仅正常图、无 mask、非空 `clsname`；每视野一张参考图+二值 mask。

## 错误处理

| 场景 | 行为 |
|------|------|
| DINO 配准失败或 warp 后空 mask | 训练中断并报明确 reason；推理同样失败，不静默全图兜底 |
| `clsname` 无参考 | 启动校准阶段失败 |
| `train_uni.jsonl` 含异常/mask | 拒绝加载 |
| 旧含 SAM 的 generation | schema/manifest 不匹配即失败 |

## 内存硬约束

1. 主进程编排只持有元数据，不持有像素。
2. Worker 峰值像素约 `batch_size` 量级；默认 `num_workers=0`，避免多进程预取放大内存。
3. DINO 预处理权重与 detector 分阶段占用 GPU，用完释放。
4. 校准/推理按 batch 流式，禁止整集 `share_memory_`。

## 测试要点

1. 单/多视野：校准与训练可跑通；峰值内存不随总样本量涨到全量级。
2. 单元：warp mask 几何正确；配准失败路径抛出明确错误。
3. 配置与 manifest 无 `sam2_*`。
4. 端到端：generation 仅含上述清单；推理可加载并出分。
5. 代码无对 `sam.py`、`prepared_source_session` 训练路径的引用。

## 成功标准

- 样本量远大于可用 RAM 时训练不因全量驻留 OOM。
- 代码与产物无 SAM 依赖。
- 每视野一张参考 mask；行为为 DINO 配准 + warp。
- 管线清晰，无旧路径兼容包袱。
