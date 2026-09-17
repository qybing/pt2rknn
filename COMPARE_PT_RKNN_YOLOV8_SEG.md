# YOLOv8-Seg：.pt vs RKNN 对比说明

脚本：

```text
rknn_model_zoo/examples/yolov8_seg/python/compare_pt_rknn_yolov8_seg.py
```

用于验收 **原权重 `.pt`** 与 **INT8/FP `.rknn`** 在实例分割上是否一致。

---

## 1. 对比什么

同一张图、同一套阈值下：

| 层级 | 指标 |
|------|------|
| 检测框 | 数量、类别、框 IoU、`|Δconf|` |
| 分割掩码 | 配对实例的 **mask IoU**（二值） |
| 整集 | pass/warn/fail、match_rate、mean_box_iou、mean_mask_iou |

RKNN 后处理对齐本目录 `yolov8_seg.py`（常见 **13 路**：每尺度 box / cls / score_sum / mask_coeff ×3 + proto）。

与 demo 的对应关系：

- **一致**：letterbox+RGB、丢弃 score_sum（用 ones）、cls 原样、DFL、filter、NMS、proto@coeff、crop、`get_real_*`
- **唯一差异**：mask 插值边长用 `--imgsz`（demo 写死 640；1280 模型必须改）

---

## 2. 环境

建议在 **RK3588 板端**跑（有 NPU）。需要：

| 依赖 | 用途 |
|------|------|
| `rknnlite` | 跑 `.rknn` |
| `torch` + `ultralytics` | `--pt-mode predict`（官方 .pt） |
| `opencv-python` / `numpy` | 读图、画图 |

若 `.pt` 是 zoo / airockchip 的 **torchscript raw**，用 `--pt-mode raw`（走 `Torch_model_container`，与 RKNN 同后处理）。

---

## 3. 用法

```bash
cd rknn_model_zoo/examples/yolov8_seg/python

python3 compare_pt_rknn_yolov8_seg.py \
  --pt /path/to/yolov8n-seg.pt \
  --rknn /path/to/yolov8n-seg_i8.rknn \
  --source ../model \
  --img_save
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--pt-mode predict` | 默认；Ultralytics 官方 seg `.pt` |
| `--pt-mode raw` | torchscript 13 路 raw，与 RKNN 同 decode |
| `--conf` / `--iou` | 默认 0.25 / 0.45（对齐 demo） |
| `--match-iou` | 框配对阈值，默认 0.5 |
| `--names` | 逗号分隔类别名；默认跟 `.pt` 或 COCO80 |
| `--classes` | 只比部分类，如 `--classes person` |
| `--max_images` | 限制张数；0=全部 |
| `--img_save` | 保存左右对比图（框+半透明 mask） |

---

## 4. 如何读结果

单图示例：

```text
[PASS] bus.jpg  pt=3 rknn=3 match=3  box_iou(mean/min)=0.97/0.94  mask_iou(mean/min)=0.91/0.85  max|dconf|=0.04
```

| 字段 | 含义 |
|------|------|
| `pt/rknn/match` | 两侧实例数与成功配对数 |
| `box_iou` | 配对框 IoU |
| `mask_iou` | 配对 mask IoU |
| `max\|dconf\|` | 最大置信度差 |

整集 `OVERALL`：

- **pass 多、fail=0**：转换可用  
- **warn**：i8 常见小漂移，看 mask_iou 与可视化  
- **fail**：检查 RKNN 是否为 seg 13 路、校准集、后处理是否对齐  

可视化目录默认：`compare_result_pt_vs_rknn_seg/`（左 PT，右 RKNN）。

---

## 5. 与转换流程的关系

1. ONNX 需为 zoo 优化的 **多输出 seg**（不是官方单输出 fused）  
2. `python convert.py <onnx> rk3588 i8` 得到 `.rknn`  
3. 用本脚本对比 **训练/导出用的 `.pt`** 与 i8  

若只有 ONNX↔RKNN，可用同目录思路另写；本脚本聚焦 **业务关心的 .pt 效果 vs 板上 i8**。

---

## 6. 注意

- RKNN 输入为 letterbox 后的 **NHWC uint8**（`mean=0,std=255`）  
- `predict` 模式：PT 与 RKNN 后处理路径不完全相同，允许小差异；若要对齐 raw 数值，用 `--pt-mode raw`  
- 分割对分辨率敏感：`--imgsz` 需与导出一致（默认 640）
