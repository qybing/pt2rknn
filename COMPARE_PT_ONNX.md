# PT vs ONNX 对比说明文档

本文说明 `compare_pt_onnx_yolo.py` 的用途、运行方法、输出含义，以及如何判断 `.pt`→`.onnx` 导出是否正确。

脚本路径：

```text
rknn_model_zoo/examples/yolo11/python/compare_pt_onnx_yolo.py
```

同类文档：`.pt`↔`.rknn` 见 `COMPARE_PT_RKNN.md`；ONNX↔RKNN 见 `COMPARE_ONNX_RKNN.md`。  
本脚本做 **框匹配 +（非 e2e 时）解码张量对比**，不是 mAP。

---

## 1. 为什么要做这个对比

转换链路通常是：

```text
.pt  →  .onnx  →  .rknn
```

先确认 **导出的 ONNX 和原 `.pt` 一致**，再查 RKNN 量化才有意义。  
若 `.pt`↔`.onnx` 就对不上，后面再比 RKNN 也分不清是导出问题还是量化问题。

本脚本：

1. 同一张图、同一套 letterbox
2. PyTorch 跑 `.pt`，ONNX Runtime 跑 `.onnx`
3. 非 e2e 时对比解码后张量（cosine / mae）
4. NMS 后对比检测框，并汇总整集数字

支持 **YOLOv8 / YOLO11 / YOLO26**（`--family auto` 按 `Detect.reg_max`：`1`→YOLO26，`16`→v8/v11）。

---

## 2. 环境与依赖

可在 PC 或板端跑（不需要 NPU）。

| 依赖 | 用途 |
|------|------|
| `torch` / `ultralytics` | 跑 `.pt` |
| `onnxruntime` | 跑 `.onnx` |
| `opencv-python` / `numpy` | 读图、解码、匹配 |

```bash
python3 -c "import torch, cv2, numpy, onnxruntime; from ultralytics import YOLO; print('deps ok')"
```

默认示例文件：

```text
examples/yolo11/python/safety_helmet_all.pt
examples/yolo11/python/safety_helmet_all.onnx   # fork 6 路
/userdata/jovan/code/rk3588/dataset/helmet/
```

---

## 3. 怎么运行

```bash
cd /userdata/jovan/code/rk3588/rknn_model_zoo/examples/yolo11/python
```

### 3.1 YOLO26 fork（6 路，推荐）

```bash
python3 compare_pt_onnx_yolo.py \
  --pt safety_helmet_all.pt \
  --onnx safety_helmet_all.onnx \
  --source /userdata/jovan/code/rk3588/dataset/helmet/ \
  --img_save
```

### 3.2 官方 fused ONNX

```bash
python3 compare_pt_onnx_yolo.py \
  --pt safety_helmet_all.pt \
  --onnx ultralytics_src/safety_helmet_all.onnx \
  --source /userdata/jovan/code/rk3588/dataset/helmet/ \
  --out_dir ./compare_result_pt_vs_onnx_fused \
  --img_save
```

### 3.3 YOLO11 / YOLOv8

```bash
python3 compare_pt_onnx_yolo.py \
  --pt helmet_y11s_best.pt \
  --onnx helmet_y11s_best.onnx \
  --family v11 \
  --source /userdata/jovan/code/rk3588/dataset/helmet/ \
  --img_save
```

（需自备对应 `.pt`。ONNX 为 9 路时 layout=`split9`。）

### 3.4 参数说明

| 参数 | 默认 | 含义 |
|------|------|------|
| `--pt` / `--onnx` | `safety_helmet_all.*` | 权重路径 |
| `--source` | helmet 目录 | 图片或目录 |
| `--family` | `auto` | `auto` / `v26` / `v11` / `v8` |
| `--pt-mode` | `auto` | PT 对齐方式，跟 ONNX 布局走 |
| `--head` | `one2many` | 拆头优先分支（fork 导出用这个） |
| `--fuse` | 开 | 仅影响对比时的 PT 侧 |
| `--box-scale` | `auto` | fused 框是否 ×imgsz |
| `--conf` / `--iou` / `--match-iou` | 0.25 / 0.7 / 0.5 | 阈值 |
| `--max_images` | `0` | 0=全部 |
| `--img_save` / `--save_limit` | 关 / 5 | 可视化 |

`--pt-mode` / `--head` / `--fuse` **只影响对比时的 `.pt` 侧**，不影响 ONNX 文件本身，部署也不用设。含义与 `COMPARE_PT_RKNN.md` 第 3.6 节相同。

---

## 4. 脚本内部做什么

### 4.1 预处理

两边同一 letterbox → RGB → `/255` → `NCHW float32`。

### 4.2 布局

| layout | 典型输出 |
|--------|----------|
| `split6` | YOLO26 fork：3 尺度 × (reg + cls) |
| `split9` | v8/v11：3 尺度 × (box + cls + score_sum) |
| `fused` | `(1, 4+nc, N)` |
| `e2e` | `(1, max_det, 6)` |

YOLO26：`reg_max=1` 直接回归；v8/v11：`reg_max=16` 走 DFL。

### 4.3 对比两层

1. **张量**（非 e2e）：解码成 `(1,4+nc,N)` 后算 cosine / box_mae / cls_mae  
2. **框**：同类且 IoU≥`--match-iou` 贪心匹配

---

## 5. 怎么读输出

### 5.1 单图

```text
[1/128] xxx.jpg | safety_helmet_all:pass 2/2/2 iou=0.99 dconf=0.00 cos=1.000000
```

| 字段 | 含义 |
|------|------|
| `pass/warn/fail` | 单图粗分（规则同 PT↔RKNN，见 `COMPARE_PT_RKNN.md` 5.3） |
| `2/2/2` | PT 框数 / ONNX 框数 / 配对成功数 |
| `cos` | 解码张量余弦相似度（有则打印） |

### 5.2 整集数字

```text
=== PT vs safety_helmet_all (split6) ===
  n_images        128
  n_pt / n_onnx   528 / 528
  n_match         528
  match_rate      1.0000
  recall_vs_pt    1.0000
  extra_rate      0.0000
  mean_iou        1.0000
  p5_iou          1.0000
  mean|dconf|     0.0000
  max|dconf|      0.0000
  high_conf(≥0.50)  recall=1.0000  ...
  mean_cosine     1.000000
  min_cosine      1.000000
  结论            导出可用：与参考模型检出高度一致。
  PASS=128 (100.0%)  WARN=0 (0.0%)  FAIL=0 (0.0%)
```

| 指标 | 含义 | 好坏 |
|------|------|------|
| `match_rate` | `n_match/max(n_pt,n_onnx)` | **越大越好**，理想≈1 |
| `recall_vs_pt` | `n_match/n_pt` | **越大越好** |
| `extra_rate` | `(n_onnx-n_match)/n_onnx` | **越小越好** |
| `mean_iou` | 已配对平均 IoU | **越大越好**，>0.9 通常很好 |
| `mean\|dconf\|` | 平均分数差 | **越小越好** |
| `mean_cosine` / `min_cosine` | 解码张量相似度 | **越大越好**，接近 1 |

#### 总结论（最重要）

| 结论文案 | 什么时候算过 | 含义 |
|----------|--------------|------|
| **导出可用** | match_rate≥0.95 且 mean_iou≥0.90 且 mean\|dconf\|≤0.05 | 直接过，可进入下一环或部署 |
| **轻微差异可接受** | match_rate≥0.85 且 mean_iou≥0.80（但不满足上面） | 勉强过，建议抽看 FAIL 图 |
| **偏差较大** | 不满足上面 | 不过，要查导出/量化/后处理 |

**三段链路各自看自己的结论**：本环节（`.pt`↔`.onnx`）期望「**导出可用**」；`.onnx`↔`.rknn` 与 `.pt`↔`.rknn` 期望「**转换可用**」——FP 期望「转换可用」，i8 最好也是「转换可用」，至少「i8 可接受」。本项目 safety_helmet_all：三段都是「可用」→ 整体通过。

#### 优先看的 3 个数字（按重要性）

| 优先级 | 指标 | 怎样算好看 |
|--------|------|------------|
| 1 | match_rate | 越大越好，i8 常见 ≥0.95 |
| 2 | mean_iou | 越大越好，一般 ≥0.90 |
| 3 | mean\|dconf\| | 越小越好，i8 常见 ≤0.05 |

这三项就是脚本打「结论」用的条件，过了这三项 ≈ 结论可用 ≈ 可以过。

#### 次要指标（辅助，不单独卡死）

| 指标 | 作用 |
|------|------|
| recall_vs_pt | 越大越好，看漏检 |
| extra_rate | 越小越好，看多检 |
| high_conf recall | 主目标（高分框）是否稳，最好接近 1 |
| mean_cosine / min_cosine | 张量层，导出/FP 常接近 1；别被背景噪声吓到 |
| PASS/WARN/FAIL 张数 | 只用来扫图，不要只看这个过不过 |

**一句话记法**：先看 match_rate，再看 mean_iou，最后看 mean\|dconf\|；三项达标 = 导出可用，直接进下一环；只到「可接受」就抽看 FAIL 图；「偏差较大」先修导出再往下走。

优先看整集数字，不要只看 PASS 张数。

---

## 6. 经验参考（本项目实测）

数据集：`/userdata/jovan/code/rk3588/dataset/helmet/`，128 张。  
权重：`safety_helmet_all.pt` ↔ `safety_helmet_all.onnx`（YOLO26 fork 6 路）。  
阈值：`conf=0.25`，NMS IoU=`0.7`，匹配 IoU=`0.5`。

命令：

```bash
python3 compare_pt_onnx_yolo.py \
  --pt safety_helmet_all.pt \
  --onnx safety_helmet_all.onnx \
  --source /userdata/jovan/code/rk3588/dataset/helmet/
```

| 指标 | 数值 |
|------|------|
| layout | `split6` |
| n_pt / n_onnx / n_match | 528 / 528 / 528 |
| **match_rate** | **1.0000** |
| recall_vs_pt | 1.0000 |
| extra_rate | 0.0000 |
| **mean_iou** | **1.0000** |
| p5_iou | 1.0000 |
| **mean\|dconf\|** | **0.0000** |
| high_conf recall | 1.0000（496/496） |
| mean_cosine / min_cosine | 1.000000 / 1.000000 |
| PASS / WARN / FAIL | 128 / 0 / 0 |
| 结论 | **导出可用** |

导出几乎无损：框完全对齐，解码张量 cosine=1。

---

## 7. 推荐流程

1. 先跑少量图：`--max_images 2 --img_save`
2. 跑全集，看 `match_rate` / `mean_iou` / `mean_cosine`
3. 导出可用后再做 ONNX↔RKNN（`compare_onnx_rknn_yolo.py`）
4. 最终可再做 PT↔RKNN（`compare_pt_rknn_yolo.py`）

---

## 8. 常见问题

- **拆头失败 / 对不上**：fork 用 `--head one2many`；官方 fused 不要强行 `--pt-mode split`
- **余弦低但框还行**：看有效区域与框匹配；背景噪声会拉低全图 cos
- **没有 `.pt` 的 YOLO11**：只能跳过本脚本，直接做 ONNX↔RKNN
- **想算 mAP**：需要 GT；本脚本不算 mAP
