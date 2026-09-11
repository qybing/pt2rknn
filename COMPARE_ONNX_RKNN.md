# ONNX vs RKNN 对比说明文档

本文说明 `compare_onnx_rknn_yolo.py` 的用途、运行方法、输出含义，以及如何判断 `.onnx`→`.rknn` 转换是否正确。

脚本路径：

```text
rknn_model_zoo/examples/yolo11/python/compare_onnx_rknn_yolo.py
```

同类文档：`.pt`↔`.onnx` 见 `COMPARE_PT_ONNX.md`；`.pt`↔`.rknn` 见 `COMPARE_PT_RKNN.md`。  
支持 **YOLOv8 / YOLO11 / YOLO26**。不依赖 PyTorch。

---

## 1. 为什么要做这个对比

板上跑的是 `.rknn`，通常由同一份 `.onnx` 转换而来。转换可能引入：

- 量化误差（尤其 i8）
- 传错 / 截断 ONNX
- mean/std、预处理不一致

本脚本：

1. 同一张图、同一套 letterbox
2. ONNX Runtime（CPU）与 RKNN NPU 各推一次
3. 对比 **原始输出张量** + **NMS 后检测框**
4. 汇总整集数字（match_rate 等）

目标：确认「这块 RKNN 是否和源 ONNX 预测一致」。

---

## 2. 环境与依赖

建议在 **RK3588 板端**运行。

| 依赖 | 用途 |
|------|------|
| `onnxruntime` / `onnx` | 跑 / 校验 ONNX |
| `rknnlite` | 跑 RKNN |
| `opencv-python` / `numpy` | 读图、解码 |

```bash
python3 -c "import onnx, onnxruntime, cv2, numpy; from rknnlite.api import RKNNLite; print('deps ok')"
```

示例模型：

```text
# YOLO26 fork 6 路
safety_helmet_all.onnx
safety_helmet_all_i8.rknn

# YOLO11 9 路
helmet_y11s_best.onnx
helmet_y11s_best_fp.rknn / helmet_y11s_best_i8.rknn

/userdata/jovan/code/rk3588/dataset/helmet/
```

> ONNX 必须完整；两边可用 `md5sum` 校验。

---

## 3. 怎么运行

```bash
cd /userdata/jovan/code/rk3588/rknn_model_zoo/examples/yolo11/python
```

### 3.1 YOLO26 fork i8

```bash
python3 compare_onnx_rknn_yolo.py \
  --onnx safety_helmet_all.onnx \
  --rknn safety_helmet_all_i8.rknn \
  --source /userdata/jovan/code/rk3588/dataset/helmet/ \
  --nc 2 --names head,helmet \
  --img_save
```

### 3.2 YOLO11 FP（建议先于 i8）

```bash
python3 compare_onnx_rknn_yolo.py \
  --onnx helmet_y11s_best.onnx \
  --rknn helmet_y11s_best_fp.rknn \
  --source /userdata/jovan/code/rk3588/dataset/helmet/ \
  --nc 2 --names head,helmet \
  --img_save
```

### 3.3 YOLO11 i8

```bash
python3 compare_onnx_rknn_yolo.py \
  --onnx helmet_y11s_best.onnx \
  --rknn helmet_y11s_best_i8.rknn \
  --out_dir ./compare_result_onnx_vs_rknn_i8 \
  --nc 2 --names head,helmet \
  --img_save
```

### 3.4 一次对比多块 RKNN

```bash
python3 compare_onnx_rknn_yolo.py \
  --onnx safety_helmet_all.onnx \
  --rknn safety_helmet_all_i8.rknn \
         ultralytics_src/safety_helmet_all-rk3588_i8.rknn \
  --nc 2 --names head,helmet \
  --img_save
```

### 3.5 参数说明

| 参数 | 默认 | 含义 |
|------|------|------|
| `--onnx` | `safety_helmet_all.onnx` | 源 ONNX（须与转 RKNN 时同一份） |
| `--rknn` | `safety_helmet_all_i8.rknn` | 一块或多块 |
| `--source` / `--img_folder` | helmet 目录 | 图片或目录（后者为别名） |
| `--nc` / `--names` | `2` / `head,helmet` | 类别数与名称 |
| `--family` | `auto` | 仅作提示；解码按通道数 |
| `--box-scale` | `auto` | fused 框是否 ×imgsz |
| `--conf` / `--iou` / `--match-iou` | 0.25 / 0.7 / 0.5 | 阈值 |
| `--max_images` | `0` | 0=全部 |
| `--img_save` / `--save_limit` | 关 / 5 | 可视化 |

---

## 4. 脚本内部做什么

### 4.1 预处理

| 后端 | 输入 |
|------|------|
| ONNX | `NCHW float32`，`/255`，`(1,3,640,640)` |
| RKNN | `NHWC uint8`，`(1,640,640,3)`（转换常配 mean=0, std=255） |

### 4.2 布局自动识别

| layout | 典型 | 模型 |
|--------|------|------|
| `split6` | 6×4D | YOLO26 fork |
| `split9` | 9×4D | YOLOv8 / YOLO11 |
| `fused` | `(1,4+nc,N)` | 官方非 e2e |
| `e2e` | `(1,max_det,6)` | 端到端 |

box 通道：`reg_max=1` 直接回归；`reg_max=16` 做 DFL。9 路时忽略每尺度 `score_sum`。

### 4.3 对比两层

1. **张量**：按 shape 对齐后算 cos / **sig_cos** / mae / max_abs（优先看 sig_cos）  
2. **框**：同类 + IoU 匹配，汇总整集数字  
3. **fused score 抹零**：若 ONNX 分数有信号、RKNN 分数通道≈0 → 该图强制 FAIL

---

## 5. 怎么读输出

### 5.1 张量表

```text
 idx   onnx_shape        rknn_shape         cos     sig_cos        mae     max_abs
```

| 列 | 怎么看 |
|----|--------|
| `cos` | 全图余弦；近零背景时可能虚低 |
| `sig_cos` | 有效区域余弦，**优先看这个** |
| `mae` / `max_abs` | 越小越好 |

### 5.2 单图

```text
[1/128] xxx.jpg | tag:pass 2/2/2 iou=0.97 dconf=0.01 sig=0.9999
```

`pass/warn/fail` 规则与 `COMPARE_PT_RKNN.md` 5.3 相同（把 PT 换成 ONNX 侧）。

### 5.3 整集数字

```text
=== ONNX vs safety_helmet_all_i8 (split6) ===
  n_images        128
  n_onnx / n_rknn 528 / 523
  n_match         520
  match_rate      0.9848
  recall_vs_onnx  0.9848
  extra_rate      0.0057
  mean_iou        0.9646
  p5_iou          0.9231
  mean|dconf|     0.0204
  max|dconf|      0.1659
  high_conf(≥0.50)  recall=1.0000  ...
  tensor          cos_min=0.990691  sig_cos_min=0.990691  mae_mean=0.254401
  结论            转换可用：与 ONNX 参考检出高度一致。
  PASS=113 (88.3%)  WARN=14 (10.9%)  FAIL=1 (0.8%)
```

| 指标 | 含义 | 好坏 |
|------|------|------|
| `match_rate` | 框对齐比例 | **越大越好** |
| `recall_vs_onnx` | ONNX 有的框 RKNN 找回多少 | **越大越好** |
| `extra_rate` | RKNN 相对多检比例 | **越小越好** |
| `mean_iou` | 配对平均重叠 | **越大越好** |
| `mean\|dconf\|` | 平均分数差 | **越小越好** |
| `sig_cos_min` | 有效区域最差相似度 | **越大越好**（FP 常 ≥0.99） |

结论：

| 结论 | 条件 |
|------|------|
| 转换可用 | match_rate≥0.95 且 mean_iou≥0.90 且 mean\|dconf\|≤0.05 |
| i8 可接受 | match_rate≥0.85 且 mean_iou≥0.80 |
| 偏差较大 | 其它 |

**FP 期望「转换可用」；i8 允许轻微分数漂移，但框应对齐。**  
出现 `score抹零` → 不要部署该 i8，需重转（常见于 fused 整段 int8）。

---

## 6. 经验参考（本项目实测）

数据集：`/userdata/jovan/code/rk3588/dataset/helmet/`，128 张。  
ONNX：`safety_helmet_all.onnx`（fork 6 路）。  
RKNN：`safety_helmet_all_i8.rknn`。  
阈值：`conf=0.25`，NMS IoU=`0.7`，匹配 IoU=`0.5`，`--nc 2`。

命令：

```bash
python3 compare_onnx_rknn_yolo.py \
  --onnx safety_helmet_all.onnx \
  --rknn safety_helmet_all_i8.rknn \
  --source /userdata/jovan/code/rk3588/dataset/helmet/ \
  --nc 2 --names head,helmet
```

| 指标 | 数值 |
|------|------|
| layout | `split6` |
| n_onnx / n_rknn / n_match | 528 / 523 / 520 |
| **match_rate** | **0.9848** |
| recall_vs_onnx | 0.9848 |
| extra_rate | 0.0057 |
| **mean_iou** | **0.9646** |
| p5_iou | 0.9231 |
| **mean\|dconf\|** | **0.0204** |
| high_conf recall | 1.0000（496/496） |
| tensor cos_min / sig_cos_min | 0.990691 / 0.990691 |
| tensor mae_mean | 0.254401 |
| PASS / WARN / FAIL | 113 / 14 / 1 |
| 结论 | **转换可用** |

唯一 FAIL 图：`000064_....jpg`，ONNX 7 框、RKNN 4 框（阈值边界），已配对 IoU≥0.90。  
与 PT↔RKNN 整集数字一致（因 PT↔ONNX 几乎完全重合）。

---

## 7. 推荐流程

1. 先比 **FP RKNN**（有的话）→ 期望高度一致  
2. 再比 **i8** → 看整集 match_rate / mean_iou / sig_cos  
3. 加 `--img_save` 抽看 FAIL 图  
4. 若 ONNX↔RKNN 差，而 PT↔ONNX 好 → 问题在量化/convert  
5. 若 PT↔ONNX 就差 → 先修导出（见 `COMPARE_PT_ONNX.md`）

---

## 8. 常见问题

### 8.1 ONNX corrupt

重新传输并用 `md5sum` 对齐转换源文件。

### 8.2 `cos` 很低但 `sig_cos≈1`

背景底噪导致；优先看 sig_cos 和框匹配。

### 8.3 FP 也 FAIL

核对：是否同一份 ONNX、mean/std=0/255、尺寸 640、`--nc` 是否正确。

### 8.4 fused i8 检不出框

看是否 `score抹零`；应用 fork 6 路导出再量化，或改混合精度。

### 8.5 输入维度报错

RKNN 需要 4 维 `(1,H,W,C)`；本脚本已按此构造。
