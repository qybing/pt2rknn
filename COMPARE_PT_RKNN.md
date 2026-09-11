# PT vs RKNN 对比说明文档

本文说明 `compare_pt_rknn_yolo.py` 的用途、运行方法、输出含义，以及如何根据整集数字判断 RKNN 转换是否和原 `.pt` 一致。

脚本路径：

```text
rknn_model_zoo/examples/yolo11/python/compare_pt_rknn_yolo.py
```

同类文档：`.pt`↔`.onnx` 见 `COMPARE_PT_ONNX.md`；ONNX↔RKNN 见 `COMPARE_ONNX_RKNN.md`。  
本脚本比的是 **NMS 后的检测框**，不是 mAP（当前头盔目录没有 GT labels）。

---

## 1. 为什么要做这个对比

板上部署用的是 `.rknn`，训练权重是 `.pt`。中间可能经过：

```text
.pt  →  .onnx  →  .rknn
```

分段对比（`.pt`↔`.onnx`、`.onnx`↔`.rknn`）适合排错。  
最终业务验收更直接的问题是：**这块 RKNN 检出的框，和原 `.pt` 像不像。**

本脚本做的事情是：

1. **同一张图、同一套 letterbox 预处理**
2. 板端用 **PyTorch 跑 `.pt`**，用 **NPU 跑 `.rknn`**
3. 两边各自解码 + NMS，得到 `(xyxy, conf, cls)`
4. 按「同类且 IoU≥匹配阈值」配对
5. 汇总 **整集数字**（match_rate / mean_iou / 分数差等），再附带按图 PASS/WARN/FAIL 扫图

目标：确认「转换出来的 RKNN，是否和源 `.pt` 预测一致」。

> 本脚本不对比原始 tensor。YOLO26 fork 是 6 路 raw，官方 Ultralytics 是 fused `(1,6,8400)`，和 `.pt` 内部张量对不齐；对齐的是后处理后的框。

---

## 2. 环境与依赖

必须在 **RK3588 板端**跑（要有 NPU，也要能 `import torch`）。

### 2.1 必需依赖

| 依赖 | 用途 |
|------|------|
| Python3 | 运行脚本 |
| `torch` / `ultralytics` | 跑 `.pt` |
| `rknnlite`（rknn-toolkit-lite2） | 板端跑 `.rknn` |
| `opencv-python` / `cv2` | 读图、画框 |
| `numpy` | 解码、NMS、匹配 |

可用下面命令快速检查：

```bash
python3 -c "import torch, cv2, numpy; from ultralytics import YOLO; from rknnlite.api import RKNNLite; print('torch', torch.__version__); print('deps ok')"
```

脚本已处理：必须先 `import torch` 再 `import rknnlite`（后者会改写 logging 级别名）。不要自己把 import 顺序改反。

### 2.2 需要准备的文件

| 文件 | 说明 |
|------|------|
| `.pt` | 训练/导出用的同一份 Ultralytics 权重 |
| `.rknn` | 待验证的 FP 或 i8（可同时传多块） |
| 测试图片 | jpg/png/bmp/jpeg，文件或目录 |

当前 YOLO26 头盔示例默认文件：

```text
examples/yolo11/python/safety_helmet_all.pt
examples/yolo11/python/safety_helmet_all_i8.rknn          # fork 6 路 i8（推荐部署）
examples/yolo11/python/ultralytics_src/safety_helmet_all-rk3588_i8.rknn  # 官方 fused i8
/userdata/jovan/code/rk3588/dataset/helmet/               # 128 张，无 GT
```

两条转换链路 **不是同一份 ONNX**：

| 链路 | ONNX | RKNN 输出 | 说明 |
|------|------|-----------|------|
| fork `format=rknn` | `safety_helmet_all.onnx` | 6 路 `reg/cls`，cls 是 logits | 推荐；脚本 layout=`split6` |
| 官方 Ultralytics | `ultralytics_src/safety_helmet_all.onnx` | fused `(1,6,8400)`，框约 0~1 | 脚本 layout=`fused`，会自动 ×640 |

---

## 3. 怎么运行

进入脚本目录：

```bash
cd /userdata/jovan/code/rk3588/rknn_model_zoo/examples/yolo11/python
```

### 3.1 对比 fork 6 路 i8（推荐部署这份）

```bash
python3 compare_pt_rknn_yolo.py \
  --pt safety_helmet_all.pt \
  --rknn safety_helmet_all_i8.rknn \
  --source /userdata/jovan/code/rk3588/dataset/helmet/ \
  --img_save
```

不带参数时默认就是上面这条（`--source` 指向 helmet 全集）。  
若只想先跑少量图确认路径和可视化，可加 `--max_images 2`。

### 3.2 对比官方 fused i8

```bash
python3 compare_pt_rknn_yolo.py \
  --pt safety_helmet_all.pt \
  --rknn ultralytics_src/safety_helmet_all-rk3588_i8.rknn \
  --source /userdata/jovan/code/rk3588/dataset/helmet/ \
  --out_dir ./compare_result_pt_vs_fused \
  --img_save
```

fused 框若是归一化坐标，`--box-scale auto` 会在 `max(|box|) < 2.5` 时自动 ×imgsz。一般不用改。

### 3.3 一次对比两块 RKNN

```bash
python3 compare_pt_rknn_yolo.py \
  --pt safety_helmet_all.pt \
  --rknn safety_helmet_all_i8.rknn \
         ultralytics_src/safety_helmet_all-rk3588_i8.rknn \
  --source /userdata/jovan/code/rk3588/dataset/helmet/ \
  --img_save
```

每块 RKNN 各自出一份整集报表。PT 按 layout 缓存，不会每块都重新前向一遍。

### 3.4 YOLO11 / YOLOv8（有对应 `.pt` 时）

```bash
python3 compare_pt_rknn_yolo.py \
  --pt helmet_y11s_best.pt \
  --rknn helmet_y11s_best_i8.rknn \
  --family v11 \
  --source /userdata/jovan/code/rk3588/dataset/helmet/ \
  --img_save
```

`--family auto` 时：`Detect.reg_max=1` 判 YOLO26，`16` 判 v8/v11。YOLO11 的 9 路头（box+cls+score_sum）会识别为 `split9`。

### 3.5 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--pt` | 当前目录 `safety_helmet_all.pt` | Ultralytics 权重 |
| `--rknn` | `safety_helmet_all_i8.rknn` | 一块或多块 `.rknn` |
| `--source` | `/userdata/jovan/code/rk3588/dataset/helmet/` | 图片文件或目录 |
| `--out_dir` | `./compare_result_pt_vs_rknn` | 可视化输出目录 |
| `--imgsz` | `640` | letterbox 边长 |
| `--conf` | `0.25` | 置信度阈值 |
| `--iou` | `0.7` | NMS IoU（Ultralytics 默认） |
| `--match-iou` | `0.5` | 两边框配对所需最低 IoU |
| `--family` | `auto` | `auto` / `v26` / `v11` / `v8` |
| `--pt-mode` | `auto` | **仅影响对比时的 PT 侧**，见下方 3.6 |
| `--head` | `one2many` | **仅影响对比时的 PT 侧**，见下方 3.6 |
| `--fuse` | 开 | **仅影响对比时的 PT 侧**，见下方 3.6 |
| `--box-scale` | `auto` | fused RKNN 框是否 ×imgsz（部署官方 fused 时也要对应处理） |
| `--max_images` | `0`（全部） | 只测前 N 张 |
| `--img_save` | 关闭 | 保存左右对比图 |
| `--save_limit` | `5` | 最多保存前 N 张可视化 |

### 3.6 `--pt-mode` / `--head` / `--fuse` 是什么？部署要改吗？

这三个参数**只用来对齐 `.pt` 参考结果**，让对比公平。板端真正部署 `.rknn` 时**不会、也不需要**写进业务代码。

| 参数 | 对比脚本里干什么 | 部署代码要不要管 |
|------|------------------|------------------|
| `--pt-mode` | 决定 PT 怎么出框：`auto` 跟 RKNN 布局走；`split` 强制从 Detect 拆头取 raw；`raw` 走解码后 concat；`e2e` 走 `predict()` | **不用管**。部署只跑 RKNN，不跑 `.pt` |
| `--head` | 拆头时优先取 YOLO26 的哪条分支：`one2many`（fork `format=rknn` 导出的是这条）或 `one2one` | **不用管**。RKNN 里已经是导出时那一路的 6 个 tensor，没有 one2many/one2one 这个开关 |
| `--fuse` | 对比前把 PT 的 Conv+BN 融合成一个算子，更接近导出后的计算图 | **不用管**。`.rknn` 转换时已经定型，板端没有 fuse 这一步 |

**部署 fork 6 路 i8（推荐）时，业务代码只要关心：**

1. 预处理：letterbox → RGB → `NHWC uint8 (1,640,640,3)`（转换时 `mean=0, std=255`）
2. `rknn.inference` 得到 6 路输出
3. 后处理：对每尺度 `reg` 解码框；对 `cls` 做 **sigmoid**（logits）；分类别 NMS
4. 把 letterbox 坐标映射回原图

**不要**在部署里写 `--pt-mode` / `--head` / `--fuse`。  
`--box-scale` 只对官方 fused 那份有意义（框可能是 0~1，要 ×640）；fork 6 路一般不需要。

默认 `--pt-mode auto --head one2many --fuse` 即可，对比 fork 模型时不用改。

---

## 4. 脚本内部做了什么（便于理解输出）

### 4.1 预处理（两边必须对齐）

对每张图：

1. `letter_box` 缩放到 `640x640`（保持比例，黑边填充）
2. BGR → RGB

然后分别构造输入：

| 后端 | 输入格式 | 说明 |
|------|----------|------|
| PT | `NCHW float32`，再 `/255` | shape: `(1,3,640,640)` |
| RKNN | `NHWC uint8` | shape: `(1,640,640,3)` |

RKNN 转换时通常配置了 `mean=0, std=255`，NPU 侧自动除以 255。两边数学上等价。

### 4.2 RKNN 布局自动识别

探测一次 dummy 推理，按输出个数和 shape 分类：

| layout | 典型输出 | 来源 |
|--------|----------|------|
| `split6` | 3 尺度 × (reg 4 通道 + cls nc 通道) | YOLO26 fork `format=rknn` |
| `split9` | 3 尺度 × (box + cls + score_sum) | YOLOv8 / YOLO11 优化头 |
| `fused` | `(1, 4+nc, N)`，头盔是 `(1,6,8400)` | 官方非 e2e ONNX |
| `e2e` | `(1,300,6)` 一类 | 带 NMS 的端到端导出 |

YOLO26 的 `reg_max=1`，box 是直接 ltrb，不做 DFL。  
v8/v11 的 `reg_max=16`，box 走 DFL。脚本按通道数自动选。

### 4.3 PT 怎么对齐 RKNN

`--pt-mode auto` 时：

| RKNN layout | PT 做法 |
|-------------|---------|
| `split6` / `split9` | 从 Detect 的 `cv2/cv3` 或 `one2many` 取 raw，同一套解码+NMS |
| `fused` | 临时关掉 `end2end`，取解码后 concat `(1,4+nc,N)` 再 NMS |
| `e2e` | 走 `yolo.predict()`，坐标已是原图像素 |

fork 这条链路必须对 **one2many**，不要拿 one2one / `predict()` 去硬比。

cls 若看起来像 logits（出现负值或大于 1），脚本会做 sigmoid。fork 的 6 路 cls 就是这种。

### 4.4 框匹配

配对条件：

- 同一类别
- IoU ≥ `--match-iou`（默认 0.5）
- 贪心：每个 PT 框找尚未占用的最高 IoU RKNN 框

匹配后统计数量、IoU、`|conf_pt - conf_rknn|`。

---

## 5. 运行日志逐段解释

下面按实际输出顺序说明。

### 5.1 加载 PT / RKNN

```text
Loading PT...
YOLO26n summary (fused): 120 layers, ...
=== PT vs RKNN ===
pt       : safety_helmet_all.pt
family   : YOLO26 (reg_max=1)
task/nc  : detect / 2  names={0: 'head', 1: 'helmet'}
pt_head  : reg_max=1 end2end=False fused=True
align    : pt-mode=auto head=one2many box-scale=auto
nms      : conf=0.25 iou=0.7 match-iou=0.5
source   : .../dataset/helmet/ (128 images)
RKNN     : safety_helmet_all_i8.rknn  layout=split6  shapes=[(1, 4, 80, 80), ...]
```

核对这几项：

- `family` 是否符合权重（YOLO26 应为 `reg_max=1`）
- `layout` 是否符合那块 RKNN（fork 是 `split6`，官方是 `fused`）
- 图片数量是否对

中间若出现：

```text
query RKNN_QUERY_INPUT_DYNAMIC_RANGE error ... static shape
```

静态 shape 模型的常见警告，**可忽略**。

### 5.2 单图进度行

前 3 张、每 32 张、以及出现 FAIL 的图会打印：

```text
[1/128] 000001_....jpg | safety_helmet_all_i8:pass 2/2/2 iou=0.968 dconf=0.019
```

| 字段 | 含义 |
|------|------|
| `pass/warn/fail` | **这一张图**的判定等级（见下一节） |
| `2/2/2` | PT 框数 / RKNN 框数 / 配对成功数 |
| `iou` | 本图已配对框的 **平均 IoU** |
| `dconf` | 本图已配对框的 **最大 \|分数差\|** |

### 5.3 单图 PASS / WARN / FAIL 怎么判

这是脚本对**每一张图**单独打的等级，用来扫图找异常；**不是业界标准，也不是最终验收依据**。最终以整集数字为准。

配对前提：同类，且 IoU ≥ `--match-iou`（默认 0.5）。记：

- `n_pt` / `n_rknn` / `n_match`：本图 PT 框数 / RKNN 框数 / 配对成功数
- `min_iou`：本图已配对框中 **最低** IoU
- `max_dconf`：本图已配对框中 **最大** `|分数差|`

判定顺序：

| 等级 | 条件（与代码一致） | 典型情况 |
|------|--------------------|----------|
| **PASS** | `n_pt == n_rknn == n_match`，且（两边都没框，**或** `min_iou ≥ 0.9` 且 `max_dconf ≤ 0.15`） | 框数完全一致、全部配对，框几乎重合、分数接近 |
| **WARN** | ① 框数完全对上且全部配对，但 `min_iou < 0.9` 或 `max_dconf > 0.15`；**或** ② `n_match ≥ max(1, floor(0.8 × max(n_pt, n_rknn)))`（框数不完全一致，但配对率仍 ≥ 约 80%） | i8 常见：框略飘、分数略漂；或阈值边界多/少 1 个框 |
| **FAIL** | 不满足上面两条，即配对成功数不到「较多那侧框数」的约 80% | 漏检/多检较多，或框对不上 |

示例：

- PT 2 框、RKNN 2 框、配对 2，且最差 IoU=0.97、最大分数差=0.02 → **PASS**
- PT 2 框、RKNN 2 框、配对 2，但最差 IoU=0.85 → **WARN**（框数对了，重叠不够严）
- PT 5 框、RKNN 6 框、配对 5（5 ≥ 0.8×6）→ **WARN**
- PT 7 框、RKNN 4 框、配对 4（4 < 0.8×7）→ **FAIL**

跑完后的汇总行形如：

```text
PASS=113 (88.3%)  WARN=14 (10.9%)  FAIL=1 (0.8%)
```

含义：128 张里有多少张被判成上述三档。i8 出现少量 WARN 很常见。**不要用 PASS 张数当最终结论**，优先看下一节的整集数字。FAIL 最多再列出 12 张文件名，方便打开可视化。

### 5.4 整集指标（跑完后看这里）

脚本对每块 RKNN 打印一份数字报表（含义写在本文，代码里不再复述）：

```text
=== PT vs safety_helmet_all_i8 (split6) ===
  n_images        128
  n_pt / n_rknn   528 / 523
  n_match         520
  match_rate      0.9848
  recall_vs_pt    0.9848
  extra_rate      0.0057
  mean_iou        0.9646
  p5_iou          0.9231
  mean|dconf|     0.0204
  max|dconf|      0.1659
  high_conf(≥0.50)  recall=1.0000  mean_iou=0.9656  mean|dconf|=0.0193  (pt_high=496, matched=496)
  结论            转换可用：与参考模型检出高度一致。
  PASS=113 (88.3%)  WARN=14 (10.9%)  FAIL=1 (0.8%)
```

#### 各指标含义

| 指标 | 含义 | 怎么看 |
|------|------|--------|
| `n_images` | 参与对比的图片数 | 应等于目录里的图数 |
| `n_pt` / `n_rknn` | 两侧 NMS 后检出框 **总数**（整集按框累加，不是按图平均） | 差太多说明漏检/多检 |
| `n_match` | 同类且 IoU≥`--match-iou`、成功配对的框数 | |
| `match_rate` | `n_match / max(n_pt, n_rknn)` | **优先看**。两边框对上的比例，越接近 1 越好；i8 常见 0.95+ |
| `recall_vs_pt` | `n_match / n_pt` | PT 有的框 RKNN 找回了多少（相对漏检）。**越大越好**，理想接近 1；偏低说明相对 PT 漏检多 |
| `extra_rate` | `(n_rknn - n_match) / n_rknn` | RKNN 多出来、没对上 PT 的比例（相对多检）。**越小越好**，理想接近 0；偏高说明 RKNN 多检多 |
| `mean_iou` | 已配对框平均重叠 | **优先看**。1=完全重合；>0.9 通常很好 |
| `p5_iou` | 已配对 IoU 的 5 分位 | 差的那一档有多差 |
| `mean\|dconf\|` | 已配对框平均 \|分数差\| | **优先看**。i8 常见 0.02~0.05 |
| `max\|dconf\|` | 已配对框最大分数差 | 单框极端值，只用来排查 |
| `high_conf(≥0.50)` | 只统计 PT conf≥0.5 的主目标：找回率 / 平均 IoU / 平均分数差 | 看高分框是否稳 |

这些是 **按框汇总**：一张图 20 个框的权重大于一张图 1 个框。比「128 张里 PASS 了多少张」更接近真实检出差异。

### 5.5 整集结论怎么读

脚本根据整集数字给一句结论（**不是 mAP**）。

#### 总结论（最重要）

| 结论文案 | 什么时候算过 | 含义 |
|----------|--------------|------|
| **转换可用** | match_rate≥0.95 且 mean_iou≥0.90 且 mean\|dconf\|≤0.05 | 和 `.pt` 高度一致，直接过，可部署 |
| **i8 可接受** | match_rate≥0.85 且 mean_iou≥0.80（但不满足「转换可用」） | 有量化漂移，勉强过，建议抽看 FAIL 图 |
| **偏差较大** | 不满足以上 | 不过，要查导出头、预处理、后处理或量化 |

**三段链路各自看自己的结论**：`.pt`↔`.onnx` 期望「**导出可用**」（见 `COMPARE_PT_ONNX.md`）；`.onnx`↔`.rknn` 期望「**转换可用**」（见 `COMPARE_ONNX_RKNN.md`）；本环节（`.pt`↔`.rknn`）FP 期望「转换可用」，i8 最好也是「转换可用」，至少「i8 可接受」。本项目 safety_helmet_all：三段都是「可用」→ 整体通过（实测数字见第 6 节）。

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
| p5_iou | 配对框里最差一档的重叠度，辅助定位 |
| PASS/WARN/FAIL 张数 | 只用来扫图，不要只看这个过不过 |

**一句话记法**：先看 match_rate，再看 mean_iou，最后看 mean\|dconf\|；FP 三项达标 = 转换可用，i8 到「i8 可接受」也基本能部署；「偏差较大」按第 7 节分段排查。

### 5.6 可视化图（加了 `--img_save`）

```text
views: compare_result_pt_vs_rknn
```

左图 PT，右图 RKNN（多块 RKNN 会继续往右拼）。默认只存前 `--save_limit` 张（默认 5）。

---

## 6. 经验参考（本项目实测）

数据集：`/userdata/jovan/code/rk3588/dataset/helmet/`，128 张。  
权重：`safety_helmet_all.pt` ↔ `safety_helmet_all_i8.rknn`（YOLO26 fork 6 路 i8）。  
阈值：`conf=0.25`，NMS IoU=`0.7`，匹配 IoU=`0.5`。

同批次链路结论（便于对照）：

| 环节 | 脚本 | 结论 |
|------|------|------|
| `.pt`↔`.onnx` | `compare_pt_onnx_yolo.py` | **导出可用**（match_rate=1.0，cosine=1.0） |
| `.onnx`↔`.rknn` | `compare_onnx_rknn_yolo.py` | **转换可用**（match_rate=0.9848） |
| `.pt`↔`.rknn` | 本脚本 | **转换可用**（与上一段数字一致） |

命令：

```bash
python3 compare_pt_rknn_yolo.py \
  --pt safety_helmet_all.pt \
  --rknn safety_helmet_all_i8.rknn \
  --source /userdata/jovan/code/rk3588/dataset/helmet/
```

### fork 6 路 i8：`safety_helmet_all_i8.rknn`

| 指标 | 数值 |
|------|------|
| layout | `split6` |
| n_pt / n_rknn / n_match | 528 / 523 / 520 |
| **match_rate** | **0.9848** |
| recall_vs_pt | 0.9848 |
| extra_rate | 0.0057 |
| **mean_iou** | **0.9646** |
| p5_iou | 0.9231 |
| **mean\|dconf\|** | **0.0204** |
| high_conf recall | 1.0000（496/496） |
| PASS / WARN / FAIL | 113 / 14 / 1 |
| 结论 | **转换可用** |

唯一 FAIL 图：`000064_....jpg`，PT 7 框、RKNN 4 框（阈值边界漏检），已配对框 IoU 仍 ≥0.90。

### 官方 fused i8：`ultralytics_src/safety_helmet_all-rk3588_i8.rknn`

| 指标 | 数值 |
|------|------|
| layout | `fused` |
| n_pt / n_rknn / n_match | 528 / 990 / 517 |
| **match_rate** | **0.5222** |
| recall_vs_pt | 0.9792（主目标大多还能找回） |
| extra_rate | **0.4778**（多检近一半） |
| **mean_iou** | **0.8102** |
| p5_iou | 0.6502 |
| mean\|dconf\| | 0.0335 |
| PASS / WARN / FAIL | 15 / 53 / 60 |
| 结论 | **偏差较大，不建议当部署模型** |

这份 fused 对 PT 的高分框还能对上，但 RKNN 多出大量框、IoU 也偏低。排错应回到「是不是同一份 ONNX、框是否归一化、量化是否把分数尺度打歪」。

---

## 7. 线上 RKNN 和 `yolo predict` 对不上怎么排查

适用场景：线上 `.rknn` 检错了 / 漏了 / 框飘了，你用同一张图跑 `yolo predict`，两边对不上。

先固定同一张问题图，再按下面顺序缩小范围。

### 7.1 先对齐「比什么」

两边必须尽量同一条件，否则看起来像转换坏了，其实是设置不同：

| 项 | 建议 |
|----|------|
| 同一张原图 | 不要一边裁过、一边没裁 |
| 同类阈值 | 例如都是 `conf=0.25`，NMS IoU 尽量一致（Ultralytics 常见 0.7） |
| 同类别名 | `head` / `helmet` 是否对上 |
| 同一块权重链路 | 线上必须是 fork 那份 `safety_helmet_all_i8.rknn`，不要拿错成官方 fused |

当前这份 YOLO26 头盔模型上，原版 `predict` 和 fork RKNN **大体一致**（前 10 张实测几乎全配对、IoU≈0.97）。  
所以「predict 对、线上错」多数不是「路径天生不同」，而是 **板端实现或部署配置** 有问题。

### 7.2 排查顺序（推荐）

**第 1 步：板上用对比脚本，确认「转换本身」还好不好**

```bash
cd /userdata/jovan/code/rk3588/rknn_model_zoo/examples/yolo11/python

python3 compare_pt_rknn_yolo.py \
  --pt safety_helmet_all.pt \
  --rknn safety_helmet_all_i8.rknn \
  --source /path/to/问题图.jpg \
  --img_save
```

| 结果 | 说明 | 下一步 |
|------|------|--------|
| 脚本里 PT 和 RKNN **一致**，但都和「你认为的正确答案」不符 | 权重本身就会错 | 看数据/训练，不是转换问题 |
| 脚本里 PT 和 RKNN **一致**，且和 `yolo predict` 也接近，但**线上服务**仍不对 | 转换 OK，坏在线上推理代码 | 查预处理 / 后处理 / 是否加载错模型（第 2 步） |
| 脚本里 PT 对、RKNN 明显差 | 转换或这块 `.rknn` 有问题 | 走第 3 步分段对比 |

**第 2 步：对照线上代码（转换已 OK、线上却不对时）**

按清单逐项核对板上业务代码是否与对比脚本一致：

1. **是不是同一份 `.rknn` 文件**（路径、版本、md5）
2. **预处理**：letterbox 到 640、保持比例、黑边；BGR→RGB；喂 `NHWC uint8 (1,640,640,3)`  
   （不要自己再 `/255`，转换一般是 `mean=0, std=255`）
3. **输出是 6 路**：每个尺度 `reg(1,4,H,W)` + `cls(1,nc,H,W)`
4. **cls 必须 sigmoid**（fork 导出的是 logits）
5. **解码 + NMS** 阈值是否和线上一致（常见 conf=0.25）
6. **坐标有没有映射回原图**（letterbox 逆变换）
7. 有没有把官方 fused 的后处理（例如框 ×640、单输出 `(1,6,8400)`）误用到 fork 6 路上

**第 3 步：转换本身就差时，分段定位**

```text
.pt  --(compare_pt_onnx_yolo.py)-->  .onnx  --(compare_onnx_rknn_yolo.py)-->  .rknn
```

- `.pt`↔`.onnx` 就差 → 导出有问题  
- `.onnx`↔`.rknn` 才差 → 量化 / convert 配置有问题（尤其看 cls 是否被量化打成全 0）  
- 两段都过，但线上仍差 → 回到第 2 步查业务代码

### 7.3 怎么读「一致 / 不一致」

不要要求像素级完全一样。i8 下常见：

- 框 IoU > 0.9、分数差 < 0.05 → 可当一致  
- 主目标都在，只是阈值边界多/少 1 个低分框 → 多半可接受  
- 类别错、主目标漏、框飞掉、分数接近 0 → 真不一致，要修

也可用对比脚本看整集：`match_rate` / `mean_iou` / `mean|dconf|`（含义见第 5.4 节）。

---

## 8. 常见问题排查

### 8.1 `FileNotFoundError: .pt 不存在` / `.rknn 不存在`

路径相对脚本当前工作目录。先 `cd` 到 `examples/yolo11/python`，或写绝对路径。

### 8.2 `仅支持 detect`

本脚本只做检测头。分类/分割权重不要拿来跑。

### 8.3 `PT 拆头失败` / `找不到 one2many`

YOLO26 fork 应对 `--head one2many`（默认）。  
若 RKNN 其实是官方 fused，layout 会变成 `fused`，PT 会走 concat，不要强行 `--pt-mode split`。

### 8.4 fused 框飞到图外，或缩成一团

官方 YOLO26 RKNN 的框经常是 0~1。`--box-scale auto` 会处理。  
若仍不对，试 `--box-scale on` 或 `off` 对比可视化。

### 8.5 框对得上，但分数差很大

i8 轻微漂移（0.02~0.05）可接受。  
若 `mean|dconf|` 到 0.2+，或某一类分数接近 0：优先怀疑分类头被整段 int8 量化打坏（fused `(1,6,8400)` 把大数值 box 和小分数挤在同一张量里时尤其容易）。这类问题用 `compare_onnx_rknn_yolo.py` 看 score 通道更清楚。

### 8.6 只有一边有检出

可能原因：

- 量化后分数跨过 0.25 阈值
- PT 对齐错了头（one2one vs one2many）
- 预处理 / 输入尺寸不一致

可临时把 `--conf` 降到 `0.1` 做诊断，正式结论仍看默认 0.25。

### 8.7 想算 mAP

当前 `dataset/helmet/` **没有 labels**，本脚本算不了 mAP。  
有 COCO/YOLO txt 标注后再单独做 GT 评估；和「转换是否等价」是两件事。

### 8.8 板子上 `import torch` 很慢或 OOM

`.pt` 走 CPU 时 128 张大约 2~3 分钟（本机实测两块 RKNN 共 166s）。  
先 `--max_images 2` 确认能跑通。

---

## 9. 推荐使用流程

1. **先跑少量图（例如 `--max_images 2 --img_save`）**  
   确认路径、layout 识别、可视化框大致重合。
2. **跑全集，先看整集数字**  
   优先 `match_rate` / `mean_iou` / `mean|dconf|`，不要只看 PASS 张数。
3. **fork i8 应用**  
   期望「转换可用」。高分框 recall 应接近 1。
4. **官方 fused 仅作对照**  
   若 match_rate 明显偏低、extra_rate 很高，不要部署这份。
5. **分段排错**（整集偏差大时）  
   - `.pt`↔`.onnx`：`compare_pt_onnx_yolo.py`（见 `COMPARE_PT_ONNX.md`）  
   - `.onnx`↔`.rknn`：`compare_onnx_rknn_yolo.py`（见 `COMPARE_ONNX_RKNN.md`，含 tensor / 分数通道）
6. **抽看 FAIL 图**  
   打开 `compare_result_pt_vs_rknn/cmp_*.jpg`。
7. **线上和 predict 对不上**  
   按第 7 节：先对比脚本分清「转换 / 线上代码 / 权重本身」。

---

## 10. 一句话结论怎么读

- **转换可用**：`.pt` 与 RKNN 检出高度一致，可以当转换成功  
- **i8 可接受**：主目标还在，有量化漂移，结合 FAIL 图决定是否上线  
- **偏差较大**：不要把这块 RKNN 当成 `.pt` 的等价模型，先查导出和量化

对当前头盔 YOLO26：**部署用 fork 6 路 i8；官方 fused i8 不要当最终模型。**
