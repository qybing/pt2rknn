# YOLO26 → 6 路 raw ONNX → INT8 RKNN 导出说明

本文记录在转换机上，用 [yolo26_rknn_ultralytics](https://gitee.com/jovan_qiao/yolo26_rknn_ultralytics) 导出 **可用 INT8 RKNN** 的完整步骤。

目标产物：

| 文件 | 形态 | 用途 |
|------|------|------|
| `safety_helmet_all.onnx` | **6 路** `reg/cls × 3 尺度`，cls 为 **logits** | 源模型 |
| `safety_helmet_all_i8.rknn` | 同上 6 路，INT8 | 板上 NPU 推理 |

板端后处理需对 cls 做 `sigmoid`，再解码框 + NMS（**不能走模型内 end2end**）。

同类文档：`COMPARE_PT_ONNX.md` / `COMPARE_ONNX_RKNN.md` / `COMPARE_PT_RKNN.md`。

---

## 0. 为什么必须按本文做

| 错误做法 | 结果 |
|----------|------|
| 用 pip 官方 Ultralytics（如 8.4.146）`format=rknn` | 常得到单输出 `(1, 6, 8400)` |
| 对单输出 ONNX 整段 INT8 | **score 被量化成 0**，板上 0 检出 |
| 用默认 COCO 小校准集、业务是头盔 | 量化更易漂 |

正确做法：**仓库改过的 Detect 导出 6 路 raw** → 用**业务校准集**转 i8 → 板端 split6 后处理。

---

## 1. 环境准备（转换机）

仓库路径下文以 `/root/code/rknn/yolo26_rknn_ultralytics` 为例，按你机器改路径。

### 1.1 克隆并安装仓库 ultralytics

```bash
cd ~/code/rknn
git clone https://gitee.com/jovan_qiao/yolo26_rknn_ultralytics
cd ~/code/rknn/yolo26_rknn_ultralytics

# 卸掉官方包，避免仍加载 site-packages 里的 8.4.146
pip uninstall ultralytics -y

pip install -U pip setuptools wheel
pip install .
```

### 1.2 关于 `PYTHONPATH`（可选但建议）

```bash
export PYTHONPATH=/root/code/rknn/yolo26_rknn_ultralytics:$PYTHONPATH
```

含义：让 Python **优先**从本仓库加载 `ultralytics`。

| 情况 | 要不要设 |
|------|----------|
| 已 `pip uninstall` 官方包，且 `pip install .` 成功 | 可省略 |
| 机器上还可能混着官方 ultralytics | **建议每次导出前都设** |

### 1.3 自检（必须过）

```bash
python -c "import ultralytics; print(ultralytics.__version__, ultralytics.__file__)"
# 期望：版本约 8.4.9
# 路径含 yolo26_rknn_ultralytics，或 site-packages 但是 pip install . 装进去的这份 fork

python -c "import inspect, ultralytics.nn.modules.head as h; print('rknn' in inspect.getsource(h.Detect.forward))"
# 必须输出：True
```

若版本是 **8.4.146** 且路径在无关的 `site-packages`，后面一定会导出成单输出，**不要继续转 i8**。

### 1.4 Torch / Torchvision 配对

导出依赖匹配的 CPU 轮子（曾出现 `torchvision::nms does not exist`）：

```bash
pip uninstall torch torchvision torchaudio -y
pip cache purge

pip install torch==2.4.0+cpu torchvision==0.19.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
```

注意：`torchvision` 版本字符串应带 **`+cpu`**。验证：

```bash
python - <<'EOF'
import torch, torchvision
print("torch", torch.__version__)
print("tv", torchvision.__version__)
from torchvision.ops import nms
print("nms ok")
EOF
```

若仍因 SAM 导入拖死 torchvision，可临时注释仓库内 `ultralytics/models/__init__.py` 里的 `from .sam import SAM`（仅导出检测模型时）。

### 1.5 其它依赖

- `onnx`
- `rknn-toolkit2`（与板端 runtime 大版本接近，例如 2.3.x）
- 权重：`weight/safety_helmet_all.pt`
- 校准列表：如 `/root/code/rknn/helmet_calib.txt`（每行一张图路径，建议几十～一两百张头盔图）

---

## 2. 导出 6 路 raw ONNX

```bash
cd ~/code/rknn/yolo26_rknn_ultralytics
export PYTHONPATH=/root/code/rknn/yolo26_rknn_ultralytics:$PYTHONPATH

yolo export model=weight/safety_helmet_all.pt format=rknn name=rk3588
```

### 2.1 成功日志特征

```text
Ultralytics 8.4.9
WARNING  RKNN export does not support end2end models, disabling end2end branch.
output shape(s) ((1, 4, 80, 80), (1, 2, 80, 80), (1, 4, 40, 40), (1, 2, 40, 40), (1, 4, 20, 20), (1, 2, 20, 20))
RKNN: starting export with torch ...
RKNN: export success, saved as 'weight/safety_helmet_all.onnx'
```

说明：

- **会关掉 end2end**：NPU 只要 raw，sigmoid/NMS 放 CPU
- 当前 fork 导出头是 **`cv2/cv3`（one2many）**；PT↔ONNX 对比请用 `--head one2many`

### 2.2 失败日志特征（立刻停）

```text
Ultralytics 8.4.146
ONNX: slimming with onnxslim ...
output shape(s) (1, 6, 8400)
```

这是官方 fused 单输出，**不要**再拿去 `convert.py --dtype i8`。

### 2.3 立刻检查 ONNX 输出

```bash
python3 - <<'EOF'
import onnx
m = onnx.load("weight/safety_helmet_all.onnx")
print("outputs:", len(m.graph.output))
for o in m.graph.output:
    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(o.name, dims)
EOF
```

**合格（6 个）：**

```text
output0_reg [1, 4, 80, 80]
output0_cls [1, 2, 80, 80]
output1_reg [1, 4, 40, 40]
output1_cls [1, 2, 40, 40]
output2_reg [1, 4, 20, 20]
output2_cls [1, 2, 20, 20]
```

**不合格：** 只有 `output0 [1, 6, 8400]` → 重做第 1～2 节。

---

## 3. 转为 INT8 RKNN

```bash
cd ~/code/rknn/yolo26_rknn_ultralytics

python rknn_export/convert.py \
  --model-path weight/safety_helmet_all.onnx \
  --platform rk3588 \
  --dtype i8 \
  --data-path /root/code/rknn/helmet_calib.txt \
  --rknn-path weight/safety_helmet_all_i8.rknn
```

### 3.1 合格 build 日志

应对 **每个** 输出分别提示 dtype→int8，例如：

```text
output0_reg ... int8
output0_cls ... int8
output1_reg ... int8
...
```

### 3.2 危险日志（单输出死路）

```text
The default output dtype of 'output0' is changed from 'float32' to 'int8'
```

且模型只有一个 `output0` → score 极易全 0。

`mean=0, std=255` 与板端 NHWC uint8 预处理一致，一般不要改。

---

## 4. 拷到板子并验收

拷贝：

- `safety_helmet_all.onnx`（6 路）
- `safety_helmet_all_i8.rknn`

板端快速探针（score 不能全 0）：

```bash
cd /userdata/jovan/code/rk3588/rknn_model_zoo/examples/yolo11/python
python3 - <<'EOF'
from rknnlite.api import RKNNLite
import numpy as np
r = RKNNLite()
r.load_rknn("safety_helmet_all_i8.rknn")
r.init_runtime()
outs = r.inference(inputs=[np.zeros((1, 640, 640, 3), np.uint8)])
print("n_outs", len(outs))
for i, o in enumerate(outs):
    o = np.asarray(o)
    print(i, o.shape, "min", float(o.min()), "max", float(o.max()), "nz", int(np.count_nonzero(o)))
r.release()
EOF
```

- **合格：** `n_outs == 6`，cls 头 `nz > 0` 且有负数 logits 很正常  
- **不合格：** `n_outs == 1` 且 shape `(1,6,8400)`，score 通道全 0

完整对比：

```bash
python3 compare_onnx_rknn_yolo.py \
  --onnx safety_helmet_all.onnx \
  --rknn safety_helmet_all_i8.rknn \
  --source /userdata/jovan/code/rk3588/dataset/helmet/ \
  --max_images 5 \
  --img_save
```

PT↔ONNX（转换机，需 PyTorch）：

```bash
python compare_pt_onnx_yolo.py \
  --pt weight/safety_helmet_all.pt \
  --onnx weight/safety_helmet_all.onnx \
  --source <测试图目录> \
  --head one2many
```

（若脚本名仍是旧的 `compare_pt_onnx_yolo26.py`，参数同上。）

---

## 5. 一页清单（复制用）

```bash
# --- 环境 ---
cd ~/code/rknn/yolo26_rknn_ultralytics
pip uninstall ultralytics -y
pip install .
export PYTHONPATH=/root/code/rknn/yolo26_rknn_ultralytics:$PYTHONPATH
python -c "import ultralytics; print(ultralytics.__version__, ultralytics.__file__)"
python -c "import inspect, ultralytics.nn.modules.head as h; print('rknn' in inspect.getsource(h.Detect.forward))"

# --- 导出 ONNX ---
yolo export model=weight/safety_helmet_all.pt format=rknn name=rk3588
python3 - <<'EOF'
import onnx
m = onnx.load("weight/safety_helmet_all.onnx")
print("outputs:", len(m.graph.output))
for o in m.graph.output:
    print(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
EOF
# 必须 6 个输出，再往下

# --- INT8 ---
python rknn_export/convert.py \
  --model-path weight/safety_helmet_all.onnx \
  --platform rk3588 \
  --dtype i8 \
  --data-path /root/code/rknn/helmet_calib.txt \
  --rknn-path weight/safety_helmet_all_i8.rknn
```

---

## 6. 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| 导出是 `(1,6,8400)` | 加载了官方 ultralytics | 设 `PYTHONPATH` / 重装仓库 / 再自检 |
| i8 score 全 0 | 单输出整段 int8 | 必须先有 6 路 ONNX |
| `torchvision::nms does not exist` | torch/tv 不匹配 | 同渠道装 `2.4.0+cpu` + `0.19.0+cpu` |
| PT↔ONNX 严阈值 FAIL，但 cos≈0.999 | 对比错成 one2one | 用 `--head one2many` |
| 板上要开 end2end？ | RKNN 图里没有 | 用 raw + sigmoid + NMS |

---

## 7. 和业务部署的关系

```text
图像 → letterbox → RKNN(6×raw) → sigmoid(cls) → 解码 → NMS → 框
```

- NPU：**不做** end2end / NMS  
- CPU：split6 后处理（与 `compare_onnx_rknn_yolo.py` 一致）  
- 短期若 i8 异常，可先用已验证的 **FP** RKNN
