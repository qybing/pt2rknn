# .pt 权重转 RKNN 完整教程(YOLO11 案例)

> 适用对象:把训练好的 Ultralytics YOLO11 `.pt` 模型部署到 RK3588(流程同样适用于 RK3562/3566/3568/3576/3576 等 RKNPU2 平台)。
> 本文命令全部来自瑞芯微官方仓库实测(2025 年核实,官方最新版本为 RKNN-Toolkit2 v2.3.2),官方文档索引见文末附录。

---

## 仓库结构与配套脚本

```text
pt2rknn/
├── pt转换rknn.md         ← 本教程
├── make_calib.py         ← 量化校准集清单生成脚本(第 4 节使用,纯 Python 标准库,无需额外安装)
├── compare_pt_onnx.py    ← .pt 与 .onnx 转换精度对比脚本(第 3 节关卡 A 使用)
├── compare_onnx_rknn.py  ← ONNX 与 RKNN 输出一致性对比脚本(第 6.2 节关卡 B/C 使用;
│                             注意:需拷到 rknn_model_zoo/examples/yolo11/python/ 目录下运行)
├── images/               ← 对比脚本默认的测试图目录(放你自己的测试图,当前有示例图)
└── weight/               ← (按需创建)按脚本默认路径放 helmet_y11s_best.pt 和 helmet_y11s_best.onnx;
                             也可不建目录,直接用 --pt/--onnx 参数指定
```

---

## 0. 总览:整体流程与机器分工

```
PC / x86_64 Linux 服务器                              RK3588 板子
─────────────────────────────────────                ──────────────────
① 环境:装 rknn-toolkit2
② .pt → ONNX(官方 fork 导出)
③ 验证关卡A:.pt vs .onnx 对比
④ 准备量化校准集(图片 + txt 清单)
⑤ ONNX → RKNN(convert.py,fp/i8)
⑥ 模拟器测试(fp16 vs i8 对比)                        ⑦ 部署上板运行
```

| 环节 | 在哪做 | 说明 |
|---|---|---|
| ①~⑥ 全部转换与测试 | **PC / x86_64 Linux 服务器**(物理机、VM、Docker 容器均可) | 纯 CPU 计算,不需要 GPU、不需要板子 |
| ⑦ 部署运行 | **RK3588 板子** | 板上装运行库和 lite2 或 C demo |

**版本配套三原则**(官方反复强调,出问题先查这里):

1. PC 端 `rknn-toolkit2`、板端 `librknnrt.so`、板端 `rknn_server` **三端版本必须一致**(建议统一装最新 v2.3.2);
2. 板端内核 rknpu 驱动建议 **≥ 0.9.2**(官方固件自带,驱动只能随固件升级);
3. `.rknn` 是按目标平台编译的:`target_platform='rk3588'` 生成的模型不能拿到其他芯片上跑。

---

## 1. 环境准备(PC / 服务器上,只做一次)

### 1.1 Python 环境

要求 Ubuntu 18.04/20.04/22.04(或 WSL2、Docker),Python **3.8~3.12**(本教程使用 **3.10**)。建议用独立环境:

```bash
conda create -n toolkit2 python=3.10
conda activate toolkit2
```

### 1.2 下载官方仓库

```bash
mkdir -p ~/code/rknn && cd ~/code/rknn
git clone https://github.com/airockchip/rknn-toolkit2.git --depth 1
git clone https://github.com/airockchip/rknn_model_zoo.git --depth 1
git clone https://github.com/airockchip/ultralytics_yolo11.git --depth 1   # 导出 fork
```

> **`--depth 1` 是什么意思**:浅克隆(shallow clone),只拉取仓库**最新一次提交**的代码,不下载完整的提交历史。好处是下载量小、克隆快(这些仓库带历史很大,我们只关心最新代码);代价是不能查看/切换历史版本。如果以后需要 checkout 旧版本对照,去掉该参数重新完整克隆即可。

三个仓库分工:

| 仓库 | 用途 |
|---|---|
| `rknn-toolkit2` | 转换工具安装包 + PDF 文档 + 板端运行库(`rknpu2/runtime/`) |
| `rknn_model_zoo` | `examples/yolo11/` 里的转换脚本、Python/C demo、FAQ |
| `ultralytics_yolo11` | 把你的 `.pt` 导出成 NPU 友好的 ONNX |

### 1.3 安装 RKNN-Toolkit2(使用官方仓库里的安装包)

安装包就在刚克隆的 rknn-toolkit2 仓库里:

- **仓库地址**:https://github.com/airockchip/rknn-toolkit2
- **安装包位置**:`rknn-toolkit2/rknn-toolkit2/packages/x86_64/`(注意:仓库目录下还嵌套一层同名目录;arm64 平台的包在 `packages/arm64/`)
- **包含内容**:
  - 各 Python 版本的 whl 安装包,如 Python 3.10 对应 `rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
  - 依赖清单文件,如 Python 3.10 对应 `requirements_cp310-2.3.2.txt`(cp36~cp312 每个版本一份)

按你的 Python 版本选对应文件(Python 3.10 → `cp310`):

```bash
cd ~/code/rknn/rknn-toolkit2/rknn-toolkit2    # 注意进入仓库内嵌套的同名目录

# ① 先装依赖清单
pip install -r packages/x86_64/requirements_cp310-2.3.2.txt

# ② 再装 toolkit2 本体(whl 包)
pip install packages/x86_64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# ③ 验证安装
python -c "from rknn.api import RKNN; print('ok')"
```

- 换 Python 版本时替换 `cp310`/`3.10`(如 Python 3.11 → `cp311`/`requirements_cp311-2.3.2.txt`);
- 有外网时也可用 PyPI 在线装(依赖自动带齐):`pip install rknn-toolkit2 -i https://pypi.org/simple`;已装旧版升级:`pip install rknn-toolkit2 --upgrade`,升级后之前转的模型要重转,保持版本一致。

---

## 2. 第一步:`.pt` → ONNX(必须用官方 fork!)

> ⚠️ **不要用 ultralytics 官方的 `yolo export`**!官方 fork 在导出时对模型做了三处关键修改
> (移除后处理结构、DFL 移到图外、增加置信度求和分支),demo 的后处理代码是按这个结构写的。
> 用错导出方式,轻则跑不准,重则全程报错。官方 FAQ 明确说:自己模型跑不对,先检查是否按官方 fork 导出。

### 2.1 安装 fork(注意:没有 requirements.txt!)

新版 ultralytics 用 `pyproject.toml` 管理依赖,在 fork 仓库根目录:

```bash
cd ~/code/rknn/ultralytics_yolo11
pip install -e .          # 自动装齐依赖并注册本地包,装完无需设 PYTHONPATH
```

**装完必须验证用的是 fork 的代码**(防止 PyPI 版混入):

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
# 输出必须指向 ~/code/rknn/ultralytics_yolo11/ultralytics/__init__.py
# 如果指向 site-packages/ultralytics/...,说明环境混了,重开一个干净环境重装
```

导出时还会用到 onnx/onnxslim,一般运行时自动安装;离线环境手动 `pip install onnx onnxslim`。

### 2.2 导出

```bash
# ① 修改 ./ultralytics/cfg/default.yaml 中的 model 字段,指向你的权重:
#    model: /root/code/rknn/helmet_y11s_best.pt
vi ultralytics/cfg/default.yaml

# ② 导出(检测/分割/姿态/旋转框任务的 pt 都支持)
python ./ultralytics/engine/exporter.py     # 生成 best.onnx(与 pt 同目录同名)
```

> 用新版 ultralytics 训练的 `.pt` 若在 fork 里加载报错,在 fork 环境中重新加载权重导出即可(结构相同,不用重训)。

---

## 3. 第二步:验证关卡 A(`.pt` ↔ `.onnx` 对比)

导出这一步理论上是**无损**的,输出应当几乎完全一致。有可见差异 = 出错了,不要往下走。

### 3.1 用配套脚本 compare_pt_onnx.py(推荐)

本仓库自带的对比脚本,对每张测试图做**两层对比**并自动判 PASS/FAIL:

- **张量层**:同一输入(letterbox 到固定尺寸)下,`.pt` 与 `.onnx` 解码后的输出逐元素对比——cosine 相似度、box/cls 的平均误差(MAE)与最大误差;
- **检测层**:两者各自 NMS 后,检测框按 IoU 配对——比对框数量、类别一致性、IoU、置信度差 |Δconf|;
- **自动适配两种 ONNX 输出布局**:官方 ultralytics 端到端单输出 `(1, 4+nc, N)` 或 `(1, N, 4+nc)`,以及 airockchip fork 的拆头输出(每尺度 box(64通道)+cls(类别数),共 6/9 路);
- **可视化**:每张图输出三联图(绿框=pt、红框=onnx、第三张两者叠加),存到 `result_views/pt_onnx_compare/`。

```bash
# 依赖:训练侧环境即可(torch + ultralytics + onnxruntime + opencv,Windows/Linux 均可运行)
pip install ultralytics onnxruntime opencv-python

# 方式一:按脚本默认路径 —— 权重放 weight/ 子目录,测试图放 images/
python compare_pt_onnx.py

# 方式二:显式指定路径
python compare_pt_onnx.py --pt best.pt --onnx best.onnx --source ./test_imgs

# 图较多、只想快速看结论时可关闭可视化
python compare_pt_onnx.py --no-vis
```

**参数说明**(compare_pt_onnx.py):

| 参数 | 默认值 | 含义与可设置值 |
|---|---|---|
| `--pt` | `weight/helmet_y11s_best.pt` | 原始 Ultralytics 权重路径 |
| `--onnx` | `weight/helmet_y11s_best.onnx` | 待对比的 ONNX 模型路径 |
| `--source` | `images/` | 测试图文件或目录(目录则逐张对比,支持 jpg/jpeg/png/bmp/webp/tif) |
| `--output` | `result_views/pt_onnx_compare/` | 可视化三联图输出目录 |
| `--imgsz` | 自动读 ONNX 输入 shape | 推理输入边长;一般不用传,仅非 640 模型需确认 |
| `--conf` / `--iou` | 0.25 / 0.7 | NMS 置信度 / IoU 阈值,设成与业务部署一致即可 |
| `--match-iou` | 0.5 | pt/onnx 检测框配对的最小 IoU |
| `--min-cosine` | 0.9999 | 张量 cosine 判定下限 |
| `--max-box-err` | 0.05 | 框坐标最大允许偏差(px,模型画布上) |
| `--max-cls-err` | 1e-3 | 类别分数最大允许偏差 |
| `--min-iou` / `--max-dconf` | 0.99 / 1e-3 | 配对框 IoU 下限 / 置信度差上限 |
| `--no-vis` | 关闭 | 加上则跳过可视化,只输出终端指标 |

**判读标准**(脚本内置阈值,可用参数调整):

| 指标 | 默认阈值 | 含义 |
|---|---|---|
| cosine | ≥ 0.9999 | 整张输出张量与 .pt 的余弦相似度 |
| box_max | ≤ 0.05 px | 框坐标最大偏差(模型画布上) |
| cls_max | ≤ 1e-3 | 类别分数最大偏差 |
| 检测框配对 | 数量一致、min_iou ≥ 0.99、max\|Δconf\| ≤ 1e-3 | NMS 后逐框配对 |

> **为什么阈值可以这么严**:关卡 A 两侧都是 FP32 计算,差异只来自算子实现与浮点运算顺序(约 1e-6 量级),不是"精度损失"。脚本固定用 CPU 推理对比,结果确定、可复现,因此采用业界"无损转换"校验的严格口径(cosine + 逐元素误差)是合理的。偶发单点超差时,结合 MAE 数值判断是系统性偏移(要查)还是浮点噪声(可忽略)。

**各转换阶段的参考阈值**(同一套指标,越往后越宽松——业界通行量级):

| 对比阶段 | cosine | box_max | cls_max | min_iou | max\|Δconf\| | 定位 |
|---|---|---|---|---|---|---|
| .pt ↔ .onnx(FP32,关卡 A) | ≥ 0.9999 | ≤ 0.05 px | ≤ 1e-3 | ≥ 0.99 | ≤ 1e-3 | 本脚本默认,判"无损" |
| .onnx ↔ .rknn fp16(关卡 B) | ≥ 0.999 | ≤ 1 px | ≤ 0.01 | ≥ 0.95 | ≤ 0.01 | fp16 舍入误差 |
| fp16 ↔ int8(关卡 C) | ≥ 0.99 | ≤ 2 px | ≤ 0.05 | ≥ 0.9 | ≤ 0.05 | 量化误差,最终以 mAP 兜底 |

> **置信度阈值边界的抖动(误报 FAIL 的常见原因)**:某框置信度恰在 `--conf 0.25` 附近时,浮点微小差异会让它在两个模型一边过阈、一边被滤掉,导致"框数量不一致"被判 FAIL。这属于边界抖动而非转换失败——排查时把 `--conf` 降一档(如 0.1)再比一次,或对边界框(置信度 ±0.02 以内)单独人工确认。
>
> **最终验收以 mAP 兜底**:框级指标是过程把关;业界惯例(如 TensorRT 量化验收)是同一带标注的小验证集上,.pt 与转换后模型各测一次 mAP——关卡 A 差值应 < 0.1 个点,INT8 量化后 mAP 相对下降 ≤ 1%(或绝对 0.5~1 个点)为接受线。

- 终端每张图打印 `[PASS]` / `[FAIL]` 及明细,最后总结 `ALL PASS` / `FAIL`;
- **退出码**:0 = 全部通过,2 = 存在失败(可直接接入 CI 自动把关);
- 不达标时先排查:fork 是否生效(见第 2.1 节)、预处理是否一致,参考第 9 节 FAQ。

### 3.2 备选:官方 model_zoo demo 对比

```bash
cd ~/code/rknn/rknn_model_zoo/examples/yolo11/python

# 官方 demo 支持直接跑 .pt / .onnx / .rknn 三种格式,同一套前后处理,直接对比
python yolo11.py --model_path /root/code/rknn/helmet_y11s_best.pt   --img_save
python yolo11.py --model_path /root/code/rknn/helmet_y11s_best.onnx --img_save
```

- 终端逐张打印 `类别 @ (xmin ymin xmax ymax) 置信度`,画框结果存在 `./result/` 下,肉眼比对两者一致性;
- 也可以先下载官方模型验证工具链:`cd ../model && ./download_model.sh` 后跑 `yolo11n.onnx`,预期输出见该目录 README(`bus @ (91 136 554 440) 0.948` 等)。

---

## 4. 第三步:准备 INT8 量化校准集

INT8 量化需要一组图片让工具统计各层数值分布。**只有转 `i8` 需要,`fp` 不需要**。

- **数量**:100~500 张;场景单一 100 张够用,最多 500(更多无收益,还费内存);
- **内容**:纯图片、**无需标注**;从训练/验证集挑或实际场景抽帧;
- **覆盖度**:所有类别都要有(比如"戴帽"和"未戴帽"两类样本都要放),覆盖不同光照、远近、角度、密度,放一些遮挡/模糊的难例;
- **自定义清单格式**:每行一个图片路径的 txt。

### 4.1 一键生成清单(本仓库脚本 make_calib.py)

脚本就在本教程同目录(`pt2rknn/make_calib.py`),纯 Python 标准库实现,服务器上无需额外安装任何依赖:

```bash
# 全量写入
python3 make_calib.py /root/code/rknn/dataset/helmet /root/code/rknn/helmet_calib.txt

# 图太多时随机抽 200 张(固定 seed 可复现)
python3 make_calib.py /root/code/rknn/dataset/helmet /root/code/rknn/helmet_calib.txt --limit 200

# 检查
head -3 /root/code/rknn/helmet_calib.txt
wc -l  /root/code/rknn/helmet_calib.txt
```

脚本会递归扫描目录(含子目录),自动识别 jpg/jpeg/png/bmp/webp(大小写都认),跳过空文件与非图片文件,写出**绝对路径**。

**参数说明**(make_calib.py,纯标准库实现,任何装了 Python3 的机器都能跑):

| 参数 | 默认值 | 含义与可设置值 |
|---|---|---|
| `img_dir`(第 1 个位置参数,必填) | — | 图像目录,**递归扫描**所有子目录;支持 jpg/jpeg/png/bmp/webp(大小写均可),自动跳过 0 字节文件和非图片文件 |
| `output`(第 2 个位置参数,可选) | `<图像目录>/../helmet_calib.txt` | 输出的清单 txt 路径;**建议写绝对路径**,方便 convert.py 直接引用 |
| `--limit N` | 0(全部保留) | 随机抽取的图片数量上限。校准集 100~500 张即可:整个训练集都丢进去会**大幅增加转换内存和耗时**(此前 OOM 就是这么来的),图多时建议 `--limit 200` |
| `--seed` | 42 | 随机抽样种子。保持不变 → 每次抽到同一批图(可复现);想换一批图换一个 seed 即可 |

### 4.2 把清单配置进 convert.py

打开 `rknn_model_zoo/examples/yolo11/python/convert.py`,把第 4 行默认的 COCO 校准集换成你的:

```python
# DATASET_PATH = '../../../datasets/COCO/coco_subset_20.txt'
DATASET_PATH = '/root/code/rknn/helmet_calib.txt'     # ← 改成你的清单
```

---

## 5. 第四步:ONNX → RKNN 转换

### 5.1 命令与参数

```bash
cd ~/code/rknn/rknn_model_zoo/examples/yolo11/python

# 用法: python convert.py <onnx路径> <平台> [dtype] [输出路径]
# dtype 二选一:i8 = INT8 量化(默认)  |  fp = FP16 不量化
# ⚠️ 平台参数(rk3588)必填,漏了会报 "Invalid model type"

# FP16 参照版(不量化,先转,当精度基准)
python convert.py helmet_y11s_best.onnx rk3588 fp  helmet_y11s_best_fp16.rknn

# INT8 部署版
python convert.py helmet_y11s_best.onnx rk3588 i8  helmet_y11s_best_i8.rknn
```

**关于 i8 / u8 / fp**:`fp` = FP16 不量化;`i8` = 有符号 8bit 量化(新架构 NPU,RK3588 用这个);`u8` = 无符号 8bit,只用于 RV1109/RV1126/RK1808 等老平台,`u8` 对 RK3588 无意义。

**两个精度怎么用**:INT8 是部署的主力(快一倍以上、体积减半,正常掉 0.5~2 mAP);FP16 几乎无损,留作精度对照和退路。**两个都转**。

### 5.2 转换脚本内部(共 4 个 API 调用)

```python
rknn = RKNN()
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
rknn.load_onnx(model='helmet_y11s_best.onnx')
rknn.build(do_quantization=True, dataset='helmet_calib.txt')   # fp 时 dataset 不生效
rknn.export_rknn('helmet_y11s_best_i8.rknn')
```

mean=0、std=255 表示**喂原始 0~255 像素,归一化在图内做**,后续推理不要自己再除 255。

### 5.3 转换过程中的正常现象与告警

- `OpFusing`/`Quantizating` 进度条:正常流程,INT8 版耗时明显比 fp 版长(要跑校准图);
- `W build: found outlier value ...`:个别权重存在离群大值,量化分辨率会受影响的提示。**先不管**,转完做精度对比,若掉点明显再用 toolkit2 混合量化单独处理这几层;
- **`Killed`**:不是报错,是内存被系统 OOM 杀掉(见第 9 节 FAQ)。

### 5.4 自定义权重需要同步修改的文件清单(修改前 → 修改后)

用自己的训练权重走完整条链路,除了命令行参数,以下文件**必须改动**。逐项对照,改完再继续:

| 文件 | 改哪里 | 不改的后果 |
|---|---|---|
| fork 的 `ultralytics/cfg/default.yaml` | `model:` 字段指向你的 .pt(见 2.2 节) | 导出的还是官方 yolov8n 演示模型 |
| model_zoo 的 `yolo11/python/convert.py` 第 4 行 | `DATASET_PATH` 指向你的校准清单(见 4.2 节) | INT8 拿 COCO 图校准,精度明显掉 |
| model_zoo 的 `yolo11/python/yolo11.py` 顶部 | `CLASSES` 改成你的类别列表 | 类别名显示错误(框/分数不受影响) |
| model_zoo 的 `yolo11/cpp/` C demo | 类别数定义与 label 文件 | C demo 解码或显示异常 |

**① convert.py 第 4 行**(转 INT8 前必改):

```python
# 修改前:
DATASET_PATH = '../../../datasets/COCO/coco_subset_20.txt'
# 修改后(换成你生成的校准清单,建议绝对路径):
DATASET_PATH = '/root/code/rknn/helmet_calib.txt'
```

**② yolo11.py 的 `CLASSES`**(模拟器测试、上板测试前改):

```python
# 修改前(COCO 80 类,节选):
CLASSES = ("person", "bicycle", "car", "motorbike ", ...)
# 修改后(顺序必须 = 训练时的类别 ID,例如 0=helmet, 1=no_helmet):
CLASSES = ("helmet", "no_helmet")
```

同一个文件里顺带检查:`IMG_SIZE = (640, 640)` 要和训练推理尺寸一致;`OBJ_THRESH = 0.25`、`NMS_THRESH = 0.45` 按业务调整。

**③ fork 的 default.yaml**(导出前改,详见 2.2 节):

```yaml
# 修改前(fork 默认指向演示权重):
model: yolov8n.pt
# 修改后:
model: /root/code/rknn/helmet_y11s_best.pt
```

> 对比脚本(compare_pt_onnx.py / compare_onnx_rknn.py)的模型路径都用**命令行参数**传,不需要改脚本本身。

---

## 6. 第五步:模拟器测试(PC 上,不需要板子)

### 6.1 跑通测试

```bash
mkdir -p /root/code/rknn/test_imgs      # 放几张你自己场景的图(有戴帽/未戴帽的都要)
# cp /你的测试图/*.jpg /root/code/rknn/test_imgs/

# 不带 --target 参数 = PC 模拟器推理
python yolo11.py --model_path helmet_y11s_best_i8.rknn --img_folder /root/code/rknn/test_imgs --img_save
```

- `--img_folder`:测试图片目录(默认 `../model` 里的 bus.jpg 是 COCO 街景,对自定义模型无意义,**必须换成自己的图**);
- `--img_save`:画框结果图存到当前目录 `./result/`;`--img_show` 需要 GUI 环境,服务器上用 `--img_save`;
- 终端打印 `类别 @ (坐标) 分数`。

**类别名显示不对是正常的**:demo 的 `CLASSES` 写死 COCO 80 类。改成自己的类别即可,如 `CLASSES = ("helmet", "no_helmet")`(**顺序必须和训练时类别 ID 一致**)。框和分数不受影响。

### 6.2 量化精度验收(推荐:compare_onnx_rknn.py 直接对比 ONNX ↔ RKNN)

本仓库的 `compare_onnx_rknn.py` 用**同一张图、同一套前后处理**分别喂 ONNX 和 RKNN,逐路输出计算指标,再对检测框配对,自动给出 PASS / WARN / FAIL 结论。fp 版和 i8 版都能用它验:

```bash
cd ~/code/rknn/rknn_model_zoo/examples/yolo11/python
cp /path/to/pt2rknn/compare_onnx_rknn.py .

# 对比 FP16 版(关卡 B,期望 PASS)
python compare_onnx_rknn.py \
    --onnx /root/code/rknn/helmet_y11s_best.onnx \
    --rknn helmet_y11s_best_fp16.rknn \
    --img_folder /root/code/rknn/test_imgs --img_save

# 对比 INT8 版(关卡 C,出现 WARN 属正常)
python compare_onnx_rknn.py \
    --onnx /root/code/rknn/helmet_y11s_best.onnx \
    --rknn helmet_y11s_best_i8.rknn \
    --img_folder /root/code/rknn/test_imgs --img_save
```

**⚠️ 放置位置**:脚本 import 了同目录的 `yolo11.py`(IMG_SIZE/CLASSES/后处理函数)和 `py_utils` 工具库,且要求运行路径中包含 `rknn_model_zoo` 目录名——**必须拷到 `rknn_model_zoo/examples/yolo11/python/` 里运行**,不能在 pt2rknn 目录下直接跑。它走 toolkit2 模拟器推理,所以同样只要求 PC 端环境,不需要板子。

**参数说明**:

| 参数 | 默认值 | 含义与可设置值 |
|---|---|---|
| `--onnx` | 脚本目录下 `helmet_y11s_best.onnx` | ONNX 模型路径;脚本启动时先做完整性校验(文件截断/损坏会直接报错并提示重新传输) |
| `--rknn` | 脚本目录下 `helmet_y11s_best_fp.rknn` | 待验证的 RKNN 模型路径(fp16 或 i8 都行) |
| `--img_folder` | `/userdata/...`(示例路径,**必须改**) | 测试图片目录;建议 20~50 张自己场景的图(有目标、有背景、有难例) |
| `--img_save` | 关闭 | 加上则输出 ONNX / RKNN 并排对比图(绿字标注)到 `--out_dir` |
| `--out_dir` | `./compare_result/` | 对比图保存目录 |
| `--max_images` | 0(全部) | 只跑前 N 张,快速冒烟用 |

**输出怎么读**:

- **sig_cos(有效区域 cosine)**:只统计 `|onnx 输出| > 0.01` 的有效像素区域——全图大部分是近零背景,普通 cosine 会被浮点底噪拖垮,这个指标更能反映真实一致性(脚本作者已经处理了这个坑);
- **PASS**(sig_cos ≥ 0.99 且检测框全部配对):FP 转换一致性良好;
- **WARN**(sig_cos ≥ 0.95,检测框 ≥ 80% 配对):i8 量化的正常表现;**若 fp16 版也只到 WARN,要深入排查**;
- **FAIL**:依次检查 ONNX 来源(是否 fork 导出)、mean/std 配置、输出顺序;
- RKNN 与 ONNX 的输出顺序可能不一致,脚本会按 shape 自动对齐并打印映射(`Output index mapping`),无需人工干预;
- 逐图打印 ONNX-only / RKNN-only 的漏检与多检序号,定位问题图很方便;
- 阈值口径与第 3.1 节的分阶段阈值表一致:fp16 参考 PASS 线,i8 参考 WARN 线,最终以 mAP 兜底。

### 6.3 备选:yolo11.py 输出对比法

没有专门脚本时,用官方 demo 的输出做对比兜底:

```bash
python yolo11.py --model_path helmet_y11s_best_fp16.rknn --img_folder /root/code/rknn/test_imgs > fp16_result.txt
python yolo11.py --model_path helmet_y11s_best_i8.rknn  --img_folder /root/code/rknn/test_imgs > i8_result.txt
diff fp16_result.txt i8_result.txt
```

**合格标准**:框一一对应、位置基本不动、置信度差 0.05 以内、无漏检/多检。建议放 20~50 张不同场景的图对比。不合格先换/补校准图重转 i8,仍不行上混合量化。

### 6.4 mAP 评估(可选,需 COCO 格式标注)

```bash
pip install pycocotools
python yolo11.py --model_path helmet_y11s_best_i8.rknn \
    --img_folder /验证集图片目录 \
    --anno_json /标注.json --coco_map_test
```

标注的类别 ID 顺序必须与训练一致;模拟器跑全量较慢,可抽子集。

---

## 7. 第六步:部署到 RK3588 板子

### 7.1 板端环境检查(三样东西)

```bash
# ① 内核 rknpu 驱动(官方固件自带,建议 ≥ 0.9.2;查不到说明是第三方固件)
dmesg | grep -i rknpu

# ② 运行库与 rknn_server 版本(必须 ≥ PC 端 toolkit2 版本)
strings /usr/bin/rknn_server   | grep -i "rknn_server version"
strings /usr/lib/librknnrt.so  | grep -i "librknnrt version"
```

版本低了就更新(在 **PC** 上执行,文件落在**板子**上):

```bash
cd ~/code/rknn/rknn-toolkit2/rknpu2
adb push runtime/Linux/rknn_server/aarch64/usr/bin/* /usr/bin
adb push runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib
adb shell   # 以下在板子上
chmod +x /usr/bin/rknn_server /usr/bin/start_rknn.sh /usr/bin/restart_rknn.sh
restart_rknn.sh
```

> Linux 板不在 PC 旁边时,`scp` 传文件即可(model_zoo README 官方认可 scp 方式)。
> 连板调试(模拟器替代不了的真机验证)需要板子 USB 插在执行命令的机器上,并启动 rknn_server。

### 7.2 方式 A:板上 Python(rknn-toolkit-lite2)

```bash
# 板上(Python 3.7~3.12)
pip install rknn-toolkit-lite2
```

把 model_zoo 的 `examples/yolo11/python/` 整个目录拷上板,按官方 FAQ 的做法把 `py_utils/rknn_executor.py` 里的
`from rknn.api import RKNN` 改成 `from rknnlite.api import RKNNLite as RKNN` 即可跑:

```bash
python yolo11.py --model_path /userdata/helmet_y11s_best_i8.rknn \
    --img_folder /userdata/test_imgs --img_save
```

RKNNLite 可用 `core_mask` 指定 NPU 核(`RKNNLite.NPU_CORE_0 / NPU_CORE_0_1_2 / NPU_CORE_AUTO`)。

### 7.3 方式 B:C/C++(性能更好,产品推荐)

```bash
# 交叉编译在 PC 上做(需 Linaro GCC 6.3.1)
export GCC_COMPILER=/path/to/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu
cd ~/code/rknn/rknn_model_zoo
./build-linux.sh -t rk3588 -a aarch64 -d yolo11

# 产物推到板子
adb push install/rk3588_linux_aarch64/rknn_yolo11_demo/ /userdata/

# 板上运行
adb shell
cd /userdata/rknn_yolo11_demo
export LD_LIBRARY_PATH=./lib
./rknn_yolo11_demo model/helmet_y11s_best_i8.rknn model/bus.jpg
# 结果图在 out.png,PC 上: adb pull /userdata/rknn_yolo11_demo/out.png .
```

> C demo 的后处理与官方 fork 导出结构配套;换自己的模型时同步修改类别数定义和 label 文件。

---

## 8. 验收流程总结(一图流)

```
✅ 关卡A:.pt vs .onnx 输出几乎一致      → 否则:检查 fork 是否生效、预处理是否一致
✅ 模拟器:fp16 与 onnx 基本一致        → 否则:检查 toolkit 版本
✅ 模拟器:i8 vs fp16 差 0.05 以内       → 否则:换/补校准图重转 → 混合量化
✅ 板上:结果与模拟器一致               → 否则:查三端版本配套
✅ mAP:较 .pt 掉 0.5~2 个点属正常      → INT8 量化的正常代价
```

---

## 9. 常见问题排查(踩坑实录)

| 现象 | 原因与解决 |
|---|---|
| `ERROR: Invalid model type: xxx.rknn` | **漏了平台参数**。正确顺序:`convert.py <onnx> rk3588 i8 <输出>`——第三参数位置被当成了 dtype |
| 转换中途 `Killed`(无报错堆栈) | 内存 OOM 被 OS 杀掉。① 校准集减到 100~200 张;② 容器查 `cat /sys/fs/cgroup/memory.max`,有上限就 `docker update --memory 8g`;③ VM 内存小就在**宿主机**加 swap 或升配 |
| `cat /sys/fs/cgroup/memory.max` 显示 `max` | 容器没设内存限制,OOM 来自宿主机整体内存不足,处理同上 |
| fp 版转换正常,i8 版被杀 | 量化要跑校准图,内存峰值大——先减校准图数量 |
| fork 仓库没有 requirements.txt | 新版用 `pyproject.toml`,在 fork 根目录 `pip install -e .` 即可 |
| 导出的 ONNX 和 .pt 差异明显 | 导出时用了 PyPI 官方 ultralytics 而非 fork 代码。验证:`python -c "import ultralytics; print(ultralytics.__file__)"` 必须指向 fork 目录 |
| demo 打印的类别名乱七八糟 | `CLASSES` 写死 COCO 80 类,改成自己的类别列表(顺序=训练时类别 ID);框和分数不受影响 |
| bus.jpg 测不出任何框 | 官方测试图是 COCO 街景,自定义模型换成自己场景的图 |
| 上板报版本不匹配 | 三端版本配套:toolkit2 = librknnrt.so = rknn_server;驱动随固件升级 |
| 量化掉点大 | 换贴近实际场景的校准图 → 补覆盖不足的类别 → `quantized_algorithm="mmse"` → 混合量化(hybrid_quantization) |
| 板端 python demo 报 rknn_server 异常 | 连板调试才需要 rknn_server;板上手动启动:`adb shell "nohup /usr/bin/rknn_server >/dev/null" &` |
| compare_pt_onnx.py 报 `Unsupported ONNX outputs` | 该脚本支持端到端单输出与拆头 6/9 路 RK 优化输出;遇到其他布局需按实际输出 shape 扩展 `decode_onnx()` |
| compare_pt_onnx.py 找不到权重 | 默认从 `weight/` 子目录读 `helmet_y11s_best.pt/.onnx`;不放该目录就用 `--pt/--onnx` 显式指定 |
| compare_onnx_rknn.py 启动报 `ValueError: 'rknn_model_zoo' is not in list` | 脚本必须放在 `rknn_model_zoo/examples/yolo11/python/` 目录下运行(它 import 同目录的 yolo11.py 与 py_utils,见 6.2 节) |
| compare_onnx_rknn.py 对比图里两边框完全不同 | 先看 `Output index mapping` 是否对齐;再检查 --onnx 是否 fork 导出、--rknn 与 onnx 是否同一权重的产物、mean/std 是否一致(FAIL 时按 6.2 节排查) |
| 不修改 YOLO 结构直接转行吗 | 可以但不推荐(官方 FAQ 3.5):量化精度差、性能差,且 demo 后处理代码对不上 |

---

## 10. 附录:官方资料索引

| 资料 | 位置 |
|---|---|
| RKNN-Toolkit2 仓库 | https://github.com/airockchip/rknn-toolkit2 |
| RKNN Model Zoo | https://github.com/airockchip/rknn_model_zoo |
| YOLO11 导出 fork | https://github.com/airockchip/ultralytics_yolo11 |
| Quick Start 手册(中文,27 页) | `rknn-toolkit2/doc/01_Rockchip_RKNPU_Quick_Start_RKNN_SDK_V2.3.2_CN.pdf` |
| User Guide(工具详细用法、混合量化) | `rknn-toolkit2/doc/02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.2_CN.pdf` |
| API 手册(Python / C) | `doc/03_*_Toolkit2_*.pdf`、`doc/04_*_RKNNRT_*.pdf` |
| 算子支持列表 | `doc/05_RKNN_Compiler_Support_Operator_List_V2.3.2.pdf` |
| WSL 使用指南 | `rknn-toolkit2/doc/WSL中使用RKNN_ToolKit2.md` |
| rknn_server 连板说明 | `rknn-toolkit2/doc/rknn_server_proxy.md` |
| Model Zoo FAQ(官方踩坑合集) | `rknn_model_zoo/FAQ_CN.md` |
| RKNPU2 SDK 网盘(镜像、预转模型) | https://console.zbox.filez.com/l/I00fc3 (提取码 rknn) |

> 文档基于官方 v2.3.2(2025-04-03 发布)整理。本目录配套两个工具脚本:`make_calib.py` 生成量化校准清单(第 4 节),`compare_pt_onnx.py` 做 .pt 与 .onnx 的转换精度对比(第 3 节)。
