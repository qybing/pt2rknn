# syntax=docker/dockerfile:1
# =============================================================
# pt2rknn 转换环境镜像(x86_64)
#
# pip 默认走清华源;torch/torchvision 锁定 CPU 版不被后续安装改动:
#   torch==2.4.0(CPU)  torchvision==0.19.0
# ultralytics 双轨设计:
#   - site-packages 里是 ultralytics==8.3.28(与训练环境一致,默认 import 用它,
#     compare_pt_onnx.py 加载 .pt 走这个版本)
#   - RKNN 导出专用 fork 在 /opt/ultralytics_yolo11(只 clone 不 pip 安装,
#     不会覆盖 8.3.28);导出时用 PYTHONPATH 切到 fork 代码:
#       cd /opt/ultralytics_yolo11
#       PYTHONPATH=/opt/ultralytics_yolo11 python ./ultralytics/engine/exporter.py
#     (先改 /opt/ultralytics_yolo11/ultralytics/cfg/default.yaml 的 model 字段)
#
# 构建:  docker build -t pt2rknn:2.3.2 .
# 运行:  docker run -it --rm -v /你的数据目录:/workspace pt2rknn:2.3.2 bash
#
# 镜像内固定路径:
#   /opt/ultralytics_yolo11                                  RKNN 导出专用 fork(PYTHONPATH 方式调用)
#   /opt/rknn_model_zoo                                      官方 model_zoo(convert.py / yolo11.py / py_utils)
#   /opt/rknn_model_zoo/examples/yolo11/python/compare_onnx_rknn.py
#   /opt/tools/make_calib.py  /opt/tools/compare_pt_onnx.py
#   /workspace                                               挂载你的权重/图片/校准清单
#
# 能力边界(详见 README.md 第 1.4 节):
#   容器内可做:导出 ONNX / 生成校准集 / 转 RKNN / 模拟器验证 / mAP(模拟器) / adb+scp 网络传板上文件
#   容器内不可做:连板调试 --target rk3588(需 USB 直通,仅限 Linux 宿主机加 --privileged)
# =============================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 基础工具 + Python 3.10(Ubuntu 22.04 自带)+ adb/ssh(网络方式传文件到板子)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git wget curl \
        python3 python3-pip \
        adb openssh-client vim \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# numpy 钉 1.26.4(toolkit2 要求 numpy<=1.26.4)
# torch/torchvision 走 pytorch 官方 CPU 源,版本锁定 2.4.0 / 0.19.0,后续安装只依赖不升级
RUN pip3 install --no-cache-dir numpy==1.26.4 \
    && pip3 install --no-cache-dir torch==2.4.0 torchvision==0.19.0 \
       --index-url https://download.pytorch.org/whl/cpu

# ultralytics 8.3.28:与训练环境一致;pip 检测到 torch/torchvision 已满足要求,不会动它们
# onnxsim:ONNX 图精简工具(README 2.3 节)
RUN pip3 install --no-cache-dir ultralytics==8.3.28 onnxsim

# RKNN-Toolkit2 2.3.2(核心转换工具)+ pycocotools(mAP 评估)
RUN pip3 install --no-cache-dir rknn-toolkit2==2.3.2 pycocotools

# RKNN 导出专用 fork:只 clone 不 pip 安装(避免覆盖 ultralytics 8.3.28),导出用 PYTHONPATH 切换
RUN git clone --depth 1 https://github.com/airockchip/ultralytics_yolo11.git /opt/ultralytics_yolo11

# 官方 model_zoo:convert.py / yolo11.py / py_utils
RUN git clone --depth 1 https://github.com/airockchip/rknn_model_zoo.git /opt/rknn_model_zoo

# 本仓库脚本;compare_onnx_rknn.py 必须放进 model_zoo 的 yolo11/python 目录才能跑(见 README 6.2 节)
COPY make_calib.py /opt/tools/make_calib.py
COPY compare_pt_onnx.py /opt/tools/compare_pt_onnx.py
COPY compare_onnx_rknn.py /opt/rknn_model_zoo/examples/yolo11/python/compare_onnx_rknn.py

WORKDIR /workspace
CMD ["/bin/bash"]
