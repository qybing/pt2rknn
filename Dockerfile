# syntax=docker/dockerfile:1
# =============================================================
# pt2rknn 转换环境镜像(x86_64)
#
# 构建:  docker build -t pt2rknn:2.3.2 .
# 运行:  docker run -it --rm -v /你的数据目录:/workspace pt2rknn:2.3.2 bash
#
# 镜像内固定路径:
#   /opt/ultralytics_yolo11                                  官方 YOLO11 fork(改 cfg/default.yaml 后跑 exporter.py)
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
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 基础工具 + Python 3.10(Ubuntu 22.04 自带)+ adb/ssh(网络方式传文件到板子)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git wget curl \
        python3 python3-pip \
        adb openssh-client vim \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# numpy 钉 1.26.4(toolkit2 要求 numpy<=1.26.4)
# torch 用 CPU 版(比默认 CUDA 版小数 GB;转换/模拟器纯 CPU 即可,toolkit2 要求 torch<=2.4.0)
RUN pip3 install --no-cache-dir numpy==1.26.4 \
    && pip3 install --no-cache-dir torch==2.4.0 torchvision==0.19.0 \
       --index-url https://download.pytorch.org/whl/cpu

# RKNN-Toolkit2 2.3.2(核心转换工具,依赖自动补齐)
RUN pip3 install --no-cache-dir rknn-toolkit2==2.3.2 pycocotools

# 官方 YOLO11 fork:.pt -> ONNX 导出;可编辑安装,改 /opt/ultralytics_yolo11/ultralytics/cfg/default.yaml 即可
RUN git clone --depth 1 https://github.com/airockchip/ultralytics_yolo11.git /opt/ultralytics_yolo11 \
    && pip3 install --no-cache-dir -e /opt/ultralytics_yolo11

# 官方 model_zoo:convert.py / yolo11.py / py_utils
RUN git clone --depth 1 https://github.com/airockchip/rknn_model_zoo.git /opt/rknn_model_zoo

# 本仓库脚本;compare_onnx_rknn.py 必须放进 model_zoo 的 yolo11/python 目录才能跑(见 README 6.2 节)
COPY make_calib.py /opt/tools/make_calib.py
COPY compare_pt_onnx.py /opt/tools/compare_pt_onnx.py
COPY compare_onnx_rknn.py /opt/rknn_model_zoo/examples/yolo11/python/compare_onnx_rknn.py

WORKDIR /workspace
CMD ["/bin/bash"]
