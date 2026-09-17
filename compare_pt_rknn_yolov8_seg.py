#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOv8-Seg：Ultralytics .pt ↔ INT8/FP .rknn 精度对比。

对比两层（实例级，原图像素坐标）：
  A. 检测框：数量、类别、IoU、|Δconf|
  B. 分割掩码：配对实例的 mask IoU（二值）

RKNN 侧后处理对齐本目录 yolov8_seg.py（13 路：每尺度 box/cls/score/mask×3 + proto）。
与 demo 一致：丢弃 score_sum（分数用 ones）、cls 不做 sigmoid；仅 mask 插值尺寸跟 --imgsz。
PT 默认走 ultralytics predict()（官方 .pt / seg 任务）；若是 zoo 风格 torchscript
raw 权重，可用 --pt-mode raw 走与 RKNN 相同的 post_process。
注意：predict 返回的 masks.data 常在 letterbox 画布上，脚本会先去 pad 再映回原图，
避免「框 IoU 高、mask IoU 假性偏低」。

注意：必须先 import torch，再 import rknnlite。

示例：
  python3 compare_pt_rknn_yolov8_seg.py \\
      --pt ../model/yolov8n-seg.pt \\
      --rknn ../model/yolov8n-seg_i8.rknn \\
      --source ../model \\
      --img_save
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
_ZOO = None
for _p in (HERE, *HERE.parents):
    if (_p / "py_utils").is_dir():
        _ZOO = _p
        break
if _ZOO is None:
    _parts = str(HERE).split(os.sep)
    if "rknn_model_zoo" in _parts:
        _ZOO = Path(os.sep.join(_parts[: _parts.index("rknn_model_zoo") + 1]))
if _ZOO is not None and str(_ZOO) not in sys.path:
    sys.path.insert(0, str(_ZOO))

from py_utils.coco_utils import COCO_test_helper  # noqa: E402
from rknnlite.api import RKNNLite  # noqa: E402

# rknnlite 会改写 logging 级别名，恢复标准名称
logging._nameToLevel.update(
    {
        "CRITICAL": 50,
        "FATAL": 50,
        "ERROR": 40,
        "WARN": 30,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
        "NOTSET": 0,
    }
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMG_SIZE = (640, 640)  # (W, H)
MAX_DETECT = 300
HIGH_CONF = 0.5

# COCO 80 默认名（可用 --names 覆盖）
DEFAULT_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def list_images(source: Path) -> list[Path]:
    """列出源路径下的图片。"""
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"不是图片: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"目录中没有图片: {source}")
    return files


def parse_names(names_str: str | None, yolo_names) -> dict[int, str]:
    """解析类别名：优先 --names，其次 ultralytics.names，否则 COCO80。"""
    if names_str:
        parts = [p.strip() for p in names_str.split(",") if p.strip()]
        return {i: n for i, n in enumerate(parts)}
    if yolo_names is not None:
        if isinstance(yolo_names, dict):
            return {int(k): str(v) for k, v in yolo_names.items()}
        return {i: str(n) for i, n in enumerate(yolo_names)}
    return {i: n for i, n in enumerate(DEFAULT_NAMES)}


def resolve_class_ids(specs: list[str] | None, names: dict[int, str]) -> set[int] | None:
    """把类别 id/名称列表解析成 id 集合；None 表示不过滤。"""
    if not specs:
        return None
    name_to_id = {str(v).lower(): int(k) for k, v in names.items()}
    out: set[int] = set()
    for s in specs:
        key = str(s).strip()
        if key.isdigit() or (key.startswith("-") and key[1:].isdigit()):
            out.add(int(key))
            continue
        lid = name_to_id.get(key.lower())
        if lid is None:
            raise ValueError(f"未知类别 '{s}'，可选: {list(names.values())[:8]}...")
        out.add(lid)
    return out


def filter_by_classes(
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray | None,
    class_ids: set[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """按类别过滤实例。masks 形状 (N,H,W)。"""
    if class_ids is None or len(boxes) == 0:
        return boxes, classes, scores, masks
    keep = np.array([int(c) in class_ids for c in classes], dtype=bool)
    boxes = boxes[keep]
    classes = classes[keep]
    scores = scores[keep]
    if masks is not None and len(masks):
        masks = masks[keep]
    return boxes, classes, scores, masks


def sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定 sigmoid。"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """成对框 IoU，xyxy。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """单个二值 mask 的 IoU。"""
    a = a.astype(bool)
    b = b.astype(bool)
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def nms_numpy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """纯 NumPy NMS，避免依赖 torchvision。"""
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return keep


# ---------------------------------------------------------------------------
# RKNN / raw 后处理（对齐 yolov8_seg.py）
# ---------------------------------------------------------------------------

def dfl_decode(position: np.ndarray) -> np.ndarray:
    """DFL 解码，对齐 yolov8_seg.py 的 dfl()（torch softmax）。"""
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y = y.softmax(2)
    acc = torch.arange(mc, dtype=torch.float32).reshape(1, 1, mc, 1, 1)
    y = (y * acc).sum(2)
    return y.numpy()


def box_process(position: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """单尺度 box → xyxy。对齐 yolov8_seg.py；stride 用整除（与 demo 一致）。"""
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    # yolov8_seg: stride = [IMG_H//gh, IMG_W//gw]，IMG_SIZE=(W,H)
    stride = np.array([imgsz[1] // grid_h, imgsz[0] // grid_w]).reshape(1, 2, 1, 1)
    position = dfl_decode(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)


def _crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """按框裁剪 mask，masks (N,H,W)，boxes (N,4) xyxy。"""
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def filter_boxes(
    boxes: np.ndarray,
    box_confidences: np.ndarray,
    box_class_probs: np.ndarray,
    seg_part: np.ndarray,
    obj_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按 obj*cls 阈值过滤。"""
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _pos = np.where(class_max_score * box_confidences >= obj_thresh)
    scores = (class_max_score * box_confidences)[_pos]
    return boxes[_pos], classes[_pos], scores, (seg_part * box_confidences.reshape(-1, 1))[_pos]


def post_process_seg(
    input_data: list[np.ndarray],
    imgsz: tuple[int, int],
    obj_thresh: float,
    nms_thresh: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """YOLOv8-Seg 后处理，逻辑对齐 yolov8_seg.py 的 post_process()。

    布局（13 路）：
      [0,4,8] box；[1,5,9] cls；[2,6,10] score_sum（demo 中丢弃）；
      [3,7,11] mask_coeff；[-1] proto。

    与 demo 一致处：
      - score 用 ones，不用 score_sum
      - cls 原样使用，不做 sigmoid
      - DFL / filter / 类偏移 NMS / proto@coeff / crop
    仅插值尺寸改为 imgsz（demo 写死 640，1280 模型必须跟输入边长）。
    """
    outs = [np.asarray(x) for x in input_data]
    if len(outs) < 4:
        raise ValueError(f"YOLOv8-Seg 期望约 13 路输出，实际 {len(outs)}")

    proto = outs[-1]
    default_branch = 3
    pair_per_branch = len(outs) // default_branch

    boxes_l, scores_l, classes_conf_l, seg_l = [], [], [], []
    for i in range(default_branch):
        base = pair_per_branch * i
        boxes_l.append(box_process(outs[base], imgsz))
        classes_conf_l.append(outs[base + 1])
        # 对齐 demo：丢弃 score_sum，用全 1
        scores_l.append(np.ones_like(outs[base + 1][:, :1, :, :], dtype=np.float32))
        # pair=4 时 mask 在 +3；pair=3 时在 +2
        seg_l.append(outs[base + 3] if pair_per_branch >= 4 else outs[base + 2])

    def sp_flatten(_in: np.ndarray) -> np.ndarray:
        ch = _in.shape[1]
        return _in.transpose(0, 2, 3, 1).reshape(-1, ch)

    boxes = np.concatenate([sp_flatten(v) for v in boxes_l])
    classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf_l])
    scores = np.concatenate([sp_flatten(v) for v in scores_l])
    seg_part = np.concatenate([sp_flatten(v) for v in seg_l])

    boxes, classes, scores, seg_part = filter_boxes(
        boxes, scores, classes_conf, seg_part, obj_thresh
    )
    if boxes.shape[0] == 0:
        return None, None, None, None

    # 按分数排序（对齐 demo）
    zipped = zip(boxes, classes, scores, seg_part)
    sort_zipped = sorted(zipped, key=lambda x: x[2], reverse=True)
    result = zip(*sort_zipped)
    max_nms = 30000
    n = boxes.shape[0]
    if n > max_nms:
        boxes, classes, scores, seg_part = [np.array(x[:max_nms]) for x in result]
    else:
        boxes, classes, scores, seg_part = [np.array(x) for x in result]

    # 类偏移 + NMS（优先 torchvision，与 demo 一致；失败则 NumPy）
    max_wh = 7680
    c = classes * max_wh
    boxes_t = torch.tensor(boxes, dtype=torch.float32) + torch.tensor(c, dtype=torch.float32).unsqueeze(-1)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    try:
        import torchvision

        ids = torchvision.ops.nms(boxes_t, scores_t, nms_thresh)
        keep = ids.tolist()[:MAX_DETECT]
    except Exception:  # noqa: BLE001
        keep = nms_numpy(boxes + c[:, None], scores, nms_thresh)[:MAX_DETECT]

    boxes = boxes[keep]
    classes = classes[keep]
    scores = scores[keep]
    seg_part = seg_part[keep]
    if len(boxes) == 0:
        return None, None, None, None

    ph, pw = proto.shape[-2:]
    proto_f = np.asarray(proto).reshape(seg_part.shape[-1], -1)
    seg_img = np.matmul(seg_part, proto_f)
    seg_img = sigmoid(seg_img)
    seg_img = seg_img.reshape(-1, ph, pw)

    # 插值到 letterbox 画布；尺寸跟 imgsz（W,H）→ Size([H,W])
    seg_t = F.interpolate(
        torch.tensor(seg_img)[None],
        torch.Size([imgsz[1], imgsz[0]]),
        mode="bilinear",
        align_corners=False,
    )[0]
    seg_t = _crop_mask(seg_t, torch.tensor(boxes))
    seg_bin = seg_t.numpy() > 0.5
    return boxes, classes, scores, seg_bin


# ---------------------------------------------------------------------------
# 推理：PT / RKNN
# ---------------------------------------------------------------------------

def load_rknn(path: Path) -> RKNNLite:
    """加载板端 RKNN。"""
    r = RKNNLite()
    if r.load_rknn(str(path)) != 0:
        raise RuntimeError(f"load_rknn 失败: {path}")
    if r.init_runtime() != 0:
        raise RuntimeError("init_runtime 失败")
    return r


def infer_rknn_seg(
    rknn: RKNNLite,
    img_src: np.ndarray,
    co_helper: COCO_test_helper,
    imgsz: tuple[int, int],
    conf: float,
    iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, list[np.ndarray]]:
    """RKNN 推理并映射回原图。返回 boxes/classes/scores/masks(原图)/raw_outs。"""
    letter = co_helper.letter_box(im=img_src.copy(), new_shape=(imgsz[1], imgsz[0]), pad_color=(114, 114, 114))
    rgb = cv2.cvtColor(letter, cv2.COLOR_BGR2RGB)
    # RKNN Lite 需要 NHWC 4 维：(1,H,W,C)
    rknn_in = np.expand_dims(rgb, 0)
    outs = rknn.inference(inputs=[rknn_in])
    if outs is None:
        raise RuntimeError("RKNN inference 返回 None")
    outs = [np.asarray(o, dtype=np.float32) for o in outs]
    boxes, classes, scores, seg = post_process_seg(outs, imgsz, conf, iou)
    if boxes is None:
        return (
            np.zeros((0, 4), np.float32),
            np.zeros((0,), np.int64),
            np.zeros((0,), np.float32),
            None,
            outs,
        )
    real_boxes = np.asarray(co_helper.get_real_box(boxes), dtype=np.float32)
    real_segs = np.asarray(co_helper.get_real_seg(seg), dtype=np.uint8)  # (N,H,W)
    return real_boxes, classes.astype(np.int64), scores.astype(np.float32), real_segs.astype(bool), outs


def masks_letterbox_to_orig(
    masks_lb: np.ndarray,
    orig_hw: tuple[int, int],
    imgsz: int,
) -> np.ndarray:
    """把 Ultralytics letterbox 画布上的 mask 映回原图像素。

    ``predict`` 返回的 ``masks.data`` 常见形状为 ``(N, imgsz, imgsz)``（含 pad）。
    若直接 ``resize`` 到原图，会把上下/左右黑边压进画面，导致 mask 纵向/横向错位，
    而框已是原图坐标，从而出现「框 IoU 很高、mask IoU 很低」的假象。

    Args:
        masks_lb: ``(N, H_lb, W_lb)``，通常 H_lb=W_lb=imgsz。
        orig_hw: 原图 ``(H, W)``。
        imgsz: predict 用的边长（正方形 letterbox）。

    Returns:
        原图尺寸二值 mask，形状 ``(N, H, W)``。
    """
    oh, ow = int(orig_hw[0]), int(orig_hw[1])
    if masks_lb.ndim != 3 or len(masks_lb) == 0:
        return np.zeros((0, oh, ow), dtype=bool)

    mh, mw = int(masks_lb.shape[1]), int(masks_lb.shape[2])
    # 已是原图尺寸：直接二值化
    if (mh, mw) == (oh, ow):
        return masks_lb > 0.5

    # 与 Ultralytics / COCO letter_box 一致：等比缩放后居中 pad
    r = min(imgsz / oh, imgsz / ow)
    nh = int(round(oh * r))
    nw = int(round(ow * r))
    dh = (imgsz - nh) / 2.0
    dw = (imgsz - nw) / 2.0
    top = int(round(dh - 0.1))
    left = int(round(dw - 0.1))

    out = []
    for m in masks_lb:
        # 若画布边长不是 imgsz（少见），按实际 mh/mw 估 pad
        h_lb, w_lb = m.shape[:2]
        if (h_lb, w_lb) == (imgsz, imgsz):
            y0, x0 = top, left
            y1, x1 = top + nh, left + nw
        else:
            r2 = min(h_lb / oh, w_lb / ow)
            nh2, nw2 = int(round(oh * r2)), int(round(ow * r2))
            y0 = int(round((h_lb - nh2) / 2.0 - 0.1))
            x0 = int(round((w_lb - nw2) / 2.0 - 0.1))
            y1, x1 = y0 + nh2, x0 + nw2
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(h_lb, y1), min(w_lb, x1)
        cropped = m[y0:y1, x0:x1]
        if cropped.size == 0:
            out.append(np.zeros((oh, ow), dtype=bool))
            continue
        resized = cv2.resize(cropped.astype(np.float32), (ow, oh), interpolation=cv2.INTER_LINEAR)
        out.append(resized > 0.5)
    return np.stack(out, axis=0)


def infer_pt_predict(
    yolo,
    img_src: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Ultralytics predict：原图像素坐标的框与 mask。"""
    res = yolo.predict(source=img_src, imgsz=imgsz, conf=conf, iou=iou, verbose=False, rect=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return (
            np.zeros((0, 4), np.float32),
            np.zeros((0,), np.int64),
            np.zeros((0,), np.float32),
            None,
        )
    boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = res.boxes.conf.cpu().numpy().astype(np.float32)
    classes = res.boxes.cls.cpu().numpy().astype(np.int64)
    masks = None
    if res.masks is not None and len(res.masks) > 0:
        # masks.data 多为 letterbox 画布，须先去 pad 再映回原图
        mdata = res.masks.data.cpu().numpy()
        oh, ow = img_src.shape[:2]
        masks = masks_letterbox_to_orig(mdata, (oh, ow), imgsz)
    return boxes, classes, scores, masks


def infer_pt_raw_torchscript(
    model,
    img_src: np.ndarray,
    co_helper: COCO_test_helper,
    imgsz: tuple[int, int],
    conf: float,
    iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """zoo 风格 torchscript .pt：与 RKNN 相同预处理与 post_process。"""
    letter = co_helper.letter_box(im=img_src.copy(), new_shape=(imgsz[1], imgsz[0]), pad_color=(114, 114, 114))
    rgb = cv2.cvtColor(letter, cv2.COLOR_BGR2RGB)
    inp = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    outs = model.run([inp])
    boxes, classes, scores, seg = post_process_seg(outs, imgsz, conf, iou)
    if boxes is None:
        return (
            np.zeros((0, 4), np.float32),
            np.zeros((0,), np.int64),
            np.zeros((0,), np.float32),
            None,
        )
    real_boxes = np.asarray(co_helper.get_real_box(boxes), dtype=np.float32)
    real_segs = np.asarray(co_helper.get_real_seg(seg), dtype=bool)
    return real_boxes, classes.astype(np.int64), scores.astype(np.float32), real_segs


# ---------------------------------------------------------------------------
# 匹配 / 判定 / 可视化
# ---------------------------------------------------------------------------

def match_instances(
    pt_boxes: np.ndarray,
    pt_cls: np.ndarray,
    pt_scores: np.ndarray,
    pt_masks: np.ndarray | None,
    rk_boxes: np.ndarray,
    rk_cls: np.ndarray,
    rk_scores: np.ndarray,
    rk_masks: np.ndarray | None,
    iou_thr: float,
) -> dict:
    """按框 IoU + 同类贪心匹配，并算 mask IoU。"""
    ious = box_iou_matrix(pt_boxes, rk_boxes)
    used: set[int] = set()
    pairs = []
    for i in range(len(pt_boxes)):
        best_j, best_iou = -1, 0.0
        for j in range(len(rk_boxes)):
            if j in used:
                continue
            if int(pt_cls[i]) != int(rk_cls[j]):
                continue
            if ious[i, j] > best_iou:
                best_iou = float(ious[i, j])
                best_j = j
        if best_j < 0 or best_iou < iou_thr:
            continue
        used.add(best_j)
        miou = float("nan")
        if pt_masks is not None and rk_masks is not None and i < len(pt_masks) and best_j < len(rk_masks):
            miou = mask_iou(pt_masks[i], rk_masks[best_j])
        pairs.append(
            {
                "pt_i": i,
                "rk_i": best_j,
                "cls": int(pt_cls[i]),
                "pt_conf": float(pt_scores[i]),
                "rk_conf": float(rk_scores[best_j]),
                "box_iou": best_iou,
                "mask_iou": miou,
                "dconf": abs(float(pt_scores[i]) - float(rk_scores[best_j])),
            }
        )

    n_pt, n_rk, n_m = len(pt_boxes), len(rk_boxes), len(pairs)
    min_biou = float(min((p["box_iou"] for p in pairs), default=1.0)) if pairs else 1.0
    mean_biou = float(np.mean([p["box_iou"] for p in pairs])) if pairs else 1.0
    mask_vals = [p["mask_iou"] for p in pairs if np.isfinite(p["mask_iou"])]
    min_miou = float(min(mask_vals)) if mask_vals else float("nan")
    mean_miou = float(np.mean(mask_vals)) if mask_vals else float("nan")
    max_dc = float(max((p["dconf"] for p in pairs), default=0.0))
    n_high = int(np.sum(pt_scores >= HIGH_CONF)) if len(pt_scores) else 0

    if n_pt == n_rk == n_m:
        ok_mask = (not mask_vals) or (min_miou >= 0.7)
        level = "pass" if (n_m == 0 or (min_biou >= 0.85 and max_dc <= 0.2 and ok_mask)) else "warn"
    elif n_m >= max(1, int(0.8 * max(n_pt, n_rk, 1))):
        level = "warn"
    else:
        level = "fail"

    return {
        "n_pt": n_pt,
        "n_rknn": n_rk,
        "n_match": n_m,
        "n_high_pt": n_high,
        "min_box_iou": min_biou,
        "mean_box_iou": mean_biou,
        "min_mask_iou": min_miou,
        "mean_mask_iou": mean_miou,
        "max_dconf": max_dc,
        "level": level,
        "pairs": pairs,
    }


class Colors:
    """简易调色板，用于画 mask。"""

    def __init__(self) -> None:
        hexs = (
            "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86",
            "1A9334", "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF",
        )
        self.palette = [tuple(int(h[i : i + 2], 16) for i in (4, 2, 0)) for h in hexs]

    def __call__(self, i: int) -> tuple[int, int, int]:
        return self.palette[int(i) % len(self.palette)]


def draw_seg_panel(
    bgr: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray | None,
    names: dict[int, str],
    title: str,
) -> np.ndarray:
    """画框 + 半透明 mask。"""
    vis = bgr.copy()
    colors = Colors()
    if masks is not None:
        for i, m in enumerate(masks):
            if m is None:
                continue
            color = colors(int(classes[i]) if i < len(classes) else i)
            overlay = vis.copy()
            overlay[m.astype(bool)] = (
                (0.5 * overlay[m.astype(bool)] + 0.5 * np.array(color, dtype=np.float32)).astype(np.uint8)
            )
            vis = overlay
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(round(float(v))) for v in boxes[i]]
        cid = int(classes[i])
        cname = names.get(cid, str(cid))
        sc = float(scores[i])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            vis,
            f"{cname} {sc:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
    cv2.putText(vis, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return vis


def aggregate(rows: list[dict]) -> dict:
    """整集汇总。"""
    n_pt = sum(r["n_pt"] for r in rows)
    n_rk = sum(r["n_rknn"] for r in rows)
    n_m = sum(r["n_match"] for r in rows)
    biou = [p["box_iou"] for r in rows for p in r["pairs"]]
    miou = [p["mask_iou"] for r in rows for p in r["pairs"] if np.isfinite(p["mask_iou"])]
    dconf = [p["dconf"] for r in rows for p in r["pairs"]]
    levels = [r["level"] for r in rows]
    return {
        "n_images": len(rows),
        "n_pt": n_pt,
        "n_rknn": n_rk,
        "n_match": n_m,
        "match_rate": float(n_m / max(n_pt, 1)),
        "mean_box_iou": float(np.mean(biou)) if biou else float("nan"),
        "mean_mask_iou": float(np.mean(miou)) if miou else float("nan"),
        "mean_dconf": float(np.mean(dconf)) if dconf else 0.0,
        "pass": levels.count("pass"),
        "warn": levels.count("warn"),
        "fail": levels.count("fail"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行。"""
    p = argparse.ArgumentParser(
        description="YOLOv8-Seg .pt vs RKNN 精度对比（框 + mask）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pt", type=str, required=True, help="Ultralytics .pt 或 zoo torchscript .pt")
    p.add_argument("--rknn", type=str, required=True, help="INT8/FP .rknn")
    p.add_argument("--source", type=str, default=str(HERE / "../model"), help="图片或目录")
    p.add_argument("--out_dir", type=str, default=str(HERE / "compare_result_pt_vs_rknn_seg"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU（对齐 yolov8_seg.py 默认 0.45）")
    p.add_argument("--match-iou", type=float, default=0.5, help="框配对 IoU")
    p.add_argument(
        "--pt-mode",
        choices=("predict", "raw"),
        default="predict",
        help="predict=ultralytics 官方推理；raw=torchscript 与 RKNN 同后处理",
    )
    p.add_argument("--names", type=str, default=None, help="逗号分隔类别名；默认跟 .pt 或 COCO80")
    p.add_argument("--nc", type=int, default=None, help="类别数；默认从 names 推断")
    p.add_argument("--classes", nargs="*", default=None, metavar="CLS", help="只比这些类（id 或名）")
    p.add_argument("--max_images", type=int, default=0, help="0=全部")
    p.add_argument("--img_save", action="store_true")
    p.add_argument("--save_limit", type=int, default=10)
    return p.parse_args()


def main() -> int:
    """逐图对比 .pt 与 .rknn 分割结果。"""
    args = parse_args()
    pt_path = Path(args.pt)
    rknn_path = Path(args.rknn)
    if not pt_path.is_file():
        raise FileNotFoundError(pt_path)
    if not rknn_path.is_file():
        raise FileNotFoundError(rknn_path)

    images = list_images(Path(args.source))
    if args.max_images > 0:
        images = images[: args.max_images]
    out_dir = Path(args.out_dir)
    if args.img_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    imgsz_wh = (args.imgsz, args.imgsz)
    yolo = None
    ts_model = None
    names: dict[int, str]

    if args.pt_mode == "predict":
        from ultralytics import YOLO

        print("Loading Ultralytics PT...")
        yolo = YOLO(str(pt_path))
        task = getattr(yolo, "task", "")
        if task not in ("segment", "detect"):
            print(f"WARN: task={task}，将按 segment/detect 结果尽力对比")
        names = parse_names(args.names, yolo.names)
        yolo.model.eval()
    else:
        from py_utils.pytorch_executor import Torch_model_container

        print("Loading torchscript PT (raw)...")
        ts_model = Torch_model_container(str(pt_path))
        names = parse_names(args.names, None)

    if args.nc is not None:
        # 截断/扩展仅影响打印；匹配仍用实际 cls id
        pass
    class_ids = resolve_class_ids(list(args.classes or []), names)

    print("Loading RKNN...")
    rknn = load_rknn(rknn_path)

    print("=== YOLOv8-Seg PT vs RKNN ===")
    print(f"pt       : {pt_path}  mode={args.pt_mode}")
    print(f"rknn     : {rknn_path}")
    print(f"nc/names : {len(names)}  sample={list(names.values())[:5]}")
    print(f"classes  : {'ALL' if class_ids is None else sorted(class_ids)}")
    print(f"nms      : conf={args.conf} iou={args.iou} match-iou={args.match_iou}")
    print(f"source   : {args.source} ({len(images)} images)")

    rows: list[dict] = []
    saved = 0
    co_helper = COCO_test_helper(enable_letter_box=True)

    for idx, img_path in enumerate(images):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"WARN: 读图失败 {img_path}")
            continue

        # 每张图独立 letterbox 状态：PT predict 不用它；RKNN / raw 用
        co_helper.letter_box_info_list.clear()

        if args.pt_mode == "predict":
            pt_boxes, pt_cls, pt_scores, pt_masks = infer_pt_predict(
                yolo, bgr, args.imgsz, args.conf, args.iou
            )
        else:
            pt_boxes, pt_cls, pt_scores, pt_masks = infer_pt_raw_torchscript(
                ts_model, bgr, co_helper, imgsz_wh, args.conf, args.iou
            )

        # RKNN 需要自己的 letterbox 记录；predict 模式时还没写入，这里再跑一次 letter 状态
        if args.pt_mode == "predict":
            co_helper.letter_box_info_list.clear()
        rk_boxes, rk_cls, rk_scores, rk_masks, rk_outs = infer_rknn_seg(
            rknn, bgr, co_helper, imgsz_wh, args.conf, args.iou
        )

        pt_boxes, pt_cls, pt_scores, pt_masks = filter_by_classes(
            pt_boxes, pt_cls, pt_scores, pt_masks, class_ids
        )
        rk_boxes, rk_cls, rk_scores, rk_masks = filter_by_classes(
            rk_boxes, rk_cls, rk_scores, rk_masks, class_ids
        )

        stat = match_instances(
            pt_boxes, pt_cls, pt_scores, pt_masks,
            rk_boxes, rk_cls, rk_scores, rk_masks,
            args.match_iou,
        )
        rows.append(stat)

        print(
            f"[{stat['level'].upper()}] {img_path.name}  "
            f"pt={stat['n_pt']} rknn={stat['n_rknn']} match={stat['n_match']}  "
            f"box_iou(mean/min)={stat['mean_box_iou']:.3f}/{stat['min_box_iou']:.3f}  "
            f"mask_iou(mean/min)={stat['mean_mask_iou']:.3f}/{stat['min_mask_iou']:.3f}  "
            f"max|dconf|={stat['max_dconf']:.3f}  rknn_outs={len(rk_outs)}"
        )
        for p in stat["pairs"][:8]:
            cname = names.get(p["cls"], str(p["cls"]))
            mi = p["mask_iou"]
            mi_s = f"{mi:.3f}" if np.isfinite(mi) else "n/a"
            print(
                f"    {cname}: pt={p['pt_conf']:.3f} rk={p['rk_conf']:.3f} "
                f"boxIoU={p['box_iou']:.3f} maskIoU={mi_s} |dconf|={p['dconf']:.3f}"
            )

        if args.img_save and saved < args.save_limit:
            left = draw_seg_panel(bgr, pt_boxes, pt_cls, pt_scores, pt_masks, names, "PT")
            right = draw_seg_panel(bgr, rk_boxes, rk_cls, rk_scores, rk_masks, names, "RKNN")
            h = max(left.shape[0], right.shape[0])
            canvas = np.zeros((h, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
            canvas[: left.shape[0], : left.shape[1]] = left
            canvas[: right.shape[0], left.shape[1] :] = right
            save_path = out_dir / f"cmp_seg_{img_path.stem}.jpg"
            cv2.imwrite(str(save_path), canvas)
            print(f"  saved: {save_path}")
            saved += 1

    agg = aggregate(rows)
    print("\n=== OVERALL ===")
    print(
        f"images={agg['n_images']}  pass/warn/fail={agg['pass']}/{agg['warn']}/{agg['fail']}"
    )
    print(
        f"dets pt/rknn/match={agg['n_pt']}/{agg['n_rknn']}/{agg['n_match']}  "
        f"match_rate={agg['match_rate']:.3f}"
    )
    print(
        f"mean_box_iou={agg['mean_box_iou']:.4f}  mean_mask_iou={agg['mean_mask_iou']:.4f}  "
        f"mean_dconf={agg['mean_dconf']:.4f}"
    )
    if agg["fail"] == 0 and agg["warn"] <= max(1, agg["n_images"] // 5):
        print("Conclusion: RKNN 与 .pt 分割结果大体一致，转换可用。")
        code = 0
    elif agg["fail"] == 0:
        print("Conclusion: 有一定漂移（常见于 i8），请抽查可视化与 mask_iou。")
        code = 0
    else:
        print("Conclusion: 存在明显不一致 — 检查 ONNX 是否为 13 路 seg、校准集、后处理。")
        code = 2

    rknn.release()
    if ts_model is not None:
        ts_model.release()
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
