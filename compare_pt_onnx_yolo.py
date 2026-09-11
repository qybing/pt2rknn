#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ultralytics .pt 与 ONNX 检测精度对比（YOLOv8 / YOLO11 / YOLO26）。

同一套 letterbox 预处理后，对比 NMS 检出框：数量、类别、IoU、|Δconf|。
非 e2e 布局时额外对比解码张量 cosine / box_mae / cls_mae。

模型族（可用 --family 覆盖，默认 auto）：
  - YOLO26 : Detect.reg_max=1，ONNX 常见 6 路 raw 或 fused (1,4+nc,N)
  - YOLOv8 / YOLO11 : Detect.reg_max=16（DFL），ONNX 常见 9 路或 fused

PT 对齐（--pt-mode auto）：
  - ONNX 为拆头 → 从 Detect 的 cv2/cv3（或 one2many/one2one）取 raw 再解码
  - ONNX 为 fused → PT 走解码后 concat（end2end=False）
  - ONNX 为 e2e   → PT 走 ultralytics predict()

示例：
  python3 compare_pt_onnx_yolo.py --pt safety_helmet_all.pt --onnx safety_helmet_all.onnx \\
      --source /userdata/jovan/code/rk3588/dataset/helmet/ --img_save
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
STRIDES = (8, 16, 32)
HIGH_CONF = 0.5


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def list_images(source: Path) -> list[Path]:
    """列出源路径下的图片文件。"""
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def maybe_sigmoid_cls(cls_flat: np.ndarray) -> np.ndarray:
    """若分类输出像 logits，则做 sigmoid。"""
    if cls_flat.min() < -1e-3 or cls_flat.max() > 1.0 + 1e-3:
        return sigmoid(cls_flat)
    return cls_flat


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """xywh → xyxy，最后一维为 4。"""
    out = np.empty_like(xywh)
    out[:, 0] = xywh[:, 0] - xywh[:, 2] * 0.5
    out[:, 1] = xywh[:, 1] - xywh[:, 3] * 0.5
    out[:, 2] = xywh[:, 0] + xywh[:, 2] * 0.5
    out[:, 3] = xywh[:, 1] + xywh[:, 3] * 0.5
    return out


def dets_to_original(dets: np.ndarray, co_helper: COCO_test_helper) -> np.ndarray:
    """将 letterbox 画布上的 xyxy 映射回原图。"""
    if len(dets) == 0:
        return dets
    out = dets.copy()
    out[:, :4] = np.asarray(co_helper.get_real_box(dets[:, :4]), dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# 模型族 / Detect 头
# ---------------------------------------------------------------------------

def get_detect_module(core: torch.nn.Module) -> torch.nn.Module:
    """取 DetectionModel 最后一个 Detect 头。"""
    return core.model[-1]


def infer_family(reg_max: int, user_family: str) -> str:
    """根据 Detect.reg_max 判断模型族。"""
    if user_family == "v26":
        return "v26"
    if user_family in ("v8", "v11"):
        return "v8_v11"
    if reg_max in (-1, 1):
        return "v26"
    return "v8_v11"


def family_label(family: str, reg_max: int) -> str:
    """用于打印的模型族文案。"""
    if family == "v26":
        return f"YOLO26 (reg_max={reg_max})"
    return f"YOLOv8/YOLO11 (reg_max={reg_max})"


def fuse_pt_model(yolo: YOLO) -> bool:
    """fuse Conv+BN；先关 end2end，避免删掉 cv2/cv3（one2many）。"""
    try:
        det = get_detect_module(yolo.model)
        if hasattr(det, "end2end"):
            det.end2end = False
        yolo.fuse()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: PT fuse 失败，继续用未 fuse 权重: {exc}")
        return False


def resolve_box_cls_modules(head: torch.nn.Module, which: str):
    """解析 one2one / one2many / cv2+cv3 的 box、cls 分支。"""
    which = which.lower()
    for key in (("one2one", "one2many") if which == "one2one" else ("one2many", "one2one")):
        obj = getattr(head, key, None)
        if isinstance(obj, dict):
            for bk, ck in (("box_head", "cls_head"), ("cv2", "cv3"), ("one2one_cv2", "one2one_cv3")):
                if bk in obj and ck in obj:
                    return obj[bk], obj[ck], key
    if which == "one2one":
        pairs = (("one2one_cv2", "one2one_cv3"), ("cv2", "cv3"))
    else:
        pairs = (("cv2", "cv3"), ("one2one_cv2", "one2one_cv3"))
    for bk, ck in pairs:
        if hasattr(head, bk) and hasattr(head, ck):
            box_ml, cls_ml = getattr(head, bk), getattr(head, ck)
            if box_ml is not None and cls_ml is not None:
                return box_ml, cls_ml, "attr"
    raise AttributeError(f"Detect 头上找不到 {which} 的 box/cls 分支")


def forward_feats(core: torch.nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    """前向到 Detect 输入特征图列表。"""
    y: list = []
    out = x
    detect = core.model[-1]
    for m in core.model:
        if m.f != -1:
            out = y[m.f] if isinstance(m.f, int) else [out if j == -1 else y[j] for j in m.f]
        if m is detect:
            return out if isinstance(out, list) else [out]
        out = m(out)
        y.append(out if m.i in core.save else None)
    raise RuntimeError("未找到 Detect 模块")


# ---------------------------------------------------------------------------
# 解码 / NMS / 布局
# ---------------------------------------------------------------------------

def dfl_or_direct_box(box: np.ndarray, stride: float) -> np.ndarray:
    """单尺度 box 解码为 xywh。"""
    _, c, h, w = box.shape
    if c % 4 != 0:
        raise ValueError(f"box 通道数必须是 4 的倍数，得到 {c}")
    reg_max = c // 4
    if reg_max == 1:
        dfl = box.astype(np.float32)
    else:
        reg = box.reshape(1, 4, reg_max, h, w).astype(np.float32)
        reg = reg - reg.max(axis=2, keepdims=True)
        exp = np.exp(reg)
        dfl = (
            exp / exp.sum(axis=2, keepdims=True) * np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1, 1)
        ).sum(axis=2)

    gy, gx = np.meshgrid(
        np.arange(h, dtype=np.float32) + 0.5,
        np.arange(w, dtype=np.float32) + 0.5,
        indexing="ij",
    )
    x1 = (gx - dfl[0, 0]) * stride
    y1 = (gy - dfl[0, 1]) * stride
    x2 = (gx + dfl[0, 2]) * stride
    y2 = (gy + dfl[0, 3]) * stride
    return np.stack(
        [
            ((x1 + x2) * 0.5).reshape(-1),
            ((y1 + y2) * 0.5).reshape(-1),
            (x2 - x1).reshape(-1),
            (y2 - y1).reshape(-1),
        ],
        axis=0,
    )


def pair_box_cls(a: np.ndarray, b: np.ndarray, nc: int) -> tuple[np.ndarray, np.ndarray]:
    """按通道数区分 box / cls，允许输出顺序颠倒。"""
    if a.ndim != 4 or b.ndim != 4:
        raise ValueError(f"拆头输出应为 4D，得到 {a.shape}, {b.shape}")
    ca, cb = int(a.shape[1]), int(b.shape[1])
    if ca % 4 == 0 and cb == nc:
        return a, b
    if cb % 4 == 0 and ca == nc:
        return b, a
    raise ValueError(f"无法按 nc={nc} 配对 box/cls: {a.shape} vs {b.shape}")


def detect_layout(outputs: list[np.ndarray], nc: int) -> str:
    """识别 ONNX 输出布局（与 RKNN 脚本命名一致）。"""
    n = len(outputs)
    if n == 1:
        o = outputs[0]
        if o.ndim != 3:
            raise ValueError(f"单输出应为 3D，得到 {o.shape}")
        if o.shape[1] == 4 + nc or o.shape[2] == 4 + nc:
            return "fused"
        if o.shape[-1] == 6 or o.shape[1] == 6:
            return "e2e"
        raise ValueError(f"无法识别单输出 shape={o.shape}，nc={nc}")
    if n in (6, 9) and all(getattr(o, "ndim", 0) == 4 for o in outputs):
        step = n // 3
        pair_box_cls(outputs[0], outputs[1], nc)
        return "split9" if step == 3 else "split6"
    raise ValueError(f"不支持的输出: n={n} shapes={[tuple(o.shape) for o in outputs]}")


def decode_split(outputs: list[np.ndarray], nc: int, imgsz: int) -> np.ndarray:
    """解码 6/9 路拆头为 (1, 4+nc, N)。9 路时忽略每尺度第 3 个 score_sum。"""
    step = len(outputs) // 3
    chunks = []
    for i in range(3):
        box, cls = pair_box_cls(outputs[i * step + 0], outputs[i * step + 1], nc)
        h = int(box.shape[2])
        stride = float(imgsz / h) if h > 0 else float(STRIDES[i])
        xywh = dfl_or_direct_box(box, stride)
        cls_flat = maybe_sigmoid_cls(cls[0].reshape(cls.shape[1], -1).astype(np.float32))
        chunks.append(np.concatenate([xywh, cls_flat], axis=0))
    return np.concatenate(chunks, axis=1)[None].astype(np.float32)


def normalize_fused(out: np.ndarray, nc: int) -> np.ndarray:
    """将 (1,4+nc,N) 或 (1,N,4+nc) 规范为 (1,4+nc,N)。"""
    if out.ndim != 3:
        raise ValueError(f"fused 维数异常: {out.shape}")
    if out.shape[1] == 4 + nc:
        return out.astype(np.float32)
    if out.shape[2] == 4 + nc:
        return out.transpose(0, 2, 1).astype(np.float32)
    raise ValueError(f"无法将 {out.shape} 映射到 nc={nc}")


def maybe_scale_fused(pred: np.ndarray, imgsz: int, mode: str) -> np.ndarray:
    """fused 框有时是 0~1 归一化，必要时 ×imgsz。"""
    box = pred[:, :4, :]
    bmax = float(np.max(np.abs(box)))
    need = mode == "on" or (mode == "auto" and bmax < 2.5)
    if not need:
        return pred
    out = pred.copy()
    out[:, :4, :] *= float(imgsz)
    return out


def parse_e2e(out: np.ndarray) -> np.ndarray:
    """解析 e2e 输出为 (N,6) xyxy/conf/cls。"""
    if out.ndim != 3:
        raise ValueError(f"e2e 维数异常: {out.shape}")
    if out.shape[-1] == 6:
        dets = out[0]
    elif out.shape[1] == 6:
        dets = out[0].T
    else:
        raise ValueError(f"无法解析 e2e shape={out.shape}")
    keep = dets[:, 4] > 1e-6
    return dets[keep].astype(np.float32)


def nms_numpy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """纯 NumPy NMS。"""
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
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ovr <= iou_thres]
    return keep


def nms_from_raw(pred: np.ndarray, conf: float, iou: float, max_det: int = 300) -> np.ndarray:
    """对 (1,4+nc,N) 分类别 NMS，返回 (K,6) xyxy/conf/cls。"""
    if pred.ndim != 3 or pred.shape[0] != 1:
        raise ValueError(f"期望 pred 形状 (1,C,N)，得到 {pred.shape}")
    p = pred[0]
    boxes = xywh_to_xyxy(p[:4].T.copy())
    cls_scores = p[4:]
    cls_ids = np.argmax(cls_scores, axis=0)
    scores = cls_scores[cls_ids, np.arange(cls_scores.shape[1])]
    mask = scores >= conf
    boxes, scores, cls_ids = boxes[mask], scores[mask], cls_ids[mask]
    if len(scores) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    final: list[np.ndarray] = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        keep = nms_numpy(boxes[idx], scores[idx], iou)
        for k in keep:
            i = int(idx[k])
            final.append(
                np.array(
                    [boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], scores[i], float(c)],
                    dtype=np.float32,
                )
            )
    if not final:
        return np.zeros((0, 6), dtype=np.float32)
    dets = np.stack(final, axis=0)
    return dets[np.argsort(dets[:, 4])[::-1][:max_det]]


def nms_xyxy(dets: np.ndarray, conf: float, iou: float) -> np.ndarray:
    """对已是 xyxy/conf/cls 的 (N,6) 再做一次分类别 NMS。"""
    if len(dets) == 0:
        return dets
    dets = dets[dets[:, 4] >= conf]
    if len(dets) == 0:
        return dets
    final = []
    for c in np.unique(dets[:, 5].astype(np.int32)):
        idx = np.where(dets[:, 5].astype(np.int32) == c)[0]
        keep = nms_numpy(dets[idx, :4], dets[idx, 4], iou)
        final.append(dets[idx][keep])
    return np.concatenate(final, axis=0) if final else np.zeros((0, 6), dtype=np.float32)


def pred_for_nms(pred: np.ndarray) -> np.ndarray:
    """fused 张量在 NMS 前对 cls 做 maybe_sigmoid。"""
    cls = maybe_sigmoid_cls(pred[0, 4:])
    return np.concatenate([pred[:, :4], cls[None]], axis=1)


def onnx_to_pred(
    outs: list[np.ndarray],
    nc: int,
    imgsz: int,
    layout: str,
    box_scale: str,
) -> np.ndarray:
    """解码 ONNX 为 (1,4+nc,N)，不做 NMS。"""
    outs = [np.asarray(o, dtype=np.float32) for o in outs]
    if layout in ("split6", "split9"):
        return decode_split(outs, nc, imgsz)
    if layout == "fused":
        return maybe_scale_fused(normalize_fused(outs[0], nc), imgsz, box_scale)
    raise ValueError(f"layout={layout} 无解码张量")


def onnx_to_dets(
    outs: list[np.ndarray],
    nc: int,
    imgsz: int,
    layout: str,
    conf: float,
    iou: float,
    box_scale: str,
) -> np.ndarray:
    """把 ONNX 输出解码为 letterbox 坐标 (K,6)。"""
    outs = [np.asarray(o, dtype=np.float32) for o in outs]
    if layout in ("split6", "split9"):
        pred = decode_split(outs, nc, imgsz)
        return nms_from_raw(pred, conf, iou)
    if layout == "fused":
        pred = maybe_scale_fused(normalize_fused(outs[0], nc), imgsz, box_scale)
        return nms_from_raw(pred_for_nms(pred), conf, iou)
    if layout == "e2e":
        return nms_xyxy(parse_e2e(outs[0]), conf, iou)
    raise ValueError(f"未知 layout={layout}")


# ---------------------------------------------------------------------------
# PT 推理
# ---------------------------------------------------------------------------

def pt_raw_split(yolo: YOLO, tensor: np.ndarray, head_name: str) -> list[np.ndarray]:
    """从 .pt 提取多尺度 box/cls raw（与 ONNX 拆头同形态）。"""
    core = yolo.model
    box_ml, cls_ml, _ = resolve_box_cls_modules(get_detect_module(core), head_name)
    with torch.no_grad():
        feats = forward_feats(core, torch.from_numpy(tensor))
        outs: list[np.ndarray] = []
        for i, feat in enumerate(feats):
            outs.append(box_ml[i](feat).detach().cpu().numpy())
            outs.append(cls_ml[i](feat).detach().cpu().numpy())
    return outs


def pt_decoded_concat(yolo: YOLO, tensor: np.ndarray, nc: int) -> np.ndarray:
    """Ultralytics 解码后的 (1,4+nc,N)，临时关闭 end2end。"""
    head = get_detect_module(yolo.model)
    old = getattr(head, "end2end", None)
    try:
        if old is not None:
            head.end2end = False
        with torch.no_grad():
            out = yolo.model(torch.from_numpy(tensor))
        pred = out[0] if isinstance(out, (tuple, list)) else out
        arr = pred.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 6:
            raise ValueError("当前是 e2e 框输出，请用 --pt-mode e2e")
        if arr.shape[1] != 4 + nc and arr.shape[2] == 4 + nc:
            arr = arr.transpose(0, 2, 1)
        return normalize_fused(arr, nc)
    finally:
        if old is not None:
            head.end2end = old


def pt_predict_dets(yolo: YOLO, bgr: np.ndarray, imgsz: int, conf: float, iou: float) -> np.ndarray:
    """ultralytics predict() 得到原图像素坐标 (N,6)。"""
    res = yolo.predict(source=bgr, imgsz=imgsz, conf=conf, iou=iou, verbose=False, rect=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = res.boxes.xyxy.cpu().numpy()
    sc = res.boxes.conf.cpu().numpy()[:, None]
    cls = res.boxes.cls.cpu().numpy()[:, None]
    return np.concatenate([xyxy, sc, cls], axis=1).astype(np.float32)


def pt_to_pred(
    yolo: YOLO,
    tensor: np.ndarray,
    layout: str,
    nc: int,
    imgsz: int,
    pt_mode: str,
    head_name: str,
) -> tuple[np.ndarray, str]:
    """按 ONNX 布局取 PT 解码张量 (1,4+nc,N)。"""
    if pt_mode == "e2e" or layout == "e2e":
        raise ValueError("e2e 布局不做张量对比")

    if pt_mode in ("auto", "split") and layout in ("split6", "split9"):
        candidates = [head_name] if pt_mode == "split" else [head_name, "one2many", "one2one"]
        last_err = None
        tried: list[str] = []
        for h in candidates:
            if h in tried:
                continue
            tried.append(h)
            try:
                raw = pt_raw_split(yolo, tensor, h)
                pred = decode_split(raw, nc, imgsz)
                return pred, f"split:{h}"
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        raise RuntimeError(f"PT 拆头失败: {last_err}")

    if pt_mode == "raw" or layout == "fused":
        pred = pt_decoded_concat(yolo, tensor, nc)
        return pred, "raw_concat:end2end=False"

    raise ValueError(f"无法为 layout={layout} pt_mode={pt_mode} 取 PT 张量")


def choose_pt_dets(
    yolo: YOLO,
    tensor: np.ndarray,
    bgr: np.ndarray,
    layout: str,
    nc: int,
    imgsz: int,
    pt_mode: str,
    head_name: str,
    conf: float,
    iou: float,
) -> tuple[np.ndarray, str]:
    """按 ONNX 布局选择对齐的 PT 检测框（letterbox 坐标，e2e 为原图坐标）。"""
    if pt_mode == "e2e" or layout == "e2e":
        return pt_predict_dets(yolo, bgr, imgsz, conf, iou), "e2e/predict"

    if pt_mode in ("auto", "split") and layout in ("split6", "split9"):
        pred, tag = pt_to_pred(yolo, tensor, layout, nc, imgsz, pt_mode, head_name)
        return nms_from_raw(pred, conf, iou), tag

    if pt_mode == "raw" or layout == "fused":
        pred = pt_decoded_concat(yolo, tensor, nc)
        return nms_from_raw(pred_for_nms(pred), conf, iou), "raw_concat:end2end=False"

    pred, tag = pt_to_pred(yolo, tensor, layout, nc, imgsz, pt_mode, head_name)
    return nms_from_raw(pred, conf, iou), tag


# ---------------------------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------------------------

def load_onnx(path: Path) -> tuple[ort.InferenceSession, str, bool]:
    """加载 ONNX，返回 session、输入名、是否 NCHW。"""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = inp.shape
    nchw = True
    if len(shape) == 4 and shape[-1] in (1, 3) and shape[1] not in (1, 3):
        nchw = False
    return sess, inp.name, nchw


def probe_onnx(sess: ort.InferenceSession, input_name: str, nchw: bool, nc: int, imgsz: int) -> tuple[str, list[tuple]]:
    """探测 ONNX 输出布局与 shape。"""
    if nchw:
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    else:
        dummy = np.zeros((1, imgsz, imgsz, 3), dtype=np.float32)
    outs = [np.asarray(o) for o in sess.run(None, {input_name: dummy})]
    layout = detect_layout(outs, nc)
    return layout, [tuple(o.shape) for o in outs]


def run_onnx(sess: ort.InferenceSession, input_name: str, nchw: bool, tensor_nchw: np.ndarray) -> list[np.ndarray]:
    """运行 ONNX；tensor_nchw 为 RGB float NCHW。"""
    if nchw:
        feed = tensor_nchw
    else:
        rgb = tensor_nchw[0].transpose(1, 2, 0)
        feed = np.ascontiguousarray(rgb[None])
    return [np.asarray(o) for o in sess.run(None, {input_name: feed})]


# ---------------------------------------------------------------------------
# 张量对比
# ---------------------------------------------------------------------------

def tensor_report(pt_pred: np.ndarray, onnx_pred: np.ndarray) -> dict:
    """计算解码张量的 cosine / MAE。"""
    if pt_pred.shape != onnx_pred.shape:
        raise ValueError(f"形状不一致: pt={pt_pred.shape} onnx={onnx_pred.shape}")
    box_diff = np.abs(pt_pred[:, :4] - onnx_pred[:, :4])
    cls_diff = np.abs(pt_pred[:, 4:] - onnx_pred[:, 4:])
    a = pt_pred.reshape(-1).astype(np.float64)
    b = onnx_pred.reshape(-1).astype(np.float64)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {
        "cosine": cosine,
        "box_mae": float(box_diff.mean()),
        "cls_mae": float(cls_diff.mean()),
    }


# ---------------------------------------------------------------------------
# 匹配 / 判定 / 可视化
# ---------------------------------------------------------------------------

def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """成对 IoU，输入 xyxy。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_dets(pt: np.ndarray, other: np.ndarray, iou_thr: float) -> dict:
    """按 IoU 且同类贪心匹配两组检测框。"""
    ious = box_iou(pt[:, :4] if len(pt) else pt, other[:, :4] if len(other) else other)
    used: set[int] = set()
    pairs = []
    for i in range(len(pt)):
        best_j, best_iou = -1, 0.0
        for j in range(len(other)):
            if j in used:
                continue
            if ious[i, j] > best_iou:
                best_iou = float(ious[i, j])
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr and int(pt[i, 5]) == int(other[best_j, 5]):
            used.add(best_j)
            pairs.append(
                {
                    "pt_cls": int(pt[i, 5]),
                    "pt_conf": float(pt[i, 4]),
                    "onnx_conf": float(other[best_j, 4]),
                    "iou": best_iou,
                    "dconf": abs(float(pt[i, 4]) - float(other[best_j, 4])),
                }
            )
    n_pt, n_onnx, n_m = int(len(pt)), int(len(other)), len(pairs)
    n_high_pt = int(np.sum(pt[:, 4] >= HIGH_CONF)) if len(pt) else 0
    min_iou = float(min((p["iou"] for p in pairs), default=1.0)) if pairs else 1.0
    mean_iou = float(np.mean([p["iou"] for p in pairs])) if pairs else 1.0
    max_dc = float(max((p["dconf"] for p in pairs), default=0.0)) if pairs else 0.0
    mean_dc = float(np.mean([p["dconf"] for p in pairs])) if pairs else 0.0
    if n_pt == n_onnx == n_m:
        level = "pass" if (n_m == 0 or (min_iou >= 0.9 and max_dc <= 0.15)) else "warn"
    elif n_m >= max(1, int(0.8 * max(n_pt, n_onnx, 1))):
        level = "warn"
    else:
        level = "fail"
    return {
        "n_pt": n_pt,
        "n_onnx": n_onnx,
        "n_match": n_m,
        "n_high_pt": n_high_pt,
        "min_iou": min_iou,
        "mean_iou": mean_iou,
        "max_dconf": max_dc,
        "mean_dconf": mean_dc,
        "level": level,
        "pairs": pairs,
    }


def aggregate_metrics(rows: list[dict]) -> dict:
    """按框汇总整集匹配指标。"""
    n_pt = sum(r["n_pt"] for r in rows)
    n_onnx = sum(r["n_onnx"] for r in rows)
    n_match = sum(r["n_match"] for r in rows)
    ious: list[float] = []
    dconfs: list[float] = []
    high_ious: list[float] = []
    high_dconfs: list[float] = []
    n_high_pt = 0
    n_high_match = 0
    for r in rows:
        n_high_pt += int(r.get("n_high_pt", 0))
        for p in r.get("pairs", []):
            ious.append(float(p["iou"]))
            dconfs.append(float(p["dconf"]))
            if float(p["pt_conf"]) >= HIGH_CONF:
                high_ious.append(float(p["iou"]))
                high_dconfs.append(float(p["dconf"]))
                n_high_match += 1

    denom = max(n_pt, n_onnx, 1)
    return {
        "n_images": len(rows),
        "n_pt": n_pt,
        "n_onnx": n_onnx,
        "n_match": n_match,
        "match_rate": n_match / denom,
        "recall_vs_pt": (n_match / n_pt) if n_pt else 1.0,
        "extra_rate": ((n_onnx - n_match) / n_onnx) if n_onnx else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 1.0,
        "p5_iou": float(np.percentile(ious, 5)) if ious else 1.0,
        "mean_dconf": float(np.mean(dconfs)) if dconfs else 0.0,
        "max_dconf": float(np.max(dconfs)) if dconfs else 0.0,
        "n_high_pt": n_high_pt,
        "n_high_match": n_high_match,
        "high_recall": (n_high_match / n_high_pt) if n_high_pt else 1.0,
        "high_mean_iou": float(np.mean(high_ious)) if high_ious else 1.0,
        "high_mean_dconf": float(np.mean(high_dconfs)) if high_dconfs else 0.0,
    }


def aggregate_tensor(tensor_rows: list[dict | None]) -> dict | None:
    """汇总张量 cosine。"""
    cosines = [float(r["cosine"]) for r in tensor_rows if r is not None]
    if not cosines:
        return None
    return {
        "mean_cosine": float(np.mean(cosines)),
        "min_cosine": float(np.min(cosines)),
    }


def conversion_verdict(m: dict) -> str:
    """根据整集数字给出导出结论。"""
    if m["match_rate"] >= 0.95 and m["mean_iou"] >= 0.90 and m["mean_dconf"] <= 0.05:
        return "导出可用：与参考模型检出高度一致。"
    if m["match_rate"] >= 0.85 and m["mean_iou"] >= 0.80:
        return "轻微差异可接受：建议抽看 FAIL/低分框。"
    return "偏差较大：请检查导出头、预处理、后处理。"


def print_dataset_report(
    tag: str,
    layout: str,
    rows: list[dict],
    counter: Counter,
    tensor_agg: dict | None,
) -> None:
    """打印整集数字与按图 PASS/WARN/FAIL 计数。"""
    m = aggregate_metrics(rows)
    print(f"\n=== PT vs {tag} ({layout}) ===")
    print(f"  n_images        {m['n_images']}")
    print(f"  n_pt / n_onnx   {m['n_pt']} / {m['n_onnx']}")
    print(f"  n_match         {m['n_match']}")
    print(f"  match_rate      {m['match_rate']:.4f}")
    print(f"  recall_vs_pt    {m['recall_vs_pt']:.4f}")
    print(f"  extra_rate      {m['extra_rate']:.4f}")
    print(f"  mean_iou        {m['mean_iou']:.4f}")
    print(f"  p5_iou          {m['p5_iou']:.4f}")
    print(f"  mean|dconf|     {m['mean_dconf']:.4f}")
    print(f"  max|dconf|      {m['max_dconf']:.4f}")
    print(
        f"  high_conf(≥{HIGH_CONF:.2f})  "
        f"recall={m['high_recall']:.4f}  mean_iou={m['high_mean_iou']:.4f}  "
        f"mean|dconf|={m['high_mean_dconf']:.4f}  "
        f"(pt_high={m['n_high_pt']}, matched={m['n_high_match']})"
    )
    if tensor_agg is not None:
        print(f"  mean_cosine     {tensor_agg['mean_cosine']:.6f}")
        print(f"  min_cosine      {tensor_agg['min_cosine']:.6f}")
    print(f"  结论            {conversion_verdict(m)}")
    n = max(len(rows), 1)
    print(
        f"  PASS={counter.get('pass', 0)} ({counter.get('pass', 0) / n:.1%})  "
        f"WARN={counter.get('warn', 0)} ({counter.get('warn', 0) / n:.1%})  "
        f"FAIL={counter.get('fail', 0)} ({counter.get('fail', 0) / n:.1%})"
    )
    fails = [x for x in rows if x["level"] == "fail"]
    for r in fails[:12]:
        print(
            f"  FAIL {r['name']}: {r['n_pt']}/{r['n_onnx']}/{r['n_match']} "
            f"min_iou={r['min_iou']:.3f} max_dconf={r['max_dconf']:.3f}"
        )


def draw_dets(
    bgr: np.ndarray,
    dets: np.ndarray,
    names: dict,
    title: str,
    co_helper: COCO_test_helper,
    letterbox_coords: bool,
) -> np.ndarray:
    """在原图画框。letterbox_coords=True 时先映射回原图。"""
    vis = bgr.copy()
    boxes = dets[:, :4].copy() if len(dets) else np.zeros((0, 4), dtype=np.float32)
    if letterbox_coords and len(boxes):
        boxes = np.asarray(co_helper.get_real_box(boxes), dtype=np.float32)
    for i in range(len(dets)):
        x1, y1, x2, y2 = [int(round(float(v))) for v in boxes[i]]
        cid = int(dets[i, 5])
        cname = names.get(cid, str(cid))
        sc = float(dets[i, 4])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            vis,
            f"{cname} {sc:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return vis


def hstack_panels(panels: list[np.ndarray]) -> np.ndarray:
    """等高横向拼接。"""
    h = max(p.shape[0] for p in panels)
    w = panels[0].shape[1]
    canvas = np.zeros((h, w * len(panels), 3), dtype=np.uint8)
    for j, panel in enumerate(panels):
        canvas[: panel.shape[0], j * w : (j + 1) * w] = panel
    return canvas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行。"""
    p = argparse.ArgumentParser(
        description="Ultralytics .pt vs ONNX 精度对比（YOLOv8 / YOLO11 / YOLO26）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pt", default=str(HERE / "safety_helmet_all.pt"), help=".pt 权重")
    p.add_argument("--onnx", default=str(HERE / "safety_helmet_all.onnx"), help=".onnx 模型")
    p.add_argument("--source", default="/userdata/jovan/code/rk3588/dataset/helmet/", help="图片或目录")
    p.add_argument("--out_dir", default=str(HERE / "compare_result_pt_vs_onnx"), help="可视化目录")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    p.add_argument("--match-iou", type=float, default=0.5, help="框匹配 IoU")
    p.add_argument(
        "--family",
        choices=("auto", "v8", "v11", "v26"),
        default="auto",
        help="模型族；auto 按 Detect.reg_max 判断（1=YOLO26，16=v8/v11）",
    )
    p.add_argument(
        "--pt-mode",
        choices=("auto", "split", "raw", "e2e"),
        default="auto",
        help="PT 对齐：auto 跟 ONNX 布局；split 强制拆头；raw 强制 concat；e2e 用 predict",
    )
    p.add_argument(
        "--head",
        choices=("one2many", "one2one"),
        default="one2many",
        help="拆头优先分支",
    )
    p.add_argument("--fuse", action=argparse.BooleanOptionalAction, default=True, help="对比前 fuse")
    p.add_argument(
        "--box-scale",
        choices=("auto", "on", "off"),
        default="auto",
        help="fused 框是否 ×imgsz",
    )
    p.add_argument("--max_images", type=int, default=0, help="0=全部")
    p.add_argument("--img_save", action="store_true", help="保存对比图")
    p.add_argument("--save_limit", type=int, default=5, help="最多保存前 N 张")
    return p.parse_args()


def main() -> int:
    """逐图对比 .pt 与 .onnx。"""
    args = parse_args()
    pt_path = Path(args.pt)
    onnx_path = Path(args.onnx)
    if not pt_path.is_file():
        raise FileNotFoundError(f".pt 不存在: {pt_path}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f".onnx 不存在: {onnx_path}")

    images = list_images(Path(args.source))
    if args.max_images > 0:
        images = images[: args.max_images]
    out_dir = Path(args.out_dir)
    if args.img_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PT...")
    yolo = YOLO(str(pt_path))
    if getattr(yolo, "task", "detect") != "detect":
        raise ValueError(f"仅支持 detect，当前 task={yolo.task}")
    yolo.model.eval()
    names = yolo.names if isinstance(yolo.names, dict) else {i: n for i, n in enumerate(yolo.names)}
    nc = len(names)

    fused = fuse_pt_model(yolo) if args.fuse else False
    head = get_detect_module(yolo.model)
    reg_max = int(getattr(head, "reg_max", -1))
    end2end = bool(getattr(head, "end2end", False))
    family = infer_family(reg_max, args.family)

    print("Loading ONNX...")
    sess, input_name, nchw = load_onnx(onnx_path)
    layout, shapes = probe_onnx(sess, input_name, nchw, nc, args.imgsz)
    onnx_tag = onnx_path.stem

    print("=== PT vs ONNX ===")
    print(f"pt       : {pt_path}")
    print(f"onnx     : {onnx_path}")
    print(f"family   : {family_label(family, reg_max)}")
    print(f"task/nc  : detect / {nc}  names={names}")
    print(f"pt_head  : reg_max={reg_max} end2end={end2end} fused={fused}")
    print(f"onnx_in  : {input_name} nchw={nchw} layout={layout} shapes={shapes}")
    print(f"align    : pt-mode={args.pt_mode} head={args.head} box-scale={args.box_scale}")
    print(f"nms      : conf={args.conf} iou={args.iou} match-iou={args.match_iou}")
    print(f"source   : {args.source} ({len(images)} images)")

    co = COCO_test_helper(enable_letter_box=True)
    rows: list[dict] = []
    tensor_rows: list[dict | None] = []
    counter: Counter = Counter()
    t0 = time.time()

    compare_tensor = layout != "e2e" and args.pt_mode != "e2e"

    for i, img_path in enumerate(images):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"读图失败: {img_path}")

        img_lb = co.letter_box(im=bgr.copy(), new_shape=(args.imgsz, args.imgsz), pad_color=(0, 0, 0))
        rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])

        onnx_outs = run_onnx(sess, input_name, nchw, tensor)
        pt_dets_lb, pt_tag = choose_pt_dets(
            yolo,
            tensor,
            bgr,
            layout,
            nc,
            args.imgsz,
            args.pt_mode,
            args.head,
            args.conf,
            args.iou,
        )
        onnx_dets_lb = onnx_to_dets(onnx_outs, nc, args.imgsz, layout, args.conf, args.iou, args.box_scale)

        pt_e2e = layout == "e2e" or args.pt_mode == "e2e"
        if pt_e2e:
            pt_match = pt_dets_lb
            onnx_match = dets_to_original(onnx_dets_lb, co)
        else:
            pt_match = dets_to_original(pt_dets_lb, co)
            onnx_match = dets_to_original(onnx_dets_lb, co)

        tstat: dict | None = None
        if compare_tensor:
            try:
                pt_pred, _ = pt_to_pred(yolo, tensor, layout, nc, args.imgsz, args.pt_mode, args.head)
                onnx_pred = onnx_to_pred(onnx_outs, nc, args.imgsz, layout, args.box_scale)
                tstat = tensor_report(pt_pred, onnx_pred)
            except Exception as exc:  # noqa: BLE001
                tstat = None
                if i == 0:
                    print(f"WARN: 张量对比跳过: {exc}")
        tensor_rows.append(tstat)

        st = match_dets(pt_match, onnx_match, args.match_iou)
        st["name"] = img_path.name
        st["pt_tag"] = pt_tag
        if tstat is not None:
            st["tensor"] = tstat
        rows.append(st)
        counter[st["level"]] += 1

        line = (
            f"[{i + 1}/{len(images)}] {img_path.name} | {st['level']} "
            f"{st['n_pt']}/{st['n_onnx']}/{st['n_match']} "
            f"iou={st['mean_iou']:.3f} dconf={st['max_dconf']:.3f}"
        )
        if tstat is not None:
            line += f" cos={tstat['cosine']:.5f} box_mae={tstat['box_mae']:.2e} cls_mae={tstat['cls_mae']:.2e}"
        fail_now = st["level"] == "fail"
        if i < 3 or fail_now or (i + 1) % 32 == 0:
            print(line)

        if args.img_save and i < args.save_limit:
            pt_lb = layout != "e2e" and args.pt_mode != "e2e"
            panels = [
                draw_dets(bgr, pt_dets_lb, names, f"PT {pt_tag}", co, letterbox_coords=pt_lb),
                draw_dets(bgr, onnx_dets_lb, names, onnx_tag[:22], co, letterbox_coords=True),
            ]
            cv2.imwrite(str(out_dir / f"cmp_{img_path.stem}.jpg"), hstack_panels(panels))

    print(f"\ndone in {time.time() - t0:.1f}s  images={len(images)}")
    print_dataset_report(onnx_tag, layout, rows, counter, aggregate_tensor(tensor_rows))

    if args.img_save:
        print(f"views: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
