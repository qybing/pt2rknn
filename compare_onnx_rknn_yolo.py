#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX Runtime 与 RKNN NPU 检测精度对比（YOLOv8 / YOLO11 / YOLO26）。

同一套 letterbox 预处理后，对比原始输出张量与 NMS 后检测框。
不依赖 PyTorch / Ultralytics。

输出布局（按 shape 自动识别，--family 仅作打印提示）：
  - split6 : 6×4D（YOLO26 常见 reg+cls×3）
  - split9 : 9×4D（v8/v11，忽略每尺度 score_sum）
  - fused  : (1, 4+nc, N)，可选 ×imgsz（--box-scale auto/on/off）
  - e2e    : (1, max_det, 6)

示例：
  python3 compare_onnx_rknn.py \\
      --onnx safety_helmet_all.onnx \\
      --rknn safety_helmet_all_i8.rknn \\
      --source /userdata/jovan/code/rk3588/dataset/helmet/ \\
      --nc 2 --names head,helmet --img_save
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

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
STRIDES = (8, 16, 32)
HIGH_CONF = 0.5
SIG_THRESH = 0.01


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


def restore_logging_levels() -> None:
    """恢复被 rknnlite 改写的 logging 级别名。"""
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


def parse_names(names_str: str, nc: int) -> dict[int, str]:
    """解析 --names 为类别 id → 名称。"""
    parts = [p.strip() for p in names_str.split(",") if p.strip()]
    if len(parts) != nc:
        raise ValueError(f"--names 需要 {nc} 个名称，得到 {len(parts)}: {names_str}")
    return {i: n for i, n in enumerate(parts)}


def family_hint_label(family: str, reg_max_hint: int | None) -> str:
    """打印用模型族文案（--family 与通道推断）。"""
    if family == "v26":
        return f"YOLO26 (hint reg_max={reg_max_hint or 1})"
    if family in ("v8", "v11"):
        return f"YOLO{family.upper()} (hint reg_max={reg_max_hint or 16})"
    if reg_max_hint == 1:
        return "auto→YOLO26 (reg_max=1)"
    if reg_max_hint and reg_max_hint > 1:
        return f"auto→YOLOv8/YOLO11 (reg_max={reg_max_hint})"
    return "auto"


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


# ---------------------------------------------------------------------------
# 布局 / 解码
# ---------------------------------------------------------------------------

def dfl_or_direct_box(box: np.ndarray, stride: float) -> np.ndarray:
    """单尺度 box 解码为 xywh（reg_max 由通道数 //4 推断）。"""
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
    """识别 ONNX/RKNN 输出布局。"""
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


def infer_reg_max_from_outputs(outputs: list[np.ndarray], layout: str, nc: int) -> int | None:
    """从拆头 box 分支通道推断 reg_max（仅用于打印）。"""
    if layout in ("split6", "split9"):
        try:
            box, _ = pair_box_cls(outputs[0], outputs[1], nc)
            return int(box.shape[1]) // 4
        except Exception:  # noqa: BLE001
            return None
    if layout == "fused":
        o = outputs[0]
        return 1
    return None


def decode_split(outputs: list[np.ndarray], nc: int, imgsz: int) -> np.ndarray:
    """解码 6/9 路拆头为 (1, 4+nc, N)。9 路时忽略 score_sum。"""
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
    """fused 框坐标若为 0~1 归一化，必要时 ×imgsz。"""
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
    """对已是 xyxy/conf/cls 的 (N,6) 再做分类别 NMS。"""
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


def outputs_to_dets(
    outs: list[np.ndarray],
    nc: int,
    imgsz: int,
    layout: str,
    conf: float,
    iou: float,
    box_scale: str,
) -> np.ndarray:
    """将模型输出解码为 letterbox 坐标 (K,6)。"""
    outs = [np.asarray(o, dtype=np.float32) for o in outs]
    if layout in ("split6", "split9"):
        pred = decode_split(outs, nc, imgsz)
        return nms_from_raw(pred, conf, iou)
    if layout == "fused":
        pred = maybe_scale_fused(normalize_fused(outs[0], nc), imgsz, box_scale)
        cls = maybe_sigmoid_cls(pred[0, 4:])
        pred = np.concatenate([pred[:, :4], cls[None]], axis=1)
        return nms_from_raw(pred, conf, iou)
    if layout == "e2e":
        return nms_xyxy(parse_e2e(outs[0]), conf, iou)
    raise ValueError(f"未知 layout={layout}")


# ---------------------------------------------------------------------------
# 推理后端
# ---------------------------------------------------------------------------

def load_onnx(path: Path) -> tuple[ort.InferenceSession, str]:
    """加载 ONNX Runtime 会话。"""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return sess, name


def validate_onnx(path: Path) -> None:
    """可选校验 ONNX 文件完整性。"""
    try:
        import onnx  # noqa: WPS433

        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        n_in, n_out = len(model.graph.input), len(model.graph.output)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"ONNX OK: size={size_mb:.2f}MB inputs={n_in} outputs={n_out}")
    except ImportError:
        print("WARN: 未安装 onnx 包，跳过 ONNX 校验")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"ONNX 文件无效: {exc}") from exc


def probe_onnx(sess: ort.InferenceSession, input_name: str, nc: int, imgsz: int) -> tuple[str, list[tuple]]:
    """探测 ONNX 输出布局。"""
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    outs = [np.asarray(o) for o in sess.run(None, {input_name: dummy})]
    layout = detect_layout(outs, nc)
    return layout, [tuple(o.shape) for o in outs]


def load_rknn(path: Path) -> RKNNLite:
    """加载并初始化 RKNN。"""
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(path))
    if ret != 0:
        raise RuntimeError(f"load_rknn 失败 ({ret}): {path}")
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime 失败 ({ret}): {path}")
    restore_logging_levels()
    return rknn


def probe_rknn(rknn: RKNNLite, nc: int, imgsz: int) -> tuple[str, list[tuple]]:
    """探测 RKNN 输出布局。"""
    dummy = np.zeros((1, imgsz, imgsz, 3), dtype=np.uint8)
    outs = [np.asarray(o) for o in rknn.inference(inputs=[dummy])]
    layout = detect_layout(outs, nc)
    return layout, [tuple(o.shape) for o in outs]


# ---------------------------------------------------------------------------
# 张量对比
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度。"""
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def tensor_pair_metrics(onnx_t: np.ndarray, rknn_t: np.ndarray) -> dict:
    """单输出张量的 cos / sig_cos / mae / max_abs。"""
    o = onnx_t.astype(np.float32)
    r = rknn_t.astype(np.float32)
    if o.shape != r.shape:
        raise ValueError(f"shape 不一致: {o.shape} vs {r.shape}")
    diff = np.abs(o - r)
    mae = float(diff.mean())
    max_abs = float(diff.max())
    cos = cosine_sim(o, r)
    mask = (np.abs(o) > SIG_THRESH) | (np.abs(r) > SIG_THRESH)
    if mask.any():
        sig_cos = cosine_sim(o[mask], r[mask])
    else:
        sig_cos = 1.0 if mae < 0.01 else cos
    return {"cos": cos, "sig_cos": sig_cos, "mae": mae, "max_abs": max_abs}


def remap_output_indices(onnx_outs: list[np.ndarray], rknn_outs: list[np.ndarray]) -> list[int] | None:
    """按 shape 为 ONNX 每个输出找 RKNN 对应下标；失败返回 None。"""
    if len(onnx_outs) != len(rknn_outs):
        return None
    unused = set(range(len(rknn_outs)))
    mapping: list[int] = []
    for o in onnx_outs:
        match_j = None
        for j in sorted(unused):
            if tuple(o.shape) == tuple(rknn_outs[j].shape):
                match_j = j
                break
        if match_j is None:
            return None
        mapping.append(match_j)
        unused.remove(match_j)
    return mapping


def compare_raw_tensors(
    onnx_outs: list[np.ndarray],
    rknn_outs: list[np.ndarray],
    nc: int,
    layout: str,
) -> tuple[list[dict] | None, list[int] | None]:
    """逐输出对比原始张量；layout 为 e2e 时不对比。"""
    if layout == "e2e":
        return None, None
    mapping = remap_output_indices(onnx_outs, rknn_outs)
    if mapping is None:
        return None, None
    rows: list[dict] = []
    for i, j in enumerate(mapping):
        m = tensor_pair_metrics(onnx_outs[i], rknn_outs[j])
        m["idx"] = i
        m["onnx_shape"] = tuple(onnx_outs[i].shape)
        m["rknn_shape"] = tuple(rknn_outs[j].shape)
        rows.append(m)
    if layout == "fused" and len(onnx_outs) == 1:
        o = normalize_fused(np.asarray(onnx_outs[0], dtype=np.float32), nc)[0]
        r = normalize_fused(np.asarray(rknn_outs[mapping[0]], dtype=np.float32), nc)[0]
        box_m = tensor_pair_metrics(o[:4], r[:4])
        box_m["idx"] = "fused_box"
        box_m["onnx_shape"] = (4, o.shape[1])
        box_m["rknn_shape"] = (4, r.shape[1])
        rows.append(box_m)
        if nc > 0:
            cls_m = tensor_pair_metrics(o[4:], r[4:])
            cls_m["idx"] = "fused_cls"
            cls_m["onnx_shape"] = (nc, o.shape[1])
            cls_m["rknn_shape"] = (nc, r.shape[1])
            rows.append(cls_m)
    return rows, mapping


def detect_score_killed_fused(onnx_out: np.ndarray, rknn_out: np.ndarray, nc: int) -> bool:
    """fused i8 分数通道被量化抹零：RKNN 分数近 0 而 ONNX 仍有有效信号。"""
    o = normalize_fused(np.asarray(onnx_out, dtype=np.float32), nc)[0, 4:]
    r = normalize_fused(np.asarray(rknn_out, dtype=np.float32), nc)[0, 4:]
    onnx_sig = float(np.max(np.abs(o)))
    rknn_sig = float(np.max(np.abs(r)))
    onnx_active = float(np.mean(np.abs(o) > SIG_THRESH))
    rknn_active = float(np.mean(np.abs(r) > SIG_THRESH))
    if onnx_sig > 0.05 and rknn_sig < 0.02 and onnx_active > 0.001 and rknn_active < 1e-4:
        return True
    if onnx_sig > 0.2 and rknn_sig < onnx_sig * 0.05:
        return True
    return False


def summarize_tensor_rows(rows: list[dict] | None) -> dict | None:
    """汇总单图张量指标。"""
    if not rows:
        return None
    cos_list = [r["cos"] for r in rows]
    sig_list = [r["sig_cos"] for r in rows]
    mae_list = [r["mae"] for r in rows]
    return {
        "cos_min": float(min(cos_list)),
        "sig_cos_min": float(min(sig_list)),
        "mae_mean": float(np.mean(mae_list)),
        "max_abs_max": float(max(r["max_abs"] for r in rows)),
    }


def print_tensor_table(rows: list[dict], mapping: list[int] | None) -> None:
    """打印单图原始张量对比表。"""
    if mapping is not None:
        print(f"  Output index mapping (onnx_i -> rknn_j): {mapping}")
    print(f"  {'idx':>4}  {'onnx_shape':<18} {'rknn_shape':<18} {'cos':>8} {'sig_cos':>8} {'mae':>10} {'max_abs':>10}")
    for r in rows:
        idx = r["idx"]
        print(
            f"  {str(idx):>4}  {str(r['onnx_shape']):<18} {str(r['rknn_shape']):<18} "
            f"{r['cos']:8.6f} {r['sig_cos']:8.6f} {r['mae']:10.6f} {r['max_abs']:10.6f}"
        )
    s = summarize_tensor_rows(rows)
    if s:
        print(
            f"  Raw summary: cos_min={s['cos_min']:.6f} sig_cos_min={s['sig_cos_min']:.6f} "
            f"mae_mean={s['mae_mean']:.6f} max_abs_max={s['max_abs_max']:.6f}"
        )


# ---------------------------------------------------------------------------
# 框匹配 / 报告
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


def match_dets(onnx: np.ndarray, rknn: np.ndarray, iou_thr: float) -> dict:
    """按 IoU 且同类贪心匹配 ONNX 与 RKNN 检测框。"""
    ious = box_iou(onnx[:, :4] if len(onnx) else onnx, rknn[:, :4] if len(rknn) else rknn)
    used: set[int] = set()
    pairs = []
    for i in range(len(onnx)):
        best_j, best_iou = -1, 0.0
        for j in range(len(rknn)):
            if j in used:
                continue
            if ious[i, j] > best_iou:
                best_iou = float(ious[i, j])
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr and int(onnx[i, 5]) == int(rknn[best_j, 5]):
            used.add(best_j)
            pairs.append(
                {
                    "onnx_cls": int(onnx[i, 5]),
                    "onnx_conf": float(onnx[i, 4]),
                    "rknn_conf": float(rknn[best_j, 4]),
                    "iou": best_iou,
                    "dconf": abs(float(onnx[i, 4]) - float(rknn[best_j, 4])),
                }
            )
    n_onnx, n_rknn, n_m = int(len(onnx)), int(len(rknn)), len(pairs)
    n_high_onnx = int(np.sum(onnx[:, 4] >= HIGH_CONF)) if len(onnx) else 0
    min_iou = float(min((p["iou"] for p in pairs), default=1.0)) if pairs else 1.0
    mean_iou = float(np.mean([p["iou"] for p in pairs])) if pairs else 1.0
    max_dc = float(max((p["dconf"] for p in pairs), default=0.0)) if pairs else 0.0
    mean_dc = float(np.mean([p["dconf"] for p in pairs])) if pairs else 0.0
    if n_onnx == n_rknn == n_m:
        level = "pass" if (n_m == 0 or (min_iou >= 0.9 and max_dc <= 0.15)) else "warn"
    elif n_m >= max(1, int(0.8 * max(n_onnx, n_rknn, 1))):
        level = "warn"
    else:
        level = "fail"
    return {
        "n_onnx": n_onnx,
        "n_rknn": n_rknn,
        "n_match": n_m,
        "n_high_onnx": n_high_onnx,
        "min_iou": min_iou,
        "mean_iou": mean_iou,
        "max_dconf": max_dc,
        "mean_dconf": mean_dc,
        "level": level,
        "pairs": pairs,
    }


def image_level(
    det: dict,
    tensor_sum: dict | None,
    score_killed: bool,
) -> str:
    """单图 PASS/WARN/FAIL（张量 + 框 + score 抹零）。"""
    if score_killed:
        return "fail"
    box_level = det["level"]
    if tensor_sum is None:
        return box_level
    sig = tensor_sum["sig_cos_min"]
    det_ok = det["n_onnx"] == det["n_rknn"] == det["n_match"]
    if sig >= 0.99 and det_ok and (det["n_match"] == 0 or (det["min_iou"] >= 0.9 and det["max_dconf"] <= 0.15)):
        return "pass"
    if sig >= 0.95 and det["n_match"] >= max(1, int(0.8 * max(det["n_onnx"], det["n_rknn"], 1))):
        return "warn" if box_level != "fail" else "fail"
    if sig < 0.95:
        return "fail"
    return box_level


def aggregate_metrics(rows: list[dict]) -> dict:
    """整集框匹配指标汇总。"""
    n_onnx = sum(r["n_onnx"] for r in rows)
    n_rknn = sum(r["n_rknn"] for r in rows)
    n_match = sum(r["n_match"] for r in rows)
    ious: list[float] = []
    dconfs: list[float] = []
    high_ious: list[float] = []
    high_dconfs: list[float] = []
    n_high_onnx = 0
    n_high_match = 0
    for r in rows:
        n_high_onnx += int(r.get("n_high_onnx", 0))
        for p in r.get("pairs", []):
            ious.append(float(p["iou"]))
            dconfs.append(float(p["dconf"]))
            if float(p["onnx_conf"]) >= HIGH_CONF:
                high_ious.append(float(p["iou"]))
                high_dconfs.append(float(p["dconf"]))
                n_high_match += 1

    denom = max(n_onnx, n_rknn, 1)
    out = {
        "n_images": len(rows),
        "n_onnx": n_onnx,
        "n_rknn": n_rknn,
        "n_match": n_match,
        "match_rate": n_match / denom,
        "recall_vs_onnx": (n_match / n_onnx) if n_onnx else 1.0,
        "extra_rate": ((n_rknn - n_match) / n_rknn) if n_rknn else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 1.0,
        "p5_iou": float(np.percentile(ious, 5)) if ious else 1.0,
        "mean_dconf": float(np.mean(dconfs)) if dconfs else 0.0,
        "max_dconf": float(np.max(dconfs)) if dconfs else 0.0,
        "n_high_onnx": n_high_onnx,
        "n_high_match": n_high_match,
        "high_recall": (n_high_match / n_high_onnx) if n_high_onnx else 1.0,
        "high_mean_iou": float(np.mean(high_ious)) if high_ious else 1.0,
        "high_mean_dconf": float(np.mean(high_dconfs)) if high_dconfs else 0.0,
    }
    t_rows = [r["tensor_sum"] for r in rows if r.get("tensor_sum")]
    if t_rows:
        out["tensor_cos_min"] = float(min(t["cos_min"] for t in t_rows))
        out["tensor_sig_cos_min"] = float(min(t["sig_cos_min"] for t in t_rows))
        out["tensor_mae_mean"] = float(np.mean([t["mae_mean"] for t in t_rows]))
    return out


def conversion_verdict(m: dict) -> str:
    """根据整集数字给出转换结论。"""
    if m["match_rate"] >= 0.95 and m["mean_iou"] >= 0.90 and m["mean_dconf"] <= 0.05:
        return "转换可用：与 ONNX 参考检出高度一致。"
    if m["match_rate"] >= 0.85 and m["mean_iou"] >= 0.80:
        return "i8 可接受：有量化漂移，建议抽看 FAIL/低分框。"
    return "偏差较大：请检查导出头、预处理、后处理或量化。"


def print_dataset_report(tag: str, layout: str, rows: list[dict], counter: Counter) -> None:
    """打印整集报告。"""
    m = aggregate_metrics(rows)
    print(f"\n=== ONNX vs {tag} ({layout}) ===")
    print(f"  n_images        {m['n_images']}")
    print(f"  n_onnx / n_rknn {m['n_onnx']} / {m['n_rknn']}")
    print(f"  n_match         {m['n_match']}")
    print(f"  match_rate      {m['match_rate']:.4f}")
    print(f"  recall_vs_onnx  {m['recall_vs_onnx']:.4f}")
    print(f"  extra_rate      {m['extra_rate']:.4f}")
    print(f"  mean_iou        {m['mean_iou']:.4f}")
    print(f"  p5_iou          {m['p5_iou']:.4f}")
    print(f"  mean|dconf|     {m['mean_dconf']:.4f}")
    print(f"  max|dconf|      {m['max_dconf']:.4f}")
    print(
        f"  high_conf(≥{HIGH_CONF:.2f})  "
        f"recall={m['high_recall']:.4f}  mean_iou={m['high_mean_iou']:.4f}  "
        f"mean|dconf|={m['high_mean_dconf']:.4f}  "
        f"(onnx_high={m['n_high_onnx']}, matched={m['n_high_match']})"
    )
    if "tensor_sig_cos_min" in m:
        print(
            f"  tensor          cos_min={m['tensor_cos_min']:.6f}  "
            f"sig_cos_min={m['tensor_sig_cos_min']:.6f}  mae_mean={m['tensor_mae_mean']:.6f}"
        )
    score_killed_n = sum(1 for r in rows if r.get("score_killed"))
    if score_killed_n:
        print(f"  score抹零       {score_killed_n} 张图检测到 fused 分数通道异常（已判 FAIL）")
    print(f"  结论            {conversion_verdict(m)}")
    n = max(len(rows), 1)
    print(
        f"  PASS={counter.get('pass', 0)} ({counter.get('pass', 0) / n:.1%})  "
        f"WARN={counter.get('warn', 0)} ({counter.get('warn', 0) / n:.1%})  "
        f"FAIL={counter.get('fail', 0)} ({counter.get('fail', 0) / n:.1%})"
    )
    fails = [x for x in rows if x["level"] == "fail"]
    for r in fails[:12]:
        extra = ""
        if r.get("score_killed"):
            extra = " [score_killed]"
        print(
            f"  FAIL {r['name']}: {r['n_onnx']}/{r['n_rknn']}/{r['n_match']} "
            f"min_iou={r['min_iou']:.3f} max_dconf={r['max_dconf']:.3f}{extra}"
        )


def draw_dets(
    bgr: np.ndarray,
    dets: np.ndarray,
    names: dict,
    title: str,
    co_helper,
    letterbox_coords: bool,
) -> np.ndarray:
    """在原图画框。"""
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
        description="ONNX vs RKNN 精度对比（YOLOv8 / YOLO11 / YOLO26）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx", default=str(HERE / "safety_helmet_all.onnx"), help="ONNX 模型")
    p.add_argument(
        "--rknn",
        nargs="+",
        default=[str(HERE / "safety_helmet_all_i8.rknn")],
        help="一个或多个 RKNN 模型",
    )
    p.add_argument(
        "--source",
        "--img_folder",
        default="/userdata/jovan/code/rk3588/dataset/helmet/",
        dest="source",
        help="图片或目录（--img_folder 为别名）",
    )
    p.add_argument("--nc", type=int, default=2, help="类别数")
    p.add_argument("--names", default="head,helmet", help="逗号分隔类别名")
    p.add_argument("--out_dir", default=str(HERE / "compare_result_onnx_vs_rknn"), help="可视化输出目录")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    p.add_argument("--match-iou", type=float, default=0.5, help="框匹配 IoU")
    p.add_argument(
        "--family",
        choices=("auto", "v8", "v11", "v26"),
        default="auto",
        help="模型族提示（解码仍按通道数）",
    )
    p.add_argument(
        "--box-scale",
        choices=("auto", "on", "off"),
        default="auto",
        help="fused 框是否 ×imgsz",
    )
    p.add_argument("--max_images", type=int, default=0, help="0=全部")
    p.add_argument("--img_save", action="store_true", help="保存对比图")
    p.add_argument("--save_limit", type=int, default=5, help="最多保存前 N 张")
    p.add_argument("--no-validate-onnx", action="store_true", help="跳过 onnx 包校验")
    return p.parse_args()


def main() -> int:
    """逐图对比 ONNX 与各 RKNN。"""
    args = parse_args()
    onnx_path = Path(args.onnx)
    rknn_paths = [Path(p) for p in args.rknn]
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX 不存在: {onnx_path}")
    for rp in rknn_paths:
        if not rp.is_file():
            raise FileNotFoundError(f".rknn 不存在: {rp}")

    names = parse_names(args.names, args.nc)
    nc = args.nc
    images = list_images(Path(args.source))
    if args.max_images > 0:
        images = images[: args.max_images]
    out_dir = Path(args.out_dir)
    if args.img_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_validate_onnx:
        print("Validate ONNX...")
        validate_onnx(onnx_path)

    print(f"Load ONNX: {onnx_path}")
    sess, input_name = load_onnx(onnx_path)
    onnx_layout, onnx_shapes = probe_onnx(sess, input_name, nc, args.imgsz)
    dummy = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
    probe_outs = [np.asarray(o) for o in sess.run(None, {input_name: dummy})]
    reg_hint = infer_reg_max_from_outputs(probe_outs, onnx_layout, nc)

    print("=== ONNX vs RKNN ===")
    print(f"onnx     : {onnx_path}")
    print(f"family   : {family_hint_label(args.family, reg_hint)}")
    print(f"nc/names : {nc}  {names}")
    print(f"onnx     : layout={onnx_layout}  shapes={onnx_shapes}")
    print(f"align    : box-scale={args.box_scale}")
    print(f"nms      : conf={args.conf} iou={args.iou} match-iou={args.match_iou}")
    print(f"source   : {args.source} ({len(images)} images)")

    models = []
    for rp in rknn_paths:
        print(f"Load RKNN: {rp}")
        rknn = load_rknn(rp)
        layout, shapes = probe_rknn(rknn, nc, args.imgsz)
        if layout != onnx_layout:
            print(f"WARN: RKNN layout={layout} 与 ONNX layout={onnx_layout} 不一致，按各自解码")
        tag = rp.stem
        models.append({"tag": tag, "path": rp, "rknn": rknn, "layout": layout, "shapes": shapes})
        print(f"RKNN     : {rp.name}  layout={layout}  shapes={shapes}")

    counters = {m["tag"]: Counter() for m in models}
    rows: dict[str, list] = {m["tag"]: [] for m in models}
    co = COCO_test_helper(enable_letter_box=True)
    t0 = time.time()

    for i, img_path in enumerate(images):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"读图失败: {img_path}")
        img_lb = co.letter_box(im=bgr.copy(), new_shape=(args.imgsz, args.imgsz), pad_color=(0, 0, 0))
        rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        onnx_in = np.ascontiguousarray((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        rknn_in = np.expand_dims(rgb, 0).astype(np.uint8)

        onnx_outs = [np.asarray(o) for o in sess.run(None, {input_name: onnx_in})]
        line = f"[{i + 1}/{len(images)}] {img_path.name}"
        panels: list[np.ndarray] = []
        onnx_dets = outputs_to_dets(onnx_outs, nc, args.imgsz, onnx_layout, args.conf, args.iou, args.box_scale)
        onnx_letterbox = onnx_layout != "e2e"

        for m in models:
            rk_outs = [np.asarray(o) for o in m["rknn"].inference(inputs=[rknn_in])]
            layout = m["layout"]
            if layout == onnx_layout:
                tensor_rows, mapping = compare_raw_tensors(onnx_outs, rk_outs, nc, layout)
            else:
                tensor_rows, mapping = None, None
            tensor_sum = summarize_tensor_rows(tensor_rows)

            score_killed = False
            if layout == "fused" == onnx_layout and len(onnx_outs) == 1:
                score_killed = detect_score_killed_fused(onnx_outs[0], rk_outs[0], nc)

            rk_dets = outputs_to_dets(rk_outs, nc, args.imgsz, layout, args.conf, args.iou, args.box_scale)
            rk_letterbox = layout != "e2e"

            st = match_dets(onnx_dets, rk_dets, args.match_iou)
            st["name"] = img_path.name
            st["tensor_sum"] = tensor_sum
            st["score_killed"] = score_killed
            st["level"] = image_level(st, tensor_sum, score_killed)
            rows[m["tag"]].append(st)
            counters[m["tag"]][st["level"]] += 1

            sig_s = f" sig={tensor_sum['sig_cos_min']:.4f}" if tensor_sum else ""
            sk = " SCORE_KILLED" if score_killed else ""
            line += (
                f" | {m['tag']}:{st['level']} "
                f"{st['n_onnx']}/{st['n_rknn']}/{st['n_match']} "
                f"iou={st['mean_iou']:.3f} dconf={st['max_dconf']:.3f}{sig_s}{sk}"
            )

            verbose = i < 2 or st["level"] == "fail" or score_killed
            if verbose and tensor_rows:
                print(f"\nInfer {i + 1}/{len(images)} {img_path.name}  ({m['tag']})")
                print(f"  ONNX outs: {len(onnx_outs)}  RKNN outs: {len(rk_outs)}")
                print_tensor_table(tensor_rows, mapping)

            if args.img_save and i < args.save_limit:
                if not panels:
                    panels.append(
                        draw_dets(bgr, onnx_dets, names, "ONNX", co, letterbox_coords=onnx_letterbox)
                    )
                panels.append(
                    draw_dets(bgr, rk_dets, names, m["tag"][:22], co, letterbox_coords=rk_letterbox)
                )

        fail_now = any(rows[m["tag"]][-1]["level"] == "fail" for m in models)
        if i < 3 or fail_now or (i + 1) % 32 == 0:
            print(line)
        if panels:
            cv2.imwrite(str(out_dir / f"cmp_{img_path.stem}.jpg"), hstack_panels(panels))

    for m in models:
        m["rknn"].release()

    print(f"\ndone in {time.time() - t0:.1f}s  images={len(images)}")
    for m in models:
        print_dataset_report(m["tag"], m["layout"], rows[m["tag"]], counters[m["tag"]])

    if args.img_save:
        print(f"views: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
