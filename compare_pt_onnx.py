# -*- coding: utf-8 -*-
"""
通用 YOLO .pt vs .onnx 转换精度对比（Ultralytics detect）。

需要准备：
  1. --pt     原权重 .pt（必填）
  2. --onnx   转换后的 .onnx（必填）
  3. --source 测试图文件或目录（建议真图；没有可用随机张量只比 tensor）
  4. --imgsz  输入边长（默认读 ONNX 固定 shape，否则 640）
  5. --conf / --iou  与业务一致的阈值（默认 0.25 / 0.7）

对比两层：
  A. 同一输入张量下，解码后的 (1, 4+nc, N) 数值：cosine / mae / max
  B. 同一 NMS 后的检测框：数量、类别、IoU、|Δconf|

支持的 ONNX 输出：
  - 官方 Ultralytics 单输出：(1, 4+nc, N) 或 (1, N, 4+nc)
  - 拆头 6 输出：每尺度 box(64,H,W) + cls(nc,H,W)
  - 拆头 9 输出：每尺度 box + cls + 额外 1 通道（会忽略第 3 路）
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from pathlib import Path

if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHT_DIR = SCRIPT_DIR / "weight"
DEFAULT_PT = WEIGHT_DIR / "helmet_y11s_best.pt"
DEFAULT_ONNX = WEIGHT_DIR / "helmet_y11s_best.onnx"
DEFAULT_IMAGES = SCRIPT_DIR / "images"
DEFAULT_OUTPUT = SCRIPT_DIR / "result_views" / "pt_onnx_compare"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
STRIDES = (8, 16, 32)
REG_MAX = 16


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def list_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Not an image: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = [p for p in sorted(source.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No images in {source}")
    return files


def resolve_imgsz(sess: ort.InferenceSession, user_imgsz: int | None) -> int:
    if user_imgsz is not None:
        return int(user_imgsz)
    shape = sess.get_inputs()[0].shape
    # expected [N, C, H, W]
    if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        if shape[2] != shape[3]:
            raise ValueError(f"Non-square ONNX input not supported yet: {shape}")
        return int(shape[2])
    return 640


def preprocess(bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """Ultralytics-style letterbox + RGB + /255 + NCHW float32."""
    letterboxed = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=bgr)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


# ---------------------------------------------------------------------------
# ONNX decode: unify to (1, 4+nc, N) xywh + class scores
# ---------------------------------------------------------------------------

def _dfl_decode_box(box: np.ndarray, stride: float) -> np.ndarray:
    """box: (1, 64, H, W) -> xywh (4, H*W)."""
    _, c, h, w = box.shape
    if c != 4 * REG_MAX:
        raise ValueError(f"Expect box channels={4 * REG_MAX}, got {c}")
    reg = box.reshape(1, 4, REG_MAX, h, w)
    reg = reg - reg.max(axis=2, keepdims=True)
    exp = np.exp(reg)
    dfl = (exp / exp.sum(axis=2, keepdims=True) * np.arange(REG_MAX, dtype=np.float32).reshape(1, 1, REG_MAX, 1, 1)).sum(
        axis=2
    )
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


def _normalize_end2end(out: np.ndarray, nc: int) -> np.ndarray:
    """Accept (1,4+nc,N) or (1,N,4+nc) -> (1,4+nc,N)."""
    if out.ndim != 3:
        raise ValueError(f"Unexpected end2end rank: {out.shape}")
    if out.shape[1] == 4 + nc:
        return out.astype(np.float32)
    if out.shape[2] == 4 + nc:
        return out.transpose(0, 2, 1).astype(np.float32)
    raise ValueError(f"Cannot map end2end shape {out.shape} to nc={nc}")


def detect_onnx_layout(outputs: list[np.ndarray], nc: int) -> str:
    n = len(outputs)
    if n == 1:
        return "end2end"
    if n in (6, 9) and all(o.ndim == 4 for o in outputs):
        # box should be 64-ch; cls should be nc-ch
        step = n // 3
        box0, cls0 = outputs[0], outputs[1]
        if box0.shape[1] == 4 * REG_MAX and cls0.shape[1] == nc:
            return "split9" if step == 3 else "split6"
    raise ValueError(
        "Unsupported ONNX outputs. Supported: "
        "1x end2end (1,4+nc,N)/(1,N,4+nc), or 6/9 split-head tensors. "
        f"Got {[o.shape for o in outputs]}"
    )


def decode_onnx(outputs: list[np.ndarray], nc: int, imgsz: int) -> tuple[np.ndarray, str]:
    layout = detect_onnx_layout(outputs, nc)
    if layout == "end2end":
        return _normalize_end2end(outputs[0], nc), layout

    step = 3 if layout == "split9" else 2
    chunks = []
    for i in range(3):
        box = outputs[i * step + 0]
        cls = outputs[i * step + 1]
        h, w = int(box.shape[2]), int(box.shape[3])
        # Prefer geometric stride; fall back to classic 8/16/32.
        stride = float(imgsz / h) if h > 0 else float(STRIDES[i])
        xywh = _dfl_decode_box(box, stride)
        cls_flat = cls[0].reshape(cls.shape[1], -1).astype(np.float32)
        # if logits (outside [0,1]), apply sigmoid
        if cls_flat.min() < -1e-3 or cls_flat.max() > 1.0 + 1e-3:
            cls_flat = 1.0 / (1.0 + np.exp(-np.clip(cls_flat, -80, 80)))
        chunks.append(np.concatenate([xywh, cls_flat], axis=0))
    return np.concatenate(chunks, axis=1)[None].astype(np.float32), layout


def pt_forward(model: YOLO, tensor: np.ndarray) -> np.ndarray:
    """Run Ultralytics DetectionModel, return (1, 4+nc, N) numpy."""
    with torch.no_grad():
        out = model.model(torch.from_numpy(tensor))
    if isinstance(out, (tuple, list)):
        pred = out[0]
    else:
        pred = out
    if not torch.is_tensor(pred):
        raise TypeError(f"Unexpected PT output type: {type(pred)}")
    return pred.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def tensor_report(pt_pred: np.ndarray, onnx_pred: np.ndarray) -> dict:
    if pt_pred.shape != onnx_pred.shape:
        raise ValueError(f"Shape mismatch: pt={pt_pred.shape} onnx={onnx_pred.shape}")
    box_diff = np.abs(pt_pred[:, :4] - onnx_pred[:, :4])
    cls_diff = np.abs(pt_pred[:, 4:] - onnx_pred[:, 4:])
    a, b = pt_pred.reshape(-1), onnx_pred.reshape(-1)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {
        "shape": list(pt_pred.shape),
        "box_mae": float(box_diff.mean()),
        "box_max": float(box_diff.max()),
        "cls_mae": float(cls_diff.mean()),
        "cls_max": float(cls_diff.max()),
        "all_mae": float(np.mean(np.abs(a - b))),
        "all_max": float(np.max(np.abs(a - b))),
        "cosine": cosine,
    }


def nms_boxes(pred: np.ndarray, conf: float, iou: float) -> np.ndarray:
    dets = non_max_suppression(
        torch.from_numpy(pred.copy()),
        conf_thres=conf,
        iou_thres=iou,
        agnostic=False,
        max_det=300,
    )[0]
    if dets is None or len(dets) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return dets.cpu().numpy()


def scale_to_original(dets: np.ndarray, imgsz: int, orig_hw: tuple[int, int]) -> np.ndarray:
    if len(dets) == 0:
        return dets
    out = dets.copy()
    out[:, :4] = scale_boxes((imgsz, imgsz), torch.from_numpy(out[:, :4].copy()), orig_hw).numpy()
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_dets(pt: np.ndarray, onnx: np.ndarray, iou_thr: float) -> dict:
    ious = box_iou(pt[:, :4] if len(pt) else pt, onnx[:, :4] if len(onnx) else onnx)
    used: set[int] = set()
    pairs = []
    for i in range(len(pt)):
        best_j, best_iou = -1, 0.0
        for j in range(len(onnx)):
            if j in used:
                continue
            if ious[i, j] > best_iou:
                best_iou = float(ious[i, j])
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr and int(pt[i, 5]) == int(onnx[best_j, 5]):
            used.add(best_j)
            pairs.append(
                {
                    "pt_cls": int(pt[i, 5]),
                    "onnx_cls": int(onnx[best_j, 5]),
                    "pt_conf": float(pt[i, 4]),
                    "onnx_conf": float(onnx[best_j, 4]),
                    "iou": best_iou,
                    "dconf": abs(float(pt[i, 4]) - float(onnx[best_j, 4])),
                }
            )
    return {
        "n_pt": int(len(pt)),
        "n_onnx": int(len(onnx)),
        "n_matched": len(pairs),
        "pairs": pairs,
        "min_iou": float(min((p["iou"] for p in pairs), default=0.0)),
        "max_dconf": float(max((p["dconf"] for p in pairs), default=0.0)),
    }


def draw_boxes(bgr: np.ndarray, dets: np.ndarray, names: dict, color: tuple[int, int, int], prefix: str) -> np.ndarray:
    vis = bgr.copy()
    for x1, y1, x2, y2, conf, cls_id in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(vis, p1, p2, color, 2)
        label = names.get(int(cls_id), str(int(cls_id)))
        cv2.putText(
            vis,
            f"{prefix}{label} {conf:.3f}",
            (p1[0], max(20, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return vis


def judge(tensor: dict, det: dict, args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons = []
    if tensor["cosine"] < args.min_cosine:
        reasons.append(f"cosine {tensor['cosine']:.6f} < {args.min_cosine}")
    if tensor["box_max"] > args.max_box_err:
        reasons.append(f"box_max {tensor['box_max']:.4g} > {args.max_box_err}")
    if tensor["cls_max"] > args.max_cls_err:
        reasons.append(f"cls_max {tensor['cls_max']:.4g} > {args.max_cls_err}")
    if det["n_pt"] != det["n_onnx"] or det["n_matched"] != det["n_pt"]:
        reasons.append(f"det count pt={det['n_pt']} onnx={det['n_onnx']} matched={det['n_matched']}")
    if det["n_matched"] and det["min_iou"] < args.min_iou:
        reasons.append(f"min_iou {det['min_iou']:.4f} < {args.min_iou}")
    if det["n_matched"] and det["max_dconf"] > args.max_dconf:
        reasons.append(f"max_dconf {det['max_dconf']:.4g} > {args.max_dconf}")
    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generic YOLO .pt vs .onnx conversion accuracy check.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pt", default=str(DEFAULT_PT), help="Ultralytics .pt weight")
    p.add_argument("--onnx", default=str(DEFAULT_ONNX), help="Converted .onnx model")
    p.add_argument("--source", default=str(DEFAULT_IMAGES), help="Image file or directory")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Visualization output dir")
    p.add_argument("--imgsz", type=int, default=None, help="Inference size; default from ONNX")
    p.add_argument("--conf", type=float, default=0.25, help="NMS confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--match-iou", type=float, default=0.5, help="Box matching IoU for pairing")
    p.add_argument("--min-cosine", type=float, default=0.9999)
    p.add_argument("--max-box-err", type=float, default=0.05, help="Max |xywh| error on model canvas (px)")
    p.add_argument("--max-cls-err", type=float, default=1e-3)
    p.add_argument("--min-iou", type=float, default=0.99, help="Min IoU of matched boxes")
    p.add_argument("--max-dconf", type=float, default=1e-3, help="Max |conf_pt - conf_onnx|")
    p.add_argument("--no-vis", action="store_true", help="Skip saving comparison images")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pt_path, onnx_path = Path(args.pt), Path(args.onnx)
    if not pt_path.is_file():
        raise FileNotFoundError(f".pt not found: {pt_path}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f".onnx not found: {onnx_path}")

    images = list_images(Path(args.source))
    out_dir = Path(args.output)
    if not args.no_vis:
        out_dir.mkdir(parents=True, exist_ok=True)

    pt_model = YOLO(str(pt_path))
    if getattr(pt_model, "task", "detect") != "detect":
        raise ValueError(f"This script currently supports detect only, got task={pt_model.task}")
    pt_model.model.eval()
    names = pt_model.names if isinstance(pt_model.names, dict) else {i: n for i, n in enumerate(pt_model.names)}
    nc = len(names)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    imgsz = resolve_imgsz(sess, args.imgsz)

    # Probe ONNX layout once
    probe = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    probe_out = sess.run(None, {input_name: probe})
    _, layout = decode_onnx(probe_out, nc, imgsz)

    print("=== YOLO PT vs ONNX compare ===")
    print(f"pt       : {pt_path}")
    print(f"onnx     : {onnx_path}")
    print(f"task/nc  : detect / {nc}  names={names}")
    print(f"onnx_in  : {input_name} {sess.get_inputs()[0].shape}")
    print(f"onnx_out : {[tuple(o.shape) for o in probe_out]}  layout={layout}")
    print(f"imgsz    : {imgsz}  conf={args.conf}  nms_iou={args.iou}")
    print(f"source   : {args.source} ({len(images)} images)")
    print(
        "need     : .pt + .onnx + test images; optional: --imgsz --conf --iou and pass thresholds"
    )

    all_ok = True
    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        tensor = preprocess(bgr, imgsz)
        pt_pred = pt_forward(pt_model, tensor)
        onnx_pred, _ = decode_onnx(sess.run(None, {input_name: tensor}), nc, imgsz)

        # Align channel count if PT returns more (rare)
        if pt_pred.shape[1] != onnx_pred.shape[1]:
            raise ValueError(f"Channel mismatch after decode: pt={pt_pred.shape} onnx={onnx_pred.shape}")

        tstat = tensor_report(pt_pred, onnx_pred)
        pt_dets = scale_to_original(nms_boxes(pt_pred, args.conf, args.iou), imgsz, bgr.shape[:2])
        onnx_dets = scale_to_original(nms_boxes(onnx_pred, args.conf, args.iou), imgsz, bgr.shape[:2])
        dstat = match_dets(pt_dets, onnx_dets, args.match_iou)
        ok, reasons = judge(tstat, dstat, args)
        all_ok = all_ok and ok

        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {img_path.name}")
        print(
            f"  tensor  shape={tstat['shape']} cosine={tstat['cosine']:.8f} "
            f"box_mae={tstat['box_mae']:.3e} box_max={tstat['box_max']:.3e} "
            f"cls_mae={tstat['cls_mae']:.3e} cls_max={tstat['cls_max']:.3e}"
        )
        print(
            f"  boxes   pt={dstat['n_pt']} onnx={dstat['n_onnx']} matched={dstat['n_matched']} "
            f"min_iou={dstat['min_iou']:.6f} max|dconf|={dstat['max_dconf']:.3e}"
        )
        for pair in dstat["pairs"]:
            label = names.get(pair["pt_cls"], str(pair["pt_cls"]))
            print(
                f"    {label}: pt={pair['pt_conf']:.6f} onnx={pair['onnx_conf']:.6f} "
                f"iou={pair['iou']:.6f} |dconf|={pair['dconf']:.3e}"
            )
        if reasons:
            print("  fail   " + "; ".join(reasons))

        if not args.no_vis:
            vis_pt = draw_boxes(bgr, pt_dets, names, (0, 180, 0), "pt:")
            vis_onnx = draw_boxes(bgr, onnx_dets, names, (0, 0, 220), "onnx:")
            vis_both = draw_boxes(vis_pt, onnx_dets, names, (0, 0, 220), "onnx:")
            banner = np.full((36, vis_both.shape[1] * 3, 3), 40, dtype=np.uint8)
            cv2.putText(
                banner,
                f"{img_path.name}  {status}  layout={layout}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            panel = np.hstack([vis_pt, vis_onnx, vis_both])
            cv2.imwrite(str(out_dir / img_path.name), np.vstack([banner, panel]))

    print(
        "\n"
        + (
            "ALL PASS: ONNX matches .pt within thresholds."
            if all_ok
            else "FAIL: conversion mismatch exceeded thresholds."
        )
    )
    if not args.no_vis:
        print(f"views : {out_dir}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
