#!/usr/bin/env python3
"""Compare ONNX vs RKNN outputs to verify conversion consistency.

Same preprocess / postprocess as yolo11.py:
  - ONNX: NCHW float32 /255
  - RKNN: NHWC uint8 (mean=0, std=255 in convert.py)

Usage:
  python3 compare_onnx_rknn.py \
    --onnx helmet_y11s_best.onnx \
    --rknn helmet_y11s_best_fp.rknn \
    --img_folder /userdata/jovan/code/rk3588/dataset/helmet/ \
    --img_save

  # also compare i8:
  python3 compare_onnx_rknn.py --onnx helmet_y11s_best.onnx \
    --rknn helmet_y11s_best_i8.rknn --img_folder ... --img_save
"""
import os
import sys
import argparse
import cv2
import numpy as np

realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0] + _sep, *realpath[1:realpath.index('rknn_model_zoo') + 1]))

from py_utils.coco_utils import COCO_test_helper
from py_utils.onnx_executor import ONNX_model_container
from py_utils.rknn_executor import RKNN_model_container
from yolo11 import IMG_SIZE, CLASSES, post_process, draw, img_check


def check_onnx_file(path):
    """Fail fast with a clear message if ONNX is truncated/corrupt."""
    size = os.path.getsize(path)
    if size < 1024:
        raise RuntimeError('ONNX too small ({} bytes): {}'.format(size, path))
    try:
        import onnx
        model = onnx.load(path)
        onnx.checker.check_model(model)
        ins = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input]
        outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model.graph.output]
        print('ONNX OK: size={:.2f}MB inputs={} outputs={}'.format(size / 1024 / 1024, len(ins), len(outs)))
        for name, shape in ins:
            print('  in :', name, shape)
        for name, shape in outs[:12]:
            print('  out:', name, shape)
        if len(outs) > 12:
            print('  ... {} more outputs'.format(len(outs) - 12))
        return model
    except Exception as e:
        raise RuntimeError(
            'ONNX file is invalid/corrupt: {}\n'
            '  size={:.2f}MB path={}\n'
            '  detail: {}\n'
            '  Please re-copy the original .onnx (scp/rsync) and verify md5 on both sides.'
            .format(type(e).__name__, size / 1024 / 1024, path, e)
        )


def cosine_sim(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def tensor_metrics(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.shape != b.shape:
        return {
            'shape_a': tuple(a.shape),
            'shape_b': tuple(b.shape),
            'cos': float('nan'),
            'mae': float('nan'),
            'max_abs': float('nan'),
            'rmse': float('nan'),
        }
    diff = np.abs(a - b)
    return {
        'shape_a': tuple(a.shape),
        'shape_b': tuple(b.shape),
        'cos': cosine_sim(a, b),
        'mae': float(diff.mean()),
        'max_abs': float(diff.max()),
        'rmse': float(np.sqrt((diff ** 2).mean())),
    }


def match_outputs_by_shape(onnx_outs, rknn_outs):
    """Align RKNN outputs to ONNX order by exact shape (fallback: same index)."""
    used = set()
    matched = []
    mapping = []
    for i, o in enumerate(onnx_outs):
        found = None
        for j, r in enumerate(rknn_outs):
            if j in used:
                continue
            if tuple(o.shape) == tuple(r.shape):
                found = j
                break
        if found is None:
            found = i if i < len(rknn_outs) else None
        if found is None:
            raise RuntimeError('Cannot match ONNX out[{}] shape {} to any RKNN output'.format(i, o.shape))
        used.add(found)
        matched.append(rknn_outs[found])
        mapping.append(found)
    return matched, mapping


def box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def match_detections(boxes_a, cls_a, scores_a, boxes_b, cls_b, scores_b, iou_thr=0.5):
    if boxes_a is None or boxes_b is None:
        n_a = 0 if boxes_a is None else len(boxes_a)
        n_b = 0 if boxes_b is None else len(boxes_b)
        return [], set(range(n_a)), set(range(n_b))

    pairs = []
    for i in range(len(boxes_a)):
        for j in range(len(boxes_b)):
            if int(cls_a[i]) != int(cls_b[j]):
                continue
            iou = box_iou(boxes_a[i], boxes_b[j])
            if iou >= iou_thr:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)

    used_a, used_b = set(), set()
    matched = []
    for iou, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j, iou))

    unmatched_a = set(range(len(boxes_a))) - used_a
    unmatched_b = set(range(len(boxes_b))) - used_b
    return matched, unmatched_a, unmatched_b


def prepare_image(img_path, co_helper):
    img_src = cv2.imread(img_path)
    if img_src is None:
        return None, None, None
    img = co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    onnx_in = img.transpose((2, 0, 1)).astype(np.float32)
    onnx_in = np.expand_dims(onnx_in, 0) / 255.0
    rknn_in = np.expand_dims(img, 0)  # NHWC uint8, matches mean/std=[0]/[255]
    return img_src, onnx_in, rknn_in


def compare_one(img_name, img_src, onnx_outs, rknn_outs, co_helper, out_dir, save_img):
    print('\n' + '=' * 72)
    print('IMG:', img_name)
    print('ONNX outs: {}  RKNN outs: {}'.format(len(onnx_outs), len(rknn_outs)))
    for i, t in enumerate(onnx_outs):
        print('  onnx[{}] {}'.format(i, t.shape))
    for i, t in enumerate(rknn_outs):
        print('  rknn[{}] {}'.format(i, t.shape))

    rknn_aligned, mapping = match_outputs_by_shape(onnx_outs, rknn_outs)
    print('Output index mapping (onnx_i -> rknn_j):', mapping)

    cos_list, mae_list, max_list = [], [], []
    # 有效区域 cosine：忽略近零背景（否则全零图 cosine 会被浮点底噪毁掉）
    sig_cos_list = []
    print('\n{:>4}  {:>18}  {:>18}  {:>8}  {:>10}  {:>10}  {:>10}'.format(
        'idx', 'onnx_shape', 'rknn_shape', 'cos', 'sig_cos', 'mae', 'max_abs'))
    for i, (o, r) in enumerate(zip(onnx_outs, rknn_aligned)):
        m = tensor_metrics(o, r)
        mask = np.abs(o.astype(np.float64)) > 0.01
        if mask.any() and o.shape == r.shape:
            sig_cos = cosine_sim(o[mask], r[mask])
        else:
            # 整张图都接近 0：看 mae 是否也很小
            sig_cos = 1.0 if m['mae'] < 0.01 else m['cos']
        cos_list.append(m['cos'])
        sig_cos_list.append(sig_cos)
        mae_list.append(m['mae'])
        max_list.append(m['max_abs'])
        print('{:>4}  {:>18}  {:>18}  {:>8.6f}  {:>8.6f}  {:>10.6f}  {:>10.6f}'.format(
            i, str(m['shape_a']), str(m['shape_b']), m['cos'], sig_cos, m['mae'], m['max_abs']))

    print('\nRaw summary: cos_min={:.6f} sig_cos_min={:.6f} mae_mean={:.6f} max_abs_max={:.6f}'.format(
        min(cos_list), min(sig_cos_list), float(np.nanmean(mae_list)), max(max_list)))

    ob, oc, os_ = post_process(onnx_outs)
    rb, rc, rs = post_process(rknn_outs)

    def _print_dets(tag, boxes, classes, scores):
        if boxes is None:
            print('{} detections (0):'.format(tag))
            return
        real = co_helper.get_real_box(boxes)
        print('{} detections ({}):'.format(tag, len(boxes)))
        for b, c, s in zip(real, classes, scores):
            cname = CLASSES[int(c)] if int(c) < len(CLASSES) else str(int(c))
            print('  {} score={:.4f} box=[{:.1f},{:.1f},{:.1f},{:.1f}]'.format(
                cname, float(s), b[0], b[1], b[2], b[3]))

    print()
    _print_dets('ONNX', ob, oc, os_)
    _print_dets('RKNN', rb, rc, rs)

    matched, ua, ub = match_detections(ob, oc, os_, rb, rc, rs, iou_thr=0.5)
    print('\nMatched dets (IoU>=0.5, same class): {}'.format(len(matched)))
    score_diffs = []
    for ia, ib, iou in matched:
        sd = abs(float(os_[ia]) - float(rs[ib]))
        score_diffs.append(sd)
        print('  onnx[{}] {} {:.3f} <-> rknn[{}] {} {:.3f} | IoU={:.3f} | |dscore|={:.4f}'.format(
            ia, CLASSES[int(oc[ia])], float(os_[ia]),
            ib, CLASSES[int(rc[ib])], float(rs[ib]),
            iou, sd))
    if ua:
        print('  ONNX-only:', sorted(ua))
    if ub:
        print('  RKNN-only:', sorted(ub))

    cos_min = float(np.nanmin(cos_list))
    sig_cos_min = float(np.nanmin(sig_cos_list))
    n_onnx = 0 if ob is None else len(ob)
    n_rknn = 0 if rb is None else len(rb)
    n_match = len(matched)
    det_ok = (n_onnx == n_rknn and n_match == n_onnx) or (
        n_match >= max(1, int(0.8 * max(n_onnx, n_rknn, 1))))

    print('\n--- Verdict ---')
    # 以有效区域 cosine + 检测框一致性为准；全图 cosine 对近零背景不可靠
    if sig_cos_min >= 0.99 and det_ok and n_onnx == n_rknn and n_match == n_onnx:
        print('PASS: highly consistent (FP conversion looks OK).')
        level = 'pass'
    elif sig_cos_min >= 0.95 and det_ok:
        print('WARN: mostly consistent (common for i8 quant). Check score/box drift.')
        level = 'warn'
    else:
        print('FAIL: large mismatch — check ONNX source / mean-std / quant / output order.')
        level = 'fail'

    if save_img:
        os.makedirs(out_dir, exist_ok=True)
        img_o = img_src.copy()
        img_r = img_src.copy()
        if ob is not None:
            draw(img_o, co_helper.get_real_box(ob), os_, oc)
        if rb is not None:
            draw(img_r, co_helper.get_real_box(rb), rs, rc)
        h = max(img_o.shape[0], img_r.shape[0])
        canvas = np.zeros((h, img_o.shape[1] + img_r.shape[1], 3), dtype=np.uint8)
        canvas[:img_o.shape[0], :img_o.shape[1]] = img_o
        canvas[:img_r.shape[0], img_o.shape[1]:] = img_r
        cv2.putText(canvas, 'ONNX', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(canvas, 'RKNN', (img_o.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        save_path = os.path.join(out_dir, 'cmp_' + os.path.splitext(img_name)[0] + '.jpg')
        cv2.imwrite(save_path, canvas)
        print('Saved compare image:', save_path)

    return {
        'img': img_name,
        'cos_min': cos_min,
        'sig_cos_min': sig_cos_min,
        'cos_mean': float(np.nanmean(cos_list)),
        'mae_mean': float(np.nanmean(mae_list)),
        'max_abs_max': float(np.nanmax(max_list)),
        'n_onnx': n_onnx,
        'n_rknn': n_rknn,
        'n_match': n_match,
        'score_mae': float(np.mean(score_diffs)) if score_diffs else None,
        'level': level,
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Compare ONNX vs RKNN for conversion check')
    parser.add_argument('--onnx', default=os.path.join(here, 'helmet_y11s_best.onnx'))
    parser.add_argument('--rknn', default=os.path.join(here, 'helmet_y11s_best_fp.rknn'))
    parser.add_argument('--img_folder', default='/userdata/jovan/code/rk3588/dataset/helmet/')
    parser.add_argument('--img_save', action='store_true')
    parser.add_argument('--out_dir', default=os.path.join(here, 'compare_result'))
    parser.add_argument('--max_images', type=int, default=0, help='0=all')
    args = parser.parse_args()

    if not os.path.isfile(args.onnx):
        raise FileNotFoundError('ONNX not found: {}'.format(args.onnx))
    if not os.path.isfile(args.rknn):
        raise FileNotFoundError('RKNN not found: {}'.format(args.rknn))

    print('Validate ONNX...')
    check_onnx_file(args.onnx)

    print('\nLoad ONNX:', args.onnx)
    onnx_model = ONNX_model_container(args.onnx)
    print('Load RKNN:', args.rknn)
    rknn_model = RKNN_model_container(args.rknn)

    img_list = sorted([p for p in os.listdir(args.img_folder) if img_check(p)])
    if args.max_images > 0:
        img_list = img_list[:args.max_images]
    if not img_list:
        raise RuntimeError('No images in {}'.format(args.img_folder))

    summaries = []
    for idx, name in enumerate(img_list):
        co_helper = COCO_test_helper(enable_letter_box=True)
        path = os.path.join(args.img_folder, name)
        img_src, onnx_in, rknn_in = prepare_image(path, co_helper)
        if img_src is None:
            print('Skip unreadable:', name)
            continue

        print('\nInfer {}/{} {}'.format(idx + 1, len(img_list), name))
        print('  onnx_in {} {}  rknn_in {} {}'.format(
            onnx_in.shape, onnx_in.dtype, rknn_in.shape, rknn_in.dtype))
        onnx_outs = onnx_model.run([onnx_in])
        rknn_outs = rknn_model.run([rknn_in])
        if rknn_outs is None:
            print('ERROR: RKNN inference failed')
            continue

        s = compare_one(name, img_src, onnx_outs, rknn_outs, co_helper, args.out_dir, args.img_save)
        summaries.append(s)

    print('\n' + '#' * 72)
    print('OVERALL')
    for s in summaries:
        print('  {img}: level={level} sig_cos_min={sig_cos_min:.6f} cos_min={cos_min:.6f} mae={mae_mean:.6f} '
              'dets onnx/rknn/match={n_onnx}/{n_rknn}/{n_match}'.format(**s))

    if summaries:
        worst = min(summaries, key=lambda x: x['sig_cos_min'])
        levels = {s['level'] for s in summaries}
        print('\nWorst sig_cos_min image: {} ({:.6f})'.format(worst['img'], worst['sig_cos_min']))
        print('How to read:')
        print('  PASS (sig_cos>=0.99, dets match) -> FP RKNN conversion OK')
        print('  WARN (sig_cos>=0.95)             -> normal for i8; FP should usually PASS')
        print('  FAIL                             -> wrong ONNX source / convert config / preprocess')
        if levels == {'pass'}:
            print('\nConclusion: RKNN matches ONNX well — conversion looks OK.')
        elif 'fail' in levels:
            print('\nConclusion: significant mismatch — do NOT trust this RKNN until fixed.')
        else:
            print('\nConclusion: acceptable drift (often i8). If this is FP RKNN and still WARN, dig deeper.')

    onnx_model.release()
    rknn_model.release()


if __name__ == '__main__':
    main()
