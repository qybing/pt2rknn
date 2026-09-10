#!/usr/bin/env python3
"""生成 RKNN 量化校准集清单文件。

扫描指定目录(含子目录)下的所有图像,把每张图的绝对路径
逐行写入 txt 文件,供 convert.py 的 DATASET_PATH 使用。

用法:
    python3 make_calib.py <图像目录> [输出txt路径] [--limit N] [--seed 42]

示例:
    # 全部图像写入清单
    python3 make_calib.py /root/code/rknn/dataset/helmet /root/code/rknn/helmet_calib.txt

    # 图太多时随机抽取 200 张(校准集 100~500 张即可,更多无收益)
    python3 make_calib.py /root/code/rknn/dataset/helmet /root/code/rknn/helmet_calib.txt --limit 200
"""
import argparse
import random
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def collect_images(img_dir: Path):
    paths = []
    for p in img_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stat().st_size > 0:
            paths.append(p.resolve())
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description='生成 RKNN 量化校准集清单')
    parser.add_argument('img_dir', help='图像目录(递归扫描子目录)')
    parser.add_argument('output', nargs='?', default=None,
                        help='输出txt路径,默认写到 <图像目录>/../helmet_calib.txt')
    parser.add_argument('--limit', type=int, default=0,
                        help='随机抽取的图片数量上限,0=全部保留')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机抽样种子,保证可复现')
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        raise SystemExit(f'错误: 目录不存在: {img_dir}')

    output = Path(args.output) if args.output else img_dir.parent / 'helmet_calib.txt'

    paths = collect_images(img_dir)
    total = len(paths)
    if total == 0:
        raise SystemExit(f'错误: 在 {img_dir} 下没找到图像,支持后缀: {sorted(IMG_EXTS)}')

    if args.limit and args.limit < total:
        random.seed(args.seed)
        paths = random.sample(paths, args.limit)

    with open(output, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(str(p) for p in paths) + '\n')

    stat = {}
    for p in paths:
        stat[p.suffix.lower()] = stat.get(p.suffix.lower(), 0) + 1
    print(f'共找到 {total} 张图像,写入 {len(paths)} 条到 {output}')
    print('后缀分布: ' + ', '.join(f'{k}: {v}' for k, v in sorted(stat.items())))
    if args.limit and args.limit < total:
        print(f'(已按 seed={args.seed} 随机抽取 {args.limit} 张,换图重抽或复现时保持同 seed)')
    print(f'下一步: 把 convert.py 第 4 行 DATASET_PATH 改为: {output}')


if __name__ == '__main__':
    main()
