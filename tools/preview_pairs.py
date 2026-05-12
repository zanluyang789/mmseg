"""
预览 mmseg img_dir / ann_dir 数据,把 原图 | mask | 红色叠加 横向拼,
多个样本垂直堆成一张大 JPG 便于人眼检查。

用法:
    python tools/preview_pairs.py \
        --root /data/azanly1/9/mmseg/data/building \
        --split train \
        --num 8 \
        --out /data/azanly1/9/mmseg/preview_building_train.jpg
"""
import argparse
import random
import sys
from pathlib import Path
import numpy as np
import cv2


def render_row(img_bgr, mask_uint8):
    h, w = img_bgr.shape[:2]
    if mask_uint8.max() <= 1:
        mask_vis = cv2.cvtColor(mask_uint8 * 255, cv2.COLOR_GRAY2BGR)
    else:
        mask_vis = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(img_bgr); red[:, :, 2] = 255
    alpha = 0.45
    bin_mask = (mask_uint8 > 0)
    mask_3c = np.stack([bin_mask] * 3, axis=-1)
    overlay = np.where(mask_3c,
                       (img_bgr * (1 - alpha) + red * alpha).astype(np.uint8),
                       img_bgr)
    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    return np.concatenate([img_bgr, sep, mask_vis, sep, overlay], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--split', default='train', choices=['train', 'val'])
    ap.add_argument('--num', type=int, default=8)
    ap.add_argument('--out', default='preview.jpg')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-h', type=int, default=512)
    args = ap.parse_args()

    root = Path(args.root)
    img_dir = root / 'img_dir' / args.split
    ann_dir = root / 'ann_dir' / args.split
    if not img_dir.is_dir():
        print(f'[error] 找不到 {img_dir}', file=sys.stderr); sys.exit(2)
    if not ann_dir.is_dir():
        print(f'[error] 找不到 {ann_dir}', file=sys.stderr); sys.exit(2)

    all_imgs = sorted(img_dir.glob('*.png'))
    if not all_imgs:
        all_imgs = sorted(img_dir.glob('*.tif'))
    suffix = all_imgs[0].suffix if all_imgs else '?'
    print(f'[info] {args.split} 集共 {len(all_imgs)} 张 ({suffix})', flush=True)
    if not all_imgs:
        print('[error] 这个 split 下没有 png/tif', file=sys.stderr); sys.exit(3)

    random.seed(args.seed)
    samples = random.sample(all_imgs, min(args.num, len(all_imgs)))

    rows = []
    title_h = 26
    for img_p in samples:
        mask_p = ann_dir / (img_p.stem + '.png')
        if not mask_p.exists():
            print(f'  [skip] {img_p.name} 没有对应 mask', flush=True); continue
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        m = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)
        if img is None or m is None:
            print(f'  [skip] {img_p.name} 读不了', flush=True); continue
        if m.ndim == 3:
            m = m[..., 0]
        if img.shape[0] > args.max_h:
            scale = args.max_h / img.shape[0]
            new_size = (int(img.shape[1] * scale), args.max_h)
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            m = cv2.resize(m, new_size, interpolation=cv2.INTER_NEAREST)
        row = render_row(img, m)
        pct = 100.0 * (m > 0).sum() / m.size
        title = f'{img_p.stem}    building: {pct:.2f}%   mask_max={int(m.max())}'
        canvas = np.full((row.shape[0] + title_h, row.shape[1], 3), 255, dtype=np.uint8)
        canvas[title_h:] = row
        cv2.putText(canvas, title, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(canvas)
        print(f'  [{len(rows)}/{args.num}] {img_p.stem}  building={pct:.2f}%', flush=True)

    if not rows:
        print('[error] 没生成任何行', file=sys.stderr); sys.exit(4)

    max_w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    sep = np.full((6, max_w, 3), 200, dtype=np.uint8)
    pieces = []
    for i, r in enumerate(rows):
        if i > 0: pieces.append(sep)
        pieces.append(r)
    big = np.concatenate(pieces, axis=0)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_p), big, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        print(f'[error] cv2.imwrite 写失败: {out_p}', file=sys.stderr); sys.exit(5)
    print(f'[ok] saved {out_p}  shape={big.shape}', flush=True)


if __name__ == '__main__':
    main()
