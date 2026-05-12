"""
把 <SRC>/{images,masks}/*.png 按 9:1 划分到 mmseg 标准目录
- images (.png) 直接复制
- masks  (.png) 直接复制；如果发现像素是 0/255 而不是 0/1，自动归一到 0/1

用法（在 GPU 服务器上跑）:
    python tools/prepare_data_building.py \
        --src /data/9/shandong/dataset/building \
        --dst /data/9/building_seg/data \
        --val-ratio 0.1

Windows 上测试也行:
    python tools/prepare_data_building.py \
        --src Z:\\shandong\\dataset\\建筑物 \
        --dst Z:\\building_seg\\data
"""
import argparse
import random
import shutil
from pathlib import Path
import numpy as np
from PIL import Image


def collect_pairs(src_img: Path, src_lbl: Path):
    """收集 image 和 mask 都存在的样本名（不带后缀）"""
    img_stems = {p.stem for p in src_img.glob("*.png")}
    lbl_stems = {p.stem for p in src_lbl.glob("*.png")}
    common = sorted(img_stems & lbl_stems)
    only_img = img_stems - lbl_stems
    only_lbl = lbl_stems - img_stems
    if only_img:
        print(f"[warn] {len(only_img)} 张图没有对应 mask，举例: {list(sorted(only_img))[:3]}")
    if only_lbl:
        print(f"[warn] {len(only_lbl)} 个 mask 没有对应图，举例: {list(sorted(only_lbl))[:3]}")
    return common


def split_train_val(names, val_ratio, seed):
    rng = random.Random(seed)
    names = list(names)
    rng.shuffle(names)
    n_val = max(1, int(round(len(names) * val_ratio)))
    return names[n_val:], names[:n_val]


def write_split(names, split, src_img: Path, src_lbl: Path, dst_root: Path,
                normalize_mask: bool):
    img_dst = dst_root / "img_dir" / split
    lbl_dst = dst_root / "ann_dir" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(names):
        # 影像：原样复制
        s_img = src_img / f"{name}.png"
        d_img = img_dst / f"{name}.png"
        if not d_img.exists():
            shutil.copy(s_img, d_img)

        # mask：要保证是 mode=L、值 0/1
        s_lbl = src_lbl / f"{name}.png"
        d_lbl = lbl_dst / f"{name}.png"
        m = Image.open(s_lbl)
        arr = np.array(m)
        if arr.ndim == 3:
            arr = arr[..., 0]                     # 万一是多通道,只取第一通道
        max_v = int(arr.max()) if arr.size else 0
        if normalize_mask and max_v > 1:
            # 0/255 -> 0/1
            arr = (arr > 0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        Image.fromarray(arr, mode="L").save(d_lbl)

        if (i + 1) % 100 == 0:
            print(f"  [{split}] {i+1}/{len(names)}")
    print(f"完成 {split}: {len(names)} 张")


def sanity_check(dst_root: Path):
    """打印一份统计：train/val 数量、随机样本尺寸、mask 值域"""
    for split in ("train", "val"):
        img_dir = dst_root / "img_dir" / split
        ann_dir = dst_root / "ann_dir" / split
        imgs = sorted(img_dir.glob("*.png"))
        anns = sorted(ann_dir.glob("*.png"))
        print(f"[stat] {split}: {len(imgs)} imgs, {len(anns)} masks")
        if imgs:
            s = Image.open(imgs[0])
            print(f"       sample img: {imgs[0].name}, size={s.size}, mode={s.mode}")
        if anns:
            a = np.array(Image.open(anns[0]))
            print(f"       sample ann: {anns[0].name}, shape={a.shape}, "
                  f"dtype={a.dtype}, unique={np.unique(a).tolist()[:5]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="原始数据根目录，里面要有 images/ 和 masks/ 两个子目录")
    ap.add_argument("--dst", required=True,
                    help="输出根目录，会生成 img_dir/{train,val} 和 ann_dir/{train,val}")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-normalize", action="store_true",
                    help="不要把 0/255 归一到 0/1（默认会归一）")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    src_img = src / "images"
    src_lbl = src / "masks"
    assert src_img.is_dir(), f"找不到 {src_img}"
    assert src_lbl.is_dir(), f"找不到 {src_lbl}"

    print(f"src = {src}")
    print(f"dst = {dst}")
    names = collect_pairs(src_img, src_lbl)
    print(f"配对成功样本数: {len(names)}")
    assert len(names) > 0, "没有任何配对样本，请检查目录结构"

    train_names, val_names = split_train_val(names, args.val_ratio, args.seed)
    print(f"train: {len(train_names)}, val: {len(val_names)}")

    write_split(train_names, "train", src_img, src_lbl, dst,
                normalize_mask=not args.no_normalize)
    write_split(val_names,   "val",   src_img, src_lbl, dst,
                normalize_mask=not args.no_normalize)

    print("---")
    sanity_check(dst)
    print("数据准备完成!")


if __name__ == "__main__":
    main()
