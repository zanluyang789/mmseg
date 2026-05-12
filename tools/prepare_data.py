"""
 /data/9/train/water/{image,label} 按 9:1 划分到 mmseg 标准目录
- image (.tif) 直接复制
- label (.tif) 转成 .png(mmseg 标准是单通道 png)
"""
import os
import random
import shutil
from pathlib import Path
import rasterio
import numpy as np
from PIL import Image

random.seed(42)

SRC_IMG = Path("/data/9/train/water/image")
SRC_LBL = Path("/data/9/train/water/label")
DST = Path("/data/9/water_seg/data")

# 1. 收集所有样本名(不带后缀)
all_names = sorted([p.stem for p in SRC_IMG.glob("*.tif")])
print(f"总样本数: {len(all_names)}")

# 2. 9:1 随机划分
random.shuffle(all_names)
n_val = max(1, len(all_names) // 10)
val_names = all_names[:n_val]
train_names = all_names[n_val:]
print(f"训练集: {len(train_names)}, 验证集: {len(val_names)}")

# 3. 处理函数
def process(names, split):
    img_dst = DST / "img_dir" / split
    lbl_dst = DST / "ann_dir" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    
    for i, name in enumerate(names):
        # 影像直接复制(.tif)
        src_img = SRC_IMG / f"{name}.tif"
        dst_img = img_dst / f"{name}.tif"
        if not dst_img.exists():
            shutil.copy(src_img, dst_img)
        
        # mask 从 .tif 转成 .png(单通道,值 0/1)
        src_lbl = SRC_LBL / f"{name}.tif"
        dst_lbl = lbl_dst / f"{name}.png"
        with rasterio.open(src_lbl) as src:
            arr = src.read(1).astype(np.uint8)
        Image.fromarray(arr).save(dst_lbl)
        
        if (i + 1) % 50 == 0:
            print(f"  [{split}] {i+1}/{len(names)}")
    print(f"完成 {split}: {len(names)} 张")

process(train_names, "train")
process(val_names, "val")
print("数据准备完成!")
