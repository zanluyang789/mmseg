# 建筑物分割模型 训练 → 部署 操作手册

整条流水线复用水体的方案，只换了类别和数据路径。所有新增脚本 / 配置以 `_building` 后缀区分，**没有改动任何水体相关文件**。

## 0. 新增的文件清单

| 文件 | 作用 |
|---|---|
| `configs/deeplabv3plus_building.py` | mmseg 训练配置（DeepLabV3+ / ResNet50 / 512×512 / 类别加权 + Dice） |
| `tools/prepare_data_building.py` | 9:1 切分 train/val，自动把 mask 归一到 0/1 |
| `tools/export_onnx_building.py` | 导出静态 512×512 ONNX（含归一化 + softmax + 取 building 通道） |
| `tools/verify_onnx_building.py` | 对比 PyTorch vs ONNX 输出一致性 |
| `tools/infer_om_building.py` | 昇腾 NPU 上做 ACL 推理 + 三联图可视化 |
| `BUILDING_RUNBOOK.md` | 本文件 |

## 1. 数据准备（GPU 服务器 / Linux 端）

```bash
# 进 mmseg 工作目录
cd /your/path/to/mmseg

# 把盘阵上的原始 images+masks 切成 train/val
# --src 替换成 盘阵在 Linux 这台机上的实际挂载路径
python tools/prepare_data_building.py \
    --src /your/mount/shandong/dataset/building \
    --dst /data/9/building_seg/data \
    --val-ratio 0.1
```

跑完会得到：

```
/data/9/building_seg/data/
├── img_dir/
│   ├── train/   *.png
│   └── val/     *.png
└── ann_dir/
    ├── train/   *.png   (mode=L, 值 0/1)
    └── val/     *.png
```

脚本末尾会自动打 sanity check（样本尺寸、mask 值域），看一眼确认没问题。

## 2. GPU 训练

单卡：

```bash
python tools/train.py configs/deeplabv3plus_building.py
```

多卡（举例 4 卡）：

```bash
./tools/dist_train.sh configs/deeplabv3plus_building.py 4
```

训练产物在 `work_dirs/deeplabv3plus_building/`，根据 `best_mIoU` 自动留 top-3 ckpt。

关键超参在 config 里可调（看注释）：
- `class_weight = [0.5, 1.5]`：建筑物正负样本不均衡时调这个
- `train_pipeline` 里加了 `RandomCrop(512, cat_max_ratio=0.85)`，避免随机裁出全背景
- `max_iters=20000`，`val_interval=1000`

## 3. 导出 ONNX（GPU 服务器）

```bash
python tools/export_onnx_building.py \
    --config configs/deeplabv3plus_building.py \
    --ckpt   work_dirs/deeplabv3plus_building/best_mIoU_iter_XXXXX.pth \
    --out    building_seg.onnx \
    --opset  11
```

输入 `(1,3,512,512) float32 [0,255]`，输出 `(1,1,512,512) float32` 概率图（已经做完归一化和 softmax，部署端直接拿概率图阈值化即可）。

## 4. 验证 ONNX 与 PyTorch 输出一致

```bash
python tools/verify_onnx_building.py \
    --config configs/deeplabv3plus_building.py \
    --ckpt   work_dirs/deeplabv3plus_building/best_mIoU_iter_XXXXX.pth \
    --onnx   building_seg.onnx \
    --img    /data/9/building_seg/data/img_dir/val/<某张>.png
```

期望：`max abs diff < 1e-3`、`binary mask IoU > 0.99`。否则停下来排查（一般是 norm_cfg 没切到 BN）。

## 5. ATC 转 OM（在装了 CANN 的机器上跑）

```bash
atc \
    --model=building_seg.onnx \
    --framework=5 \
    --output=building_seg \
    --input_format=NCHW \
    --input_shape="input:1,3,512,512" \
    --soc_version=Ascend910B \
    --log=error
```

跑完会生成 `building_seg.om`。`--soc_version` 按你机器实际型号填（910B / 310P / 310B 等）。

## 6. NPU 推理（910B 板）

```bash
python tools/infer_om_building.py \
    --om  building_seg.om \
    --img <任意一张测试图.png> \
    --out result_building.png \
    --threshold 0.5
```

会打印推理耗时、建筑物像素占比，然后保存一张三联图（原图 / mask / 红色半透明叠加）。

## 常见坑

- **导出 ONNX 时 `data_preprocessor` 必须设成 None**，因为归一化已经塞进 `BuildingSegONNX` 的 forward 里了。脚本里已经处理。
- **SyncBN 不能直接导出**，脚本会把 `norm_cfg` 改成普通 `BN`。
- **ATC 转换要静态 shape**，所以 export 时 `dynamic_axes=None`，输入固定 `1,3,512,512`。
- **推理大图**：原图如果是 1024×1024，自己切 4 块送进去，或者先 resize 到 512×512（精度会掉一点）。后续可以再写一个 sliding window 推理脚本。
- **mask 值如果是 0/255**：`prepare_data_building.py` 默认会自动归一到 0/1，加 `--no-normalize` 可关闭。
