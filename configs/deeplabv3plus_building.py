# DeepLabV3+ 建筑物分割,二分类,512x512 部署,Ascend 910B
# 沿用水体那套(deeplabv3plus_water.py),只改:
#   1. 类别名 / 调色板:'building' + 红色
#   2. data_root 指向 /data/9/building_seg/data
#   3. img_suffix 从 .tif 改成 .png(建筑物数据集已经是 PNG)
#   4. class_weight 微调:建筑物正样本比例通常 10~20%,比水体的 2% 高,
#      所以背景权重适当提高,正类权重略降。可按实际正负比调整。

# ============ 数据集 ============
dataset_type = 'BaseSegDataset'
data_root = '/data/azanly1/9/mmseg/data/building/'

metainfo = dict(
    classes=('background', 'building'),
    palette=[[0, 0, 0], [220, 20, 60]]   # 背景黑,建筑物深红
)

crop_size = (512, 512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

# 训练增强 pipeline:
#   原图是 1024x1024,这里先随机 crop 到 512x512,再做翻转 + 色彩抖动。
#   crop 是关键 —— 既能给模型多视角,也保证 GPU 训练时显存可控。
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.85),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# 验证 pipeline:全图缩放到 512x512(与 ONNX 部署一致)
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=metainfo,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=metainfo,
        pipeline=val_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator

# ============ 模型 ============
norm_cfg = dict(type='SyncBN', requires_grad=True)

# 建筑物类别加权:典型场景建筑物像素占比 10~20%
# 比例如果偏差大,可调。先给一个相对温和的不平衡补偿:
class_weight = [0.5, 1.5]

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        # 类别加权 CE + Dice 联合 loss,提升小建筑物边界召回
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False,
                 loss_weight=1.0, class_weight=class_weight),
            dict(type='DiceLoss', loss_weight=1.0)
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.4, class_weight=class_weight)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ============ 训练策略 ============
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=200),
    dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=200, end=20000, by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============ Hook 配置 ============
default_scope = 'mmseg'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000,
                    save_best='mIoU', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

randomness = dict(seed=42)
