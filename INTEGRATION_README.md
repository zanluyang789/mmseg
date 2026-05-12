# 系统集成总览（GPU/NPU 训练 + GPU/NPU 推理）

按"地灾项目-算法集成"文档实现的统一入口。**昨天的离线流程一行都没删**，只是在上面套了一层 task.conf 驱动的系统调度入口。

## 全景图

```
mmseg/
├── run.py                              # 系统训练入口  (python run.py train)
├── predict.py                          # 系统推理入口  (python predict.py infer --custom-config='task')
│
├── configs/
│   ├── task.conf                       # 系统下发的【训练】参数文件 (位置固定)
│   ├── deeplabv3plus_building.py       # mmseg base config (建筑物二分类)
│   └── deeplabv3plus_water.py          # 旧的水体 config (保留)
│
├── clie_lib/
│   └── configs/
│       └── task.conf                   # 系统下发的【推理】参数文件 (位置固定)
│
├── integration/                        # 系统集成层 (新增)
│   ├── conf_parser.py                  # task.conf 解析器 (带类型推断)
│   ├── device_utils.py                 # GPU/NPU 自动探测
│   ├── filelist_dataset.py             # 支持 train_img_list 文件列表的 dataset
│   ├── config_builder.py               # task.conf 参数注入 mmseg Config
│   ├── train_runner.py                 # 训练核心
│   └── infer_runner.py                 # 推理核心 (pth/onnx/om 三栈统一)
│
├── tools/                              # 昨天的离线工具 (全保留)
│   ├── prepare_data_building.py
│   ├── export_onnx_building.py
│   ├── verify_onnx_building.py
│   └── infer_om_building.py
│
├── deploy_building/                    # 昨天的 NPU 部署目录 (全保留)
│   ├── run_building.sh                 # env 驱动的旧入口
│   ├── infer_large_image.py            # 滑窗推理 (被 integration/infer_runner 复用)
│   ├── postprocess.py                  # 栅格 -> SHP
│   ├── send_kafka_msg.py               # Kafka 进度上报
│   └── Dockerfile                      # 昨天的推理镜像 (保留)
│
├── docker/                             # 系统集成镜像 (新增)
│   ├── Dockerfile.gpu                  # GPU 训练 + GPU 推理 一体镜像
│   ├── Dockerfile.npu                  # NPU 训练 + NPU 推理 一体镜像
│   ├── entrypoint.sh                   # 根据 train|infer 分发
│   └── export_image.sh                 # docker save -> tar.gz
│
├── BUILDING_RUNBOOK.md                 # 昨天的离线流程手册 (保留)
└── INTEGRATION_README.md               # 本文件
```

## 两条流程的关系

| 流程 | 入口 | 配置来源 | 适用场景 |
|---|---|---|---|
| 离线 (昨天) | `tools/train.py`、`tools/export_onnx_building.py`、`deploy_building/run_building.sh` | 命令行参数 / 环境变量 | 算法工程师本地调试 |
| 系统集成 (今天新增) | `run.py train`、`predict.py infer --custom-config='task'` | `configs/task.conf`、`clie_lib/configs/task.conf` | 平台调度 |

两套流程互不影响，互相不调用，**底层算法代码（mmseg config、滑窗推理函数、矢量化函数）100% 共享**。

## 训练（GPU / NPU 通用）

### 1. 系统填好 task.conf

平台会在容器启动前把参数写入 `configs/task.conf`，**位置固定，不要改**。常用字段（来自 PDF 表）：

```ini
env = 'building-train-3266'
work_dir = '/data/train/model/train-3266'
checkpoint_path = '/data/train/model/train-3266/pth'
log_path = '/data/train/model/train-3266/log'
tensorboard_log_path = '/data/train/tensorboard/train-3266'
use_tensorboard_scalar = True
use_tensorboard_image = True
pretrained = '/data/train/common_pth'
retrain_pth_url = ''                      # 再训练时给 .pth 绝对路径

use_filelist = True
train_img_list = '.../config/train_img_list.txt'
train_gt_list  = '.../config/train_gt_list.txt'
val_img_list   = '.../config/val_img_list.txt'
val_gt_list    = '.../config/val_gt_list.txt'
train_img_suffix = '.png'
train_gt_suffix  = '.png'
val_img_suffix   = '.png'
val_gt_suffix    = '.png'
mean_file = '.../config/mean_value.txt'
std_file  = '.../config/std_value.txt'
num_classes = 2
classes_name = ('background','building')
palette = [[0,0,0],[220,20,60]]

batch_size = 8
max_iters = 20000
val_interval = 1000
gpu_num = 1
```

> 文档里还可能下发其它字段，整套 PDF 通用表都已被 `integration/conf_parser.py` + `integration/config_builder.py` 处理，对不上的字段会被忽略。

### 2. 单卡

```bash
python run.py train
```

设备自动探测：装了 `torch_npu` 走 NPU，否则走 CUDA，再否则 CPU。也可强制指定：

```bash
INTEGRATION_DEVICE=npu  python run.py train
INTEGRATION_DEVICE=cuda python run.py train
```

### 3. 多卡（按文档原文）

文档给的模板：

```bash
python --nnodes=${nnodes} --node_rank=${node_rank} \
       --nproc_per_node=${nproc_per_node} --master_addr=${master_addr} \
       --master_port=${master_port} run.py train
```

平台会替换变量。实际等价于：

```bash
python -m torch.distributed.run \
       --nnodes=1 --node_rank=0 --nproc_per_node=4 \
       --master_addr=127.0.0.1 --master_port=29500 \
       run.py train
```

`run.py` 探测到 `RANK` 环境变量后会把 `cfg.launcher` 切成 `'pytorch'`，mmengine Runner 自己处理后续分布式细节。NPU 上 dist backend 自动切到 `hccl`，norm_cfg 自动从 `SyncBN` 退化到 `BN`。

## 推理（GPU pth / NPU om / ONNX 三栈）

### 1. 系统填好 clie_lib/configs/task.conf

```ini
env = 'building-infer-3266'
load_model_path = '/data/train/model/train-3266/pth/best_mIoU_iter_18000.pth'
# 或:
# load_model_path = '/data/.../building_seg.onnx'
# load_model_path = '/data/.../building_seg.om'

img_list_file = '/data/.../infer/img_list.txt'
output_root   = '/share/.../output/predict/building_3266'
band_list_file   = '/data/.../config/band_list.txt'    # 文件里写 "3,2,1"
color_table_file = '/data/.../config/color_table.txt'
mean_file = '...'
std_file  = '...'

bootstrap_servers = '192.168.1.166:12007'
topic = 'ib-theme.algorithm_callback_topic'
projectId = 288
taskId = 5224

use_color_out = True
use_shapefile_out = True

# 可选
num_classes = 2
classes_name = ('background','building')
palette = [[0,0,0],[220,20,60]]
threshold = 0.5
tile = 512
stride = 384
min_area_px = 25
save_prob = False
```

### 2. 启动

```bash
python predict.py infer --custom-config='task'
```

`--custom-config='task'` 是文档约定，对应 `clie_lib/configs/task.conf`。也可以传绝对路径：

```bash
python predict.py infer --custom-config=/path/to/whatever.conf
```

### 3. 模型后缀决定后端

| 后缀 | runner | 设备 | 说明 |
|---|---|---|---|
| `.pth` | `PthRunner` (mmseg.apis) | cuda / npu | GPU 推理（直接吃训练产物） |
| `.onnx` | `ONNXRunner` (onnxruntime) | cpu / cuda | 跨平台调试 |
| `.om` | `OMRunner` (pyACL) | npu | 昇腾 NPU 推理（昨天部署用的） |

切换全自动，**不需要改任何代码**。系统下发哪种 `load_model_path` 就跑哪种栈。

## Docker 镜像

文档要求："将代码运行环境打包成一个 docker 镜像"。提供两个：

### GPU 镜像

```bash
cd C:\Users\zanly\Desktop\mmseg
docker build -f docker/Dockerfile.gpu -t building-seg-gpu:v1 .

# 训练
docker run --rm --gpus all \
    -v /data/train:/data/train \
    -v $PWD/configs/task.conf:/workspace/mmseg/configs/task.conf:ro \
    building-seg-gpu:v1 train

# 推理
docker run --rm --gpus all \
    -v /data/predict:/data/predict \
    -v /share:/share \
    -v $PWD/clie_lib/configs/task.conf:/workspace/mmseg/clie_lib/configs/task.conf:ro \
    building-seg-gpu:v1 infer
```

### NPU 镜像

```bash
docker build -f docker/Dockerfile.npu -t building-seg-npu:v1 .

# 训练
docker run --rm \
    --device=/dev/davinci0 --device=/dev/davinci_manager \
    --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /data/train:/data/train \
    -v $PWD/configs/task.conf:/workspace/mmseg/configs/task.conf:ro \
    building-seg-npu:v1 train

# 推理（同 GPU 镜像，把 task.conf 里的 load_model_path 换成 .om 即可）
docker run --rm \
    --device=/dev/davinci0 ... \
    -v $PWD/clie_lib/configs/task.conf:/workspace/mmseg/clie_lib/configs/task.conf:ro \
    building-seg-npu:v1 infer
```

### 导出 docker save 文件

文档要求"最好是 docker save 后的文件"：

```bash
bash docker/export_image.sh gpu     # -> dist/building-seg-gpu-v1.tar.gz
bash docker/export_image.sh npu     # -> dist/building-seg-npu-v1.tar.gz
```

## 离线流程入口（昨天那套，全保留）

* 数据准备：`python tools/prepare_data_building.py --src ... --dst ...`
* 训练：`python tools/train.py configs/deeplabv3plus_building.py`
* 导 ONNX：`python tools/export_onnx_building.py --ckpt ... --out building_seg.onnx`
* 验证一致性：`python tools/verify_onnx_building.py ...`
* ATC 转 OM：`atc --model=building_seg.onnx --output=building_seg ...`
* NPU 单图推理：`python tools/infer_om_building.py --om ... --img ...`
* NPU 大图部署：`cd deploy_building && bash run_building.sh`（env 驱动）

完整步骤还是看 [BUILDING_RUNBOOK.md](./BUILDING_RUNBOOK.md)。

## 设备 / 后端兼容矩阵

|  | GPU (CUDA) | NPU (Ascend) |
|---|---|---|
| 训练 | `INTEGRATION_DEVICE=cuda python run.py train`（默认） | `INTEGRATION_DEVICE=npu python run.py train` |
| 训练 norm_cfg | SyncBN | 自动降级 BN |
| 训练 dist backend | nccl | hccl |
| 推理 - pth | ✓ | ✓（torch_npu） |
| 推理 - onnx | ✓ | x（ORT 没 NPU EP，建议 om） |
| 推理 - om | x | ✓（pyACL） |

## 调试小贴士

* `python -m integration.conf_parser configs/task.conf` 单独看 task.conf 解析效果
* `python -m integration.device_utils` 看当前设备探测结果
* `python -m integration.config_builder configs/deeplabv3plus_building.py configs/task.conf` 打印注入后的完整 Config，方便排查
* CPU 模式跑通最快：`INTEGRATION_DEVICE=cpu python run.py train`（用一两个样本 + max_iters=10 测路径有没有错）

## 改动文件清单（昨天 vs 今天）

新增：

* `run.py` `predict.py`
* `integration/`（6 个 .py）
* `docker/Dockerfile.gpu` `docker/Dockerfile.npu` `docker/entrypoint.sh` `docker/export_image.sh`
* `INTEGRATION_README.md`

修改（**0 个**）—— 昨天的脚本和 config 完全没动。

保留：

* `configs/task.conf` `clie_lib/configs/task.conf`（只是位置和命名按文档对齐，内容由平台覆盖）
* `tools/*` `deploy_building/*` `configs/*.py`
* `BUILDING_RUNBOOK.md` `deploy_building/README.md`
