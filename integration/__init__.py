"""
mmseg 系统集成层
================

把"昨天的离线流程（直接跑 tools/train.py + tools/export_onnx_building.py + 部署目录）"
和"系统调度（run.py train / predict.py infer --custom-config='task'）"统一在这里。

核心约定：
- 训练参数文件: configs/task.conf
- 推理参数文件: clie_lib/configs/task.conf
- 任何参数都通过 task.conf 注入，不再传命令行（除了文档规定的多卡 ${nnodes} 等）
- GPU / NPU 由 device_utils.detect_device() 自动探测（torch_npu 可用则 NPU 优先）

子模块：
    conf_parser      解析 task.conf 的 key=value 行（带注释、类型推断）
    device_utils     探测 GPU/NPU、给出 dist_backend / norm_cfg
    config_builder   把 task.conf 的内容注入 mmseg Config（覆盖 dataset/类别等）
    filelist_dataset 自定义 dataset：从 train_img_list 这种"每行一个绝对路径"文件读样本
    train_runner     训练核心，给 run.py 调
    infer_runner     推理核心（GPU pth / NPU om / ONNX 三栈共用），给 predict.py 调
"""

from .conf_parser import load_task_conf, get_task_config
from .device_utils import detect_device, get_dist_backend, get_norm_cfg

__all__ = [
    "load_task_conf",
    "get_task_config",
    "detect_device",
    "get_dist_backend",
    "get_norm_cfg",
]
