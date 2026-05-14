"""
task.conf -> mmseg Config 注入
==============================

输入：base config 路径 + task.conf 解析出来的 dict
输出：注入完所有参数的 mmengine.config.Config 对象，可直接喂给 Runner

注入的字段（来自 PDF 表）：
    通用：
        work_dir          -> cfg.work_dir
        checkpoint_path   -> cfg.default_hooks.checkpoint.out_dir（如果给）
        log_path          -> cfg.log_dir（mmengine 的 work_dir 子目录会自动拼）
        tensorboard_log_path 启动 TB hook，dir 指向这里
        pretrained        -> 仅作为查找 backbone 预训练 pth 的根目录提示
        retrain_pth_url   -> cfg.load_from（再训练）
        use_tensorboard_scalar / use_tensorboard_image -> 配置 TB hook
    语义分割：
        use_filelist=True 时，把 train/val dataloader 切到 FileListSegDataset
        num_classes / classes_name / palette -> metainfo + decode_head + auxiliary_head
        mean_file / std_file -> data_preprocessor.mean / .std（如果给）
        train_img_suffix / train_gt_suffix / val_img_suffix / val_gt_suffix
            （use_filelist=False 时控制 BaseSegDataset 的后缀）
    设备：
        norm_cfg -> SyncBN/BN
        dist_cfg.backend -> nccl / hccl
    其它：
        batch_size / max_iters / val_interval / crop_size 这几个虽然 PDF 没列，
            但项目原 task.conf 里有，按需注入。
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional

from mmengine.config import Config

from .device_utils import detect_device, get_dist_backend, get_norm_cfg


def _read_floats_file(path: str, fallback):
    """读 mean/std 文件，每行一个 float（或一行逗号分隔），返回 list[float]"""
    if not path or not os.path.exists(path):
        return fallback
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for x in line.replace(",", " ").split():
                try:
                    vals.append(float(x))
                except ValueError:
                    pass
    if not vals:
        return fallback
    return vals


def _ensure_tuple(v):
    if isinstance(v, str):
        # "('a','b')" 已经在 conf_parser 那一关被 literal_eval；
        # 万一这里拿到字符串，按逗号切
        return tuple(s.strip().strip("'\"") for s in v.strip("()[] ").split(","))
    if isinstance(v, (list, tuple)):
        return tuple(v)
    return (str(v),)


def _find_local_pth(directory: str, key: str) -> Optional[str]:
    """
    在 directory 及其子目录（最多深 2 层）里找文件名包含 key 的 .pth。
    例如 key='resnet50_v1c' -> 命中 'resnet50_v1c-2cccc1ad.pth'。
    """
    if not os.path.isdir(directory):
        return None
    for root, dirs, files in os.walk(directory):
        rel = os.path.relpath(root, directory)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 2:
            dirs[:] = []
            continue
        for f in files:
            if f.endswith(".pth") and key in f:
                return os.path.join(root, f)
    return None


def _resolve_pretrained(cfg: Config, pretrained_dir: str) -> None:
    """
    把 cfg.model.pretrained 从 'open-mmlab://xxx' / URL 替换成 pretrained_dir
    下的本地 .pth；找不到时设 TORCH_HOME=pretrained_dir，让 torch.hub 后续
    下载也落到这个目录（路径会是 pretrained_dir/hub/checkpoints/xxx.pth）。
    """
    if not pretrained_dir or not os.path.isdir(pretrained_dir):
        return

    current = cfg.model.get("pretrained", None)
    if current and (os.path.isabs(current) and os.path.exists(current)):
        return  # 已经是本地有效路径

    # 从 'open-mmlab://resnet50_v1c' / URL 中提取关键词
    key = None
    if current:
        # e.g. 'open-mmlab://resnet50_v1c' -> 'resnet50_v1c'
        #      'https://.../resnet50_v1c-2cccc1ad.pth' -> 'resnet50_v1c-2cccc1ad.pth'
        last = current.split("//")[-1].split("/")[-1].split(":")[-1]
        key = last.replace(".pth", "")

    local = _find_local_pth(pretrained_dir, key) if key else None
    if local:
        cfg.model.pretrained = local
        print(f"[pretrained] {current} -> {local}", flush=True)
    else:
        # 让 torch.hub 后续下载落到 pretrained_dir/hub/checkpoints/
        os.environ["TORCH_HOME"] = pretrained_dir
        print(
            f"[pretrained] 本地没找到 '{key}' 相关 pth，"
            f"已设 TORCH_HOME={pretrained_dir}，"
            f"torch.hub 会下载到 {pretrained_dir}/hub/checkpoints/",
            flush=True,
        )


def _patch_segmentor_classes(cfg: Config, num_classes: int):
    """两处 head 都要改 num_classes"""
    if "decode_head" in cfg.model:
        cfg.model.decode_head.num_classes = num_classes
        # 类别加权 loss 维度对不上会炸，简单兜底
        for loss in cfg.model.decode_head.get("loss_decode", []) or []:
            cw = loss.get("class_weight")
            if cw and len(cw) != num_classes:
                # 数量不对就丢掉旧权重，让 loss 走默认等权
                loss.pop("class_weight", None)
    if "auxiliary_head" in cfg.model and cfg.model.auxiliary_head is not None:
        cfg.model.auxiliary_head.num_classes = num_classes
        for loss_aux in [cfg.model.auxiliary_head.get("loss_decode", {})]:
            if isinstance(loss_aux, dict):
                cw = loss_aux.get("class_weight")
                if cw and len(cw) != num_classes:
                    loss_aux.pop("class_weight", None)


def _switch_to_filelist_dataset(
    loader_cfg: Dict, img_list: str, gt_list: Optional[str], metainfo: Dict, pipeline=None
):
    """把现成的 dataloader.dataset 改成 FileListSegDataset"""
    new_ds = dict(
        type="FileListSegDataset",
        img_list_file=img_list,
        gt_list_file=gt_list,
        metainfo=metainfo,
        pipeline=pipeline if pipeline is not None else loader_cfg["dataset"].get("pipeline"),
    )
    # reduce_zero_label / ignore_index 沿用原 dataset 的设定
    for k in ("reduce_zero_label", "ignore_index"):
        if k in loader_cfg["dataset"]:
            new_ds[k] = loader_cfg["dataset"][k]
    loader_cfg["dataset"] = new_ds


def build_train_cfg(
    base_cfg_path: str,
    task_cfg: Dict[str, Any],
    device: Optional[str] = None,
) -> Config:
    """
    主入口：base config + task.conf -> 注入后的 Config
    """
    if not os.path.exists(base_cfg_path):
        raise FileNotFoundError(f"base config 不存在: {base_cfg_path}")

    cfg: Config = Config.fromfile(base_cfg_path)
    device = device or detect_device()

    # -------- 输出 / 工作目录 --------
    # mmengine 的 work_dir 决定了 timestamp 子目录、默认 log 文件、默认 vis_data
    # 全部位置。按 PDF 表语义：log_path 是日志根目录，所以让 work_dir = log_path，
    # 这样 mmengine 自动生成的所有"训练痕迹"都在 log/ 下，符合预期。
    log_path = task_cfg.get("log_path")
    work_dir_cfg = task_cfg.get("work_dir") or task_cfg.get("work_space")
    if log_path:
        cfg.work_dir = log_path
    elif work_dir_cfg:
        cfg.work_dir = work_dir_cfg
    os.makedirs(cfg.work_dir, exist_ok=True)

    # ckpt 不再通过 CheckpointHook 的 out_dir 改路径（mmengine 会把它再叠
    # 一层 basename(work_dir) 子目录，结果路径不干净）。改用 CkptSyncHook
    # 在 work_dir 下生成 .pth 后实时同步到 checkpoint_path。
    ckpt_dir = task_cfg.get("checkpoint_path")
    if ckpt_dir:
        # 触发自定义 hook 注册
        from . import hooks  # noqa: F401

        os.makedirs(ckpt_dir, exist_ok=True)
        custom_hooks = list(cfg.get("custom_hooks", []) or [])
        custom_hooks.append(dict(type="CkptSyncHook", ckpt_dir=ckpt_dir, link=False))
        cfg.custom_hooks = custom_hooks

    # -------- 再训练 --------
    retrain = task_cfg.get("retrain_pth_url")
    if retrain:
        cfg.load_from = retrain
        cfg.resume = bool(task_cfg.get("resume_mode") and task_cfg["resume_mode"] != "None")

    # -------- 预训练 backbone 权重 --------
    pretrained_dir = task_cfg.get("pretrained")
    if pretrained_dir:
        _resolve_pretrained(cfg, pretrained_dir)

    # -------- 类别 / palette --------
    num_classes = task_cfg.get("num_classes")
    classes_name = task_cfg.get("classes_name")
    palette = task_cfg.get("palette")
    if num_classes is not None:
        _patch_segmentor_classes(cfg, int(num_classes))
    if classes_name and palette:
        classes_name = _ensure_tuple(classes_name)
        metainfo = dict(classes=classes_name, palette=list(palette))
        # 同步到 train / val / test dataloader
        for loader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
            loader = cfg.get(loader_name)
            if loader and "dataset" in loader:
                loader["dataset"]["metainfo"] = metainfo
    else:
        # 没传 classes_name/palette 时退而求其次：保留 base config 自带的 metainfo
        metainfo = None
        for loader_name in ("train_dataloader",):
            loader = cfg.get(loader_name)
            if loader and "dataset" in loader and "metainfo" in loader["dataset"]:
                metainfo = loader["dataset"]["metainfo"]
                break

    # -------- mean / std --------
    mean = _read_floats_file(task_cfg.get("mean_file", ""), None)
    std = _read_floats_file(task_cfg.get("std_file", ""), None)
    if mean and std and "data_preprocessor" in cfg.model:
        cfg.model.data_preprocessor.mean = mean
        cfg.model.data_preprocessor.std = std

    # -------- 数据集：文件列表 模式 --------
    if task_cfg.get("use_filelist"):
        # 触发自定义 dataset 注册
        from . import filelist_dataset  # noqa: F401

        # 用 base config 自带的 metainfo 兜底
        if metainfo is None:
            for loader_name in ("train_dataloader",):
                loader = cfg.get(loader_name)
                if loader and "dataset" in loader and "metainfo" in loader["dataset"]:
                    metainfo = loader["dataset"]["metainfo"]
                    break

        train_img_list = task_cfg.get("train_img_list")
        train_gt_list = task_cfg.get("train_gt_list")
        val_img_list = task_cfg.get("val_img_list")
        val_gt_list = task_cfg.get("val_gt_list")

        if train_img_list and cfg.get("train_dataloader"):
            _switch_to_filelist_dataset(
                cfg.train_dataloader, train_img_list, train_gt_list, metainfo
            )
        if val_img_list and cfg.get("val_dataloader"):
            _switch_to_filelist_dataset(
                cfg.val_dataloader, val_img_list, val_gt_list, metainfo
            )
            # test 跟 val 一致（mmseg 习惯）
            if cfg.get("test_dataloader"):
                cfg.test_dataloader = deepcopy(cfg.val_dataloader)

    # -------- 批大小 / 训练长度 --------
    batch_size = task_cfg.get("batch_size")
    if batch_size and cfg.get("train_dataloader"):
        cfg.train_dataloader.batch_size = int(batch_size)

    max_iters = task_cfg.get("max_iters")
    val_interval = task_cfg.get("val_interval")
    if max_iters and cfg.get("train_cfg"):
        cfg.train_cfg.max_iters = int(max_iters)
    if val_interval and cfg.get("train_cfg"):
        cfg.train_cfg.val_interval = int(val_interval)

    # -------- 设备 / 分布式 --------
    cfg.model.setdefault("backbone", {})
    cfg.model.backbone["norm_cfg"] = get_norm_cfg(device)
    if "decode_head" in cfg.model:
        cfg.model.decode_head["norm_cfg"] = get_norm_cfg(device)
    if "auxiliary_head" in cfg.model and cfg.model.auxiliary_head is not None:
        cfg.model.auxiliary_head["norm_cfg"] = get_norm_cfg(device)
    cfg.env_cfg.dist_cfg.backend = get_dist_backend(device)

    # -------- TensorBoard --------
    # tensorboard_log_path 是绝对路径，独立于 work_dir / log_path。
    # 这里直接给 TensorboardVisBackend 一个绝对 save_dir，避免 mmengine 把
    # vis_data 挂到 work_dir 下。
    tb_dir = task_cfg.get("tensorboard_log_path")
    use_scalar = task_cfg.get("use_tensorboard_scalar", False)
    use_image = task_cfg.get("use_tensorboard_image", False)
    if tb_dir and (use_scalar or use_image):
        os.makedirs(tb_dir, exist_ok=True)
        backends = list(cfg.get("vis_backends", []) or [])
        backends.append(dict(type="TensorboardVisBackend", save_dir=tb_dir))
        cfg.vis_backends = backends
        if "visualizer" in cfg:
            cfg.visualizer["vis_backends"] = backends
            cfg.visualizer["save_dir"] = tb_dir

    # -------- env 标记，便于日志区分 --------
    env = task_cfg.get("env")
    if env:
        cfg.experiment_name = str(env)

    return cfg


if __name__ == "__main__":
    # 调试入口：python -m integration.config_builder configs/deeplabv3plus_building.py configs/task.conf
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else "configs/deeplabv3plus_building.py"
    conf = sys.argv[2] if len(sys.argv) > 2 else "configs/task.conf"
    from .conf_parser import load_task_conf

    tc = load_task_conf(conf)
    cfg = build_train_cfg(base, tc)
    print(cfg.pretty_text)
