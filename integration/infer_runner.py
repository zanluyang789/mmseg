"""
推理核心（三栈统一：pth / onnx / om）
=====================================

按文档约定，推理时参数从 clie_lib/configs/task.conf 读：
    必填：
        load_model_path     模型路径（.pth / .onnx / .om）
        img_list_file       影像列表（每行一个绝对路径），单时相
        output_root         输出目录
    可选：
        img2_list_file      变化检测后时项，本项目用不到
        band_list_file      多波段 RGB 映射 "3,2,1"（也可以是文件里一行）
        color_table_file    调色板/类别表
        mean_file / std_file
        bootstrap_servers / topic / projectId / taskId   Kafka 上报
        use_color_out       输出彩色 PNG
        use_shapefile_out   输出 SHP
        num_classes / classes_name / palette   类别信息（建筑物=2 类）
        threshold           二值化阈值，默认 0.5（建筑物 / 水体 适用）
        tile / stride       滑窗尺寸 / 步长，默认 512 / 384
        save_prob           输出概率图
        min_area_px         矢量化最小图斑像素数，默认 25
        device              强制 device（cuda/npu/cpu/om）

设备/后端选择：
    load_model_path 后缀决定走哪个 runner：
        .om   -> NPU pyACL  （昨天的逻辑）
        .onnx -> onnxruntime（CPU/GPU 回落）
        .pth  -> PyTorch    （GPU 推理，NPU 也行，看 device）

复用昨天的 slide_infer 滑窗：从 deploy_building/infer_large_image.py 抽出来；
但要支持非 GeoTIFF 输入（PNG/JPG 走 cv2，不带 GeoTIFF 头时不出 SHP）。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# ============= 复用 deploy_building 已有的滑窗、矢量化、Kafka =============
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPLOY = os.path.join(os.path.dirname(_HERE), "deploy_building")
if _DEPLOY not in sys.path:
    sys.path.insert(0, _DEPLOY)

# 注意：deploy_building 里 import acl 是可选的，这里也跟着走
from infer_large_image import (  # type: ignore  # noqa: E402
    OMRunner,
    ONNXRunner,
    apply_wkt_mask,
    build_blend_weight,
    pick_rgb,
    slide_infer,
)
from postprocess import vectorize  # type: ignore  # noqa: E402
from send_kafka_msg import send as _kafka_send_env  # type: ignore  # noqa: E402

from .conf_parser import load_task_conf
from .config_builder import _read_floats_file, _patch_segmentor_classes
from .device_utils import detect_device, setup_device_env


# ============================================================
# PyTorch (pth) 推理 runner —— 给 GPU / NPU 直接吃 pth 用
# ============================================================
class PthRunner:
    """
    直接吃 mmseg ckpt（.pth + 同目录下的 config）做 forward。

    用法：
        runner = PthRunner('best_mIoU_iter_xxx.pth',
                           config='configs/deeplabv3plus_building.py',
                           device='cuda',
                           tile=512,
                           target_class=1)   # 取建筑物通道
    输入: (1,3,tile,tile) float32 [0,255] RGB
    输出: (1,1,tile,tile) float32 概率图（target_class 通道 softmax 后）
    """

    def __init__(
        self,
        ckpt: str,
        config: str,
        device: str = "cuda",
        tile: int = 512,
        target_class: int = 1,
        task_cfg: Optional[dict] = None,
    ):
        import torch
        from mmengine.config import Config
        from mmseg.apis import init_model

        self.torch = torch
        cfg = Config.fromfile(config)
        # SyncBN 单进程跑不动，强制 BN
        if "norm_cfg" in cfg.model:
            cfg.model.norm_cfg = dict(type="BN", requires_grad=True)
        for k in ("backbone", "decode_head", "auxiliary_head"):
            if k in cfg.model and isinstance(cfg.model[k], dict):
                cfg.model[k]["norm_cfg"] = dict(type="BN", requires_grad=True)

        # 同步 task.conf 的 num_classes（推理时若 task_cfg.num_classes 跟
        # base config 不一致，必须先调整 head 再 load ckpt，否则形状对不上）
        task_cfg = task_cfg or {}
        nc_task = task_cfg.get("num_classes")
        if nc_task is not None:
            try:
                _patch_segmentor_classes(cfg, int(nc_task))
            except Exception as e:
                print(f"[PthRunner] num_classes 同步失败({e})，沿用 base config", flush=True)

        # 优先用 task.conf 的 mean_file/std_file —— 跟训练保持一致
        mean_from_file = _read_floats_file(task_cfg.get("mean_file", ""), None)
        std_from_file = _read_floats_file(task_cfg.get("std_file", ""), None)
        dp = cfg.model.get("data_preprocessor", {}) or {}
        base_mean = dp.get("mean", [123.675, 116.28, 103.53])
        base_std = dp.get("std", [58.395, 57.12, 57.375])
        final_mean = mean_from_file if mean_from_file else base_mean
        final_std = std_from_file if std_from_file else base_std
        if mean_from_file or std_from_file:
            print(f"[PthRunner] 用 task.conf mean/std: mean={final_mean} std={final_std}",
                  flush=True)
        else:
            print(f"[PthRunner] 用 base config mean/std: mean={final_mean} std={final_std}",
                  flush=True)

        torch_device = "cuda:0" if device == "cuda" else (
            "npu:0" if device == "npu" else "cpu"
        )
        self.model = init_model(cfg, ckpt, device=torch_device)
        self.model.eval()
        self.device = torch_device
        self.tile = tile
        self.target_class = target_class
        self.mean = torch.tensor(final_mean).view(1, 3, 1, 1).to(torch_device)
        self.std = torch.tensor(final_std).view(1, 3, 1, 1).to(torch_device)
        self.bgr_to_rgb = dp.get("bgr_to_rgb", False)

    @property
    def num_classes(self) -> int:
        head = getattr(self.model, "decode_head", None)
        if head is not None and hasattr(head, "num_classes"):
            return int(head.num_classes)
        return 2

    def infer(self, x_nchw_fp32: np.ndarray) -> np.ndarray:
        """跟 OMRunner.infer 同签名，输入 (1,3,T,T) float32 [0,255] RGB"""
        torch = self.torch
        assert x_nchw_fp32.shape[1] == 3
        with torch.no_grad():
            x = torch.from_numpy(x_nchw_fp32).to(self.device)
            if self.bgr_to_rgb:
                # 调用方传的已经是 RGB（pick_rgb 输出），这里 BGR->RGB 跳过
                pass
            x = (x - self.mean) / self.std
            feats = self.model.extract_feat(x)
            logits = self.model.decode_head.predict_by_feat(
                self.model.decode_head(feats),
                batch_img_metas=[{"img_shape": (self.tile, self.tile),
                                  "ori_shape": (self.tile, self.tile)}],
            )
            prob = torch.softmax(logits, dim=1)
            target = prob[:, self.target_class : self.target_class + 1, :, :]
            return target.detach().cpu().numpy().astype(np.float32)

    def close(self):
        del self.model


def make_runner_any(model_path: str, device: str, tile: int = 512,
                    base_config: Optional[str] = None,
                    target_class: int = 1,
                    task_cfg: Optional[dict] = None):
    """根据后缀和设备选 runner"""
    suffix = Path(model_path).suffix.lower()
    if suffix == ".om":
        print(f"[runner] OM (NPU): {model_path}", flush=True)
        return OMRunner(model_path)
    if suffix == ".onnx":
        print(f"[runner] ONNX ({device}): {model_path}", flush=True)
        return ONNXRunner(model_path)
    if suffix in (".pth", ".pt"):
        if base_config is None:
            raise ValueError("pth 推理需要 base_config（mmseg config 路径）")
        print(f"[runner] Pth ({device}): {model_path}  cfg={base_config}", flush=True)
        return PthRunner(model_path, base_config, device=device,
                         tile=tile, target_class=target_class,
                         task_cfg=task_cfg)
    raise ValueError(f"不识别的模型后缀: {suffix}")


# ============================================================
# 大图 / 单图 统一处理
# ============================================================
def _read_list(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"列表文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _is_geotiff(path: str) -> bool:
    return Path(path).suffix.lower() in (".tif", ".tiff")


def _resolve_band_order(task_cfg: dict) -> str:
    """
    band_list_file 兼容多种格式：
      - 直接字符串      "3,2,1"
      - 文件，逗号分隔   "3,2,1"
      - 文件，空白分隔   "3 2 1"
      - 文件，每行一个   "0\n1\n2"
    并且自动识别 0-based / 1-based：
      - 最小值是 0 -> 系统下发的 0-based 通道索引，全部 +1 转成 1-based
      - 最小值是 1 -> 已经是 1-based，原样返回
    返回值始终是 1-based 的逗号分隔字符串（喂给 deploy_building.pick_rgb 后会 -1）。
    """
    import re

    blf = task_cfg.get("band_list_file")
    if not blf:
        return "3,2,1"

    if os.path.isfile(blf):
        with open(blf, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        # 不是文件就当字符串
        raw = str(blf)

    # 用任何空白 / 逗号切
    parts = [p for p in re.split(r"[,\s]+", raw.strip()) if p]
    if not parts:
        return "3,2,1"

    try:
        nums = [int(p) for p in parts]
    except ValueError:
        # 解析不了纯数字时退回原字符串（极端兜底）
        return raw.strip().replace("\n", ",")

    if len(nums) < 3:
        print(f"[band_order] 警告：只识别到 {len(nums)} 个波段 {nums}，"
              f"用默认 '3,2,1'", flush=True)
        return "3,2,1"

    if min(nums) == 0:
        nums = [n + 1 for n in nums]
        print(f"[band_order] 检测到 0-based 索引 {parts}, 自动转 1-based {nums}",
              flush=True)

    # 只取前 3 个，对应 R/G/B
    return ",".join(str(n) for n in nums[:3])


def _save_mask_geotiff(mask: np.ndarray, ref_profile: dict, out_path: str):
    import rasterio

    profile = dict(ref_profile)
    profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def _imwrite_utf8(out_path: str, img_arr: np.ndarray, ext: str = ".png") -> None:
    """跨平台中文路径友好的 cv2.imwrite 替代品。"""
    import cv2

    ok, buf = cv2.imencode(ext, img_arr)
    if not ok:
        raise RuntimeError(f"cv2.imencode 失败: {out_path}")
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())


def _save_color_png(mask: np.ndarray, palette: List[List[int]], out_path: str):
    """根据 palette 上色后保存 PNG（多分类时按类索引上色）"""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, rgb in enumerate(palette):
        color[mask == idx] = rgb[::-1]  # BGR -> 给 cv2
    _imwrite_utf8(out_path, color, ext=".png")


def _read_non_geotiff(path: str, band_order_str: str) -> np.ndarray:
    """读 PNG/JPG 返回 (3,H,W) uint8 RGB"""
    import cv2

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"读不到影像: {path}")
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1)  # (3,H,W)


def _process_one(
    img_path: str,
    output_root: str,
    runner,
    task_cfg: dict,
    progress_cb=None,
) -> dict:
    """
    输出：{'mask': xxx, 'shp': xxx or None, 'color': xxx or None}
    """
    out = {"mask": None, "shp": None, "color": None}
    threshold = float(task_cfg.get("threshold", 0.5))
    tile = int(task_cfg.get("tile", 512))
    stride = int(task_cfg.get("stride", 384))
    save_prob = bool(task_cfg.get("save_prob", False))
    band_order = _resolve_band_order(task_cfg)
    palette = task_cfg.get("palette") or [[0, 0, 0], [220, 20, 60]]
    use_color = bool(task_cfg.get("use_color_out", False))
    use_shp = bool(task_cfg.get("use_shapefile_out", False))
    min_area_px = int(task_cfg.get("min_area_px", 25))
    stem = Path(img_path).stem
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if _is_geotiff(img_path):
        import rasterio

        with rasterio.open(img_path) as src:
            arr = src.read()
            transform = src.transform
            crs = src.crs
            profile = src.profile.copy()
        rgb = pick_rgb(arr, band_order)
        del arr
        prob = slide_infer(rgb, runner, tile=tile, stride=stride, progress_cb=progress_cb)
        wkt_str = task_cfg.get("vector_wkt") or os.environ.get("VECTOR_WKT", "")
        if wkt_str:
            prob = apply_wkt_mask(prob, transform, crs, wkt_str)
        mask = (prob >= threshold).astype(np.uint8)

        mask_path = output_root / f"{stem}_mask.tif"
        _save_mask_geotiff(mask, profile, str(mask_path))
        out["mask"] = str(mask_path)

        if save_prob:
            prob_u8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
            prob_path = output_root / f"{stem}_prob.tif"
            _save_mask_geotiff(prob_u8, profile, str(prob_path))

        if use_shp:
            shp_path = output_root / f"{stem}.shp"
            vectorize(str(mask_path), str(shp_path), min_area_px=min_area_px)
            out["shp"] = str(shp_path)

        if use_color:
            color_path = output_root / f"{stem}_color.png"
            _save_color_png(mask, palette, str(color_path))
            out["color"] = str(color_path)

    else:
        # PNG/JPG 单图（小图）路径
        rgb = _read_non_geotiff(img_path, band_order)
        prob = slide_infer(rgb, runner, tile=tile, stride=stride, progress_cb=progress_cb)
        mask = (prob >= threshold).astype(np.uint8)

        mask_path = output_root / f"{stem}_mask.png"
        _imwrite_utf8(str(mask_path), (mask * 255).astype(np.uint8), ext=".png")
        out["mask"] = str(mask_path)

        if use_color:
            color_path = output_root / f"{stem}_color.png"
            _save_color_png(mask, palette, str(color_path))
            out["color"] = str(color_path)

    return out


# ============================================================
# Kafka 上报：优先用 task.conf 里的 bootstrap_servers / topic / taskId
# 没有就回落到 deploy_building/send_kafka_msg 里的 env 版本
# ============================================================
def _kafka_send(progress: int, status: str, info: str, task_cfg: dict):
    bs = task_cfg.get("bootstrap_servers")
    topic = task_cfg.get("topic")
    task_id = task_cfg.get("taskId")
    if bs and topic:
        # 临时塞 env，复用 deploy_building 的 send 逻辑
        os.environ["KAFKA_SERVER_IP_PORT"] = str(bs)
        os.environ["KAFKA_TOPIC"] = str(topic)
        if task_id is not None:
            os.environ["KAFKA_TASK_ID"] = str(task_id)
    _kafka_send_env(progress, status, info)


# ============================================================
# 主入口
# ============================================================
def run_inference(
    task_conf_path: str = "clie_lib/configs/task.conf",
    device: Optional[str] = None,
    base_config: Optional[str] = None,
):
    # 让 GDAL / fiona / rasterio 把所有路径字符串当 UTF-8 处理
    # —— 中文 output_root（例：'/share/.../建筑物识别'）才不会写文件失败
    os.environ.setdefault("GDAL_FILENAME_IS_UTF8", "YES")
    os.environ.setdefault("SHAPE_ENCODING", "UTF-8")

    print(f"[run_inference] task_conf = {os.path.abspath(task_conf_path)}", flush=True)
    task_cfg = load_task_conf(task_conf_path)
    print(f"[run_inference] keys = {sorted(task_cfg.keys())}", flush=True)

    model_path = task_cfg.get("load_model_path")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"load_model_path 不存在: {model_path}")

    img_list_file = task_cfg.get("img_list_file")
    if not img_list_file:
        raise ValueError("img_list_file 未配置")
    images = _read_list(img_list_file)
    if not images:
        raise ValueError(f"img_list_file 是空的: {img_list_file}")

    output_root = task_cfg.get("output_root")
    if not output_root:
        raise ValueError("output_root 未配置")

    # base_config 选择：pth 必须，om/onnx 不需要
    suffix = Path(model_path).suffix.lower()
    if suffix in (".pth", ".pt") and not base_config:
        base_config = (
            task_cfg.get("base_config")
            or "configs/deeplabv3plus_building.py"
        )

    device = setup_device_env(device or task_cfg.get("device") or detect_device())
    if suffix == ".om" and device != "npu":
        print(f"[warn] 模型是 .om 但 device={device}，强制切到 npu", flush=True)
        device = "npu"

    tile = int(task_cfg.get("tile", 512))
    # target_class 默认取最后一类（建筑物=1 / 水体=1 / 多类时取末类）
    # task.conf 可以用 target_class 字段显式覆盖
    nc = task_cfg.get("num_classes")
    default_target = (int(nc) - 1) if nc else 1
    target_class = int(task_cfg.get("target_class", default_target))
    print(f"[run_inference] num_classes={nc}, target_class={target_class}", flush=True)

    runner = make_runner_any(model_path, device, tile=tile,
                             base_config=base_config,
                             target_class=target_class,
                             task_cfg=task_cfg)

    _kafka_send(0, "running", "推理任务启动", task_cfg)

    try:
        total = len(images)
        for i, img in enumerate(images, 1):
            print(f"\n===== [{i}/{total}] {img} =====", flush=True)

            def cb(done, tot, _i=i, _total=total):
                # 把单图内的瓦片进度也带上整体百分比
                per_img = (done / max(tot, 1))
                overall = int((_i - 1 + per_img) / _total * 90) + 5
                _kafka_send(min(overall, 95), "running",
                            f"[{_i}/{_total}] 瓦片 {done}/{tot}", task_cfg)

            try:
                res = _process_one(img, output_root, runner, task_cfg,
                                   progress_cb=cb)
                print(f"  -> {res}", flush=True)
            except Exception as e:
                print(f"[error] {img} 推理失败: {e}", flush=True)
                _kafka_send(int(i / total * 95), "running",
                            f"[{i}/{total}] 失败: {e}", task_cfg)
                continue

        _kafka_send(100, "completed", f"完成 {total} 张影像推理", task_cfg)
    finally:
        runner.close()


def main_cli(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser("mmseg predict (integration)")
    parser.add_argument("action", nargs="?", default="infer", choices=["infer"])
    parser.add_argument("--custom-config", default="task",
                        help="文档要求传 'task'，对应 clie_lib/configs/task.conf；"
                        "也可以传文件绝对路径")
    parser.add_argument("--device", default=None,
                        choices=[None, "cuda", "npu", "cpu"])
    parser.add_argument("--base", default=None,
                        help="pth 推理时的 mmseg config（不传则用 configs/deeplabv3plus_building.py）")
    args = parser.parse_args(argv)

    cc = args.custom_config
    if cc == "task" or cc is None:
        task_conf_path = "clie_lib/configs/task.conf"
    elif os.path.isabs(cc) or cc.endswith(".conf"):
        task_conf_path = cc
    else:
        task_conf_path = f"clie_lib/configs/{cc}.conf"

    run_inference(task_conf_path=task_conf_path,
                  device=args.device,
                  base_config=args.base)
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
