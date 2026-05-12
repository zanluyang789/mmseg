"""
建筑物大图 NPU 推理(滑窗 + 重叠 + 概率叠加)

输入: 一张多波段 GeoTIFF (DATA_INPUT_DIR{1,2})
输出:
    <stem>_building_mask.tif      二值 mask, 与原图同坐标系/分辨率
    <stem>_building_prob.tif      (可选) 概率图 uint8 [0,255]

环境变量:
    PATH_MODEL_RESOURCE   模型目录,默认到这下面找 building_seg.om
    DATA_OUTPUT_DIR       输出根目录(必填)
    PATH_WORKING          临时工作目录(裁剪后影像放这,默认 DATA_OUTPUT_DIR/working)
    VECTOR_WKT            WKT 字符串(可选,WGS84)
    BAND_ORDER            通道映射,默认 "3,2,1" (GF 多光谱 -> RGB)
    THRESHOLD             二值化阈值,默认 0.5
    TILE                  瓦片边长,默认 512(必须与 om 模型 input shape 一致)
    STRIDE                滑窗步长,默认 384(重叠 128)
    SAVE_PROB             "1" 同时输出概率图,默认 "0"

命令行 (覆盖 env):
    python infer_large_image.py --input <tif> --output-dir <dir> \
        [--om <om path>] [--clip-shp <shp>] [--threshold 0.5]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_geom
from rasterio.features import geometry_mask
from shapely import wkt as shapely_wkt
from shapely.geometry import mapping

# Kafka 进度上报(失败也无所谓,会 fallback 到 stdout)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from send_kafka_msg import send as kafka_send  # noqa: E402

# 尝试加载 pyACL;PC 上没有这个包,自动回落到 onnxruntime 走 ONNX 模型(用于调试)
HAS_ACL = False
try:
    import acl  # noqa: F401
    HAS_ACL = True
except Exception:
    pass


# ============================================================
# 模型封装
# ============================================================
class OMRunner:
    """pyACL 推理. 与 tools/infer_om_building.py 里的 OMInfer 同款"""
    def __init__(self, om_path, device_id=0):
        import acl
        self._acl = acl
        ret = acl.init(); self._chk(ret, 'acl.init')
        ret = acl.rt.set_device(device_id); self._chk(ret, 'set_device')
        self.ctx, ret = acl.rt.create_context(device_id); self._chk(ret, 'create_context')
        self.model_id, ret = acl.mdl.load_from_file(om_path); self._chk(ret, 'load_model')
        self.model_desc = acl.mdl.create_desc()
        self._chk(acl.mdl.get_desc(self.model_desc, self.model_id), 'get_desc')
        self.input_size = acl.mdl.get_input_size_by_index(self.model_desc, 0)
        self.output_size = acl.mdl.get_output_size_by_index(self.model_desc, 0)

    @staticmethod
    def _chk(ret, msg):
        if ret != 0:
            raise RuntimeError(f'{msg} failed, ret={ret}')

    def infer(self, x_nchw_fp32):
        acl = self._acl
        assert x_nchw_fp32.dtype == np.float32
        assert x_nchw_fp32.shape == (1, 3, 512, 512)
        in_dev, ret = acl.rt.malloc(self.input_size, 0); self._chk(ret, 'malloc in')
        acl.rt.memcpy(in_dev, self.input_size, x_nchw_fp32.ctypes.data,
                      x_nchw_fp32.nbytes, 1)
        in_buf = acl.create_data_buffer(in_dev, self.input_size)
        in_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(in_ds, in_buf)

        out_dev, ret = acl.rt.malloc(self.output_size, 0); self._chk(ret, 'malloc out')
        out_buf = acl.create_data_buffer(out_dev, self.output_size)
        out_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(out_ds, out_buf)

        self._chk(acl.mdl.execute(self.model_id, in_ds, out_ds), 'execute')

        out = np.zeros(1 * 1 * 512 * 512, dtype=np.float32)
        acl.rt.memcpy(out.ctypes.data, out.nbytes, out_dev, self.output_size, 2)

        acl.destroy_data_buffer(in_buf)
        acl.destroy_data_buffer(out_buf)
        acl.mdl.destroy_dataset(in_ds)
        acl.mdl.destroy_dataset(out_ds)
        acl.rt.free(in_dev)
        acl.rt.free(out_dev)
        return out.reshape(1, 1, 512, 512)

    def close(self):
        acl = self._acl
        acl.mdl.unload(self.model_id)
        acl.mdl.destroy_desc(self.model_desc)
        acl.rt.destroy_context(self.ctx)
        acl.rt.reset_device(0)
        acl.finalize()


class ONNXRunner:
    """ONNX 回落, 仅 PC 调试用"""
    def __init__(self, onnx_path):
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def infer(self, x_nchw_fp32):
        return self.sess.run([self.out_name], {self.in_name: x_nchw_fp32})[0]

    def close(self):
        pass


def make_runner(model_path):
    suffix = Path(model_path).suffix.lower()
    if suffix == '.om':
        if not HAS_ACL:
            raise RuntimeError('要跑 .om 但没有 pyACL,请在 NPU 环境运行')
        print(f'[runner] OM (NPU): {model_path}', flush=True)
        return OMRunner(model_path)
    elif suffix == '.onnx':
        print(f'[runner] ONNX (CPU/GPU 回落): {model_path}', flush=True)
        return ONNXRunner(model_path)
    raise ValueError(f'不识别的模型扩展名: {suffix}')


# ============================================================
# 影像处理
# ============================================================
def pick_rgb(arr_chw, band_order_str):
    """从多波段影像里取 3 个波段拼 RGB
    arr_chw: (C, H, W) uint8/uint16, band_order_str: '3,2,1' 等
    """
    idx = [int(x) - 1 for x in band_order_str.split(',')]  # 1-based -> 0-based
    assert len(idx) == 3, f'BAND_ORDER 必须 3 个: {band_order_str}'
    rgb = arr_chw[idx]  # (3, H, W)
    # 兼容 uint16(GF 原始可能是 16bit), 等比拉伸到 uint8
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        # 2/98 百分位拉伸 (稳健)
        out = np.zeros_like(rgb, dtype=np.uint8)
        for c in range(3):
            ch = rgb[c]
            lo, hi = np.percentile(ch[ch > 0], (2, 98)) if (ch > 0).any() else (0, 255)
            if hi <= lo:
                hi = lo + 1
            ch = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255)
            out[c] = ch.astype(np.uint8)
        rgb = out
    return rgb  # (3, H, W) uint8


def build_blend_weight(tile=512, ramp=64):
    """生成羽化权重(中心 1,边缘平滑下降),减弱拼缝"""
    w1 = np.ones(tile, dtype=np.float32)
    if ramp > 0:
        r = np.linspace(0, 1, ramp, dtype=np.float32)
        w1[:ramp] = r
        w1[-ramp:] = r[::-1]
    w2 = np.outer(w1, w1)  # (tile, tile)
    return w2


def slide_infer(rgb_chw_u8, runner, tile=512, stride=384, ramp=64,
                progress_cb=None):
    """
    rgb_chw_u8: (3, H, W) uint8, RGB
    返回: prob_hw float32 [0,1]
    """
    _, H, W = rgb_chw_u8.shape
    weight = build_blend_weight(tile, ramp)

    prob_acc = np.zeros((H, W), dtype=np.float32)
    w_acc = np.zeros((H, W), dtype=np.float32)

    # 滑窗起点,确保覆盖边缘
    def gen_starts(L, t, s):
        if L <= t:
            return [0]
        starts = list(range(0, L - t + 1, s))
        if starts[-1] + t < L:
            starts.append(L - t)
        return starts

    ys = gen_starts(H, tile, stride)
    xs = gen_starts(W, tile, stride)
    total = len(ys) * len(xs)
    print(f'[slide] image {H}x{W}, tiles {len(ys)}x{len(xs)} = {total}', flush=True)

    t0 = time.time()
    done = 0
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            tile_rgb = rgb_chw_u8[:, y:y+tile, x:x+tile]
            # 边缘 pad
            ph = tile - tile_rgb.shape[1]
            pw = tile - tile_rgb.shape[2]
            if ph > 0 or pw > 0:
                tile_rgb = np.pad(tile_rgb, ((0, 0), (0, ph), (0, pw)), mode='reflect')

            inp = tile_rgb.astype(np.float32)[None]  # (1,3,512,512), [0,255]
            out = runner.infer(inp)                  # (1,1,512,512)
            prob = out[0, 0]                          # (512,512)

            # 累加到 acc, 截掉 pad
            eff_h = tile - ph
            eff_w = tile - pw
            prob_acc[y:y+eff_h, x:x+eff_w] += (prob[:eff_h, :eff_w]
                                                * weight[:eff_h, :eff_w])
            w_acc[y:y+eff_h, x:x+eff_w] += weight[:eff_h, :eff_w]
            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f'  [{done}/{total}] elapsed={elapsed:.1f}s eta={eta:.1f}s',
                      flush=True)
                if progress_cb:
                    progress_cb(done, total)

    # 平均
    w_acc[w_acc == 0] = 1  # 避免除零(理论上不会发生)
    prob = prob_acc / w_acc
    return prob


def apply_wkt_mask(prob, src_transform, src_crs, wkt_str):
    """WKT 在 WGS84,投影到影像坐标系后,把 ROI 外置为 0"""
    if not wkt_str or wkt_str.lower() == 'none':
        return prob
    geom = shapely_wkt.loads(wkt_str)
    geom_geo = mapping(geom)
    # 重投影到栅格 CRS
    if str(src_crs).upper() != 'EPSG:4326':
        geom_geo = transform_geom('EPSG:4326', src_crs, geom_geo)
    h, w = prob.shape
    mask = geometry_mask([geom_geo], out_shape=(h, w),
                         transform=src_transform, invert=True)
    out = prob.copy()
    out[~mask] = 0
    return out


# ============================================================
# 主流程
# ============================================================
def process_one(input_tif, output_dir, model_path, wkt_str,
                band_order, threshold, tile, stride,
                save_prob, progress_range=(0, 100)):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_tif).stem
    out_mask = output_dir / f'{stem}_building_mask.tif'
    out_prob = output_dir / f'{stem}_building_prob.tif'

    p_lo, p_hi = progress_range

    with rasterio.open(input_tif) as src:
        print(f'[input] {input_tif}', flush=True)
        print(f'         crs={src.crs}  size={src.width}x{src.height}'
              f'  bands={src.count}  dtype={src.dtypes[0]}', flush=True)
        # 全图读入(超大图后续可改成 windowed,这版先简单)
        arr = src.read()  # (C, H, W)
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

    rgb = pick_rgb(arr, band_order)
    del arr  # 释放原始多波段

    runner = make_runner(model_path)

    def cb(done, total):
        # 进度按 progress_range 映射
        prog = p_lo + (p_hi - p_lo) * done / max(total, 1)
        kafka_send(int(prog), 'running', f'推理中 {done}/{total} 瓦片')

    try:
        prob = slide_infer(rgb, runner, tile=tile, stride=stride,
                            progress_cb=cb)
    finally:
        runner.close()

    # ROI 裁剪
    prob = apply_wkt_mask(prob, transform, crs, wkt_str)
    mask = (prob >= threshold).astype(np.uint8)
    print(f'[stats] 建筑物像素占比: {100.0*mask.sum()/mask.size:.3f}%', flush=True)

    # 写 mask GeoTIFF (保留坐标系)
    profile.update(count=1, dtype='uint8', nodata=255, compress='lzw')
    with rasterio.open(out_mask, 'w', **profile) as dst:
        dst.write(mask, 1)
    print(f'[ok] mask -> {out_mask}', flush=True)

    if save_prob:
        prob_u8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        with rasterio.open(out_prob, 'w', **profile) as dst:
            dst.write(prob_u8, 1)
        print(f'[ok] prob -> {out_prob}', flush=True)

    return str(out_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', help='输入 GeoTIFF(覆盖 DATA_INPUT_DIR1)')
    ap.add_argument('--output-dir', help='输出目录(覆盖 DATA_OUTPUT_DIR)')
    ap.add_argument('--om', help='om/onnx 模型路径(覆盖 PATH_MODEL_RESOURCE)')
    ap.add_argument('--wkt', help='WKT 字符串(覆盖 VECTOR_WKT)')
    ap.add_argument('--band-order', default=None)
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--tile', type=int, default=None)
    ap.add_argument('--stride', type=int, default=None)
    ap.add_argument('--save-prob', action='store_true')
    ap.add_argument('--progress-lo', type=int, default=0)
    ap.add_argument('--progress-hi', type=int, default=100)
    args = ap.parse_args()

    # env 优先级: 命令行 > env > 默认
    input_tif = args.input or os.environ.get('DATA_INPUT_DIR1')
    output_dir = args.output_dir or os.environ.get('DATA_OUTPUT_DIR')
    model_path = args.om
    if not model_path:
        mdir = os.environ.get('PATH_MODEL_RESOURCE', '/app/module')
        # 优先 om, 回落 onnx
        for cand in ('building_seg.om', 'building_seg.onnx'):
            p = os.path.join(mdir, cand)
            if os.path.exists(p):
                model_path = p
                break
    wkt_str = args.wkt if args.wkt is not None else os.environ.get('VECTOR_WKT', '')
    band_order = args.band_order or os.environ.get('BAND_ORDER', '3,2,1')
    threshold = args.threshold if args.threshold is not None \
        else float(os.environ.get('THRESHOLD', '0.5'))
    tile = args.tile or int(os.environ.get('TILE', '512'))
    stride = args.stride or int(os.environ.get('STRIDE', '384'))
    save_prob = args.save_prob or (os.environ.get('SAVE_PROB', '0') == '1')

    print('======== infer_large_image ========', flush=True)
    print(f'input       = {input_tif}', flush=True)
    print(f'output_dir  = {output_dir}', flush=True)
    print(f'model       = {model_path}', flush=True)
    print(f'wkt         = {(wkt_str[:80] + "...") if wkt_str and len(wkt_str)>80 else wkt_str}', flush=True)
    print(f'band_order  = {band_order}', flush=True)
    print(f'threshold   = {threshold}', flush=True)
    print(f'tile/stride = {tile}/{stride}  save_prob={save_prob}', flush=True)
    print('===================================', flush=True)

    if not input_tif or not os.path.isfile(input_tif):
        print(f'[fatal] input 不存在: {input_tif}', file=sys.stderr); sys.exit(2)
    if not output_dir:
        print('[fatal] output_dir 未设置', file=sys.stderr); sys.exit(2)
    if not model_path or not os.path.isfile(model_path):
        print(f'[fatal] 模型不存在: {model_path}', file=sys.stderr); sys.exit(2)

    process_one(input_tif, output_dir, model_path, wkt_str,
                band_order, threshold, tile, stride,
                save_prob, progress_range=(args.progress_lo, args.progress_hi))


if __name__ == '__main__':
    main()
