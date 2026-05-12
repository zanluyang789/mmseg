"""
ACL 推理 + 三联图可视化(建筑物版)
依赖: pyACL (NPU 镜像里应该自带), opencv-python, numpy
"""
import argparse
import numpy as np
import cv2
import acl
import time


def chk(ret, msg=''):
    if ret != 0:
        raise RuntimeError(f'{msg} failed, ret={ret}')


class OMInfer:
    def __init__(self, om_path, device_id=0):
        chk(acl.init(), 'acl.init')
        chk(acl.rt.set_device(device_id), 'set_device')
        self.ctx, ret = acl.rt.create_context(device_id); chk(ret, 'create_context')

        self.model_id, ret = acl.mdl.load_from_file(om_path); chk(ret, 'load_model')
        self.model_desc = acl.mdl.create_desc()
        chk(acl.mdl.get_desc(self.model_desc, self.model_id), 'get_desc')

        self.input_size  = acl.mdl.get_input_size_by_index(self.model_desc, 0)
        self.output_size = acl.mdl.get_output_size_by_index(self.model_desc, 0)

    def infer(self, x_float32_nchw):
        """x: numpy (1,3,512,512) float32 [0,255]"""
        assert x_float32_nchw.dtype == np.float32
        assert x_float32_nchw.shape == (1, 3, 512, 512)

        in_host_ptr = x_float32_nchw.ctypes.data
        in_dev_ptr, ret = acl.rt.malloc(self.input_size, 0); chk(ret, 'malloc in')
        chk(acl.rt.memcpy(in_dev_ptr, self.input_size,
                          in_host_ptr,  x_float32_nchw.nbytes,
                          1),
            'memcpy h2d')

        in_data = acl.create_data_buffer(in_dev_ptr, self.input_size)
        in_dataset = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(in_dataset, in_data)

        out_dev_ptr, ret = acl.rt.malloc(self.output_size, 0); chk(ret, 'malloc out')
        out_data = acl.create_data_buffer(out_dev_ptr, self.output_size)
        out_dataset = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(out_dataset, out_data)

        t0 = time.time()
        chk(acl.mdl.execute(self.model_id, in_dataset, out_dataset), 'execute')
        dt = (time.time() - t0) * 1000

        out_host = np.zeros(1 * 1 * 512 * 512, dtype=np.float32)
        chk(acl.rt.memcpy(out_host.ctypes.data, out_host.nbytes,
                          out_dev_ptr, self.output_size,
                          2),
            'memcpy d2h')

        acl.destroy_data_buffer(in_data)
        acl.destroy_data_buffer(out_data)
        acl.mdl.destroy_dataset(in_dataset)
        acl.mdl.destroy_dataset(out_dataset)
        acl.rt.free(in_dev_ptr)
        acl.rt.free(out_dev_ptr)

        return out_host.reshape(1, 1, 512, 512), dt

    def close(self):
        acl.mdl.unload(self.model_id)
        acl.mdl.destroy_desc(self.model_desc)
        acl.rt.destroy_context(self.ctx)
        acl.rt.reset_device(0)
        acl.finalize()


def make_triptych(img_bgr, mask_uint8, out_path):
    """三联图: 原图 / mask / 红色半透明叠加"""
    h, w = img_bgr.shape[:2]
    mask_vis = cv2.cvtColor(mask_uint8 * 255, cv2.COLOR_GRAY2BGR)

    # 建筑物半透明红色叠加 (BGR: 红=(0,0,255))
    red = np.zeros_like(img_bgr); red[:, :, 2] = 255
    alpha = 0.5
    mask_3c = np.stack([mask_uint8] * 3, axis=-1).astype(bool)
    overlay = np.where(mask_3c,
                       (img_bgr * (1 - alpha) + red * alpha).astype(np.uint8),
                       img_bgr)

    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    triptych = np.concatenate([img_bgr, sep, mask_vis, sep, overlay], axis=1)

    title_h = 30
    canvas = np.full((h + title_h, triptych.shape[1], 3), 255, dtype=np.uint8)
    canvas[title_h:] = triptych
    for text, x in [('Original', 10), ('Mask', w + 14), ('Overlay', 2 * w + 28)]:
        cv2.putText(canvas, text, (x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imwrite(out_path, canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--om',  required=True)
    ap.add_argument('--img', required=True, help='输入 RGB 512x512 图(或更大,自动 resize)')
    ap.add_argument('--out', default='result_building.png')
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    img_bgr = cv2.imread(args.img, cv2.IMREAD_UNCHANGED)
    if img_bgr.shape[:2] != (512, 512):
        img_bgr = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
    assert img_bgr.ndim == 3 and img_bgr.shape[2] == 3, f'got shape {img_bgr.shape}'

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = img_rgb.transpose(2, 0, 1)[None].astype(np.float32)   # (1,3,512,512)

    runner = OMInfer(args.om)
    # 预热一次再正式计时
    runner.infer(x)
    prob, latency_ms = runner.infer(x)

    print(f'inference latency: {latency_ms:.2f} ms')
    print(f'prob range: [{prob.min():.4f}, {prob.max():.4f}]')

    mask = (prob[0, 0] > args.threshold).astype(np.uint8)
    print(f'building pixels: {mask.sum()} / {mask.size} ({100*mask.sum()/mask.size:.2f}%)')

    make_triptych(img_bgr, mask, args.out)
    print(f'saved triptych to {args.out}')

    runner.close()


if __name__ == '__main__':
    main()
