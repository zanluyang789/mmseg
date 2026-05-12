"""
导出 DeepLabV3+ 水体分割模型为 ONNX
- 输入: float32, NCHW, [0,255] 范围的 RGB
- 内部做归一化(ImageNet 均值方差)
- 输出: water 通道 softmax 概率图 (1,1,512,512), float32
"""
import torch
import torch.nn as nn
import argparse
from mmengine.config import Config
from mmseg.apis import init_model

IMAGENET_MEAN = [123.675, 116.28, 103.53]   # 0-255 尺度
IMAGENET_STD  = [58.395, 57.12, 57.375]

class WaterSegONNX(nn.Module):
    """把预处理 + 主干 + softmax + 取 water 通道 全部塞进 ONNX 图里"""
    def __init__(self, mmseg_model):
        super().__init__()
        self.model = mmseg_model
        # 注册为 buffer,会被一起导出到 ONNX
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        # x: (1,3,512,512) float32, [0,255], RGB 顺序
        x = (x - self.mean) / self.std
        # mmseg 模型用 extract_feat + decode_head 走纯前向,不带后处理
        feats = self.model.extract_feat(x)
        logits = self.model.decode_head.predict_by_feat(
            self.model.decode_head(feats),
            batch_img_metas=[{'img_shape': (512, 512), 'ori_shape': (512, 512)}]
        )
        # logits: (1,2,512,512)
        prob = torch.softmax(logits, dim=1)
        water = prob[:, 1:2, :, :]   # (1,1,512,512)
        return water

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',  default='configs/deeplabv3plus_water.py')
    ap.add_argument('--ckpt',    required=True, help='best_mIoU_iter_xxxx.pth 路径')
    ap.add_argument('--out',     default='water_seg.onnx')
    ap.add_argument('--opset',   type=int, default=11)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    # 强制关掉 SyncBN(导出 ONNX 用不了)
    cfg.model.data_preprocessor = None  # 我们自己做归一化
    if 'norm_cfg' in cfg.model:
        cfg.model.norm_cfg = dict(type='BN', requires_grad=True)

    model = init_model(cfg, args.ckpt, device='cuda:0')
    model.eval()

    wrapper = WaterSegONNX(model).cuda().eval()

    # 用一个真实尺度的假输入做 trace
    dummy = torch.rand(1, 3, 512, 512, device='cuda') * 255.0

    with torch.no_grad():
        out = wrapper(dummy)
        print(f'[smoke test] output shape: {out.shape}, range: [{out.min():.4f}, {out.max():.4f}]')

    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        opset_version=args.opset,
        input_names=['input'],
        output_names=['water_prob'],
        dynamic_axes=None,                 # 静态 shape
        do_constant_folding=True,
    )
    print(f'[ok] exported to {args.out}')

    # onnxsim 简化
    try:
        import onnx, onnxsim
        m = onnx.load(args.out)
        m_sim, ok = onnxsim.simplify(m)
        assert ok, 'onnxsim simplify failed'
        onnx.save(m_sim, args.out)
        print(f'[ok] simplified, saved to {args.out}')
    except Exception as e:
        print(f'[warn] onnxsim skipped: {e}')

if __name__ == '__main__':
    main()
