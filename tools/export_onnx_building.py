"""
导出 DeepLabV3+ 建筑物分割模型为 ONNX
- 输入: float32, NCHW, [0,255] RGB
- 内部做归一化(ImageNet 均值方差)
- 输出: building 通道 softmax 概率图 (1,1,512,512), float32
"""
import torch
import torch.nn as nn
import argparse
from mmengine.config import Config
from mmseg.apis import init_model

IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD  = [58.395, 57.12, 57.375]


class BuildingSegONNX(nn.Module):
    """把预处理 + 主干 + softmax + 取 building 通道 全部塞进 ONNX 图里"""
    def __init__(self, mmseg_model):
        super().__init__()
        self.model = mmseg_model
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        # x: (1,3,512,512) float32, [0,255], RGB
        x = (x - self.mean) / self.std
        feats = self.model.extract_feat(x)
        logits = self.model.decode_head.predict_by_feat(
            self.model.decode_head(feats),
            batch_img_metas=[{'img_shape': (512, 512), 'ori_shape': (512, 512)}]
        )
        prob = torch.softmax(logits, dim=1)
        building = prob[:, 1:2, :, :]    # (1,1,512,512)
        return building


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',  default='configs/deeplabv3plus_building.py')
    ap.add_argument('--ckpt',    required=True, help='best_mIoU_iter_xxxx.pth 路径')
    ap.add_argument('--out',     default='building_seg.onnx')
    ap.add_argument('--opset',   type=int, default=11)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    # 关掉 SyncBN(导出 ONNX 用不了)
    cfg.model.data_preprocessor = None
    if 'norm_cfg' in cfg.model:
        cfg.model.norm_cfg = dict(type='BN', requires_grad=True)

    model = init_model(cfg, args.ckpt, device='cuda:0')
    model.eval()

    wrapper = BuildingSegONNX(model).cuda().eval()

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
        output_names=['building_prob'],
        dynamic_axes=None,
        do_constant_folding=True,
    )
    print(f'[ok] exported to {args.out}')

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
