"""
对比 PyTorch 原模型 和 导出的 ONNX,在同一张图上推理,看输出差异。
差异应该 < 1e-4。如果大于这个,说明 ONNX 转坏了。
"""
import argparse
import numpy as np
import torch
import cv2
import onnxruntime as ort
from mmengine.config import Config
from mmseg.apis import init_model

def torch_infer(cfg_path, ckpt, img_chw_uint8):
    cfg = Config.fromfile(cfg_path)
    cfg.model.data_preprocessor = None
    if 'norm_cfg' in cfg.model:
        cfg.model.norm_cfg = dict(type='BN', requires_grad=True)
    model = init_model(cfg, ckpt, device='cuda:0').eval()

    mean = torch.tensor([123.675,116.28,103.53]).view(1,3,1,1).cuda()
    std  = torch.tensor([58.395,57.12,57.375]).view(1,3,1,1).cuda()
    x = torch.from_numpy(img_chw_uint8).float().unsqueeze(0).cuda()
    x = (x - mean) / std
    with torch.no_grad():
        feats = model.extract_feat(x)
        logits = model.decode_head.predict_by_feat(
            model.decode_head(feats),
            batch_img_metas=[{'img_shape':(512,512),'ori_shape':(512,512)}]
        )
        prob = torch.softmax(logits, dim=1)[:,1:2]
    return prob.cpu().numpy()

def onnx_infer(onnx_path, img_chw_uint8):
    sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    x = img_chw_uint8.astype(np.float32)[None]   # (1,3,512,512)
    out = sess.run(['water_prob'], {'input': x})[0]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/deeplabv3plus_water.py')
    ap.add_argument('--ckpt',   required=True)
    ap.add_argument('--onnx',   default='water_seg.onnx')
    ap.add_argument('--img',    required=True, help='验证集里随便挑一张 .tif')
    args = ap.parse_args()

    img = cv2.imread(args.img, cv2.IMREAD_UNCHANGED)  # BGR uint8
    assert img.shape == (512,512,3), f'got shape {img.shape}'
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_chw = img_rgb.transpose(2,0,1)               # (3,512,512) uint8

    p_torch = torch_infer(args.config, args.ckpt, img_chw)
    p_onnx  = onnx_infer(args.onnx, img_chw)

    diff = np.abs(p_torch - p_onnx)
    print(f'torch out: {p_torch.shape}, range [{p_torch.min():.4f}, {p_torch.max():.4f}]')
    print(f'onnx  out: {p_onnx.shape},  range [{p_onnx.min():.4f}, {p_onnx.max():.4f}]')
    print(f'max abs diff: {diff.max():.6e}')
    print(f'mean abs diff: {diff.mean():.6e}')

    # mask 一致性
    m_torch = (p_torch[0,0] > 0.5).astype(np.uint8)
    m_onnx  = (p_onnx[0,0]  > 0.5).astype(np.uint8)
    iou = (m_torch & m_onnx).sum() / max((m_torch | m_onnx).sum(), 1)
    print(f'binary mask IoU (torch vs onnx): {iou:.4f}')

    if diff.max() < 1e-3:
        print('[OK] ONNX 和 PyTorch 输出一致,可以拿去 ATC 转换了')
    else:
        print('[WARN] 差异偏大,检查导出过程')

if __name__ == '__main__':
    main()