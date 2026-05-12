"""
FileListSegDataset
==================

文档里训练参数有 use_filelist / train_img_list / train_gt_list / val_img_list / val_gt_list，
每个 *_list 都是"列表文件的绝对路径，文件里每行一个样本的绝对路径"。

mmseg 自带的 BaseSegDataset 是按目录扫的（data_prefix + img_suffix），所以要扩一下。
做法：override load_data_list()，按外部 txt 给的列表直接构造 data_info 列表。
按 sample 名（不带后缀）配对 img 和 gt，行数不一致就报错。

注册到 mmseg 的 DATASETS：dataset_type='FileListSegDataset'
"""

from __future__ import annotations

import os
from typing import List, Optional

from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


def _read_list(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"列表文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


@DATASETS.register_module()
class FileListSegDataset(BaseSegDataset):
    """
    通过 *_list.txt 提供样本列表的语义分割 dataset。

    Args:
        img_list_file: 训练/验证影像列表 txt（每行一个绝对路径）
        gt_list_file:  对应标签列表 txt
        metainfo:      和 BaseSegDataset 一样
        pipeline:      和 BaseSegDataset 一样
        其余参数透传 BaseSegDataset，但 data_root / data_prefix / ann_file 一般不传
    """

    METAINFO = dict(classes=("background",), palette=[[0, 0, 0]])

    def __init__(
        self,
        img_list_file: str,
        gt_list_file: Optional[str] = None,
        pipeline=None,
        metainfo=None,
        reduce_zero_label: bool = False,
        ignore_index: int = 255,
        **kwargs,
    ) -> None:
        self._img_list_file = img_list_file
        self._gt_list_file = gt_list_file
        # 屏蔽掉父类那些"按目录扫"用的参数，全置空
        kwargs.pop("data_root", None)
        kwargs.pop("data_prefix", None)
        kwargs.pop("ann_file", None)
        super().__init__(
            pipeline=pipeline,
            metainfo=metainfo,
            data_root="",
            data_prefix=dict(img_path="", seg_map_path=""),
            ann_file="",
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        imgs = _read_list(self._img_list_file)
        if self._gt_list_file:
            gts = _read_list(self._gt_list_file)
            if len(imgs) != len(gts):
                raise ValueError(
                    f"图像与标签数量不一致: imgs={len(imgs)} vs gts={len(gts)} "
                    f"(检查 {self._img_list_file} 和 {self._gt_list_file})"
                )
        else:
            gts = [None] * len(imgs)

        data_list = []
        for img_path, gt_path in zip(imgs, gts):
            info = dict(
                img_path=img_path,
                seg_map_path=gt_path,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
            )
            data_list.append(info)
        return data_list
