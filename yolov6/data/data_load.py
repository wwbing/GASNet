#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py

import os
import torch.distributed as dist
from torch.utils.data import dataloader, distributed

from .datasets import TrainValDataset
from yolov6.utils.events import LOGGER
from yolov6.utils.torch_utils import torch_distributed_zero_first


def create_dataloader(
    path,                       # 训练集的路径
    img_size,                   # 准备输入网络的图像大小
    batch_size,                 # 批大小
    stride,                     # 最大的步长，感受野
    hyp=None,                   # 图像增强的超参数
    augment=False,              # 是否进行图像增强
    check_images=False,         # 是否检查图像
    check_labels=False,         # 是否检查标签
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,                  # 工作进程数
    shuffle=False,
    data_dict=None,             # 数据集yaml文件的信息
    task="Train",
    specific_shape=False,
    height=1088,
    width=1920,
    cache_ram=False             # 是否缓存到内存
    ):
    """Create general dataloader.

    Returns dataloader and dataset
    """
    if rect and shuffle:
        LOGGER.warning(
            "WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False"
        )
        shuffle = False
    with torch_distributed_zero_first(rank):
        dataset = TrainValDataset(
            path,                                   # 训练集的路径
            img_size,                               # 准备输入网络的图像大小    
            batch_size,                             # 批大小
            augment=augment,                         # 是否进行图像增强
            hyp=hyp,                                 # 图像增强的超参数
            rect=rect,                               # 是否使用矩形图像
            check_images=check_images,              # 是否检查图像
            check_labels=check_labels,              # 是否检查标签
            stride=int(stride),                     # 最大的步长，感受野
            pad=pad,                                 # 填充
            rank=rank,                               
            data_dict=data_dict,                     # 数据集yaml文件的信息
            task=task,
            specific_shape = specific_shape,
            height=height,
            width=width,
            cache_ram=cache_ram                        
        )

    batch_size = min(batch_size, len(dataset))
    
    workers = min(
        [
            os.cpu_count() // int(os.getenv("WORLD_SIZE", 1)),
            batch_size if batch_size > 1 else 0,
            workers,
        ]
    )  # number of workers

    # in DDP mode, if GPU number is greater than 1, and set rect=True,
    # DistributedSampler will sample from start if the last samples cannot be assigned equally to each
    # GPU process, this might cause shape difference in one batch, such as (384,640,3) and (416,640,3)
    # will cause exception in collate function of torch.stack.
    # 在DDP模式下，如果GPU数量大于1，并且设置了rect=True，DistributedSampler将从开头进行采样，如果最后的样本无法均匀分配给每个GPU进程
    # 这可能导致一个批次中出现形状不一致的情况，比如(384,640,3)和(416,640,3)，这将导致torch.stack的拼接函数出现异常。

    # 避免异常
    drop_last = rect and dist.is_initialized() and dist.get_world_size() > 1
    
    sampler = (
        None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
    )
    
    return (
        TrainValDataLoader(
            dataset,                                            # TrainValDataset创建的dataset对象
            batch_size=batch_size,                              
            shuffle=shuffle and sampler is None,                 
            num_workers=workers,                               
            sampler=sampler,
            pin_memory=True,                                    # 父类DataLoader中使用的参数，加速读取
            collate_fn=TrainValDataset.collate_fn,              # 父类DataLoader中使用的参数，表示实现自定义的batch输出，这里是一个batch的数据转成张量进行输出
        ),
        dataset,
    )


class TrainValDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
