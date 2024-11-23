#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import math

import torch
import torch.nn as nn

from yolov6.utils.events import LOGGER


def build_optimizer(cfg, model):
    """ Build optimizer from cfg file."""
    g_bnw, g_w, g_b = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_w.append(v.weight)

    assert cfg.solver.optim in ['SGD', 'Adam', 'AdamW'], 'ERROR: unknown optimizer, use SGD defaulted'

    # 把bn层的参数加入优化器
    if cfg.solver.optim == 'SGD':
        optimizer = torch.optim.SGD(g_bnw, lr=cfg.solver.lr0, momentum=cfg.solver.momentum, nesterov=True)
    elif cfg.solver.optim == 'Adam':
        optimizer = torch.optim.Adam(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))
    elif cfg.solver.optim == 'AdamW':
        optimizer = torch.optim.AdamW(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))

    # 把其他层的参数加进去
    optimizer.add_param_group({'params': g_w, 'weight_decay': cfg.solver.weight_decay})
    optimizer.add_param_group({'params': g_b})

    del g_bnw, g_w, g_b
    return optimizer


# 
def build_lr_scheduler(cfg, optimizer, epochs):
    """Build learning rate scheduler from cfg file."""
    if cfg.solver.lr_scheduler == 'Cosine':
        # lf 函数接受一个参数 x，这个参数表示当前的训练周期数，函数的返回值是一个介于 cfg.solver.lrf 和 1 之间的数
        # 当 x 从 0 增加到 epochs 时，lf(x) 的值会从 1 逐渐降低到 cfg.solver.lrf。这就实现了余弦退火的效果.    热身阶段的学习率变化在update_optimizer实现
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg.solver.lrf - 1) + 1       
    elif cfg.solver.lr_scheduler == 'Constant':
        lf = lambda x: 1.0
    else:
        LOGGER.error('unknown lr scheduler, use Cosine defaulted')

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler, lf




