#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.layers.common import *
from yolov6.utils.torch_utils import initialize_weights
from yolov6.models.efficientrep import *
from yolov6.models.reppan import *
from yolov6.utils.events import LOGGER

# m_lvt模型
from yolov6.models.lvt_ds import *

# neck消融实验
from yolov6.models.cm_fpn import *
from yolov6.models.fpn import *
from yolov6.models.pafpn import *

def build_model(cfg, num_classes, device, fuse_ab=False, distill_ns=False):
    model = Model(cfg, channels=3, num_classes=num_classes, fuse_ab=fuse_ab, distill_ns=distill_ns).to(device)
    return model

class Model(nn.Module):
    export = False
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, config, channels=3, num_classes=None, fuse_ab=False, distill_ns=False):  # model, input channels, number of classes
        super().__init__()
        # 头网络的层数
        num_layers = config.model.head.num_layers

        # 根据配置文件构建网络
        if config.model.backbone.type in ["lvt_ds", "e_lvt_ds"]:
            if config.model.neck.type in["CM_FPN","FPN","PAFPN"]:
                self.backbone, self.neck, self.detect = lvt_neck_v6s(config, channels, num_classes, num_layers, fuse_ab=fuse_ab, distill_ns=distill_ns)
            else:
                self.backbone, self.neck, self.detect = lvt_v6s(config, channels, num_classes, num_layers, fuse_ab=fuse_ab, distill_ns=distill_ns)
        else:
            self.backbone, self.neck, self.detect = build_network(config, channels, num_classes, num_layers, fuse_ab=fuse_ab, distill_ns=distill_ns)



        # 每层特征对应的步长，感受野
        self.stride = self.detect.stride

        # 初始化cls_preds和reg_preds里面卷积的bias参数
        self.detect.initialize_biases()

        # 改变bn层的eps和momentum参数，并且把所有激活函数的inplace参数设为True，不会对Conv2d有任何的初始化
        initialize_weights(self)

    def forward(self, x):
        export_mode = torch.onnx.is_in_onnx_export() or self.export
        x = self.backbone(x)  # backbone输出的多层的特征
        x = self.neck(x)      # neck输出的多层特征图  s:64 128 256 
        if not export_mode:
            featmaps = []
            featmaps.extend(x)  # 保存neck输出的特征图
        x = self.detect(x)

        # 返回的x包括[[neck的三层特征], cls_score_list, reg_distri_list , reg_lrtb_list]    featmaps也是neck的三层特征
        return x if export_mode is True else [x, featmaps]

    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network(config, channels, num_classes, num_layers, fuse_ab=False, distill_ns=False):
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    fuse_P2 = config.model.backbone.get('fuse_P2')
    cspsppf = config.model.backbone.get('cspsppf')
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    block = get_block(config.training_mode) # training_mode,默认是conv_relu
    # 得到backbone和neck的函数名
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)

    if 'CSP' in config.model.backbone.type:

        if "stage_block_type" in config.model.backbone:
            stage_block_type = config.model.backbone.stage_block_type
        else:
            stage_block_type = "BepC3"  #default

        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.backbone.csp_e,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf,
            stage_block_type=stage_block_type
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.neck.csp_e,
            stage_block_type=stage_block_type
        )
    else:
        # 执行backbone和neck的函数，返回对应的模型
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

    if distill_ns:
        from yolov6.models.heads.effidehead_distill_ns import Detect, build_effidehead_layer
        if num_layers != 3:
            LOGGER.error('ERROR in: Distill mode not fit on n/s models with P6 head.\n')
            exit()
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    elif fuse_ab:
        from yolov6.models.heads.effidehead_fuseab import Detect, build_effidehead_layer
        anchors_init = config.model.head.anchors_init
        head_layers = build_effidehead_layer(channels_list, 3, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, anchors_init, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    else:
        from yolov6.models.effidehead import Detect, build_effidehead_layer
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return backbone, neck, head


def lvt_v6s(config, channels, num_classes, num_layers, fuse_ab=False, distill_ns=False):
    num_repeat = [1, 2, 4, 6, 2, 4, 4, 4, 4]
    
    channels_list = [0,64,64,160,256,
                     64,64,64,128,128,256]   # 最后一个维度试了一下，发现256效果好得多 192不行

    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max

    block = get_block(config.training_mode) # training_mode,默认是conv_relu

    # 得到backbone和neck的函数名
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)
    

    # 执行backbone和neck的函数，返回对应的模型
    if 'CSP' in config.model.neck.type:  # mbla模块

        if "stage_block_type" in config.model.neck:
            stage_block_type = config.model.neck.stage_block_type
        else:
            stage_block_type = "BepC3"  #default

        backbone = BACKBONE()

        checkpoint_path  = r"C:\Users\jiahao\Desktop\JIAHAO\paper_with_code\project\Gasnet\yolov6\models\lvt_imagenet.pth.tar"
        param_dict = torch.load(checkpoint_path)
        # 加载新的 state_dict 到模型中
        backbone.load_state_dict(param_dict, strict=False)
        
        
        if config.model.backbone.type == "e_lvt_ds":
            print("==================== e_lvt_ds load backbone imagent pretrained model successfully ====================")
        elif config.model.backbone.type == "lvt_ds":
            print("==================== lvt_ds load backbone imagent pretrained model successfully ====================")
        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.neck.csp_e,
            stage_block_type=stage_block_type
        )
    else:       # 非mbla模块
        backbone = BACKBONE()

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )


    if distill_ns:
        from yolov6.models.heads.effidehead_distill_ns import Detect, build_effidehead_layer
        if num_layers != 3:
            LOGGER.error('ERROR in: Distill mode not fit on n/s models with P6 head.\n')
            exit()
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    elif fuse_ab:
        from yolov6.models.heads.effidehead_fuseab import Detect, build_effidehead_layer
        anchors_init = config.model.head.anchors_init
        head_layers = build_effidehead_layer(channels_list, 3, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, anchors_init, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    else:
        from yolov6.models.effidehead import Detect, build_effidehead_layer
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return backbone, neck, head


def lvt_neck_v6s(config, channels, num_classes, num_layers, fuse_ab=False, distill_ns=False):
    
    if config.model.neck.type == "PAFPN":
        channels_list = [0,0,0,0,0,
                         0,64,0,160,0,256]
    else:
        channels_list = [0,0,0,0,0,
                         0,128,0,256,0,512]  

    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max

    # 得到backbone和neck的函数名
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)
    

    # 执行backbone和neck的函数，返回对应的模型
    
    backbone = BACKBONE()
    checkpoint_path  = r"C:\Users\jiahao\Desktop\JIAHAO\paper_with_code\project\Gasnet\yolov6\models\lvt_imagenet.pth.tar"
    param_dict = torch.load(checkpoint_path)

    # 加载新的 state_dict 到模型中
    backbone.load_state_dict(param_dict, strict=False)
         
    if config.model.backbone.type == "e_lvt_ds":
        print("==================== e_lvt_ds load backbone imagent pretrained model successfully ====================")
    elif config.model.backbone.type == "lvt_ds":
        print("==================== lvt_ds load backbone imagent pretrained model successfully ====================")
    
    neck = NECK()

    if distill_ns:
        from yolov6.models.heads.effidehead_distill_ns import Detect, build_effidehead_layer
        if num_layers != 3:
            LOGGER.error('ERROR in: Distill mode not fit on n/s models with P6 head.\n')
            exit()
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    elif fuse_ab:
        from yolov6.models.heads.effidehead_fuseab import Detect, build_effidehead_layer
        anchors_init = config.model.head.anchors_init
        head_layers = build_effidehead_layer(channels_list, 3, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, anchors_init, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    else:
        from yolov6.models.effidehead import Detect, build_effidehead_layer
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return backbone, neck, head
