import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sys
import os

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from yolov6.layers.common import ConvBNSiLU,RepVGGBlock


def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x

# low-fam
class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])
        
        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d
        
        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)
        
        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out
    

# low-ifm
low_IFM = nn.Sequential(
                ConvBNSiLU(in_channels=544,
                    out_channels=128, 
                    kernel_size=1, 
                    stride=1,
                    padding=0),
                *[RepVGGBlock(in_channels=128, out_channels=128) for _ in range(3)],
                ConvBNSiLU(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
        )

from mmcv.cnn import ConvModule

def get_avg_pool():
    if torch.onnx.is_in_onnx_export():
        avg_pool = onnx_AdaptiveAvgPool2d
    else:
        avg_pool = nn.functional.adaptive_avg_pool2d
    return avg_pool


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    
    def forward(self, x):
        return self.relu(x + 3) / 6

class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            activations=None,
            global_inp=None,
    ) -> None:
        super().__init__()
        self.norm_cfg = norm_cfg
        
        if not global_inp:
            global_inp = inp
        
        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()
    
    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H
        
        local_feat = self.local_embedding(x_l)
        
        global_act = self.global_act(x_g)
        global_feat = self.global_embedding(x_g)
        
        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])
            
            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)
        
        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        out = local_feat * sig_act + global_feat
        return out


Inject_bf = InjectionMultiSum_Auto_pool(64,64, norm_cfg=dict(type='SyncBN', requires_grad=True),
                                                     activations=nn.ReLU6)



if __name__ == '__main__':
    x3 = torch.randn(1, 64, 160, 160)
    x2 = torch.randn(1, 64, 80, 80)
    x1 = torch.randn(1, 160, 40, 40)
    x0 = torch.randn(1, 256, 20, 20)

    input = [x3, x2, x1, x0]

    # 提取全局特征的过程
    low_FAM = SimFusion_4in()
    Inject_bf = InjectionMultiSum_Auto_pool(64,64, norm_cfg=dict(type='SyncBN', requires_grad=True),activations=nn.ReLU6)

    low_align_feat = low_FAM(input)
    low_fuse_feat = low_IFM(low_align_feat)
    low_global_info = low_fuse_feat.split([64,64], dim=1)
    

    bf0 = torch.randn(1, 64, 40, 40)
    bf1 = torch.randn(1, 64, 80, 80)

    # 注入全局特征的过程
    bf0_out = Inject_bf(bf0, low_global_info[0])
    bf1_out = Inject_bf(bf1, low_global_info[1])

    print(bf0_out.shape)
    print(bf1_out.shape)

