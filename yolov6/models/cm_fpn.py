import torch
import torch.nn as nn

# 激活函数的选择
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

# 初始化权重
def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True

def kaiming_init(module: nn.Module,
                 a: float = 0,
                 mode: str = 'fan_out',
                 nonlinearity: str = 'relu',
                 bias: float = 0,
                 distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        # 隐藏层的通道数：默认是输出通道数的一半
        hidden_channels = int(out_channels * expansion)

        # 是否选择使用深度可分离卷积:默认不用
        Conv = DWConv if depthwise else BaseConv

        # 通道的缩放
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)

        # 是否使用残差连接:只要输入输出通道数相同，默认是使用残差连接
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)

        # 创建ks大小为 5, 9, 13 的三个最大池化层,自动计算了padding保证了输出形状和输入相同
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        

        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes, pool='att', fusions=['channel_add'], ratio=8):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions

        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1) #context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 通道的缩放
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        
        # 通道的缩放
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    # 空间池化
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)#softmax操作
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, H * W] ---->  [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))

            # [N, C, H, W] * [N, C, 1, 1] --> [N, C, H, W]
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            # [N, C, H, W] + [N, C, H, W] --> [N, C, H, W]
            out = out + channel_add_term

        return out


class CM_FPN(nn.Module):
    """
    Channel-modulated FPN (CM-FPN) in paper
    """

    def __init__(
        self,
        depth=0.5,
        width=0.5,
        
        in_channels=[256, 512, 1024],
        out_channels=[64,160,256],
        depthwise=False,
       
        act="silu",
        low_fea=False,
    ):
        super().__init__()
        
        
        self.in_channels = in_channels
        self.low_fea=low_fea
        Conv = DWConv if depthwise else BaseConv


        self.SPP = SPPBottleneck(int(out_channels[2]), int(out_channels[2]))


        self.CSP_out0 = CSPLayer(int(out_channels[2]), int(in_channels[2] * width), shortcut=False)


        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")


        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )


        self.C3_p4 = CSPLayer(
            int(out_channels[1]+in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth), 
            False,
            depthwise=depthwise,
            act=act,
        )  # cat


        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )


        self.C3_p3 = CSPLayer(
            int(out_channels[0]+in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.gc1 = ContextBlock2d(int(out_channels[2]),int(out_channels[2]))
        self.gc2 = ContextBlock2d(int(out_channels[1]),int(out_channels[1]))
        self.gc3 = ContextBlock2d(int(out_channels[0]),int(out_channels[0]))


    def forward(self, input):
        """
        Args:
            inputs: Tuple[Tensor],backbone处理后的后三个阶段的特征.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        
        (x3, x2, x1, x0) = input

        out0=self.SPP(x0)  # 256
        out0 = self.gc1(out0) # 256
        out0=self.CSP_out0(out0)  #  1024

        out1 = self.lateral_conv0(out0) # 1024 -> 512
        out1 = self.upsample(out1)  # 512
        x1 = self.gc2(x1) # 160
        out1 = torch.cat([out1, x1], 1)   # 672
        out1 = self.C3_p4(out1) # 672 -> 512 

        out2 = self.reduce_conv1(out1)  # 512 -> 256
        out2 = self.upsample(out2)  
        x2 = self.gc3(x2) # 64
        out2 = torch.cat([out2, x2], 1)  # 320
        out2 = self.C3_p3(out2) # 320 -> 256

        outputs = [out2, out1, out0] 

        return outputs


