import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

# ===============================lvt原版部分
class ds_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, 
                 dilation=[1,3,5], groups=1, bias=True, 
                 act_layer='nn.SiLU(True)', init='kaiming'):
        super().__init__()
        assert in_planes % groups == 0  # 输入通道数必须是组数的整数倍
        assert kernel_size == 3 # 目前只支持卷积核大小为3
        
        # 初始化卷积层的参数
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.with_bias = bias
        
        # 使用正态分布张量初始化卷积权重
        # 卷积对应out_planes个卷积，每个卷积有 (输入通道数//分组数)个通道,每个卷积的形状k*k
        self.weight = nn.Parameter(torch.randn(out_planes, in_planes // groups, kernel_size, kernel_size), requires_grad=True)
        
        if bias:
            # 使用零初始化偏置项
            self.bias = nn.Parameter(torch.Tensor(out_planes))
        else:
            self.bias = None
            
        # 激活函数
        self.act = eval(act_layer)
        
        self.init = init
        self._initialize_weights()

    def _initialize_weights(self):
        # 进一步初始化卷积权重和偏置项的方法
        if self.init == 'dirac':
            nn.init.dirac_(self.weight, self.groups)
        elif self.init == 'kaiming':
            nn.init.kaiming_uniform_(self.weight)
        else:
            raise NotImplementedError
            
        if self.with_bias:
            if self.init == 'dirac':
                nn.init.constant_(self.bias, 0.)
            elif self.init == 'kaiming':
                # 根据 kaiming 初始化偏置项
                bound = self.groups / (self.kernel_size**2 * self.in_planes)
                bound = math.sqrt(bound)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                raise NotImplementedError

    def forward(self, x):
        output = 0
        # 递归进行膨胀卷积操作:残差连接
        for dil in self.dilation:
            output += self.act(
                F.conv2d(
                    x, weight=self.weight, bias=self.bias, stride=self.stride, padding=dil,
                    dilation=dil, groups=self.groups,
                )
            )
        return output

class CSA(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, kernel_size=3, padding=1, stride=2,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 多头注意力机制的头数和每个头的维度
        self.num_heads = num_heads
        head_dim = out_dim // num_heads
        
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        # 注意力缩放因子
        self.scale = qk_scale or head_dim**-0.5
        
        # 多头注意力机制的线性映射
        self.attn = nn.Linear(in_dim, kernel_size**4 * num_heads)
        self.attn_drop = nn.Dropout(attn_drop)

        # Unfold 操作用于提取图像块，AvgPool2d 用于下采样
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        
        # CSA（Channel-wise Spatial Attention）卷积的参数
        self.csa_group = 1
        assert out_dim % self.csa_group == 0
        self.weight = nn.Conv2d(
            self.kernel_size*self.kernel_size*out_dim, 
            self.kernel_size*self.kernel_size*out_dim, 
            1, 
            stride=1, padding=0, dilation=1, 
            groups=self.kernel_size*self.kernel_size*self.csa_group, 
            bias=qkv_bias,
        )
        assert qkv_bias == False
        
        # 初始化 CSA 卷积的权重
        fan_out = self.kernel_size*self.kernel_size*self.out_dim
        fan_out //= self.csa_group
        self.weight.weight.data.normal_(0, math.sqrt(2.0 / fan_out)) # 初始化
        
        # 最终投影的线性层和 dropout
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, v=None):
        # 获取输入图像的维度
        B, H, W, _ = x.shape
        # 计算 CSA 操作后的图像高度和宽度
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        
        # 对输入进行下采样，并进行多头注意力计算
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
            self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4) # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 对输入进行转置，然后使用 Unfold 操作提取图像块
        v = x.permute(0, 3, 1, 2) # B,C,H, W
        v = self.unfold(v).reshape(
            B, self.out_dim, self.kernel_size*self.kernel_size, h*w
        ).permute(0,3,2,1).reshape(B*h*w, self.kernel_size*self.kernel_size*self.out_dim, 1, 1)
        
        # 使用 CSA 卷积处理图像块
        v = self.weight(v)
        v = v.reshape(B, h*w, self.kernel_size*self.kernel_size, self.num_heads, 
                      self.out_dim//self.num_heads).permute(0,3,1,2,4).contiguous() # B,H,N,kxk,C/H
        
        # 使用多头注意力权重对处理后的图像块进行加权求和
        x = (attn @ v).permute(0, 1, 4, 3, 2)
        x = x.reshape(B, self.out_dim * self.kernel_size * self.kernel_size, h * w)
        
        # 使用 fold 操作还原图像大小
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, 
                   padding=self.padding, stride=self.stride)

        # 最终的投影操作和 dropout
        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0., with_depconv=True):
        super().__init__()
        
        # 如果未指定隐藏特征和输出特征，则默认与输入特征相同
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 是否使用深度可分离卷积（depthwise separable convolution）
        self.with_depconv = with_depconv
        
        if self.with_depconv:
            # 使用深度可分离卷积的情况
            
            # 第一个卷积层，将输入特征映射到隐藏特征
            self.fc1 = nn.Conv2d(
                in_features, hidden_features, 1, stride=1, padding=0, dilation=1, 
                groups=1, bias=True,
            )
            
            # 深度可分离卷积层，增加非线性变换
            self.depconv = nn.Conv2d(
                hidden_features, hidden_features, 3, stride=1, padding=1, dilation=1, 
                groups=hidden_features, bias=True,
            )
            
            # 激活函数
            self.act = act_layer()
            
            # 第二个卷积层，将隐藏特征映射到输出特征
            self.fc2 = nn.Conv2d(
                hidden_features, out_features, 1, stride=1, padding=0, dilation=1, 
                groups=1, bias=True,
            )
        else:
            # 不使用深度可分离卷积的情况
            
            # 第一个全连接层，将输入特征映射到隐藏特征
            self.fc1 = nn.Linear(in_features, hidden_features)
            
            # 激活函数
            self.act = act_layer()
            
            # 第二个全连接层，将隐藏特征映射到输出特征
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        if self.with_depconv: # 使用深度可分离卷积的情况

            # 调整输入张量的维度顺序
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.fc1(x) # 第一个卷积层
            x = self.depconv(x) # 深度可分离卷积层
            x = self.act(x) # 非线性变换
            x = self.drop(x)
            x = self.fc2(x) # 第二个卷积层
            x = self.drop(x)
            
            # 调整张量维度顺序，返回结果
            x = x.permute(0, 2, 3, 1).contiguous()
            
            return x
        else: # 不使用深度可分离卷积的情况
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            
            return x
    
class Attention(nn.Module):
    def __init__(
        self, 
        dim, num_heads=8, qkv_bias=False, 
        qk_scale=None, attn_drop=0., 
        proj_drop=0., 
        rasa_cfg=None, sr_ratio=1, 
        linear=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.linear = linear
        self.rasa_cfg = rasa_cfg
        self.use_rasa = rasa_cfg is not None
        self.sr_ratio = sr_ratio
        
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        
        if self.use_rasa:
            if self.rasa_cfg.atrous_rates is not None:
                self.ds = ds_conv2d(
                    dim, dim, kernel_size=3, stride=1, 
                    dilation=self.rasa_cfg.atrous_rates, groups=dim, bias=qkv_bias, 
                    act_layer=self.rasa_cfg.act_layer, init=self.rasa_cfg.init,
                )
            if self.rasa_cfg.r_num > 1:
                self.silu = nn.SiLU(True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _inner_attention(self, x):
        B, H, W, C = x.shape
        q = self.q(x).reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.use_rasa:
            if self.rasa_cfg.atrous_rates is not None:
                q = q.permute(0,1,3,2).reshape(B, self.dim, H, W).contiguous()
                q = self.ds(q)
                q = q.reshape(B, self.num_heads, self.dim//self.num_heads, H*W).permute(0,1,3,2).contiguous()
        
        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0,3,1,2)
                x_ = self.sr(x_).permute(0,2,3,1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
              raise NotImplementedError
        
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
                
    def forward(self, x):
        if self.use_rasa:
            x_in = x
            x = self._inner_attention(x)
            if self.rasa_cfg.r_num > 1:
                x = self.silu(x)
            for _ in range(self.rasa_cfg.r_num-1):
                x = x + x_in
                x_in = x
                x = self._inner_attention(x)
                x = self.silu(x)
        else:
            x = self._inner_attention(x)
        return x

class Transformer_block(nn.Module):
    def __init__(self, dim,
                 num_heads=1, mlp_ratio=3., attn_drop=0.,
                 drop_path=0., sa_layer='sa', rasa_cfg=None, sr_ratio=1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 qkv_bias=False, qk_scale=None, with_depconv=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if sa_layer == 'csa':
            self.attn = CSA(
                dim, dim, num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop)
        elif sa_layer in ['rasa', 'sa']:
            self.attn = Attention(
                dim, num_heads=num_heads, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                attn_drop=attn_drop, rasa_cfg=rasa_cfg, sr_ratio=sr_ratio)
        else:
            raise NotImplementedError
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            with_depconv=with_depconv)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        if self.patch_size[0] == 7:
            x = self.proj(x)
            x = x.permute(0,2,3,1)
            x = self.norm(x)
        else:
            x = x.permute(0,3,1,2)
            x = self.proj(x)
            x = x.permute(0,2,3,1)
            x = self.norm(x)
        return x


#===============================下采样部分
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module): # (B, C, H, W) -> (B, C, H, W) 不改变形状
    # 标准卷积
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # 激活函数的选择，默认是SiLU激活函数
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


    
class Downsampling_Stem(nn.Module): # (B, C_in, H, W) -> (B, H/4, W/4, C_out)
    # Stem
    def __init__(self, c_in, c_out, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Downsampling_Stem, self).__init__()
        
        c_hid = int(c_out/2)  # hidden channels
        self.cv1 = Conv(c1=c_in, c2=c_hid, k=3, s=2) # 会减半宽高,因为s=2.注意赋值，没有指定args=xxx就是按顺序
        self.cv2 = Conv(c_hid, c_hid, 1, 1)
        self.cv3 = Conv(c_hid, c_hid, 3, 2) # 会减半宽高,因为s=2
        self.pool = torch.nn.AvgPool2d(2, stride=2) # 会减半宽高,因为s=2
        self.cv4 = Conv(2 * c_hid, c_out, 1, 1)

    def forward(self, x):
        x = self.cv1(x) 
        # cv1-cv2-cv3:(b, c_in, h/2, w/2) -> (b, c_hid, h/2, w/2) -> (b, c_hid, h/2, w/2) ->(b, c_hid, h/4, w/4)
        # cv1-pool:(b, c_in, h/2, w/2) -> (b, c_hid, h/4, w/4)

        # 再将cv3和pool的结果按照通道维度拼接,再进行cv4：(b, c_hid*2, h/4, w/4) -> (b, c_out, h/4, w/4)
        # 最后变换维度顺序:(b, c_out, h/4, w/4) -> (b, h/4, w/4, c_out)
        return self.cv4(torch.cat((self.cv3(self.cv2(x)), self.pool(x)), dim=1)).permute(0,2,3,1)





class lite_vision_transformer(nn.Module):
    
    def __init__(self, layers, in_chans=3, num_classes=1000, patch_size=4,
                 embed_dims=None, num_heads=None,
                 sa_layers=['csa', 'rasa', 'rasa', 'rasa'], rasa_cfg=None,
                 mlp_ratios=None, mlp_depconv=None, sr_ratios=[1,1,1,1], 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0., norm_layer=nn.LayerNorm, with_cls_head=True):

        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_depconv = mlp_depconv
        self.sr_ratios = sr_ratios
        self.layers = layers
        self.num_classes = num_classes
        self.sa_layers = sa_layers
        self.rasa_cfg = rasa_cfg
        self.with_cls_head = with_cls_head # set false for downstream tasks
        
        network = []
        for stage_idx in range(len(layers)):
            if stage_idx>0:
                _patch_embed = OverlapPatchEmbed(
                    patch_size=7 if stage_idx == 0 else 3,
                    stride=4 if stage_idx == 0 else 2,
                    in_chans=in_chans if stage_idx == 0 else embed_dims[stage_idx - 1],
                    embed_dim=embed_dims[0] if stage_idx == 0 else embed_dims[stage_idx],
                )
            else:
                _patch_embed=Downsampling_Stem(3,embed_dims[0])
            
            _blocks = []
            for block_idx in range(layers[stage_idx]):
                block_dpr = drop_path_rate * (block_idx + sum(layers[:stage_idx])) / (sum(layers) - 1)
                _blocks.append(Transformer_block(
                    embed_dims[stage_idx], 
                    num_heads=num_heads[stage_idx], 
                    mlp_ratio=mlp_ratios[stage_idx],
                    sa_layer=sa_layers[stage_idx],
                    rasa_cfg=self.rasa_cfg if sa_layers[stage_idx] == 'rasa' else None, # I am here
                    sr_ratio=sr_ratios[stage_idx],
                    qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    attn_drop=attn_drop_rate, drop_path=block_dpr,
                    with_depconv=mlp_depconv[stage_idx]))
            _blocks = nn.Sequential(*_blocks)
            
            network.append(nn.Sequential(
                _patch_embed, 
                _blocks
            ))
        
        # backbone
        self.backbone = nn.ModuleList(network)

        # classification head
        if self.with_cls_head:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.downstream_norms = nn.ModuleList([norm_layer(embed_dims[idx]) 
                                                   for idx in range(len(embed_dims))])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.with_cls_head:
            for idx, stage in enumerate(self.backbone):
                x = stage(x)
            x = self.norm(x)
            x = self.head(x.mean(dim=(1,2)))
            return x
        else:
            outs = []
            for idx, stage in enumerate(self.backbone):
                x = stage(x)
                x = self.downstream_norms[idx](x)
                outs.append(x.permute(0,3,1,2).contiguous())
            return outs


rasa_cfg = dict(
        atrous_rates= [1,3,5,7], # None, [1,3,5]
        act_layer= 'nn.SiLU(True)',
        init= 'kaiming',
        r_num = 2,
    ),

class lvt_ds(lite_vision_transformer):
    def __init__(self, rasa_cfg=None, with_cls_head=False, **kwargs):
        super().__init__(
            layers=[2,2,2,2],
            patch_size=4,
            embed_dims=[64,64,160,256],
            num_heads=[2,2,5,8],
            mlp_ratios=[4,8,4,4],
            mlp_depconv=[False, True, True, True],
            sr_ratios=[8,4,2,1],
            sa_layers=['csa', 'rasa', 'rasa', 'rasa'],
            rasa_cfg=rasa_cfg,
            with_cls_head=with_cls_head,
        )

class e_lvt_ds(lite_vision_transformer):
    def __init__(self, rasa_cfg=None, with_cls_head=False, **kwargs):
        super().__init__(
            layers=[1,1,2,2],
            patch_size=4,
            embed_dims=[64,64,160,256],
            num_heads=[2,2,5,8],
            mlp_ratios=[4,8,4,4],
            mlp_depconv=[False, True, True, True],
            sr_ratios=[8,4,2,1],
            sa_layers=['csa', 'rasa', 'rasa', 'rasa'],
            rasa_cfg=rasa_cfg,
            with_cls_head=with_cls_head,
        )

if __name__ == '__main__':
    input = torch.randn(1,3,640,640)
    model = e_lvt_ds()

    from thop import profile
    import time

    flop, para = profile(model, inputs=(input, ))  # 必须加上逗号，否者会报错
    print('Flops:',"%.2fG" % (flop/1e9), 'Params:',"%.2fM" % (para/1e6))
    start_time = time.time()
    outs = model(input)
    end_time = time.time()
    print("Time:", end_time - start_time)
    print(outs[0].shape,outs[1].shape,outs[2].shape,outs[3].shape)   