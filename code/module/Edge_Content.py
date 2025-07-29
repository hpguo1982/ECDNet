# Realtive import
import sys

from apex.amp import amp

sys.path.append('../blocks')
import torch.nn.functional as F
import torch
from torch import nn
# from module.ResNet import ConvNextBlock
import math
from inspect import isfunction
# from torch.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class EMA():
    def __init__(self, beta):
        super(EMA, self).__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, context=None, *args, **kwargs):
        return self.fn(x, context, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb,self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

from einops.layers.torch import Rearrange
def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.norm(x)
        return self.fn(x,context)



# helpers functions
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.ModuleList):
            for j, k in m.named_children():
                for a, s in k.named_children():
                    if isinstance(s, nn.Conv2d):
                        nn.init.kaiming_normal_(s.weight, mode='fan_in', nonlinearity='relu')
                        if s.bias is not None:
                            nn.init.zeros_(s.bias)
                    elif isinstance(s, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                        nn.init.ones_(s.weight)
                        if s.bias is not None:
                            nn.init.zeros_(s.bias)

        else:
            pass

import copy

from einops import rearrange, reduce,repeat
from torch import nn, einsum


class GlobalAvgPool(nn.Module):
    def __init__(self, flatten=False):
        super(GlobalAvgPool, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)

class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.selu(x)

        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x, context=None):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class LinearCorssAttention(nn.Module):
    def __init__(self, dim, context_in=None, heads = 4, dim_head = 32):
        super(LinearCorssAttention, self).__init__()

        context_in = default(context_in, dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias = False)
        self.to_k = nn.Conv2d(context_in, hidden_dim, 1, bias = False)
        self.to_v = nn.Conv2d(context_in, hidden_dim, 1, bias = False)

        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, context=None, mask=None):
        b, c, h, w = x.shape
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # b, c, h, w = x.shape
        # qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), (q, k, v))
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        # sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        #
        # if exists(mask):
        #     mask = rearrange(mask, 'b ... -> b (...)')
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, 'b j -> (b h) () j', h=h)
        #     sim.masked_fill_(~mask, max_neg_value)
        #
        # # attention, what we cannot get enough of
        # attn = sim.softmax(dim=-1)
        # out = torch.einsum('b h d e, b h d n -> b h e n', attn, q)

        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class EdgeDetectionModule(nn.Module):
    def __init__(self, in_channels):
        super(EdgeDetectionModule, self).__init__()
        self.eem = nn.ModuleList([EdgeEnhancementModule(c) for c in in_channels])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, features):
        # features: [f1, f2, f3, f4] (从浅到深)
        f = [self.eem[i](features[i]) for i in range(4)]
        f4 = f[3]
        f3 = self.eem[2](f[2] + self.upsample(f4))
        f2 = self.eem[1](f[1] + self.upsample(f3))
        f1 = self.eem[0](f[0] + self.upsample(f2))
        return [f1, f2, f3, f4]
class EdgeEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(EEM, self).__init__()

        # Sobel filters (fixed, non-learnable)
        sobel_h = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_v = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_h', sobel_h)
        self.register_buffer('sobel_v', sobel_v)

        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        edge_h = F.conv2d(x.view(b * c, 1, h, w), self.sobel_h, padding=1)
        edge_v = F.conv2d(x.view(b * c, 1, h, w), self.sobel_v, padding=1)

        edge_hv = torch.sqrt(edge_h ** 2 + edge_v ** 2 + 1e-6)  # small constant for numerical stability
        edge_hv = edge_hv.view(b, c, h, w)

        edge_feat = self.bn_relu(edge_hv)
        f_grf = edge_feat * x  # Element-wise multiplication
        f_e = f_grf + x  # Element-wise addition

        return f_e


class ContentDetectionModule(nn.Module):
    def __init__(self, in_channels):
        super(ContentDetectionModule, self).__init__()
        self.cem = nn.ModuleList([ContentEnhancementModule(c) for c in in_channels])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, features):
        # features: [f1, f2, f3, f4] (从浅到深)
        f = [self.cem[i](features[i]) for i in range(4)]
        f4 = f[3]
        f3 = self.cem[2](f[2] + self.upsample(f4))
        f2 = self.cem[1](f[1] + self.upsample(f3))
        f1 = self.cem[0](f[0] + self.upsample(f2))
        return [f1, f2, f3, f4]


class ContentEnhancementModule(nn.Module):
    def __init__(self, in_c):
        super(ContentEnhancementModule, self).__init__()
        self.acb = ACB(in_c)
        self.cab = CAB(in_c)
        self.sab = SAB()
        self.out_conv = nn.Conv2d(in_c, in_c, kernel_size=1)

    def forward(self, x):
        f_d = self.acb(x)            # (a) ACB
        f_c = self.cab(f_d)          # (b) CAB
        f_s = self.sab(f_d)          # (c) SAB
        f_a = f_c * f_s              # 元素级乘法 f^a = f^c * f^s
        f_2 = f_a * f_d              # 再次乘上 f^d
        f_out = self.out_conv(f_2)   # 1x1卷积
        return x + f_out             # 残差连接

class ACB(nn.Module):
    def __init__(self, in_c):
        super(ACB, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_c, in_c, 3, padding=1, dilation=1), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(in_c, in_c, 3, padding=2, dilation=2), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(in_c, in_c, 3, padding=3, dilation=3), nn.ReLU(inplace=True))
        ])
        self.out_conv = nn.Conv2d(in_c * 3, in_c, kernel_size=1)

    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.convs], dim=1)
        return self.out_conv(out)

class CAB(nn.Module):
    def __init__(self, in_c):
        super(CAB, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_c, in_c // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_c // 2, in_c)
        )
        self.sigmoid = nn.Sigmoid()
        self.in_c = in_c

    def forward(self, x):
        b, c, _, _ = x.size()
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_out = self.mlp(max_pool)
        avg_out = self.mlp(avg_pool)
        out = self.sigmoid(max_out + avg_out).view(b, c, 1, 1)
        return out

class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        return self.sigmoid(self.conv(pool))



class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, context=None):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class ConvBlock(nn.Module):
    def  __init__(self, dim, dim_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        h = self.relu(h)
        return h




class DilConv(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(DilConv, self).__init__()

        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class ConditionExtractor(nn.Module):
    def __init__(
        self,
        dim,
        cp_condition_net,
        dim_mults=(1, 2, 4, 8),
        self_condition = True,
        with_time_emb = True,
        residual = False,
    ):
        super(ConditionExtractor, self).__init__()


        # determizne dimensions
        input_img_channels = 1
        mask_channels= 1
        self.input_img_channels = input_img_channels
        self.mask_channels = mask_channels
        self.self_condition = self_condition


        # init_dim = default(init_dim, dim)

        # self.cond_init_conv = nn.Conv2d(input_img_channels, init_dim, 7, padding = 3)

        output_channels = mask_channels
        mask_channels = mask_channels * (2 if self_condition else 1)
        self.init_conv = nn.Conv2d(mask_channels, dim, 7, padding = 3)
        self.init_conv_cond = nn.Conv2d(input_img_channels, dim, 7, padding = 3)

        self.channels = self.input_img_channels
        self.residual = residual
        dims_rgb = [dim, *map(lambda m: dim * m, dim_mults)]
        dims_mask = [dim, *map(lambda m: dim * m, dim_mults)]

        in_out_rgb = list(zip(dims_rgb[:-1], dims_rgb[1:]))
        in_out_mask = list(zip(dims_mask[:-1], dims_mask[1:]))
        from functools import partial
        block_klass = partial(ResnetBlock, groups = 8)


        full_self_attn: tuple = (False, False, False, True)

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        # attention related params

        attn_kwargs = dict(
            dim_head = 32,
            heads = 4
        )


        self.downs_input = nn.ModuleList([])
        self.downs_label_noise = nn.ModuleList([])

        self.side_out_for_content = nn.ModuleList([])
        self.side_out_for_boundary = nn.ModuleList([])


        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out_mask)

        in_out_mask = [(2,64),(64,128),(128,320),(320,512)]


        for ind, (dim_in, dim_out) in enumerate(in_out_mask):
            is_last = ind >= (num_resolutions - 1)


            self.side_out_for_content.append(nn.ModuleList([
                # nn.Conv2d(dim_in, 128, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(128),
                ConvBlock(dim_out, 64)
            ]))

            self.side_out_for_boundary.append(nn.ModuleList([
                # nn.Conv2d(dim_in, 128, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(128),
                ConvBlock(dim_out, 64)
            ]))

        from module.pvt_v2 import PyramidVisionTransformerV2, pvt_v2_b2
        backbone = pvt_v2_b2(True, cp_condition_net)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]
        self.decoder_content = EdgeDetectionModule((64, 128, 256, 512), (8, 16, 32, 64), out_c =64)
        self.decoder_boundary = ContentDetectionModule((64, 128, 256, 512), (8, 16, 32, 64), out_c =64)





    def normalization(channels):
        """
        Make a standard normalization layer.

        :param channels: number of input channels.
        :return: an nn.Module for normalization.
        """
        return GroupNorm32(32, channels)


    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid
    # def forward(self, input):
    def forward(self, input):

        B,C,H,W, = input.shape
        cond = input

        pyramid = self.get_pyramid(cond)

        if self.residual:
            orig_x = input

        side_out_edge = []
        for i, ModuleList in enumerate(self.side_out_for_content):
            for conv in ModuleList:
                side = conv(pyramid[i])
                side_out_edge.append(side)

        side_out_detail = []
        for i, ModuleList in enumerate(self.side_out_for_boundary):
            for conv in ModuleList:
                side = conv(pyramid[i])
                side_out_detail.append(side)

        edge_side = [tensor.clone() for tensor in side_out_edge]
        # edge_side = edge_side[::-1]

        detail_side = [tensor.clone() for tensor in side_out_detail]
        # detail_side = detail_side[::-1]

        _, out_edge = self.decoder_content(edge_side)
        _, out_detail = self.decoder_boundary(detail_side)

        out_edge_detail = self.content_boundary(out_edge, out_detail)
        return out_edge, out_detail, out_edge_detail, pyramid[-1]


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    torch.cuda.set_device(1)
    model = ConditionExtractor(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        with_time_emb=True,
        residual=True
    ).cuda()
    input_R = torch.randn(1,1,256,256).cuda()
    label_noise_t = torch.randn(1,1,256,256).cuda()
    time = torch.randn(2).cuda()
    X=model(input_R)