import math
from argparse import Namespace
from typing import Any, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import Conv2d
from torch.nn import init as init
from torch.nn.functional import pad
from torch.nn.parameter import Parameter


class KAT(nn.Module):
    def __init__(self, args: Namespace):
        super(KAT, self).__init__()

        n_blocks = args.n_blocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale

        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        m_head = [conv(3, n_feats, kernel_size)]

        m_body: list[nn.Module] = [
            KABlock(
                n_feats, kernel_size, res_scale=args.res_scale, legacy=args.legacy
            ) for _ in range(n_blocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        m_tail = [
            Upsampler(scale, n_feats, act='none'),
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # type: ignore
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict: Any, strict: bool=True): # type: ignore
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, Parameter):
                    param = param.data
                own_state[name].copy_(param)
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))



def conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class KAM(nn.Module):
    def __init__(self, n_feat: int, kernel_size: int=3, bias: bool=False, legacy: bool=False):
        super().__init__()
        self.kernel_size = kernel_size

        self.qk = Conv2d(n_feat, n_feat * 2, kernel_size=3, bias=False, stride=3)

        self.weight = Parameter(
            torch.ones(n_feat, n_feat, kernel_size, kernel_size))
        init.kaiming_normal_(self.weight)
        self.weight.data *= 0.1
        if bias:
            self.bias = Parameter(torch.zeros(n_feat, n_feat))
        else:
            self.bias = None

        self.temperature = Parameter(torch.zeros(1, 1, 1))
        self.pre_normalize = True
        self.relu = nn.ReLU(inplace=True)
        self.legacy = legacy

    def xtx(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
            q: B, 1, C, H, W
            k: B, 1, C, H, W
            return: B, K^2C, K^2C
        """

        K = self.kernel_size

        xtx = cal_xtx(q, k, K)  # N, C_in, C_in, d_size, d_size

        return xtx

    def forward(self, x: torch.Tensor): # type: ignore
        B, C, _, _ = x.shape

        qk = self.qk(x)
        q, k = qk.chunk(2, dim=1)

        if self.pre_normalize:
            _, _, H2, W2 = q.shape

            q = rearrange(q, 'B C H W -> B (C H W)')
            k = rearrange(k, 'B C H W -> B (C H W)')

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            q = rearrange(q, 'B (C H W) -> B C H W', H=H2, W=W2)
            k = rearrange(k, 'B (C H W) -> B C H W', H=H2, W=W2)

        attn = self.xtx(q.unsqueeze(1),
                        k.unsqueeze(1))  # B, K^2C, K^2C

        attn = attn.view(-1, C, attn.size(-2), attn.size(-1))
        atten_kernel = conv2d(attn, self.weight,
                              sample_wise=False)
        atten_kernel = atten_kernel.view(B, C, C, self.kernel_size,
                                         self.kernel_size).permute(
                                             0, 2, 1, 3, 4).contiguous()

        if self.legacy:
            viewed_atten_kernel = atten_kernel.view(atten_kernel.size(0), -1)
            atten_kernel = ((atten_kernel - viewed_atten_kernel.mean(dim=1).view(-1, 1, 1, 1, 1)) / 
                            viewed_atten_kernel.std(dim=1).view(-1, 1, 1, 1, 1))
        else:
            atten_kernel = rearrange(atten_kernel, 'N B C D1 D2 -> N B (C D1 D2)')
            atten_kernel = torch.nn.functional.normalize(atten_kernel, dim=-1)
            atten_kernel = rearrange(atten_kernel, 'N B (C D1 D2) -> N B C D1 D2', D1=self.kernel_size, D2=self.kernel_size)

        atten_kernel = self.weight.unsqueeze(0).repeat(
            (x.size(0), 1, 1, 1,
            1)) + atten_kernel * self.temperature

        out = conv2d(x,
                     atten_kernel,
                     padding=self.kernel_size // 2,
                     sample_wise=True)

        out = F.relu(out)

        return out


def cal_xtx(x1: Any, x2: Any, d_size: int):
    """
        x: N, 1, C_in, H, W
        d_size: kernel (d) size
    """
    padding = d_size - 1
    xtx = conv3d(x1,
                 x2.view(x2.size(0), x2.size(2), 1, 1, x2.size(3), x2.size(4)),
                 padding,
                 sample_wise=True)

    return xtx


def conv2d(input: Any,
           weight: Any,
           padding: Union[int, List[int]] = 0,
           sample_wise: bool = False) -> torch.Tensor:
    """
        sample_wise=False, normal conv2d:
            input - (N, C_in, H_in, W_in)
            weight - (C_out, C_in, H_k, W_k)
        sample_wise=True, sample-wise conv2d:
            input - (N, C_in, H_in, W_in)
            weight - (N, C_out, C_in, H_k, W_k)
    """
    if isinstance(padding, int):
        padding = [padding] * 4
    if sample_wise:
        # input - (N, C_in, H_in, W_in) -> (1, N * C_in, H_in, W_in)
        input_sw = input.view(1,
                              input.size(0) * input.size(1), input.size(2),
                              input.size(3))

        # weight - (N, C_out, C_in, H_k, W_k) -> (N * C_out, C_in, H_k, W_k)
        weight_sw = weight.view(
            weight.size(0) * weight.size(1), weight.size(2), weight.size(3),
            weight.size(4))

        # group-wise convolution, group_size==batch_size
        out = F.conv2d(pad(input_sw, padding),
                       weight_sw,
                       groups=input.size(0))
        out = out.view(input.size(0), weight.size(1), out.size(2), out.size(3))
    else:
        out = F.conv2d(pad(input, padding), weight)
    return out


def conv3d(x: Any,
           weight: Any,
           padding: Union[int, List[int]] = 0,
           sample_wise: bool = False) -> torch.Tensor:
    """
        sample_wise=False, normal conv3d:
            x - (N, C_in, D_in, H_in, W_in)
            weight - (C_out, C_in, D_k, H_k, W_k)
        sample_wise=True, sample-wise conv3d:
            x - (N, C_in, D_in, H_in, W_in)
            weight - (N, C_out, C_in, D_k, H_k, W_k)
    """
    if isinstance(padding, int):
        padding = [padding] * 4 + [0, 0]
    if sample_wise:
        # x - (N, C_in, D_in, H_in, W_in) -> (1, N * C_in, D_in, H_in, W_in)
        input_sw = x.view(1,
                              x.size(0) * x.size(1), x.size(2),
                              x.size(3), x.size(4))

        # weight - (N, C_out, C_in, D_k, H_k, W_k) -> (N * C_out, C_in, D_k, H_k, W_k)
        weight_sw = weight.reshape(
            weight.size(0) * weight.size(1), weight.size(2), weight.size(3),
            weight.size(4), weight.size(5))
        # group-wise Sconvolution, group_size==batch_size
        input_sw = pad(input_sw, padding)
        out = F.conv3d(input_sw, weight_sw, groups=x.size(0))
        out = out.view(x.size(0), weight.size(1), out.size(2), out.size(3),
                       out.size(4))
    else:
        out = F.conv3d(pad(x, padding), weight)
    return out

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range: int,
        rgb_mean: Tuple[float, float, float]=(0.4488, 0.4371, 0.4040), rgb_std: Tuple[float, float, float]=(1.0, 1.0, 1.0), sign: float=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class KABlock(nn.Module):
    def __init__(
        self, n_feats: int, kernel_size: int, bias: bool=True, bn: str='none', res_scale: float=1, legacy: bool=False):

        super(KABlock, self).__init__()
        m = []
        m.append(KAM(n_feats, kernel_size, bias=bias, legacy=legacy))
        m.append(nn.ReLU(True))
        if bn == 'bn':
            m.append(nn.BatchNorm2d(n_feats))
        elif bn == 'ln':
            m.append(LayerNorm(n_feats))
        m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
        if bn == 'bn':
            m.append(nn.BatchNorm2d(n_feats))
        elif bn == 'ln':
            m.append(LayerNorm(n_feats))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor): # type: ignore
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, scale: int, n_feats: int, bn: str='none', act: str='none', bias: bool=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn == 'bn':
                    m.append(nn.BatchNorm2d(n_feats))
                elif bn == 'ln':
                    m.append(LayerNorm(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
                elif act == 'gelu':
                    m.append(nn.GELU())

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn == 'bn':
                m.append(nn.BatchNorm2d(n_feats))
            elif bn == 'ln':
                m.append(LayerNorm(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
            elif act == 'gelu':
                m.append(nn.GELU())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

def to_3d(x: Any):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x: Any, h: int, w: int):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):

    def __init__(self, dim: int):
        super(BiasFree_LayerNorm, self).__init__()
        normalized_shape = (dim, )
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor): # type: ignore
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):

    def __init__(self, dim: int):
        super(WithBias_LayerNorm, self).__init__()
        normalized_shape = (dim, )
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor): # type: ignore
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):

    def __init__(self, dim: int, bias: bool=False):
        super(LayerNorm, self).__init__()
        if not bias:
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x: torch.Tensor): # type: ignore
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
