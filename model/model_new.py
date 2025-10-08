import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import *


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, -1).transpose(1, 2)
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        x = x.transpose(1, 2).view(b, c, h, w)
        return x


class FFN(nn.Module):
    def __init__(self, dim, hidden=4):
        super().__init__()

        self.norm = LayerNorm(dim)

        self.fc1 = nn.Conv2d(dim, dim * hidden, 1)
        self.pos = nn.Conv2d(dim * hidden, dim * hidden, 3, padding=1, groups=dim * hidden)
        self.fc2 = nn.Conv2d(dim * hidden, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x


class LCAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)
        return x


# WLA+FFN
class WLTrans(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = LCAttention(dim)
        self.ffn = FFN(dim)
        self.w1 = nn.Parameter(5e-2 * torch.ones((dim, 1, 1)), requires_grad=True)
        self.w2 = nn.Parameter(5e-2 * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = self.w1 * self.attn(x) + x
        x = self.w2 * self.ffn(x) + x
        return x


class CAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=False, act='sigmoid'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        if act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'softmax':
            self.act = nn.Softmax(dim=-1)
        else:
            self.act = nn.ReLU()

    def forward(self, q, k, v):
        B, C, H, W = q.shape
        q = q.view(B, self.num_heads, C, H * W // self.num_heads)
        k = k.view(B, self.num_heads, C, H * W // self.num_heads)
        v = v.view(B, self.num_heads, C, H * W // self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = self.act(attn)

        out = attn @ v
        out = out.view(B, self.num_heads * (C // self.num_heads), H, W)
        return out


# GCA+FFN
class CATrans(nn.Module):
    def __init__(self, dim, num_head=4):
        super().__init__()
        self.dim = dim
        self.p2f = nn.Conv2d(1, self.dim, 1)
        self.q1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.kv1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.k2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim)
        )
        self.v2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim)
        )

        self.gca1 = CAttention(dim, num_heads=num_head)
        self.gca2 = CAttention(dim, num_heads=num_head)
        self.ffn = FFN(dim)

    def forward(self, x, R, y):
        x_rec = self.p2f(R(y))
        q1 = self.q1(x_rec)
        q2 = self.q2(x)
        k1, v1 = self.kv1(x).chunk(2, dim=1)
        ca_f1 = self.gca1(q1, k1, v1)
        k2, v2 = self.k2(ca_f1), self.v2(ca_f1)

        xa = self.gca2(q2, k2, v2) + x
        x = self.ffn(xa) + x

        return x


# DGD
class GRAD(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv2p = nn.Conv2d(dim, 1, kernel_size=1)
        self.conv2f = nn.Conv2d(1, dim, kernel_size=1)
        self.res = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )

    def forward(self, x, y, S, R):
        De = self.conv2f(
            R(
                y - S(self.conv2p(x))  # pixel error
            )
        )
        De = De + self.res(De)
        x = x + De
        return x


# TSSD
class DENO(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.GELU(),
        )

        self.res = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        self.conv_out = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)

    def up2(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear')

    def forward(self, x, x_as=None):
        # db
        x_down0 = x
        x_down1 = self.down1(x_down0) + x_as[2] if x_as else self.down1(x_down0)
        x_down2 = self.down2(x_down1) + x_as[1] if x_as else self.down2(x_down1)
        x_down3 = self.down3(x_down2) + x_as[0] if x_as else self.down3(x_down2)
        x_up0 = x_down3
        x_up1 = self.mix(self.up2(x_up0) + x_down2)
        x_up2 = self.mix(self.up2(x_up1) + x_down1)
        x_up3 = self.mix(self.up2(x_up2) + x_down0)

        x_out = x_up3 + self.res(x_up3)

        x = self.conv_out(torch.concat([x, x_out], dim=1))
        return x, [x_up0, x_up1, x_up2]


class GDB(nn.Module):
    def __init__(self, dim):
        super(GDB, self).__init__()

        self.grad = GRAD(dim)
        self.deno = DENO(dim)

    def forward(self, x, y, S, R, x_as_old):
        x = self.grad(x, y, S, R)
        x, x_as_new = self.deno(x, x_as_old)
        return x, x_as_new


class SModule(nn.Module):
    def __init__(self, patch, s_weight):
        super().__init__()
        self.patch = patch
        self.s_weight = s_weight

        self.res = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv2d(16, 8, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=7, padding=3),
            nn.GELU()
        )

    def forward(self, x):
        x = x + self.res(x)
        y = F.conv2d(x, self.s_weight, stride=self.patch)
        return y


class RModule(nn.Module):
    def __init__(self, patch, s_weight):
        super().__init__()
        self.patch = patch
        self.s_weight = s_weight

        self.res = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, y):
        x = F.conv_transpose2d(y, self.s_weight, stride=self.patch)
        x = x + self.res(x)
        return x


class CSDemo(nn.Module):
    def __init__(self, ratio, iter_num=10, model_dim=16, patch=32):
        super(CSDemo, self).__init__()

        self.model_dim = model_dim
        self.iter_num = iter_num
        self.patch = patch
        self.full_dim = patch ** 2
        self.cs_dim = int(ratio * self.full_dim)

        self.s_weight = nn.Parameter(
            kaiming_normal_(torch.Tensor(self.cs_dim, 1, self.patch, self.patch))
        )
        self.S = SModule(self.patch, self.s_weight)
        self.R = RModule(self.patch, self.s_weight)

        self.proj_in = nn.Conv2d(1, self.model_dim, kernel_size=3, padding=1)
        self.proj_out = nn.Conv2d(self.model_dim, 1, kernel_size=1)

        self.gdb = nn.ModuleList([GDB(model_dim) for _ in range(self.iter_num)])
        self.cat = nn.ModuleList([CATrans(model_dim) for _ in range(self.iter_num)])
        self.wlt = nn.ModuleList([WLTrans(model_dim) for _ in range(self.iter_num)])

    def forward(self, x):
        y = self.S(x)
        x_r = self.R(y)
        x = self.proj_in(x_r)

        x_as = None
        for i in range(self.iter_num):
            x, x_as = self.gdb[i](x, y, self.S, self.R, x_as)
            x = self.cat[i](x, self.R, y)
            x = self.wlt[i](x)

        x = self.proj_out(x)

        return x, x.clone()
