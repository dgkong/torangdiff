# referred https://huggingface.co/blog/annotated-diffusion

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, reduce


class ESMProj(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Linear(c_in, c_out)

    def forward(self, esm):
        return self.proj(esm)


class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim, t_max=400):
        super().__init__()
        self.dim = dim
        self.t_max = t_max
        half_dim = self.dim // 2
        scaling = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim) * -scaling)
        self.register_buffer('freqs', freqs)

    def forward(self, time):
        t_norm = time.float() / (self.t_max - 1)
        angles = t_norm[:, None] * self.freqs[None, :]
        embeddings = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv1d(nn.Conv1d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1", partial(torch.var, unbiased=False))
        weight_norm = (weight - mean) / (var + eps).sqrt()

        return F.conv1d(
            x,
            weight_norm,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv1d(c_in, c_out, 5, padding=2)
        self.norm = nn.GroupNorm(groups, c_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim, groups=8):
        super().__init__()
        self.block1 = ConvBlock(c_in, c_out, groups=groups)
        self.block2 = ConvBlock(c_out, c_out, groups=groups)
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(t_dim, c_out * 2))
        )
        self.res_conv = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, t_emb):
        t_emb = self.mlp(t_emb)
        t_emb = rearrange(t_emb, "b c -> b c 1")
        scale_shift = t_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class MultiheadAttention(nn.Module):
    def __init__(self, dim, n_head=4, head_dim=32):
        super().__init__()
        self.scale = head_dim**-0.5
        self.n_head = n_head
        hidden_dim = n_head * head_dim
        self.qkv_proj = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.out_proj = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1) # (B,L,d_k)
        q = rearrange(q, "b l (h d) -> b h l d", h=self.n_head)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.n_head)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.n_head)
        
        q = q * self.scale
        attn_scores = einsum(q, k, "... q d_k, ... k d_k -> ... q k")
        attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True).detach()

        attn = einsum(torch.softmax(attn_scores, -1), v, "... q k, ... k d_v -> ... q d_v")
        attn = rearrange(attn, "b h l d -> b (h d) l")
        out = self.out_proj(attn)
        return out


class Downsample1D(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.down = nn.Conv1d(dim*2, dim_out, 1)

    def forward(self, x):
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1))
        x = rearrange(x, "b c (l p) -> b (c p) l", p=2) # doubles channels and halves length to not lose information
        out = self.down(x)
        return out


class Upsample(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.conv = nn.Conv1d(dim, dim_out, 5, padding=2)

    def forward(self, x):
        x = self.up(x)
        out = self.conv(x)
        return out


class PreRMSNorm(nn.Module):
    def __init__(self, dim, fn, eps=1e-5):
        super().__init__()
        self.fn = fn
        self.norm = nn.RMSNorm(dim, eps)

    def forward(self, x):
        x = rearrange(x, "b c l -> b l c")
        x = self.norm(x)
        return self.fn(x)


class UNet1D(nn.Module):
    def __init__(self, c_in, c_cond, init_dim=128, dim_mults=(1,2,4), t_dim=128, resnet_block_groups=8):
        super().__init__()
        self.t_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(t_dim),
            nn.Linear(t_dim, t_dim*4), 
            nn.GELU(), 
            nn.Linear(t_dim*4, t_dim)
        )

        in_channels = c_in + c_cond
        self.init_conv = nn.Conv1d(in_channels, init_dim, 1)
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_in, t_dim, resnet_block_groups),
                        ResnetBlock(dim_in, dim_in, t_dim, resnet_block_groups),
                        PreRMSNorm(dim_in, MultiheadAttention(dim_in)),
                        Downsample1D(dim_in, dim_out) if not is_last
                        else nn.Conv1d(dim_in, dim_out, 5, padding=2),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, t_dim, resnet_block_groups)
        self.mid_attn = PreRMSNorm(mid_dim, MultiheadAttention(mid_dim))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, t_dim, resnet_block_groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out + dim_in, dim_out, t_dim, resnet_block_groups),
                        ResnetBlock(dim_out + dim_in, dim_out, t_dim, resnet_block_groups),
                        PreRMSNorm(dim_out, MultiheadAttention(dim_out)),
                        Upsample(dim_out, dim_in) if not is_last
                        else nn.Conv1d(dim_out, dim_in, 5, padding=2),
                    ]
                )
            )

        self.final_res = ResnetBlock(init_dim*2, init_dim, t_dim, resnet_block_groups)
        self.final_conv = nn.Conv1d(init_dim, c_in, 1)
        

    def forward(self, x_t, cond, t):
        seq_len = x_t.shape[-1]
        x = torch.cat((x_t, cond), dim=1)

        x = self.init_conv(x)
        h0 = x.clone()
        t = self.t_mlp(t)

        h = []
        for b1, b2, attn, downsample in self.downs:
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            h_skip = h.pop()
            if h_skip.shape[-1] != x.shape[-1]:
                h_skip = F.pad(h_skip, (0, x.shape[-1] - h_skip.shape[-1]))
            x = torch.cat((x, h_skip), dim=1)
            x = block1(x, t)

            h_skip = h.pop()
            if h_skip.shape[-1] != x.shape[-1]:
                h_skip = F.pad(h_skip, (0, x.shape[-1] - h_skip.shape[-1]))
            x = torch.cat((x, h_skip), dim=1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        if h0.shape[-1] != x.shape[-1]:
            h0 = F.pad(h0, (0, x.shape[-1] - h0.shape[-1]))
        x = torch.cat((x, h0), dim=1)

        x = self.final_res(x, t)
        out = self.final_conv(x)
        out = out[..., :seq_len]
        return out
