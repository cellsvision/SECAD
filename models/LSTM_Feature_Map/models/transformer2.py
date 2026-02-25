import torch
import torch.nn as nn

from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer_Layer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Transformer(nn.Module):
    def __init__(self, num_class=6, embedding_dim=2048, nhead=4, num_layers=5, need_embedding=False, hidden_dim=256):
        super(Transformer, self).__init__()
        self.need_embedding = need_embedding
        if need_embedding:
            # self.embedding = nn.Embedding(num_embeddings=859, embedding_dim=embedding_dim, padding_idx=858)
            from models.FMC_0 import ChannelAttention, SpatialAttention
            self.conv0 = nn.Conv2d(embedding_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.relu = nn.ReLU(inplace=True)
            self.ca0 = ChannelAttention(hidden_dim)
            self.sa0 = SpatialAttention()
            embedding_dim = hidden_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.dropout = nn.Dropout(0.)

        self.transformer = Transformer_Layer(embedding_dim, num_layers, nhead, dim_head=64, mlp_dim=256, dropout=0.)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_class)
        )

    def init_weights(self) -> None:
        initrange = 0.1
        self.transformer_encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        if self.need_embedding:
            # input = self.embedding(input)  # [batch_size,seq_len,200]
            x = self.conv0(x)
            x = self.ca0(x) * x
            x = self.sa0(x) * x
        x = rearrange(x, 'b c h w -> b (h w) c')
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == '__main__':
    model = Transformer(nhead=2, num_layers=2)
    input = torch.zeros((2, 2048, 50, 50))
    out = model(input)
    print(out.shape)
