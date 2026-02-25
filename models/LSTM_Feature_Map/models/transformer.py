import torch
import torch.nn as nn

from einops import rearrange, reduce, asnumpy, parse_shape,repeat
from models.mish import Mish


class Transformer(nn.Module):
    def __init__(self, num_class=6, embedding_dim=2048, nhead=4, num_layers=5, need_embedding=False, hidden_dim=512):
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.mlp = nn.Linear(embedding_dim,num_class)
        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 256),
            nn.Dropout(0.2),
            Mish(),
            nn.Linear(256, num_class),
            # nn.Sigmoid()
        )
        # self.init_weights()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

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

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer_encoder(x)
        # x = x.mean(dim=1)
        # x = x.view(x.size(0), -1)  # 将三维张量reshape成二维，然后直接通过全连接层将高维数据映射为classes
        # x = rearrange(x, 'b c l -> b (c l)')
        x= x[:, 0]
        # print(x.shape)
        x = self.mlp(x)
        return x


if __name__ == '__main__':
    model = Transformer(nhead=2, num_layers=2)
    input = torch.zeros((2, 2048, 50, 50))
    out = model(input)
    print(out.shape)
