import torch.nn as nn
import torch
import torch.nn.functional as F
from models.mish import Mish
from einops import rearrange, reduce, asnumpy, parse_shape


class LSTM(nn.Module):
    def __init__(self, num_class=17, need_embedding=False, embedding_dim=128, hidden_dim=512):
        super(LSTM, self).__init__()
        if need_embedding:
            # self.embedding = nn.Embedding(num_embeddings=859, embedding_dim=embedding_dim, padding_idx=858)
            from models.FMC_0 import ChannelAttention, SpatialAttention
            self.conv0 = nn.Conv2d(embedding_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.relu = nn.ReLU(inplace=True)
            self.ca0 = ChannelAttention(hidden_dim)
            self.sa0 = SpatialAttention()
            embedding_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=128, num_layers=2, batch_first=True,
                            bidirectional=True,
                            dropout=0.5)
        # self.fc1 = nn.Linear(128 * 2, 128)
        # self.fc2 = nn.Linear(128, num_class)
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128 * 2, 256),
            nn.Dropout(0.2),
            Mish(),
            nn.Linear(256, num_class),
            # nn.Sigmoid()
        )
        self.need_embedding = need_embedding

    def forward(self, input):
        '''
        :param input:
        :return:
        '''
        assert input.size(0)==1
        if self.need_embedding:
            # input = self.embedding(input)  # [batch_size,seq_len,200]
            input = self.conv0(input)
            input = self.ca0(input) * input
            input = self.sa0(input) * input
        input = rearrange(input, 'b c h w -> b (h w) c')
        output, (h_n, c_n) = self.lstm(input)

        output = rearrange(output, 'b l (d h) -> b l d h', d=2)
        fc_input = []
        for i in range(output.size(1)):
            new_input = torch.cat((h_n[-1, 0, :],output[0,i,0,:]), dim=-1)
            fc_input.append(new_input)
        fc_input = torch.stack(fc_input,dim=0)
        # out_1 = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)  # 拼接正向最后一个输出和反向最后一个输出
        # out = torch.cat((output[:,:,128:],output[:,:,:128]), dim=-1)
        # 进行全连接
        out = self.fc(fc_input)

        return out


if __name__ == '__main__':
    model = LSTM(embedding_dim=2048)
    input = torch.zeros((1, 2048, 50, 50))
    out = model(input)
    print(out.shape)
