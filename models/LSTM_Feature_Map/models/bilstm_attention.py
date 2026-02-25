import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, asnumpy, parse_shape
from torch.autograd import Variable

from .mish import Mish
class BiLSTM_Attention(nn.Module):
    def __init__(self,num_class=17,need_embedding=False,embedding_dim=128,hidden_dim=512):
        super(BiLSTM_Attention, self).__init__()
        self.need_embedding = need_embedding
        if need_embedding:
            # self.embedding = nn.Embedding(num_embeddings=859, embedding_dim=embedding_dim, padding_idx=858)
            from .FMC_0 import ChannelAttention,SpatialAttention
            self.conv0 = nn.Conv2d(embedding_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.relu = nn.ReLU(inplace=True)
            self.ca0 = ChannelAttention(hidden_dim)
            self.sa0 = SpatialAttention()
            embedding_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_size=128, bidirectional=True)
        # self.out = nn.Linear(hidden_dim * 2, num_class)
        self.out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128*2, 256),
            nn.Dropout(0.2),
            Mish(),
            nn.Linear(256, num_class),
        # nn.Sigmoid()
        )
        self.hidden_dim = 128
    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.hidden_dim * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, input):
        if self.need_embedding:
            # input = self.embedding(input)  # [batch_size,seq_len,200]
            input = self.conv0(input)
            input = self.ca0(input) * input
            input = self.sa0(input) * input
        if len(input.shape)==4:
            input = rearrange(input, 'b c h w -> (h w) b c')
        else:
            input = rearrange(input, 'b c l -> l b c')


        # input = input.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]
        hidden_state = Variable(torch.zeros(1*2, input.shape[1], self.hidden_dim)).cuda() # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1*2, input.shape[1], self.hidden_dim)).cuda()  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        # print(attn_output.shape)

        return self.out(attn_output) # model : [batch_size, num_classes], attention : [batch_size, n_step]
if __name__=='__main__':
    model = BiLSTM_Attention(embedding_dim=2048)
    input = torch.zeros((126,2048,50,50))
    out = model(input)
    print(out.shape)