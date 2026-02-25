# 主要作用和特点如下：

# 自注意力池化: 通过自注意力机制，模块学习了如何从多个层的隐藏状态中为每个层分配权重。这使得它可以考虑来自模型的多个层的信息，而不仅仅是最后一层或第一层。

# 获取固定长度的表示: 不管输入文本有多长，这种池化方法都可以输出一个固定长度的向量。这对于某些下游任务，如分类，非常有用，因为它们需要固定大小的输入。

# 考虑多层信息: 通过考虑来自多个隐藏层的信息，这种方法可能更好地捕捉输入数据的不同抽象层次的特征。

# 具体地，以下是模块的工作流程：

# 对于给定的所有隐藏状态，它首先选择每个层的第一个token（例如，BERT中的CLS token）的隐藏状态，并将它们堆叠起来。

# 使用一个查询向量q和堆叠的隐藏状态来计算自注意力权重。

# 使用得到的注意力权重对隐藏状态进行加权平均，得到一个池化的表示。

# 使用一个线性变换（由参数w_h表示）进一步转换池化的表示。

# 这种自注意力池化方法常常被用于自然语言处理任务，特别是当我们想从Transformer模型中获取整个输入序列的固定大小表示时。
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification,AutoModel
class AttentionPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_fc):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = num_layers  # 隐藏层的数量
        self.hidden_size = hidden_size  # 隐藏状态的大小
        self.hiddendim_fc = hiddendim_fc  # 全连接层的隐藏维度
        self.dropout = nn.Dropout(0.1)  # dropout层，用于正则化

        # 初始化q为小的随机值
        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        # self.q = nn.Parameter(torch.from_numpy(q_t)).float().cuda()
        self.register_buffer("q",torch.from_numpy(q_t).float())

        # self.q = nn.Parameter(torch.from_numpy(q_t)).float()

        # 初始化w_h为小的随机值
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        # self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()
        # self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().cuda()
        self.register_buffer("w_h",torch.from_numpy(w_ht).float())

    def forward(self, all_hidden_states):
        # 对所有隐藏层的第一个token（通常是CLS token）的隐藏状态进行堆叠
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)

        # 通过注意力机制得到输出
        out = self.attention(hidden_states)
        out = self.dropout(out)  # 应用dropout
        return out

    def attention(self, h):
        # 使用q和h计算注意力分数
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)  # 对分数进行softmax操作，得到注意力权重

        # 根据注意力权重和h计算加权平均的隐藏状态
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)

        # 使用w_h对加权平均的隐藏状态进行线性变换
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v

class Transformer(nn.Module):
    def __init__(self, num_class,d_model=1024, layer_norm_eps= 1e-5, num_encoder_layers: int = 2,hidden_size=768):
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=512, dropout=0.1,
                                                activation=F.relu, layer_norm_eps=layer_norm_eps, batch_first=True, norm_first=False,
                                                )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.model = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # 从预训练模型中加载模型
        # self.model = torch.nn.Transformer(d_model=512,nhead=8)
        # 从预训练模型中加载配置
        # 获取合适的池化层
        self.pooler = AttentionPooling(2, d_model, d_model)

        self.fc = nn.Linear(d_model, num_class)

        # 初始化全连接层的权重
        for module in self.model.modules():
            self._init_weights(module)

        # # 如果配置文件中指定，冻结模型的前半部分层
        # if freeze_method is not None:
        #     freeze_method = get_freezing_method(freeze_method)
        #     freeze_method(self.model)

    def _init_weights(self, module):
        # 初始化权重的辅助方法
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input):
        # 前向传播函数
        # 通过transformer模型获得输出
        input = rearrange(input,'b c w h -> b (w h) c ')
        out = self.model(input)
        all_hidden_states = torch.stack(out.hidden_states)
        # 将transformer的输出通过池化层
        attention_pooling_embeddings = self.pooler(all_hidden_states)
        # 将池化后的输出通过全连接层获得最终输出
        outputs = self.fc(attention_pooling_embeddings)
        return outputs