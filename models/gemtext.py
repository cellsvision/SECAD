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
from einops import rearrange,repeat
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification,AutoModel
class GeMText(nn.Module):
    def __init__(self, dim=1, p=3, eps=1e-6):
        super(GeMText, self).__init__()  # 调用父类构造函数

        # 池化操作的维度
        self.dim = dim

        # 广义均值池化中的指数参数，可学习
        self.p = Parameter(torch.ones(1) * p)

        # 用于防止除以零的小常数
        self.eps = eps

        # 特征倍乘因子，用于控制输出特征的大小
        self.feat_mult = 1

    def forward(self, last_hidden_state, attention_mask):
        # 扩展注意力掩码以与last_hidden_state的形状匹配
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape)

        # 使用注意力掩码和eps防止除以零，对隐藏状态应用广义均值池化
        x = (last_hidden_state.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)

        # 通过除以注意力掩码的和来计算广义均值
        ret = x / attention_mask_expanded.sum(self.dim).clamp(min=self.eps)

        # 最后通过取p次方根来完成广义均值计算
        ret = ret.pow(1 / self.p)

        return ret

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
        # self.pooler = GeMText(2, d_model, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))


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
        b, _, _, _ = input.shape  # 单独先将batch缓存起来

        input = rearrange(input,'b c w h -> b (w h) c ')
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        input = torch.cat([cls_tokens, input], dim=1)

        out = self.model(input)

        all_hidden_states = torch.stack(out.hidden_states)
        # 将transformer的输出通过池化层
        attention_pooling_embeddings = self.pooler(all_hidden_states)
        # 将池化后的输出通过全连接层获得最终输出
        outputs = self.fc(attention_pooling_embeddings)
        return outputs