import torch
from einops import repeat
import torch.nn.functional as F


class TransformerEncoder(torch.nn.Module):
    def __init__(self, n_sequences=400, raw_dim=7, d_model=64, dim_feedforward=64, n_class=5, pool='mean',
                 pos_encoding=False, cls_token=False):
        super(TransformerEncoder, self).__init__()
        self.pool = pool
        self.embedding = torch.nn.Linear(raw_dim, d_model)
        self.cls_token_layer = None
        self.pos_encoding_layer = None
        if cls_token:
            self.cls_token_layer = torch.nn.Parameter(torch.randn(1, 1, d_model))
        if pos_encoding:
            self.pos_encoding_layer = torch.nn.Parameter(torch.randn(1, n_sequences + 1, d_model))

        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=dim_feedforward,
                                                              batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.mlp_head = torch.nn.Linear(d_model, n_class)

    def forward(self, x):
        b, n, _ = x.shape
        x = self.embedding(x)
        if self.cls_token_layer is not None:
            cls_tokens = repeat(self.cls_token_layer, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_encoding_layer is not None:
            x += self.pos_encoding[:, :(n + 1)]
        x = self.transformer_encoder(x)
        x = x[:, 0] if self.pool == 'cls' else x.mean(dim=1)
        x = self.mlp_head(x)
        return x


def get_post_result(model_output, label=None, thresh=0, class_order=None):

    # model_output: torch_tensor (shape: 1 x 5)
    if class_order is None:
        class_order = ['nilm', 'asc_us', 'lsil', 'hsil', 'agc']

    conf = F.softmax(model_output.detach().cpu(), dim=1).squeeze(0).numpy()
    orig_pred, max_conf = class_order[conf.argmax()], conf.max()
    pred = 'asc_us' if max_conf < thresh and orig_pred == 'nilm' else orig_pred
    ret = dict(
        pred=pred.upper(),
        ori_pred=orig_pred.upper(),
        max_conf=max_conf,
        NILM=conf[0], ASC_US=conf[1], LSIL=conf[2], HSIL=conf[3], AGC=conf[4],
    )
    if label is not None:
        ret['gt'] = label.upper()
    return ret


def main():
    model = TransformerEncoder(pos_encoding=False)
    x = torch.rand((1, 400, 7))
    y = model(x)
    print(y)


if __name__ == '__main__':
    main()