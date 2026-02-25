import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Multi_Class_Weight_Loss(nn.Module):
    def __init__(self, class_num, alpha_matrix=None, size_average=True):
        super(Multi_Class_Weight_Loss, self).__init__()
        if alpha_matrix is None:
            self.alpha_matrix = Variable(torch.ones(class_num, class_num))
        else:
            if isinstance(alpha_matrix, Variable):
                self.register_buffer("alpha_matrix",alpha_matrix)

            else:
                self.register_buffer("alpha_matrix", torch.tensor(alpha_matrix))

        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        # N = inputs.size(0)
        # C = inputs.size(1)
        # P = F.softmax(inputs)

        self.alpha_matrix = self.alpha_matrix.to(inputs.device)
        preds_softmax = F.softmax(inputs, dim=1)
        opp_preds_softmax = 1 - preds_softmax
        opp_preds_softmax = opp_preds_softmax.to(torch.double)
        # opp_preds_softmax = torch.where(opp_preds_softmax>0.01,opp_preds_softmax,0.01)
        opp_preds_softmax = torch.clamp(opp_preds_softmax,min=0.01)
        opp_preds_logsoft = -torch.log(opp_preds_softmax)


        target_weight = self.alpha_matrix[targets]
        loss = torch.mul(target_weight, opp_preds_logsoft)


        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


if __name__ == '__main__':
    alpha_matrix = [
        [0, 1, 2, 4, 4],
        [2, 0, 2, 2, 2],
        [4, 2, 0, 2, 2],
        [8, 4, 2, 0, 2],
        [8, 4, 2, 2, 0],
    ]
    loss_func = Multi_Class_Weight_Loss(class_num=5, alpha_matrix=alpha_matrix)
    # inputs = torch.ones((7, 5))
    inputs = torch.Tensor([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ])
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1])
    loss = loss_func(inputs, labels)
    print(loss)
