import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class PLCCLoss(nn.Module):

    def __init__(self):
        super(PLCCLoss, self).__init__()

    def forward(self, input, target):
        input0 = input - torch.mean(input)
        target0 = target - torch.mean(target)

        self.loss = torch.sum(input0 * target0) / (torch.sqrt(torch.sum(input0 ** 2)) * torch.sqrt(torch.sum(target0 ** 2)))
        return self.loss


# PLCC unittest
if __name__ == '__main__':
    plcc_loss = PLCCLoss()
    test_input = Variable(torch.randn(10).double(), requires_grad=True)
    test_target = Variable(torch.randn(10).double())
    optimizer = optim.SGD([test_input], lr=1)

    while True:
        plcc = plcc_loss(test_input, test_target)
        print(plcc.data[0])
        negplcc = -plcc
        negplcc.backward()
        optimizer.step()
