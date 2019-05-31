import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xsobel = torch.Tensor(
                            [[1, 0, 1],
                            [2, 0, -2],
                            [1, 0, -1]])

        self.ysobel = torch.Tensor(
                            [[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])

    def forward(self, x):
        xgrad = F.conv2d(x, self.xsobel)
        ygrad = F.conv2d(x, self.ysobel)
        grad = torch.abs(xgrad) + torch.abs(ygrad)
        loss = torch.sum(grad)
        return loss
