import torch
from torch import nn

class TripletSemiHardLoss(nn.Module):
    """
    Shape:
        - Input: :math:`(N, C)` where `C = number of channels`
        - Target: :math:`(N)`
        - Output: scalar.
    """

    def __init__(self, device, margin=0, size_average=True):
        super(TripletSemiHardLoss, self).__init__()
        self.device = device
        self.margin = margin
        self.size_average = size_average

    def forward(self, input, target):

