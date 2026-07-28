from torch import nn
from torch.nn import functional as F


class ObjectNormalizedL2Loss(nn.Module):

    def __init__(self):
        super(ObjectNormalizedL2Loss, self).__init__()

    def forward(self, output, dmap, num_objects):
        return ((output - dmap) ** 2).sum() / num_objects


class CountL1Loss(nn.Module):

    def __init__(self):
        super(CountL1Loss, self).__init__()

    def forward(self, output, dmap):
        pred_count = output.flatten(1).sum(dim=1)
        target_count = dmap.flatten(1).sum(dim=1)
        return F.l1_loss(pred_count, target_count)
