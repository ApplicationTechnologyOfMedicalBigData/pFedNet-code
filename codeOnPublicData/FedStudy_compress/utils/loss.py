import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import rearrange
DEVICE="cuda:0"

class DiceLoss(nn.Module):
    def __init__(self, n_classes, alpha=0.5):
        "dice_loss_plus_cetr_weighted"
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        # self.weight = weight
        self.alpha = alpha

    def forward(self, input, target):
        smooth = 0.01  # 防止分母为0
        input1 = F.softmax(input, dim=1)

        target1 = F.one_hot(target, self.n_classes)

        input1 = rearrange(input1, 'b n h w s -> b n (h w s)')
        target1 = rearrange(target1, 'b h w s n -> b n (h w s)')

        input1 = input1[:, 1:, :]
        target1 = target1[:, 1:, :].float()

        inter = torch.sum(input1 * target1)
        union = torch.sum(input1) + torch.sum(target1) + smooth
        dice = 2.0 * inter / union

        loss = F.cross_entropy(input, target)

        total_loss = (1 - self.alpha) * loss + (1 - dice) * self.alpha

        return total_loss


class CrossEntropy3D(nn.Module):
    def forward(self, input, target):
        n, c, h, w, d = input.size()
        input1 = F.log_softmax(input, dim=1)
        input1 = input1.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
        target = target.view(target.numel())
        loss = F.cross_entropy(input1, target)

        return loss

class CrossEntropy2D(nn.Module):
    def forward(self, input, target):
        n, c, h, w = input.size()
        target=target.flatten().long()
        target = F.one_hot(target).float()
        if target.shape[1]!=c:
            target=torch.cat((target,torch.zeros((target.shape[0],c-target.shape[1]),device=DEVICE)),dim=1)
        input1 = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        loss = F.cross_entropy(input1, target)
        return loss


