import torch
import torch.nn as nn
import torch.nn.functional as F

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsKLDLoss(nn.Module):
    def __init__(self):
        super(JointsKLDLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        # self.softmax = F.Softmax(dim=1)
        # self.logsoftmax = nn.(dim=1)

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        width = output.size(2)
        height = output.size(3)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        # [B, 4096]
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            heatmap_pred = F.log_softmax(heatmap_pred.mul(target_weight[:, idx]), dim=1)
            heatmap_gt = F.softmax(heatmap_gt.mul(target_weight[:, idx]), dim=1)

            loss += self.criterion(
                heatmap_pred, heatmap_gt
            )

        loss = torch.sum(torch.sum(loss, dim=1), dim=0)

        return loss / batch_size / (width * height)
