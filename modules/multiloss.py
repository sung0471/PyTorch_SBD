import torch
import torch.nn as nn
from modules.layers.multi_detector import MultiDetector


class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()

        self.reg_loss = nn.SmoothL1Loss()
        self.conf_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        loc_pred, conf_pred = predictions
        loc_target, conf_target = targets[:, :2], targets[:, -1].to(torch.long)

        alpha = 0.5
        loss_loc = self.reg_loss(loc_pred, loc_target) * alpha
        loss_conf = self.conf_loss(conf_pred, conf_target) * (1. - alpha)

        loss = loss_loc + loss_conf
        return loss


if __name__ == '__main__':
    kernel_size = (16, 4, 4)
    layer = MultiDetector(512 * 4, kernel_size=kernel_size)
    input_t = torch.ones([1, 512*4, 16, 4, 4], dtype=torch.float32)
    pred = layer(input_t)
    print('loc : {}, size : {}'.format(pred[0], pred[0].size()))
    print('conf : {}, size : {}'.format(pred[1], pred[1].size()))

    targets = torch.ones([1, 3])
    criterion = MultiLoss()
    loss = criterion(pred, targets)
    print(loss)
