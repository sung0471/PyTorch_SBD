import torch
import torch.nn as nn
from modules.layers.multi_detector import MultiDetector
from lib.utils import encoding


class MultiLoss(nn.Module):
    def __init__(self, extra_layers=False, sample_duration=16):
        super(MultiLoss, self).__init__()

        self.extra_layers = extra_layers
        self.sample_duration = sample_duration
        self.reg_loss = nn.SmoothL1Loss()
        # self.reg_loss = nn.MSELoss()
        self.conf_loss = nn.CrossEntropyLoss()

        self.default_bar_number_list = dict()
        self.default_bar_number_list[16] = [15, 7, 3, 1]
        self.default_bar_number_list[32] = [31, 15, 7, 3, 1]
        self.default_bar = dict()
        self.default_bar[16] = torch.zeros(26, 2)
        self.default_bar[32] = torch.zeros(57, 2)
        for sample_duration in self.default_bar_number_list.keys():
            count = 0
            length = 2
            for default_bar_number in self.default_bar_number_list[sample_duration]:
                for start in range(default_bar_number):
                    self.default_bar[sample_duration][count][0] = start * (length / 2)
                    self.default_bar[sample_duration][count][1] = start * (length / 2) + length - 1
                    count += 1
                length *= 2

    def forward(self, predictions, targets):
        loc_pred, conf_pred = predictions
        if not self.extra_layers:
            loc_target = targets[:, :-1].clone().detach().data
            conf_target = targets[:, -1].clone().detach().to(torch.long).data
            loc_target = encoding(loc_target)

            alpha = 0.5
            loss_loc = self.reg_loss(loc_pred, loc_target) * alpha
            loss_conf = self.conf_loss(conf_pred, conf_target) * (1. - alpha)
        else:
            batch_size = targets.size(0)
            default_bar_num = self.default_bar[self.sample_duration].size(0)
            loc_t = torch.Tensor(batch_size, default_bar_num, 2)
            conf_t = torch.LongTensor(batch_size, default_bar_num)
            for idx in range(batch_size):
                truths = targets[idx, :, :-1].clone().detach().data
                labels = targets[idx, :, -1].clone().detach().to(torch.long).data

            # alpha = 0.5
            # loss_loc = self.reg_loss(loc_pred, loc_target) * alpha
            # loss_conf = self.conf_loss(conf_pred, conf_target) * (1. - alpha)

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
    criterion = MultiLoss(extra_layers=True, sample_duration=16)
    loss = criterion(pred, targets)
    print(loss)
