import torch
import torch.nn as nn
from modules.layers.multi_detector import MultiDetector
from lib.utils import encoding, cal_iou


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
            loc_target = encoding(loc_target, self.sample_duration)
            conf_target = targets[:, -1].clone().detach().to(torch.long).data

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
                # jaccard index
                overlaps = cal_iou(
                    truths,
                    self.default_bar_num
                )
                # (Bipartite Matching)
                # [1,num_objects] best prior for each ground truth
                best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
                # [1,num_priors] best ground truth for each prior
                best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
                best_truth_idx.squeeze_(0)
                best_truth_overlap.squeeze_(0)
                best_prior_idx.squeeze_(1)
                best_prior_overlap.squeeze_(1)
                best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
                # TODO refactor: index  best_prior_idx with long tensor
                # ensure every gt matches with its prior of max overlap
                for j in range(best_prior_idx.size(0)):
                    best_truth_idx[best_prior_idx[j]] = j
                matches = truths[best_truth_idx]  # Shape: [num_priors,4]
                conf = labels[best_truth_idx] + 1  # Shape: [num_priors]
                conf[best_truth_overlap < 0.5] = 0  # label as background
                loc = encoding(matches)
                loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
                conf_t[idx] = conf  # [num_priors] top class label for each prior

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
