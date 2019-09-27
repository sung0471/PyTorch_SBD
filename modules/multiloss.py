import torch
import torch.nn as nn
from modules.layers.multi_detector import MultiDetector
from lib.utils import encoding, cal_iou, log_sum_exp, Configure


class MultiLoss(nn.Module):
    def __init__(self, device, extra_layers=False, sample_duration=16, num_classes=3,
                 data_type='normal', policy='first', neg_ratio=3, neg_threshold=(0.33, 0.5)):
        super(MultiLoss, self).__init__()

        self.device = device
        self.extra_layers = extra_layers
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        self.negpos_ratio = neg_ratio
        self.neg_threshold = neg_threshold

        self.reg_loss = nn.SmoothL1Loss()
        # self.reg_loss = nn.MSELoss()
        self.conf_loss = nn.CrossEntropyLoss()
        c = Configure(sample_duration=sample_duration, data_type=data_type, policy=policy)
        self.default_bar = c.default_bar()

    def forward(self, predictions, targets):
        total_length = self.sample_duration
        loc_pred, conf_pred = predictions
        if not self.extra_layers:
            # loc_pred : [batch_size, 2]
            # conf_pred : [batch_size, 3]
            loc_target = targets[:, :-1].clone().detach().data
            loc_target = encoding(loc_target, total_length)
            conf_target = targets[:, -1].clone().detach().to(torch.long).data

            loss_loc = self.reg_loss(loc_pred, loc_target)
            loss_conf = self.conf_loss(conf_pred, conf_target)

        else:
            # loc_pred : [batch_size, default_bar_num, 2]
            # conf_pred : [batch_size, default_bar_num, 3]
            batch_size = targets.size(0)
            default_bar_num = self.default_bar.size(0)
            self.reg_loss = nn.SmoothL1Loss(reduction='sum')
            self.conf_loss = nn.CrossEntropyLoss(reduction='sum')

            loc_t = torch.Tensor(batch_size, default_bar_num, 2)
            conf_t = torch.LongTensor(batch_size, default_bar_num)
            # wrap targets
            with torch.no_grad():
                self.default_bar = self.default_bar.to(self.device)
                loc_t = loc_t.to(self.device)
                conf_t = conf_t.to(self.device)

            for idx in range(batch_size):
                truths = targets[idx, :-1].view(-1, 2).data                # [1, 2]
                labels = targets[idx, -1].view(-1, 1).to(torch.long).data  # [1, 1]
                default = self.default_bar.clone().detach().data           # [default_bar_num, 2]
                # jaccard index
                # truths = truths.view(-1, 1, 2)      # [1, 1, 2]
                # labels = labels.view(-1, 1)         # [1, 1]
                # default = default.view(1, -1, 2)    # [1, default_bar_num, 2]
                overlaps = cal_iou(                 # [1, default_bar_num]
                    truths,
                    default,
                    use_default=True
                )
                # (Bipartite Matching)
                # [1(gt_num), 1] best prior for each ground truth
                best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)  # GT와 가장 많이 겹치는 default
                # [1, default_bar_num] best ground truth for each prior
                best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)  # default와 가장 많이 겹치는 GT
                best_prior_idx.squeeze_(1)          # [1(gt_num)]
                best_prior_overlap.squeeze_(1)      # [1(gt_num)]
                best_truth_idx.squeeze_(0)          # [default_bar_num]
                best_truth_overlap.squeeze_(0)      # [default_bar_num]
                best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
                # TODO refactor: index  best_prior_idx with long tensor
                # ensure every gt matches with its prior of max overlap
                for j in range(best_prior_idx.size(0)):
                    best_truth_idx[best_prior_idx[j]] = j
                matches = truths[best_truth_idx]    # Shape: [default_bar_num, 2]
                conf = labels[best_truth_idx] + 1   # Shape: [default_bar_num]

                background_conf_idx = conf == 1         # get index of background
                conf[best_truth_overlap < self.neg_threshold] = 0     # label as negative
                conf[background_conf_idx] = 1           # set label to background

                assert matches.size() == self.default_bar.size(),\
                    "matches_size : {}, default_bar_size : {}".format(matches.size(), default.size())
                loc = encoding(matches, total_length, default_bar=default)
                loc_t[idx] = loc  # [default_bar_num,2] encoded offsets to learn
                conf_t[idx] = conf.squeeze(1)  # [default_bar_num] top class label for each prior

            pos = conf_t > 0
            loc_pos = conf_t > 1

            # Localization Loss (Smooth L1)
            # Shape: [batch, default_bar_num, 2]
            pos_idx = loc_pos.unsqueeze(loc_pos.dim()).expand_as(loc_pred)
            loc_p = loc_pred[pos_idx].view(-1, 2)
            loc_t = loc_t[pos_idx].view(-1, 2)
            loss_loc = self.reg_loss(loc_p, loc_t)

            # Compute max conf across batch for hard negative mining
            # batch_conf : [batch_size * default_bar_num, num_classes]
            # loss_conf : [batch_size * default_bar_num, 1]
            batch_conf = conf_pred.view(-1, self.num_classes)
            conf_t = torch.clamp(conf_t - 1, min=0)
            loss_conf = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

            # Hard Negative Mining
            loss_conf = loss_conf.view(batch_size, -1)  # [batch_size, default_bar_num]
            loss_conf[pos] = 0   # filter out pos boxes for now / positive에 해당하는 bar들을 filtering
            _, loss_idx = loss_conf.sort(1, descending=True)    # loss_conf가 큰 idx의 내림차순 : [8, 26]
            _, idx_rank = loss_idx.sort(1)  # 각 idx의 등수 : [batch_size, default_bar_num]
            num_pos = pos.long().sum(1, keepdim=True)
            num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)     # [batch_size, 1]
            neg = idx_rank < num_neg.expand_as(idx_rank)         # [batch_size, default_bar_num]

            # Confidence Loss Including Positive and Negative Examples
            pos_idx = pos.unsqueeze(2).expand_as(conf_pred)
            neg_idx = neg.unsqueeze(2).expand_as(conf_pred)
            conf_p = conf_pred[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            targets_weighted = conf_t[(pos + neg).gt(0)]
            loss_conf = self.conf_loss(conf_p, targets_weighted)

            # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

            N = num_pos.data.sum()
            loss_loc /= N
            loss_conf /= N

            # N_pos = num_pos.data.sum()
            # N_neg = num_neg.data.sum()
            # loss_loc /= N_pos
            # loss_conf /= N_pos + N_neg

        return loss_loc, loss_conf


if __name__ == '__main__':
    from models.detector import Bottleneck
    block = Bottleneck
    kernel_size = (16, 4, 4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    layer = MultiDetector(block, 512, kernel_size=kernel_size, num_classes=3, extra_layers=True, phase='train')
    input_t = torch.ones([8, 512*4, 16, 4, 4], dtype=torch.float32)
    boundary = torch.zeros(8)
    for i in range(8):
        boundary[i] = i * 8
    data = layer.to(device)(input_t.to(device), boundary.to(device))
    print('loc_size : {}'.format(data[0].size()))
    print('conf_size : {}'.format(data[1].size()))

    targets = torch.ones([8, 3]).to(device)
    criterion = MultiLoss(device, extra_layers=True, sample_duration=16)
    loss = criterion(data, targets)
    print(loss, loss.size())
