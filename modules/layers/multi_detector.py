import torch
import torch.nn as nn
import numpy as np
from lib.utils import channel_list, decoding, default_bar, nms


class MultiDetector(nn.Module):
    def __init__(self, block, in_planes, kernel_size=(16, 2, 2), num_classes=3, extra_layers=False,
                 phase='train', conf_thresh=0.01, nms_thresh=0.45, top_k=5):
        super(MultiDetector, self).__init__()

        self.num_classes = num_classes
        self.extra_layers = extra_layers
        self.phase = phase
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k
        self.sample_duration = kernel_size[0]
        self.default_bar = default_bar(sample_duration=self.sample_duration)

        in_channel = in_planes * block.expansion
        if not self.extra_layers:
            self.loc_pool = nn.AvgPool3d(kernel_size, stride=1)
            self.loc_fc = nn.Linear(in_channel, 2)
            self.conf_pool = nn.AvgPool3d(kernel_size, stride=1)
            self.conf_fc = nn.Linear(in_channel, num_classes)
        else:
            self.loc_layer = list()
            self.conf_layer = list()
            self.extra_layer = list()

            self.channel_list = channel_list(sample_duration=self.sample_duration)
            filter_size = (2, 1, 1)
            kernel_size = (2, kernel_size[1], kernel_size[2])

            self.loc_layer += [nn.Conv3d(in_channel, 2, kernel_size=kernel_size, padding=0, bias=False)]
            self.conf_layer += [nn.Conv3d(in_channel, num_classes, kernel_size=kernel_size, padding=0, bias=False)]
            for in_channel, mid_channel, out_channel in self.channel_list:
                self.extra_layer += [nn.Conv3d(in_channel, mid_channel, kernel_size=1, padding=0, bias=False)]
                self.extra_layer += [nn.Conv3d(mid_channel, out_channel, kernel_size=3, padding=filter_size, bias=False,
                                               dilation=filter_size, stride=filter_size)]

                self.loc_layer += [nn.Conv3d(out_channel, 2, kernel_size=kernel_size, padding=0, bias=False)]
                self.conf_layer += [nn.Conv3d(out_channel, num_classes, kernel_size=kernel_size, padding=0, bias=False)]

            self.extra_layer = nn.Sequential(*self.extra_layer)
            self.loc_layer = nn.Sequential(*self.loc_layer)
            self.conf_layer = nn.Sequential(*self.conf_layer)

    def forward(self, x, start_boundaries):
        batch_size = x.size(0)
        if not self.extra_layers:
            loc_x = self.loc_pool(x)
            loc_x = loc_x.view(batch_size, -1)
            loc_x = self.loc_fc(loc_x)

            conf_x = self.conf_pool(x)
            conf_x = conf_x.view(batch_size, -1)
            conf_x = self.conf_fc(conf_x)
            out = (loc_x.view(batch_size, 2), conf_x.view(batch_size, self.num_classes))

            # detection
            # loc[8, 2], conf[8, 3]
            if self.phase == 'test':
                total_length = self.sample_duration - 1

                loc, conf = out
                loc = decoding(loc, total_length)

                frame_pos = torch.zeros(loc.size(0), loc.size(1))
                for i in range(batch_size):
                    res = loc[i] + start_boundaries[i]
                    frame_pos[i, :] = res

                labels = torch.argmax(conf, dim=1)

                out = (frame_pos, labels)
        else:
            loc_list = list()
            conf_list = list()
            loc_list += [self.loc_layer[0](x).view(batch_size, -1)]
            conf_list += [self.conf_layer[0](x).view(batch_size, -1)]
            for i in range(0, len(self.extra_layer), 2):
                x = self.extra_layer[i](x)
                x = self.extra_layer[i + 1](x)

                idx = int(i / 2 + 1)
                loc_list += [self.loc_layer[idx](x).view(batch_size, -1)]
                conf_list += [self.conf_layer[idx](x).view(batch_size, -1)]

            loc_x = loc_list[0]
            conf_x = conf_list[0]
            for i in range(1, len(loc_list)):
                loc_x = torch.cat((loc_x, loc_list[i]), 1)
                conf_x = torch.cat((conf_x, conf_list[i]), 1)
            out = (loc_x.view(batch_size, -1, 2), conf_x.view(batch_size, -1, self.num_classes))

            # detection
            # loc[8, 26, 2], conf[8, 26, 3]
            if self.phase == 'test':
                loc, conf = out
                frame_pos = torch.zeros(loc.size(0), loc.size(1), loc.size(2))
                labels = torch.zeros(conf.size(0), conf.size(1), 1)

                output = torch.zeros(batch_size, self.num_classes, self.top_k, 3)
                conf_pred = conf.transpose(2, 1)
                for i in range(batch_size):
                    total_length = self.sample_duration - 1

                    # assert loc.size(1) == self.default_bar.size(0)
                    # frame_pos[i, :] = decoding(loc[i, :], total_length, default_bar=self.default_bar)
                    # frame_pos[i, :] += start_boundaries[i]                  # [default_num, 2]
                    # label = conf[i, :]
                    # labels[i, :] = torch.argmax(label, dim=1).view(-1, 1)   # [default_num, 1]

                    decoded_bars = decoding(loc[i, :], total_length, default_bar=self.default_bar)  # [default_num, 2]
                    conf_scores = conf_pred[i].clone().detach()   # [3, default_num]

                    for cl in range(1, self.num_classes):
                        c_mask = conf_scores[cl].gt(self.conf_thresh)   # [default_num]
                        # for i in default_num,
                        # if conf[i] > conf_thresh, num += 1
                        scores = conf_scores[cl][c_mask]    # [num]
                        if scores.size(0) == 0:
                            continue
                        l_mask = c_mask.unsqueeze(1).expand_as(decoded_bars)    # [default_num, 2]
                        bars = decoded_bars[l_mask].view(-1, 2)     # [num, 2]
                        # idx of highest scoring and non-overlapping boxes per class
                        ids, count = nms(bars, scores, self.nms_thresh, self.top_k)
                        output[i, cl, :count] = \
                            torch.cat((bars[ids[:count]],
                                       scores[ids[:count]].unsqueeze(1)), 1)    # [count, 3]

                # [batch_size, all_result_bars_num, 3]
                # all_result_bars_num : 클래스 별 bar의 갯수(최대 top_k * (num_classes - 1))
                # 3 : start, end, conf
                flt = output.contiguous().view(batch_size, -1, 3)
                _, idx = flt[:, :, -1].sort(1, descending=True)     # [batch_size, bars_num]
                _, rank = idx.sort(1)                               # [batch_size, bars_num], 각 idx 의 등수
                flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

                # [batch_size, class, num_bars, data]
                # data = [start, end, conf]
                pred_num = torch.zeros(batch_size, self.num_classes, dtype=torch.int32)
                for batch_num in range(batch_size):
                    for cls in range(self.num_classes):
                        i = 0
                        while output[batch_num, cls, i, -1] >= 0.6:
                            output[batch_num, cls, i, :-1] = torch.round(
                                output[batch_num, cls, i, :-1] + start_boundaries[batch_num])
                            i += 1
                            if i == self.top_k:
                                break
                        pred_num[batch_num, cls] = i

                total_bars_num = pred_num.int().sum().clone().detach().item()
                frame_pos = torch.zeros(total_bars_num, 2, dtype=torch.int32)
                labels = torch.zeros(total_bars_num, dtype=torch.int32)
                num = 0
                for batch_num in range(batch_size):
                    for cls in range(self.num_classes):
                        result_num = pred_num[batch_num, cls].data
                        for i in range(result_num):
                            frame_pos[num] = output[batch_num, cls, i, :-1]
                            labels[num] = cls
                            num += 1

                # frame_pos = [bars_num, 2]
                # labels = [bars_num]
                out = (frame_pos, labels)

        return out


if __name__ == '__main__':
    from models.detector import Bottleneck
    block = Bottleneck
    kernel_size = (16, 4, 4)

    layer = MultiDetector(block, 512, kernel_size=kernel_size, num_classes=3, extra_layers=True, phase='test')
    print(layer)
    input_t = torch.randn([8, 512*4, 16, 4, 4], dtype=torch.float32)
    boundary = torch.zeros(8)
    for i in range(8):
        boundary[i] = i * 8
    data = layer(input_t, boundary)
    loc, conf = data
    print('loc : {}, size : {}'.format(loc, loc.size()))
    print('conf : {}, size : {}'.format(conf, conf.size()))

    # loc_numpy = loc.clone().detach().cpu().numpy()
    # conf_numpy = conf.clone().detach().cpu().numpy()
    # labels, frame_pos = list(), list()
    #
    # boundary = list()
    # for i in range(len(loc)):
    #     boundary += [0 + 8 * i]
    #
    # sample_duration = 16
    # for i, (center, length) in enumerate(loc_numpy):
    #     end = int((center * 2 + length) / 2 * sample_duration) + boundary[i]
    #     start = int((center * 2 - length) / 2 * sample_duration) + boundary[i]
    #     frame_pos += [[start, end]]
    #
    # for row in conf_numpy:
    #     labels.append(np.argmax(row))
    #
    # print(loc_numpy)
    # print(frame_pos)
    # print(conf_numpy)
