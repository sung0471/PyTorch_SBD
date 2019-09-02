import torch
import torch.nn as nn
import numpy as np
from lib.utils import channel_list, decoding, default_bar


class MultiDetector(nn.Module):
    def __init__(self, block, in_planes, kernel_size=(16, 2, 2), num_classes=3, extra_layers=False, phase='train'):
        super(MultiDetector, self).__init__()

        self.num_classes = num_classes
        self.extra_layers = extra_layers
        self.phase = phase
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
            # loc[8, 26, 2], conf[8, 26, 1]
            if self.phase == 'test':
                loc, conf = out
                frame_pos = torch.zeros(loc.size(0), loc.size(1), loc.size(2))
                labels = torch.zeros(conf.size(0), conf.size(1), 1)
                for i in range(batch_size):
                    total_length = self.sample_duration - 1

                    assert loc.size(1) == self.default_bar.size(0)
                    frame_pos[i, :] = decoding(loc[i, :], total_length, default_bar=self.default_bar)
                    frame_pos[i, :] += start_boundaries[i]                  # [default_num, 2]
                    label = conf[i, :]
                    labels[i, :] = torch.argmax(label, dim=1).view(-1, 1)   # [default_num, 1]

                # frame_pos = [batch_size, default_num, 2]
                # lables = [batch_size, default_num, 1]
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
