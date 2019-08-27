import torch
import torch.nn as nn
import numpy as np


class MultiDetector(nn.Module):
    def __init__(self, block, in_planes, kernel_size=(16, 2, 2), num_classes=3, extra_layers=False):
        super(MultiDetector, self).__init__()

        self.num_classes = num_classes
        self.extra_layers = extra_layers

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
            channel_list = dict()
            channel_list[16] = [(2048, 512, 1024), (1024, 256, 512), (512, 128, 256)]
            channel_list[32] = [(2048, 512, 1024), (1024, 256, 512), (512, 128, 256), (256, 128, 256)]
            sample_duration = kernel_size[0]
            filter_size = (2, 1, 1)
            kernel_size = (2, kernel_size[1], kernel_size[2])

            self.loc_layer += [nn.Conv3d(in_channel, 2, kernel_size=kernel_size, padding=0, bias=False)]
            self.conf_layer += [nn.Conv3d(in_channel, num_classes, kernel_size=kernel_size, padding=0, bias=False)]
            for in_channel, mid_channel, out_channel in channel_list[sample_duration]:
                self.extra_layer += [nn.Conv3d(in_channel, mid_channel, kernel_size=1, padding=0, bias=False)]
                self.extra_layer += [nn.Conv3d(mid_channel, out_channel, kernel_size=3, padding=filter_size, bias=False,
                                               dilation=filter_size, stride=filter_size)]

                self.loc_layer += [nn.Conv3d(out_channel, 2, kernel_size=kernel_size, padding=0, bias=False)]
                self.conf_layer += [nn.Conv3d(out_channel, num_classes, kernel_size=kernel_size, padding=0, bias=False)]

    def forward(self, x):
        batch_size = x.size(0)
        if not self.extra_layers:
            loc_x = self.loc_pool(x)
            loc_x = loc_x.view(batch_size, -1)
            loc_x = self.loc_fc(loc_x)

            conf_x = self.conf_pool(x)
            conf_x = conf_x.view(batch_size, -1)
            conf_x = self.conf_fc(conf_x)
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
            print(loc_x.size(), conf_x.size())
            for i in range(1, len(loc_list)):
                loc_x = torch.cat((loc_x, loc_list[i]), 1)
                conf_x = torch.cat((conf_x, conf_list[i]), 1)
        out = (loc_x.view(batch_size, -1, 2), conf_x.view(batch_size, -1, self.num_classes))

        return out


if __name__ == '__main__':
    from models.detector import Bottleneck
    block = Bottleneck
    kernel_size = (16, 4, 4)

    layer = MultiDetector(block, 512, kernel_size=kernel_size, extra_layers=True)
    print(layer)
    input_t = torch.randn([8, 512*4, 16, 4, 4], dtype=torch.float32)
    data = layer(input_t)
    loc, conf = data
    print('loc : {}, size : {}'.format(loc, loc.size()))
    print('conf : {}, size : {}'.format(conf, conf.size()))

    loc_numpy = loc.clone().detach().cpu().numpy()
    conf_numpy = conf.clone().detach().cpu().numpy()
    labels, frame_pos = list(), list()

    boundary = list()
    for i in range(len(loc)):
        boundary += [0 + 8 * i]

    sample_duration = 16
    for i, (center, length) in enumerate(loc_numpy):
        end = int((center * 2 + length) / 2 * sample_duration) + boundary[i]
        start = int((center * 2 - length) / 2 * sample_duration) + boundary[i]
        frame_pos += [[start, end]]

    for row in conf_numpy:
        labels.append(np.argmax(row))

    print(loc_numpy)
    print(frame_pos)
    print(conf_numpy)
