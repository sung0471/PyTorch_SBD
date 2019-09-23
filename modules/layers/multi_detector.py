import torch
import torch.nn as nn
from lib.utils import get_channel_list, decoding, detection


class MultiDetector(nn.Module):
    def __init__(self, block, in_planes, kernel_size=(16, 2, 2), num_classes=3, extra_layers=False,
                 phase='train', data_type='normal', conf_thresh=0.01, nms_thresh=0.33, top_k=5):
        super(MultiDetector, self).__init__()

        self.num_classes = num_classes
        self.extra_layers = extra_layers
        assert phase in ['train', 'test'], 'phase in ["train", "test"]'
        self.phase = phase
        self.data_type = data_type
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k
        self.sample_duration = kernel_size[0]

        in_channel = in_planes * block.expansion
        if not self.extra_layers:
            self.loc_pool = nn.AvgPool3d(kernel_size, stride=1)
            self.loc_fc = nn.Linear(in_channel, 2)
            self.conf_pool = nn.AvgPool3d(kernel_size, stride=1)
            self.conf_fc = nn.Linear(in_channel, num_classes)
        else:
            self.extra_layer = list()
            self.loc_layer = list()
            self.conf_layer = list()

            channel_list = get_channel_list(sample_duration=self.sample_duration)
            kernel_size = (2, kernel_size[1], kernel_size[2])
            # kernel_size = (1, kernel_size[1], kernel_size[2])
            filter_size = (2, 1, 1)

            if self.data_type in ['normal', 'cut']:
                self.loc_layer += [nn.Conv3d(in_channel, 2, kernel_size=kernel_size, padding=0, bias=False)]
                self.conf_layer += [nn.Conv3d(in_channel, num_classes, kernel_size=kernel_size, padding=0, bias=False)]

            if self.data_type in ['normal', 'gradual']:
                for in_channel, mid_channel, out_channel in channel_list:
                    self.extra_layer += [nn.Conv3d(in_channel, mid_channel, kernel_size=1, padding=0, bias=False)]
                    self.extra_layer += [nn.Conv3d(mid_channel, out_channel, kernel_size=3, padding=filter_size, bias=False,
                                                   dilation=filter_size, stride=filter_size)]
                    self.loc_layer += [nn.Conv3d(out_channel, 2, kernel_size=kernel_size, padding=0, bias=False)]
                    self.conf_layer += [nn.Conv3d(out_channel, num_classes, kernel_size=kernel_size, padding=0, bias=False)]

            self.loc_layer = nn.Sequential(*self.loc_layer)
            self.conf_layer = nn.Sequential(*self.conf_layer)
            if self.data_type in ['normal', 'gradual']:
                self.extra_layer = nn.Sequential(*self.extra_layer)
                self.relu = nn.ReLU(inplace=True)

            if self.phase == 'test':
                self.softmax = nn.Softmax(dim=-1)

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
                total_length = self.sample_duration

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
                x = self.relu(x)

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
            # normal : loc[8, 26, 2], conf[8, 26, num_classes]
            # cut : loc[8, 15, 2], conf[8, 15, num_classes]
            # gradual : loc[8, 11, 2], conf[8, 11, num_classes]
            if self.phase == 'test':
                out = (out[1], self.softmax(out[1]))
                output = detection(out, self.sample_duration, self.num_classes, self.data_type,
                                   self.top_k, self.conf_thresh, self.nms_thresh)

                # output = [batch_size, num_classes, num_bars, [start, end, conf]]
                pred_num = torch.zeros(batch_size, self.num_classes, dtype=torch.int32)
                for batch_num in range(batch_size):
                    for cls in range(1, self.num_classes):
                        i = 0
                        while output[batch_num, cls, i, -1] >= 0.6:
                            bound_start = start_boundaries[batch_num].float().data
                            bound_end = start_boundaries[batch_num].float().data + self.sample_duration - 1
                            output_boundary = torch.round(output[batch_num, cls, i, :-1] + start_boundaries[batch_num])
                            if bound_start <= output_boundary[0] <= bound_end and bound_start <= output_boundary[1] <= bound_end:
                                output[batch_num, cls, i, :-1] = output_boundary
                            else:
                                output[batch_num, cls, i, :-1] = torch.zeros(1, 2)
                            i += 1
                            if i == self.top_k:
                                break
                        pred_num[batch_num, cls] = i

                total_bars_num = pred_num.int().sum().clone().detach().data
                frame_pos = torch.zeros(total_bars_num, 2)
                labels = torch.zeros(total_bars_num, 1)
                num = 0
                for batch_num in range(batch_size):
                    for cls in range(1, self.num_classes):
                        result_num = pred_num[batch_num, cls].data
                        for i in range(result_num):
                            bound_start = output[batch_num, cls, i, 0].data
                            bound_end = output[batch_num, cls, i, 1].data
                            if bound_start == 0 and bound_end == 0:
                                pass
                            else:
                                frame_pos[num, :] = output[batch_num, cls, i, :-1].clone().detach()
                                labels[num, :] = cls
                                num += 1

                # frame_pos = [bars_num, 2]
                # labels = [bars_num, 1]
                out = (frame_pos[:num] + 1, labels[:num])

        return out


if __name__ == '__main__':
    from models.detector import Bottleneck
    block = Bottleneck
    kernel_size = (16, 4, 4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    layer = MultiDetector(block, 512, kernel_size=kernel_size, num_classes=3, extra_layers=True, phase='test').to(device)
    print(layer)
    input_t = torch.randn([8, 512*4, 16, 4, 4], dtype=torch.float32).to(device)
    boundary = torch.zeros(8).to(device)
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
