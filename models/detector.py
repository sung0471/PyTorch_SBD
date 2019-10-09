import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import os
from modules.layers.DepthwiseSeparableConv import DepthwiseSeparableConv
from modules.layers.multi_detector import MultiDetector

__all__ = ['ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200',
           'ResNeXt', 'resnext50', 'resnext101']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.to(device)

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


# `19.8.6. modules.layers.DepthwiseSeparableConv.py로 이동
# class DepthwiseSeparableConv3d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False):
#         super(DepthwiseSeparableConv3d, self).__init__()
#
#         self.channelwise = nn.Conv3d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                                      groups=in_planes, bias=bias)
#         # self.bn1 = nn.BatchNorm3d(in_planes)
#         # self.pointwise = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)
#         # self.bn2 = nn.BatchNorm3d(out_planes)
#         # self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.channelwise(x)
#         # x = self.bn1(x)
#         # x = self.relu(x)
#         # x = self.pointwise(x)
#         # x = self.bn2(x)
#         # x = self.relu(x)
#
#         return x


# Resnet code start
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', num_classes=400,
                 use_depthwise=False, loss_type=None, use_extra_layer=False, phase='train',
                 data_type='normal', policy='first'):
        self.inplanes = 64
        self.Detector_layer = None
        if loss_type == 'multiloss':
            self.Detector_layer = MultiDetector

        super(ResNet, self).__init__()
        self.sample_size = sample_size
        # if self.sample_size == 128:
        #     self.avgpool_128 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        #     sample_size = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=(1, 2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=(1, 2, 2))
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=(1, 2, 2))
        # last_duration = math.ceil(sample_duration / 16)
        last_duration = sample_duration
        last_size = math.ceil(sample_size / 32)
        kernel_size = (last_duration, last_size, last_size)
        if self.Detector_layer is not None:
            self.Detector_layer = self.Detector_layer(block, 512, kernel_size=kernel_size,
                                                      num_classes=num_classes, extra_layers=use_extra_layer,
                                                      phase=phase, data_type=data_type, policy=policy)
        else:
            self.avgpool = nn.AvgPool3d(kernel_size, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # m.eval()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=(1, 1, 1)):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                bn = nn.BatchNorm3d(planes * block.expansion)
                # bn.eval()
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    bn
                )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, boundaries=None):
        # x = x.cuda()
        # x = x.to(device)

        # if self.sample_size == 128:
        #     x = self.avgpool_128(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.Detector_layer is not None:
            out = self.Detector_layer(x, boundaries)
        else:
            x = self.avgpool(x)

            x = x.view(x.size(0), -1)
            out = self.fc(x)

        return out

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            pretrained = torch.load(base_file, map_location=lambda storage, loc: storage)['state_dict']
            pretrained = {"{}".format(s[7:]):v for s,v in pretrained.items()}
            current_param = self.state_dict()
            pretrained = {k: v for k, v in pretrained.items() if k in current_param and k[:2] != 'fc'}
            current_param.update(pretrained)
            print(pretrained.keys())
            # print(self.state_dict().keys())
            self.load_state_dict(current_param)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# ResNeXt code start
class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None, DS_Conv3d=None):
        super(ResNeXtBottleneck, self).__init__()
        # 19.8.6. remove
        # cardinality = planes와 Channelwise Conv 대체와 동일한 기능이므로 제거
        # if use_depthwise:
        #     cardinality = planes

        mid_planes = cardinality * int(planes / cardinality)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        if DS_Conv3d is not None:
            self.conv2 = DS_Conv3d(mid_planes, mid_planes,  kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        else:
            self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3,
                                   stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', cardinality=32, num_classes=400,
                 use_depthwise=False, loss_type=None, use_extra_layer=False, phase='train',
                 data_type='normal', policy='first'):
        self.inplanes = 64
        self.DS_Conv3d = None
        if use_depthwise:
            self.DS_Conv3d = DepthwiseSeparableConv(dimension=3)
        self.Detector_layer = None
        if loss_type == 'multiloss':
            self.Detector_layer = MultiDetector

        super(ResNeXt, self).__init__()
        self.sample_size = sample_size
        # if self.sample_size == 128:
        #     self.avgpool_128 = nn.AvgPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        #     sample_size = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, cardinality, stride=(1, 2, 2))
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type, cardinality, stride=(1, 2, 2))
        self.layer4 = self._make_layer(block, 1024, layers[3], shortcut_type, cardinality, stride=(1, 2, 2))
        # last_duration = math.ceil(sample_duration / 16)
        last_duration = sample_duration
        last_size = math.ceil(sample_size / 32)
        kernel_size = (last_duration, last_size, last_size)
        if self.Detector_layer is not None:
            self.Detector_layer = self.Detector_layer(block, cardinality * 32, kernel_size=kernel_size,
                                                      num_classes=num_classes, extra_layers=use_extra_layer,
                                                      phase=phase, data_type='normal', policy=policy)
        else:
            self.avgpool = nn.AvgPool3d(kernel_size, stride=1)
            self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # m.eval()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=(1, 1, 1)):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                bn = nn.BatchNorm3d(planes * block.expansion)
                # bn.eval()
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    bn
                )

        layers = list()
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample, DS_Conv3d=self.DS_Conv3d))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality, DS_Conv3d=self.DS_Conv3d))

        return nn.Sequential(*layers)

    def forward(self, x, boundaries=None):
        # x = x.to(device)
        # x = x.cuda()
        # if self.sample_size == 128:
        #     x = self.avgpool_128(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.Detector_layer is not None:
            out = self.Detector_layer(x, boundaries)
        else:
            x = self.avgpool(x)

            x = x.view(x.size(0), -1)
            out = self.fc(x)

        return out

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            pretrained = torch.load(base_file, map_location=lambda storage, loc: storage)['state_dict']
            pretrained = {"{}".format(s[7:]):v for s,v in pretrained.items()}
            current_param = self.state_dict()
            pretrained = {k: v for k, v in pretrained.items() if k in current_param and k[:2] != 'fc'}
            current_param.update(pretrained)
            print(pretrained.keys())
            # print(self.state_dict().keys())
            self.load_state_dict(current_param)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def get_detector(model_type, depth, **kwargs):
    assert model_type in ['resnet', 'resnext']
    if model_type == 'resnet':
        assert depth in [10, 18, 34, 50, 101, 152, 200]
        if depth == 10:
            return resnet34(**kwargs)
        elif depth == 18:
            return resnet18(**kwargs)
        elif depth == 34:
            return resnet34(**kwargs)
        elif depth == 50:
            return resnet50(**kwargs)
        elif depth == 101:
            return resnet101(**kwargs)
        elif depth == 152:
            return resnet152(**kwargs)
        elif depth == 200:
            return resnet200(**kwargs)
    else:
        assert depth in [50, 101, 152]
        if depth == 50:
            return resnext50(**kwargs)
        elif depth == 101:
            return resnext101(**kwargs)
        elif depth == 152:
            return resnext152(**kwargs)


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


def resnext50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model


if __name__ == '__main__':
    net = resnet50(num_classes=3, sample_size=128, sample_duration=16, use_depthwise=False, loss_type='multiloss')
    print(net)
    print("net length : ", len(list(net.children())))
