import torch.nn as nn


def DepthwiseSeparableConv(dimension=3, **kwargs):
    assert dimension in [2, 3]
    if dimension == 2:
        return DepthwiseSeparableConv2d(**kwargs)
    else:
        return DepthwiseSeparableConv3d(**kwargs)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()

        self.channelwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                     groups=in_planes, bias=bias)
        # self.bn1 = nn.BatchNorm2d(in_planes)
        # self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)
        # self.bn2 = nn.BatchNorm2d(out_planes)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.channelwise(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.pointwise(x)
        # x = self.bn2(x)
        # x = self.relu(x)

        return x


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False):
        super(DepthwiseSeparableConv3d, self).__init__()

        self.channelwise = nn.Conv3d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                     groups=in_planes, bias=bias)
        # self.bn1 = nn.BatchNorm3d(in_planes)
        # self.pointwise = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)
        # self.bn2 = nn.BatchNorm3d(out_planes)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.channelwise(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.pointwise(x)
        # x = self.bn2(x)
        # x = self.relu(x)

        return x