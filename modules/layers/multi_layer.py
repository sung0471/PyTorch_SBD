import torch.nn as nn


class MultiClassifier(nn.Module):
    def __init__(self, in_planes, kernel_size=3, num_classes=3):
        super(MultiClassifier, self).__init__()

        self.loc_layer = nn.Conv3d(in_planes, 2,
                                   kernel_size=kernel_size, padding=0)
        self.conf_avgpool = nn.AvgPool3d(kernel_size, stride=1)
        self.conf_layer = nn.Linear(in_planes, num_classes)

    def forward(self, x):
        loc_x = self.loc_layer(x)

        conf_pool = self.conf_avgpool(x)
        conf_pool = conf_pool.view(conf_pool.size(0), -1)
        conf_x = self.conf_layer(conf_pool)

        out = (loc_x, conf_x)
        return out
