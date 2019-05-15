from torchvision import models
import torch.nn as nn
import torch

class SqueezeNetFeature(nn.Module):  # feature_size: 86528 -> 1024
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __init__(self, feature='squeezenet1_1'):
        super(SqueezeNetFeature, self).__init__()
        squeezenet = models.squeezenet1_1(pretrained=True)
        # squeezenet.float()
        squeezenet.to(self.device)
        squeezenet.eval()

        module_list = list(squeezenet.children())
        self.final_conv = nn.Sequential(*module_list[:-1])  # last layer:86528 feature size

    def forward(self, x):
        # x = x.to(self.device)
        final_conv = self.final_conv(x)
        # dense_layer = nn.Linear(86528, 1024).cuda()
        # final_conv = final_conv.view(final_conv.size(0), -1)
        # final_feature = dense_layer(final_conv)

        return final_conv
