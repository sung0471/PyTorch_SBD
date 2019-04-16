# -*- coding: utf-8 -*-
#https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c

from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import torch
import torch.nn.functional as F

class Rescale(object):
    """Rescale a image to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image) : PIL.Image object to rescale
        """
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img

class ResNetFeature(nn.Module): #feature_size: 2048
    def __init__(self, feature='resnet101'):
        """
        Args:
            feature (string): resnet101 or resnet152
        """
        super(ResNetFeature, self).__init__()
        if feature == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        else:
            resnet = models.resnet34(pretrained=True) #resnet152
        resnet.float()
        resnet.cuda()
        resnet.eval()

        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[:-2])
        self.pool5 = module_list[-2]

    def forward(self, x):
        res5c = self.conv5(x.cuda())
        pool5 = self.pool5(res5c)
        pool5 = pool5.view(pool5.size(0), -1)
        return res5c, pool5

resnet_transform = transforms.Compose([
    transforms.Scale((224, 224)),# Rescale(224, 224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

## options #Pad(4) #RandomHorizontalFlip() #RandomCrop([32, 32])

class InceptionFeature(nn.Module): #feature_size: 2048 --> 1024
    def __init__(self, feature='inception_v3'):
        """
        Args:
            feature (string): Inception v3
        """
        super(InceptionFeature, self).__init__()
        self.model = models.inception_v3(pretrained=True)

        # self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()
        # tf slim's inception implementation has a named prelogits layer.
        # torchvision's does the pooling across the mixed_7c layers as an anonymous op.
        # so if we want the output from that prelogits layer we need to get the mixed_7c output and then do the pooling ourselves.
        layer = self.model._modules.get('Mixed_7c')
        # this is where we'll copy the prelogits
        self.prelogits = torch.zeros(2048)
        # add a hook that copies the result of the 
        def copy_prelogits(m, i, o):
            # pool the result of the mixed_7c
            # inception layers
            v = F.avg_pool2d(o.data, kernel_size=8)
            v = v.squeeze()           
            self.prelogits.copy_(v.data)#(v.data)

        # install the copy data hook
        hook = layer.register_forward_hook(copy_prelogits)
        # we're not training
        self.model.eval()

    # given an image, return the "prelogits"
    def forward(self, x):
        img = x.cuda()
        # run inference
        self.model(img)
        # copy the prelogits out of our placeholder
        d_layer = nn.Linear(2048, 1024)#.cuda()
        feature = d_layer(self.prelogits.clone())

        return feature#.numpy()

inception_transform = transforms.Compose([
    transforms.Scale((299, 299)), #Rescale(299, 299), #
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SqueezeNetFeature(nn.Module): #feature_size: 86528-> 1024
    def __init__(self, feature='squeezenet1_1'):
        super(SqueezeNetFeature, self).__init__()
        squeezenet = models.squeezenet1_1(pretrained=True)
        # squeezenet.float()
        squeezenet.cuda()
        squeezenet.eval()

        module_list = list(squeezenet.children())
        self.final_conv  = nn.Sequential(*module_list[:-1]) # last layer:86528 feature size

    def forward(self, x):
        final_conv  = self.final_conv(x.cuda())
        dense_layer = nn.Linear(86528, 1024).cuda()
        final_conv  = final_conv.view(final_conv.size(0), -1)           
        final_feature = dense_layer(final_conv)
       
        return final_feature

squeezenet_transform = transforms.Compose([
    transforms.Scale((224, 224)),
    # transforms.Scale(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

## https://www.kaggle.com/renatobmlr/pytorch-densenet-as-feature-extractor
class MyDenseNetConv(nn.Module): #feature_size: 1024
    def __init__(self, fixed_extractor = True):
        super(MyDenseNetConv,self).__init__()
        # torch.cuda.set_device(1)
        self.device = torch.device('cuda:0') # in pytorch > 0.4.0

        original_model = models.densenet121(pretrained=True)
        original_model = original_model.cuda().to(self.device)  #cpu: without cuda()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
        if fixed_extractor:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x.cuda().to(self.device)) #cpu: without cuda()
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x

class MyDenseNetDens(nn.Module):
    def __init__(self, nb_out=28):
        super().__init__()
        self.dens1 = nn.Linear(in_features=2208, out_features=512)
        self.dens2 = nn.Linear(in_features=512, out_features=128)
        self.dens3 = nn.Linear(in_features=128, out_features=nb_out)
        
    def forward(self, x):
        x = self.dens1(x.cuda())
        x = nn.functional.selu(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.dens2(x)
        x = nn.functional.selu(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.dens3(x)
        return x

class MyDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mrnc = MyDenseNetConv()
        self.mrnd = MyDenseNetDens()
    def forward(self, x):
        x = self.mrnc(x)
        x = self.mrnd(x)
        return x 

densenet_transform = transforms.Compose([
    transforms.Scale((224, 224)), # Rescale(224, 224), #
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])