import torch
from torch import nn

from models.deepSBD import deepSBD
import models.resnet as resnet
import models.resnext as resnext
import models.detector as detector


# 19.6.26.
# add parameter=model_type instead opt.model
def generate_model(opt, model_type):
    assert model_type in ['resnet', 'alexnet', 'resnext', 'detector']
    assert opt.alexnet_type in ['origin', 'dropout']

    if model_type == 'alexnet':
        model = deepSBD(model_type=opt.alexnet_type)
    elif model_type == 'resnet':
        from models.resnet import get_fine_tuning_parameters

        model = resnet.resnet18(num_classes=opt.n_classes,
                                sample_size=opt.sample_size, sample_duration=opt.sample_duration)
    elif model_type == 'resnext':
        model = resnext.resnet101(num_classes=opt.n_classes,
                                  sample_size=opt.sample_size, sample_duration=opt.sample_duration)
    elif model_type == 'detector':
        model = detector.resnet101(num_classes=opt.n_classes,
                                  sample_size=opt.sample_size, sample_duration=opt.sample_duration, use_depthwise=True)
    else:
        raise Exception("Unknown model name")

    return model
