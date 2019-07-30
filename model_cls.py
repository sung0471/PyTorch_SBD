import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models import *
from thop import profile
import os


# 19.6.26.
# add parameter=model_type instead opt.model
# 19.7.16.
# moved from models/__init__.py
def generate_model(opt, model_type):
    assert model_type in ['resnet', 'alexnet', 'resnext', 'detector']

    if model_type == 'alexnet':
        assert opt.alexnet_type in ['origin', 'dropout']
        model = deepSBD.deepSBD(model_type=opt.alexnet_type)
    elif model_type == 'resnet':
        from models.resnet import get_fine_tuning_parameters

        model = resnet.resnet18(num_classes=opt.n_classes,
                                sample_size=opt.sample_size, sample_duration=opt.sample_duration)
    elif model_type == 'resnext':
        model = resnext.resnext101(num_classes=opt.n_classes,
                                   sample_size=opt.sample_size, sample_duration=opt.sample_duration)
    elif model_type == 'detector':
        model = detector.resnext101(num_classes=opt.n_classes,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration, use_depthwise=True)
    else:
        raise Exception("Unknown model name")

    for_test_tensor = torch.randn(opt.batch_size, 3, opt.sample_duration, opt.sample_size, opt.sample_size)
    flops, params = profile(model, inputs=(for_test_tensor,))
    print('Model : {}, (FLOPS: {}, Params: {})'.format(model_type, flops, params))

    return model


# 19.6.26.
# add parameter=model_type
# for knowledge distillation
def build_model(opt, model_type, phase, device):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return

    # num_classes = opt.n_classes
    # 19.6.26. add 'model_type' parameter
    model = generate_model(opt, model_type)

    # model=gradual_cls(opt.sample_duration,opt.sample_size,opt.sample_size,model,num_classes)
    # print(model)
    if opt.pretrained_model:
        print("use pretrained model")
        if phase == 'train' and opt.pretrain_path:
            model.load_weights(opt.pretrain_path)
    else:
        print("no pretrained model")

    # `19.3.8
    # model = model.cuda(device)

    # `19.6.4
    # remove opt.model_type > 'new' is not trainable
    # 19.6.26.
    # opt.model_type = 'new' 다시 사용
    # benchmark도 전체적으로 적용하여 재시도
    # 19.6.28. remove opt.model_type
    # if opt.cuda and opt.model_type == 'old':
    if opt.cuda:
        # `19.6.24. 다시 추가
        # `19.7.1. 주석 처리
        # 19.7.10. benchmark=modelwise 재사용
        torch.backends.benchmark = True

        # `19.??
        # Parallel > model.to(device) 순서로 설정
        # `19.5.14.
        # use multi_gpu for training and testing
        model = nn.DataParallel(model, device_ids=range(opt.gpu_num))
        # `19.7.2.
        # model.to(device) > model = model.to(device)로 변경
        # `19.7.7.
        # model = model.to(device)를 main_baseline.py로 이동
        # 19.7.10.
        # model.inplace를 다시 modelwise로 rollback
        # model.cuda()
        # model.to(device)
        model = model.to(device)

        # `19.6.27.
        # use model.to(device)>Parallel 순서로 설정
        # `19.7.2. 주석처리
        # # model.cuda()
        # # model = model.to(device)
        # model.to(device)
        # # `19.5.14.
        # # use multi_gpu for training and testing
        # model = nn.DataParallel(model, device_ids=range(opt.gpu_num))

    if phase == 'train':
        # for debug
        # print(opt.gpu_num)

        # `19.3.8
        # I'll use only one gpu
        # model = nn.DataParallel(model, device_ids=range(1))
        # use multi gpu for only training
        # model = nn.DataParallel(model, device_ids=range(opt.gpu_num))
        model.train()
    else:
        # model = nn.DataParallel(model, device_ids=range(1))
        model.eval()

    return model
