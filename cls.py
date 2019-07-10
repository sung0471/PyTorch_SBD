import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models import generate_model
import os


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
        # torch.backends.benchmark = True

        # `19.??
        # Parallel > model.to(device) 순서로 설정
        # `19.5.14.
        # use multi_gpu for training and testing
        model = nn.DataParallel(model, device_ids=range(opt.gpu_num))
        # `19.7.2.
        # model.to(device) > model = model.to(device)로 변경
        model = model.to(device)
        # model.cuda()
        # model.to(device)

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
