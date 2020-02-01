import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models import *
from thop import profile
import os
import copy


# 19.6.26. add parameter=model_type instead opt.model
# 19.7.16. moved from models/__init__.py
def generate_model(opt, model_type):
    assert model_type in ['resnet', 'alexnet', 'resnext', 'detector']

    if model_type == 'alexnet':
        assert opt.alexnet_type in ['origin', 'dropout']
        model = deepSBD.deepSBD(model_type=opt.alexnet_type)
    else:
        assert opt.model_depth in [18, 34, 50, 101, 152]
        if model_type == 'resnet':
            from models.resnet import get_fine_tuning_parameters
            model = resnet.get_resnet(opt.model_depth, num_classes=opt.n_classes,
                                      sample_size=opt.sample_size, sample_duration=opt.sample_duration)
        elif model_type == 'resnext':
            model = resnext.get_resnext(opt.model_depth, num_classes=opt.n_classes,
                                       sample_size=opt.sample_size, sample_duration=opt.sample_duration)
        else:
            model = detector.get_detector(opt.baseline_model, opt.model_depth,
                                          num_classes=opt.n_classes, sample_size=opt.sample_size,
                                          sample_duration=opt.sample_duration, use_depthwise=False,
                                          loss_type=opt.loss_type, use_extra_layer=opt.use_extra_layer,
                                          phase=opt.phase, data_type=opt.train_data_type, policy=opt.layer_policy)

    # 19.7.31. add deepcopy
    test_model = copy.deepcopy(model).to(opt.device)

    for_test_tensor = torch.randn(opt.batch_size, 3, opt.sample_duration, opt.sample_size, opt.sample_size).to(opt.device)
    if 'use_extra_layer' not in vars(opt).keys() or not opt.use_extra_layer:
        flops, params = profile(test_model, inputs=(for_test_tensor,))
    else:
        start_boundaries = torch.zeros(opt.batch_size).to(opt.device)
        for i in range(opt.batch_size):
            start_boundaries[i] = i * opt.sample_duration / 2
        flops, params = profile(test_model, inputs=(for_test_tensor, start_boundaries))
    print('Model : {}, (FLOPS: {}, Params: {})'.format(model_type, flops, params))

    return model


# 19.6.26. add parameter=model_type
# for knowledge distillation
def build_model(opt, model_type, phase):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return

    # num_classes = opt.n_classes
    # 19.6.26. add 'model_type' parameter
    model = generate_model(opt, model_type)

    # model=gradual_cls(opt.sample_duration,opt.sample_size,opt.sample_size,model,num_classes)
    # 19.5.15 if not opt.no_pretrained_model 추가
    # 19.5.23 opt.no_pretrained_model > opt.pretrained_model로 변경
    # 19.8.5 train에서만 사전학습 모델을 불러오도록 수정
    if phase == 'train' and opt.pretrained_model:
        # 19.8.2 pretrained_path 설정하는 부분 추가
        pretrained_model_type = model_type if model_type not in ['detector'] else opt.baseline_model
        pretrained_model_name = pretrained_model_type + '-' + str(opt.model_depth) + '-kinetics.pth'
        pretrained_path = os.path.join(opt.pretrained_dir, pretrained_model_name)
        if os.path.exists(pretrained_path):
            print("use pretrained model")
            model.load_weights(pretrained_path)
        else:
            raise Exception("there is no pretrained model : {}".format(pretrained_path))
    else:
        print("no pretrained model")

    # 19.4.17 model > cuda > parallel > train 순서
    # 19.5.15 model > benchmark > cuda > parallel > train 순서 (model_cls.py/build_model())
    # 19.5.16 model > train > parallel > benchmark > cuda 순서 (main_baseline.py/train_dataset)
    # 19.5.16 benchmark > model > cuda > parallel > train 순서 (model_cls.py/build_model())
    # 19.5.20 opt.model_type == 'old' 조건 추가
    # if opt.no_cuda and opt.model_type == 'old':
    # model_type=='new', benchmark > model > train > parallel > cuda 순서 (main_baseline.py/build_final_model())
    # model_type=='old', benchmark > model > cuda > parallel > train 순서 (model_cls.py/build_model())
    # 19.5.23 opt.no_cuda > opt.cuda로 변경
    # 19.5.30(부정확) benchmark > model > parallel > cuda > train 순서 (model_cls.py/build_model())
    # 19.6.4 remove opt.model_type == 'old' : 'new' is not trainable
    # 19.6.24 model > benchmark > parallel > cuda > train 순서 (model_cls.py/build_model())
    # 19.6.26. opt.model_type == 'new' 다시 사용
    # benchmark도 전체적으로 적용하여 재시도 (model_type=new일 때, main_baseline.py/build_final_model())
    # model_type=='new', model > train > benchmark > parallel > cuda (main_baseline.py/build_final_model())
    # model_type=='old', model > benchmark > parallel > cuda > train 순서 (model_cls.py/build_model())
    # 19.6.28. remove opt.model_type == 'old'
    # if opt.cuda and opt.model_type == 'old':
    if opt.cuda:
        # 19.5.15 benchmark 추가(model.to(device) 위에)
        # 19.5.16 main_baseline.py/train_dataset() > main()으로 이동
        # 19.6.24 다시 추가 (model_cls.py/build_model())
        # 19.6.26 model generate 후에 적용되게 수정 (model_type=old일 때, main_baseline.py/main())
        # 19.7.1 main_baseline.py/main()으로 이동
        # 19.7.10 benchmark=modelwise 방식으로 다시 사용 (model_cls.py/build_model())
        torch.backends.benchmark = True

        # 원본 phase='train'과 phase='test'에 개별로 동작
        # 19.5.14 use multi-gpu for training and testing
        # test의 device_ids=range(1) > device_ids=range(opt.gpu_num)로 수정
        # 19.5.15 if phase == 'train' 위로 이동(개별로 동작하던 것을 통합)
        # 19.5.16 main_baseline.py/main()으로 이동 / 다시 롤백
        # 19.5.20 if opt.no_cuda에서 실행되게 변경 (model_type=old일 때 조건이므로, 기능은 변화없음)
        # 19.5.30(정확한 날짜X) Parallel > model.to(device) 순서로 변경
        model = nn.DataParallel(model, device_ids=range(opt.gpu_num))
        # origin    model = model.cuda()
        # 19.3.8    model.to(device)
        # 19.4.17   model = model.cuda(device)
        # 19.5.15 코드 수정 및 benchmark 추가
        #           model.to(device)
        # 19.5.16 main()으로 이동 / 다시 롤백
        # 19.5.16   model = model.to(device)
        # 19.5.30(정확한 날짜X) Parallel > model = model.to(device) 순서로 변경
        # 19.5.31   model.to(device)
        # 19.7.2    model = model.to(device)
        # 19.7.7 main_baseline.py/main으로 이동
        # 19.7.10 model.inplace를 다시 modelwise로 rollback
        # 19.10.18 device > opt.device
        model = model.to(opt.device)

    if phase == 'train':
        model.train()
    else:
        model.eval()

    return model
