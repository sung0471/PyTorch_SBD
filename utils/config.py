import os
import argparse

parser: argparse.ArgumentParser
opt: argparse.ArgumentParser.parse_args


def set_dataset_cfg():
    phase = opt.phase
    dataset = opt.dataset
    if phase in ['train', 'full']:
        assert dataset == 'ClipShots'
    elif phase == 'test':
        assert dataset in ['ClipShots', 'RAI', 'TRECVID07']

    dataset_cfg = {
        "ClipShots": {
            "train_subdir": ['train', 'only_gradual'],
            "train_list_path": 'data/data_list/detector.txt',
            "gt_dir": 'annotations/test_191213.json'
        },
        "RAI": {
            "gt_dir": 'annotations/test.json'
        },
        "TRECVID": {
            "gt_dir": 'annotations/test.json'
            # "gt_dir": 'annotations/test2.json'
        }
    }
    dataset_name = dataset if dataset in ['ClipShots', 'RAI'] else dataset[:-2]
    dataset_path = dataset_cfg[dataset_name]

    if dataset == 'ClipShots':
        parser.add_argument('--train_list_path', default=dataset_path["train_list_path"], type=str)
        parser.add_argument('--train_subdir', type=list, default=dataset_path["train_subdir"],
                            help='subdirectory for training set')
    parser.add_argument('--gt_dir', default=dataset_path["gt_dir"], type=str,
                        help='directory contains ground truth for test set')


def set_model_cfg():
    model = opt.model
    assert model in ['alexnet', 'resnet', 'resnext', 'detector']

    if model == 'alexnet':
        parser.add_argument('--alexnet_type', default='dropout', type=str, help='origin | dropout')
    else:
        if model in ['resnet', 'resnext']:
            pass
        else:
            parser.add_argument('--use_extra_layer', default=True, help='if model==detector, use (True | False)')
            parser.add_argument('--baseline_model', default='resnet', type=str,
                                help='if you use detector, select baseline model(alexnet, resnet, resnext)')

        parser.add_argument('--model_depth', default=50, type=int,
                            help='only resnet-(18, 34, 50, 101, 152), resnext-101 are supported')


def set_optimizer_cfg():
    optimizer = opt.optimizer
    assert optimizer in ['sgd', 'adam']

    if optimizer == 'sgd':
        parser.add_argument('--learning_rate', default=1e-4, type=float,
                            help='Initial learning rate (divided by 10 while training by lr scheduler)')
        parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
        parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
        parser.set_defaults(nesterov=False)
    else:
        parser.add_argument('--learning_rate', default=2e-5, type=float,
                            help='Initial learning rate (divided by 10 while training by lr scheduler)')
        parser.add_argument('--betas', default=(0.9, 0.999), help="Adam's betas")
        parser.add_argument('--eps', default=1e-8, type=float, help="Adam's eps")

    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')


def set_loss_cfg():
    loss_type = opt.loss_type
    assert loss_type in ['normal', 'multiloss', 'KDloss']

    if loss_type == 'normal':
        pass
    elif loss_type == 'multiloss':
        parser.add_argument('--neg_threshold', default=(0.33, 0.5), help='negative threshold for cut, gradual')
        parser.add_argument('--nms_threshold', default=0.33, type=float, help='nms threshold')
    else:
        parser.add_argument('--KDloss_type', default='new', help='origin | new | dual')
        parser.add_argument('--teacher_model', default='alexnet', help='only alexnet supported')
        parser.add_argument('--teacher_model_path', default='models/Alexnet-final.pth', type=str,
                            help='Pretrained model (.pth)')
        parser.add_argument('--alexnet_type', default='dropout', type=str, help='origin | dropout')


def set_cfg(parser_class: argparse.ArgumentParser):
    global parser, opt
    parser = parser_class
    opt = parser_class.parse_known_args()[0]

    set_dataset_cfg()
    set_loss_cfg()
    set_optimizer_cfg()
    set_model_cfg()

    opt = parser.parse_args()

    # `19.10.8. add
    # check dataset and set opt.root_dir
    # `19.10.18 revise
    # opt.root_dir > opt.video_dir
    # `20.1.7. move from main_baseline.py/main()
    if opt.dataset[:-2] == 'TRECVID':
        dataset_path = os.path.join(opt.dataset[:-2], opt.dataset[-2:])
    else:
        dataset_path = opt.dataset
    opt.video_dir = os.path.join(opt.root_dir, dataset_path, opt.video_dir)
    opt.test_list_path = os.path.join(opt.root_dir, dataset_path, opt.test_list_path)
    opt.gt_dir = os.path.join(opt.root_dir, dataset_path, opt.gt_dir)

    # iter_per_epoch을 opt.is_full_data와 opt.batch_size에 맞게 자동으로 조정
    # `20.1.7. move from main_baseline.py/main()
    if opt.iter_per_epoch == 0:
        if opt.is_full_data:
            opt.iter_per_epoch = 500000
            # if not opt.use_extra_layer:
            #     opt.iter_per_epoch = 500000
            # else:
            #     opt.iter_per_epoch = 250000
        else:
            opt.iter_per_epoch = 70000
            # if not opt.use_extra_layer:
            #     opt.iter_per_epoch = 70000
            # else:
            #     opt.iter_per_epoch = 34000

    opt.iter_per_epoch = int(opt.iter_per_epoch / opt.batch_size)
    print("iter_per_epoch : {}, batch_size : {}".format(opt.iter_per_epoch, opt.batch_size))

    # set n_classes automatically
    # `20.1.7. move from main_baseline.py/main()
    assert opt.train_data_type in ['normal', 'cut', 'gradual']
    if opt.train_data_type == 'normal':
        opt.n_classes = 3
    else:
        opt.n_classes = 2
        if opt.train_data_type == 'cut':
            if opt.layer_policy == 'second':
                opt.neg_threshold = opt.neg_threshold[1]
            else:
                opt.neg_threshold = opt.neg_threshold[0]
        else:
            opt.neg_threshold = opt.neg_threshold[1]

    return opt
