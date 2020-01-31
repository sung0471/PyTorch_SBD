import argparse
from utils.config import set_cfg


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='test', type=str, help='train | test | full')
    parser.add_argument('--misaeng', default=False, help='if true, test misaeng')
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', help='If true, cuda is used.')
    parser.set_defaults(cuda=True)
    parser.add_argument('--root_dir', default='data', type=str, help='Root directory path of data')
    parser.add_argument('--dataset', default='ClipShots', type=str, help='ClipShots | RAI | TRECVID07')
    parser.add_argument('--video_dir', default='videos', type=str, help='Root directory path of data')
    # parser.add_argument('--train_list_path', default='data/data_list/detector.txt', type=str)
    parser.add_argument('--test_list_path', default='video_lists/test.txt', type=str, help='test list path')
    # parser.add_argument('--train_subdir', type=list, default=['train', 'only_gradual'],
    #                     help='subdirectory for training set')
    # parser.add_argument('--gt_dir', default='annotations/test_191213.json', type=str,
    #                     help='directory contains ground truth for test set')
    parser.add_argument('--test_subdir', type=str, default='test', help='subdirectory for testing set')
    parser.add_argument('--result_dir', default='results', type=str, help='Result directory path')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--test_weight', default='results/model_final.pth', type=str, help='test model path')
    parser.add_argument('--input_type', default='RGB', help='RGB | HSV')
    parser.add_argument('--is_full_data', default=True, help='explain whether full data or not')
    parser.add_argument('--train_data_type', default='cut', type=str, help='normal(cut, gradual) | cut | gradual')
    parser.add_argument('--layer_policy', default='second', type=str, help='first | second')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes, automatic set by train_data_type'
                                                                 'if train_data_type=="normal", 3'
                                                                 'else, 2')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='when training start with different batch size, adjust this value, else 0')
    parser.add_argument('--iter_per_epoch', default=0, type=int,
                        help='if iter=0, adjust automatic, elif iter>0, set this value')
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--model', default='detector', type=str, help='alexnet | resnet | resnext | detector')
    # parser.add_argument('--use_extra_layer', default=True, help='if model==detector, use (True | False)')
    # parser.add_argument('--baseline_model', default='resnet', type=str,
    #                     help='if you use detector, select baseline model(alexnet, resnet, resnext)')
    # parser.add_argument('--model_depth', default=50, type=int,
    #                     help='only resnet-(18, 34, 50, 101, 152), resnext-101 are supported')
    # parser.add_argument('--model_type', default='old', type=str, help='old | new'
    #                                                                   'old: model>parallel>cuda>train'
    #                                                                   'new: model>train>parallel>cuda')
    # parser.add_argument('--alexnet_type', default='dropout', type=str, help='origin | dropout')
    parser.add_argument('--pretrained_model', default=True, help='if true, use pre-trained model')
    parser.add_argument('--pretrained_dir', default='kinetics_pretrained_model/', type=str, help='pre-trained model dir')
    parser.add_argument('--loss_type', default='multiloss', help='normal(cross entropy)'
                                                                 'multiloss(cross entropy + regression)'
                                                                 'KDloss(teacher student loss)')
    # parser.add_argument('--KDloss_type', default='new', help='origin | new | dual')
    # parser.add_argument('--teacher_model', default='alexnet', help='only alexnet supported')
    # parser.add_argument('--teacher_model_path', default='models/Alexnet-final.pth', type=str,
    #                     help='Pretrained model (.pth)')
    parser.add_argument('--candidate', default=False, help='if true, use candidate extraction')
    parser.add_argument('--sample_size', default=64, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=2, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--use_save_timing', default=False,
                        help='if True, adjust save timing from 2000 to 5000. else, iter_per_epoch / 5')
    parser.add_argument('--shuffle', default=True, help="shuffle the dataset")
    # parser.add_argument('--learning_rate', default=2e-5, type=float,
    #                     help='Initial learning rate (divided by 10 while training by lr scheduler)')
    # parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    # parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    # parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    # parser.set_defaults(nesterov=False)
    # parser.add_argument('--betas', default=(0.9, 0.999), help="Adam's betas")
    # parser.add_argument('--eps', default=1e-8, type=float, help="Adam's eps")
    parser.add_argument('--optimizer', default='adam', type=str, help='sgd | adam')
    # parser.add_argument('--neg_threshold', default=(0.33, 0.5), help='negative threshold for cut, gradual')
    # parser.add_argument('--nms_threshold', default=0.33, type=float, help='nms threshold')
    parser.add_argument('--lr_patience', default=10, type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--norm_value', default=1, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--manual_seed', type=int, default=16)
    parser.add_argument('--n_scales', default=5, type=int,
                        help='Number of scales for multiscale cropping')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--train', action='store_true', help='If true, training is performed.')
    parser.set_defaults(train=True)
    parser.set_defaults(no_val=False)
    parser.set_defaults(test=False)
    # args = parser.parse_args()
    args = set_cfg(parser)

    return args


def parse_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True, type=str, help='Root directory path of data')
    parser.add_argument('--test_list_path', default='data/ClipShots/Video_lists/test.txt', type=str,
                        help='test list path')
    parser.add_argument('--model', default='resnet', type=str)
    parser.add_argument('--weights', default='resnet', type=str)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--n_classes', default=3, type=int,
                        help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--norm_value', default=1, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--sample_duration', default=32, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--gt_dir', type=str)
    parser.add_argument('--spatial_size', type=int, default=128)
    parser.add_argument('--test_subdir', type=str, default='test')
    parser.add_argument('--auto_resume', action='store_true')
    args = parser.parse_args()
    return args


def parse_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path', type=str)
    parser.add_argument('--gt_dir', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_opts()

    print(opt)