import os
import json
from torch import nn
from torch import optim

from opts import parse_opts
from lib.spatial_transforms import *

from data.train_data_loader import DataSet as train_DataSet
from data.test_data_loader import DataSet as test_DataSet
from cls import build_model
import time

from lib.utils import AverageMeter, calculate_accuracy
from torch.autograd import Variable
from torch.optim import lr_scheduler

import cv2

from candidate_extracting import candidate_extraction
from raw import eval_res

from tensorboardX import SummaryWriter
writer = SummaryWriter()


def get_mean(norm_value=255):
    return [114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value]


def get_label(res_tensor):
    res_numpy=res_tensor.data.cpu().numpy()
    labels=[]
    for row in res_numpy:
        labels.append(np.argmax(row))
    return labels


def get_labels_from_candidate(video, temporal_length, model, spatial_transform, batch_size, device, boundary_index, **args):
    print(boundary_index)
    clip_batch = []
    labels = []
    all_frames = []
    info_boundary = []

    print("[INFO] transform video for test")
    video_length = 0
    for i, im in enumerate(video):
        frame = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).convert('RGB')
        frame = spatial_transform(frame)
        all_frames.append(frame)
        video_length+=1

    print("[INFO] start video test")
    for i, candidate_frame_number in enumerate(boundary_index):
        start_frame = int(candidate_frame_number-(temporal_length/2-1))
        start_frame = 0 if start_frame < 0 else start_frame
        end_frame = int(candidate_frame_number+(temporal_length/2)+1)
        end_frame = video_length if end_frame > video_length else end_frame

        image_clip = all_frames[start_frame:end_frame]
        info_boundary.append([start_frame, end_frame])
        image_clip += [image_clip[-1] for _ in range(temporal_length - len(image_clip))]

        if len(image_clip) == temporal_length:
            clip = torch.stack(image_clip, 0).permute(1, 0, 2, 3)
            clip_batch.append(clip)
            # image_clip = image_clip[int(temporal_length / 2):]

        if len(clip_batch) == batch_size or i==(len(boundary_index)-1):
            clip_tensor = torch.stack(clip_batch, 0)
            # Alexnet
            # clip_tensor = Variable(clip_tensor)
            # resnet
            clip_tensor = Variable(clip_tensor).cuda(device)
            results = model(clip_tensor)
            labels += get_label(results)
            clip_batch = []

    print("[INFO] get predicted label")
    res = [0]*video_length
    for i, label in enumerate(labels):
        range_of_frames = info_boundary[i]
        for j in range(range_of_frames[0],range_of_frames[1]):
            res[j] = label if label == 1 or res[j] == 0 else res[j]

    final_res = []
    i = 0
    while i < len(res):
        if res[i] > 0:
            label = res[i]
            begin = i
            i += 1
            while i < len(res) and res[i] == res[i - 1]:
                i += 1
            end = i - 1
            final_res.append((begin, end, label))
        else:
            i += 1
    return final_res


def load_checkpoint(model, opt_model):
    if opt_model=='alexnet':
        path = 'models/Alexnet-final.pth'
    elif opt_model=='resnet':
        path = 'results/model_final.pth'
    else:
        print("[ERR] incorrect opt.model : ", opt_model)
        assert False
    print("load model... : ", opt_model)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])


def test_misaeng():
    opt = parse_opts()

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)

    opt.mean = get_mean(opt.norm_value)
    print(opt)

    torch.manual_seed(opt.manual_seed)

    # 19.3.8. add
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("cuda is available : ", torch.cuda.is_available())

    root_dir = 'misaeng_test'
    video_file = '1001.0001.0001.0001.0008.mp4'
    video_name = str(os.path.splitext(video_file)[0])
    video_full_name = os.path.join(root_dir, video_name)

    # input video (imageio - sometimes error)
    # imageio.plugins.ffmpeg.download()
    # video = imageio.get_reader('test.mp4')

    print('[INFO] extracting candidate')
    video, num_frame, fps, boundary_index = candidate_extraction(root_dir, video_file, device)

    print('[INFO] testing deepSBD')
    print("[INFO] test video : {}".format(video_name), flush=True)

    model = build_model(opt, 'test', device)
    print(model)
    load_checkpoint(model, opt.model)
    spatial_transforms = get_test_spatial_transform(opt)
    res = {}

    # 19.4.11.
    # opt.sample_duration must be 16 in alexnet
    labels = get_labels_from_candidate(video, opt.sample_duration, model,
                                       spatial_transforms, opt.batch_size, device, boundary_index[1:-1])
    # labels = deepSBD(video_dir, opt.sample_duration, model,
    #                  spatial_transforms, opt.batch_size, device)

    boundary_index_final = []
    _res = {'cut': [], 'gradual': []}
    for begin, end, label in labels:
        if label == 2:
            _res['cut'].append((begin, end))
        else:
            _res['gradual'].append((begin, end))
        boundary_index_final.append([begin, end, label])
    res[video_name] = _res
    boundary_index_final.insert(0, 0)
    boundary_index_final.append(num_frame)

    print(res)
    print(boundary_index_final)
    srt_index=0
    do_srt = True
    if do_srt:
        with open(video_full_name + '.final.srt', 'w', encoding='utf-8') as f:
            for bound_ind in range(len(boundary_index_final) - 1):
                if type(boundary_index_final[bound_ind])==list:
                    transition='cut' if boundary_index_final[bound_ind][2]==2 else 'gradual'
                    startframe = boundary_index_final[bound_ind][0]
                    endframe = boundary_index_final[bound_ind][1]
                    starttime = float(startframe / fps)
                    endtime = float(endframe / fps)
                    f.write(str(srt_index) + '\n')
                    f.write(str(starttime) + ' --> ' + str(endtime) + '\n')
                    f.write(transition + ': frame [' + str(startframe) + ',' + str(endframe) + ']\n')
                    f.write('\n')

                startframe = boundary_index_final[bound_ind][1] if bound_ind!=0 else 0.0
                endframe = boundary_index_final[bound_ind + 1][0] if bound_ind!=(len(boundary_index_final)-2) else boundary_index_final[bound_ind + 1]
                starttime = float(startframe / fps)
                endtime = float(endframe / fps)
                f.write(str(srt_index) + '\n')
                f.write(str(starttime) + ' --> ' + str(endtime) + '\n')
                f.write('shot# ' + str(bound_ind) + ' & frame [' + str(startframe) +',' + str(endframe) + ']\n')
                f.write('\n')

                srt_index += 1


def test_dataset():
    opt = parse_opts()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(opt, "test", device)
    load_checkpoint(model, opt.model)
    model.cuda(device)
    model.eval()

    spatial_transform = get_test_spatial_transform(opt)
    temporal_transform = None
    target_transform = None
    # list_root_path : train path, only_gradual path
    # `19.3.7 : add only_gradual path
    root_dir = os.path.join(opt.root_dir, opt.test_subdir)
    print(root_dir)
    print(opt.test_list_path)
    with open(opt.test_list_path,'r') as f:
        video_name_list = [line.strip('\n') for line in f.readlines()]

    res = {}
    print('\n====> Testing Start')
    for idx, videoname in enumerate(video_name_list):
        print("Process {} {}".format(idx, videoname), flush=True)
        test_data = test_DataSet(os.path.join(root_dir, videoname),
                                 spatial_transform=spatial_transform,
                                 temporal_transform=temporal_transform,
                                 target_transform=target_transform,
                                 sample_duration=opt.sample_duration)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                       num_workers=opt.n_threads, pin_memory=True)

        labels = []
        for _, clip in enumerate(test_data_loader):
            clip = Variable(clip).cuda(device)
            results = model(clip)
            labels += get_label(results)

        final_res = []
        i = 0
        while i < len(labels):
            if labels[i] > 0:
                label = labels[i]
                begin = i
                i += 1
                while i < len(labels) and labels[i] == labels[i - 1]:
                    i += 1
                end = i - 1
                final_res.append((begin * opt.sample_duration / 2 + 1, end * opt.sample_duration / 2 + 16 + 1, label))
            else:
                i += 1

        _res = {'cut': [], 'gradual': []}
        for begin, end, label in final_res:
            if label == 2:
                _res['cut'].append((begin, end))
            else:
                _res['gradual'].append((begin, end))
        res[videoname] = _res

    print("[INFO] finish test!!")
    out_path = os.path.join(opt.result_dir, 'results.json')
    if not os.path.exists(out_path):
        json.dump(res, open(out_path, 'w'))
    eval_res.eval(out_path, opt.gt_dir)


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data

    return n_correct_elems / batch_size


# 19.3.8 revision
# add parameter : "device"
def train(cur_iter, total_iter, data_loader, model, criterion, optimizer, scheduler, opt, device):
    model.train()

    # 19.3.14. add
    print("device : ", torch.cuda.get_device_name(0))
    # torch.set_default_tensor_type('torch.cuda.DoubleTensor')\

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    i = cur_iter

    # for debug
    # print(not(opt.no_cuda)) : True
    print('\n====> Training Start')
    while i < total_iter:
        for _, (inputs, targets) in enumerate(data_loader):

            # 19.3.7 add
            # if not opt.no_cuda:
            #     targets = targets.cuda(async=True)
            #     inputs = inputs.cuda(async=True)

            # 19.3.8. revision
            if not opt.no_cuda:
                targets = targets.cuda(device, async=True)

            targets = Variable(targets)
            inputs = Variable(inputs)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            acc = calculate_accuracy(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.data)

            print('Iter:{} Loss_conf:{} acc:{} lr:{}'.format(
                i + 1, loss.data, acc, optimizer.param_groups[0]['lr']), flush=True)
            i += 1

            if i % 2000 == 0:
                save_file_path = os.path.join(opt.result_dir, 'model_iter{}.pth'.format(i))
                print("save to {}".format(save_file_path))
                states = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_file_path)
            if i >= total_iter:
                break

    save_file_path = os.path.join(opt.result_dir, 'model_final.pth'.format(opt.checkpoint_path))
    print("save to {}".format(save_file_path))
    states = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path)


def get_lastest_model(opt):
    if opt.resume_path != '':
        return 0
    if os.path.exists(os.path.join(opt.result_dir, 'model_final.pth')):
        opt.resume_path = os.path.join(opt.result_dir, 'model_final.pth')
        return opt.total_iter

    iter_num = -1
    for filename in os.listdir(opt.result_dir):
        if filename[-3:] == 'pth':
            _iter_num = int(filename[len('model_iter'):-4])
            iter_num = max(iter_num, _iter_num)
    if iter_num > 0:
        opt.resume_path = os.path.join(opt.result_dir, 'model_iter{}.pth'.format(iter_num))
    return iter_num


def train_misaeng():
    opt = parse_opts()

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)

    opt.mean = get_mean(opt.norm_value)
    print(opt)

    torch.manual_seed(opt.manual_seed)

    # 19.3.8. add
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("cuda is available : ", torch.cuda.is_available())

    print('[INFO] training resnet')
    model = build_model(opt, 'train', device)

    cur_iter = 0
    if opt.auto_resume and opt.resume_path == '':
        cur_iter = get_lastest_model(opt)
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        model.load_state_dict(checkpoint['state_dict'])

    parameters = model.parameters()
    criterion = nn.CrossEntropyLoss()
    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.momentum

    optimizer = optim.SGD(parameters, lr=opt.learning_rate,
                          momentum=opt.momentum, dampening=dampening,
                          weight_decay=opt.weight_decay, nesterov=opt.nesterov)

    # 19.3.8 revision
    if not opt.no_cuda:
        criterion = criterion.cuda(device)

    if not opt.no_train:
        spatial_transform = get_train_spatial_transform(opt)
        temporal_transform = None
        target_transform = None
        # list_root_path : train path, only_gradual path
        # `19.3.7 : add only_gradual path
        list_root_path = list()
        list_root_path.append(os.path.join(opt.root_dir, opt.train_subdir))
        list_root_path.append(os.path.join(opt.root_dir, 'only_gradual'))
        print(list_root_path)
        print("[INFO] reading : ", opt.video_list_path)
        training_data = train_DataSet(list_root_path, opt.video_list_path,
                                      spatial_transform=spatial_transform,
                                      temporal_transform=temporal_transform,
                                      target_transform=target_transform, sample_duration=opt.sample_duration)

        weights = torch.DoubleTensor(training_data.weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=opt.batch_size,
                                                           num_workers=opt.n_threads, sampler=sampler, pin_memory=True)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=60000)

        # 19.3.8. add
        # train(cur_iter,opt.total_iter,training_data_loader, model, criterion, optimizer,scheduler,opt)
        train(cur_iter, opt.total_iter, training_data_loader, model, criterion, optimizer, scheduler, opt, device)


if __name__ == '__main__':
    # train_misaeng()
    test_dataset()
    # test_misaeng()
