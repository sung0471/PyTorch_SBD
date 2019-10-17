#-*- coding: utf-8 -*-

import os
import json
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from lib.spatial_transforms import *

from data.train_data_loader import DataSet as train_DataSet
from data.test_data_loader import DataSet as test_DataSet
import time
import datetime

from lib.utils import AverageMeter, calculate_accuracy, Configure, nms
from lib.pickle_utils import PickleUtils
from modules.teacher_student_module import TeacherStudentModule
from modules.knowledge_distillation_loss import KDloss
from modules.multiloss import MultiLoss
from model_cls import build_model

import cv2
from tensorboardX import SummaryWriter
import eval_res


def get_mean(norm_value=255):
    return [114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value]


def get_label(res_tensor):
    # res_numpy = res_tensor.clone().detach().cpu().numpy()
    # labels = list()
    # for row in res_numpy:
    #     labels.append(np.argmax(row))
    res = torch.argmax(res_tensor, dim=1, keepdim=True)
    # labels = [label.item() for label in res]

    return res
    # return labels


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
        for j in range(range_of_frames[0], range_of_frames[1]):
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


def get_result(frame_pos, labels, opt):
    # print(labels)
    # print(frame_pos, flush=True)
    do_origin = False

    cut_priority = False
    gradual_priority = False
    train_type = opt.train_data_type
    final_res = list()
    if not do_origin:
        for i, label in enumerate(labels):
            # cut, gradual only
            if label > 0:
                if opt.loss_type != 'multiloss':
                    start = int(frame_pos[i])
                    # transition 데이터가 없을 때
                    if len(final_res) == 0:
                        final_res.append((start, start + opt.sample_duration, label))
                    else:
                        last_start = final_res[-1][0]
                        last_end = final_res[-1][1]
                        last_label = final_res[-1][2]
                        # 범위가 겹치지 않을때
                        if last_end < start:
                            final_res.append((start, start + opt.sample_duration, label))
                        # 범위가 겹칠 때
                        else:
                            # cut이 gradual보다 우선하는 정책
                            if cut_priority:
                                # 레이블이 같을 때
                                if last_label == label:
                                    final_res[-1] = (last_start, start + opt.sample_duration, label)
                                # 나중에 나온 레이블이 cut
                                elif last_label < label:
                                    final_res[-1] = (last_start, start, last_label)
                                    final_res.append((start, start + opt.sample_duration, label))
                                # 나중에 나온 레이블이 gradual
                                else:
                                    final_res.append((last_end, start + opt.sample_duration, label))
                            # gradual이 cut보다 우선하는 정책
                            elif gradual_priority:
                                # 레이블이 같을 때
                                if last_label == label:
                                    final_res[-1] = (last_start, start + opt.sample_duration, label)
                                # 나중에 나온 레이블이 gradual
                                elif last_label > label:
                                    final_res[-1] = (last_start, start, last_label)
                                    final_res.append((start, start + opt.sample_duration, label))
                                # 나중에 나온 레이블이 cut
                                else:
                                    final_res.append((last_end, start + opt.sample_duration, label))
                            # 나중에 오는 transition이 우선하는 정책
                            else:
                                if last_label == label:
                                    final_res[-1] = (last_start, start + opt.sample_duration, label)
                                else:
                                    final_res[-1] = (last_start, start, last_label)
                                    final_res.append((start, start + opt.sample_duration, label))
                else:
                    start, end = int(frame_pos[i][0]), int(frame_pos[i][1])
                    if start < end:
                        if len(final_res) == 0:
                            final_res.append((start, end, label))
                        else:
                            last_start = final_res[-1][0]
                            last_end = final_res[-1][1]
                            last_label = final_res[-1][2]
                            if label == 2 or (label == 1 and train_type == 'cut'):
                                final_res.append((start, end, label))
                                # # 안 겹칠 때
                                # if last_end < start or end < last_start:
                                #     if last_end < start:
                                #         final_res.append((start, end, label))
                                #     else:
                                #         final_res.insert(-1, (start, end, label))
                                # # 겹칠 때
                                # else:
                                #     # last_label=2, label=2
                                #     if last_label == label:
                                #         if last_start <= start <= last_end < end:
                                #             final_res[-1] = (last_start, end, label)
                                #         elif start <= last_start <= end < last_end:
                                #             final_res[-1] = (start, last_end, label)
                                #         elif start <= last_start < last_end <= end:
                                #             final_res[-1] = (start, end, label)
                                #         else:
                                #             final_res[-1] = (last_start, last_end, label)
                            else:
                                if last_label == label:
                                    # last_label=1, label=1
                                    if (start - last_end) <= 8:
                                        final_res[-1] = (last_start, end, label)
                                    else:
                                        final_res.append((start, end, label))
                                else:
                                    # 안 겹칠 경우
                                    if last_end < start or end < last_start:
                                        if last_end < start:
                                            final_res.append((start, end, label))
                                        else:
                                            final_res.insert(-1, (start, end, label))
                                    # 겹칠 경우
                                    else:
                                        if last_start <= start <= last_end < end:
                                            final_res[-1] = (last_start, start - 1, last_label)
                                            final_res.append((start, end, label))
                                        elif start <= last_start <= end < last_end:
                                            final_res[-1] = (start, end, label)
                                            final_res.append((end + 1, last_end, last_label))
                                        elif start <= last_start < last_end <= end:
                                            final_res[-1] = (start, end, label)
                                        else:
                                            final_res[-1] = (last_start, last_end, last_label)
    else:
        i = 0
        while i < len(labels):
            if labels[i] > 0:
                label = labels[i]
                begin = i
                i += 1
                while i < len(labels) and labels[i] == labels[i - 1]:
                    i += 1
                end = i - 1
                begin_frame = int(begin * opt.sample_duration / 2 + 1)
                end_frame = int(end * opt.sample_duration / 2 + 16 + 1)
                final_res.append((begin_frame, end_frame, label))
            else:
                i += 1
    return final_res


def get_frames_labels(prediction_list, opt):
    if opt.loss_type == 'multiloss':
        all_result = torch.zeros(prediction_list.size(0), 4).to(opt.device)
        start_count = 0
        end_count = 0
        if prediction_list.size(0) != 0:
            for cls in range(1, opt.n_classes):
                cls_idx = prediction_list[:, -1] == cls
                nms_list = prediction_list[cls_idx]
                ids, count = nms(nms_list[:, :2], nms_list[:, -2], overlap=0.5)
                end_count += count
                all_result[start_count:end_count] = nms_list[ids[:count]]
                start_count += count

        print(all_result[:end_count])
        _, idx_sort = all_result[:end_count, 0].sort(0)
        new_result = all_result[idx_sort]
        frame_pos, labels = new_result[:, :2], new_result[:, -1]
    else:
        frame_pos, labels = prediction_list[:, 0], prediction_list[:, 1]

    return frame_pos, labels


def test(video_path, test_data_loader, model, opt):
    # frame_pos = list()
    # labels = list()
    prediction_list = torch.Tensor().to(opt.device)

    batch_time = time.time()
    total_iter = len(test_data_loader)
    for i, (clip, boundary) in enumerate(test_data_loader):
        # for check size
        # print(sys.getsizeof(inputs))

        # clip = Variable(clip)
        # if opt.cuda:
        #     # clip = clip.to(device)
        #     clip = clip.cuda(device, non_blocking=True)
        with torch.no_grad():
            clip = clip.to(opt.device, non_blocking=True)
            boundary = boundary.to(opt.device, non_blocking=True)
        # # if use teacher student network, only get result of student network output
        # if opt.loss_type == 'KDloss':
        #     results = results[1]

        # 19.8.8. add
        # multiloss 일 때, 처리하는 부분 추가
        if opt.loss_type == 'multiloss':
            # frame_info, label = model(clip, boundary)
            prediction = model(clip, boundary)

            # frame_pos += [[start.item(), end.item()] for start, end in frame_info]
            # labels += [cls.item() for cls in label]
            # frame_pos += [frame_info.view(-1, 2)]
            # labels += [label.view(-1, 1)]
            prediction_list = torch.cat((prediction_list, prediction), 0)
        else:
            results = model(clip)
            # frame_pos += [frame.item() for frame in boundary]
            # labels += get_label(results)
            prediction = torch.cat((boundary.view(-1, 1), get_label(results).view(-1, 1)), 1)
            prediction_list = torch.cat((prediction_list, prediction), 0)

        if (i+1) % 10 == 0 or i+1 == total_iter:
            end_time = time.time() - batch_time
            print("iter {}/{} : {}".format(i + 1, total_iter, end_time), flush=True)
            batch_time = time.time()

    # if opt.loss_type == 'multiloss':
    #     all_result = torch.zeros(prediction_list.size(0), 4).to(opt.device)
    #     start_count = 0
    #     end_count = 0
    #     if prediction_list.size(0) != 0:
    #         for cls in range(1, opt.n_classes):
    #             cls_idx = prediction_list[:, -1] == cls
    #             nms_list = prediction_list[cls_idx]
    #             ids, count = nms(nms_list[:, :2], nms_list[:, -2], overlap=0.33)
    #             end_count += count
    #             all_result[start_count:end_count] = nms_list[ids[:count]]
    #             start_count += count
    #
    #     print(all_result[:end_count])
    #     _, idx_sort = all_result[:end_count, 0].sort(0)
    #     new_result = all_result[idx_sort]
    #     frame_pos, labels = new_result[:, :2], new_result[:, -1]

    # return [frame_pos, labels]
    return prediction_list


def load_checkpoint(model, opt_model):
    path = 'results/model_final.pth'

    print("load model... : ", opt_model)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])


def test_misaeng(opt, model):
    # opt = parse_opts()

    # if opt.loss_type == 'KDloss':
    #     teacher_model_path = 'models/Alexnet-final.pth'
    #     model = teacher_student_net(opt, teacher_model_path, 'test', device)
    # else:
    #     model = build_model(opt, 'test', device)
    load_checkpoint(model, opt.model)
    # model.eval()

    spatial_transform = get_test_spatial_transform(opt)
    temporal_transform = None
    target_transform = None

    root_dir = 'E:/video/misaeng'
    # print(root_dir, flush=True)
    # print(opt.test_list_path, flush=True)
    misaeng_list_path = 'E:/video/misaeng/misaeng_filename_list.txt'
    with open(misaeng_list_path, 'r') as f:
        video_name_list = [line.strip('\n') for line in f.readlines()]

    # data_name_list = ['frame_pos', 'labels']
    data_name_list = ['predictions']
    pickle_utils = PickleUtils(opt, video_name_list, data_name_list)

    res = {}
    # print('\n====> Testing Start', flush=True)
    epoch_time = time.time()
    for idx, video_name in enumerate(video_name_list[:1]):
        video_time = time.time()

        print("Make dataset {}...".format(video_name), flush=True)
        test_data = test_DataSet(root_dir, video_name,
                                 spatial_transform=spatial_transform,
                                 temporal_transform=temporal_transform,
                                 target_transform=target_transform,
                                 sample_duration=opt.sample_duration,
                                 input_type=opt.input_type, candidate=opt.candidate)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                       num_workers=opt.n_threads, pin_memory=True)

        total_frame, fps = test_data.get_video_attr()

        # 이미 처리한 결과가 있다면 pickle 로드
        print("Process {}".format(idx + 1), flush=True)
        if pickle_utils.check_pickle_data(video_name):
            video_path = os.path.join(root_dir, video_name)
            # frame_pos, labels = test(video_path, test_data_loader, model, device, opt)
            predictions = [test(video_path, test_data_loader, model, opt)]
            pickle_utils.save_pickle(video_name, predictions)
        else:
            # frame_pos, labels = load_pickle(frame_pos_path, labels_path)
            predictions = pickle_utils.load_pickle(video_name)

        # final_res = get_result(frame_pos, labels, opt)
        frame_pos, labels = get_frames_labels(predictions[0], opt)
        final_res = get_result(frame_pos, labels, opt)
        # print(final_res)

        boundary_index_final = []
        _res = {'cut': [], 'gradual': []}
        for begin, end, label in final_res:
            if label == 2:
                _res['cut'].append((begin, end))
            else:
                _res['gradual'].append((begin, end))
            boundary_index_final.append([begin, end, label])
        res[video_name] = _res

        boundary_index_final.insert(0, 1)
        boundary_index_final.append(total_frame)

        srt_index = 0
        do_srt = True
        if do_srt:
            with open(os.path.join(root_dir, video_name) + '.final.srt', 'w', encoding='utf-8') as f:
                for bound_ind in range(len(boundary_index_final) - 1):
                    if type(boundary_index_final[bound_ind]) == list:
                        transition = 'cut' if boundary_index_final[bound_ind][2] == 2 else 'gradual'
                        startframe = boundary_index_final[bound_ind][0]
                        endframe = boundary_index_final[bound_ind][1]
                        starttime = float(startframe / fps)
                        endtime = float(endframe / fps)
                        f.write(str(srt_index) + '\n')
                        f.write(str(starttime) + ' --> ' + str(endtime) + '\n')
                        f.write(transition + ': frame [' + str(startframe) + ',' + str(endframe) + ']\n')
                        f.write('\n')

                        if endframe == total_frame:
                            break

                    startframe = boundary_index_final[bound_ind][1] + 1 if bound_ind != 0 else 1
                    endframe = boundary_index_final[bound_ind + 1][0] - 1 if bound_ind != (
                                len(boundary_index_final) - 2) else boundary_index_final[bound_ind + 1]
                    starttime = float(startframe / fps)
                    endtime = float(endframe / fps)
                    f.write(str(srt_index) + '\n')
                    f.write(str(starttime) + ' --> ' + str(endtime) + '\n')
                    f.write('shot# ' + str(bound_ind) + ' & frame [' + str(startframe) + ',' + str(endframe) + ']\n')
                    f.write('\n')

                    srt_index += 1

        end_time = time.time() - video_time
        end_time = datetime.timedelta(seconds=end_time)
        print("{} Processing time : {}".format(video_name, end_time))

    total_time = time.time() - epoch_time
    total_time = datetime.timedelta(seconds=total_time)
    print("[INFO] finish test!!, {}".format(total_time), flush=True)

    out_path = os.path.join(root_dir, 'miseang_results.json')
    if not os.path.exists(out_path):
        json.dump(res, open(out_path, 'w'))


def test_dataset(opt, model):
    # opt = parse_opts()

    # if opt.loss_type == 'KDloss':
    #     teacher_model_path = 'models/Alexnet-final.pth'
    #     model = teacher_student_net(opt, teacher_model_path, 'test', device)
    # else:
    #     model = build_model(opt, 'test', device)
    load_checkpoint(model, opt.model)
    # model.eval()

    spatial_transform = get_test_spatial_transform(opt)
    temporal_transform = None
    target_transform = None
    # list_root_path : train path, only_gradual path
    # `19.3.7 : add only_gradual path
    root_dir = os.path.join(opt.root_dir, opt.test_subdir)
    # print(root_dir, flush=True)
    # print(opt.test_list_path, flush=True)
    with open(opt.test_list_path, 'r') as f:
        video_name_list = [line.strip('\n') for line in f.readlines()]

    # data_name_list = ['frame_pos', 'labels']
    data_name_list = ['predictions']
    pickle_utils = PickleUtils(opt, video_name_list, data_name_list)

    res = {}
    # print('\n====> Testing Start', flush=True)
    epoch_time = time.time()
    for idx, video_name in enumerate(video_name_list):
        video_time = time.time()
        print("Process {} : {}".format(idx + 1, video_name), flush=True)
        test_data = test_DataSet(root_dir, video_name,
                                 spatial_transform=spatial_transform,
                                 temporal_transform=temporal_transform,
                                 target_transform=target_transform,
                                 sample_duration=opt.sample_duration,
                                 input_type=opt.input_type, candidate=opt.candidate)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                       num_workers=opt.n_threads, pin_memory=True)
        if pickle_utils.check_pickle_data(video_name):
            video_path = os.path.join(root_dir, video_name)
            # frame_pos, labels = test(video_path, test_data_loader, model, device, opt)
            predictions = [test(video_path, test_data_loader, model, opt)]
            pickle_utils.save_pickle(video_name, predictions)
        else:
            # frame_pos, labels = load_pickle(frame_pos_path, labels_path)
            predictions = pickle_utils.load_pickle(video_name)

        # final_res = get_result(frame_pos, labels, opt)
        frame_pos, labels = get_frames_labels(predictions[0], opt)
        final_res = get_result(frame_pos, labels, opt)
        # print(final_res)

        if opt.train_data_type == 'normal':
            _res = {'cut': list(), 'gradual': list()}
        else:
            _res = {opt.train_data_type: list()}

        for begin, end, label in final_res:
            if label == 2:
                _res['cut'].append((begin, end))
            else:
                if opt.train_data_type == 'cut':
                    _res['cut'].append((begin, end))
                else:
                    _res['gradual'].append((begin, end))
        print(_res)
        res[video_name] = _res
        # print(videoname," : ", _res)

        # json.dump(res, open("test.json", 'w'))

        end_time = time.time() - video_time
        print("{} Processing time : {:.3f}s".format(video_name, end_time))

    total_time = time.time() - epoch_time
    total_time = datetime.timedelta(seconds=total_time)
    print("[INFO] finish test!!, {}".format(total_time), flush=True)

    return res


# 19.3.8 revision
# add parameter : "device"
def train(cur_iter, iter_per_epoch, epoch, data_loader, model, criterion, optimizer, scheduler, opt):
    # 19.3.14. add
    # print("device : ", torch.cuda.get_device_name(0), flush=True)
    # torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    # 19.5.10. revision
    for i in range(opt.gpu_num):
        print("device {} : {}".format(i, torch.cuda.get_device_name(i)), flush=True)

    # batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    i = cur_iter
    total_acc, epoch_acc, avg_acc = dict(), dict(), dict()
    if opt.loss_type == 'multiloss':
        keys = ['loc', 'conf']
    else:
        keys = ['conf']

    for key in keys:
        total_acc[key] = [0.0] * epoch
        epoch_acc[key] = 0.0
        avg_acc[key] = 0.0

    # for debug
    # print(opt.cuda) : True

    total_iter = epoch * iter_per_epoch
    save_timing = int(iter_per_epoch / 5)
    if opt.use_save_timing:
        if save_timing < 2000:
            save_timing = 2000
        elif save_timing > 5000:
            save_timing = 5000
    epoch_time = time.time()

    writer = SummaryWriter('runs/')
    c = Configure(sample_duration=opt.sample_duration, data_type=opt.train_data_type, policy=opt.layer_policy)
    default = c.get_default_bar()
    print('\n====> Training Start', flush=True)
    while i < total_iter:
        start_time = time.time()
        for _, (inputs, targets) in enumerate(data_loader):
            # for check size
            # print(sys.getsizeof(inputs))

            # 19.3.7 add
            # if opt.cuda:
            #     targets = targets.cuda(async=True)
            #     inputs = inputs.cuda(async=True)

            # 19.3.8. revision
            # 19.7.16
            # .cuda > .to
            # with torch.no_grad() for target
            # remove 'if opt.cuda:'
            inputs = inputs.to(opt.device, non_blocking=True)
            with torch.no_grad():
                targets = targets.to(opt.device, non_blocking=True)

            # 19.7.16. no use
            # targets = Variable(targets)
            # inputs = Variable(inputs)

            outputs = model(inputs)

            # if opt.loss_type in ['normal', 'KDloss']:
            #     targets = targets[:, -1].to(torch.long)

            loss_loc, loss_conf = criterion(outputs, targets)
            alpha = 0.5
            loss = loss_loc * alpha + loss_conf * (1. - alpha)

            if opt.loss_type in ['KDloss']:
                outputs = outputs[1]

            acc = calculate_accuracy(outputs, targets, opt.sample_duration, default, opt.device)
            for key in keys:
                avg_acc[key] += acc[key] / 10
                epoch_acc[key] += acc[key] / iter_per_epoch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            i += 1

            if i % 10 == 0:
                batch_time = time.time() - start_time
                lr = optimizer.param_groups[0]['lr']
                print('Iter:{} Loss(per 10 batch):{} lr:{} batch_time:{:.3f}s'.format(
                    i, loss.item(), lr, batch_time), flush=True)

                writer.add_scalar('loss/loss_loc', loss_loc.item(), i)
                writer.add_scalar('loss/loss_conf', loss_conf.item(), i)
                writer.add_scalar('loss/loss', loss.item(), i)
                writer.add_scalar('learning_rate', lr, i)
                writer.add_scalar('acc_loss/loss_loc', loss_loc.item(), i)
                writer.add_scalar('acc_loss/loss_conf', loss_conf.item(), i)
                writer.add_scalar('acc_loss/loss', loss.item(), i)

                for key in keys:
                    print('avg_acc:{:.5f}({}) epoch_acc:{:.9f}({})'.format(
                        avg_acc[key], key, epoch_acc[key], key), end=' ', flush=True)
                    writer.add_scalar('accuracy/acc_{}'.format(key), avg_acc[key], i)
                    writer.add_scalar('acc_loss/acc_{}'.format(key), avg_acc[key], i)
                    avg_acc[key] = 0.0
                print(flush=True)
                start_time = time.time()

            if i % save_timing == 0:
                save_file_path = os.path.join(opt.result_dir, 'model_iter{}.pth'.format(i))
                print("save to {}".format(save_file_path))
                states = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_file_path)
            if i % iter_per_epoch == 0 and i != 0:
                print("epoch {} accuracy : ".format(i / iter_per_epoch), end='', flush=True)
                for key in keys:
                    print('{}({}) '.format(epoch_acc[key], key), end='', flush=True)
                    total_acc[key][int(i / iter_per_epoch)-1] = float(epoch_acc[key])
                    epoch_acc[key] = 0.0
                print(flush=True)
            if i >= total_iter:
                break

    total_time = time.time() - epoch_time
    total_time = datetime.timedelta(seconds=total_time)
    print("Training Time : {}".format(total_time), flush=True)
    total_acc['Training_Time'] = str(total_time)

    save_file_path = os.path.join(opt.result_dir, 'model_final.pth'.format(opt.checkpoint_path))
    print("save to {}".format(save_file_path), flush=True)
    states = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path)

    json.dump(total_acc, open(os.path.join(opt.result_dir, 'epoch_accuracy_and_total_time.json'), 'w'))
    writer.close()


def get_lastest_model(opt):
    if opt.resume_path != '':
        return 0
    if os.path.exists(os.path.join(opt.result_dir, 'model_final.pth')):
        opt.resume_path = os.path.join(opt.result_dir, 'model_final.pth')
        return opt.epoch * opt.iter_per_epoch

    iter_num = -1
    for filename in os.listdir(opt.result_dir):
        if filename[-3:] == 'pth':
            _iter_num = int(filename[len('model_iter'):-4])
            iter_num = max(iter_num, _iter_num)
    if iter_num > 0:
        opt.resume_path = os.path.join(opt.result_dir, 'model_iter{}.pth'.format(iter_num))
    return iter_num


def train_dataset(opt, model):
    # opt = parse_opts()

    # opt.scales = [opt.initial_scale]
    # for i in range(1, opt.n_scales):
    #     opt.scales.append(opt.scales[-1] * opt.scale_step)
    #
    # opt.mean = get_mean(opt.norm_value)
    # print(opt)
    #
    # torch.manual_seed(opt.manual_seed)

    # # 19.3.8. add
    # print("cuda is available : ", torch.cuda.is_available(), flush=True)

    # # 19.5.16 add
    # # set default tensor type
    # if torch.cuda.is_available() and opt.cuda:
    #     torch.backends.benchmark = True
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # else:
    #     torch.set_default_tensor_type('torch.FloatTensor')

    # 19.5.15. add
    print('[INFO] training {}'.format(opt.model), flush=True)

    # # 19.5.7. add
    # # teacher student option add
    # if opt.loss_type == 'KDloss':
    #     teacher_model_path = 'models/Alexnet-final.pth'
    #     model = teacher_student_net(opt, teacher_model_path, 'train', device)
    # else:
    #     model = build_model(opt, 'train', device)
    # print(model)

    # # `19.5.14.
    # # use multi_gpu for training and testing
    # model = nn.DataParallel(model, device_ids=range(opt.gpu_num))
    #
    # # `19.5.16. : from model_cls.py to main_baseline.py
    # # `19.3.8
    # # model = model.cuda(device)
    # if opt.cuda:
    #     torch.backends.benchmark = True
    #     model = model.to(device)
    #     # model.cuda()

    parameters = model.parameters()

    if opt.optimizer == 'sgd':
        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.momentum

        optimizer = optim.SGD(parameters, lr=opt.learning_rate,
                              momentum=opt.momentum, dampening=dampening,
                              weight_decay=opt.weight_decay, nesterov=opt.nesterov)
    elif opt.optimizer == 'adam':
        optimizer = optim.Adam(parameters, lr=opt.learning_rate,
                               betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=opt.weight_decay)

    cur_iter = 0
    if opt.auto_resume and opt.resume_path == '':
        cur_iter = get_lastest_model(opt)
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path), flush=True)
        checkpoint = torch.load(opt.resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if opt.start_iter != 0:
        cur_iter = opt.start_iter

    # 19.5.7. add
    # teacher student option add
    # 19.8.7. add
    # Multiloss add
    assert opt.loss_type in ['KDloss', 'multiloss', 'normal']
    if opt.loss_type == 'KDloss':
        criterion = KDloss(loss_type=opt.KD_type)
    elif opt.loss_type == 'multiloss':
        criterion = MultiLoss(device=opt.device, extra_layers=opt.use_extra_layer,
                              sample_duration=opt.sample_duration, num_classes=opt.n_classes,
                              data_type=opt.train_data_type, policy=opt.layer_policy, neg_ratio=3, neg_threshold=opt.neg_threshold)
    else:
        criterion = nn.CrossEntropyLoss()

    # 19.3.8. revision
    # add 'if opt.no_cuda:'
    # 19.7.16. revision
    # remove 'if opt.cuda:
    # if opt.cuda:
    criterion = criterion.to(opt.device)

    if opt.train:
        spatial_transform = get_train_spatial_transform(opt)
        temporal_transform = None
        target_transform = None
        # list_root_path : train path, only_gradual path
        # `19.3.7 : add only_gradual path
        # `19.8.12 : deepSBD_new.txt / detector.txt를 사용 시 필요없음
        # list_root_path = list()
        # list_root_path.append(os.path.join(opt.root_dir, opt.train_subdir))
        # list_root_path.append(os.path.join(opt.root_dir, opt.only_gradual_subdir))
        # print(list_root_path, flush=True)
        print("[INFO] reading : ", opt.video_list_path, flush=True)
        training_data = train_DataSet(opt.root_dir, opt.video_list_path, opt,
                                      spatial_transform=spatial_transform,
                                      temporal_transform=temporal_transform,
                                      target_transform=target_transform,
                                      sample_duration=opt.sample_duration)

        weights = torch.DoubleTensor(training_data.weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=opt.batch_size,
                                                           num_workers=opt.n_threads, sampler=sampler, pin_memory=True)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.iter_per_epoch)

        # 19.3.8. add
        # train(cur_iter,opt.total_iter,training_data_loader, model, criterion, optimizer,scheduler,opt)
        train(cur_iter, opt.iter_per_epoch, opt.epoch, training_data_loader, model, criterion, optimizer, scheduler, opt)


def build_final_model(opt):
    assert opt.phase in ['train', 'test']
    # 19.6.4 remove
    # is not used
    # 19.6.26 revive
    # 19.6.28 remove
    # assert opt.model_type in ['old', 'new']

    # 19.5.7. add
    # teacher student option add
    if opt.loss_type == 'KDloss':
        # 19.6.26.
        # remove teacher_model_path
        # because use opt.teacher_model_path

        # teacher_model_path = 'models/Alexnet-final.pth'
        model = TeacherStudentModule(opt)
    else:
        model = build_model(opt, opt.model, opt.phase)

    # 19.6.4.
    # remove below lines > opt.model_type = 'new' is not trainable
    # 19.6.26.
    # benchmark를 new type model일 때 전체적으로 적용 > 시도해볼 필요 있음
    # 19.6.28. remove
    # if opt.cuda and opt.model_type == 'new':
    #     torch.backends.benchmark = True
    #     # use multi_gpu for training and testing
    #     model = nn.DataParallel(model, device_ids=range(opt.gpu_num))
    #     # model.cuda()
    #     # model = model.cuda(device)
    #     model.to(device)

    print(model)

    return model


def main():
    # 19.5.17 add
    print(torch.__version__)

    # 19.5.17 for ubuntu
    # torch.multiprocessing.set_start_method('spawn')

    # # 19.5.7 add
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt = parse_opts()

    # 19.10.17 add
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # `19.10.8 move from train_dataset()
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)

    opt.mean = get_mean(opt.norm_value)
    print(opt)

    torch.manual_seed(opt.manual_seed)

    # 19.3.8. add
    print("cuda is available : ", torch.cuda.is_available(), flush=True)

    # `19.7.7. add
    # parallel>benchmark>cuda 순서 적용한 부분 추가
    # `19.7.10. 주석처리
    # if torch.cuda.is_available() and opt.cuda:
    #     torch.backends.benchmark = True
    #     model=model.to(device)

    # `19.10.8. add
    # check dataset and set opt.root_dir
    assert opt.dataset in ['ClipShots', 'RAI']
    opt.root_dir = os.path.join('data/', opt.dataset, 'videos/')
    opt.test_list_path = os.path.join('data/', opt.dataset, 'video_lists/test.txt')
    opt.gt_dir = os.path.join('data/', opt.dataset, 'annotations/test.json')

    # iter_per_epoch을 opt.is_full_data와 opt.batch_size에 맞게 자동으로 조정
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

    # 19.5.16 add
    # set default tensor type
    # 19.6.26.
    # model generate 시 적용되게 수정
    # 19.7.1.
    # model 완성전에 benchmark 수행하게 설정
    # 19.7.10. 주석처리
    # if torch.cuda.is_available() and opt.cuda:
    #     torch.backends.benchmark = True

    # ubuntu에서 주석처리해서 > 에러해결
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # else:
    #     torch.set_default_tensor_type('torch.FloatTensor')

    # assert opt.phase in ['train', 'test']
    # # 19.5.7. add
    # # teacher student option add
    # if opt.loss_type == 'KDloss':
    #     teacher_model_path = 'models/Alexnet-final.pth'
    #     model = teacher_student_net(opt, teacher_model_path, device)
    # else:
    #     model = build_model(opt, opt.phase, device)
    # print(model)

    assert opt.input_type in ['RGB', 'HSV']
    out_path = os.path.join(opt.result_dir, 'results.json')
    if opt.phase == 'full':
        phase_list = ['train', 'test']
    else:
        phase_list = [opt.phase]

    for phase in phase_list:
        opt.phase = phase
        model = build_final_model(opt)
        if phase == 'train':
            train_dataset(opt, model)
        else:
            if opt.misaeng:
                test_misaeng(opt, model)
            else:
                if not os.path.exists(out_path):
                    res = test_dataset(opt, model)
                    json.dump(res, open(out_path, 'w'))
                eval_res.eval(opt.result_dir, opt.gt_dir, opt.train_data_type)


if __name__ == '__main__':
    # # 19.5.7 add
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()
    # train_dataset(device)
    # test_dataset(device)
    # test_misaeng(device)
