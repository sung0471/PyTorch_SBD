import time
import datetime
import torch
import numpy as np
import json
import torch.nn as nn
import cv2
import os
import time
from opts import parse_opts
from lib.spatial_transforms import get_test_spatial_transform, Image

epoch_time = time.time()


def python_test():
    print('Test : make 16 length image clip')
    temporal_length = 16
    image_clip = []
    for i in range(temporal_length - len(image_clip)):
        print('temporal_length - len(image_clip) : {}'.format(temporal_length - len(image_clip)))
        image_clip.append(i)
    print('image_clip : {}'.format(image_clip))

    print('Test : control arr')
    arr, arr2 = list(), list()
    for i in range(10):
        arr.append(i)
        arr2.append(10-i)
    print('arr : {}'.format(arr))
    print('arr[1:-2] : {}'.format(arr[1:-2]))

    if type(arr) is list:
        print("arr : list")
    else:
        print("arr : int")

    print("arr[{}:] = {}\n".format(1, arr[1:]))
    start_index = 0
    for i, _ in enumerate(arr, 2):
        print("arr[{}::{}] = {}".format(start_index, i, arr[start_index::i]))

    print('Test : bool type value test')
    flag = False
    for _ in range(10):
        print('(1, 3)[flag] : {}'.format((1, 3)[flag]))
        flag = not flag

    print('Test : Tuple test')
    tup = (arr, arr2)
    for i, arr in enumerate(tup):
        print('tup[i] : {}'.format(tup[i]))

    tup2 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    print('tup2[1::2] : {}'.format(tup2[1::2]))

    print('Test : torch.randn() test')
    a = torch.randn(4)
    print("torch.randn(4) = {}".format(a))
    b = a.mul(0.9)           # a에 0.9를 mul
    print("b = a.mul(0.9) = {}".format(b))
    print("b.add(1) = ", b.add(1))        # b에 1를 add
    print("b.add(0.1,1) = ", b.add(0.1,1))    # 0.1*1을 b에 add
    print("a.mul(0.9).add(0.1,1) = ", a.mul(0.9).add(0.1,1))

    print('Test : for i in range(4, 6, 2)')
    for i in range(4, 6, 2):
        print(i)

    print('Test : line.split(' ')')
    line = '5014499309.mp4 1868 2'
    print(line.split(' '))

    def f(arr):
        return np.array(sum(arr, []))

    print('Test : sum(arr, [])')
    array = [[1, 2], [3, 4], [5, 6]]
    print(f(array))

    boundary = [120, 130]
    print([1] * len(boundary))
    print(boundary + [1] * len(boundary))

    checklist=[1,2,3,4,5]
    no_list={"json":123}

    print(isinstance(checklist,list))
    print(isinstance(no_list,list))
    print(isinstance(no_list,dict))

    # os.makedirs(os.path.join('test', 'test2', 'test3'))

    test = [1,2,3]
    assign = test.copy()
    test[0]=5
    print(test, assign)


def pytorch_tensor_test():
    print('Test : torch.Tensor.size(), view(), cat()')
    x = torch.randn(4, 4)
    print('x : {}, size : {}'.format(x, x.size()))
    y = x.view(4, -1, 4)
    print('y = x.view(4, -1, 4) : {}, size : {}'.format(y, y.size()))
    y_cat = torch.cat((y, y, y), 1)
    print('cat((y, y, y), 1) : {}, size : {}'.format(y_cat, y_cat.size()))
    z = y.view(4, 4)
    print('y.view(4, 4) : {}, size : {}'.format(z, z.size()))

    print('Test : permute, contiguous, view')
    img_t = torch.randn(2, 3, 5, 6)
    print('img_t : {}'.format(img_t))
    img_t = img_t.permute(0, 2, 3, 1).contiguous()
    print('img_t.permute(0, 2, 3, 1).contiguous() : {}'.format(img_t))

    fc = img_t.view(img_t.size(0), -1)
    print('img_t.view(img_t.size(0), -1) : {}, size : {}'.format(fc, fc.size()))
    fc = fc.view(fc.size(0), -1, 6)
    print('img_t.view(fc.size(0), -1, 6) : {}, size : {}'.format(fc, fc.size()))

    weight = torch.DoubleTensor([0.3, 0.3, 0.3, 0.5, 0.5])
    print('torch.multinomial : ', torch.multinomial(weight, len(weight), replacement=True))

    print('Test : max, squeeze, unsqueeze')
    arr_test = torch.Tensor([[1, 2, 3, 4], [9, 10, 11, 12], [5, 6, 7, 8]])
    # 원본 : [3,4]
    # dim=0 끼리 비교
    # return : [1,4], [1,4] / 최댓값 배열, 최댓값 index
    dim0_value, dim0_index = arr_test.max(0, keepdim=True)
    print(dim0_value, dim0_index)
    # squeeze :
    print(dim0_value.squeeze_(0), dim0_index.squeeze_(0))
    # dim=1 끼리 비교
    # return : [3,1], [3,1] / 최댓값 배열, 최댓값 index
    dim1_value, dim1_index = arr_test.max(1, keepdim=True)
    print(dim1_value, dim1_index)
    print(dim1_value.squeeze_(1), dim1_index.squeeze_(1))

    print(arr_test, arr_test.size(), arr_test.dim())
    for i in range(3):
        arr_unsq = arr_test.unsqueeze(i)
        print(arr_unsq, arr_unsq.size(), arr_unsq.dim())

    print('Test : Tensor > int')
    pos = arr_test > 6
    print('pos = arr_test > 6 : ', pos)
    print('pos.sum(dim=1, keepdim=True) : ', pos.sum(dim=1, keepdim=True))
    print('pos.sum(dim=1, keepdim=False) : ', pos.sum(dim=1, keepdim=False))

    box1 = torch.Tensor([[1, 1, 3, 3], [2, 2, 4, 4]])
    box2 = torch.Tensor()

    print('Test : CrossEntropy Loss')
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    print('input : {}\n'
          'target : {}\n'
          'output : {}'.format(input, target, output))

    print('Test : .topk(), .t()')
    arr_test = torch.Tensor([[1, 2, 3, 4], [9, 10, 11, 12], [5, 6, 7, 8]])
    _, predict = arr_test.topk(1, 1, True)
    predict = predict.t()
    print(predict, type(predict), predict.size())

    print('Test : /=int * int, /=(int * int)')
    test = torch.Tensor([10.0, 10.0])
    test_ = torch.Tensor([10.0, 10.0])
    test /= 3 * 2
    test_ /= (3 * 2)
    print('test /= 3 * 2 : {}\n'
          'test_ /= (3 * 2) : {}'.format(test, test_))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor1 = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8]]).to(device)
    tensor2 = torch.Tensor([2, 2, 2, 2]).to(torch.long).to(device)
    pos = tensor2 > 0
    new = tensor1[tensor2]
    print(new)
    new_2 = tensor1[pos]
    print(new_2)

    print(tensor1, tensor2)
    print(tensor1.gather(1, tensor2.view(-1, 1)))

    tensor1[pos] = 0
    print(tensor1)


def json_test():
    voc = {
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'VOC',
    }

    coco = {
        'num_classes': 201,
        'lr_steps': (280000, 360000, 400000),
        'max_iter': 400000,
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [21, 45, 99, 153, 207, 261],
        'max_sizes': [45, 99, 153, 207, 261, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'COCO',
    }

    num_classes=21
    cfg = (coco, voc)[num_classes == 21]
    print(cfg)

    data = dict()
    data["cut"] = [(1, 2)]
    temp = np.array([3, 4])
    new = []
    for _ in temp:
        new += [int(_ + 1)]
    data["gradual"] = [new]
    json.dump(data, open("test.json", 'w'))

    total_acc = [0.0] * 5
    for i in range(len(total_acc)):
        total_acc[i] = 95.0 + i * 0.2
    total_time = time.time() - epoch_time
    total_time = datetime.timedelta(seconds=total_time)
    print("Training Time : {}".format(total_time), flush=True)
    total_acc.append(str(total_time))
    json.dump(total_acc, open('epoch_accuracy_test.json', 'w'))


def video_inout_test():
    opt = parse_opts()
    with open(opt.test_list_path, 'r') as f:
        video_name_list = [line.strip('\n') for line in f.readlines()]

    length_of_test = 0.0
    root_dir = os.path.join(opt.root_dir, opt.test_subdir)
    for video_name in video_name_list:
        video_dir = os.path.join(root_dir, video_name)
        cap = cv2.VideoCapture(video_dir)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length_of_test += total_frame

    print("test length frame : {}".format(length_of_test))

    misaeng_list_path = 'E:/video/misaeng/misaeng_filename_list.txt'
    with open(misaeng_list_path, 'r') as f:
        video_name_list = [line.strip('\n') for line in f.readlines()]

    length_of_test = 0.0
    root_dir = 'E:/video/misaeng'
    for video_name in video_name_list:
        video_dir = os.path.join(root_dir, video_name)
        cap = cv2.VideoCapture(video_dir)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length_of_test += total_frame

    print("misaeng length frame : {}".format(length_of_test))

    video_path = os.path.join(root_dir, video_name_list[0])
    print(video_path)
    video_cap = cv2.VideoCapture(video_path)
    video = list()
    spatial_transform = get_test_spatial_transform(opt)
    for i in range(8):
        status, frame = video_cap.read()
        if not status:
            break
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame = Image.fromarray(hsv, 'HSV')
            frame = spatial_transform(frame)
            video.append(frame)
    video = torch.stack(video, 0)
    print(torch.Tensor.size(video))


if __name__ == '__main__':
    # python_test()
    pytorch_tensor_test()
    # json_test()
    # video_inout_test()
