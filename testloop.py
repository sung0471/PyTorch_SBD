import time
import datetime

epoch_time = time.time()

temporal_length=16
image_clip=[]
for i in range(temporal_length -len(image_clip)):
    print(temporal_length -len(image_clip))
    image_clip.append(i)

print(image_clip)

arr, arr2 = list(), list()
for i in range(10):
    arr.append(i)
    arr2.append(10-i)

print(arr)
print(arr[1:-2])

if type(arr) is list:
    print("arr : list")
else:
    print("arr : int")

print("arr[{}:] = {}\n".format(1, arr[1:]))
start_index = 0
for i, _ in enumerate(arr, 2):
    print("arr[{}::{}] = {}".format(start_index,i,arr[start_index::i]))

flag=False
for _ in range(10):
    print((1,3)[flag])
    flag=not flag

tup=(arr,arr2)
for i, arr in enumerate(tup):
    print(tup[i])

tup2=(1,2,3,4,5,6,7,8,9,10)
print(tup2[1::2])

import torch
a=torch.randn(4)
print("a = ", a)
b=a.mul(0.9)           # a에 0.9를 mul
print("b = .a.mul(0.9) = ", b)
print("b.add(1) = ", b.add(1))        # b에 1를 add
print("b.add(0.1,1) = ", b.add(0.1,1))    # 0.1*1을 b에 add
print("a.mul(0.9).add(0.1,1) = ", a.mul(0.9).add(0.1,1))

for i in range(4,6,2):
    print(i)

line='5014499309.mp4 1868 2'
print(line.split(' '))

import numpy as np
def f(array):
    return np.array(sum(array, []))


array = [[1, 2], [3, 4], [5, 6]]
print(f(array))

x = torch.randn(4, 4)
print(x, x.size())
y = x.view(4,-1,4)
print(y, y.size())
y_cat = torch.cat((y,y,y),1)
print(y_cat, y_cat.size())
z = y.view(4,4)
print(z, z.size())

imgg = torch.randn(2,3,5,6)
print(imgg)
imgg=imgg.permute(0,2,3,1).contiguous()
print(imgg)

fc=imgg.view(imgg.size(0),-1)
print(fc, fc.size())
fc=fc.view(fc.size(0),-1,6)
print(fc, fc.size())

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
cfg = (coco, voc)[num_classes==21]
print(cfg)

weight=torch.DoubleTensor([0.3, 0.3, 0.3, 0.5, 0.5])
print(torch.multinomial(weight, len(weight), replacement=True))

for i in range(5):
    print(i,flush=True)

boundary=[120, 130]
print([1]*len(boundary))
print(boundary+[1]*len(boundary))

import json
data={}
data["cut"]=[(1,2)]
temp=np.array([3,4])
new=[]
for _ in temp:
    new+=[int(_+1)]
data["gradual"]=[new]
json.dump(data,open("test.json",'w'))

arr_test = torch.Tensor([[1,2,3,4],[9,10,11,12],[5,6,7,8]])
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

pos=arr_test > 6
print(pos)
print(pos.sum(dim=1, keepdim=True))
print(pos.sum(dim=1, keepdim=False))

box1=torch.Tensor([[1,1,3,3],[2,2,4,4]])
box2=torch.Tensor()

import torch.nn as nn
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
print(input, target, output)

checklist=[1,2,3,4,5]
no_list={"json":123}

print(isinstance(checklist,list))
print(isinstance(no_list,list))
print(isinstance(no_list,dict))

import cv2
import os
import time
from opts import parse_opts

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

arr_test = torch.Tensor([[1,2,3,4],[9,10,11,12],[5,6,7,8]])
_, predict = arr_test.topk(1,1,True)
predict = predict.t()
print(predict, type(predict), predict.size())

total_acc = [0.0] * 5
for i in range(len(total_acc)):
    total_acc[i] = 95.0 + i*0.2
total_time = time.time() - epoch_time
total_time = datetime.timedelta(seconds=total_time)
print("Training Time : {}".format(total_time), flush=True)
total_acc.append(str(total_time))
json.dump(total_acc, open('epoch_accuracy_test.json', 'w'))

from lib.spatial_transforms import *
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

os.makedirs(os.path.join('test', 'test2', 'test3'))
