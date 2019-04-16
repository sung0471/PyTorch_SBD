import torch
import torch.utils.data as data
from PIL import Image
import os
import math
# import functools
# import json
# import copy
# import numpy as np
import random
import cv2


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def video_loader(video_path, frame_pos, sample_duration, img=False):
    video = []
    try:
        if not img:
            video_cap = cv2.VideoCapture(video_path)
            video_cap.set(1, frame_pos)
            for i in range(sample_duration - len(video)):
                status, frame = video_cap.read()
                if not status:
                    break
                else:
                    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
                    video.append(frame)
            video += [video[-1] for _ in range(sample_duration - len(video))]

            return video
    except Exception as e:
        pass


def get_default_video_loader():
    return video_loader


def make_dataset(video_path, sample_duration):
    video_list=[]
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_num < total_frame:
        info = {"video_path": video_path, "begin": frame_num, "sample_duration": sample_duration}
        video_list.append(info)
        frame_num += sample_duration/2

    return video_list


class DataSet(data.Dataset):
    def __init__(self, video_path,
                 spatial_transform=None, temporal_transform=None, target_transform=None,
                 sample_duration=16, get_loader=get_default_video_loader):
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.video_list = make_dataset(video_path, sample_duration)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (clip, target) where target is class_index of the target class.
        """

        video_path = self.video_list[index]['video_path']
        begin_indicate = self.video_list[index]['begin']
        sample_duration = self.video_list[index]['sample_duration']

        clip = self.loader(video_path, begin_indicate, sample_duration)
        # raw_clip=clip

        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip

    def __len__(self):
        return len(self.video_list)
