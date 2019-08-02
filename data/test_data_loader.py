import torch
import torch.utils.data as data
from PIL import Image
import os
import math
# import functools
import json
# import copy
# import numpy as np
import random
import cv2
from lib.candidate_extracting import candidate_extraction
from models.squeezenet import SqueezeNetFeature


def no_candidate_frame_pos(root_dir, video_name, sample_duration):
    video_dir = os.path.join(root_dir, video_name)

    frame_index_list = list()
    # input video (cv2)
    cap = cv2.VideoCapture(video_dir)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    i = 0
    while i < total_frame:
        frame_index_list.append(i)
        i += sample_duration / 2

    return frame_index_list, total_frame, fps


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def video_loader(video_path, frame_pos, sample_duration, input_type='RGB', img=False):
    video = list()
    try:
        if not img:
            video_cap = cv2.VideoCapture(video_path)
            video_cap.set(1, frame_pos)
            for i in range(sample_duration - len(video)):
                status, frame = video_cap.read()
                if not status:
                    break
                else:
                    # 19.7.31. add HSV version
                    # 19.8.2. add 'if'
                    if input_type == 'RGB':
                        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
                    elif input_type == 'HSV':
                        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), 'HSV')
                    else:
                        assert "input_type must be 'RGB' or 'HSV'"
                    video.append(frame)
            video += [video[-1] for _ in range(sample_duration - len(video))]

            return video
    except Exception as e:
        pass


def get_default_video_loader():
    return video_loader


def make_clip_list(video_root, video_name, sample_duration, candidate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    video_list = list()
    if candidate:
        model = SqueezeNetFeature()
        frame_index_list, total_frame, fps = candidate_extraction(video_root, video_name, model, adjacent=True)
        frame_index_list = frame_index_list[1:-1]
    else:
        frame_index_list, total_frame, fps = no_candidate_frame_pos(video_root, video_name, sample_duration)
        frame_index_list = frame_index_list[:]

    video_path = os.path.join(video_root, video_name)
    for frame_pos in frame_index_list:
        info = {"video_path": video_path, "frame_pos": frame_pos, "sample_duration": sample_duration}
        video_list.append(info)

    json_path = os.path.join(video_root, os.path.splitext(video_name)[0] + ".json")
    json.dump(video_list, open(json_path, 'w'), indent=2)

    return video_list, total_frame, fps


def make_dataset(video_root, video_name_list, sample_duration, candidate):
    if not isinstance(video_name_list, list):
        video_list, total_frame, fps = make_clip_list(video_root, video_name_list, sample_duration, candidate)
        return video_list, total_frame, fps
    else:
        video_all_list = list()
        video_length_list = list()
        video_fps_list = list()
        for video_name in video_name_list:
            video_list, total_frame, fps = make_clip_list(video_root, video_name, sample_duration, candidate)
            video_all_list += video_list
            video_length_list += [total_frame]
            video_fps_list += [fps]

        return video_all_list, video_length_list, video_fps_list


class DataSet(data.Dataset):
    def __init__(self, video_root, video_name,
                 spatial_transform=None, temporal_transform=None, target_transform=None,
                 sample_duration=16, input_type = 'RGB', candidate=False,
                 get_loader=get_default_video_loader):
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.video_list, self.total_frame, self.fps = make_dataset(
            video_root, video_name, sample_duration, candidate)
        self.input_type = input_type
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.candidate = candidate
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (clip, target) where target is class_index of the target class.
        """

        video_path = self.video_list[index]['video_path']
        frame_pos = self.video_list[index]['frame_pos']
        sample_duration = self.video_list[index]['sample_duration']

        if self.candidate:
            start_frame = int(frame_pos - (sample_duration / 2 - 1))
            start_frame = 0 if start_frame < 0 else start_frame
        else:
            start_frame = frame_pos

        clip = self.loader(video_path, start_frame, sample_duration, self.input_type)
        # raw_clip=clip

        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip, start_frame

    def __len__(self):
        return len(self.video_list)

    def get_video_attr(self):
        return self.total_frame, self.fps
