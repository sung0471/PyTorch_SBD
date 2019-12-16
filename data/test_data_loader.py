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
        assert False, 'Error msg : {}\n' \
                      'video : {}'.format(e, video)


def get_default_video_loader():
    return video_loader


def no_candidate_frame_pos(root_dir, video_name, total_frame, sample_duration):
    video_dir = os.path.join(root_dir, video_name)

    frame_index_list = list()

    i = 0
    while i < total_frame:
        frame_index_list.append(i)
        i += sample_duration / 2

    return frame_index_list


def check_video_frames_and_fps(video_root, video_name):
    # input video (cv2)
    video_path = os.path.join(video_root, video_name)
    cap = cv2.VideoCapture(video_path)
    cv2_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frame = cv2_total_frames
    while True:
        cap.set(1, total_frame - 1)
        status, frame = cap.read()
        if status:
            break
        total_frame -= 1

    return cv2_total_frames, total_frame, fps


def make_dataset(video_root, video_name, total_frame, sample_duration, candidate):
    video_path = os.path.join(video_root, video_name)
    video_list = list()
    if candidate:
        model = SqueezeNetFeature()
        frame_pos_list = candidate_extraction(video_root, video_name, total_frame, model, adjacent=True)
        frame_index_list = list()
        for frame_pos in frame_pos_list[1:-1]:
            interpolated_frame_pos = int(frame_pos - (sample_duration / 2 - 1))
            interpolated_frame_pos = 0 if interpolated_frame_pos < 0 else interpolated_frame_pos
            frame_index_list.append(interpolated_frame_pos)
    else:
        frame_index_list = no_candidate_frame_pos(video_root, video_name, total_frame, sample_duration)
        frame_index_list = frame_index_list[:]

    for frame_pos in frame_index_list:
        info = {"video_path": video_path, "frame_pos": frame_pos, "sample_duration": sample_duration}
        video_list.append(info)

    return video_list


class DataSet(data.Dataset):
    def __init__(self, video_root, video_name, gt_total_frame,
                 spatial_transform=None, temporal_transform=None, target_transform=None,
                 sample_duration=16, input_type='RGB', candidate=False,
                 get_loader=get_default_video_loader):
        cv2_total_frame, total_frame, fps = check_video_frames_and_fps(video_root, video_name)
        self.video_list = make_dataset(video_root, video_name, sample_duration, total_frame, candidate)
        print('Video length info : cv2({}), gt({}), real_data({}), lastF({})'.format(cv2_total_frame, gt_total_frame,
                                                                                     total_frame,
                                                                                     self.video_list[-1]['frame_pos']))
        self.total_frame = total_frame
        self.fps = fps
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.input_type = input_type
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

        clip = self.loader(video_path, frame_pos, sample_duration, self.input_type)
        # raw_clip=clip

        if self.spatial_transform is not None:
            # `19.10.8
            # multiscalecornercrop
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip, frame_pos

    def __len__(self):
        return len(self.video_list)

    def get_video_attr(self):
        return self.total_frame, self.fps
