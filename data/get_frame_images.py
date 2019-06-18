import os
import json
import cv2
import time
import datetime
from PIL import Image

test_list_path = "ClipShots/video_lists/test.txt"
with open(test_list_path, 'r') as f:
    video_name_list = [line.strip('\n') for line in f.readlines()]

# file_name = "hUoDOxOxK1I.mp4"

gts = {}
gt_base_dir = "ClipShots/annotations/"
train_gt_dir = os.path.join(gt_base_dir, "train.json")
only_gradual_gt_dir = os.path.join(gt_base_dir, "only_gradual.json")
test_gt_dir = os.path.join(gt_base_dir, "test.json")
gts["train"] = json.load(open(train_gt_dir, 'r'))
gts["only_gradual"] = json.load(open(only_gradual_gt_dir, 'r'))
gts["test"] = json.load(open(test_gt_dir, 'r'))

total_start_time = time.time()
for i, file_name in enumerate(video_name_list):
    per_video_start_time = time.time()

    pos_info = gts["test"][file_name]["transitions"]

    video_base_dir = "ClipShots/videos/"
    video_path = os.path.join(video_base_dir, 'test', file_name)
    video_cap = cv2.VideoCapture(video_path)

    images_path = os.path.join('images', file_name)
    if not os.path.exists('images'):
        os.mkdir('images')
    if not os.path.exists(images_path):
        os.mkdir(images_path)
        os.mkdir(os.path.join(images_path, 'cut'))
        os.mkdir(os.path.join(images_path, 'gradual'))

    if file_name == 'hUoDOxOxK1I.mp4':
        pos_info += [[3113, 3133]]

    for frame_pos, frame_end in pos_info:
        if frame_end - frame_pos == 1:
            transition_type = 'cut'
            frame_pos -= 3
            frame_end += 3
        else:
            transition_type = 'gradual'
        video_cap.set(1, frame_pos)
        for i in range(frame_pos, frame_end+1):
            status, frame = video_cap.read()
            if not status:
                break
            else:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
                image_path = os.path.join(images_path, transition_type)
                frame.save(os.path.join(image_path, str(i) + '.png'))

    per_video_end_time = time.time() - per_video_start_time

    print('video #{} {} : {}'.format(i, file_name, per_video_end_time), flush=True)

total_time = time.time() - total_start_time
print('Total video : {}'.format(datetime.timedelta(seconds=total_time)), flush=True)
