import os
import json
import cv2
import time
import datetime
from PIL import Image


def print_frames(video_path, frame_pos, frame_end, image_path):
    video_cap = cv2.VideoCapture(video_path)
    video_cap.set(1, frame_pos)
    for j in range(frame_pos, frame_end + 1):
        status, frame = video_cap.read()
        if not status:
            break
        else:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
            frame.save(os.path.join(image_path, str(j) + '.png'))


def get_gt_dirs(path_type, check_dataset, dataloader_name=None):
    video_name_list = dict()
    gts = dict()
    dataset_root_dir = 'ClipShots/'
    dataloader_list = 'data_list/'
    video_root = os.path.join(dataset_root_dir, 'videos/')
    list_path_dir = os.path.join(dataset_root_dir, 'video_lists/')
    gt_base_dir = os.path.join(dataset_root_dir, 'annotations/')
    if dataloader_name is None:
        for i, p_type in enumerate(path_type):
            if check_dataset[i]:
                if path_type != 'result':
                    file_list_path = os.path.join(list_path_dir, p_type + '.txt')
                    with open(file_list_path, 'r') as f:
                        video_name_list[p_type] = [line.strip('\n') for line in f.readlines()]

                    gt_path = os.path.join(gt_base_dir, p_type + '.json')
                    gts[p_type] = json.load(open(gt_path, 'r'))
                else:
                    result_dir = os.path.join('../results/', 'results.json')
                    gts["result"] = json.load(open(result_dir, 'r'))
        return video_name_list, gts
    else:
        for i, dataloader_file_name in enumerate(dataloader_name):
            if check_dataset[i]:
                dataloader_path = os.path.join(dataloader_list, dataloader_file_name + '.txt')
                with open(dataloader_path, 'r') as f:
                    for line in f.readlines():
                        words = line.split(' ')
                        video_name = words[0]
                        begin = int(words[1])
                        label = int(words[2])

                        if len(words) == 3:
                            for idx in range(2):
                                video_path = os.path.join(video_root, path_type[idx], video_name)
                                if os.path.exists(video_path):
                                    break
                                elif idx == 1:
                                    assert (os.path.exists(video_path))
                                else:
                                    continue
                        else:
                            video_path = os.path.join(video_root, words[3].split('\n')[0], video_name)
                        info = {'video_path': video_path, 'file_name': video_name, 'pos': (begin, begin + 15), 'label': label}
                        if dataloader_file_name not in gts.keys():
                            gts[dataloader_file_name] = list()
                        gts[dataloader_file_name] += [info]
                        
        total_start_time = time.time()
        for dataloader_file_name, data in gts.items():
            for dic in data:
                per_video_start_time = time.time()

                video_path = dic['video_path']
                file_name = dic['file_name']
                pos_info = dic['pos']
                label = dic['label']
                if label == 2:
                    transition_type = 'cut'
                elif label == 1:
                    transition_type = 'gradual'
                else:
                    continue

                images_path = os.path.join('images', dataloader_file_name, file_name)
                if not os.path.exists(images_path):
                    os.makedirs(images_path)
                    os.mkdir(os.path.join(images_path, 'cut'))
                    os.mkdir(os.path.join(images_path, 'gradual'))

                start, end = pos_info
                image_path = os.path.join(images_path, transition_type)
                print_frames(video_path, start, end, image_path)

                per_video_end_time = time.time() - per_video_start_time

                print('video {}({} - {}) : {}'.format(file_name, start, end, per_video_end_time), flush=True)
            total_time = time.time() - total_start_time
            print('Total video({}) : {}'.format(dataloader_file_name, datetime.timedelta(seconds=total_time)), flush=True)
        
        return 0


def get_images(path_type, video_name_list, gts):
    for dataset_idx, p_type in enumerate(path_type):
        # file_name = "hUoDOxOxK1I.mp4"

        total_start_time = time.time()
        for i, file_name in enumerate(video_name_list[p_type]):
            per_video_start_time = time.time()

            video_base_dir = "ClipShots/videos/"
            video_path = os.path.join(video_base_dir, p_type, file_name)

            if path_type != 'result':
                pos_info = gts[p_type][file_name]["transitions"]
                if file_name == 'hUoDOxOxK1I.mp4':
                    pos_info += [[3113, 3133]]
            else:
                pos_info = gts[p_type][file_name]

            images_path = os.path.join('images', p_type, file_name)

            if not os.path.exists(images_path):
                os.makedirs(images_path)
                os.mkdir(os.path.join(images_path, 'cut'))
                os.mkdir(os.path.join(images_path, 'gradual'))

            if path_type != 'result':
                for frame_pos, frame_end in pos_info:
                    if frame_end - frame_pos == 1:
                        transition_type = 'cut'
                        frame_pos -= 3
                        frame_end += 3
                    else:
                        transition_type = 'gradual'
                    image_path = os.path.join(images_path, transition_type)
                    print_frames(video_path, frame_pos, frame_end, image_path)

            else:
                for type, data in pos_info.items():
                    for frame_pos, frame_end in data:
                        transition_type = type
                        image_path = os.path.join(images_path, transition_type)
                        print_frames(video_cap, frame_pos, frame_end, image_path)

            per_video_end_time = time.time() - per_video_start_time

            print('video #{} {} : {}'.format(i, file_name, per_video_end_time), flush=True)
        total_time = time.time() - total_start_time
        print('Total video({}) : {}'.format(p_type, datetime.timedelta(seconds=total_time)), flush=True)


if __name__ == '__main__':
    path_type = ['train', 'only_gradual', 'test', 'result']
    check_dataset = [False, False, False, False]
    dataloader_name = ['deepSBD', 'detector', 'detector_new']
    check_dataloader = [False, True, False]

    if True in check_dataset:
        video_name_list, gts = get_gt_dirs(path_type, check_dataset)
        get_images(path_type, video_name_list, gts)
    elif True in check_dataloader:
        get_gt_dirs(path_type, check_dataloader, dataloader_name)
