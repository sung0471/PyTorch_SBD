import os
import json
import cv2
import time
import datetime
from PIL import Image

dataset = ['ClipShots/', 'RAI/', 'TRECVID/']
dataset_idx = 2
dataset_root_dir = os.path.join(dataset[dataset_idx]) if dataset_idx != 2 else os.path.join(dataset[dataset_idx], '07/')
data_loader_list = 'data_list/'
video_root = os.path.join(dataset_root_dir, 'videos/')
list_path_dir = os.path.join(dataset_root_dir, 'video_lists/')
gt_base_dir = os.path.join(dataset_root_dir, 'annotations/')
gt_path = os.path.join(dataset_root_dir, 'annotations/', 'test.json')


def print_frames(video_path, images_root, pos_info=None, print_all=False):
    video_cap = cv2.VideoCapture(video_path)
    if not print_all:
        for frame_pos, frame_end in pos_info:
            video_cap.set(1, frame_pos)
            if frame_end - frame_pos == 1:
                transition_type = 'cut'
            else:
                transition_type = 'gradual'
            image_path = os.path.join(images_root, transition_type)
            for j in range(frame_pos, frame_end + 1):
                status, frame = video_cap.read()
                if not status:
                    break
                else:
                    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
                    frame.save(os.path.join(image_path, str(j) + '.png'))
    else:
        image_path = os.path.join(images_root, 'all')
        status = True
        j = 0
        while status:
            status, frame = video_cap.read()
            if not status:
                break
            else:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
                frame.save(os.path.join(image_path, str(j) + '.png'))
                j += 1


def get_gt_dirs(path_type, check_dataset, dataloader_name=None):
    video_name_list = dict()
    gts = dict()
    if dataloader_name is None:
        for i, p_type in enumerate(path_type):
            if check_dataset[i]:
                if p_type in ['train', 'only_gradual']:
                    file_list_path = os.path.join(list_path_dir, p_type + '.txt')
                else:
                    file_list_path = os.path.join(list_path_dir, 'test.txt')

                with open(file_list_path, 'r') as f:
                    video_name_list[p_type] = [line.strip('\n') for line in f.readlines()]

                if p_type != 'result':
                    gt_path = os.path.join(gt_base_dir, p_type + '.json')
                else:
                    gt_path = os.path.join('../results/', 'results.json')

                gts[p_type] = json.load(open(gt_path, 'r'))
        return video_name_list, gts
    else:
        for i, dataloader_file_name in enumerate(dataloader_name):
            if check_dataset[i]:
                dataloader_path = os.path.join(data_loader_list, dataloader_file_name + '.txt')
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


def get_images(path_type, video_name_list, gts, print_all=False):
    for p_type in path_type:
        # file_name = "hUoDOxOxK1I.mp4"

        total_start_time = time.time()
        if p_type in video_name_list.keys():
            for i, file_name in enumerate(video_name_list[p_type]):
                images_root = os.path.join('images', p_type, file_name)
                if not os.path.exists(images_root):
                    os.makedirs(images_root)
                    os.mkdir(os.path.join(images_root, 'cut'))
                    os.mkdir(os.path.join(images_root, 'gradual'))
                    os.mkdir(os.path.join(images_root, 'all'))

                print('video {} start'.format(file_name), flush=True)
                per_video_start_time = time.time()

                if not print_all:
                    if p_type != 'result':
                        video_path = os.path.join(video_root, p_type, file_name)
                    else:
                        video_path = os.path.join(video_root, 'test', file_name)

                    if p_type != 'result':
                        pos_info = gts[p_type][file_name]["transitions"]
                        if file_name == 'hUoDOxOxK1I.mp4':
                            pos_info += [[3113, 3133]]
                    else:
                        pos_info = gts[p_type][file_name]

                    if p_type != 'result':
                        for j, (frame_pos, frame_end) in enumerate(pos_info):
                            if frame_end - frame_pos == 1:
                                frame_pos -= 1
                                frame_end += 1
                                pos_info[j] = [frame_pos, frame_end]
                        print_frames(video_path, images_root, pos_info)
                    else:
                        for _, data in pos_info.items():
                            print_frames(video_path, images_root, data)
                else:
                    video_path = os.path.join(video_root, 'test', file_name)
                    print_frames(video_path, images_root, print_all=print_all)

                per_video_end_time = time.time() - per_video_start_time

                print('video #{} {} : {:.3f}s'.format(i + 1, file_name, per_video_end_time), flush=True)
            total_time = time.time() - total_start_time
            print('Total video({}) : {}'.format(p_type, datetime.timedelta(seconds=total_time)), flush=True)


if __name__ == '__main__':
    path_type = ['train', 'only_gradual', 'test', 'result']
    check_dataset = [False, False, False, False]
    data_loader_name = ['deepSBD', 'detector', 'detector_new']
    check_data_loader = [False, False, False]

    if True in check_dataset:
        video_name_list, gts = get_gt_dirs(path_type, check_dataset)
        get_images(path_type, video_name_list, gts)
    elif True in check_data_loader:
        get_gt_dirs(path_type, check_data_loader, data_loader_name)

    video_to_frame = True
    print_all = True
    if video_to_frame:
        video_name = ['BG_11362.mpg']
        video_name_list = {'test': video_name}
        # gt_path = os.path.join('../results/_draw/', 'results.json')
        gt = json.load(open(gt_path, 'rt'))
        gts = {'test': gt}
        get_images(['test'], video_name_list, gts, print_all)
