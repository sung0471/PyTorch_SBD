import os
import json

make_deepSBD_new = False
make_detector_from_groundTruth_and_deepSBD_new = False
make_detector_using_deepSBD_new = True
check_count_of_data_list = True

data_list_root = 'data_list'
deepSBD_new_name = 'deepSBD_new.txt'
detector_name = 'detector.txt'

dataset_root = 'ClipShots'
video_list_dir = os.path.join(dataset_root, 'video_lists')
video_gt_dir = os.path.join(dataset_root, 'annotations')
category = ['train', 'only_gradual']

gt = dict()
for dir_name in category:
    video_list_path = os.path.join(video_list_dir, dir_name + '.txt')
    video_gt_path = os.path.join(video_gt_dir, dir_name + '.json')

    gt[dir_name] = dict()
    video_gt = json.load(open(video_gt_path, 'r'))
    for video_name, data in video_gt.items():
        gt[dir_name][video_name] = dict()
        gt[dir_name][video_name]['cut'] = list()
        gt[dir_name][video_name]['gradual'] = list()
        for start, end in data['transitions']:
            if end - start == 1:
                gt[dir_name][video_name]['cut'] += [(start, end)]
            elif end - start > 1:
                gt[dir_name][video_name]['gradual'] += [(start, end)]

deepSBD_path = os.path.join(data_list_root, 'deepSBD.txt')
deepSBD_new_path = os.path.join(data_list_root, deepSBD_new_name)
if make_deepSBD_new:
    print('Start making deepSBD_new.txt')
    destination = open(deepSBD_new_path, 'w', encoding='utf-8')
    with open(deepSBD_path, 'rt', encoding='utf-8') as f:
        lines = f.readlines()
        total_length = len(lines)
        print('Read {} : {}'.format(deepSBD_path, total_length))
        for count, line in enumerate(lines):
            words = line.split(' ')
            video_name = words[0]
            start = words[1]
            class_type = int(words[2].split('\n')[0])
            for i in range(2):
                video_dir = os.path.join(dataset_root, 'videos/', category[i])
                video_path = os.path.join(video_dir, video_name)
                if os.path.exists(video_path):
                    dir_name = category[i]
                    break
                elif i == 1:
                    assert (os.path.exists(video_path))
                else:
                    continue
            destination.write('{} {} {} {}\n'.format(video_name, start, class_type, dir_name))
            if (count + 1) % 50000 == 0:
                print('processing {}/{} ...'.format(count + 1, total_length))
    print('Finish making deepSBD_new.txt')

new_dataset_path = os.path.join(data_list_root, detector_name)
if make_detector_from_groundTruth_and_deepSBD_new:
    print('Start making detector dataset from groundTruth and deepSBD new')
    with open(new_dataset_path, 'wt', encoding='utf-8') as f:
        for dir_name, data_list in gt.items():
            # data_list_path = os.path.join(data_list_root, dir_name + '.json')
            # json.dump(data_list, open(data_list_path, 'wt'))

            for video_name, transitions in data_list.items():
                for transition_type, transition_list in transitions.items():
                    for (start, end) in transition_list:
                        class_type = 2 if transition_type == 'cut' else 1

                        start_1 = start - start % 8
                        f.write('{} {} {} {} {} {}\n'.format(
                            video_name, start_1, class_type, dir_name, start, end))
                        if transition_type == 'cut':
                            start_2 = start_1 - 8
                            if start_2 >= 0:
                                f.write('{} {} {} {} {} {}\n'.format(
                                    video_name, start_2, class_type, dir_name, start, end))
                        else:
                            start_2 = start_1 + 8
                            while (end - start_2) > 2:
                                f.write('{} {} {} {} {} {}\n'.format(
                                    video_name, start_2, class_type, dir_name, start, end))
                                start_2 += 8
    print('Finish making detector dataset from groundTruth')

    deepSBD_new = deepSBD_new_path
    destination = open(new_dataset_path, 'at', encoding='utf-8')
    with open(deepSBD_new, 'rt', encoding='utf-8') as f:
        for line in f.readlines():
            words = line.split(' ')
            video_name = words[0]
            start = words[1]
            class_type = int(words[2])
            dir_name = words[3]
            if class_type == 0:
                destination.write('{} {} {} {}'.format(video_name, start, class_type, dir_name))
    print('Finish making detector dataset from deepSBD new')

if make_detector_using_deepSBD_new:
    print('Start making detector dataset using deepSBD new')

    deepSBD_new = deepSBD_new_path
    destination = open(new_dataset_path, 'wt', encoding='utf-8')
    transition_type = ['no', 'gradual', 'cut']
    sample_duration = 16
    count = [0, 0]
    no_gt = dict()
    with open(deepSBD_new, 'rt', encoding='utf-8') as f:
        for line in f.readlines():
            words = line.split(' ')
            video_name = words[0]
            start = int(words[1])
            label = int(words[2])
            dir_name = words[3].split('\n')[0]
            if label != 0:
                end = start + sample_duration - 1
                check = False
                for gt_s, gt_e in gt[dir_name][video_name][transition_type[label]]:
                    if start <= gt_s < end <= gt_e:
                        check = not check
                    elif start <= gt_s < gt_e <= end:
                        check = not check
                    elif gt_s <= start < end <= gt_e:
                        check = not check
                    elif gt_s <= start < gt_e <= end:
                        check = not check
                    else:
                        continue

                    if check:
                        destination.write('{} {} {} {} {} {}\n'.format(video_name, start, label, dir_name, gt_s, gt_e))
                        break
                count[check] += 1
                if check:
                    check = not check
                else:
                    if dir_name not in no_gt.keys():
                        no_gt[dir_name] = list()
                    no_gt[dir_name] += [[video_name, start, label, dir_name]]
                    print('{}/{} : no gt'.format(dir_name, video_name))
            else:
                destination.write('{} {} {} {}\n'.format(video_name, start, label, dir_name))
                count[1] += 1

    print('gt/no_gt : {}/{}'.format(count[1], count[0]))
    json.dump(no_gt, open('no_gt.json', 'wt', encoding='utf-8'), indent=2)

if check_count_of_data_list:
    count = dict()
    with open(new_dataset_path, 'rt', encoding='utf-8') as f:
        for line in f.readlines():
            class_type = line.split(' ')[2]
            class_type = class_type.split('\n')[0]
            if class_type not in count.keys():
                count[class_type] = 0
            count[class_type] += 1

    print(count)
