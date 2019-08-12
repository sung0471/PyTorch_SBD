import os
import json

make_dataset_from_groundTruth = False
merge_dataset_with_deepSBD = False
make_dataset_using_deepSBD = True
check_count_of_data_list = True

data_list_root = 'data_list'
new_file_name = 'detector.txt'
dataset_path = os.path.join(data_list_root, new_file_name)

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

if make_dataset_from_groundTruth:
    count = dict()
    with open(dataset_path, 'wt', encoding='utf-8') as f:
        for dir_name, data_list in gt.items():
            data_list_path = os.path.join(data_list_root, dir_name + '.json')
            json.dump(data_list, open(data_list_path, 'w'))

            count[dir_name] = dict()
            count[dir_name]['cut'] = 0
            count[dir_name]['gradual'] = 0
            for video_name, transitions in data_list.items():
                for transition_type, transition_list in transitions.items():
                    for (start, end) in transition_list:
                        class_type = 2 if transition_type == 'cut' else 1

                        start_1 = start - start % 8
                        count[dir_name][transition_type] += 1
                        f.write('{} {} {} {} {}\n'.format(video_name, start_1, class_type, start, end))
                        if transition_type == 'cut':
                            start_2 = start_1 - 8
                            if start_2 >= 0:
                                count[dir_name][transition_type] += 1
                                f.write('{} {} {} {} {}\n'.format(video_name, start_2, class_type, start, end))
                        else:
                            start_2 = start_1 + 8
                            while (end - start_2) > 2:
                                count[dir_name][transition_type] += 1
                                f.write('{} {} {} {} {}\n'.format(video_name, start_2, class_type, start, end))
                                start_2 += 8

    print(count)

if merge_dataset_with_deepSBD:
    deepSBD_path = os.path.join(data_list_root, 'deepSBD.txt')
    destination = open(dataset_path, 'at', encoding='utf-8')
    with open(deepSBD_path, 'rt', encoding='utf-8') as f:
        for line in f.readlines():
            video_name, start, class_type = line.split(' ')
            class_type = int(class_type.split('\n')[0])
            if class_type == 0:
                destination.write('{} {} {}\n'.format(video_name, start, class_type))

if make_dataset_using_deepSBD:
    deepSBD_path = os.path.join(data_list_root, 'deepSBD.txt')
    destination = open(dataset_path, 'wt', encoding='utf-8')
    transition_type = ['no', 'gradual', 'cut']
    sample_duration = 16
    count = [0, 0]
    with open(deepSBD_path, 'rt', encoding='utf-8') as f:
        for line in f.readlines():
            video_name, start, label = line.split(' ')
            start = int(start)
            label = int(label.split('\n')[0])
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
                        destination.write('{} {} {} {} {}\n'.format(video_name, start, label, gt_s, gt_e))
                        break
                count[check] += 1
                if check:
                    check = not check
                    print('{} {} {} {} {}'.format(video_name, start, label, gt_s, gt_e))
                else:
                    print('{}/{} : no gt'.format(dir_name, video_name))
            else:
                destination.write('{} {} {}\n'.format(video_name, start, label))
                count[1] += 1

    print('gt/no_gt : {}/{}'.format(count[1], count[0]))

if check_count_of_data_list:
    count = dict()
    with open(dataset_path, 'rt', encoding='utf-8') as f:
        for line in f.readlines():
            class_type = line.split(' ')[2]
            class_type = class_type.split('\n')[0]
            if class_type not in count.keys():
                count[class_type] = 0
            count[class_type] += 1

    print(count)