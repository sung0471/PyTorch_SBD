import os
import json

make_dataset_from_groundTruth = True
merge_dataset_with_deepSBD = True
check_data_list = True

data_list_root = 'data_list'
new_file_name = 'detector_list.txt'
dataset_path = os.path.join(data_list_root, new_file_name)

if make_dataset_from_groundTruth:
    video_root = 'ClipShots'
    video_list_dir = os.path.join('ClipShots', 'video_lists')
    video_gt_dir = os.path.join('ClipShots', 'annotations')
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
                if end-start == 1:
                    gt[dir_name][video_name]['cut'] += [(start, end)]
                elif end-start > 1:
                    gt[dir_name][video_name]['gradual'] += [(start, end)]

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

if check_data_list:
    count = dict()
    with open(dataset_path, 'rt', encoding='utf-8') as f:
        for line in f.readlines():
            class_type = line.split(' ')[2]
            class_type = class_type.split('\n')[0]
            if class_type not in count.keys():
                count[class_type] = 0
            count[class_type] += 1

    print(count)