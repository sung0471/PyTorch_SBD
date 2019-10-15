import os
import json
from data.train_data_loader import DataSet as train_dataset
from data.test_data_loader import DataSet as test_dataset
from opts import parse_opts


def check_data_list_function(video_list_path, dataset_type_list):
    train_dataset_count = dict()
    for dataset_type in dataset_type_list:
        train_dataset_count[dataset_type] = 0
        train_dataset_count["duplicate"] = list()
    with open(video_list_path, 'r') as f:
        for line in f.readlines():
            check_duplicate = 0
            words = line.split(' ')
            for i, dataset_type in enumerate(dataset_type_list):
                if words[0] in video_name_list[dataset_type]["list"]:
                    train_dataset_count[dataset_type] += 1
                    check_duplicate += 1
            if check_duplicate > 1:
                train_dataset_count["duplicate"] += [{"video_name": words[0], "duplicate": check_duplicate}]

    print("training dataset count : {}".format(train_dataset_count))


def check_train_dataset_function(video_root, video_list_path, opt):
    train_root_dir_list = list()
    train_root_dir_list.append(os.path.join(video_root, opt.train_subdir))
    train_root_dir_list.append(os.path.join(video_root, opt.only_gradual_subdir))
    print("[INFO] reading : ", video_list_path, flush=True)
    dataset = train_dataset(train_root_dir_list, video_list_path, opt)
    train_data_list = dataset.video_list

    total_count = dict()
    total_count["train"] = dict()
    total_count["only_gradual"] = dict()
    for key in total_count.keys():
        total_count[key]["cut"] = 0
        total_count[key]["gradual"] = 0
        total_count[key]["background"] = 0

    video_count = dict()
    video_count["train"] = dict()
    video_count["only_gradual"] = dict()
    for key in video_count.keys():
        video_count[key]["list"] = list()
        video_count[key]["count"] = 0

    for line in train_data_list:
        line_video_path = line["video_path"]
        if "train" in line_video_path:
            dataset_type = "train"
        else:
            dataset_type = "only_gradual"

        if not (line_video_path in video_count[dataset_type]["list"]):
            video_count[dataset_type]["list"].append(line_video_path)
            video_count[dataset_type]["count"] += 1

        if line["label"] == 2:
            total_count[dataset_type]["cut"] += 1
        elif line["label"] == 1:
            total_count[dataset_type]["gradual"] += 1
        else:
            total_count[dataset_type]["background"] += 1

    print("train dataset total : {}".format(total_count))
    print("train video count : train({}), only_gradual({})".
          format(video_count["train"]["count"], video_count["only_gradual"]["count"]))


def check_test_dataset_function(video_root, video_name_list, opt, gts):
    test_root_dir = os.path.join(video_root, opt.test_subdir)
    datset = test_dataset(test_root_dir, video_name_list["test"]["list"])
    test_data_list = datset.video_list

    test_total = dict()
    test_total["train"] = dict()
    test_total["only_gradual"] = dict()
    test_total["test"] = dict()
    for key in test_total.keys():
        test_total[key]["cut"] = 0
        test_total[key]["gradual"] = 0
        test_total[key]["total"] = 0
        test_total[key]["else"] = {"length": 0, "list": list()}

    test_video_count = []
    for line in test_data_list:
        line_video_path = line["video_path"]

        if not (line_video_path in test_video_count):
            test_video_count.append(line_video_path)

    max_transition = {"info": (0, 0), "value": 0}
    min_transition = {"info": (0, 0), "value": 128}
    for dir_name, gt in gts.items():
        for video_name, data in gt.items():
            _gts = data['transitions']
            gt_cuts = [(begin, end) for begin, end in _gts if end - begin == 1]
            gt_graduals = [(begin, end) for begin, end in _gts if end - begin > 1]
            gt_else = [(begin, end) for begin, end in _gts if end - begin < 1]
            test_total[dir_name]["cut"] += len(gt_cuts)
            test_total[dir_name]["gradual"] += len(gt_graduals)
            test_total[dir_name]["total"] += len(_gts)
            test_total[dir_name]["else"]["list"] += gt_else

            for begin, end in gt_graduals:
                if end - begin > max_transition["value"]:
                    max_transition["value"] = end - begin
                    max_transition["info"] = (begin, end)
                if end - begin < min_transition["value"]:
                    min_transition["value"] = end - begin
                    min_transition["info"] = (begin, end)

            # for begin, end in gt_else:
            #     if begin - end > max_transition["value"]:
            #         max_transition["value"] = begin - end
            #         max_transition["info"] = (begin, end)
            #     if begin - end < min_transition["value"]:
            #         min_transition["value"] = begin - end
            #         min_transition["info"] = (begin, end)

        test_total[dir_name]["else"]["length"] = len(test_total[dir_name]["else"]["list"])
        print("{} : {}".format(dir_name, test_total[dir_name]))
    print("transition length : max({}), min({})".format(max_transition, min_transition))


def check_data_set_function(root_dir, video_list_path, video_name_list, gts):
    correct = dict()
    correct["cut"] = 0
    correct["gradual"] = 0
    correct["incorrect"] = 0
    case = list()
    with open(video_list_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            video_name, frame, label = line.split(' ')
            frame, label = int(frame), int(label)

            if i % 50000 == 0:
                print("finish {}".format(i))
            dataset_type = ""
            for gt_location in gts.keys():
                if video_name in video_name_list[gt_location]["list"]:
                    dataset_type = gt_location
                    break
            find = False
            if label > 0:
                for begin, end in gts[dataset_type][video_name]["transitions"]:
                    if frame <= begin <= (frame + 16) or frame <= end <= (frame + 16) or begin <= frame <= end:
                        if label == 2:
                            correct["cut"] += 1
                        elif label == 1:
                            correct["gradual"] += 1
                        else:
                            pass
                        find = True
                        break
                if not find:
                    correct["incorrect"] += 1
    print(correct)


def check_gt_transition_min_max_function(gts):
    rank_of_transition_length = dict()
    possible_detection = dict()
    max_length = 10

    for type in ['16','32','64']:
        possible_detection[type] = dict()
        possible_detection[type]['total'] = [0, 0]

    for path_name, gt in gts.items():
        for type in ['16', '32', '64']:
            possible_detection[type][path_name] = [0, 0]
        for video_name, data in gt.items():
            _gts = data['transitions']
            gt_cuts = [(begin, end) for begin, end in _gts if end - begin == 1]
            gt_graduals = [(begin, end) for begin, end in _gts if end - begin > 1]
            gt_else = [(begin, end) for begin, end in _gts if end - begin < 1]

            for type in ['16', '32', '64']:
                possible_detection[type][path_name][1] += len(gt_graduals)

            for (begin, end) in gt_graduals:
                for type in ['16', '32', '64']:
                    if end - begin < int(type) + 1:
                        possible_detection[type][path_name][0] += 1

                if path_name not in rank_of_transition_length.keys():
                    rank_of_transition_length[path_name] = [(video_name, begin, end, end - begin)]
                else:
                    for i, (v_name, s, e, l) in enumerate(rank_of_transition_length[path_name]):
                        if end-begin > l:
                            rank_of_transition_length[path_name].insert(i, (video_name, begin, end, end - begin))
                            if len(rank_of_transition_length[path_name]) >= max_length:
                                rank_of_transition_length[path_name] = rank_of_transition_length[path_name][:max_length]
                            break

        for type in ['16', '32', '64']:
            for i, value in enumerate(possible_detection[type][path_name]):
                possible_detection[type]['total'][i] += value
        print(rank_of_transition_length[path_name])

    for type in ['16', '32', '64']:
        print('---------------------{} frame---------------------'.format(type))
        for key in possible_detection[type].keys():
            i, j = possible_detection[type][key][0], possible_detection[type][key][1]
            print('{} : {} / {} ({:.3f})'.format(key, i, j, i/j))


if __name__ == '__main__':
    opt = parse_opts()
    root_dir = "ClipShots"

    video_list_root = os.path.join(root_dir, "video_lists")
    dataset_type_list = ["train", "only_gradual", "test"]

    video_path_dict = dict()
    for dataset_type in dataset_type_list:
        video_path_dict[dataset_type] = os.path.join(video_list_root, dataset_type + ".txt")

    video_name_list = dict()
    for dataset_type, list_path in video_path_dict.items():
        with open(list_path, 'r') as f:
            video_name_list[dataset_type] = dict()
            video_name_list[dataset_type]["list"] = [line.strip('\n') for line in f.readlines()]
            video_name_list[dataset_type]["count"] = len(video_name_list[dataset_type]["list"])

    print("all video count : train({}), only_gradual({}), test({})".
          format(video_name_list["train"]["count"],
                 video_name_list["only_gradual"]["count"],
                 video_name_list["test"]["count"]))

    video_list_path = "data_list/deepSBD.txt"

    check_data_list = False
    check_train_dataset = False
    check_test_dataset = False
    check_data_set = False
    check_gt_transition_min_max = False
    dataset_category_check = True

    if check_data_list:
        check_data_list_function(video_list_path, dataset_type_list)

    video_root = os.path.join(root_dir, "videos")
    if check_train_dataset:
        check_train_dataset_function(video_root, video_list_path, opt)

    gts = dict()
    if check_test_dataset or check_data_set or check_gt_transition_min_max:
        gt_base_dir = os.path.join(root_dir, "annotations")
        gt_dir_dic = dict()
        for path_name in dataset_type_list:
            gt_dir_dic[path_name] = os.path.join(gt_base_dir, path_name + ".json")
            gts[path_name] = json.load(open(gt_dir_dic[path_name], 'r'))
        print('gt_dir_dic : {}'.format(gt_dir_dic))

    if check_test_dataset:
        check_test_dataset_function(video_root, video_name_list, opt, gts)

    if check_data_set:
        check_data_set_function(root_dir, video_list_path, video_name_list, gts)

    if check_gt_transition_min_max:
        check_gt_transition_min_max_function(gts)

    if dataset_category_check:
        l = list()
        test_dataset_list = ['board', 'draw', 'etc+religion', 'living_things', 'marine_activity', 'performance', 'playing', 'sport', 'vehicle']
        for file_name in test_dataset_list:
            txt_path = os.path.join(video_list_root, 'test_' + file_name + '.txt')
            with open(txt_path, 'r') as f:
                l += [line.strip('\n') for line in f.readlines()]
        txt_path = os.path.join(video_list_root, 'test.txt')
        rest = list()
        with open(txt_path, 'r') as f:
            total = [line.strip('\n') for line in f.readlines()]
            for name in total:
                if name not in l:
                    rest += [name]
        print(rest)