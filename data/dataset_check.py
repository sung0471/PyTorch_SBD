import os
import json
from data.train_data_loader import DataSet as train_dataset
from data.test_data_loader import DataSet as test_dataset
from opts import parse_opts

if __name__ == '__main__':
    opt = parse_opts()
    root_dir = "ClipShots"

    video_path_list = []
    video_path_list.append(os.path.join(root_dir, "video_lists", "train.txt"))
    video_path_list.append(os.path.join(root_dir, "video_lists", "only_gradual.txt"))
    video_path_list.append(os.path.join(root_dir, "video_lists", "test.txt"))
    video_name_list = []
    for list_path in video_path_list:
        with open(list_path, 'r') as f:
            video_name_list.append([line.strip('\n') for line in f.readlines()])

    print("video list result : train({}), only_gradual({}), test({})".format(len(video_name_list[0]),len(video_name_list[1]),len(video_name_list[2])))

    list_root_path = list()
    video_list_path = "data_list/deepSBD.txt"
    count = {}
    count["train"] = 0
    count["only_gradual"] = 0
    count["test"] = 0
    count["duplicate"] = []
    with open(video_list_path, 'r') as f:
        for line in f.readlines():
            check_duplicate = 0
            words = line.split(' ')
            if words[0] in video_name_list[0]:
                count["train"] += 1
                check_duplicate += 1
            if words[0] in video_name_list[1]:
                count["only_gradual"] += 1
                check_duplicate += 1
            if words[0] in video_name_list[2]:
                count["test"] +=1
                check_duplicate += 1
            if check_duplicate > 1:
                count["duplicate"] += [{"video_name": words[0], "duplicate": check_duplicate}]

    print("video count : {}".format(count))

    list_root_path.append(os.path.join(root_dir, 'videos', opt.train_subdir))
    list_root_path.append(os.path.join(root_dir, 'videos', opt.only_gradual_subdir))
    print(list_root_path, flush=True)
    print("[INFO] reading : ", video_list_path, flush=True)
    training_data = train_dataset(list_root_path, video_list_path, opt)
    train_dataset = training_data.video_list

    total = {}
    total["train"] = {}
    total["only_gradual"] = {}
    for key in total.keys():
        total[key]["cut"] = 0
        total[key]["gradual"] = 0

    video = {}
    video["train"] = []
    video["only_gradual"] = []
    for line in train_dataset:
        line_video_path = line["video_path"]
        if "train" in line_video_path:
            video_location = "train"
        else:
            video_location = "only_gradual"

        if not line_video_path in video[video_location]:
            video[video_location].append(line_video_path)

        if line["label"] == 2:
            total[video_location]["cut"] += 1
        else:
            total[video_location]["gradual"] += 1
    print(total, "video number : train({}), only_gradual({})".format(len(video["train"]), len(video["only_gradual"])))

    root_dir = os.path.join("ClipShots", "videos", opt.test_subdir)
    test_list_path = "ClipShots/video_lists/test.txt"
    with open(test_list_path, 'r') as f:
        video_name_list = [line.strip('\n') for line in f.readlines()]
    test_data = test_dataset(root_dir, video_name_list)
    video_list = test_data.video_list

    total = {}
    total["train"] = {}
    total["only_gradual"] = {}
    total["test"] = {}
    for key in total.keys():
        total[key]["cut"] = 0
        total[key]["gradual"] = 0
        total[key]["total"] = 0
        total[key]["else"] = {"length":0, "list":[]}

    video = []
    for line in video_list:
        line_video_path = line["video_path"]

        if not line_video_path in video:
            video.append(line_video_path)

    gts = {}
    gt_base_dir = "ClipShots/annotations"
    train_gt_dir = os.path.join(gt_base_dir, "train.json")
    only_gradual_gt_dir = os.path.join(gt_base_dir, "only_gradual.json")
    test_gt_dir = os.path.join(gt_base_dir, "test.json")
    gts["train"] = json.load(open(train_gt_dir, 'r'))
    gts["only_gradual"] = json.load(open(only_gradual_gt_dir, 'r'))
    gts["test"] = json.load(open(test_gt_dir, 'r'))

    max_transition = {"info": (0, 0), "value": 0}
    min_transition = {"info": (0, 0), "value": 128}
    for dir_name, gt in gts.items():
        for video_name, data in gt.items():
            _gts = data['transitions']
            gt_cuts = [(begin, end) for begin, end in _gts if end - begin == 1]
            gt_graduals = [(begin, end) for begin, end in _gts if end - begin > 1]
            gt_else = [(begin, end) for begin, end in _gts if end - begin < 1]
            total[dir_name]["cut"] += len(gt_cuts)
            total[dir_name]["gradual"] += len(gt_graduals)
            total[dir_name]["total"] += len(_gts)
            total[dir_name]["else"]["list"] += gt_else

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

        total[dir_name]["else"]["length"] = len(total[dir_name]["else"]["list"])
        print(total[dir_name])
    print("transition length : max({}), min({})".format(max_transition, min_transition))
