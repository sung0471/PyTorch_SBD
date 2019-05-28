import os
import json
from data.train_data_loader import DataSet as train_dataset
from data.test_data_loader import DataSet as test_dataset
from opts import parse_opts

if __name__ == '__main__':
    opt = parse_opts()
    root_dir = "ClipShots"

    video_list_root = os.path.join(root_dir, "video_lists")
    video_location_list = ["train", "only_gradual", "test"]

    video_path_dict = dict()
    for video_location in video_location_list:
        video_path_dict[video_location] = os.path.join(video_list_root, video_location + ".txt")

    video_name_list = dict()
    for video_location, list_path in video_path_dict.items():
        with open(list_path, 'r') as f:
            video_name_list[video_location] = dict()
            video_name_list[video_location]["list"] = [line.strip('\n') for line in f.readlines()]
            video_name_list[video_location]["count"] = len(video_name_list[video_location]["list"])

    print("all video count : train({}), only_gradual({}), test({})".
          format(video_name_list["train"]["count"],
                 video_name_list["only_gradual"]["count"],
                 video_name_list["test"]["count"]))

    video_list_path = "data_list/deepSBD.txt"

    check_data_list = False
    check_train_dataset = False
    check_test_dataset = False

    if check_data_list:
        train_dataset_count = dict()
        for video_location in video_location_list:
            train_dataset_count[video_location] = 0
            train_dataset_count["duplicate"] = list()
        with open(video_list_path, 'r') as f:
            for line in f.readlines():
                check_duplicate = 0
                words = line.split(' ')
                for i, video_location in enumerate(video_location_list):
                    if words[0] in video_name_list[video_location]["list"]:
                        train_dataset_count[video_location] += 1
                        check_duplicate += 1
                if check_duplicate > 1:
                    train_dataset_count["duplicate"] += [{"video_name": words[0], "duplicate": check_duplicate}]

        print("training dataset count : {}".format(train_dataset_count))

    video_root = os.path.join(root_dir, "videos")
    if check_train_dataset:
        train_root_dir_list = list()
        train_root_dir_list.append(os.path.join(video_root, opt.train_subdir))
        train_root_dir_list.append(os.path.join(video_root, opt.only_gradual_subdir))
        print("[INFO] reading : ", video_list_path, flush=True)
        train_dataset = train_dataset(train_root_dir_list, video_list_path, opt)
        train_data_list = train_dataset.video_list

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
                video_location = "train"
            else:
                video_location = "only_gradual"

            if not (line_video_path in video_count[video_location]["list"]):
                video_count[video_location]["list"].append(line_video_path)
                video_count[video_location]["count"] += 1

            if line["label"] == 2:
                total_count[video_location]["cut"] += 1
            elif line["label"] == 1:
                total_count[video_location]["gradual"] += 1
            else:
                total_count[video_location]["background"] += 1

        print("train dataset total : {}".format(total_count))
        print("train video count : train({}), only_gradual({})".
              format(video_count["train"]["count"], video_count["only_gradual"]["count"]))

    if check_test_dataset:
        test_root_dir = os.path.join(video_root, opt.test_subdir)
        test_dataset = test_dataset(test_root_dir, video_name_list["test"]["list"])
        test_data_list = test_dataset.video_list

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

    check_data_set = True
    if check_data_set:
        gts = {}
        gt_base_dir = os.path.join(root_dir, "annotations")
        train_gt_dir = os.path.join(gt_base_dir, "train.json")
        only_gradual_gt_dir = os.path.join(gt_base_dir, "only_gradual.json")
        gts["train"] = json.load(open(train_gt_dir, 'r'))
        gts["only_gradual"] = json.load(open(only_gradual_gt_dir, 'r'))

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
                video_location = ""
                for gt_location in gts.keys():
                    if video_name in video_name_list[gt_location]["list"]:
                        video_location = gt_location
                        break
                find = False
                if label > 0:
                    for begin, end in gts[video_location][video_name]["transitions"]:
                        if frame <= begin <= (frame+16) or frame <= end <= (frame+16) or begin <= frame <= end:
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
