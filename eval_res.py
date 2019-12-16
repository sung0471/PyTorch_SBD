import json
import os
from opts import parse_opts


def if_overlap(begin1, end1, begin2, end2):
    if begin1 > begin2:
        begin1, end1, begin2, end2 = begin2, end2, begin1, end1

    return end1 >= begin2


def get_union_cnt(set1, set2):
    cnt = 0
    tp = list()
    fp = list()
    fn = list()
    for begin, end in set1:
        gt_check = False
        for _begin, _end in set2:
            if if_overlap(begin, end, _begin, _end):
                if [_begin, _end] not in tp:
                    cnt += 1
                    tp.append([_begin, _end])
                    gt_check = not gt_check
                    break
        if not gt_check:
            fn.append([begin, end])
    for transition in set2:
        if transition not in tp:
            fp.append(transition)
    return cnt, (tp, fp, fn)


def pre_recall_f1(a, b, c):
    precision = a / c if c != 0 else\
        (1.0 if b == 0 else 0)
    recall = a / b if b != 0 else\
        (1.0 if c == 0 else 0)
    f1 = 2 * precision * recall / (precision + recall) if (precision != 0 or recall != 0) else 0
    return precision, recall, f1


def eval(opt):
    predict_path = os.path.join(opt.result_dir, 'results.json')
    predicts = json.load(open(predict_path))
    gts = json.load(open(opt.gt_dir))
    print(len(predicts))

    if opt.train_data_type == 'normal':
        transition_type = ['cut', 'gradual']
    else:
        transition_type = [opt.train_data_type]

    gt = dict()
    predict = dict()
    correct = dict()
    correct_sum = dict()
    gt_sum = dict()
    predict_sum = dict()

    tp_tn_fp_fn = dict()
    gt_len = dict()
    pred_len = dict()
    tp = dict()
    tn = dict()
    fp = dict()
    fn = dict()
    tp_fp_fn_list = dict()
    # for videoname, labels in gts.items():
    for videoname, _predicts in predicts.items():
        if videoname in gts:
            _gts = gts[videoname]['transitions']
            gt['cut'] = [(begin, end) for begin, end in _gts if end - begin == 1]
            gt['gradual'] = [(begin, end) for begin, end in _gts if end - begin > 1]
            gt['all'] = gt['cut'] + gt['gradual']

            # _predicts = predicts[videoname]
            tp_fp_fn_list[videoname] = dict()
            for type in transition_type:
                predict[type] = _predicts[type]
                correct[type], value_tuple = get_union_cnt(gt[type], predict[type])
                tp_fp_fn_list[videoname][type] = dict()
                tp_fp_fn_list[videoname][type]['tp'] = value_tuple[0]
                tp_fp_fn_list[videoname][type]['fp'] = value_tuple[1]
                tp_fp_fn_list[videoname][type]['fn'] = value_tuple[2]
                if type in correct_sum.keys():
                    correct_sum[type] += correct[type]
                else:
                    correct_sum[type] = 0

                gt_len[type] = len(gt[type])
                pred_len[type] = len(predict[type])

                if type in gt_sum.keys():
                    gt_sum[type] += gt_len[type]
                else:
                    gt_sum[type] = 0
                if type in predict_sum.keys():
                    predict_sum[type] += pred_len[type]
                else:
                    predict_sum[type] = 0

                tp[type] = correct[type]
                tn[type] = 0
                fp[type] = pred_len[type] - correct[type]
                fn[type] = gt_len[type] - correct[type]

            if len(transition_type) == 2:
                correct['all'], _ = get_union_cnt(gt['all'], predict['cut'] + predict['gradual'])
                if 'all' in correct_sum.keys():
                    correct_sum['all'] += correct['all']
                else:
                    correct_sum['all'] = 0

            # precision = tp / (tp + fp)
            # recall = tp / (tp + fn)
            tp_tn_fp_fn[videoname] = dict()
            for type in transition_type:
                precision, recall, f1 = pre_recall_f1(float(tp[type]), float(gt_len[type]), float(pred_len[type]))
                tp_tn_fp_fn[videoname][type] = {'tp': tp[type], 'tn': tn[type], 'fp': fp[type], 'fn': fn[type],
                                                'gt_len': gt_len[type], 'pred_len': pred_len[type],
                                                'precision': precision, 'recall': recall, 'f1_score': f1}
        else:
            print("{} not found in Ground Truth".format(videoname))
            raise Exception()

    tp_tn_fp_fn_path = os.path.join(opt.result_dir, 'tp_tn_fp_fn.json')
    tp_fp_fn_list_path = os.path.join(opt.result_dir, 'tp_fp_fn_list.json')
    json.dump(tp_tn_fp_fn, open(tp_tn_fp_fn_path, 'w'), indent=2)
    json.dump(tp_fp_fn_list, open(tp_fp_fn_list_path, 'w'), indent=2)

    print("group\tprecision\trecall\tf1score")
    for type in transition_type:
        print("{}\t{}\t{}\t{}".format(type, *pre_recall_f1(correct_sum[type], gt_sum[type], predict_sum[type])))
    if len(transition_type) == 2:
        gt_sum['all'] = gt_sum['cut'] + gt_sum['gradual']
        predict_sum['all'] = predict_sum['cut'] + predict_sum['gradual']
        print("all\t{}\t{}\t{}".format(*pre_recall_f1(correct_sum['all'], gt_sum['all'], predict_sum['all'])))


def candidate_eval(opt, gt_path):
    import torch
    from models.squeezenet import SqueezeNetFeature
    from lib.candidate_extracting import candidate_extraction

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SqueezeNetFeature().cuda(device)
    root_dir = os.path.join(opt.root_dir, opt.test_subdir)
    # print(root_dir, flush=True)
    # print(opt.test_list_path, flush=True)
    with open(opt.test_list_path, 'r') as f:
        video_name_list = [line.strip('\n') for line in f.readlines()]


    res = {}
    # print('\n====> Testing Start', flush=True)
    gts = json.load(open(gt_path))
    result = {}
    for idx, videoname in enumerate(video_name_list):
        print("Process {} {}".format(idx + 1, videoname), flush=True)
        result[videoname] = {}
        boundary_index_list = candidate_extraction(root_dir, videoname, model, adjacent=True)
        labels = gts[videoname]["transitions"]
        total_length = len(labels)
        count_included = 0

        boundary_index = 0
        boundary = boundary_index_list[0]
        for label in labels:
            while boundary < len(boundary_index_list):
                boundary = boundary_index_list[boundary_index]
                if boundary < label[0]:
                    pass
                elif label[0] <= boundary and boundary <= label[1]:
                    count_included += 1
                else:
                    break
                boundary_index += 1
        if total_length != 0:
            result[videoname] = count_included / total_length
        else:
            result[videoname] = 1.0

    return result


if __name__ == "__main__":
    opt = parse_opts()
    check_gt = False
    check_candidate = False
    if check_gt:
        out_path = os.path.join(opt.result_dir, 'results.json')
        eval(out_path,opt.gt_dir)
    elif check_candidate:
        if not os.path.exists('candidate_result.json'):
            result = candidate_eval(opt, opt.gt_dir)
            json.dump(result, open('candidate_result.json', 'w'))
        else:
            result = json.load(open('candidate_result.json'))
            print(result)
            total = 0
            for _, value in result.items():
                total += value
            print(total / len(result.keys()))
