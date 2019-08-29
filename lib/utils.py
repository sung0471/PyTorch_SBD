import csv
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets, sample_duration):
    batch_size = targets.size(0)
    total_length = sample_duration

    n_iou_sum = None
    if targets.dim() > 1:
        loc_pred = outputs[0].clone().detach()
        loc_pred = decoding(loc_pred, total_length)
        loc_target = targets[:, :-1].clone().detach()
        # iou = list()
        # for i in range(batch_size):
        #     iou += [cal_iou(loc_pred[i], loc_target[i])]
        # iou = torch.Tensor(iou).to(torch.float)
        iou = cal_iou(loc_pred, loc_target)
        n_iou_sum = iou.sum().clone().detach().data

        outputs = outputs[1]
        targets = targets[:, -1].to(torch.long)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().clone().detach()

    out = dict()
    if n_iou_sum is not None:
        out['loc'] = n_iou_sum / batch_size
    out['conf'] = n_correct_elems / batch_size

    return out


def encoding(loc, total_length, default_bar=None):
    variances = [0.1, 0.2]

    if default_bar is None:
        center = (loc[:, 1] + loc[:, 0]) / 2
        center /= variances[0] * total_length
        center = center.view(-1, 1)

        length = loc[:, 1] - loc[:, 0]
        length = torch.log(length / total_length) / variances[1]
        length = length.view(-1, 1)
    else:
        pass

    return torch.cat([center, length], 1)


def decoding(loc, total_length, default_bar=None):
    variances = [0.1, 0.2]

    if default_bar is None:
        center = loc[:, 0] * variances[0] * total_length
        length = torch.exp(loc[:, 1] * variances[1]) * total_length

        start = center - length / 2
        start = start.view(-1, 1)

        end = center + length / 2
        end = end.view(-1, 1)
    else:
        pass

    return torch.cat([start, end], 1)


def cal_iou(loc_pred, loc_target):
    inter_start = torch.max(loc_pred[:, 0], loc_target[:, 0])
    inter_end = torch.min(loc_pred[:, 1], loc_target[:, 1])
    inter = torch.clamp((inter_end - inter_start), min=0)
    area_a = loc_pred[:, 1] - loc_pred[:, 0]
    area_b = loc_target[:, 1] - loc_target[:, 0]
    union = area_a + area_b - inter

    return inter / union

    # start1, end1 = set1
    # start2, end2 = set2
    # if start1 < start2 < end1:
    #     return (end1 - start2 + 1) / (end2 - start1 + 1)
    # elif start2 < start1 < end2:
    #     return (end2 - start1 + 1) / (end1 - start2 + 1)
    # else:
    #     return 0.0
