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


def calculate_accuracy(outputs, targets, sample_duration, device):
    batch_size = targets.size(0)
    total_length = sample_duration - 1

    n_iou_sum = None
    if targets.dim() > 1:
        loc_out = outputs[0].clone().detach()
        if loc_out.dim() == 2:
            # loc_out = [batch_size, 2]
            # targets = [batch_size, 3]
            loc_pred = decoding(loc_out, total_length)
            loc_target = targets[:, :-1].clone().detach()
        else:
            # loc_out = [batch_size, default_bar_num, 2]
            # targets = [batch_size, 3]
            loc_pred = torch.zeros(batch_size, loc_out.size(1), loc_out.size(2)).to(device)
            for i in range(batch_size):
                loc_pred[i, :] = decoding(loc_out[i], total_length, default_bar=default_bar(sample_duration).to(device))
            loc_target = targets.unsqueeze(1)[:, :, :-1].expand(batch_size, loc_out.size(1), loc_out.size(2))

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


def get_coordinate(loc):
    # loc = number * [center, length]
    start = loc[:, 0] - loc[:, 1] / 2
    end = loc[:, 0] + loc[:, 1] / 2
    return torch.cat((start.view(-1, 1),    # start
                      end.view(-1, 1)), 1)  # end


def get_center_length(loc):
    # loc = number * [start, end]
    center = (loc[:, 1] + loc[:, 0]) / 2
    length = loc[:, 1] - loc[:, 0]
    return torch.cat((center.view(-1, 1),       # center
                      length.view(-1, 1)), 1)   # length


def encoding(loc, total_length, default_bar=None):
    variances = [0.1, 0.2]

    loc_data = get_center_length(loc)
    if default_bar is None:
        # loc = [batch_size, 2] : start, end
        # loc_data = [batch_size, 2] : center, length
        center = loc_data[:, 0] / (variances[0] * total_length)
        length = torch.log(loc_data[:, 1] / total_length) / variances[1]
    else:
        # loc = [default_bar_num, 2] : start, end
        # loc_data = [default_bar_num, 2] : center, length
        default = get_center_length(default_bar)

        center = (loc_data[:, 0] - default[:, 0]) / (variances[0] * default[:, 1])
        length = torch.log(loc_data[:, 1] / default[:, 1]) / variances[1]

    return torch.cat((center.view(-1, 1), length.view(-1, 1)), 1)


def decoding(loc, total_length, default_bar=None):
    variances = [0.1, 0.2]

    if default_bar is None:
        # loc = [batch_size, 2] : center, length
        center = loc[:, 0] * variances[0] * total_length
        length = torch.exp(loc[:, 1] * variances[1]) * total_length
    else:
        # loc = [default_bar_num, 2] : center, length
        default = get_center_length(default_bar)

        center = loc[:, 0] * variances[0] * default[:, 1] + default[:, 0]
        length = torch.exp(loc[:, 1] * variances[1]) * default[:, 1]

    center = center.view(-1, 1)
    length = length.view(-1, 1)

    new_loc = torch.cat((center, length), 1)

    return get_coordinate(new_loc)


def cal_iou(loc_a, loc_b, default_num=None):
    if default_num is None:
        inter_start = torch.max(loc_a[:, 0], loc_b[:, 0])
        inter_end = torch.min(loc_a[:, 1], loc_b[:, 1])
        inter = inter_end - inter_start + 1
        inter = torch.clamp(inter, min=0)
        area_a = loc_a[:, 1] - loc_a[:, 0] + 1
        area_b = loc_b[:, 1] - loc_b[:, 0] + 1
    else:
        truths = loc_a.view(-1, 1, 2)
        default = loc_b.view(1, -1, 2)
        A = truths.size(0)
        B = default.size(1)

        inter_start = torch.max(truths[:, :, 0].unsqueeze(2).expand(A, B, 1),   # [A, B, 1]
                                default[:, :, 0].unsqueeze(2).expand(A, B, 1))
        inter_end = torch.min(truths[:, :, 1].unsqueeze(2).expand(A, B, 1),     # [A, B, 1]
                              default[:, :, 1].unsqueeze(2).expand(A, B, 1))
        inter = inter_end - inter_start + 1         # [A, B, 1]
        inter = torch.clamp(inter, min=0)
        inter = inter.squeeze(inter.dim() - 1)      # [A, B]
        area_a = (truths[:, :, 1] - truths[:, :, 0] + 1).expand_as(inter)       # [A,B]
        area_b = (default[:, :, 1] - default[:, :, 0] + 1).expand_as(inter)     # [A,B]

    union = area_a + area_b - inter

    return inter / union


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (tensor): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def channel_list(sample_duration=16):
    assert sample_duration in [16, 32]

    channel_l = dict()
    channel_l[16] = [(2048, 512, 1024), (1024, 256, 512), (512, 128, 256)]
    channel_l[32] = [(2048, 512, 1024), (1024, 256, 512), (512, 128, 256), (256, 128, 256)]

    return channel_l[sample_duration]


def default_bar(sample_duration=16):
    assert sample_duration in [16, 32]
    default_bar_number_list = dict()
    default_bar_number_list[16] = [15, 7, 3, 1]
    default_bar_number_list[32] = [31, 15, 7, 3, 1]
    default_bar_list = dict()
    default_bar_list[16] = torch.zeros(26, 2)
    default_bar_list[32] = torch.zeros(57, 2)

    for key in default_bar_number_list.keys():
        count = 0
        length = 2
        for default_bar_number in default_bar_number_list[key]:
            for start in range(default_bar_number):
                default_bar_list[key][count][0] = start * (length / 2)
                default_bar_list[key][count][1] = start * (length / 2) + length - 1
                count += 1
            length *= 2

    return default_bar_list[sample_duration]


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(bars, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        bars: (tensor) The location preds for the img, Shape: [num_priors,2].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    # num : conf > 0.01인 default_bar의 갯수
    # bars = [num, 2] / scores = [num]
    # overlap = nms_threshold(0.45) / top_k = 5(default)
    keep = scores.new_zeros(scores.size(0)).to(torch.long)
    if bars.numel() == 0:   # number of elements
        return keep
    start = bars[:, 0]      # [num]
    end = bars[:, 1]        # [num]
    length = end - start + 1    # [num]
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bars of next highest vals
        ss = torch.index_select(start, 0, idx)   # [--num]
        ee = torch.index_select(end, 0, idx)     # [--num]
        # store element-wise max with next highest score
        ss = torch.clamp(ss, min=start[i].data)      # [--num]
        ee = torch.clamp(ee, max=end[i].data)        # [--num]
        l = ee - ss + 1     # [--num]
        # check length.. after each iteration
        inter = torch.clamp(l, min=0.0)         # [--num]
        # IoU = i / (area(a) + area(b) - i)
        rem_lengths = torch.index_select(length, 0, idx)  # [--num], load remaining lengths, not include top_1
        union = length[i] + rem_lengths - inter
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


if __name__ == '__main__':
    print(default_bar(16), default_bar(32))
