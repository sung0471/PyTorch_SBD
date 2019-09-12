import csv
import torch
import torch.nn as nn


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
    total_length = sample_duration

    iou_avg = None
    n_correct_avg = None
    if targets.dim() > 1:
        loc_out, conf_pred = outputs
        loc_target, conf_target = targets[:, :-1], targets[:, -1].to(torch.long)
        if loc_out.dim() == 2:
            # outputs = ([batch_size, 2], [batch_size, 3])
            # targets = [batch_size, 3]
            loc_pred = decoding(loc_out, total_length)

            # iou = list()
            # for i in range(batch_size):
            #     iou += [cal_iou(loc_pred[i], loc_target[i])]
            # iou = torch.Tensor(iou).to(torch.float)
            iou = cal_iou(loc_pred, loc_target)
            n_iou_sum = iou.sum().clone().detach().data
            iou_avg = n_iou_sum / batch_size

        else:
            # outputs = ([batch_size, default_bar_num, 2], [batch_size, default_bar_num, 3])
            # targets = [batch_size, 3]
            top_k = 5
            outputs = (outputs[0], nn.Softmax(dim=-1)(outputs[1]))
            output = detection(outputs, sample_duration, num_classes=3,
                               top_k=top_k, conf_thresh=0.01, nms_thresh=0.45)
            # output = [batch_size, num_classes, num_bars, [start, end, conf]]
            # pred_num = [batch_size, num_classes]
            num_classes = output.size(1)

            # output = [batch_size, num_classes, num_bars, [start, end, conf]]
            pred_num = torch.zeros(batch_size, num_classes, dtype=torch.int32)
            for batch_num in range(batch_size):
                for cls in range(num_classes):
                    i = 0
                    while output[batch_num, cls, i, -1] >= 0.6:
                        i += 1
                        if i == top_k:
                            break
                    pred_num[batch_num, cls] = i

            total_bars_num = pred_num.int().sum().clone().detach().data
            loc_pred = torch.zeros(total_bars_num, 2).to(device)
            conf_pred = torch.zeros(total_bars_num, 1, dtype=torch.long).to(device)

            iou_sum = torch.zeros(batch_size, 1).to(device)
            num_label_sum = torch.zeros(batch_size, 1).to(device)
            no_background_valid_bars_num = 0.0
            all_valid_bars_num = 0.0
            for batch_num in range(batch_size):
                bars_num_per_batch = 0
                for cls in range(num_classes):
                    result_num = pred_num[batch_num, cls].data
                    for i in range(result_num):
                        # if cls == 0:
                        #     loc_pred[num, :] = 0
                        # else:
                        loc_pred[bars_num_per_batch, :] = output[batch_num, cls, i, :-1]
                        conf_pred[bars_num_per_batch, :] = cls
                        bars_num_per_batch += 1
                if bars_num_per_batch == 0:
                    iou_sum[batch_num, 0] = 0.0
                    num_label_sum[batch_num] = 0
                    continue
                # cal_iou per batch : [bars_num_per_batch, 1]
                iou = cal_iou(
                    loc_pred[:bars_num_per_batch, :].data,
                    loc_target[batch_num].data,
                    use_default=True
                )
                no_background_idx = conf_pred[:bars_num_per_batch] > 0
                no_background_valid_bars_num += no_background_idx.sum().data
                no_background_iou = iou[no_background_idx]
                iou_sum[batch_num, 0] = no_background_iou.sum().clone().detach().data

                # cal_correct_label per batch
                all_valid_bars_num += bars_num_per_batch
                correct_idx = conf_pred[:bars_num_per_batch] == conf_target[batch_num]
                iou_select = iou[correct_idx] > 0
                num_label_sum[batch_num] = iou_select.sum().clone().detach().data
            iou_avg = iou_sum.float().sum().clone().detach().data / no_background_valid_bars_num
            n_correct_avg = num_label_sum.float().sum().clone().detach().data / all_valid_bars_num
            # iou_avg, n_correct_avg == NaN, 0 할당
            if iou_avg != iou_avg:
                iou_avg = 0.0
            if n_correct_avg != n_correct_avg:
                n_correct_avg = 0.0
    else:
        conf_pred = outputs
        conf_target = targets[:, -1].to(torch.long)

    out = dict()
    if iou_avg is not None:
        out['loc'] = iou_avg

    if n_correct_avg is None:
        _, pred = conf_pred.topk(1, 1, True)
        pred = pred.t()
        correct = pred.eq(conf_target.view(1, -1))
        n_correct_elems = correct.float().sum().clone().detach()

        out['conf'] = n_correct_elems / batch_size
    else:
        out['conf'] = n_correct_avg

    return out


def get_center_length(loc):
    # loc = number * [start, end]
    center = (loc[:, 1] + loc[:, 0]) / 2
    length = loc[:, 1] - loc[:, 0] + 1
    return torch.cat((center.view(-1, 1),       # center
                      length.view(-1, 1)), 1)   # length


def get_coordinate(loc):
    # loc = number * [center, length]
    start = loc[:, 0] - (loc[:, 1] - 1) / 2
    end = loc[:, 0] + (loc[:, 1] - 1) / 2
    return torch.cat((start.view(-1, 1),    # start
                      end.view(-1, 1)), 1)  # end


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


def cal_iou(loc_a, loc_b, use_default=False):
    if not use_default:
        A = loc_a.size(0)
        B = loc_b.size(1)
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

    return (inter / union).view(A, B)


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


def detection(out, sample_duration, num_classes, top_k, conf_thresh, nms_thresh):
    loc, conf = out
    # frame_pos = torch.zeros(loc.size(0), loc.size(1), loc.size(2))
    # labels = torch.zeros(conf.size(0), conf.size(1), 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    default = default_bar(sample_duration=sample_duration).to(device)
    batch_size = loc.size(0)

    output = torch.zeros(batch_size, num_classes, top_k, 3).to(device)
    conf_pred = conf.transpose(2, 1)
    for i in range(batch_size):
        total_length = sample_duration

        # assert loc.size(1) == self.default_bar.size(0)
        # frame_pos[i, :] = decoding(loc[i, :], total_length, default_bar=self.default_bar)
        # frame_pos[i, :] += start_boundaries[i]                  # [default_num, 2]
        # label = conf[i, :]
        # labels[i, :] = torch.argmax(label, dim=1).view(-1, 1)   # [default_num, 1]

        decoded_bars = decoding(loc[i, :], total_length, default_bar=default)  # [default_num, 2]
        conf_scores = conf_pred[i].clone().detach()  # [3, default_num]

        for cl in range(num_classes):
            c_mask = conf_scores[cl].gt(conf_thresh)  # [default_num]
            # for i in default_num,
            # if conf[i] > conf_thresh, num += 1
            scores = conf_scores[cl][c_mask]  # [num]
            if scores.size(0) == 0:
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_bars)  # [default_num, 2]
            bars = decoded_bars[l_mask].view(-1, 2)  # [num, 2]

            if cl == 0:
                v, idx = scores.sort(0, descending=True)  # sort in descending order
                total_num = idx.size(0) if idx.size(0) < top_k else top_k
                idx = idx[:total_num]
                output[i, cl, :total_num] = \
                    torch.cat((bars[idx],
                               scores[idx].unsqueeze(1)), 1)  # [top_k, 3]
            else:
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(bars, scores, nms_thresh, top_k)
                output[i, cl, :count] = \
                    torch.cat((bars[ids[:count]],
                               scores[ids[:count]].unsqueeze(1)), 1)  # [count, 3]

    # [batch_size, all_result_bars_num, 3]
    # all_result_bars_num : 클래스 별 bar의 갯수(최대 top_k * num_classes)
    # 3 : start, end, conf
    flt = output.contiguous().view(batch_size, -1, 3)
    _, idx = flt[:, :, -1].sort(1, descending=True)  # [batch_size, bars_num]
    _, rank = idx.sort(1)  # [batch_size, bars_num], 각 idx 의 등수
    flt[(rank < top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

    return output


if __name__ == '__main__':
    print(default_bar(16), default_bar(32))
