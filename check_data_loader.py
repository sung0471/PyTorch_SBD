import os
from opts import parse_opts
from data.train_data_loader import DataSet
from lib.spatial_transforms import *
import torch

opt = parse_opts()

opt.scales = [opt.initial_scale]
for i in range(1, opt.n_scales):
    opt.scales.append(opt.scales[-1] * opt.scale_step)

opt.mean = get_mean(opt.norm_value)
print(opt)

torch.manual_seed(opt.manual_seed)

spatial_transform = get_train_spatial_transform(opt)
temporal_transform = None
target_transform = None
list_root_path = list()
list_root_path.append(os.path.join(opt.root_dir, opt.train_subdir))
list_root_path.append(os.path.join(opt.root_dir, opt.only_gradual_subdir))
print(list_root_path, flush=True)
print("[INFO] reading : ", opt.video_list_path, flush=True)
training_data = DataSet(list_root_path, opt.video_list_path, opt,
                        spatial_transform=spatial_transform,
                        temporal_transform=temporal_transform,
                        target_transform=target_transform, sample_duration=opt.sample_duration)
weights = torch.DoubleTensor(training_data.weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=opt.batch_size, sampler=sampler)

if opt.iter_per_epoch == 0:
    if opt.is_full_data:
        opt.iter_per_epoch = 500000
    else:
        opt.iter_per_epoch = 70000

    opt.iter_per_epoch = int(opt.iter_per_epoch / opt.batch_size)
print("iter_per_epoch : {}, batch_size : {}".format(opt.iter_per_epoch, opt.batch_size))

iter_count = list([0, 0, 0])
i = 0
print("start_check")
for _, (data, labels) in enumerate(training_data_loader):
    if i % 1000 == 0:
        print(i)
    if i == opt.iter_per_epoch or i==1000:
        break
    for label in labels:
        iter_count[int(label)] += 1
        i += 1

print(iter_count)
