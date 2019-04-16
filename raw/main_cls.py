from torch import nn
from torch import optim

from opts import parse_opts
from lib.spatial_transforms import *

from data.train_data_loader import DataSet
from raw.cls import build_model
import time
import os

from lib.utils import AverageMeter, calculate_accuracy
from torch.autograd import Variable
from torch.optim import lr_scheduler
from raw.test_cls import test

def get_mean(norm_value=255):
    return [114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value]

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data

    return n_correct_elems / batch_size
    
# 19.3.8 revision
# add parameter : "device"
def train(cur_iter, total_iter,data_loader, model, criterion, optimizer,scheduler, opt, device):
    model.eval()

    # 19.3.14. add
    print("device : ",torch.cuda.get_device_name(0))
    # torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    model.to(device)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    i=cur_iter

    # for debug
    # print(not(opt.no_cuda)) : True
    print('\n====> Training Start')
    while i<total_iter:
        for _,(inputs,targets) in enumerate(data_loader):

            # 19.3.7 add
            # if not opt.no_cuda:
            #     targets = targets.cuda(async=True)
            #     inputs = inputs.cuda(async=True)

            targets = Variable(targets)
            inputs = Variable(inputs)

            # 19.3.8. revision
            if not opt.no_cuda:
                targets= targets.to(device)
                inputs= inputs.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            
            acc=calculate_accuracy(outputs,targets)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.data)

            print('Iter:{} Loss_conf:{} acc:{} lr:{}'.format(i+1,loss.data,acc,optimizer.param_groups[0]['lr']),flush=True)
            i+=1
                
            if i%2000==0:
                save_file_path = os.path.join(opt.result_dir, 'model_iter{}.pth'.format(i))
                print("save to {}".format(save_file_path))
                states = {
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }
                torch.save(states, save_file_path)
            if i>=total_iter:
                break

    save_file_path = os.path.join(opt.result_dir, 'model_final.pth'.format(opt.checkpoint_path))
    print("save to {}".format(save_file_path))
    states = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }
    torch.save(states, save_file_path)

def get_lastest_model(opt):
    if opt.resume_path!='':
        return 0
    if os.path.exists(os.path.join(opt.result_dir,'model_final.pth')):
        opt.resume_path=os.path.join(opt.result_dir,'model_final.pth')
        return opt.total_iter
    
    iter_num=-1
    for filename in os.listdir(opt.result_dir):
        if filename[-3:]=='pth':
            _iter_num=int(filename[len('model_iter'):-4])
            iter_num=max(iter_num,_iter_num)
    if iter_num>0:
        opt.resume_path=os.path.join(opt.result_dir,'model_iter{}.pth'.format(iter_num))
    return iter_num

if __name__ == '__main__':
    opt = parse_opts()

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)

    opt.mean = get_mean(opt.norm_value)
    print(opt)

    torch.manual_seed(opt.manual_seed)

    # 19.3.8. add
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("cuda is available : ",torch.cuda.is_available())

    # model = build_model(opt,"train")
    model=build_model(opt,"train",device)

    cur_iter=0
    if opt.auto_resume and opt.resume_path=='':
        cur_iter=get_lastest_model(opt)
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)

        model.load_state_dict(checkpoint['state_dict'])

    parameters = model.parameters()
    criterion = nn.CrossEntropyLoss()
    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.momentum

    optimizer = optim.SGD(parameters, lr=opt.learning_rate,
                              momentum=opt.momentum, dampening=dampening,
                              weight_decay=opt.weight_decay, nesterov=opt.nesterov)

    # 19.3.8 revision
    if not opt.no_cuda:
        # criterion = criterion.cuda()
        criterion=criterion.to(device)

    if not opt.no_train:
        spatial_transform = get_train_spatial_transform(opt)
        temporal_transform = None
        target_transform = None
        # list_root_path : train path, only_gradual path
        # `19.3.7 : add only_gradual path
        list_root_path=[]
        list_root_path.append(os.path.join(opt.root_dir,opt.train_subdir))
        list_root_path.append(os.path.join(opt.root_dir,'only_gradual'))
        print(list_root_path)
        print(opt.image_list_path)
        training_data = DataSet(list_root_path,opt.image_list_path,
                                        spatial_transform=spatial_transform,
                                        temporal_transform=temporal_transform,
                                        target_transform=target_transform, sample_duration=opt.sample_duration)

        weights = torch.DoubleTensor(training_data.weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=opt.batch_size,
                                                            num_workers=opt.n_threads,sampler=sampler, pin_memory=True)


        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=60000)

        # 19.3.8. add
        # train(cur_iter,opt.total_iter,training_data_loader, model, criterion, optimizer,scheduler,opt)
        train(cur_iter,opt.total_iter,training_data_loader, model, criterion, optimizer,scheduler,opt, device)
        test(opt,model,device)

    # 19.3.21 add
    # add else:
    # train, test부분 분리
    else:
        test(opt,model,device)
