from models.deepSBD import *
from opts import parse_opts
from cls import build_model
import os


class teacher_student_net(nn.Module):
    def __init__(self, opt, device):
        self.phase = opt.phase
        # self.device = device
        super(teacher_student_net, self).__init__()

        # 19.6.26.
        # opt.model = '' > 주석처리
        # opt.teacher_model과 opt.model을 parameter로 삽입

        # opt.model = 'alexnet'
        self.teacher_model = build_model(opt, opt.teacher_model, 'test', device)
        self.load_checkpoint(self.teacher_model, opt.teacher_model_path)

        # opt.model = 'resnext'
        self.student_model = build_model(opt, opt.model, self.phase, device)

    def forward(self, x):
        # x = x.to(self.device)
        # x = x.cuda()
        if self.phase == 'train':
            teacher_x = self.teacher_model(x)
            student_x = self.student_model(x)
            out = (teacher_x, student_x)
        else:
            student_x = self.student_model(x)
            out = student_x

        return out

    def load_checkpoint(self, model, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)


if __name__ == '__main__':
    opt = parse_opts()
    opt.pretrain_path = '../kinetics_pretrained_model/resnext-101-kinetics.pth'
    teacher_model_path = 'Alexnet-final.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = teacher_student_net(opt, teacher_model_path, device)
    print(model)
