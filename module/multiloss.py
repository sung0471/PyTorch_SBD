import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class multiloss(nn.Module):
    def __init__(self, loss_type='new'):
        assert loss_type in ['origin', 'new', 'dual']
        super(multiloss, self).__init__()

        self.loss_type = loss_type
        if self.loss_type in ['origin', 'dual']:
            self.KLDiv = nn.KLDivLoss()
        self.CrossEnt = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        teacher_pred, student_pred = predictions

        if self.loss_type == 'origin':
            alpha = 0.5
            loss_teacher = self.KLDiv(F.log_softmax(student_pred, dim=1), F.softmax(teacher_pred, dim=1)) * alpha
            loss_student = self.CrossEnt(student_pred, targets) * (1. - alpha)
        elif self.loss_type == 'new':
            # print(teacher_pred, student_pred, targets)
            # loss_teacher = self.criterion(student_pred, teacher_pred)
            # loss_teacher = torch.norm(student_pred - teacher_pred, p=2)  # l2 loss
            _, pred = teacher_pred.topk(1, 1, True)
            pred = pred.t()
            pred = pred.squeeze(0).to(torch.long)
            # print(pred)
            loss_teacher = self.CrossEnt(student_pred, pred)
            loss_student = self.CrossEnt(student_pred, targets)
            # loss_teacher /= 2
            # loss_student /= 2
        elif self.loss_type == 'dual':
            # alpha = 0.5
            _, pred = teacher_pred.topk(1, 1, True)
            pred = pred.t()
            pred = pred.squeeze(0).to(torch.long)
            loss_teacher = self.KLDiv(F.log_softmax(student_pred, dim=1), F.softmax(teacher_pred, dim=1)) + \
                           self.CrossEnt(student_pred, pred)
            loss_student = self.CrossEnt(student_pred, targets)
        else:
            assert False

        loss = loss_teacher + loss_student
        return loss
