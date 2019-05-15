import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class multiloss(nn.Module):
    def __init__(self):
        super(multiloss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        teacher_pred, student_pred = predictions
        # print(teacher_pred, student_pred, targets)
        # loss_teacher = self.criterion(student_pred, teacher_pred)
        # loss_teacher = torch.norm(student_pred - teacher_pred, p=2)  # l2 loss
        _, pred = teacher_pred.topk(1, 1, True)
        pred = pred.t()
        pred = pred.squeeze(0).to(torch.long)
        # print(pred)
        loss_teacher = self.criterion(student_pred, pred)
        loss_student = self.criterion(student_pred, targets)
        # loss_teacher /= 2
        # loss_student /= 2

        loss = loss_teacher + loss_student
        return loss
