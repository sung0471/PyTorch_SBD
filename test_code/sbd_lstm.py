import torch
import torch.nn as nn


class SBD_LSTM(nn.Module):
    def __init__(self, in_dim=1024, hid_dim=512, num_layers=3):
        super(SBD_LSTM, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.rnn = nn.LSTM(self.in_dim, self.hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(self.hid_dim*2, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.squeeze(0)[-3:] #마지막 3개 선택
        p = torch.softmax(self.fc1(x.unsqueeze(0)))
        return p #마지막 3개의 probability
