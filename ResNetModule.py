import torch
import torch.nn as nn

def conv3(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = conv3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if(in_channels != out_channels):
            self.shortcut = nn.Sequential(
               nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm1d(out_channels))

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResidualLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, downsample=None):
        super(ResidualLSTM, self).__init__()
        self.expansion = expansion
        # self.downsample = downsample
        self.lstm = nn.LSTM(in_channels, out_channels//2, 1, batch_first=False, bidirectional=True)
        self.bn = nn.BatchNorm1d(out_channels)
        self.tanh = nn.Tanh()
        self.shortcut = nn.Sequential()
        self.hid = []
        if(in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        residual = self.shortcut(x.permute(1, 2, 0))
        temp1, self.hid = self.lstm(x)
        # if self.downsample:
        #     residual = self.downsample(x)
        temp2 = temp1.permute(1, 2, 0)
        temp2 = self.bn(temp2)
        temp2 = temp2.permute(2, 0, 1)
        out = temp2 + residual.permute(2, 0, 1)
        out = self.tanh(out)
        return out, self.hid