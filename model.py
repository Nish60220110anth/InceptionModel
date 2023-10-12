# Google inception module

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard.writer import SummaryWriter
from torch.cuda import is_available

device = torch.device('cuda:0' if is_available() else 'cpu')

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(ConvBlock, self).__init__()

        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, **kwargs)
        # applied to output before activation
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1max):
        super(InceptionBlock, self).__init__()

        self.branch1 = ConvBlock(
            in_channels=in_channels, out_channels=out1x1, kernel_size=1, padding=0, stride=1)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels,
                      out_channels=red_3x3, kernel_size=1, padding=0, stride=1),
            ConvBlock(in_channels=red_3x3,
                      out_channels=out_3x3, kernel_size=3, padding=1, stride=1)
        )

        self.branch3 = nn. Sequential(
            ConvBlock(in_channels=in_channels,
                      out_channels=red_5x5, kernel_size=1, padding=0, stride=1),
            ConvBlock(in_channels=red_5x5,
                      out_channels=out_5x5, kernel_size=5, padding=2, stride=1)
        )

        self.branch4 = nn. Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels=in_channels,
                      out_channels=out_1x1max, kernel_size=1, padding=0, stride=1),
        )

    # N x filters x [image height x image width] | along filters dimension
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogLeNet, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.conv2 = ConvBlock(
            in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.inception3a = InceptionBlock(
            in_channels=192, out1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1max=32)
        self.inception3b = InceptionBlock(
            in_channels=256, out1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1max=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.inception4a = InceptionBlock(
            in_channels=480, out1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1max=64)
        self.inception4b = InceptionBlock(
            in_channels=512, out1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1max=64)
        self.inception4c = InceptionBlock(
            in_channels=512, out1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1max=64)
        self.inception4d = InceptionBlock(
            in_channels=512, out1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1max=64)
        self.inception4e = InceptionBlock(
            in_channels=528, out1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1max=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.inception5a = InceptionBlock(
            in_channels=832, out1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1max=128)
        self.inception5b = InceptionBlock(
            in_channels=832, out1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1max=128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, padding=0, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(in_features=1024, out_features=num_classes)

        # softmax not needed as it is included in the loss function

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x