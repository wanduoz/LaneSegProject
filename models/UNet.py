import torch
from torch import nn
from torchsummary import summary
import numpy as np

from .common import conv2d, BasicConv2d, SeperableConv2d, SepConvBlock, SepBlock

class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UNetConvBlock, self).__init__()

        self.block = nn.ModuleList()

        # 一个convBlock包含2个conv-bn-relu块
        for i in range(2):
            self.block.append(BasicConv2d())

        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_chans))

        block.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_chans))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out





class UNet(nn.Module):
    def __init__(self, num_class):
        super(UNet, self).__init__()
        self.num_class = num_class


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    model = UNet(8)
    summary(model, (3, 224, 224))