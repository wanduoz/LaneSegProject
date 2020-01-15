import torch
from torch import nn
from torchsummary import summary
import numpy as np

from .common import conv2d, BasicConv2d, SeperableConv2d, SepConvBlock, SepBlock


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []
        # conv -> relu -> bn
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))  # padding传入False或者int。当padding=False，int(padding)=0
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        # conv -> relu -> bn
        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

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