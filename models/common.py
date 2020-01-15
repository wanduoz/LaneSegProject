import numpy as np
import torch.nn as nn

# 使用tensorflow的padding方式，免去每次手动计算pad的麻烦
def getpadding(k, s, p, d):
    if isinstance(p, int):
        return p
    if p == 'same':
        if s == 1:
            p = (k - 1) // 2
        if s > 1:
            p = int(np.ceil((k - s) / 2))
    if p == 'valid':
        p = 0
    if d != 1:
        p = d * (k // 2)
    return p

# 定义conv
def conv2d(in_chans, out_chans, k=1, s=1, p='same', d=1, g=1):
    p = getpadding(k, s, p, d)
    return nn.Conv2d(in_chans, out_chans, k, stride=s, padding=p, dilation=d, groups=g, bias=False)


# 1.定义relu-bn-conv的基础block（默认p是same，这样调用的时候就自动计算pad）
class BasicConv2d(nn.Module):
    def __init__(self, in_chans, out_chans, k=1, s=1, p='same', d=1, g=1):
        super(BasicConv2d, self).__init__()
        self.conv = conv2d(in_chans, out_chans, k, s, p, d, g)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# 2.建立可分离卷积，使用group conv=inchannel实现depth wise
class SeperableConv2d(nn.Module):
    def __init__(self, in_chans, out_chans, k=1, s=1, p='same', d=1):
        super(SeperableConv2d, self).__init__()
        self.depthwise = conv2d(in_chans, in_chans, k, s, p, d, g=in_chans)
        self.pointwise = conv2d(in_chans, out_chans, 1, 1, 0, d)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# 3.建立可分离卷积的block。包括 relu，可分离卷积，bn.
class SepConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, k=1, s=1, p='same', d=1):
        super(SepConvBlock, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.speconv = SeperableConv2d(in_chans, out_chans, k, s, p, d)
        self.bn = nn.BatchNorm2d(out_chans)

    def forward(self, x):
        return self.bn(self.speconv(self.relu(x)))


# 3个sepconv为一个大块。用于entry flow 与 middle flow。传入的k和s是3的list
class SepBlock(nn.Module):
    def __init__(self, in_chans, out_chans, k, s, is_down=True):
        super(SepBlock, self).__init__()
        self.is_down = is_down
        self.sep_conv1 = SepConvBlock(in_chans, out_chans, k[0], s[0])
        self.sep_conv2 = SepConvBlock(out_chans, out_chans, k[1], s[1])
        self.sep_conv3 = SepConvBlock(out_chans, out_chans, k[2], s[2])
        if self.is_down:
            self.down = SeperableConv2d(in_chans, out_chans, 1, 2)

    def forward(self, x):
        identy = x
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)
        if self.is_down:
            identy = self.down(identy)
        x += identy
        return x

