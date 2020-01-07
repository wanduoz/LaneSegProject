import torch
import torch.nn as nn
from torchsummary import summary
from torch.nn import functional as F
import numpy as np
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

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


# 1. Entry flow
class EntryFlow(nn.Module):
    def __init__(self):
        super(EntryFlow, self).__init__()
        self.conv1 = BasicConv2d(3, 32, 3, 2)
        self.conv2 = BasicConv2d(32, 64, 3)
        self.downblock1 = SepBlock(64, 128, [3, 3, 3], [1, 1, 2], is_down=True)
        self.downblock2 = SepBlock(128, 256, [3, 3, 3], [1, 1, 2], is_down=True)
        self.downblock3 = SepBlock(256, 728, [3, 3, 3], [1, 1, 2], is_down=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.downblock1(x)
        short_cut = x  # 这个地方降了4倍
        x = self.downblock2(x)
        x = self.downblock3(x)
        return x, short_cut


# 2. Middle Flow
class MiddleFlow(nn.Module):
    def __init__(self, repeat=16):
        super(MiddleFlow, self).__init__()
        self.repeat = repeat
        self.layers = []
        for i in range(self.repeat):
            self.layers.append(SepBlock(728, 728, [3, 3, 3], [1, 1, 1], is_down=False))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


# 3. Exit Flow（用SepConvBlock）
class ExitFlow(nn.Module):
    def __init__(self, repeat=16):
        super(ExitFlow, self).__init__()
        self.sepconv1 = SepConvBlock(728, 728, k=3)
        self.sepconv2 = SepConvBlock(728, 1024, k=3)
        self.sepconv3 = SepConvBlock(1024, 1024, k=3, d=2)
        self.short_cut = SeperableConv2d(728, 1024, d=2)

        self.sepconv4 = SeperableConv2d(1024, 1536, k=3)
        self.sepconv5 = SeperableConv2d(1536, 1536, k=3)
        self.sepconv6 = SeperableConv2d(1536, 2048, k=3)

    def forward(self, x):
        identity = x
        identity = self.short_cut(identity)
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = self.sepconv3(x)
        x += identity
        x = self.sepconv4(x)
        x = self.sepconv5(x)
        x = self.sepconv6(x)
        return x

# Xceeption backbone
class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()
        self.entry_flow = EntryFlow()
        self.middle_flow = MiddleFlow()
        self.exit_flow = ExitFlow()

    def forward(self, x):
        x, shortcut = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        return x,shortcut


class ASPPPooling(nn.Sequential):
    def __init__(self, in_chans, out_chans):
        super(ASPPPooling, self).__init__(nn.AdaptiveAvgPool2d(1), BasicConv2d(in_chans, out_chans))

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


# encode层 ASPP out_chans参考paddle的数量
class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()

        modules = []
        out_chans = 256
        atrous_rates = [6, 12, 18]  # output stride=16时对应的ratio
        self.ConvBlock1 = BasicConv2d(2048, out_chans)

        # 3个不同ratio的膨胀卷积。分别设置d=3个rate
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.aspp1 = BasicConv2d(2048, out_chans, k=3, d=rate1)
        self.aspp2 = BasicConv2d(2048, out_chans, k=3, d=rate2)
        self.aspp3 = BasicConv2d(2048, out_chans, k=3, d=rate3)
        self.aspppooling = ASPPPooling(2048, out_chans)

        # concat后还有一个conv，bn，relu的块。
        self.project_1 = BasicConv2d(5 * out_chans, out_chans)
        self.project_2 = nn.Dropout(0.9)

    def forward(self, x):
        aspp0 = self.ConvBlock1(x)
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspppool = self.aspppooling(x)
        res = torch.cat([aspp0, aspp1, aspp2, aspp3, aspppool], dim=1)
        return self.project_1(res)

# decode层
class Decode(nn.Module):
    def __init__(self):
        super(Decode, self).__init__()

        # 把shortcut的chan从128变成48
        self.block1 = BasicConv2d(128, 48)
        # concat后的维度48+256。（256是aspp encode的输出）
        self.block2 = SepConvBlock(256 + 48, 256, 3)
        # 再进行一次sep conv
        self.block3 = SepConvBlock(256, 256, 3)

    def forward(self, x, shortcut):
        decode_shortcut = self.block1(shortcut)
        encode_data = F.interpolate(x, size=decode_shortcut.shape[2:], mode='bilinear', align_corners=False)
        encode_data = torch.cat([encode_data, decode_shortcut], dim=1)
        encode_data = self.block2(encode_data)
        encode_data = self.block3(encode_data)
        return encode_data

# 分类层
class Classifier(nn.Module):
    def __init__(self, num_class):
        super(Classifier, self).__init__()
        # decode输出维度256
        self.conv1 = conv2d(256, num_class)

    def forward(self, x, img):
        x = self.conv1(x)
        x = F.interpolate(x, size=img.shape[2:], mode='bilinear', align_corners=False)
        return x

# deeplabv3+
class deeplabv3p(nn.Module):
    def __init__(self, num_class):
        super(deeplabv3p, self).__init__()
        # xception backbone：输入channel=3，默认10个类别
        self.xception_backbone = Xception()
        # aspp：输入channel是2048，atrous_rates是[6,12,18]
        self.encode = ASPP()
        # decode
        self.decode = Decode()
        # clf
        self.clf = Classifier(num_class)

    def forward(self, x):
        out, shortcut = self.xception_backbone(x)
        out = self.encode(out)
        out = self.decode(out, shortcut)
        out = self.clf(out, x)
        return out

if __name__ == '__main__':
    model = deeplabv3p(8)
    summary(model, (3, 224, 224))