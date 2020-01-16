import torch
import torch.nn as nn
from torchsummary import summary
from torch.nn import functional as F
from models.common import conv2d, BasicConv2d, SeperableConv2d, SepConvBlock, SepBlock

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

# 池化，线性插值
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out, shortcut = self.xception_backbone(x)
        out = self.encode(out)
        out = self.decode(out, shortcut)
        out = self.clf(out, x)
        return out

if __name__ == '__main__':
    model = deeplabv3p(8)
    summary(model, (3, 224, 224))