import torch
from torch import nn
from torchsummary import summary
from models.ResNetBackbone import ResNet
from models.common import conv2d, BasicConv2d, SeperableConv2d, SepConvBlock, SepBlock

# UNet两个conv的结构
class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UNetConvBlock, self).__init__()

        self.block = nn.ModuleList()
        self.block.append(BasicConv2d(in_chans, out_chans, k=3))
        self.block.append(BasicConv2d(out_chans, out_chans, k=3))

    def forward(self, x):
        for block in self.block:
            out = block(x)
        return out

# 双线性插值对小feature mao上采样。裁剪bridge，concat，进行UNet两个conv的结构
class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                nn.Conv2d(in_chans, out_chans, kernel_size=1))
        self.conv_block = UNetConvBlock(in_chans, out_chans)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out



class UNet(nn.Module):
    def __init__(self, name, num_class):
        super(UNet, self).__init__()
        self.encode = ResNet(name)
        prev_channels = 2048
        self.up_path = nn.ModuleList()
        for i in range(3):
            self.up_path.append(UNetUpBlock(prev_channels, prev_channels // 2))
            prev_channels //= 2

        self.cls_conv_block1 = BasicConv2d(prev_channels, 32 ,k=3)
        self.cls_conv_block2 = BasicConv2d(32, 16, k=3)
        self.last = conv2d(16, num_class, k=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.size()[2:]
        blocks = self.encode(x)
        x = blocks[-1]
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 2])
        x = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)(x)
        x = self.cls_conv_block1(x)
        x = self.cls_conv_block2(x)
        x = self.last(x)
        return x


if __name__ == '__main__':
    model = UNet('resnet18',8)
    summary(model, (3, 256, 256))