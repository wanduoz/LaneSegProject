from torch import nn
from models.common import conv2d, BasicConv2d, SeperableConv2d, SepConvBlock, SepBlock

# 定义resnet18，resnet50用到的basic block。考虑靠这里第二个relu是在shortcut后，所以不能直接调用basicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chans, out_chans, stride=1, downsample=None):
        # downsample维度一致才能操作。深度一致，长款一致。
        super(BasicBlock, self).__init__()
        # 1
        self.conv3x3block = BasicConv2d(in_chans, out_chans, k=3, s=stride) # , p='same'算出来就是p=1
        # 2
        self.conv2 = conv2d(out_chans, out_chans, k=3)
        self.bn2 = nn.BatchNorm2d(out_chans)
        # x downsample
        self.downsample = downsample
        #
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        # 1
        out = self.conv3x3block(x)
        # 2
        out = self.bn2(self.conv2(out))
        # downsample x and add x
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# 定义resnet101，resnet152用到block
class BottleBlock(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans, stride=1, downsample=None):
        super(BottleBlock, self).__init__()
        # 1
        self.conv1x1Block= BasicConv2d(in_chans, out_chans) # k=1, s=1, pad=same
        # 2
        self.conv3x3Block= BasicConv2d(out_chans,out_chans,k=3,s=stride) # k=3, s=stride, padding=same自动计算
        # 3
        self.conv3 = conv2d(out_chans, self.expansion * out_chans) # 1x1, k=1, s=1, pad=same
        self.bn3 = nn.BatchNorm2d(out_chans * self.expansion)
        #
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # 1
        out = self.conv1x1Block(x)
        # 2
        out = self.conv3x3Block(out)
        # 3
        out = self.conv3(out)
        out = self.bn3(out)
        #
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, name):
        super(ResNet, self).__init__()

        if name == 'resnet18':
            block, layers = BasicBlock, [2,2,2,2]
        elif name == 'resnet34':
            block, layers = BasicBlock, [3,4,6,3]
        elif name == 'resnet50':
            block, layers = BottleBlock, [3,4,6,3]
        elif name == 'resnet101':
            block, layers = BottleBlock, [3,4,23,3]
        elif name == 'resnet152':
            block, layers = BottleBlock, [3,8,36,3]
        else:
            raise Exception("unknown resnet structure")

        self.out_chans = 64
        # 初，降2
        self.convBlock1 = BasicConv2d(3, self.out_chans, k=7, s=2, p='same') # k=7, stride=2, pad算出来=3
        # pooling，降4
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,  padding=1)
        # 1，不降，stride=1
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 2，降8
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 3，降16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 4，降32
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    # block：basicblock/BottleBlock，
    def _make_layer(self, block, out_chans, blocks, stride=1):
        downsample = None

        # stride!=1长宽不同，第二个深度不同
        if stride != 1 or self.out_chans != out_chans * block.expansion:
            downsample = nn.Sequential(
                conv2d(self.out_chans, out_chans * block.expansion,k=1,s=stride,p='valid'), # 这个地方不padding，选择valid
                nn.BatchNorm2d(out_chans * block.expansion)
            )

        layers = []
        layers.append(block(self.out_chans, out_chans, stride, downsample))
        self.out_chans = out_chans * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.out_chans, out_chans))
        return nn.Sequential(*layers)

    def forward(self, x):
        f1 = self.convBlock1(x) # 降2
        f2 = self.maxpool(f1) # 降4

        f3 = self.layer1(f2) # 降4
        f4= self.layer2(f3) # 降8
        f5 = self.layer3(f4) # 降16
        f6 = self.layer4(f5) # 降32

        return [f3, f4, f5, f6]