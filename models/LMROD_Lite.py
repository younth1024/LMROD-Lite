import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def get_model_name() -> str:
    return "LMROD_lite"


#定义深度可分离卷积BLOCK用于构建骨干网络
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


#使用深度可分离卷积代替普通卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # 使用深度可分离卷积
        self.conv1 = DepthwiseSeparableConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


#构建新的骨干网络
class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


#定义基于自适应多层掩码模块的稀疏卷积模块
class AdaptiveSparseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=False, activation=False):
        super(AdaptiveSparseConvBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.activation = nn.ReLU() if activation else None

    def generate_adaptive_mask(input_feature):
        # 该函数用于根据输入特征图生成自适应掩码
        # 掩码应该是一个二进制矩阵，指示了输入中的活跃区域
        # 使用输入的绝对值大于某个阈值的区域作为活跃区域
        threshold = 0.1
        mask = torch.abs(input_feature) > threshold
        return mask

    def forward(self, input_feature):
        # 生成自适应掩码
        mask = generate_adaptive_mask(input_feature)
        # 应用掩码到输入特征图
        input_feature = input_feature * mask.float()
        # 深度卷积
        depthwise = self.depthwise_conv(input_feature)
        # 逐点卷积
        pointwise = self.pointwise_conv(depthwise)

        # 应用批量归一化和激活函数（如果需要）
        if self.norm is not None:
            pointwise = self.norm(pointwise)
        if self.activation is not None:
            pointwise = self.activation(pointwise)
        return pointwise


#检测网络中的回归检测头，在其中引入稀疏卷积模块
class BoxNetWithSparseConv(nn.Module):

    def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False):
        super(BoxNetWithSparseConv, self).__init__()
        # 使用自适应稀疏卷积块替换原有的分离卷积块
        self.conv_list = nn.ModuleList(
            [AdaptiveSparseConvBlock(in_channels, in_channels, norm=False, activation=False) for _ in range(num_layers)]
        )

    def forward(self, inputs):
        feats = []
        # 对每个特征层循环
        for feat, bn_list in zip(inputs, self.bn_list):
            # 每个特征层需要进行num_layer次卷积+标准化+激活函数
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)
        # 进行一个堆叠
        feats = torch.cat(feats, dim=1)

        return feats


#检测模型中的分类检测头
class ClassNetWithSparseConv(nn.Module):

    def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False):
        super(ClassNetWithSparseConv, self).__init__()
        self.conv_list = nn.ModuleList(
            [AdaptiveSparseConvBlock(in_channels, in_channels, norm=False, activation=False) for _ in range(num_layers)]
        )

    def forward(self, inputs):
        feats = []
        # 对每个特征层循环
        for feat, bn_list in zip(inputs, self.bn_list):
            # 每个特征层需要进行num_layer次卷积+标准化+激活函数
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)
        # 进行一个堆叠
        feats = torch.cat(feats, dim=1)

        return feats
