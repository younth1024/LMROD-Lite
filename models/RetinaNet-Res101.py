from typing import List, Any

import torch
import torch.nn as nn
import torchvision.models.detection as detection
import torchvision.models._utils as _utils

#混合知识蒸馏体系中的教师网络使用基于Resnet101的RetinaNet作为教师模型。
class RetinaNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(RetinaNetHead, self).__init__()
        # 分类头
        super().__init__()
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)
        # 回归头
        self.box_head = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x):
        cls_logits = self.cls_head(x)
        bboxes = self.box_head(x)
        return cls_logits, bboxes

#教师网络在蒸馏训练过程中不进行参数的更新。
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)
        if self.reduction == 'mean':
            return loss.mean()
        return loss

    # 使用torchvision提供的backbone，它已经包含了FPN结构
    def resnet_fpn_backbone(name='resnet101', pretrained=True):
        backbone = getattr(detection, name)(pretrained=pretrained)
        return backbone

    def get_retinanet_resnet101(pretrained=True, num_classes=80, num_anchors=9):
        # 获取预训练的ResNet101 FPN骨干网络
        backbone = resnet_fpn_backbone('resnet101', pretrained)
        # 定义RetinaNet的头部
        num_channels = backbone.out_channels
        head = RetinaNetHead(num_channels, num_anchors, num_classes)
        # 实例化Focal Loss
        loss_evaluator = FocalLoss()
        return backbone, head, loss_evaluator




#三重注意力模块
class TripletAttention(nn.Module):
    def __init__(self, embed_dim, attention_dim, dropout):
        super(TripletAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.attention = nn.MultiheadAttention(attention_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(attention_dim, embed_dim)

    def forward(self, anchor, positive, negative):
        # 假设anchor, positive, negative都是形状为(batch_size, seq_len, embed_dim)的张量
        # 将三个输入合并并传递给注意力机制
        combined = torch.cat([anchor, positive, negative], dim=1)

        # 使用多头注意力机制
        attn_output, _ = self.attention(combined, combined, combined)

        # 应用dropout和输出层
        attn_output = self.dropout(self.out_proj(attn_output))

        # 计算三重损失
        anchor_output = attn_output[:, :, :self.embed_dim]
        positive_output = attn_output[:, :, self.embed_dim:2 * self.embed_dim]
        negative_output = attn_output[:, :, 2 * self.embed_dim:]

        triplet_loss = self.triplet_loss(anchor_output, positive_output, negative_output)

        return attn_output, triplet_loss

    def triplet_loss(self, anchor, positive, negative, margin=0.2):
        # 计算锚点和正样本、锚点和负样本之间的距离
        pos_dist = (anchor - positive).pow(2).sum(dim=2)
        neg_dist = (anchor - negative).pow(2).sum(dim=2)

        # 计算三重损失
        loss = torch.clamp(neg_dist - pos_dist + margin, min=0.0)
        return loss.mean()


#级联三重注意力模块:在bifpn的基础上，加入三重注意力机制进行
class CTAModule(nn.Module):
    def __init__(self, features, embed_dim, attention_dim, dropout=0.1):
        super(BiFPNBlock, self).__init__()
        self.top_down = nn.ModuleList([nn.Conv2d(features, embed_dim, 1) for _ in range(len(features))])
        self.lateral = nn.ModuleList([nn.Conv2d(embed_dim, embed_dim, 1) for _ in range(len(features) - 1)])
        self.attention = TripletAttention(embed_dim, attention_dim, dropout)
        self.fusion = nn.ModuleList(
            [nn.Conv2d(2 * embed_dim, embed_dim, 3, padding=1) for _ in range(len(features) - 1)])

    def forward(self, features):
        laterals: List[Any] = []
        P = [self.top_down[i](features[i]) for i in range(len(features))]
        P = [F.relu(_) for _ in P]

        # Top-down pass
        for i in range(len(features) - 1, 0, -1):
            P[i - 1] = P[i - 1] + F.upsample(P[i], size=P[i - 1].size()[2:], mode='nearest')

        # Re-apply the 1x1 convolution
        for i in range(len(features) - 1):
            P[i] = self.lateral[i](P[i])

        # Attention and fusion
        for i in range(1, len(features)):
            P[i] = self.fusion[i - 1](torch.cat([P[i], P[i - 1]], dim=1))
            P[i] = P[i] * self.attention(P[i - 1])

        return P

# 对模型进行分支池化
class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x

# 上下文信息增强模块
class ContextInformationEnhancementModule(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ContextInformationEnhancementModule, self).__init__()
        #引入三重级联注意力模块
        self.channel_attention = ChannelAttention(channels, reduction_ratio)

    def forward(self, x):
        # 假设x是多尺度的特征图
        # 应用通道注意力机制
        out = self.channel_attention(x)
        return out