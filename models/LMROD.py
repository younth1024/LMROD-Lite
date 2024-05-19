import torch
import torch.nn as nn
import torch.nn.functional as F


#此处的学生模型仍使用基于Retinanet检测框架为基准的模型，在其中加入上下文信息增强模块和级联三重注意力模块。

def get_model_name() -> str:
    return "LMROD"


class Retinanet_1(nn.Module):

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # 获得标准化的参数
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        # -------------------------------------------------#
        #   网络主干部分开始
        #   设定输入进来的是RGB三通道图像
        #   利用round_filters可以使得通道可以被8整除
        # -------------------------------------------------#
        in_channels = 3
        out_channels = round_filters(32, self._global_params)
        # -------------------------------------------------#
        #   创建stem部分
        # -------------------------------------------------#
        self._conv_stem = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._blocks = nn.ModuleList([])
        for i in range(len(self._blocks_args)):
            self._blocks_args[i] = self._blocks_args[i]._replace(
                input_filters=round_filters(self._blocks_args[i].input_filters, self._global_params),
                output_filters=round_filters(self._blocks_args[i].output_filters, self._global_params),
                num_repeat=round_repeats(self._blocks_args[i].num_repeat, self._global_params)
            )
            # -------------------------------------------------------------#
            #   都需要考虑步长和输入通道数
            # -------------------------------------------------------------#
            self._blocks.append(MBConvBlock(self._blocks_args[i], self._global_params))
            if self._blocks_args[i].num_repeat > 1:
                self._blocks_args[i] = self._blocks_args[i]._replace(input_filters=self._blocks_args[i].output_filters,
                                                                     stride=1)
            for _ in range(self._blocks_args[i].num_repeat - 1):
                self._blocks.append(MBConvBlock(self._blocks_args[i], self._global_params))

        in_channels = self._blocks_args[len(self._blocks_args) - 1].output_filters
        out_channels = round_filters(1280, self._global_params)

        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        # swish函数
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, load_weights=True, advprop=True, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        if load_weights:
            load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b' + str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


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


#对模型进行分支池化
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


#上下文信息增强模块
class ContextInformationEnhancementModule(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ContextInformationEnhancementModule, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)

    def forward(self, x):
        # 假设x是多尺度的特征图
        # 应用通道注意力机制
        out = self.channel_attention(x)
        return out
