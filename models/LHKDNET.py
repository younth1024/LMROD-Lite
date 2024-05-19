import torch
import torch.nn as nn
import torch.nn.functional as F


#这里定义使用基于Resnet18作为骨干网络的学生模型。

class LHKDNET(nn.Module):

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
        # swish函数
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)
    def extract_features(self, inputs):
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


def get_model_name() -> str:
    return "LHKDNET"


#增强前景区域的特征蒸馏模块对应的损失函数EFR-FKD
class EFR_FKD(nn.Module):
    def __init__(self, alpha=1.0):
        super(EFR_FKD, self).__init__()
        self.alpha = alpha  # 用于平衡不同损失项的权重

    def forward(self, student_features, teacher_features):
        # 假设 student_features 和 teacher_features 都是列表形式的特征，
        # 其中每个元素对应于网络中的一个特征层的输出

        # 对每个特征层的输出计算均值中心化特征
        student_features_centered = [F.normalize(feature) for feature in student_features]
        teacher_features_centered = [F.normalize(feature) for feature in teacher_features]

        # 计算均值中心化特征之间的交互损失
        mcf_loss = 0
        for sf, tf in zip(student_features_centered, teacher_features_centered):
            # 这里使用简单的L2损失作为交互损失的示例
            mcf_loss += torch.norm(sf - tf, dim=-1).mean()

        # EFR_FKD对应的损失
        mcf_loss = mcf_loss.mean() * self.alpha

        return mcf_loss

#自适应可变温度的 logits 蒸馏损失函数
class AVT_KD(nn.Module):
    def __init__(self, temperature=1.0, alpha=1.0, beta=1.0):
        super(AVT_KD, self).__init__()
        self.temperature = temperature
        #指定自适应的温度系数
        self.alpha = alpha
        self.beta = beta  # 平衡任务损失和蒸馏损失的权重

    def forward(self, student_logits, teacher_logits, labels):
        # 任务损失,交叉熵损失
        task_loss = F.cross_entropy(student_logits, labels)
        # 蒸馏损失，使用KL散度
        tau = self.temperature
        student_probs = F.softmax(student_logits / tau, dim=1)
        teacher_probs = F.softmax(teacher_logits / tau, dim=1)
        distillation_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (tau ** 2)
        # 总损失
        total_loss = self.alpha * task_loss + self.beta * distillation_loss
        return total_loss

