import torch
import torch.nn as nn
import torch.nn.init as init

# PS:这和一个不带 bias 的 BatchNorm2d 有什么区别吗
class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        # 防止除0的一个小数
        self.eps = 1e-10
        # 网络在训练中需要更新的参数定义为 Parameter
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        # 将self.weight 用 self.gamma进行填充
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        # x的输入是 batch_sizez, channel, height, width #是将通道上的所有数字进行相加了
        norm = x.pow(2).sum(dim = 1, keepdim = True).sqrt() + self.eps
        # 进行归一化
        x = torch.div(x, norm)
        # 把weight 从 一个 一维的 n_channels 的参数 扩展到 1, n_channels, 1, 1 然后在将n_channels 放大到和x相同的体积上
        # PS:貌似直接用weight * x效果一样
        # 这里按照默认是初始乘以一个20
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out