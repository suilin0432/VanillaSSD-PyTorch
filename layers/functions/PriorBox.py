from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch

# itertools 是产生用来迭代的工具, 具体用法查手册或者搜索吧
# itertools.product(A, B)用法 产生 A 与 B 的笛卡尔积
# eg: itertools.product([1,2],[3,4]) -> for 迭代之后产生的是(1,3), (1,4), (2,3), (2,4)

class PriorBox(object):
    """
    用来计算每个特征图的左上角坐标(变换偏移坐标)
    """
    # PS: [3, 2] + [1, 2] = [3, 2, 1, 2]
    mean = []
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg["min_dim"]
        # 表示没有 cfg["variance"]定义的时候才使用 [0.1]否则使用设置的
        self.variance = cfg["variance"] or [0.1]
        self.feature_maps = cfg["feature_maps"]
        self.min_sizes = cfg["min_sizes"]
        self.max_sizes = cfg["max_sizes"]
        self.steps = cfg["steps"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.clip = cfg["clip"]
        self.version = cfg["name"]
        for v in self.variance:
            if v <= 0:
                raise ValueError("Variances must be greater than 0")

    def forward(self):
        mean = []
        # 对每个feature map的尺寸进行遍历生成 default box
        for k, f in enumerate(self.feature_maps):
            # 对每个feature的网格的节点进行遍历
            for i, j in product(range(f), repeat=2):
                # 获取 stride
                # TODO: 为什么不用 self.image_size 表示步长呢 反而用设置的步长进行表示虽然说差不太多
                f_k = self.image_size / self.steps[k]
                # 计算中心 cx, cy
                # PS: 这里 无论 cx, cy 还是w, h都是相对于当前featureMap 的尺寸进行的
                cx, cy = (j + 0.5) / f_k, (i + 0.5) / f_k
                # aspect_ratio = 1
                # size = min_size 的 default bbox
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio = 1
                # size = sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 针对其他的aspect_ratio
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # 设置为Tensor格式然后设置为一个个的anchor
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max = 1, min = 0)
        return output