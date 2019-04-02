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
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg["min_dim"]
        # 表示没有 cfg["variance"]定义的时候才使用 [0.1]否则使用设置的
        self.variance = cfg["variance"] or [0.1]
