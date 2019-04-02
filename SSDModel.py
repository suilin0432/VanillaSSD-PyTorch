import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from data import voc, coco
import os

class SSD(nn.Module):
    """
    使用VGG Network作为backbone 然后在后面加上一些conv层
    然后每个 multibox 层包含了
        1)  conv2d for class conf scores
        2)  conv2d for localization predictions
        3)  生成 default bbox
    """
    def __init__(self, phase, size, base, extras, head, num_classes):
        """
        :param phase:  test / train 表明状态
        :param size: input 的 image size
        :param base: VGG16 layers
        :param extras: 额外天津爱的layers 用来生成multi-feature map的层
        :param head: mutlibox head 由 loc 和 conf 的 conv 层构成
        :param num_classes: 类别
        """
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # 如果 num_classes == 21 那么就是 (coco, voc)[True] == (coco, voc)[1] 选择的就是voc的配置
        self.cfg = (coco, voc)[num_classes == 21]
        # 设置先验框
        self.priorbox = PriorBox(self.cfg)
