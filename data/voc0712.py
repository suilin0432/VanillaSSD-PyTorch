# 这个文件时VOC数据集的相关信息以及VOC数据集操作的DataLoader的相关信息

import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# 共20个类别 然后 0-index 默认的类别是 背景 所以在取标签的时候应该进行 -1 操作
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# VOC数据集的根目录
VOC_ROOT = "./data/VOCdevkit"

class VOCAnnotationTransform(object):
    """
        将VOC的注解标签转换成bbox 和 label index的Tensor
    """
    def __init__(self, class_to_ind=None, keep_difficult=False):
        """

        :param class_to_ind: (dict, optional) 类别标签, 默认是 VOC 的20种 格式是 {标签: index, ...}
        :param keep_difficult: (bool, optional) 是否保留困难的instance
        """
        self.class_to_ind = class_to_ind or dict(zip(
            VOC_CLASSES, range(len(VOC_CLASSES))
        ))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """

        :param target: (annotation) 需要进行解析的annotation
        :return: 返回list 包含 [bbox coordination, class name]
        """
        res = []
        for obj in target.iter("object"):
            pass