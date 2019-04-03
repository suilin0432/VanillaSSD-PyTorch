# 从配置文件中引入 SSD的相关设置
from .config import *
# from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT
# from .coco import COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT, get_label_map
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """
    自定义的用来处理包含不同个数的obj标签的函数(PS: 就是我上一次写YoloV3的时候
    遇到的问题: pytorch自带的DataLoader要求必须所有返回的维度是相同的)

    :param batch: (tuple) a tuple of tensor image and lists of annotations
    :return: (tuple) contains:
                1.  (tensor) batch of images stacked on their 0 dim
                2.  (list of tensors) annotations for a given image are stacked on 0 dim
    PS: torch.stack(tuple, dim) 要求第一个必须是包含一组Tensor的tuple, 然后作用是按照指定的维度进行stack
    """
    # targets 和 imgs 分别表示GT标签和图片
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

def base_transform(image, size, mean):
    """
        就是将图片进行一个resize 然后减去数据集的平均图三通道的色彩值然后进行返回就可以了
    """
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

# 将base_transform函数进行封装而已
class BaseTransform(object):
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes = None, labels = None):
        return base_transform(image, self.size, self.mean)

