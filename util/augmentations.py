import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

# 计算重叠面积
def intersect(box_a, box_b):
    """
    :param box_a: 输入的a shape: [n, 4]
    :param box_b: 输入的b shape: [4]
    """
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * int[:, 1]

# 计算IoU
def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0])*
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0])*
              (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union

class Compose(object):
    """
        作用是将多个 augument 操作进行合并
        源代码给的Examples:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        """
        :param transforms: transforms 是 list 包含了一系列的transform操作
        """
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        """
        :param img: 要进行图片变换的图片
        :param boxes: 是要进行变换的GT框
        :param labels: 是GT框对应的GT label 不需要变换, 但是会进行筛选
        :return: 返回变换过后的img, boxes, labels
        """
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class Lambda(object):
    """
        进行一个lambda形式的变换
    """
    def __init__(self, lamda):
        assert isinstance(lamda, types.LambdaType)
        self.lamda = lamda
    def __call__(self, img, boxes=None, labels=None):
        return self.lamda(img, boxes, labels)

class ConvertFromInts(object):
    """
        将img的类型从float32变为int类型
    """
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class SubtractMeans(object):
    """
        进行平均图的减法
    """
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels

class ToAbsoluteCoords(object):
    """
        将相对与原图的百分比的box坐标更改为绝对大小
    """
    def __call__(self, image, boxes=None, labels=None):
        height,width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 1] *= height
        boxes[:, 2] *= width
        boxes[:, 3] *= height

        return image, boxes, labels

class ToPercentCoords(object):
    """
        将绝对坐标更改为相对坐标
    """
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 1] /= height
        boxes[:, 2] /= width
        boxes[:, 3] /= height

        return image, boxes, labels

class Resize(object):
    """
        图片size变换操作(PS: 因为这里暂时没有进行相对坐标什么的变换, 只是相对于图片的, 所以不用改box大小什么的)
    """
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


"""
PS: 下面的三个操作都是针对 HSV 颜色空间来说的 
色相(Hue)、饱和度(Saturation)、明度(Value)
"""
class RandomSaturation(object):
    """
        进行随机的 Saturation 操作
    """
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        # 以 1/2 的概率进行饱和度变换
        if random.randint(2):
            # 在 lower -> upper 的比例之间进行一个乘法
            # 只对 饱和度 通道进行操作
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels

class RandomHue(object):
    """
        随机色彩偏差
    """
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels

class RandomLightingNoise(object):
    """
        进行通道的转换
    """
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels

class ConvertColor(object):
    """
        进行颜色空间的转换 BGR与HSV之间的转换
    """
    def __init__(self, current="BGR", transform = "HSV"):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels
