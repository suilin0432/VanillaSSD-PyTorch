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


class RandomContrast(object):
    """
        进行随机乘法加权
    """
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    # 从 opencv 读取的 height, width, channel 的形式变为 torch.Tensor的 channel, height, width形式
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels

# 最终要的augument操作部分
class RandomSampleCrop(object):
    """
        Crop操作
        Arguments:
            img: (numpy.ndarray)要用来进行训练的图片
            boxes: (Tensor): GT box的坐标
            labels: (Tensor): GT box的类别标号
            mode(float tuple): 选择的jaccard overlaps的大小
    """
    def __init__(self):
        self.sample_options=(
            # 不对图片进行任何操作
            None,
            # crop 一个 patch 使其最小的 jaccard overlaps w/obj 是 0.1, 0.3 0.5, 0.7, 0.9
            # PS: 原来的实现中没有0.5这个选项, 但是论文中是有的
            # 每个 IoU 都不要超过 那个设定的值, 但是最后筛选 box 的时候会将所有的 中心不在 patch 范围内的box删除掉
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # 随机选取一个patch
            (None, None)
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # 随机进行一个模式的选择
            mode = random.choice(self.sample_options)
            if mode is None:
                # 不进行任何操作
                return image, boxes, labels
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float("-inf")
            if max_iou is None:
                max_iou = float("inf")

            # 最多进行尝试 50 次没有的话重新进行选择一次 mode
            for _ in range(50):
                current_image = image

                # 随机选取在 0.3
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # 论文中提及到长宽比要在 1/2 - 2 之间
                if h/w < 0.5 or h/w > 2:
                    continue

                # 选取左上角的坐标
                left = random.uniform(width-w)
                top = random.uniform(height-h)

                # 获取截取的形状
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # 计算IoU
                overlap = jaccard_numpy(boxes, rect)

                # 是否满足IoU的条件, 只要有任意一个物体被截取的不适当都不可以
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # 获取新的图片的范围
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # 计算中心坐标
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # 找到所有中心在 patch 中的 boxes 的 mask
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                mask = m1 * m2

                # 如果没有一个box符合要求 就放弃这次的
                if not mask.any():
                    continue

                current_boxes = boxes[mask, :].copy()

                current_labels = labels[mask]

                # 将左上和右下角坐标限制在 patch 的范围之内
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])

                # 将坐标减去 patch 在原图的左上角坐标 -> 使得能够最后协调到调整后的位置上
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels

class Expand(object):
    """
        用平均图的值进行expand操作 填充空白的区域
        这个操作是使用的那个 为了训练检测小物体能力而使用的 先将图片的目标尺寸放大,
        然后用一个平均图进行填充, 然后将image放入到一个随机的位置中, 然后会在后面的步骤中进行图片的缩放
    """
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)
        expand_image = np.zeros( (int(height*ratio), int(width*ratio), depth), dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height), int(left):int(left+width)] = image
        image =expand_image
        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels

class RandomMirror(object):
    """
        将图像镜像反转
    """
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]

class SwapChannels(object):
    """
        进行 channel 的转换
    """
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image

class PhotometricDistort(object):
    """
        光度畸变... 说实话不是很明白其中的那个 rand_light_noise 为什么要交换channel
        其实仔细想想就是切换了一下通道改变了一下色彩->使得对色彩畸变的鲁棒性更好???
    """
    def __int__(self):
        self.pd = [
            RandomContrast(),
            # 将image 从BGR->HSV
            ConvertColor(transforms="HSV"),
            # 进行随机饱和度的变换
            RandomSaturation(),
            # 进行随机色相的变换
            RandomHue(),
            # 再次将iamge 从HSV->BGR
            ConvertColor(current="HSV", transforms="BGR"),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.rand(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)

class SSDAugmentation(object):
    """
        从上到下:
            1. 将图片的格式变为 float
            2. 将坐标从相对坐标改为绝对坐标
            (因为在取出来坐标的时候VOCAnnotationTransform进行了一次标签的从绝对坐标到相对坐标的变换)
            3. 光度畸变
            4. Zoom-off
            5. 随机Crop
            6. 随机镜像
            7. 将坐标从绝对坐标改为相对坐标
            8. Resize
            9. 减去平均图
    """
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)