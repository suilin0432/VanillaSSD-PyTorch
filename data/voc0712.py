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
            # 进行判断物体是否是 difficult的
            difficult = int(obj.find("difficult").text) == 1
            # 如果是 difficult 的物体 而且并不要求保存difficult
            if difficult and not self.keep_difficult:
                continue
            # 找到 obj 的 name 以及 bbox 信息, name 信息应该是可以用之前建立的class_to_ind做一个映射然后得到ID的
            name = obj.find("name").text.lower().strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                # 通过循环获取 bbox的信息, 然后将其按照图片的大小进行缩放
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            # 找到映射的类别号码
            label_idx = self.class_to_ind[name]
            # 把类别号码也放在bndbox之中
            bndbox.append(label_idx)
            res += [bndbox]

        return res

class VOCDetection(data.Dataset):
    """
        VOC 数据集检测
        input是 image target是 annotation
    """
    def __init__(self, root, image_sets=[("2007", "trainval"), ("2012", "trainval")], transform=None, target_transform=VOCAnnotationTransform(), dataset_name = "VOC0712"):
        """

        :param root: VOCdevkit 文件夹的 url
        :param image_sets: 使用的数据集, 并标明是 train、val、test
        :param transform: 对 input image 进行的变换操作
        :param target_transform: 对 input annotation 进行的变换操作
        :param dataset_name: 使用的数据集的名字
        """
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        # annotation的路径地址模板
        self._annopath = osp.join("%s", "Annotations", "%s.xml")
        # 图片的路径模板
        self._imgpath = osp.join("%s", "JPEGImages", "%s.jpg")
        self.ids = list()
        for (year, name) in self.image_set:
            # 把使用的所有的image_set的信息进行遍历
            rootpath = osp.join(self.root, "VOC"+year)
            # 打开所有要是用的数据的记录部分的id 记录其根目录和图片id
            # 这两个部分分别对应于根目录和后面文件id的部分 正好和__init__中的两个函数对应上了
            for line in open(osp.join(rootpath, "ImageSets", "Main", name + ".txt")):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    # 获取单个index对应图片的所有信息，包括 图片，GT，高，宽
    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        # 如果进行变换的话就进行所有的变换
        # 这里面是标签变换  这里面会将标签的信息变为相对于图片的相对大小
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            # 第二个参数bbox只取前4个维度 表示gt bbox信息 第三个参数表示类别
            img, boxes, labels = self.transform(img, target[:, :4], target[:,4])
            # 将图像进行通道变换BGR->RGB使得其支持torch的图片想格式
            img = img[:, :, ::-1].copy()
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # PS: 通道调整是在返回值做的
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


    # 获取单个index对应的图片
    def pull_image(self, index):
        img_id = self.ids[index]
        # 返回图片
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    # 获取单个inex对应的annotation
    def pull_anno(self, index):
        img_id = self.ids[index]
        # 进行xml文件的解析
        anno = ET.parse(self._annopath % img_id).getroot()
        # 大小不进行缩放 所以后两个参数都是1
        gt = self.target_transform(anno, 1, 1)
        # 返回图片的标号以及gt信息
        return img_id[1], gt

    # 获取单个index对应的img tensor
    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)