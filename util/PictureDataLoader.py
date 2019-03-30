import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

class PictureDataLoader(Dataset):
    # 不进行cuda操作
    def __init__(self, imgRecordFile, networkInputShape = (416, 416)):
        """

        :param imgRecordFile: 记录 文件目录的文件
        :param networkInputShape: 输入到网络中的时候图像的输入size
        """
        self.networkInputShape = networkInputShape
        # 读取记录所有图片的路径
        self.imgRecordFile = imgRecordFile
        imgSrcListFile = open(imgRecordFile, 'r')
        self.imgSrcList = imgSrcListFile.readlines()
        # 去除 \n
        self.imgSrcList = [i.strip() for i in self.imgSrcList]
        # 去除空行
        self.imgSrcList = [i for i in self.imgSrcList if i != ""]

        # label文件路径
        self.labelSrcList = [i.replace(".jpg", ".txt").replace(".png", ".txt") for i in self.imgSrcList]

        # 图片总数量
        self.length = len(self.imgSrcList)

        # 一张图片最大的anchor数目
        self.maxAnchor = 300



    def __getitem__(self, index):
        # 返回target:
        # target = {"img": img, "label": labels, "imgPath": path, "originHeight": originHeight, "originWidth": originWidth} 其中originSize 为 (height, width) 方式保存
        # 但是输出的时候会被合并所以要用每个的index进行对应
        target = {}
        # 获取图片
        imgSrc = self.imgSrcList[index % self.length].strip()

        img = cv2.imread(imgSrc)
        # print(imgSrc)
        # 进行图片补全变换 PS:因为一定是偶数尺寸 因为要/32 所以不会出错，如果是奇数尺寸会出错的
        height, width = img.shape[:2] #获取读取图片的大小
        target["originHeight"] = height
        target["originWidth"] = width
        target_h, target_w = self.networkInputShape #获取目标图片尺寸大小
        scale_w = target_w / width  # 获取按照宽度放缩的比例
        scale_h = target_h / height #获取按照长度放缩的比例
        minScale = min(scale_h, scale_w)

        after_w = int(width * minScale)
        after_h = int(height * minScale)
        pad_w = target_w - after_w
        pad_h = target_h - after_h
        pad = ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2), (0,0))
        img = cv2.resize(img, (after_w, after_h))
        img = np.pad(img, pad, "constant", constant_values=128)

        # 变换numpy格式为pytorch格式
        img = img/255.0
        img = img[:,:,::-1].copy()
        img = img.transpose(2,0,1)
        # print(img.shape)
        img = torch.from_numpy(img).float()

        target["img"] = img

        # 开始进行所有标签的处理
        labelSrc = self.labelSrcList[index % self.length].strip()
        labels = np.loadtxt(labelSrc).reshape(-1,5)
        # 因为上面对图片进行补全了，所以要改变检测框的位置

        # 首先变换为 (x1, y1) (x2, y2) 方式记录
        x1 = (labels[:,1] - labels[:,3]) * after_w
        y1 = (labels[:,2] - labels[:,4]) * after_h
        x2 = (labels[:,1] + labels[:,3]) * after_w
        y2 = (labels[:,2] + labels[:,4]) * after_h
        x1 += pad[1][0]
        x2 += pad[1][0]
        y1 += pad[0][0]
        y2 += pad[0][0]
        labels[:,1] = (x1 + x2) / 2 / target_w
        labels[:,2] = (y1 + y2) / 2 / target_h
        labels[:,3] *= after_w / target_w
        labels[:,4] *= after_h / target_h

        empty_label = np.zeros((self.maxAnchor, 5))
        if len(labels) > self.maxAnchor:
            empty_label = labels[0:self.maxAnchor]
        else:
            empty_label[0:len(labels),:] = labels
        target["label"] = empty_label
        # print(len(target["label"]))
        target["imgPath"] = imgSrc
        return target

    def __len__(self):
        return self.length

"""
RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 115 and 176 in dimension 1 at /Users/soumith/b101_2/2019_02_08/wheel_build_dirs/wheel_3.6/pytorch/aten/src/TH/generic/THTensorMoreMath.cpp:1307

报错原因是因为 target["label"] 尺寸大小不能对应
所以在上面设置了一个 最大数量...
"""