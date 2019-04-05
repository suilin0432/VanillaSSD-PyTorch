import torch
import torch.nn as nn
import torch.nn.functional as F
from data import voc as cfg
from ..box_utils import match, log_sum_exp

class MultiBoxLoss(nn.Module):
    """
        本类的作用是用作计算SSD的损失函数
        PS: 这里第1步按照论文 不是先对所有的 GT bbox 匹配一个iou最大的 然后在匹配所有的jaccard overlap > threshold 的吗?? 这么做没有太大的问题吗?
        1. 匹配 GT bbox, 规则是 jaccard index > threshold(默认 0.5)
        2. 通过 encode 处理 match 的 bbox
        3. 进行 hard negative mining -> 筛选出 3:1 的 negative:positive 的框进行训练

        损失函数:
            L(x, c, l, g) = (Lconf(x, c) + αLloc(x, l, g)) / N
            Lconf 使用 CrossEntropy Loss, Lloc 使用 SmoothL1 Loss
            其中:
                x: 匹配情况矩阵
                c: class confidences
                l: predict bbox
                g: GT bbox
                N: 匹配到的 default box 的数量
    """
    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label,
                 neg_mining, neg_pos, neg_overlap, encode_target, use_gpu=True):
        """
        :param num_classes: 物品类别数目
        :param overlap_thresh: 设置的算作positive 的 overlap的大小
        :param prior_for_matching: ???
        :param bkg_label: 背景 的 index
        :param neg_mining: 是否使用neg_mining
        :param neg_pos: negative挖掘的比例系数
        :param neg_overlap: ???
        :param encode_target: ???
        :param use_gpu: 是否使用GPU
        PS: 这几个???都没用到... 不是很了解是干什么用的
        """
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """

        :param predictions:
        :param targets:
        :return:
        """
        pass