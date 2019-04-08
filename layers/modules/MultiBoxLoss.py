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

        :param predictions: 输入图片 x 经过 net 后产生的 predictions 信息
        :param targets: 目标标签
            conf_data: shape:(batchSize, numPriors, num_classes)
            loc_data: shape:(batchSize, numPriors, 4)
            priors: shape:(num_priors, 4)
        """
        # 获取 SSDModel 在 train 模式下的返回值 loc信息, conf信息 以及预测的 priors bbox 信息
        # 下面的分别是 位置信息, 置信度信息, 先验框信息
        loc_data, conf_data, priors = predictions
        # 获取batchSize 数目
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        # 获取numPrior 数目
        num_priors = (priors.size(0))
        num_classes = self.num_class

        # 将每个priors与 ground truth boxes进行匹配
        loc_t = torch.Tensor(num, num_priors, 4)
        # 为什么是 LongTensor -> 因为 conf_t 一定是 0/1 因为是GT的
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            # 单个batchSize 的 target是 [[lefttopx, y, rightdownx, y, 类别编号], [...], ...]
            # 获取当前batch的所有不包含类别的标签 就是 所有的GT的两角坐标
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            # 进行匹配
            # PS: list 什么的 在函数中传递的都是引用, 所以很明显的会被更改掉
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        loc_t.requires_grad = False
        conf_t.requires_grad = False

        pos = conf_t > 0

