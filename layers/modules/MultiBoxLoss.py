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
        num_classes = self.num_classes

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
        if self.use_gpu and torch.cuda.is_available():
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        loc_t.requires_grad = False
        conf_t.requires_grad = False

        # PS: conf 其实拿到的是 label 序号 而不是置信度系数
        # 所以这里判断的就是将背景排除掉
        pos = conf_t > 0
        # pos shape: batchSize, numPriors, 1
        num_pos = pos.sum(dim=1, keepdim=True)
        # 还是不知道为什么要keepdim
        # 扩充 pos 到 loc_data 的形状上,
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # 将 loc_data 所有 batch 的 fore数据整合到一起
        loc_p = loc_data[pos_idx].view(-1, 4)
        # 同理处理 loc_t
        loc_t = loc_t[pos_idx].view(-1, 4)
        # loc_p 获取的是predict的数据 loc_t 则是每个prior 对应的GT数据
        # 选择不进行size_average的原因是因为 最后才 /N ...
        # 计算边框的损失函数 smooth_l1_loss -> 在差距 <1 的时候是 1/2*(target-prediction)**2   >=1 的时候 L1 - 1/2
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # 将获取到的 conf 信息所有batch进行整合
        batch_conf = conf_data.view(-1, self.num_classes)
        # 计算
        """
            torch.target(input, dim, index, out=None)
            意思是在指定的维度上进行其他维度元素的选择, 而不是按照原来的单一的维度进行选择, 而是按照你给定List的index进行当前维度的选择
            dim 是索引的维度
            index 是要进行聚合的下标
            out[i][j][k] = tensor[index[i][j][k]][j][k]  # dim=0
            eg:
            a = torch.tensor([[1,2,3],[4,5,6]])
            # a-> [ [1,2,3],
            #       [4,5,6]]
            b = a.gather(0, torch.LongTensor([[0,1,0]]))
            # 表示选择 a[0,0], a[1,1], a[0,2]
            # PS: index那个维度数量一定要和 a 对应
        """
        # 这里的 loss_c 不是计算 conf 的损失函数, 而是用来排序然后筛选出 hard_mining的个体
        # log_sum_exp 的正常目的是应该 得到所有的 loss 的相对值
        # batch_conf.gather(1, conf_t,view(-1,1)) 的目的是为了 获取其正确分类的实际得分
        # TODO: 有点不明白为什么这么做???
        # 所以这里的 loss_c 是 计算相对loss之后 还要减去其所属类别 所预测到的分数
        # 其实batch_conf.gather(1, conf_t.view(-1, 1)) 的目的是为了找到最好的那个，但是实际因为进行的目的找到negative mining 所以其对应的就是0
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # 这里原来的版本有一个Bug 把上下两行颠倒一下, 在这里保证维度的对齐就好了
        loss_c = loss_c.view(num, -1)
        # 进行hard_miner
        # 将所有positive的框刨除
        loss_c[pos] = 0
        # 进行分数的排序  shape: [batchSize, numPrior] 因为要针对每张图片进行hardMining而不是一起进行
        _, loss_idx = loss_c.sort(1, descending=True)
        # 分数排序之后对 idx 号进行从小到大排序 目的是为了找到各个序号对应的下角标 也就是下角标对应数字 在分数排名的位置
        _, idx_rank = loss_idx.sort(1)
        # 获取postive的框的数量
        num_pos = pos.long().sum(1, keepdim=True)
        # 筛选出 hard mining 倍数的negative框, 但是不超过 框的数量
        # TODO: 按照论文是应该不超过 200 个框吧...
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # 筛选出进行hard mining的编号
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # 开始进行conf_loss的计算
        # 原来 pos shape:[num, numPriors] 仅仅表示这个对象是否是positive贡献的对象
        # 将其扩充第三个维度到 shape:[num, numPriors, classNumber]
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # 同理进行neg 的扩充
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # 提取出来所有 对应位置 > 0 的数据 然后进行 view()
        # 将shape [batchSize, ...] 的batchSize进行融合
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)] #获取正确的类别
        # 计算conf_loss
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # 获取positive框的数量， 就是论文所说的匹配到的数目, 而不是所有的 贡献到损失函数中的数目
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


