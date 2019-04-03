import torch
from torch.autograd import Function
from ..box_utils import decode, nms
# 原来的代码直接引入 voc 作为 cfg 这里 改一下, 加一个参数 允许用其他的config进行训练
# from data import voc as cfg

class Detect(Function):
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, cfg):
        """
        :param num_classes: 类别数
        :param bkg_label: 背景标签的下标
        :param top_k: 取最高的 top_k 个框
        :param conf_thresh: conf_thresh之下的会被筛掉
        :param nms_thresh: 超过 nms_thresh 的会被筛掉
        :param cfg: 配置
        """
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.cfg = cfg
        self.variance = cfg["variance"]

    def forward(self, loc_data, conf_data, prior_data):
        """
        :param loc_data: location 数据
            Shape: [batchSize, num_priors, 4]
        :param conf_data: confidence 数据
            Shape: [batchSize * num_priors, num_classes]
        :param prior_data: prior boxes 以及 prior layers 的 variances
            Shape: [num_priors, 4]
        """
        # 获取batchSize
        num = loc_data.size(0)
        # 先验框个数 按照论文标准的是 8732个
        num_priors = prior_data.size(0)
        # 输出 维度 batchSize, numClass, topK, 6 相比于源代码扩展了一个 项 作为类别信息的存储
        output = torch.zeros(num, self.num_classes, self.top_k, 6)
        # 将conf_data维度改为 batchSize, numPriors, numClasses
        # 然后将后两个维度倒置 -> batchSize, numClasses, numPriors 这么做是为了后面nms对类别处理方便
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # 将predictions转化为bbox
        # 对每个batch进行
        for i in range(num):
            # 输入 预测的位置信息, 先验框信息, 以及variance
            # PS:decode 后得到的是 左上 右下格式的boxes
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # 对每个class 进行nms的处理
            conf_scores = conf_preds[i].clone()
            # 因为 默认 0是背景，所以不会对背景进行取框
            for cl in range(1, self.num_classes):
                # gt()是逐个元素比较 >= conf_thresh时才为True
                # shape: numPriors
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # 过滤之后的当前 batch 当前 class 的分数
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    # 如果都没够超过阈值 跳过
                    continue
                # 添加一个维度(PS, 其实不用expand_as也可以进行下面的操作而且不用 view(-1, 4))
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # 进行 预测信息的 筛选 筛掉所有的不足 conf_thresh 的对象
                boxes = decoded_boxes[l_mask].view(-1,4)
                # 进行nms操作
                # 在 nms 里面进行top_k 的筛选(这里是针对每个类别都进行 最大 200个的限制, 而原论文是 每张图片 200个 最大)
                # PS: 后面还会进行一个top_k 的筛选... 没有问题
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                # 对output[当前batch, 当前class, 前检测到的个数] 进行赋值
                # output 最后那个 维度 6 是按照 分数, cx, cy, w, h, 类别 进行排列的
                classMessage = torch.zeros(count, 1)
                torch.nn.init.constant(classMessage, cl)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]], classMessage), 1)
        # 最后调整output 到 batchSize, -1(numClass*top_k), 6 的形状
        flt = output.contiguous().view(num, -1, 6)
        # 对flt按照分数进行排序 获取排序后的坐标顺序
        _, idx = flt[:, :, 0].sort(1, descending=True)

        # 取前200个框
        _flt = flt.view(-1, 6)
        _idx = idx.view(-1)[:200]

        _flt = _flt[_idx]
        _flt_mask = _flt[:,0].gt(self.conf_thresh)
        # 这是我修改的, 返回分数最高的200个框 下面注释掉的是原来的(如果不足过滤掉置信系数 < conf_thresh的)
        return _flt[_flt_mask]
        # _, rank = idx.sort(1)
        # flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        # return output
