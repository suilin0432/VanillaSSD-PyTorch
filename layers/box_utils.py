import torch


# TODO:(已经解决，但是值得记录) encode 和 decode 不是很理解 作用说是为了放大梯度 但是还是不能理解...
def decode(loc, priors, variances):
    """

    :param loc: 预测 location
    :param priors: 先验 location
    :param variances: 先验box 的 variances
    :return: decoded bbox prediction
    """

    # boxes
    #   1. 对cx, cy 来说 是将先验框 + 预测值 * variances[0](也就是 0.1) 然后*priors[:, :2]
    #   首先 priors[:, :2]是基本偏移值, 这个值是应该加的, 但是为什么 要把loc[:, :2] * variances[0] * priors[:, :2]呢
    # PS: 查找之后是说是为了使得prediction 与 GT的误差放大，加大loss 加速收敛 在之前有一个 encode步骤，在这里decode
    #   2. 对h, w来说 进行先验框 * exp(预测值 * variances[1])

    boxes = torch.cat(
        (priors[:, :2] + loc[:, :2] * variances[0] * priors[:, :2],
         priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1
    )
    # 解码之后就是 cx,cy: (matched[:2] + matched[2:])/2  w,h: (matched[:, 2:] - matched[:, :2])
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def encode(matched, priors, variances):
    """
    对priorbox 与 GT框进行encode
    :param matched: 对于每个先验框的匹配的GT  ***** matched 的格式是 按照 左上 右下顶点坐标的格式******
    :param priors: 先验框
    :param variances:
    :return:
    """
    # PS: 不除以 variance 的部分就是正常的损失函数版本 因为学习参数是 t_x,t_y =  (g_cx - d_cx)/d_w, (g_cy - d_cy)/d_h
    #     所以说只用 / variances是进行放大的部分 是和论文直接说的不一样的 但是论文作者在 caffe 版本也这么做了 而且 包括 fasterRcnn 等也都这么做了
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])

    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


def nms(boxes, scores, nms_thresh=0.5, top_k=200):
    """
    进行 NMS 操作 PS: 这里的 nms 是针对每个类别单独进来筛选的
    :param boxes: 筛选过的prediction boxes
    :param scores: 筛选过的boxes 的分数
    :param nms_thresh: 设置的 nms_thresh
    :param top_k: 设置的最大的筛选后的box个数
    :return: keep 存储的是按照置信度系数进行排序放入的index, count 是数量
    """
    # 因为scores的 shape 是 num 的 所以创建 一个 batchSize大小的 初始化为0 的 long 类型的Tensor
    keep = scores.new(scores.size(0)).zero_().long()
    # 如果数量boxes 是 参数数量是0个返回 keep
    if boxes.numel() == 0:
        return keep
    # 获取左上 右下坐标
    # print(boxes, boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]  # 获取 top_k个最高分对象
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # 获取分数最高的对象
        keep[count] = i  # 把对象放入
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]

        # 就是下角标选数字啊...
        # 取出来最高分数后剩下的对象的信息 参数分别是 input, dim, indexLists, output
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # 将所有的框限制在 选中框内 -> 用来计算重叠面积IoU
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h

        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union

        # le 是 <=
        idx = idx[IoU.le(nms_thresh)]
    return keep, count


def point_form(boxes):
    """
        进行中心xy, wh 向 左上右下格式的转换
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)


def intersect(box_a, box_b):
    """
        input两个Tensor 一个 [A, 4] 一个 [B, 4]
        然后目的是将二者对齐然后计算两者的 intersect
        [A, 2] -> [A, 1, 2] -> [A, B, 2]
        [B, 2] -> [1, B ,2] -> [A, B, 2]

        返回就是 [A, B] 形状
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, 0:2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 0:2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """
        计算IoU
    """
    # 首先 计算二者的重叠面积
    inter = intersect(box_a, box_b)
    # 因为剪完之后维度会变化 从 A, 4 -> A  然后 进行 A -> A, 1 -> A, B 的操作
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)

    union = area_a + area_b - inter
    return inter / union


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """
    将每个 prior 框和 GT bbox 进行匹配, 然后encode bounding boxes, 最后返回(其实是直接赋值)匹配信息
    :param threshold: IoU shreshold 阈值
    :param truths: GT bbox shape:[GT_detection_number, 4] 这个4 是 左上 右下的坐标
    :param priors: 先验框 shape:[priorNum, 4] 这个4 表示的是中心xy, 宽高
    :param variances: 配置中的variances, 用来进行梯度的放大
    :param labels: GT bbox 标签 shape:[GT_detection_number] 就是传入之前 target 分出来的一部分
    :param loc_t: 对prior box匹配上的box的location数据 shape:[batchSize, numPrior, 4]
    :param conf_t: 对prior box 匹配上的conf_t的数据 shape:[batchSize, numPrior]
    :param idx: batch 序号, 进行的第idx个batch
    :return: 并没有直接的return值, 实际上是对 loc_t 和 conf_t 进行直接的修改, 然后就不用进行返回值了
    """
    # 计算IoU  但是先要将priors 的格式 从中心xy, wh 改为 左上右下坐标的格式 进行二者的格式的统一
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # 先匹配最佳匹配的项  因为overlaps的格式是[A,B]，且我们把 Truth 当做了A, prior当做了B, 因此我们想要找每个 GY 对应的
    # 最佳的prior 需要 按照第一个维度进行取最大值
    # PS: 原来的代码 用的是overlaps.max(1, keepdim=True) 然后后面又 squeeze_ 所以keepdim好像没什么必要, 就去掉了
    # 维度 GT数量
    best_prior_overlap, best_prior_idx = overlaps.max(1)
    # 找和每个prior匹配的最好的 GT box
    # 维度 prior数量
    best_truth_overlap, best_truth_idx = overlaps.max(0)

    # 因为best_prior_idx是GT匹配上的prior框的序号, best_prior_idx是prior上匹配到的GT的序号
    # 现在我们要将 best_prior_idx 中 GT 匹配到的 最佳的 prior 框进行 匹配到的GT框的编号的设置
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
        # PS: 有一个问题, 就是会不会因为一个best_truth_overlap的值确实比较小, 但是其确实一个GT_box匹配的最好的, 这么的话就会被忽略了
        # 所以在这里我强制更改一下 GT 匹配的最好的 那个框的 overlap直接为1
        best_truth_overlap[best_prior_idx[j]] = 1
    # shape: numPrior, 4
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx] + 1
    # 将没有达到 threshold 的看做背景
    conf[best_truth_overlap < threshold] = 0
    # 进行编码 一是先将 左上右下点的格式 转变为 中心宽高的格式 然后进行中心点偏差的计算 以及宽高偏差计算, 具体信息看 paper 或者 笔记就可以了
    # PS: 在Augumentation中已经将 GT 的坐标变成了相对坐标的格式了
    loc = encode(matches, priors, variances)
    # 进行传入的参数的赋值 python 传递的是引用, 直接可以用
    loc_t[idx] = loc
    # PS: conf 其实拿到的是 label 序号 而不是置信度系数
    conf_t[idx] = conf

def log_sum_exp(x):
    """
        PS: 网上找到了解释, 减去那个 x_max 是为了数值稳定性
        目的是计算loss, 找到最大的 3 倍于 positive 数量的loss, 作为negative的loss贡献
    """
    # 找到所有数据中最大的那一个
    x_max = x.data.max()
    # 减去最大的那个并不会影响大小关系, 只是降低了值的关系
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

