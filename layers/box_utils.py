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
    #PS: 查找之后是说是为了使得prediction 与 GT的误差放大，加大loss 加速收敛 在之前有一个 encode步骤，在这里decode
    #   2. 对h, w来说 进行先验框 * exp(预测值 * variances[1])

    boxes = torch.cat(
        (priors[:, :2] + loc[:, :2] * variances[0] * priors[:, :2],
         priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])),1
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
    g_cxcy /= (variances[0] * priors[:,2:])

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
    idx = idx[-top_k:] #获取 top_k个最高分对象
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1] # 获取分数最高的对象
        keep[count] = i #把对象放入
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
        inter = w*h

        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union

        # le 是 <=
        idx = idx[IoU.le(nms_thresh)]
    return keep, count


