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
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:,2:])

    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)
