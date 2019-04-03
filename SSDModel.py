import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from data import voc, coco
import os

class SSD(nn.Module):
    """
    使用VGG Network作为backbone 然后在后面加上一些conv层
    然后每个 multibox 层包含了
        1)  conv2d for class conf scores
        2)  conv2d for localization predictions
        3)  生成 default bbox
    """
    def __init__(self, phase, size, base, extras, head, num_classes):
        """
        :param phase:  test / train 表明状态
        :param size: input 的 image size
        :param base: VGG16 layers
        :param extras: 额外添加的layers 用来生成multi-feature map的层
        :param head: mutlibox head 由 loc 和 conf 的 conv 层构成
        :param num_classes: 类别
        """
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # 如果 num_classes == 21 那么就是 (coco, voc)[True] == (coco, voc)[1] 选择的就是voc的配置
        self.cfg = (coco, voc)[num_classes == 21]
        # 创建先验框类
        self.priorbox = PriorBox(self.cfg)
        # 这里因为 pytorch1.0 将Tensor和Variable进行了合并, 所以用Tensor了 然后计算的时候记得要用 torch.no_grad()
        self.priors = torch.Tensor(self.priorbox.forward())
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # 将 conv4_3 层的输出进行 L2 Normalized的 Layer
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        # 如果设置为 test 模式那么就进行一下softmax操作(SSD的类别损失函数计算的时候使用了softmax, PS:yolo好像只是简单地进行了sigmoid)
        # 以及进行Detect操作
        if phase == "test":
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0,200, 0.01, 0.45, self.cfg)

    # 正向传播函数构建
    def forward(self, x):
        #PS: 值得注意的是在 构建SSD时候传入的 phase 是test还是train会决定进行的操作是什么(最后的步骤)
        #sources 是featuremap
        sources = list()
        loc = list()
        conf = list()

        # 添加conv4_3层作为第一个featureMap 38 * 38
        # 对vgg的conv4_3进行L2Norm
        # 23的原因在下面有介绍
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # 添加fc7(或者说 conv7) 作为第二个featureMap 19 * 19
        # PS: fc7是VGG截取掉fc6与fc7层并将pool5层进行修改之后添加conv6与conv7层
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # 添加conv8_2作为第三个 featureMap 10 * 10
        # 添加conv9_2作为第四个 featureMap 5 * 5
        # 添加conv10_2作为第五个 featureMap 3 * 3
        # 添加conv11_2作为第六个 featureMap 1 * 1
        for k, v in enumerate(self.extras):
            # PS: inplace表示原地计算, 即覆盖计算, 如果不进行覆盖计算的时候会创建一个新的变量, 覆盖计算则不会 eg: 比如 y = x + 2 和 x+=2的关系
            x = F.relu(v(x), inplace=True)
            # 每两层加一个feature map
            if k % 2 == 1:
                sources.append(x)

        # 用multibox进行 feature map 的计算
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # 将所有featuremap的loc 和 conf进行拼接
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            # 这里面置信系数 需要进行softmax操作, 但是cxcy,wh没有进行处理
            output = self.detect(
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                self.priors.type(type(x.data))
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        # 截取最后一个"."符号
        other, ext = os.path.splitext(base_file)
        if ext == ".pkl" or ".pth":
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc:storage))
            print("Loading weights finished!")
        else:
            raise ValueError('Only .pth and .pkl files supported.')

# 构件SSD的函数, 用来创建SSD Module
def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        raise ValueError("The Phase {0} can not be recognized!".format(phase))
    if size != 300:
        raise ValueError("Sorry the current config only support 300 as the input size!")
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_,num_classes)

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "C":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # 因为删除了 fc6 fc7 然后改 pool5 还加上了conv6 和 conv7
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # 这层大小不会变 但是会因为 空洞卷积而增加感受野
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers

def add_extras(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != "S":
            if v =="S":
                layers += [nn.Conv2d(in_channels, cfg[k+1], kernel_size=(1,3)[flag], stride=2,padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1,3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def multibox(vgg, extra_layers, cfg, num_classes):
    """

    :param vgg: 这里输入的是 vgg Layers 这个List
    :param extra_layers: 这里输入的是 extra Layers 这个List
    :param cfg: 这里输入的是 mbox数量那个配置文件List
    :param num_classes: 类别数目
    :return:
    """
    loc_layers = []
    conf_layers = []
    # TODO: 这里为什么，以及为什么extra层添加的都没有激活函数 直接全卷积... PS:这里没有损失函数可以理解, 预测不希望在这里将一些信息损失, 但是为什么extra层都没有激活函数
    # PS: 取出来的是 conv4_3 没有进行relu之前的 conv之后的层
    #     同样 fc7 (conv7) 取出来的也是 没有进行relu之前的conv7
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]

    # enumerate 后面的那个2表示从2开始进行index 的值
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


#PS: 对 VGG base结构 进行解析
"""
    1. 数字代表通道数 当不包含batch_norm的时候由 一个conv2d和一个ReLU构成 
       所以layer中会加入两层, 当包含batch_norm的时候会在layer中加入三层
    2. M和C都表示 进行MaxPool2d 且都是kernel_size=2, stride=2 但是
       C会多一个ceil_mode的参数，ceil表示计算结束后的shape的时候是否对shape
       进行ceil()即 37.5->38 不设置的话会 默认为 floor() 即 37.5->37
    # 因此上面设置的23是代表着 到达表示产生conv4_3的层数
    # 因为每一次maxPool算作 conv下标+1 所以在C之后是 conv4_3, 在这里会进行一个batch_norm
"""
base = {
    "300": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
    "512": [],
}

"""
    extras层是纯卷积 1*1*256 -> 3*3*512s2 -> 1*1*1*128 -> 3*3*256s2 -> 1*1*128 -> 3*3*256 -> 1*1*128 -> 3*3*256
    其中S表示下一层将会是步长为2
"""
extras = {
    "300": [256, "S", 512, 128, "S", 256, 128, 256, 128, 256],
    "512": [],
}
# 就是每层的default box的个数
mbox={
    "300": [4, 6, 6, 6, 4, 4],
    "512": [],
}

