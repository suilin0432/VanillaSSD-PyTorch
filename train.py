from data import *
# TODO: 变换是要进一步完成的内容
# from util.argumentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from SSDModel import build_ssd
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Training for VanillaSSD-Pytorch")
# add_mutually_exclusive_group 是设定一组互相排斥的参数 并且其允许一个required参数，如果这个参数是True, 那么这个互斥组是必须包含的
# PS: 但是这里这个互斥组什么都没有...
# PS: *** 同时包含这个train_set的时候会导致在 help 或者 -h 的时候出现解析的错误... 所以在这里先注释掉了
# train_set = parser.add_mutually_exclusive_group()
parser.add_argument("--dataset", default="VOC", choices=["VOC", "COCO"], type=str, help="指定的数据集,用来指定加载的数据集到底是什么")
parser.add_argument("--dataset_root", default=VOC_ROOT, help="数据集的根目录地址")
parser.add_argument("--basenet", default="vgg16_reducedfc.pth", help="预训练网络参数的权重文件")
parser.add_argument("--batch_size", default=32, type=int, help="训练所采用的batch_size")
parser.add_argument("--resume", default=None, type=str, help="从一次训练的中途进行参数文件的读取")
parser.add_argument("--start_iter", default=0, type=int, help="resume的时候进行的iter数目")
parser.add_argument("--num_workers", default=4, type=int, help="进行图片数据加载的时候使用的线程数量")
parser.add_argument("--cuda", default=True, type=str2bool, help="是否使用cuda进行加速")
parser.add_argument("--lr", "--learning-rate", default=1e-3, type=float, help="最初始的学习率的设定")
parser.add_argument("--momentum", default=0.9, type=float, help="设置的momentum的大小")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="SGD的权重衰减的速率参数")
parser.add_argument("--gamma", default=0.1, type=float, help="SGD的gamma参数")
parser.add_argument("--visdom", default=False, type=str2bool, help="是否进行训练的可视化")
parser.add_argument("--save_folder", default="weights/", help="进行权重参数记录的地方")
args = parser.parse_args()

# 进行默认变量类型的设置
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("The CUDA is available on your PC, but you set the cuda parameter False, please change the train.py to modify if you want.")
        torch.set_default_tensor_type("torch.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def train():
    if args.dataset == "COCO":
        print("Sorry, the code to support the COCO dataset is not avaliable now.")
        exit()
    elif args.dataset == "VOC":
        cfg = voc
        # 这里面VOCDetection除了变换操作没有使用默认设置其他的都是用了默认设置但是问题是 好像这里暂时没有VOC2012的数据...
        # TODO: 这里的SSDAugmentation还没有完成...
        # PS: cfg["min_dim"] 表示的图片的大小尺度
        # dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(cfg["min_dim"], MEANS))
        dataset = VOCDetection(root=args.dataset_root, image_sets=[("2007", "trainval")])

    # 可视化工具的初始化
    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd("train", cfg["min_dim"])
    # ???为什么要在这里进行这么一个步骤
    net = ssd_net

    if args.cuda and torch.cuda.is_available():
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print("Resuming training, loading {} ...".format(args.resume))
        # ssd的load_weights强制进行了CPU加载, 所以不会出错
        ssd_net.load_weights(args.resume)
    else:
        # 如果不使用训练好的完整模型参数 或者 训练中途时终止的模型参数 那么就进行base net 的加载
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print("Loading base network...")
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda and torch.cuda.is_available():
        net = net.cuda()

    # 如果没有进行 resume的时候那么将所有模型的非backbone层(extra的层以及loc和conf添加的conv层)进行参数的初始化
    if not args.resume:
        print("Initializing weights...")
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # 函数优化器选择使用SGD
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # 损失函数计算类
    # TODO: 需要将这个MultiBoxLoss类进行完成
    criterion = MultiBoxLoss(cfg['num_class'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)

    # 设置为训练模式
    net.train()

    # 损失记录
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print("Loading the dataset...")

    # 获取图片的数目
    epoch_size = len(dataset)
    print("Training SSD on:",dataset.name)
    print("Using the specified args: ")
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = "VanillaSSD on "+dataset.name
        vis_legend = ["Loc loss", "Conf Loss", "Total Loss"]
        iter_plot = create_vis_plot("Iteration", "Loss", vis_title, vis_legend, viz)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend, viz)

    # 加载数据集到 DataLoader 中
    # PS: collate_fn 这个函数是用来处理一个批次数据的返回格式的 防止默认设置要求所有的维度必须相同 导致需要进行冗余的 detection框的数量限制
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    batch_iterator = iter(data_loader)
    # 开始进行迭代, 从设置的iter开始到最后结束
    for iteration in range(args.start_iter, cfg["max_iter"]):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(viz, epoch, loc_loss, conf_loss, epoch_plot, None, "append", epoch_size)

            loc_loss = 0
            conf_loss = 0
            epoch += 1
        if iteration in cfg["lr_steps"]:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # 加载训练数据
        # 每次迭代都会取出batchsize张图片
        # 根据collate_fn的设置, 返回值前面是 Tensor格式的图片信息 batchSize, channel, height, width  后面是 annotation 值
        images, targets = next(batch_iterator)

        if args.cuda and torch.cuda.is_available():
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]

        # 前向传播
        t0 = time.time()
        out = net(images)

        # 反向传播
        optimizer.zero_grad()

        # 进行损失函数的计算
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()

        # 更新
        optimizer.step()
        t1 = time.time()
        # PS: 这个 [0] 要看一下为什么取 0
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        if iteration % 10 == 0:
            print("timer: %.4f sec." % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')

        if args.visdom:
            update_vis_plot(viz, iteration, loss_l.data[0], loss_c.data[0], iter_plot, epoch_plot, "append")

        # 每 500 个 iteration 进行一次参数信息的保存
        if iteration != 0 and iteration % 5000 == 0:
            print("Saving state, iter:",iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                       repr(iteration) + '.pth')
    # 迭代结束后进行weights的保存
    torch.save(ssd_net.state_dict(), args.save_folder+""+args.dataset+".pth")

def create_vis_plot(_xlabel, _ylabel, _title, _legend, viz):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def update_vis_plot(viz, iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


def xavier(param):
    # PS: 查找相关资料 xavier 也称作 Glorot initialisation 是一种权重的初始化方式
    init.xavier_uniform_(param)
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    train()