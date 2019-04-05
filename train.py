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

