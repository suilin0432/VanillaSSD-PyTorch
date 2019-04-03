from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from SSDModel import build_ssd

parser = argparse.ArgumentParser(description="Vanilla SSD Pytorch")
parser.add_argument("--trained_model", default="weights/ssd300_mAP_77.43_v2.pth", type=str, help="SSD的预训练.pth权重文件")
parser.add_argument("--save_folder", default="eval/", type=str, help="存储结果的路径")
parser.add_argument("--visual_threshold", default=0.6, type=float, help="最终展现在图片上要求的detection置信系数")
parser.add_argument("--cuda", default=True, type=bool, help="是否使用CUDA进行加速")
parser.add_argument("--voc_root", default=VOC_ROOT, help="VOC数据库的根文件目录")
parser.add_argument("-f", default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def test_net(save_folder, net, cuda, testset, transform, thresh):

    # 记录结果的文件名
    filename = save_folder+"test1.txt"
    # 获取测试图片数量
    num_images = len(testset)
    for i in range(num_images):
        print("Testing image {:d}/{:d}...".format((i+1), num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        # transform 进行图像的输入变换
        # 测试的时候只是简单的进行了一下变换而已
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1).contiguous()

        with open(filename, mode="a") as f:
            f.write("\nGROUND TRUTH FOR: "+img_id+"\n")
            for box in annotation:
                f.write("label: "+" || ".join(str(b) for b in box) + "\n")

        if cuda:
            x = x.cuda()

        y = net(x)
        # 获取到检测结果
        detections = y.data

        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        pred_num = 0

        # 因为按照代码, 一次是只有一张图片输入的, 所以不用对每个batch进行遍历, 默认batch那个维度为0即可
        # PS: 我在这里把返回函数改了 所以不应该这么写...
        # for i in range(detections.size(1)):
        j = 0
        # 小于 0.6 的被忽视了
        while detections[j, 0] >= 0.6:
            if pred_num == 0:
                with open(filename, mode="a") as f:
                    f.write("PREDICTIONS: \n")
            score = detections[j, 0]
            #
            label_name = labelmap[scale[j, 5]]
            # 取出来所有的预测值 然后放大到原来的尺寸
            # 因为预测的值经过编码(encode)解码(decode)之后得到的仍然是相对于原图大小的比例，而原图进行resize好像 SSD并没又进行 像Yolo, fasterRcnn 一样的padding
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3])
            pred_num += 1
            with open(filename, mode = "a") as f:
                f.write(str(pred_num) + ' label: ' + label_name + ' score: ' +
                        str(score) + ' ' + ' || '.join(str(c) for c in coords) + '\n')
            j += 1

def test_voc():
    os.environ["CUDA_VISIBLE_DEVICES"] = [0]
    num_classes = len(VOC_CLASSES) + 1
    net = build_ssd("test", 300, num_classes)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print("Finished loading model!")

    testset = VOCDetection(args.voc_root, [("2007", "test")], None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        net = nn.DataParallel(net)
        cudnn.benchmark = True

    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)
if __name__ == "__main__":
    test_voc()
