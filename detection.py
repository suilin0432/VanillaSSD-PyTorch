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
from data import
