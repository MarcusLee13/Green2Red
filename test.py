import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
# parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
# parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
# 测试数据集需要自己去写新的名称
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)



# 很重要的参数
input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
# 模型定义，很重要，不能删除，实例化了残差生成网络
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

# 这里是将网络参数输入cuda，很重要
if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()


G_AB.load_state_dict(torch.load("saved_models/yingguangseg/G_AB_195.pth"))
G_BA.load_state_dict(torch.load("saved_models/yingguangseg/G_BA_195.pth"))
D_A.load_state_dict(torch.load("saved_models/yingguangseg/D_A_195.pth"))
D_B.load_state_dict(torch.load("saved_models/yingguangseg/D_B_195.pth"))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


# Test data loader
val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=4,
    shuffle=False,
    num_workers=1,
)

# 测试的重要部分，后面运用到的时候就直接放一个batch
def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)
    # Arange images along x-axis, 4张图形刚好为一张，test A里面放绿的， test B里面放红的。一个dataset里面只有test有放的东西。
    real_A = make_grid(real_A, nrow=4, normalize=True)
    real_B = make_grid(real_B, nrow=4, normalize=True)
    fake_A = make_grid(fake_A, nrow=4, normalize=True)
    fake_B = make_grid(fake_B, nrow=4, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)

batches_done = 0

sample_images(batches_done)
