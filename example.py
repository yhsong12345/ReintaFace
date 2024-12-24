import torch
import argparse
from torchsummary import summary
import torchvision
from plots import *
from data.datasets import create_datasets, create_data_loaders
from utils.loss import *
from model.net import MobileNetV1
import os
from utils.engine import *
from model.retinaFace import RetinaFace
from data.prior_box import PriorBox
import time


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=300, help='number of epochs to train our network for')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=512, help='Batch Size')
parser.add_argument('-s', '--image_size', type=int, default=640, help='image size')
parser.add_argument('-m', '--model', nargs='+', type=str, default= 'mobilenet0.25', help='Model Selection')
parser.add_argument('-d', '-sav_dir', type=str, dest='save_dir', help='directory', default='outputs')
parser.add_argument('-f', '--files', type=str, default='./data')
args = vars(parser.parse_args())


# learning_parameters 
lr = args['learning_rate']
epochs = args['epochs']
BATCH_SIZE = args['batch_size']
s = args['image_size']
m = args['model']
d = args['save_dir']
f = args['files']

# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

model = RetinaFace(m, phase='train')

# model = MobileNetV1()
# model = model.to(device)

# output = summary(model, (3, 640, 640))
# print(model)
path = '.\data\celeba'
print(os.getcwd())
path = path.replace('.',f'{os.getcwd()}')
