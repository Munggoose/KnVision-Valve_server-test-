"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import cv2

dirname = './output\\ganomaly/none\\test\\images'
for i, filenames in enumerate(os.listdir(dirname)):
    os.rename(dirname + '/' + filenames, dirname + '/' + str(i) + 'test.jpg')

