"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from ganomaly.lib.networks import NetG, NetD, weights_init
from ganomaly.lib.visualizer import Visualizer
from ganomaly.lib.loss import l2_loss
from ganomaly.lib.evaluate import evaluate

from ganomaly.lib.model import *

from ganomaly.options import Options
from ganomaly.lib.data import load_data
from ganomaly.lib.model import Ganomaly
from ganomaly.lib.casting_dataset import *
import torch

##
def test():
    """ Training
    """

    ##
    # ARGUMENTS
    print("set option")
    opt = Options().parse()
    ##
    # LOAD DATA
    print("get data")
    #dataset = Castingdataset(root='data', train=True)
    ##
    # LOAD MODEL
    print('set model')
    dataloader = load_data(opt)
    model = Ganomaly(opt, dataloader)
    ##
    # TRAIN MODEL
    print('start test')
    result = model.test()
    print(result)

def load_model(data_path):
    '''load pretrained model'''
    opt = Options().parse()
    opt.dataset = 'none'
    opt.dataroot = data_path


if __name__ == '__main__':
    test()