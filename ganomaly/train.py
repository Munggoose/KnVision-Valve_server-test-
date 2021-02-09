"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly
from lib.casting_dataset import *
import torch

##
def train():
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
    print('start train')
    model.train()

if __name__ == '__main__':
    train()
