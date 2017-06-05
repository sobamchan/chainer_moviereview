import chainer
from chainer import datasets
from chainer import optimizers
import chainer.links as L
import chainer.functions as F
from chainer import Variable

import numpy as np
from tqdm import tqdm

from sobamchan.sobamchan_moviereview import train

import argparse
import os, sys
# sys.path.remove('/Users/sochan/project/sobamchan')

from model import MLP, CNN


if __name__ == '__main__':
    opts = {}
    opts['model'] = CNN
    opts['optimizer'] = optimizers.AdaGrad(lr=.0001)
    train(opts)
