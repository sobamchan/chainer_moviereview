import chainer
from chainer import datasets
from chainer import optimizers
import chainer.links as L
import chainer.functions as F
from chainer import Variable

import numpy as np
from tqdm import tqdm

import argparse
import os

from sobamchan.sobamchan_chainer import Model
from sobamchan.sobamchan_moviereview import train
from sobamchan.sobamchan_chainer_link import PreTrainedEmbedId

class MLP(Model):
    def __init__(self, class_n, vocab_n, d, vocab, fpath):
        super(MLP, self).__init__(
            embed=PreTrainedEmbedId(vocab_n, d, vocab, fpath, False),
            fc1=L.Linear(None, 10),
            fc2=L.Linear(10, 10),
            fc3=L.Linear(10, class_n),
        )
    def __call__(self, x, t, train=True):
        x = self.fwd(x, train)
        return F.softmax_cross_entropy(x, t), F.accuracy(x, t)
    def fwd(self, x, train):
        h = self.embed(x)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h


if __name__ == '__main__':
    opts = {}
    opts['model'] = MLP
    opts['optimizer'] = optimizers.AdaGrad()
    train(opts)
