import chainer
from chainer import datasets
from chainer import optimizers
import chainer.links as L
import chainer.functions as F
from chainer import Variable

from sobamchan.sobamchan_chainer import Model
from sobamchan.sobamchan_chainer_link import PreTrainedEmbedId

class MLP(Model):
    def __init__(self, class_n, vocab_n, d, vocab, fpath):
        super(MLP, self).__init__(
            embed=PreTrainedEmbedId(vocab_n, d, vocab, fpath, False),
            fc1=L.Linear(None, 10),
            fc2=L.Linear(None, 10),
            fc3=L.Linear(None, class_n),
        )
    def __call__(self, x, t, train=True):
        x = self.fwd(x, train)
        return F.softmax_cross_entropy(x, t), F.accuracy(x, t)
    def fwd(self, x, train):
        h = self.embed(x)
        h = F.tanh(self.fc1(h))
        h = F.tanh(self.fc2(h))
        h = self.fc3(h)
        return h


class CNN(Model):

    def __init__(self, class_n, vocab_n, d, vocab, fpath):
        embed_learn=PreTrainedEmbedId(len(vocab),d,vocab,fpath,False)
        super(CNN, self).__init__(
            embed_learn=embed_learn,
            conv_f3=L.Convolution2D(2, 100, (3, d)),
            conv_f4=L.Convolution2D(2, 100, (4, d)),
            conv_f5=L.Convolution2D(2, 100, (5, d)),
            fc=L.Linear(None, class_n),
        )
        self.embedW = embed_learn.W
        self.embed_static = F.embed_id

    def __call__(self, x, t, train=True):
        y = self.fwd(x, train)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def fwd(self, x, train):
        embedW = self.embedW
        embed_h1 = self.embed_learn(x)
        embed_h2 = self.embed_static(x, embedW)
        b, embed_h, embed_w = embed_h1.shape
        embed_h1 = F.reshape(embed_h1, (b, 1, embed_h, embed_w))
        embed_h2 = F.reshape(embed_h2, (b, 1, embed_h, embed_w))
        embed_concat = F.concat((embed_h1, embed_h2))

        # conv
        h_f3 = F.relu(self.conv_f3(embed_concat))
        h_f4 = F.relu(self.conv_f4(embed_concat))
        h_f5 = F.relu(self.conv_f5(embed_concat))
        # pool
        _, _, h_f3_h, h_f3_w = h_f3.shape
        _, _, h_f4_h, h_f4_w = h_f4.shape
        _, _, h_f5_h, h_f5_w = h_f5.shape
        pooled_f3 = F.max_pooling_2d(h_f3, (h_f3_h, 1))
        pooled_f4 = F.max_pooling_2d(h_f4, (h_f4_h, 1))
        pooled_f5 = F.max_pooling_2d(h_f5, (h_f5_h, 1))
        batch, _, _, _ = pooled_f3.shape
        pooled_concat = F.concat((pooled_f3,pooled_f4,pooled_f5),3)
        h = F.reshape(pooled_concat, (batch, -1))
        h = F.dropout(self.fc(h), train=train)
        return h
