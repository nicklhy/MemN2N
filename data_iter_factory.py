import os
import random
import mxnet as mx
import numpy as np
from collections import Counter

def read_data(fname, count, word2idx):
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise("[!] Data %s not found" % fname)

    words = []
    for line in lines:
        words.extend(line.split())

    if len(count) == 0:
        count.append(['<eos>', 0])

    count[0][1] += len(lines)
    count.extend(Counter(words).most_common())

    if len(word2idx) == 0:
        word2idx['<eos>'] = 0

    for word, _ in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    data = list()
    for line in lines:
        for word in line.split():
            index = word2idx[word]
            data.append(index)
        data.append(word2idx['<eos>'])

    print("Read %s words from %s" % (len(data), fname))
    return data


class PTBDataIter(mx.io.DataIter):
    def __init__(self,
                 data,
                 nwords,
                 batch_size=64,
                 edim=150,
                 mem_size=100,
                 init_hid=0.1):
        super(PTBDataIter, self).__init__()
        self.data = data
        self.nwords = nwords
        self.batch_size = batch_size
        self.edim = edim
        self.mem_size = mem_size
        self.init_hid = init_hid

        self.num_data = len(data)
        self.cursor = -self.batch_size

        self.provide_data = [('data', (self.batch_size, self.edim)),
                             ('time', (self.batch_size, self.mem_size)),
                             ('context', (self.batch_size, self.mem_size))]
        #  self.provide_label = [('target', (self.batch_size, self.nwords)),]
        self.provide_label = [('target', (self.batch_size,)),]

        self.x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        self.time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        #  self.target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded
        self.target = np.zeros([self.batch_size,])
        self.context = np.ndarray([self.batch_size, self.mem_size])

    def reset(self):
        self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size

        self.x.fill(self.init_hid)
        for t in xrange(self.mem_size):
            self.time[:,t].fill(t)

        self.target.fill(0)
        for b in xrange(self.batch_size):
            m = random.randrange(self.mem_size, len(self.data))
            #  self.target[b][self.data[m]] = 1
            self.target[b] = self.data[m]
            self.context[b] = self.data[m - self.mem_size:m]

        return self.cursor < self.num_data

    def getdata(self):
        return map(mx.nd.array, [self.x, self.time, self.context])

    def getlabel(self):
        return [mx.nd.array(self.target)]

    def getpad(self):
        if self.cursor+self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0
