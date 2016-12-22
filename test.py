import os
import logging
import argparse
import mxnet as mx
from metric_factory import Perplexity
from data_iter_factory import read_data, PTBDataIter
from model import get_memnn

parser = argparse.ArgumentParser(description='Train a memory neural network')
parser.add_argument('--network', type=str, required=True,
                    help='network json file')
parser.add_argument('--params', type=str, required=True,
                    help='network params file')
parser.add_argument('--gpus', type=str, default='0',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size [128]')
parser.add_argument('--edim', type=int, default=150,
                    help='internal state dimension [150]')
parser.add_argument('--lindim', type=int, default=75,
                    help='linear part of the state [75]')
parser.add_argument('--nhop', type=int, default=6,
                    help='number of hops [6]')
parser.add_argument('--mem_size', type=int, default=100,
                    help='memory size [100]')
parser.add_argument('--data-dir', type=str, default='data',
                    help='data directory [data]')
parser.add_argument('--data-name', type=str, default='ptb',
                    help='data set name [ptb]')
args = parser.parse_args()

def test(args):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    count = []
    word2idx = {}

    train_data = read_data('%s/%s.train.txt' % (args.data_dir, args.data_name), count, word2idx)
    valid_data = read_data('%s/%s.valid.txt' % (args.data_dir, args.data_name), count, word2idx)
    test_data = read_data('%s/%s.test.txt' % (args.data_dir, args.data_name), count, word2idx)

    #  idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    nwords = len(word2idx)

    test_data_iter = PTBDataIter(test_data,
                                 nwords=nwords,
                                 batch_size=args.batch_size,
                                 edim=args.edim,
                                 mem_size=args.mem_size,
                                 init_hid=args.init_hid)

    ctx = mx.cpu() if args.gpus=='cpu' else [mx.gpu(int(i)) for i in args.gpus.split(',')]

    sym_memnn = mx.sym.load(args.network)

    arg_params = {}
    aux_params = {}
    for k,v in mx.nd.load(args.params).items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        elif tp == 'aux':
            aux_params[name] = v

    sym_memnn = get_memnn(edim=args.edim,
                          mem_size=args.mem_size,
                          nwords=nwords,
                          nhop=args.nhop,
                          lindim=args.lindim)
    mod_memnn = mx.mod.Module(symbol=sym_memnn,
                              data_names=('data', 'time', 'context'),
                              label_names=('target',),
                              context=ctx)
    mod_memnn.bind(data_shapes=test_data_iter.provide_data,
                   label_shapes=test_data_iter.provide_label)
    mod_memnn.init_params(arg_params=arg_params, aux_params=aux_params)

    eval_metric = []
    eval_metric.append(mx.metric.create('ce'))
    eval_metric.append(mx.metric.np(Perplexity))

    def print_metric(metrics):
        if isinstance(metrics, mx.metric.EvalMetric):
            metrics = [metrics]
        for m in metrics:
            name, value = m.get()
            logging.info('%s: %f' % (name, value))

    for preds, i_batch, batch in mod_memnn.iter_predict(test_data_iter):
        pred_label = preds[0].asnumpy().argmax(axis=1)
        label = batch.label[0].asnumpy().astype('int32')

        for metric in eval_metric:
            metric.update(label, pred_label)

        print_metric()

    print_metric()

if __name__ == '__main__':
    test(args)
