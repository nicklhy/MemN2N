import os
import logging
import argparse
import mxnet as mx
import numpy as np
from metric_factory import Perplexity
from data_iter_factory import read_data, PTBDataIter
from model import get_memnn

parser = argparse.ArgumentParser(description='Train a memory neural network')
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
parser.add_argument('--nepoch', type=int, default=100,
                    help='number of epoch to use during training [100]')
parser.add_argument('--init-lr', type=float, default=0.01,
                    help='initial learning rate [0.01]')
parser.add_argument('--lr-factor', type=float, default=1.0/1.5,
                    help='times the lr with a factor for every lr-factor-epoch epoch [0.666]')
parser.add_argument('--lr-factor-epoch', type=float, default=1.0,
                    help='the number of epoch to factor the lr, must be larger than 1 [None(1 epoch)]')
parser.add_argument('--init-hid', type=float, default=0.1,
                    help='initial internal state value [0.1]')
parser.add_argument('--init-std', type=float, default=0.05,
                    help='weight initialization std [0.05]')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='clip gradients to this norm [50]')
parser.add_argument('--checkpoint-step', type=int, default=10,
                    help='checkpoint step [10]')
parser.add_argument('--checkpoint-dir', type=str, default='checkpoint',
                    help='checkpoint directory [checkpoints]')
parser.add_argument('--data-dir', type=str, default='data',
                    help='data directory [data]')
parser.add_argument('--data-name', type=str, default='ptb',
                    help='data set name [ptb]')
parser.add_argument('--params', type=str, default=None,
                    help='pretrained model file')
args = parser.parse_args()

def train(args):
    ctx = [mx.cpu(0), mx.cpu(1)] if args.gpus=='cpu' else [mx.gpu(int(i)) for i in args.gpus.split(',')]

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    count = []
    word2idx = {}

    train_data = read_data('%s/%s.train.txt' % (args.data_dir, args.data_name), count, word2idx)
    valid_data = read_data('%s/%s.valid.txt' % (args.data_dir, args.data_name), count, word2idx)
    test_data = read_data('%s/%s.test.txt' % (args.data_dir, args.data_name), count, word2idx)

    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    args.nwords = len(word2idx)

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    train_data_iter = PTBDataIter(train_data,
                                  nwords=args.nwords,
                                  batch_size=args.batch_size,
                                  edim=args.edim,
                                  mem_size=args.mem_size,
                                  init_hid=args.init_hid)
    valid_data_iter = PTBDataIter(valid_data,
                                  nwords=args.nwords,
                                  batch_size=args.batch_size,
                                  edim=args.edim,
                                  mem_size=args.mem_size,
                                  init_hid=args.init_hid)

    lr_factor_step = int(args.lr_factor_epoch*len(train_data)/args.batch_size)
    #  opt = mx.optimizer.Adam(learning_rate=args.init_lr,
                            #  wd=0.0,
                            #  lr_scheduler=mx.lr_scheduler.FactorScheduler(lr_factor_step,
                                                                         #  args.lr_factor,
                                                                         #  stop_factor_lr=1e-5),
                            #  beta1=0.1,
                            #  clip_gradient=args.max_grad_norm)
    opt = mx.optimizer.SGD(learning_rate=args.init_lr,
                           momentum=0.0,
                           wd=0.0,
                           lr_scheduler=mx.lr_scheduler.FactorScheduler(lr_factor_step,
                                                                        args.lr_factor,
                                                                        stop_factor_lr=1e-7),
                           clip_gradient=args.max_grad_norm)
    #  opt = mx.optimizer.RMSProp(learning_rate=args.init_lr,
                               #  wd=0.00001,
                               #  lr_scheduler=mx.lr_scheduler.FactorScheduler(lr_factor_step,
                                                                            #  args.lr_factor,
                                                                            #  stop_factor_lr=1e-7),
                               #  clip_gradient=args.max_grad_norm)


    eval_metric = []
    #  eval_metric.append('ce')
    eval_metric.append(mx.metric.np(Perplexity))

    batch_end_callback = [mx.callback.Speedometer(args.batch_size, 50), ]

    def norm_stat(d):
        return mx.nd.norm(d)/np.sqrt(d.size)
    mon = mx.mon.Monitor(50, norm_stat, '.*backward.*')
    mon = None

    arg_params = {}
    aux_params = {}
    if args.params is not None:
        for k,v in mx.nd.load(args.params).items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                arg_params[name] = v
            elif tp == 'aux':
                aux_params[name] = v

    sym_memnn = get_memnn(edim=args.edim,
                          mem_size=args.mem_size,
                          nwords=args.nwords,
                          nhop=args.nhop,
                          lindim=args.lindim)
    mod_memnn = mx.mod.Module(symbol=sym_memnn,
                              data_names=('data', 'time', 'context'),
                              label_names=('target',),
                              context=ctx)
    mod_memnn.bind(data_shapes=train_data_iter.provide_data,
                   label_shapes=train_data_iter.provide_label)
    mod_memnn.init_params(initializer=mx.init.Normal(args.init_std),
                          arg_params=arg_params,
                          aux_params=aux_params,
                          allow_missing=True)
    mod_memnn.init_optimizer(optimizer=opt)

    mod_memnn.fit(train_data_iter,
                  valid_data_iter,
                  eval_metric=eval_metric,
                  batch_end_callback=batch_end_callback,
                  epoch_end_callback=mx.callback.do_checkpoint(os.path.join(args.checkpoint_dir, 'memnn'), period=args.checkpoint_step),
                  num_epoch=args.nepoch,
                  monitor=mon)


if __name__ == '__main__':
    train(args)
