import os
import mxnet as mx
from model import get_memnn

if not os.path.exists('cache'):
    os.mkdir('cache')

sym = get_memnn(edim=150, mem_size=100, nwords=5000, nhop=1, lindim=150)
a = mx.viz.plot_network(sym,
                        save_format='png',
                        shape={'data': (128, 150), 'time': (128, 100), 'target': (128,), 'context': (128, 100)})
a.render('cache/network_structure')
