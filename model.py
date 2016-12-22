import mxnet as mx

def get_memnn(edim, mem_size, nwords, nhop, lindim):
    data = mx.sym.Variable('data', shape=(-1, edim))
    time = mx.sym.Variable('time', shape=(-1, mem_size))
    target = mx.sym.Variable('target', shape=(-1, nwords))
    context = mx.sym.Variable('context', shape=(-1, mem_size))

    hid = []
    hid.append(data)
    share_list = []
    share_list.append([])

    #  build memory
    A = mx.sym.Variable('A_weight', shape=(nwords, edim))
    B = mx.sym.Variable('B_weight', shape=(nwords, edim))
    C = mx.sym.Variable('C_weight', shape=(edim, edim))
    #  Temporal Encoding
    T_A = mx.sym.Variable('TA_weight', shape=(mem_size, edim))
    T_B = mx.sym.Variable('TB_weight', shape=(mem_size, edim))
    # m_i = sum A_ij * x_ij + T_A_i
    Ain_c = mx.sym.Embedding(data=context,
                             weight=A,
                             input_dim=nwords,
                             output_dim=edim)
    Ain_t = mx.sym.Embedding(data=time,
                             weight=T_A,
                             input_dim=mem_size,
                             output_dim=edim)
    Ain = Ain_c+Ain_t
    # c_i = sum B_ij * u + T_B_i
    Bin_c = mx.sym.Embedding(data=context,
                             weight=B,
                             input_dim=nwords,
                             output_dim=edim)
    Bin_t = mx.sym.Embedding(data=time,
                             weight=T_B,
                             input_dim=mem_size,
                             output_dim=edim)
    Bin = Bin_c+Bin_t

    for h in xrange(nhop):
        #  hid3dim = mx.sym.expand_dims(data=hid[-1], axis=1)
        Aout = mx.sym.Flatten(mx.sym.batch_dot(lhs=mx.sym.expand_dims(hid[-1], axis=1),
                                               rhs=mx.sym.transpose(Ain, axes=(0, 2, 1))))
        P = mx.sym.SoftmaxActivation(Aout, name='P')
        Bout = mx.sym.Flatten(mx.sym.batch_dot(lhs=mx.sym.expand_dims(P, axis=1),
                                               rhs=Bin))
        Cout = mx.sym.dot(hid[-1], C)
        Dout = Cout+Bout
        share_list[0].append(Cout)

        if lindim == edim:
            hid.append(Dout)
        elif lindim == 0:
            hid.append(mx.sym.Activation(data=Dout, act_type='relu'))
        else:
            F = mx.sym.Crop(mx.sym.Reshape(Dout,
                                           shape=(-1, 1, 1, edim)),
                            num_args=1,
                            offset=(0, 0),
                            h_w=(1, lindim))
            F = mx.sym.Reshape(F, shape=(-1, lindim))
            G = mx.sym.Crop(mx.sym.Reshape(Dout,
                                           shape=(-1, 1, 1, edim)),
                            num_args=1,
                            offset=(0, lindim),
                            h_w=(1, edim-lindim))
            G = mx.sym.Reshape(G, shape=(-1, edim-lindim))
            K = mx.sym.Activation(data=G, act_type='relu')
            hid.append(mx.sym.Concat(*[G, K], num_args=2, axis=1))
        clf = mx.sym.FullyConnected(data=hid[-1],
                                    num_hidden=nwords,
                                    no_bias=True,
                                    name='clf')
        loss = mx.sym.SoftmaxOutput(data=clf, label=target, name='prob')
        return loss

if __name__ == '__main__':
    sym = get_memnn(edim=150, mem_size=100, nwords=5000, nhop=6, lindim=75)
