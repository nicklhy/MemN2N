End-To-End Memory Networks in MemN2N
========================================

MXNet implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895v4) for language modelling. The original Tensorflow code from [carpedm20](https://github.com/carpedm20) can be found [here](https://github.com/carpedm20/MemN2N-tensorflow).

![alt tag](http://i.imgur.com/nv89JLc.png)

**Known issue: SGD does not converge, ADAM converges but is not able to reach a good result(![details]).**

Setup
--------------

This code requires [MXNet](https://github.com/dmlc/mxnet). Also, it uses CUDA to run on GPU for faster training. There is a set of sample Penn Tree Bank (PTB) corpus in `data` directory, which is a popular benchmark for measuring quality of these models. But you can use your own text data set which should be formated like [this](data/).

Usage
--------------

To train a model with 6 hops and memory size of 100, run the following command:

    $ python train.py --nhop 6 --mem_size 100

To see all training options, run:

    $ python train.py --help

To test a model, run the script file test.py like:

    $ python test.py --network checkpoint/memnn-symbol.json --params checkpoint/memnn-0100.params --gpus 0
