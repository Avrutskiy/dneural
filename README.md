# dneural

**For high-order derivative training, see [cPDE](https://github.com/Avrutskiy/cPDE).**
cPDE provides an easy-to-use Python interface to efficient forward and backward
propagation for fully connected multilayer perceptrons (MLPs), with CUDA and CPU
backends and training with input derivatives through sixth order, including mixed
derivatives. It includes documented function-approximation and PDE examples, plus
[performance benchmarks](https://github.com/Avrutskiy/cPDE#function-approximation-benchmark).

An minimal example of training a neural network and it's first derivatives. The target is a 2D function created by a random Fourier series. See "nn_research_gpu.ipynb".

An interesting effect related to overfitting can be observed.

Illustrates the paper by V.I. Avrutskiy "Enhancing function approximation abilities of neural networks by training derivatives", courtesy of K.Rybakov
