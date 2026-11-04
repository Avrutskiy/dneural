# dneural

**For high-order derivative training, see [cPDE](https://github.com/Avrutskiy/cPDE).**
cPDE provides an easy-to-use Python interface to efficient forward and backward
propagation for fully connected multilayer perceptrons (MLPs), with CUDA and CPU
backends and training with input derivatives through sixth order, including mixed
derivatives. It includes documented function-approximation and PDE examples, plus
[performance benchmarks](https://github.com/Avrutskiy/cPDE#function-approximation-benchmark).

**This repository** contains a minimal example of training a neural network to
approximate a function and its first derivatives. The target is a two-dimensional
function defined by a random Fourier series. See
[nn_research_gpu.ipynb](nn_research_gpu.ipynb).

Across 50 independent weight initializations of this example, the function-value
RMSE was **lower on the test grid** than on the training grid in every run after
1,000 epochs. The median reduction was 10.0%, ranging from 2.0% to 16.4%, while the
combined value-and-derivative cost was higher on the test grid in all 50 runs.
Training balances errors in values and derivatives, so minimizing their sum does
not impose the same train–test ordering on each component.

The example illustrates V. I. Avrutskiy's paper,
[“Enhancing Function Approximation Abilities of Neural Networks by Training
Derivatives”](https://doi.org/10.1109/TNNLS.2020.2979706), and is provided courtesy of K. Rybakov.
