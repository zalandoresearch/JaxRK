# JaxRK

JaxRK is a library for working with (vectors of) RKHS elements and RKHS operators using [JAX](https://github.com/google/jax) for automatic differentiation. This library includes implementations of [kernel transfer operators](https://arxiv.org/abs/1712.01572) and [conditional density operators](https://arxiv.org/abs/1905.11255).

## Installation
First you have to make sure to have jax and jaxlib installed. Please follow the [JAX installation instructions](https://github.com/google/jax) depending on whether you want a CPU or GPU/TPU installation. After that you only need
```
$ pip install jaxrk
```

## Quick start examples

For some examples of what you can do with JaxRK, see [examples/Quick_start.ipynb](https://github.com/zalandoresearch/JaxRK/blob/master/examples/Quick_start.ipynb).


## Development

To help in developing JaxRK, clone the github repo and change to the cloned directory on the command line. Then 
```
$ pip install -e .
$ pytest test
```
will install the package into your python path. Changes to files in the directory are reflected in the python package when loaded.
