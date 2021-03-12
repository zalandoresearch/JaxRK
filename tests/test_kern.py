import copy

import jax.numpy as np
import pytest
from jax import random

from numpy.testing import assert_allclose

from jaxrk.kern import GaussianKernel, LaplaceKernel, GenGaussKernel, SplitDimsKernel
from jaxrk.rkhs import (Cdo, Cmo, CombVec, CovOp, FiniteMap, FiniteVec, SpVec,
                        inner)
from jaxrk.utilities.array_manipulation import all_combinations
from jaxrk.utilities.constraints import InitConst, Sigmoid

def test_rbf():
    key1, key2 = random.split(random.PRNGKey(0), 2)
    x = np.arange(8).reshape(4,2)
    gk = GaussianKernel(scale_init = (1)) # var: 1
    lk = LaplaceKernel(scale_init = 1) # var: 2
    ggk = GenGaussKernel(shape_init = (1.9)) #var: 0.52866
    for k in (gk, lk,  ggk):
        params = k.init(key2, x)
        print(k.apply(params, x), k.apply(params, method = k.var))


def test_SplitDimsKernel():
    (intervals, kernels) = ([0, 2, 5], [GaussianKernel(0.1), GaussianKernel(1)])
    X = np.arange(15).reshape((3,5))
    Y = (X + 3)[:-1,:]
    for op in lambda x: np.sum(x, 0), lambda x: np.prod(x, 0):
        k = SplitDimsKernel(intervals, kernels, op)
        assert(k(X, Y).shape == (len(X), len(Y)))
        assert(k(X).shape == (len(X), len(X)))
        assert(k(X, diag = True).shape == (len(X),))
