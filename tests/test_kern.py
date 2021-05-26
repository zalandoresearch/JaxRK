import copy
from jax._src.numpy.lax_numpy import linspace

import jax.numpy as np
import pytest
from jax import random

from numpy.testing import assert_allclose
from scipy.stats import norm, laplace, gennorm

from jaxrk.kern import GenGaussKernel, SplitDimsKernel
from jaxrk.rkhs import (Cdo, Cmo, CombVec, CovOp, inner)
from jaxrk.utilities.array_manipulation import all_combinations

def test_rbf():
    key1, key2 = random.split(random.PRNGKey(0), 2)
    x = np.arange(8).reshape(-1,1)
    tests = []
    for ls in np.linspace(0.9, 2., 5):
        gauss_ref = norm(loc = 0., scale=ls)
        lapl_ref = laplace(loc = 0., scale=ls)
 
        tests.append((f"Gauss_sc_{ls}",
                        GenGaussKernel.make_gauss(length_scale = ls), 
                        gauss_ref.var(),
                        gauss_ref.pdf(x)))
        tests.append((f"Lapl_sc_{ls}",
                        GenGaussKernel.make_laplace(length_scale = ls),
                        lapl_ref.var(),
                        lapl_ref.pdf(x)))
        for shape in np.linspace(0.9, 2., 5):
            gg_ref = gennorm(beta = shape, scale = ls)
            tests.append((f"GG__sh_{shape}__sc_{ls}",
                        GenGaussKernel.make(shape=shape, length_scale = ls),
                        gg_ref.var(),
                        gg_ref.pdf(x)))


    for (n, k, v, pdf) in tests:
        #print(n)
        assert np.abs(v - k.var()) < 1e-4, f"{n}: {v} != {k.var()}"
        g = k(x)
        assert np.allclose(pdf.squeeze(), g[0, :], atol=1e-1)


def test_SplitDimsKernel():
    (intervals, kernels) = ([0, 2, 5], [GenGaussKernel.make_gauss(0.1), GenGaussKernel.make_gauss(1)])
    X = np.arange(15).reshape((3,5))
    Y = (X + 3)[:-1,:]
    for op in lambda x: np.sum(x, 0), lambda x: np.prod(x, 0):
        k = SplitDimsKernel(intervals, kernels, op)
        assert(k(X, Y).shape == (len(X), len(Y)))
        assert(k(X).shape == (len(X), len(X)))
        assert(k(X, diag = True).shape == (len(X),))
