
from typing import Callable

import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats
from jax.experimental.vectorize import vectorize
from jax.numpy import exp, log, sqrt
from jax.scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from jaxrk.utilities.eucldist import eucldist
from jaxrk.kern.base import DensityKernel, Kernel



class SplitDimsKernel(Kernel):
    def __init__(self, intervals, kernels, operation = "*", weights = None):
        assert(len(intervals) - 1 == len(kernels))
        self.intervals = intervals
        self.kernels = kernels
        if operation == "*":
            self.weights = np.ones(len(kernels))
            self.op = lambda x: np.prod(x, 0)
            self.log_op = lambda x: np.sum(x, 0)
            if weights is None:
                self.weights = np.ones(len(kernels))
            else:
                self.weights = weights
        else:
            assert(operation == '+')
            self.op = lambda x: np.sum(x, 0)
            self.log_op = lambda x: logsumexp(x, 0)
            if weights is None:
                self.weights = np.ones(len(kernels))
            else:
                self.weights = weights
        
    def gram(self, X, Y=None, diag = False, logsp = False):
        assert(not logsp)
        
        split_X = [X[:, self.intervals[i]:self.intervals[i + 1]] for i in range(len(self.kernels))]
        if Y is None:
            split_Y = [None] * len(self.kernels)
        else:
            split_Y = [Y[:, self.intervals[i]:self.intervals[i + 1]] for i in range(len(self.kernels))]
        sub_grams = np.array([self.kernels[i].gram(split_X[i], split_Y[i], diag = diag) * self.weights[i]  for i in range(len(self.kernels))])
        return self.op(sub_grams)

def test_SplitDimsKernel():
    (intervals, kernels) = ([0, 2, 5], [GaussianKernel(0.1), GaussianKernel(1)])
    X = np.arange(15).reshape((3,5))
    Y = (X + 3)[:-1,:]
    for op in "+", "*":
        k = SplitDimsKernel(intervals, kernels, op)
        assert(k.gram(X, Y).shape == (len(X), len(Y)))
        assert(k.gram(X).shape == (len(X), len(X)))
        assert(k.gram(X, diag = True).shape == (len(X),))
                

class SKlKernel(Kernel):
    def __init__(self, sklearn_kernel):
        self.skl = sklearn_kernel

    def gram(self, X, Y=None, diag = False, logsp = False):
        if diag:
            assert(Y is None)
            rval = self.skl.diag(X)
        else:
            rval = self.skl(X, Y)
        if logsp:
            rval = log(rval)
        return rval