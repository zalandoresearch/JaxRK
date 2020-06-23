
from typing import Callable

import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats
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