"""
Created on Thu Jan 10 10:01:56 2019

@author: Ingmar Schuster
"""


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


def median_heuristic(data, distance, per_dimension = True):
    if isinstance(distance, str):
        dist_fn = lambda x: pdist(x, distance)
    else:
        dist_fn = distance
    if per_dimension is False:
        return np.median(dist_fn(data))
    else:
        def single_dim_heuristic(data_dim):
            return median_heuristic(data_dim[:, None], dist_fn, per_dimension = False)
        return np.apply_along_axis(single_dim_heuristic, 0, data)

class Kernel(object):
    """A generic kernel type."""
    def __call__(self, *args, **kwargs):
        return self.gram(*args, **kwargs)

    def gram(self, X, Y = None, diag = False) -> np.array:
        """Compute the gram matrix, i.e. the kernel evaluated at every element of X paired with each element of Y (if not None, otherwise each element of X).

        Args:
            X: input space points, one per row.
            Y: input space points, one per row. If none, default to Y = X.
            diag: if `True`, compute only the diagonal elements of the gram matrix.

        Returns:
            The gram matrix or its diagonal, depending on passed parameters."""
        raise NotImplementedError()

    def get_params(self) -> np.array:
        """Get parameters of this particular kernel.

        Returns: Unconstrained parameteres as a flat numpy array."""
        assert()

    def set_params(self, params : np.array) -> None:
        """Set unconstrained parameters, possibly after transformin them.

        Args:
            params: Unconstrained parameters.
        """
        assert()

class DensityKernel(Kernel):
    """Type for positive definite kernels that are also densities."""
    def rvs(self, nsamps):
        raise NotImplementedError()    


class FeatMapKernel(Kernel):
    """A kernel that is defined by a feature map.
    
    Args:
        feat_map: A callable that computes the feature map, i.e. given input space points it returns real valued features, one per input space point."""

    def __init__(self, feat_map:Callable):
        self.features = feat_map

    def features_mean(self, samps):
        return self.features(samps).mean(0)

    def gram(self, X, Y = None, diag = False):
        f_X = self.features(X)
        if Y is None:
            f_Y = f_X
        else:
            f_Y = self.features(Y)
        if diag:
            return np.sum(f_X * f_Y, 1)
        else:
            return f_X.dot(f_Y.T)


class LinearKernel(FeatMapKernel):
    def __init__(self):
        FeatMapKernel.__init__(self, lambda x: x)


def pos_param(x):
    return np.where(x >= 1, x, log(1 + exp(x - 1)))


def inv_pos_param(y):
    return np.where(y >= 1, y,  log(exp(y) - 1) + 1)


class GenGaussKernel: #this is the gennorm distribution from scipy
    def __init__(self, sigma = np.array([1.]), power = np.array([2.])):
        assert(np.all(sigma > 0))
        assert(np.all(power >= 0))
        assert(np.all(power <= 2))
        self.sigma = sigma
        self.__set_standard_dev(sigma)
        self.power = power


    def __set_standard_dev(self, sd):
        assert(np.all(sd > 0))
        self._sd = sd
        self.var = sd**2
        self._const_factor = -0.5 / sd**2
        self._normalization = (sqrt(2*np.pi)*sd)
        self._log_norm = log(self._normalization)

    def get_var(self):
        return self._sd**2

    def gram(self, X, Y=None, diag = False, logsp = False):
        assert(len(np.shape(X))==2)

        # if X=Y, use more efficient pdist call which exploits symmetry

        if diag:
            if Y is None:
                sq_dists = np.zeros(X.shape[0])
            else:
                assert(X.shape == Y.shape)
                sq_dists = np.sum((X - Y)**2, 1)
        else:
            sq_dists = eucldist(X, Y, power = self.power)
            
        rval = self._const_factor* sq_dists - self._log_norm * np.shape(X)[1]
        if not logsp:
            return exp(rval)
        return rval

class GaussianKernel(DensityKernel):
    def __init__(self, sigma = np.array([1]), diffable = False):
        self.set_params(pos_param(sigma))
        self.diffable = diffable

    def get_params(self):
        return self.params

    def get_double_var_kern(self):
        return GaussianKernel(np.sqrt(2) * self._sd)

    def set_params(self, params):
        self.params = np.atleast_1d(params).flatten()[0]
        self.__set_standard_dev(pos_param(self.params))

    def __set_standard_dev(self, sd):
        assert(np.all(sd > 0))
        self._sd = sd
        self.var = sd**2
        self._const_factor = -0.5 / sd**2
        self._normalization = (sqrt(2*np.pi)*sd)
        self._log_norm = log(self._normalization)

    def get_var(self):
        return self._sd**2

    def gram(self, X, Y=None, diag = False, logsp = False):
        assert(len(np.shape(X))==2)

        # if X=Y, use more efficient pdist call which exploits symmetry

        if diag:
            if Y is None:
                sq_dists = np.zeros(X.shape[0])
            else:
                assert(X.shape == Y.shape)
                sq_dists = np.sum((X - Y)**2, 1)
        else:
            sq_dists = eucldist(X, Y, power = 2)
            
        rval = self._const_factor* sq_dists - self._log_norm * np.shape(X)[1]
        if not logsp:
            return exp(rval)
        return rval

    def rvs(self, nrows, ncols):
        return np.random.randn(nrows, ncols) * self._sd

class LaplaceKernel(DensityKernel):
    def __init__(self, sigma, diffable = False):
        self.set_params(log(exp(sigma) - 1))
        self.diffable = diffable

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = np.atleast_1d(params).flatten()[0]
        self.__set_standard_dev(log(exp(self.params) + 1))

    def __set_standard_dev(self, sd):
        self._sd = sd
        self._scale = sd
        self._const_factor = -1./self._scale
        self._normalization = 2 * self._scale
        self._log_norm = log(self._normalization)

    def get_var(self):
        return 2 * self._scale**2

    def gram(self, X, Y=None, diag = False, logsp = False):
        assert(len(np.shape(X))==2)

        # if X=Y, use more efficient pdist call which exploits symmetry

        if diag:
            if Y is None:
                dists = np.zeros(X.shape[0])
            else:
                assert(X.shape == Y.shape)
                dists = np.sum((X - Y)**2, 1)
        else:
            dists = sq_dists = eucldist(X, Y, power = 1.)

        rval = self._const_factor * dists - self._log_norm * np.shape(X)[1]
        if not logsp:
            return exp(rval)
        return rval

    def rvs(self, nrows, ncols):
        return stats.laplace.rvs(scale=self._scale, size = (nrows, ncols))


class PeriodicKernel(Kernel):
    def __init__(self, period, lengthscale, ):
        #self.set_params(log(exp(np.array([s2,df])) - 1))
        self.ls = lengthscale
        self.period = period
        self.diffable = False


    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

    def gram(self, X, Y=None, diag = False, logsp = False):
        assert(len(np.shape(X))==2)

        # if X=Y, use more efficient pdist call which exploits symmetry

        if diag:
            assert()
        else:
            dists = eucldist(X, Y, power = 1.)
        assert(not logsp)
        return exp(- 2* np.sin(np.pi*dists/self.period)**2 / self.ls**2)


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
