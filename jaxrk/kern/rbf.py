
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


def pos_param(x):
    return np.where(x >= 1, x, log(1 + exp(x - 1)))


def inv_pos_param(y):
    return np.where(y >= 1, y,  log(exp(y) - 1) + 1)


class GenGaussKernel(Kernel): #this is the gennorm distribution from scipy
    def __init__(self, scale : np.array = np.array([1.]), shape :np.array = np.array([2.])):
        """Kernel derived from the pdf of the generalized Gaussian distribution (https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1).

        Args:
            scale (np.array, optional): Scale parameter. Defaults to np.array([1.]).
            shape (np.array, optional): Shape parameter in the half-open interval (0,2]. Lower values result in pointier kernel functions. Shape == 2 results in usual Gaussian kernel, shape == 1 results in Laplace kernel. Defaults to np.array([2.]).
        """
        assert(np.all(scale > 0))
        assert(np.all(shape >= 0))
        assert(np.all(shape <= 2))
        self.scale = scale
        self.__set_standard_dev(scale)
        self.shape = shape


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
            sq_dists = eucldist(X, Y, power = self.shape)
            
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
