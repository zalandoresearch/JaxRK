
from typing import Callable

import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats

from jax.numpy import exp, log, sqrt
from jax.scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm

from jaxrk.utilities.eucldist import eucldist
from jaxrk.kern.base import DensityKernel, Kernel


def pos_param(x):
    return np.where(x >= 1, x, log(1 + exp(x - 1)))


def inv_pos_param(y):
    return np.where(y >= 1, y,  log(exp(y) - 1) + 1)

class StationaryKernel(Kernel):

    def __init__(self, sqdist_custom = True) -> None:
        """Base class for stationary kernels, depending only on

        sqdist = ‖x - x'‖^2

    Derived classes should implement either of:

        gram_sqdist(self, sqdist, logsp): Returns the kernel evaluated on sqdist,
        (squared un-scaled Euclidean distance), operating element-wise sqdist.

        gram_dist(self, dist, logsp): Returns the kernel evaluated on dist,
        (un-scaled Euclidean distance), operating element-wise dist.


        Args:
            sqdist_custom (bool, optional): Wether to use JaxRK custom euclidean distance computation. Defaults to True.
        """
        self.sqdist_custom = sqdist_custom

    def gram(self, X, Y=None, diag = False, logsp = False):
        assert(len(np.shape(X))==2)
        inp_dim = np.shape(X)[1]
        if diag:
            if Y is None:
                sq_dists = np.zeros(X.shape[0])
            else:
                assert(X.shape == Y.shape)
                sq_dists = np.sum((X - Y)**2, 1)
        else:
            sq_dists = eucldist(X, Y, power = 2.)
        return self.gram_sqdist(sq_dists, inp_dim, logsp = logsp)
        
    def gram_sqdist(self, sqdist, inp_dim, logsp = False):
        if hasattr(self, "gram_dist"):
            r = np.sqrt(np.clip(sqdist, 1e-36))
            return self.gram_dist(sqdist, inp_dim, logsp = logsp) 
        raise NotImplementedError


class GenGaussKernel(StationaryKernel): #this is the gennorm distribution from scipy
    def __init__(self, scale : np.array = np.array([1.]), shape :np.array = np.array([2.]), sqdist_custom = True):
        """Kernel derived from the pdf of the generalized Gaussian distribution (https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1).

        Args:
            scale (np.array, optional): Scale parameter. Defaults to np.array([1.]).
            shape (np.array, optional): Shape parameter in the half-open interval (0,2]. Lower values result in pointier kernel functions. Shape == 2 results in usual Gaussian kernel, shape == 1 results in Laplace kernel. Defaults to np.array([2.]).
        """
        super().__init__(sqdist_custom)
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

    def gram_sqdist(self, sqdist, inp_dim, logsp = False):            
        rval = self._const_factor* sqdist - self._log_norm * inp_dim
        if not logsp:
            return exp(rval)
        return rval

class GaussianKernel(DensityKernel, StationaryKernel):
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

    def gram_sqdist(self, sqdist, inp_dim, logsp = False):
        rval = self._const_factor* sqdist - self._log_norm * inp_dim
        if not logsp:
            return exp(rval)
        return rval

    def rvs(self, nrows, ncols):
        return norm.rvs(size = (nrows, ncols)) * self._sd

class LaplaceKernel(StationaryKernel):
    def __init__(self, sigma, diffable = False, sqdist_custom = True):
        super().__init__(sqdist_custom)
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

    def gram_dist(self, dist, inp_dim, logsp = False):
        rval = self._const_factor * dist - self._log_norm * inp_dim
        if not logsp:
            return exp(rval)
        return rval

    def rvs(self, nrows, ncols):
        return stats.laplace.rvs(scale=self._scale, size = (nrows, ncols))
