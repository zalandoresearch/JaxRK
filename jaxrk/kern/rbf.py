
from typing import Callable, Tuple, Union, Optional
from jaxrk.typing import PRNGKeyT, Shape, Dtype, Array
from jaxrk.utilities.constraints import SoftPlus, Bijection, ConstInit, Sigmoid
from dataclasses import dataclass
from flax.linen.module import compact, Module
import flax.linen as ln

import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats
import flax.linen as ln
from jax.random import PRNGKey

from jax.numpy import exp, log, sqrt
from jax.scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm

from jaxrk.utilities.eucldist import eucldist
from jaxrk.kern.base import DensityKernel, Kernel

class NoScaler(ln.Module):
    def __call__(self, inp):
        return inp
    
    def inv(self):
        return 1.
    
    def scale(self):
        return 1.

class Scaler(ln.Module):
    bij: Bijection = SoftPlus()
    init: Callable[[PRNGKeyT, Shape, Dtype, Bijection], Array] = ConstInit(np.ones(1))
    scale_shape: Shape = (1, 1)


    def setup(self):
        self.s = self.bij(self.param("s", self.init, self.scale_shape, np.float32, self.bij))

    def inv(self):
        return 1./self.s
    
    def scale(self):
        return self.s

    def __call__(self, inp):
        assert len(inp.shape) == len(self.scale_shape)
        
        return self.s * inp





class ScaledPairwiseDistance(ln.Module):
    """A class for computing scaled pairwise distance for stationary/RBF kernels, depending only on

        dist = ‖x - x'‖^p

    For some power p.
    """

    scale:bool = True
    power:float = 2.
    scale_dim:int = 1
    scale_init:Callable[[PRNGKeyT, Shape, Dtype, Bijection], Array] = ConstInit(np.ones(1))

    def setup(self,):
        assert self.scale_dim >= 1
        if not self.scale:
            self.ds = self.gs = NoScaler()
        else:
            if self.scale_dim > 0:
                self.gs = Scaler(init = self.scale_init)
                self.ds = NoScaler()
            else:
                self.gs = NoScaler()
                self.ds = Scaler(scale_shape = (1, self.scale_dim), init = self.scale_init)  


    
    def _get_scale_param(self):
        if self.scale_dim > 1:
            return self.gs.inv()**(1./self.power)
        else:
            return self.ds.inv()

    def __call__(self, X, Y=None, diag = False,):
        assert len(X.shape) == 2
      

        if diag:
            if Y is None:
                dists = np.zeros(X.shape[0])
            else:
                assert(X.shape == Y.shape)
                dists = self.gs(np.sum(np.abs(self.ds(X) - self.ds(Y))**self.power, 1))
        else:
            sY = None
            if Y is not None:
                sY = self.ds(Y)
            dists = self.gs(eucldist(self.ds(X), sY, power = self.power))
        return dists


class GenGaussKernel(Kernel, ln.Module): #this is the gennorm distribution from scipy
    """Kernel derived from the pdf of the generalized Gaussian distribution (https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1).

        Args:
            scale (np.array, optional): Scale parameter. Defaults to np.array([1.]).
            shape (np.array, optional): Shape parameter in the half-open interval (0,2]. Lower values result in pointier kernel functions. Shape == 2 results in usual Gaussian kernel, shape == 1 results in Laplace kernel. Defaults to np.array([2.]).
    """
    per_dim_scale:bool = False
    input_dim:int = None
    scale_init:Callable[[PRNGKeyT, Shape, Dtype, Bijection], Array] = ConstInit(np.ones(1))

    shape_bij: Bijection = Sigmoid(0., 2.)
    shape_init: Callable[[PRNGKeyT, Shape, Dtype, Bijection], Array] = ConstInit(np.ones(1))

#   FIXME: it also works to declare
#
#   shape: Union[Callable[[PRNGKeyT, Shape, Dtype, Bijection], Array], float, Array] = ConstInit(np.ones(1))
#
#   and then setup the class with
    # if isinstance(self.shape, Callable[[PRNGKeyT, Shape, Dtype, Bijection], Array]):
    #     #initialization function
    #     self.shape_val = self.shape_bij(self.param("shape", self.shape_init, (1,), np.float32, self.shape_bij))
    # else:
    #     #fixed float or array
    #     assert np.all(self.shape > 0)
    #     assert np.all(self.shape <= 2)
    #     self.shape_val = self.shape


    def setup(self):
        scale_dim = 1
        if self.per_dim_scale:
            assert self.dim is not None, "If scaling per dimension, input dimension must be provided"
            scale_dim = self.dim            
        self.shape = self.shape_bij(self.param("shape", self.shape_init, (1,), np.float32, self.shape_bij))
        self.dist = ScaledPairwiseDistance(power = self.shape, scale_dim = scale_dim, scale_init = self.scale_init)

    def get_sd(self):
        return np.sqrt(self.get_var())
    
    def get_var(self):
        f = np.exp(sp.special.gammaln(np.array([3, 1]) / self.shape))
        return self.dist._get_scale_param()**2 * f[0] / f[1]

    def __call__(self, X, Y=None, diag = False,):
        return exp(-self.dist(X, Y, diag))


class GaussianKernel(Kernel, ln.Module):
    per_dim_scale:bool = False
    input_dim:int = None

    scale_init:Callable[[PRNGKeyT, Shape, Dtype, Bijection], Array] = ConstInit(np.ones(1))

    def setup(self):
        scale_dim = 1
        if self.per_dim_scale:
            assert self.dim is not None, "If scaling per dimension, input dimension must be provided"
            scale_dim = self.dim
        self.dist = ScaledPairwiseDistance(power = 2., scale_dim = scale_dim, scale_init = self.scale_init)

    def get_sd(self):
        return self.dist._get_scale_param()
    
    def get_var(self):
        return self.get_sd()**2

    def __call__(self, X, Y=None, diag = False,):
        return exp(-self.dist(X, Y, diag) / 2)


class LaplaceKernel(Kernel, ln.Module):
    per_dim_scale:bool = False
    input_dim:int = None
    scale_init:Callable[[PRNGKeyT, Shape, Dtype, Bijection], Array] = ConstInit(np.ones(1))

    def setup(self):
        scale_dim = 1
        if self.per_dim_scale:
            assert self.dim is not None, "If scaling per dimension, input dimension must be provided"
            scale_dim = self.dim
        self.dist = ScaledPairwiseDistance(power = 1., scale_dim = scale_dim, scale_init = self.scale_init)


    def get_sd(self):
        return np.sqrt(2) * self.dist._get_scale_param()
    
    def get_var(self):
        return self.get_sd()**2

    def __call__(self, X, Y=None, diag = False,):
        return exp(-self.dist(X, Y, diag))