
from typing import Callable, Tuple, Union, Optional
from jaxrk.core.typing import PRNGKeyT, Shape, Dtype, Array, ConstOrInitFn
from functools import partial
from jaxrk.core.constraints import SoftPlus, Bijection, Sigmoid
from jaxrk.core.init_fn import ConstFn
from dataclasses import dataclass
from flax.linen.module import compact
from jaxrk.core import Module
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
from jaxrk.kern.util import ScaledPairwiseDistance


class GenGaussKernel(DensityKernel, Module, ln.Module): #this is the gennorm distribution from scipy
    """Kernel derived from the pdf of the generalized Gaussian distribution (https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1).

        Args:
            scale (np.array, optional): Scale parameter. Defaults to np.array([1.]).
            shape (np.array, optional): Shape parameter in the half-open interval (0,2]. Lower values result in pointier kernel functions. Shape == 2 results in usual Gaussian kernel, shape == 1 results in Laplace kernel. Defaults to np.array([2.]).
    """    
    scale_init:ConstOrInitFn = ConstFn(np.ones(1))
    shape_init: ConstOrInitFn = ConstFn(np.ones(1))

    dim:int = None
    per_dim_scale:bool = False    
    
    shape_bij: Bijection = Sigmoid(0., 2.)

    def setup(self):
        scale_dim = 1
        if self.per_dim_scale:
            assert self.dim is not None, "If scaling per dimension, input dimension must be provided"
            scale_dim = self.dim            
        self.shape = self.shape_bij(self.const_or_param("shape", self.shape_init, (1,), np.float32, self.shape_bij))
        self.dist = ScaledPairwiseDistance(power = self.shape, scale_dim = scale_dim, scale_init = self.scale_init)

    def std(self):
        return np.sqrt(self.var())
    
    def var(self):
        f = np.exp(sp.special.gammaln(np.array([3, 1]) / self.shape))
        return self.dist._get_scale_param()**2 * f[0] / f[1]

    def __call__(self, X, Y=None, diag = False,):
        return exp(-self.dist(X, Y, diag))


GaussianKernel = partial(GenGaussKernel, shape_init = 2.)
LaplaceKernel = partial(GenGaussKernel, shape_init = 1.)

# class PeriodicKernelStd(Kernel):
#     ls:float = 1.
#     per_dim_period:bool = False
#     dim:int = None
#     period_init:ConstOrInitFn = ConstFn(1.)

#     def setup(self):
#         assert self.period_init is not None
#         scale_dim = 1
#         if self.per_dim_scale:
#             assert self.dim is not None, "For one period per dimension, input dimension must be provided"
#             scale_dim = self.dim
#         self.dist = ScaledPairwiseDistance(power = 1., scale_dim = scale_dim, scale_init = self.period_init)

#     def __call__(self, X, Y=None, diag = False):
#         assert(len(np.shape(X))==2)
#         return exp(- 2* np.sin(np.pi * self.dist(X,Y, diag))**2 / self.ls**2)

class PeriodicKernel(Kernel, Module):
    lengthscale_bij: Bijection = SoftPlus()
    lengthscale_init: ConstOrInitFn = ConstFn(np.ones(1))

    per_dim_scale:bool = False
    dim:int = None
    scale_init:ConstOrInitFn = ConstFn(np.ones(1))

    distance_pow_bij: Bijection = Sigmoid(0., 2.)
    distance_pow_init: ConstOrInitFn = ConstFn(np.ones(1))

    def setup(self):
        self.ggkern = GenGaussKernel(per_dim_scale = self.per_dim_scale,
                                     dim = self.dim,
                                     scale_init = self.scale_init,
                                     shape_bij = self.distance_pow_bij, 
                                     shape_init = self.distance_pow_init)
        self.ls = self.lengthscale_bij(self.const_or_param("ls", self.lengthscale_init, (1,), np.float32, self.lengthscale_bij))

    def __call__(self, X, Y=None, diag = False,):
        return exp(-2* np.sin(self.ggkern.dist(X, Y, diag))**2 /self.ls)

class ThreshSpikeKernel(Kernel, Module):
    """Takes on spike falue if squared euclidean distance is below a certain threshold, else non_spike value

        Args:
            spike (float, optional): Kernel value > 0 when input point is equal. Defaults to 1.
            non_spike (float, optional): Kernel value when input point is unequal. abs(non_spike) < spike. Defaults to 0.5.
    """
    spike:float = 1.
    non_spike:float = 0.5
    threshold_distance = 0.
    def setup(self, ):
        
        assert self.spike > 0
        assert abs(self.non_spike)  < self.spike
        self.dist = ScaledPairwiseDistance(power = 2., scale = False)

    def __call__(self, X, Y=None, diag = False):
        assert(len(np.shape(X))==2)
        assert not diag
        return np.where(self.dist(X,Y, diag) <= self.threshold_distance, self.spike, self.non_spike)
