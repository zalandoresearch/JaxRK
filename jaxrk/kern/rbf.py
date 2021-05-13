
from typing import Callable, Tuple, Union, Optional
from jaxrk.core.typing import PRNGKeyT, Shape, Dtype, Array, ConstOrInitFn
from functools import partial
from jaxrk.core.constraints import SoftPlus, Bijection, SoftBd, Sigmoid
from jaxrk.core.init_fn import ConstFn
from dataclasses import dataclass
from flax.linen.module import compact
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
from jaxrk.kern.util import ScaledPairwiseDistance, SimpleScaler


class GenGaussKernel(DensityKernel): #this is the gennorm distribution from scipy
    """Kernel derived from the pdf of the generalized Gaussian distribution (https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1). """    


    def __init__(self, dist:ScaledPairwiseDistance):
        super().__init__()
        self.dist = dist

    @classmethod
    def make_unconstr(cls,
                      scale:Array,
                      shape:float,
                      scale_bij: Bijection = SoftPlus(),
                      shape_bij: Bijection = Sigmoid(0., 2.)) -> "GenGaussKernel":
        """Factory for constructing a GenGaussKernel from unconstrained parameters.
           The constraints for each parameters are then guaranteed by applying their accompanying bijections.
            Args:
                scale (Array): Scale parameter, unconstrained. 
                shape (float): Shape parameter, unconstrained. Lower values result in pointier kernel functions. 
                scale_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftPlus.
                shape_bij (Bijection): Bijection mapping from unconstrained real numbers to half-open interval (0,2]. Defaults to Sigmoid(0., 2.).
        """
        return cls.make(scale_bij(scale), shape_bij(shape))
    
    @classmethod
    def make(cls,
             scale:Array,
             shape:float) -> "GenGaussKernel":
        """Factory for constructing a GenGaussKernel from scale and shape parameters.
            Args:
                scale (Array): Scale parameter, nonnegative.
                shape (float): Shape parameter, in half-open interval (0,2]. Lower values result in pointier kernel functions. Shape == 2 results in usual Gaussian kernel, shape == 1 results in Laplace kernel.
        """
        assert shape > 0 and shape <= 2
        dist = ScaledPairwiseDistance(scaler = SimpleScaler(scale), power = shape)
        return GenGaussKernel(dist)
    
    @classmethod
    def make_laplace(cls,
             scale:Array) -> "GenGaussKernel":
        """Factory for constructing a Laplace kernel from scale parameter.
            Args:
                scale (Array): Scale parameter, nonnegative.
        """
        return GenGaussKernel.make(scale, 1.)
    
    @classmethod
    def make_gauss(cls,
             scale:Array) -> "GenGaussKernel":
        """Factory for constructing a Laplace kernel from scale parameter.
            Args:
                scale (Array): Scale parameter, nonnegative.
        """
        return GenGaussKernel.make(scale, 2.)

    def std(self):
        return np.sqrt(self.var())
    
    def var(self):
        f = np.exp(sp.special.gammaln(np.array([3, 1]) / self.dist.power))
        return self.dist._get_scale_param()**2 * f[0] / f[1]

    def __call__(self, X, Y=None, diag = False,):
        return exp(-self.dist(X, Y, diag))


# class GenPeriodicKernel(Kernel):

#     def __init__(self, dist:ScaledPairwiseDistance, length_scale:float, sin_power:float = 2.):
#         """Periodic kernel class. A periodic kernel is defined by
#             exp(-2 * sin(dist(X, Y, diag))**sin_power / length_scale)

#         Args:
#             dist (ScaledPairwiseDistance): Class computing the scaled pairwise distance between data points.
#             length_scale (float): Length scale.
#             sin_power (float): Power to which the sine is raised.
#         """
#         super().__init__()
#         self.dist = dist

#         assert np.all(length_scale > 0)
#         assert sin_power > 0
#         self.ls = length_scale
#         self.pow = sin_power

#     @classmethod
#     def make_unconstr(cls,
#                       period:Array,
#                       shape:float,
#                       length_scale:float,
#                       sin_power:float = 2.,
#                       scale_bij: Bijection = SoftPlus(),
#                       shape_bij: Bijection = Sigmoid(0., 2.),
#                       length_scale_bij: Bijection = SoftPlus(),
#                       sin_power_bij: Bijection = Sigmoid(0., 2.)) -> "PeriodicKernel":
#         """Factory for constructing a PeriodicKernel from unconstrained parameters.
#            The constraints for each parameters are then guaranteed by applying their accompanying bijections.
#             Args:
#                 period (Array): Scale parameter, unconstrained. 
#                 shape (float): Shape parameter, unconstrained. Lower values result in pointier kernel functions. 
#                 length_scale (float): Lengscale parameter, unconstrained.
#                 sin_power (float): Parameter for power to which the sine value is raised, unconstrained.
#                 scale_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftPlus.
#                 shape_bij (Bijection): Bijection mapping from unconstrained real numbers to half-open interval (0,2]. Defaults to Sigmoid(0., 2.).
#                 length_scale_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftPlus.
#                 sin_power_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to Sigmoid(0., 2.).

#         Returns:
#             PeriodicKernel: The constructed kernel.
#         """
#         return cls.make(scale_bij(period), shape_bij(shape), length_scale_bij(length_scale), sin_power_bij(sin_power))
    
#     @classmethod
#     def make(cls,
#              period:Array,
#              shape:float,
#              length_scale:float,
#              sin_power:float = 2.) -> "PeriodicKernel":
#         """Factory for constructing a PeriodicKernel.
#             Args:
#                 period (Array): Scale parameter, nonnegative.
#                 shape (float): Shape parameter, in half-open interval (0,2]. Lower values result in pointier kernel functions. Shape == 2 results in usual Gaussian kernel, shape == 1 results in Laplace kernel.
#                 length_scale (float): Lengscale parameter, nonnegative.
#                 sin_power (float): Parameter for power to which the sine value is raised, nonnegative.
#             Returns:
#                 PeriodicKernel: The constructed kernel.
#         """
#         assert shape > 0 and shape <= 2
#         dist = ScaledPairwiseDistance(scaler = SimpleScaler(period), power = shape)
#         return cls(dist, length_scale, sin_power)

#     def __call__(self, X, Y=None, diag = False,):
#         return exp(-2 * (np.sin(np.pi * self.dist(X, Y, diag)) /self.ls)**self.pow)

class PeriodicKernel(Kernel):

    def __init__(self, period:float, length_scale:float,):
        """Periodic kernel class. A periodic kernel is defined by
            exp(-2 * sin(dist(X, Y, diag))**sin_power / length_scale)

        Args:
            period (float): period.
            length_scale (float): Length scale.
        """
        super().__init__()
        self.dist = ScaledPairwiseDistance(scaler = SimpleScaler(period), power = 1.)
        print(self.dist.gs, self.dist.gs.scale().size, self.dist.ds, self.dist.ds.scale().size)
        self.ls = length_scale        
    

    def __call__(self, X, Y=None, diag = False,):
        assert len(np.shape(X))==2
        assert not diag
        scale_param = self.dist._get_scale_param()
        #scale_param = 1./self.dist.gs.scale()
        X = X * scale_param
        if Y is not None:
            Y = Y * scale_param

        dists = eucldist(X, Y, power = 1.)
        d2 = self.dist(X, Y, diag)
        assert np.allclose(dists, d2)
        return exp(- 2* np.sin(np.pi*dists)**2 / self.ls**2)

class ThreshSpikeKernel(Kernel):
    def __init__(self,  dist:ScaledPairwiseDistance, spike:float, non_spike:float, threshold:float):
        """Takes on spike value if squared euclidean distance is below a certain threshold, else non_spike value

        Args:
            spike (float): Kernel value when distance between input points is below threshold_distance, nonnegative. 
            non_spike (float): Kernel value when distance between input points is above threshold_distance. Has to satisfy abs(non_spike) < spike.
            threshold_distance (float): Distance threshold.
    """
        super().__init__()
        assert spike > 0
        assert abs(non_spike) < spike
        assert threshold >= 0
        self.dist = dist
        self.spike, self.non_spike, self.threshold = spike, non_spike, threshold
    
    @classmethod
    def make_unconstr(cls,
             scale:Array,
             shape:float,
             spike:float,
             non_spike:float,
             threshold:float,
             scale_bij: Bijection = SoftPlus(),
             shape_bij: Bijection = SoftPlus(),
             spike_bij: Bijection = SoftPlus(),
             non_spike_bij: Bijection = SoftBd(upper_bound = 1.),
             threshold_bij: Bijection = SoftBd(lower_bound = 0.)) -> "ThreshSpikeKernel":
        """Factory for constructing a PeriodicKernel from unconstrained parameters.
            Args:
                scale (Array): Scale parameter for distance computation.
                shape (float): Shape parameter for distance computation. 
                spike (float): Kernel value when distance between input points is above threshold_distance. 
                non_spike (float): Non-spike value, has to satisfy abs(non_spike) < spike
                threshold (float): Below theshold
                scale_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftPlus.
                shape_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftPlus.
                spike_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftPlus.
                non_spike_bij (Bijection): Bijection mapping from unconstrained real numbers to numbers smaller than 1. Defaults to SoftBd(upper_bound = 1.).
                threshold_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftBd(lower_bound = 0.).
            Returns:
                ThreshSpikeKernel: Distance threshold
        """
        dist = ScaledPairwiseDistance(scaler = SimpleScaler(scale_bij(scale)), power = shape_bij(shape))
        return cls(dist, spike_bij(spike), non_spike_bij(non_spike), threshold_bij(threshold))
        

    def __call__(self, X, Y=None, diag = False):
        assert(len(np.shape(X))==2)
        assert not diag
        return np.where(self.dist(X,Y, diag) <= self.threshold, self.spike, self.non_spike)
