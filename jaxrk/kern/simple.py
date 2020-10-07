
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
        """A simple linear kernel.
        """
        FeatMapKernel.__init__(self, lambda x: x)



class PeriodicKernel(Kernel):
    def __init__(self, period:float, lengthscale:float, ):
        """Periodic kernel, i.e. exp(- 2 sin(π dists/period)^2 / lengthscale^2).

        Args:
            period (float): Period length.
            lengthscale (float): Lengthscale
        """
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
            dists = eucldist(X/self.period, Y/self.period, power = 1.)
        assert(not logsp)
        return exp(- 2* np.sin(np.pi*dists)**2 / self.ls**2)