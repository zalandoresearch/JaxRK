"""
Created on Thu Jan 10 10:01:56 2019

@author: Ingmar Schuster
"""


from typing import Callable

import numpy as onp
import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats
import flax.linen as ln
from jax.numpy import exp, log, sqrt
from jax.scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.spatial.distance import pdist

from jaxrk.utilities.eucldist import eucldist


class Kernel(object):
    """A generic kernel type."""
    def __call__(self, X, Y = None, diag = False) -> np.array:
        """Compute the gram matrix, i.e. the kernel evaluated at every element of X paired with each element of Y (if not None, otherwise each element of X).

        Args:
            X: input space points, one per row.
            Y: input space points, one per row. If none, default to Y = X.
            diag: if `True`, compute only the diagonal elements of the gram matrix.

        Returns:
            The gram matrix or its diagonal, depending on passed parameters."""
        raise NotImplementedError()


class DensityKernel(Kernel):
    """Type for positive definite kernels that are also densities."""
    def rvs(self, nsamps):
        raise NotImplementedError()