"""
Created on Thu Jan 10 10:01:56 2019

@author: Ingmar Schuster
"""


from typing import Callable

import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats
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