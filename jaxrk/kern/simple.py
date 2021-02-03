from pathlib2 import Path
from typing import Callable

import numpy as onp
import jax.numpy as np, jax.scipy as sp, jax.scipy.stats as stats
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
        X = X / self.period
        if Y is not None:
            Y = Y / self.period

        if diag:
            assert()
        else:
            dists = eucldist(X, Y, power = 1.)
        assert(not logsp)
        return exp(- 2* np.sin(np.pi*dists)**2 / self.ls**2)

class ThreshSpikeKernel(Kernel):
    def __init__(self, spike:float = 1., non_spike:float = 0.5, threshold_sq_eucl_distance = 0.):
        """Takes on spike falue if squared euclidean distance is below a certain threshold, else non_spike value

        Args:
            spike (float, optional): Kernel value > 0 when input point is equal. Defaults to 1.
            non_spike (float, optional): Kernel value when input point is unequal. abs(non_spike) < spike. Defaults to 0.5.
        """
        assert spike > 0
        assert abs(non_spike)  < spike

        self.spike = spike
        self.non_spike = non_spike
        self.threshold_sq_eucl_distance = threshold_sq_eucl_distance

    def gram(self, X, Y=None, diag = False, logsp = False):
        assert(len(np.shape(X))==2)
        if diag:
            assert()
        else:
            dists = eucldist(X, Y, power = 2.)
        assert(not logsp)
        return np.where(dists <= self.threshold_sq_eucl_distance, self.spike, self.non_spike)

class DictKernel(Kernel):
    def __init__(self, inspace_vals:np.array, gram_values:np.array):
        """Kernel for a fixed dictionary of input space values and accompanying gram values. Example:
        ```
            k = DictKernel(np.array([1,3]), np.array([(2, -1), (-1, 1.2)]))
            assert k(np.array(1), np.array(1)) == 2
            assert k(np.array(3), np.array(1)) == -1
            assert k(np.array(3), np.array(3)) == 1.2
            k(np.array(2), np.array(3)) #this will throw an exception, as 2 is not a valid inspace value
        ```
        Args:
            inspace_vals: Order of input space values that this DictKernel is valid for.
            gram_values: Gram values of shape `[len(inspace_vals), len(inspace_vals)]`.
        """

        assert len(inspace_vals.shape) == 1
        assert gram_values.shape == (len(inspace_vals), len(inspace_vals))
        assert np.all(gram_values == gram_values.T)

        self.inspace_vals = inspace_vals
        self.gram_values = gram_values
    
    def convert_data(self, inspace_data:np.array):
        return np.array(list(map(lambda insp: self.insp_to_pos[insp], inspace_data)))
    
    @classmethod
    def read_file(cls, p:Path, dict_file:Path):
        with open(p) as matrix_file:
            lines = matrix_file.readlines()

        with open(dict_file) as df:
            d = df.read()
            d = onp.array(d.strip().split())
        
        header = None
        col_header = []
        m = []
        for idx, row in enumerate(lines):
            row = row.strip()
            if row[0] == '#' or len(row) == 0:
                continue
            entries = row.split()
            if header is None:
                header = entries
                continue
            else:
                col_header.append(entries.pop(0))
                m.append(list(map(float, entries)))
        header, col_header = onp.array(header), onp.array(col_header)
        m = np.array(m)
        assert np.all(header == col_header)
        assert len(header) == m.shape[0] and m.shape[0] == m.shape[1]

        if header[-1] == '*':
            header = header[:-1]
            m = m[:-1, :-1]
        
        reorder = np.argmax(header[:,None] == d[None,:], 0)
       # print(header, m, "\n", d, m[reorder,:][:,reorder])

        return cls(d, m[reorder,:][:,reorder])

    def gram(self, idx_X, idx_Y=None, diag = False, logsp = False):
        assert len(np.shape(idx_X))==2 and (idx_Y is None or len(np.shape(idx_Y))==2)
        assert idx_X.shape[1] == 1 and (idx_Y is None or idx_X.shape[1] == idx_Y.shape[1])
        if idx_Y is None:
            idx_Y = idx_X
        idx_X = idx_X.reshape(-1)
        idx_Y = idx_Y.reshape(-1)
        if diag:
            return self.gram_values[idx_X, idx_Y]
        else:
            return self.gram_values[np.repeat(idx_X, idx_Y.size), np.tile(idx_Y, idx_X.size)].reshape((idx_X.size, idx_Y.size))