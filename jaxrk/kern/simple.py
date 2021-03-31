from pathlib2 import Path
from typing import Callable
from jaxrk.core.typing import PRNGKeyT, Shape, Dtype, Array
from jaxrk.core import Module
from functools import partial

import numpy as onp
import jax.numpy as np, jax.scipy as sp, jax.scipy.stats as stats
from jax.numpy import exp, log, sqrt
from jax.scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from jaxrk.kern.base import DensityKernel, Kernel
from jaxrk.kern.util import ScaledPairwiseDistance
from jaxrk.core.init_fn import ConstFn, ConstIsotropicFn
import flax.linen as ln
from jaxrk.core.typing import ConstOrInitFn
from jaxrk.core.init_fn import ConstFn, ConstIsotropicFn
from jaxrk.core.constraints import SoftPlus, Bijection, CholeskyBijection
from jaxrk.utilities.views import tile_view
from jaxrk.core import Module

class FeatMapKernel(Kernel, Module):
    """A kernel that is defined by a feature map.
    
    Args:
        feat_map: A callable that computes the feature map, i.e. given input space points it returns real valued features, one per input space point."""

    feat_map:Callable[[Array], Array] = None

    def setup(self,):
        assert self.feat_map is not None

    def __call__(self, X, Y = None, diag = False):
        f_X = self.feat_map(X)
        if Y is None:
            f_Y = f_X
        else:
            f_Y = self.feat_map(Y)
        if diag:
            return np.sum(f_X * f_Y, 1)
        else:
            return f_X.dot(f_Y.T)


LinearKernel = partial(FeatMapKernel, feat_map = lambda x:x)

class DictKernel(Kernel, Module):
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
    inspace_vals:Array = None
    gram_values_init:ConstOrInitFn = ConstIsotropicFn(np.ones(1))
    diag_bij:Bijection = SoftPlus()

    def setup(self, ):
        assert self.inspace_vals is None or len(self.inspace_vals.shape) == 1

        self.gram_bij = CholeskyBijection(diag_bij = self.diag_bij)
        self.gram_values = self.gram_bij(self.const_or_param("gram_param", gram_values_init, dim = (self.inspace_vals), bij = self.gram_bij))
        assert self.gram_values.shape == (len(self.inspace_vals), len(self.inspace_vals))
    
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

    def __call__(self, idx_X, idx_Y=None, diag = False):
        assert len(np.shape(idx_X))==2 and (idx_Y is None or len(np.shape(idx_Y))==2)
        assert idx_X.shape[1] == 1 and (idx_Y is None or idx_X.shape[1] == idx_Y.shape[1])
        if idx_Y is None:
            idx_Y = idx_X
        idx_X = idx_X.reshape(-1)
        idx_Y = idx_Y.reshape(-1)
        if diag:
            return self.gram_values[idx_X, idx_Y]
        else:
            #FIXME: repeat_view
            #using https://stackoverflow.com/questions/5564098/repeat-numpy-array-without-replicating-data
            #and https://github.com/google/jax/issues/3171
            #as starting points
            return self.gram_values[np.repeat(idx_X, idx_Y.size), tile_view(idx_Y, idx_X.size)].reshape((idx_X.size, idx_Y.size))