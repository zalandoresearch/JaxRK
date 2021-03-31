from copy import copy
from dataclasses import field

from numpy.core.fromnumeric import squeeze
from jaxrk.reduce.lincomb import LinearReduce
from jaxrk.reduce.base import BalancedSum, Center, Prefactors, Sum, Mean
from time import time
from typing import Generic, TypeVar, List

import jax
import jax.numpy as np
import numpy as onp
import scipy as osp
from jax import grad
from jax.numpy import dot, exp, log
from jax.scipy.special import logsumexp
from numpy.random import rand

from jaxrk.reduce import Reduce, NoReduce
from jaxrk.kern import Kernel
from jaxrk.utilities.gram import rkhs_gram_cdist, rkhs_gram_cdist_ignore_const, gram_projection
from jaxrk.core.typing import AnyOrInitFn
from jaxrk.core import Module


from .base import Vec, LinOp

#from jaxrk.utilities.frank_wolfe import frank_wolfe_pos_proj


class FiniteVec(Vec, Module):
    """
        RKHS feature vector using input space points. This is the simplest possible vector.
    """
    k:Kernel
    insp_pts_init: AnyOrInitFn
    reduce:List[Reduce] = field(default_factory=list)

    def setup(self, ):        
        if self.reduce is None:
            self.reduce = []

        self.insp_pts = self.const_or_param("insp_pts", self.insp_pts_init,)
        assert(len(self.insp_pts.shape) == 2)
        self.__len = self.param("__len", lambda rngkey, pts, red: Reduce.final_len(len(pts), red), self.insp_pts, self.reduce)
        self._raw_gram_cache = None

    def __eq__(self, other):
        assert False, "not yet implemented checking equality of reduce"
        return (isinstance(other, self.__class__) and
                np.all(other.inspace_points == self.insp_pts) and
                other.k == self.k)

    def __len__(self):
        return self.__len
    
    def __neg__(self):
        self.updated(-self.prefactors)
    
    def inner(self, Y=None, full=True):
        if not full and Y is not None:
            raise ValueError(
                "Ambiguous inputs: `full` and `y` are not compatible.")
        if not full:
            return  self.reduce_gram(self.reduce_gram(self.k(self.insp_pts, full = full), axis = 0), axis = 1)
        if Y is not None:
            assert(self.k == Y.k)
        else:
            Y = self
        gram = self.k(self.insp_pts, Y.insp_pts).astype(float)
        r1 = self.reduce_gram(gram, axis = 0)
        r2 = Y.reduce_gram(r1, axis = 1)
        return r2
    
    def normalized(self) -> "FiniteVec":
        r = self.reduce
        if isinstance(r[-1], Prefactors):
            p = r[-1].prefactors/np.sum(r[-1].prefactors)
            return self.updated(p)
        elif isinstance(r[-1], LinearReduce):
            p = r[-1].linear_map/np.sum(r[-1].linear_map, 1, keepdims=True)
            return self.updated(p)
        else:
            assert False
            p = np.ones(len(self))/ len(self)
    
    def updated(self, prefactors) -> "FiniteVec":
        _r = copy(self.reduce)
        previous_reduction = None
        if len(_r) > 0 and (isinstance(_r[-1], Prefactors) or isinstance(_r[-1], LinearReduce)):
            final_len = Reduce.final_len(len(self.insp_pts), _r)
            assert (final_len == prefactors.shape[0]) or (final_len == 1 and len(prefactors.shape) == 1)
            previous_reduction = _r[-1]
            _r = _r[:-1]
        
        if len(prefactors.shape) == 1:
            assert previous_reduction is None or isinstance(previous_reduction, Prefactors)            
            _r.append(Prefactors(prefactors))
        elif len(prefactors.shape) == 2:
            assert previous_reduction is None or isinstance(previous_reduction, LinearReduce)   
            assert len(self) == prefactors.shape[0]
            _r.append(LinearReduce(prefactors))
        return FiniteVec(self.k, self.insp_pts, _r)

    def centered(self) -> "FiniteVec":
        return self.extend_reduce([Center()])

    def extend_reduce(self, r:List[Reduce]) -> "FiniteVec":
        if r is None or len(r) == 0:
            return self
        else:
            _r = copy(self.reduce)
            _r.extend(r)
            return FiniteVec(self.k, self.insp_pts, _r)

    def reduce_gram(self, gram, axis = 0):
        return Reduce.apply(gram, self.reduce, axis) 
    
    def nsamps(self, mean = False) -> np.ndarray:
        n = len(self.insp_pts)
        rval = Reduce.apply(np.ones(n)[:, None] * n, self.reduce, 0) 
        return rval.mean() if mean else rval
    
    def get_mean_var(self, keepdims = False) -> np.ndarray:
        mean = self.reduce_gram(self.insp_pts, 0)
        variance_of_expectations = self.reduce_gram(self.insp_pts**2, 0) - mean**2
        var = self.k.get_var() + variance_of_expectations

        if keepdims:
            return (mean, var)
        else:
            return (np.squeeze(mean), np.squeeze(var))
    
    def sum(self, use_linear_reduce = False) -> "FiniteVec":
        if use_linear_reduce:
            return self.extend_reduce([LinearReduce(np.ones((1, len(self))))])
        else:
            return self.extend_reduce([Sum()])
    
    def mean(self,) -> "FiniteVec":
        return self.extend_reduce([Mean()])
    
    @classmethod
    def construct_RKHS_Elem(cls, kern, inspace_points, prefactors = None) -> "FiniteVec":
        assert len(prefactors.squeeze().shape) == 1
        return cls(kern, inspace_points, [LinearReduce(prefactors.squeeze()[None, :])])
    
    @classmethod
    def construct_RKHS_Elem_from_estimate(cls, kern, inspace_points, estimate = "support", unsigned = True, regul = 0.1) -> "FiniteVec":
        prefactors = distr_estimate_optimization(kern, inspace_points, est=estimate)
        return cls(kern, inspace_points, prefactors, points_per_split = len(inspace_points))
    
    @property
    def _raw_gram(self):
        if self._raw_gram_cache is None:
            self._raw_gram_cache = self.k(self.insp_pts).astype(np.float64)
        return self._raw_gram_cache
    
    def point_representant(self, method:str = "inspace_point", keepdims:bool=False):
        if method == "inspace_point":
            assert isinstance(self.reduce[-1], Prefactors) or isinstance(self.reduce[-1], LinearReduce)
            G_orig_repr = Reduce.apply(self._raw_gram, self.reduce, 1)
            repr_idx = gram_projection(G_orig_repr, Reduce.apply(G_orig_repr, self.reduce, 0), self._raw_gram, method = "representer").squeeze()
            rval = self.insp_pts[repr_idx,:]
        elif method == "mean":
            rval = self.get_mean_var(keepdims=keepdims)[0]
        else:
            assert False, "No known method selected for point_representant"
        if not keepdims:
            return rval.squeeze()
        else:
            return rval
    
    def pos_proj(self, nsamps:int = None) -> "FiniteVec":
        """Project to RKHS element with purely positive prefactors. Assumes `len(self) == 1`.

        Args:
            nsamps (int, optional): Number of input space points. Defaults to None, in which case the input space points of self are reused.

        Returns:
            FiniteVec: The result of the projection.
        """
        assert(len(self) == 1)
        if nsamps is None:
            lin_map = gram_projection(Reduce.apply(self._raw_gram, self.reduce, 0), G_repr = self._raw_gram, method = "pos_proj")
            return FiniteVec(self.k, self.insp_pts, [LinearReduce(lin_map)])
        else:
            assert False, "Frank-Wolfe needs attention."
            #the problem are circular imports.

            #return frank_wolfe_pos_proj(self, self.updated(pos_proj(self.inspace_points, self.prefactors, self.k)), nsamps - self.inspace_points.shape[0])

    def dens_proj(self, nsamps:int = None) -> "FiniteVec":
        """Project to an RKHS object that is also a density in the usual sense. In particular, a projection to positive prefactors and then normalization so prefactors sum to 1.

        Returns:
            FiniteVec: The result of the projection
        """
        return self.normalized().pos_proj(nsamps).normalized()
    
    def __call__(self, argument):
        return inner(self, FiniteVec(self.k, argument, []))

def distr_estimate_optimization(kernel, support_points, est="support"):
    def __casted_output(function):
        return lambda x: onp.asarray(function(x), dtype=np.float64)
    G = kernel(support_points).astype(np.float64)

    if est == "support":
        #solution evaluated in support points should be positive constant
        cost = lambda f: np.abs(dot(f, G) - 1).sum()
    elif est == "density":
        #minimum negative log likelihood of support_points under solution
        cost = lambda f: -log(dot(f, G)).sum()

    bounds = [(0., None)] * len(support_points)

    res = osp.optimize.minimize(__casted_output(cost), rand(len(support_points))+ 0.0001, jac = __casted_output(grad(cost)), bounds = bounds)

    return res["x"]/res["x"].sum()

VrightT = TypeVar("VrightT", bound=Vec)
VleftT = TypeVar("VleftT", bound=Vec)

class CombVec(Vec, Generic[VrightT, VleftT], Module):
    vR:VrightT
    vL:VleftT
    operation:callable
    reduce:List[Reduce] = field(default_factory=list)


    def setup(self,):
        assert(len(self.vR) == len(self.vL))
        self.__len = self.variable("__len", lambda pts, red: Reduce.final_len(len(pts), red), self.vR, self.reduce)

    
    def reduce_gram(self, gram, axis = 0):
        carry = gram
        if self.reduce is not None:
            for gr in self.reduce:
                carry = gr(carry, axis)
        return carry

    def inner(self, Y:"CombVec[VrightT, VleftT]"=None, full=True):
        if Y is None:
            Y = self
        else:
            assert(Y.combine == self.combine)
        return self.reduce_gram(Y.reduce_gram(self.operation(self.vR.inner(Y.vR), self.vL.inner(Y.vL)), 1), 0)
    
    def extend_reduce(self, r:List[Reduce]) -> "CombVec":
        if r is None or len(r) == 0:
            return self
        else:
            _r = copy(self.reduce)
            _r.extend(r)
            return CombVec(self.vR, self.vL, self.operation, _r)

    def centered(self) -> "CombVec":
        return self.extend_reduce([Center()])
 

    def __len__(self):
        if self._reduce is None:
            return len(self.vR)
        else:
            return self.__len

    def updated(self, prefactors):
        raise NotImplementedError()

def inner(X, Y=None, full=True):
    return X.inner(Y, full)
