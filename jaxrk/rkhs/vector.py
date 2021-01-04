from copy import copy

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
from jaxrk.utilities.rkhsdist import rkhs_cdist, rkhs_cdist_ignore_const

from .base import Map, RkhsObject, Vec

#from jaxrk.utilities.frank_wolfe import frank_wolfe_pos_proj



def __casted_output(function):
    return lambda x: onp.asarray(function(x), dtype=np.float64)


class FiniteVec(Vec):
    """
        RKHS feature vector using input space points. This is the simplest possible vector.
    """
    def __init__(self, kern:Kernel, inspace_points, reduce:List[Reduce] = None):
        row_splits = None
        self.k = kern
        self.inspace_points = inspace_points
        assert(len(self.inspace_points.shape) == 2)
        final_len = Reduce.final_len(len(inspace_points), reduce)
        if reduce is None:
            reduce = []# Prefactors(np.ones(final_len)/final_len)
            self.is_simple = True
        self.reduce = reduce

        self.__len = final_len
        
        self.prngkey = jax.random.PRNGKey(np.int64(time()))
        self._raw_gram_cache = None

    def __eq__(self, other):
        assert False, "not yet implemented checking equality of reduce"
        return (isinstance(other, self.__class__) and
                np.all(other.inspace_points == self.inspace_points) and
                other.k == self.k)

    def __len__(self):
        return self.__len
    
    def __neg__(self):
        self.updated(-self.prefactors)
    
    def inner(self, Y=None, full=True):
        if not full and Y is not None:
            raise ValueError(
                "Ambiguous inputs: `diagonal` and `y` are not compatible.")
        if not full:
            return  self.reduce_gram(self.reduce_gram(self.k(self.inspace_points, full = full), axis = 0), axis = 1)
        if Y is not None:
            assert(self.k == Y.k)
        else:
            Y = self
        gram = self.k(self.inspace_points, Y.inspace_points).astype(float)
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
            #return FiniteVec(self.k, self.inspace_points, self.reduce[:-1]).extend_reduce([LinearReduce(p)])
        else:
            assert False
            p = np.ones(len(self))/ len(self)
        
    
    def __getitem__(self, index):
        return FiniteVec(self.k, self.inspace_points[index], self.prefactors[index])
    
    def __getitem_balanced__(self, index):
        start, stop = (index * self.points_per_split, index+1 * self.points_per_split)
        return FiniteVec(self.k, self.inspace_points[start, stop], self.prefactors[start, stop], points_per_split = self.points_per_split)
    
    def __getitem_ragged__(self, index):
        raise NotImplementedError()
    
    def updated(self, prefactors) -> "FiniteVec":
        _r = copy(self.reduce)
        if len(_r) > 0 and (isinstance(_r[-1], Prefactors) or isinstance(_r[-1], LinearReduce)): 
            assert Reduce.final_len(len(self.inspace_points), _r) == prefactors.shape[0]
            _r = _r[:-1]
        
        if len(prefactors.shape) == 1:            
            _r.append(Prefactors(prefactors))
        elif len(prefactors.shape) == 2:
            assert len(self) == prefactors.shape[0]
            _r.append(LinearReduce(prefactors))
        return FiniteVec(self.k, self.inspace_points, _r)

    def centered(self) -> "FiniteVec":
        return self.extend_reduce([Center()])

    def extend_reduce(self, r:List[Reduce]) -> "FiniteVec":
        if r is None or len(r) == 0:
            return self
        else:
            _r = copy(self.reduce)
            _r.extend(r)
            return FiniteVec(self.k, self.inspace_points, _r)

    def reduce_gram(self, gram, axis = 0):
        return Reduce.apply(gram, self.reduce, axis) 
    
    def nsamps(self, mean = False) -> np.ndarray:
        n = len(self.inspace_points)
        rval = Reduce.apply(np.ones(n)[:, None] * n, self.reduce, 0) 
        return rval.mean() if mean else rval
    
    def get_mean_var(self, keepdims = False) -> np.ndarray:
        mean = self.reduce_gram(self.inspace_points, 0)
        variance_of_expectations = self.reduce_gram(self.inspace_points**2, 0) - mean**2
        var = self.k.var + variance_of_expectations

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
            self._raw_gram_cache = self.k(self.inspace_points).astype(np.float64)
        return self._raw_gram_cache
    
    def point_representant(self, method:str = "inspace_point", keepdims:bool=False):
        if method == "inspace_point":
            assert isinstance(self.reduce[-1], Prefactors) or isinstance(self.reduce[-1], LinearReduce)
            G_orig_repr = Reduce.apply(self._raw_gram, self.reduce, 1)
            repr_idx = gram_projection(G_orig_repr, Reduce.apply(G_orig_repr, self.reduce, 0), self._raw_gram, method = "representer").squeeze()
            rval = self.inspace_points[repr_idx,:]
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
            return FiniteVec(self.k, self.inspace_points, [LinearReduce(lin_map)])
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
    
    def cdist(self, other:"FiniteVec", norm_power:float = 2.):
        """Compute RKHS distance between RKHS elements in vector self and in vector other.
        """

        if self == other:
            return rkhs_cdist(self.inner(other))
        else:
            return rkhs_cdist(self.inner(other), self.inner(), other.inner())

    def rvs(self, nsamps:int = 1) -> np.array:
        assert np.all(self.prefactors >= 0.)

        #use residual resampling from SMC theory
        if nsamps is None:
            nsamps = len(pop)        
        prop_w = log(self.normalized().prefactors)
        mult = exp(prop_w + log(nsamps))
        count = np.int32(np.floor(mult))
        resid = log(mult - count)
        resid = resid - logsumexp(resid)
        count = count + onp.random.multinomial(nsamps - count.sum(), exp(resid))

        rval = np.repeat(self.inspace_points, count, 0) + self.k.rvs(nsamps, self.inspace_points.shape[1])
        return rval

    
    def __call__(self, argument):
        return inner(self, FiniteVec(self.k, argument, []))

def choose_representer(support_points, factors, kernel):
    return choose_representer_from_gram(kernel(support_points).astype(np.float64), factors)

def choose_representer_from_gram(G, factors):
    fG = np.dot(factors, G)
    rkhs_distances_sq = (np.dot(factors, fG).flatten() + np.diag(G) - 2 * fG).squeeze()
    rval = np.argmin(rkhs_distances_sq)
    assert rval < rkhs_distances_sq.size
    return rval

def gram_projection(G_orig_repr:np.array,  G_orig:np.array=None, G_repr:np.array=None, method:str = "representer"):
    if method == "representer":
        return np.argmin(rkhs_cdist(G_orig_repr, G_repr, G_orig), 0)
    elif method == "pos_proj":
        assert G_repr is not None
        s = G_orig_repr.shape
        n_pref = np.prod(np.array(s))
        def cost(M):
            M = M.reshape(s)
            return np.trace(rkhs_cdist_ignore_const(G_orig_repr @ M.T, M @ G_repr@ M.T))

        res = osp.optimize.minimize(__casted_output(cost),
                               rand(n_pref)+ 0.0001,
                               jac = __casted_output(grad(cost)),
                               bounds = [(0., None)] * n_pref)
        return res["x"].reshape(s)
    else:
        assert False, "No valid method selected"

def distr_estimate_optimization(kernel, support_points, est="support"):
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

V1T = TypeVar("V1T", bound=Vec)
V2T = TypeVar("V2T", bound=Vec)

class CombVec(Vec, Generic[V1T, V2T]):
    def __init__(self, v1:V1T, v2:V2T, operation, reduce:List[Reduce] = []):
        assert(len(v1) == len(v2))
        self.__len = Reduce.final_len(len(v1), reduce)
        (self.v1, self.v2) = (v1, v2)
        self.combine = operation
        self._reduce = reduce
    
    def reduce_gram(self, gram, axis = 0):
        carry = gram
        if self._reduce is not None:
            for gr in self._reduce:
                carry = gr(carry, axis)
        return carry

    def inner(self, Y:"CombVec[V1T, V2T]"=None, full=True):
        if Y is None:
            Y = self
        else:
            assert(Y.combine == self.combine)
        return self.reduce_gram(Y.reduce_gram(self.combine(self.v1.inner(Y.v1), self.v2.inner(Y.v2)), 1), 0)
    
    def extend_reduce(self, r:List[Reduce]) -> "CombVec":
        if r is None or len(r) == 0:
            return self
        else:
            _r = copy(self._reduce)
            _r.extend(r)
            return CombVec(self.v1, self.v2, self.combine, _r)

    def centered(self) -> "CombVec":
        return self.extend_reduce([Center()])
 

    def __len__(self):
        if self._reduce is None:
            return len(self.v1)
        else:
            return self.__len

    def updated(self, prefactors):
        raise NotImplementedError()

def inner(X, Y=None, full=True):
    return X.inner(Y, full)
