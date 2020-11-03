from copy import copy
from jaxrk.reduce.base import BalancedSum, Center, Prefactors, Sum
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
        return (isinstance(other, self.__class__) and
                np.all(other.prefactors == self.prefactors) and
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
    
    def normalized(self):
        r = self.reduce
        if isinstance(r[-1], Prefactors):
            p = r[-1].prefactors/np.sum(r[-1].prefactors)
        else:
            assert False
            p = np.ones(len(self))/ len(self)
        return self.updated(p)
    
    def __getitem__(self, index):
        return FiniteVec(self.k, self.inspace_points[index], self.prefactors[index])
    
    def __getitem_balanced__(self, index):
        start, stop = (index * self.points_per_split, index+1 * self.points_per_split)
        return FiniteVec(self.k, self.inspace_points[start, stop], self.prefactors[start, stop], points_per_split = self.points_per_split)
    
    def __getitem_ragged__(self, index):
        raise NotImplementedError()
    
    def updated(self, prefactors):
        assert len(self) == len(prefactors)
        _r = copy(self.reduce)
        if isinstance(_r[-1], Prefactors):
            _r = _r[:-1]
        _r.append(Prefactors(prefactors))
        return FiniteVec(self.k, self.inspace_points, _r)

    def centered(self):
        return self.extend_reduce([Center()])
        
    def extend_reduce(self, r:List[Reduce]):
        if r is None or len(r) == 0:
            return self
        else:
            _r = copy(self.reduce)
            _r.extend(r)
            return FiniteVec(self.k, self.inspace_points, _r)

    def reduce_gram(self, gram, axis = 0):
        carry = gram
        if self.reduce is not None:
            for gr in self.reduce:
                carry = gr(carry, axis)
        return carry
    
    def get_mean_var(self, keepdims = False):
        mean = self.reduce_gram(self.inspace_points, 0)
        variance_of_expectations = self.reduce_gram(self.inspace_points**2, 0) - mean**2
        var = self.k.var + variance_of_expectations

        if keepdims:
            return (mean, var)
        else:
            return (np.squeeze(mean), np.squeeze(var))
    
    def sum(self,):
        reduce = copy(self.reduce)
        reduce.append(Sum)
        return FiniteVec(self.k, self.inspace_points, reduce)
    
    @classmethod
    def construct_RKHS_Elem(cls, kern, inspace_points, prefactors = None):
        return cls(kern, inspace_points, [Prefactors(prefactors), BalancedSum(len(inspace_points))])
    
    @classmethod
    def construct_RKHS_Elem_from_estimate(cls, kern, inspace_points, estimate = "support", unsigned = True, regul = 0.1) -> "FiniteVec":
        prefactors = distr_estimate_optimization(kern, inspace_points, est=estimate)
        return cls(kern, inspace_points, prefactors, points_per_split = len(inspace_points))
    
    def point_representant(self, method = "inspace_point"):
        
        if method == "inspace_point":
            n = self.normalized()
            if self._raw_gram_cache is None:
                self._raw_gram_cache = n.k(n.inspace_points).astype(np.float64)
            repr_idx = choose_representer_from_gram(self._raw_gram_cache, n.prefactors.flatten())
            return n.inspace_points[repr_idx]
        elif method == "mean":
            return np.atleast_1d([self.get_mean_var()[0]])
        else:
            n = self.dens_proj()
            return n.inspace_points[jax.random.categorical(self.prngkey, log(n.prefactors.flatten())), :]
    
    def pos_proj(self, nsamps:int = None) -> "FiniteVec":
        """Project to RKHS element with purely positive prefactors. Assumes `len(self) == 1`.

        Args:
            nsamps (int, optional): Number of input space points. Defaults to None, in which case the input space points of self are reused.

        Returns:
            FiniteVec: The result of the projection.
        """
        assert(len(self) == 1)
        if nsamps is None:
            G = kernel(self.inspace_points).astype(np.float64)
            c = 2*self.reduce_gram(G)
            def cost(f):
                f = f.reshape((len(self), G.shape[0]))
                return np.sum(dot(dot(f, G), f.T) - dot(f, c.T))
            res = osp.optimize.minimize(__casted_output(cost),
                                    rand(len(factors))+ 0.0001,
                                    jac = __casted_output(grad(cost)),
                                    bounds = [(0., None)] * len(factors))
            return self.updated(pos_proj(self.inspace_points, res["x"], self.k))
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
        return inner(self, FiniteVec(self.k, argument, np.ones(len(argument))))

def choose_representer(support_points, factors, kernel):
    return choose_representer_from_gram(kernel(support_points).astype(np.float64), factors)

def choose_representer_from_gram(G, factors):
    fG = np.dot(factors, G)
    rkhs_distances_sq = (np.dot(factors, fG).flatten() + np.diag(G) - 2 * fG).squeeze()
    rval = np.argmin(rkhs_distances_sq)
    assert rval < rkhs_distances_sq.size
    return rval

def pos_proj(support_points, factors, kernel):
    G = kernel(support_points).astype(np.float64)
    c = 2*np.dot(factors, G)
    cost = lambda f: dot(dot(f, G), f) - dot(c, f)
    res = osp.optimize.minimize(__casted_output(cost),
                               rand(len(factors))+ 0.0001,
                               jac = __casted_output(grad(cost)),
                               bounds = [(0., None)] * len(factors))

    return res["x"]

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
    def __init__(self, v1:V1T, v2:V2T, operation, reduce:List[Reduce] = None):
        assert(len(v1) == len(v2))
        self.__len = len(v1)
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

    def __len__(self):
        if self._reduce is None:
            return self.__len
        else:
            return self._reduce.new_len(self.__len)

    def updated(self, prefactors):
        raise NotImplementedError()

def inner(X, Y=None, full=True):
    return X.inner(Y, full)
