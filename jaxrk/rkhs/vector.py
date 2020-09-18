from copy import copy
from time import time
from typing import Generic, TypeVar

import jax
import jax.numpy as np
import numpy as onp
import scipy as osp
from jax import grad
from jax.numpy import dot, exp, log
from jax.scipy.special import logsumexp
from numpy.random import rand

from jaxrk.reduce import GramReduce, NoReduce

from .base import Map, RkhsObject, Vec

#from jaxrk.utilities.frank_wolfe import frank_wolfe_pos_proj



def __casted_output(function):
    return lambda x: onp.asarray(function(x), dtype=np.float64)


class FiniteVec(Vec):
    """
        RKHS feature vector using input space points. This is the simplest possible vector.
    """
    def __init__(self, kern, inspace_points, prefactors = None, points_per_split = None, center = False):
        row_splits = None
        self.k = kern
        self.inspace_points = inspace_points
        assert(len(self.inspace_points.shape) == 2)
        if prefactors is None:
            if points_per_split is None:
                prefactors = np.ones(len(inspace_points))/len(inspace_points)
            else:
                prefactors = np.ones(len(inspace_points))/points_per_split

        assert(prefactors.shape[0] == len(inspace_points))
        assert(len(prefactors.shape) == 1)
        self.prngkey = jax.random.PRNGKey(np.int64(time()))
        self.__reconstruction_kwargs = {"center" : center}

        self.prefactors = prefactors
        self.center = center


        if (points_per_split is not None) or (row_splits is not None):
            self.__reduce_gram__ = self.__reduce_balanced_ragged__
            self.is_simple = False
            if points_per_split is not None:
                assert row_splits is None, "Either of points_per_split or row_splits can be set, but not both."
                #balanced split: each feature vector element has and equal number of input space points
                self.points_per_split = points_per_split
                self.__len = len(self.inspace_points) // self.points_per_split
                self.__reshape_gram__ = self.__reshape_balanced__
                self.__getitem__ = self.__getitem_balanced__
                self.normalized = self.__normalized_balanced__
                self.__reconstruction_kwargs["points_per_split"] = points_per_split
            else:
                #ragged split: each feature vector element can be composed of a different number of input space points
                self.row_splits = row_splits
                self.__len = len(self.row_splits) - 1
                self.__reshape_gram__ = self.__reshape_ragged__
                self.normalized = self.__normalized_ragged__
                self.__reconstruction_kwargs["row_splits"] = row_splits
        else:
            self.__reduce_gram__ = lambda gram, axis: gram
            self.is_simple = True
            self.__len = len(self.inspace_points)
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
    
    def __normalized_balanced__(self):
        upd_pref = self.__reshape_balanced__(self.prefactors.reshape((-1, 1)))#.squeeze()
        upd_pref = upd_pref / upd_pref.sum(1, keepdims=True)
        return self.updated(upd_pref.reshape(self.prefactors.shape))

    def __normalized_ragged__(self):
        assert()

    def __reshape_balanced__(self, gram):
        return np.reshape(gram, (-1, self.points_per_split, gram.shape[-1]))
    
    def __reshape_ragged__(self, gram):
        assert()
        #return tf.RaggedTensor.from_row_splits(values=gram, row_splits=self.row_splits)
    
    def __reduce_balanced_ragged__(self, gram, axis):
        perm = list(range(len(gram.shape)))
        perm[0] = axis
        perm[axis] = 0

        gram = np.transpose(gram, perm)
        gram = np.sum(self.__reshape_gram__(gram), axis = 1) 
        gram =  np.transpose(gram, perm)
        return gram
    
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
        return self.updated(np.ones_like(self.prefactors))
    
    def __getitem__(self, index):
        return FiniteVec(self.k, self.inspace_points[index], self.prefactors[index])
    
    def __getitem_balanced__(self, index):
        start, stop = (index * self.points_per_split, index+1 * self.points_per_split)
        return FiniteVec(self.k, self.inspace_points[start, stop], self.prefactors[start, stop], points_per_split = self.points_per_split)
    
    def __getitem_ragged__(self, index):
        raise NotImplementedError()
    
    def updated(self, prefactors):
        assert(len(self.prefactors) == len(prefactors))
        return FiniteVec(self.k, self.inspace_points, prefactors, **self.__reconstruction_kwargs)
    
    def centered(self):
        kwargs = copy(self.__reconstruction_kwargs)
        kwargs["center"] = True

        return FiniteVec(self.k, self.inspace_points, self.prefactors, **kwargs)

    def reduce_gram(self, gram, axis = 0):
        gram = gram.astype(self.prefactors.dtype) * np.expand_dims(self.prefactors, axis=(axis+1)%2)
        gram = self.__reduce_gram__(gram, axis)
        if self.center:
            gram = gram - np.mean(gram, axis, keepdims = True)
        return gram
    
    def get_mean_var(self, keepdims = False):
        mean = self.reduce_gram(self.inspace_points, 0)
        variance_of_expectations = self.reduce_gram(self.inspace_points**2, 0) - mean**2
        var = self.k.var + variance_of_expectations

        if keepdims:
            return (mean, var)
        else:
            return (np.squeeze(mean), np.squeeze(var))
    
    def sum(self,):
        return FiniteVec(self.k, self.inspace_points, self.prefactors, points_per_split = len(self.inspace_points))
    
    @classmethod
    def construct_RKHS_Elem(cls, kern, inspace_points, prefactors = None):
        return cls(kern, inspace_points, prefactors, points_per_split = len(inspace_points))
    
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

V1T = TypeVar("V1T")
V2T = TypeVar("V2T")

class CombVec(Vec, Generic[V1T, V2T]):
    def __init__(self, v1:V1T, v2:V2T, operation, gram_reduce:GramReduce = NoReduce()):
        assert(len(v1) == len(v2))
        self.__len = len(v1)
        (self.v1, self.v2) = (v1, v2)
        self.combine = operation
        self._gram_reduce = gram_reduce
    
    def reduce_gram(self, gram, axis = 0):
        return self._gram_reduce(gram, axis)

    def inner(self, Y:"CombVec[V1T, V2T]"=None, full=True):
        if Y is None:
            Y = self
        else:
            assert(Y.combine == self.combine)
        return self.reduce_gram(Y.reduce_gram(self.combine(self.v1.inner(Y.v1), self.v2.inner(Y.v2)), 1), 0)

    def __len__(self):
        if self._gram_reduce is None:
            return self.__len
        else:
            return self._gram_reduce.new_len(self.__len)

    def updated(self, prefactors):
        raise NotImplementedError()

def inner(X, Y=None, full=True):
    return X.inner(Y, full)
