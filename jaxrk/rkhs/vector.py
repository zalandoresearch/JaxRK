import numpy as onp
import jax.numpy as np

from typing import Generic, TypeVar

from .base import Vec, Op, RkhsObject


class FiniteVec(Vec):
    """
        RKHS feature vector using input space points. This is the simplest possible vector.
    """
    def __init__(self, kern, inspace_points, prefactors = None, points_per_split = None):
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

        self.__reconstruction_kwargs = {}

        self.prefactors = prefactors


        if (points_per_split is not None) or (row_splits is not None):
            self.__reduce_gram__ = self.__reduce_balanced_ragged__
            self.is_simple = False
            if points_per_split is not None:
                assert(row_splits is None, "Either of points_per_split or row_splits can be set, but not both.")
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

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                np.all(other.prefactors == self.prefactors) and
                np.all(other.inspace_points == self.inspace_points) and
                other.k == self.k)

    def __len__(self):
        return self.__len
    
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

    def reduce_gram(self, gram, axis = 0):
        gram = gram.astype(self.prefactors.dtype) * np.expand_dims(self.prefactors, axis=(axis+1)%2)
        return self.__reduce_gram__(gram, axis)
    
    def get_mean_var(self, keepdims = False):
        mean = self.reduce_gram(self.inspace_points, 0)
        variance_of_expectations = self.reduce_gram(self.inspace_points**2, 0) - mean**2
        var = self.k.var + variance_of_expectations

        if keepdims:
            return (mean, var)
        else:
            return (np.squeeze(mean), np.squeeze(var))
    
    def sum(self,):
        return FiniteVec(self.k, self.inspace_points, self.prefactors)
    
    @classmethod
    def construct_RKHS_Elem(cls, kern, inspace_points, prefactors = None):
        return cls(kern, inspace_points, prefactors, points_per_split = len(inspace_points))
    
    @classmethod
    def construct_RKHS_Elem_from_estimate(cls, kern, inspace_points, estimate = "support", unsigned = True, regul = 0.1):
        if False:
            
            prefactors = distr_estimate_optimization(kern, inspace_points, est=estimate, unsigned = unsigned, use_jax = True)
            return cls(kern, inspace_points, prefactors, points_per_split = len(inspace_points))
        else:
            assert(estimate == "support")

            G = kern(inspace_points) 
            G_inv = np.linalg.inv(G + np.eye(inspace_points.shape[0]) * inspace_points.shape[0] * regul)
            rval = cls.construct_RKHS_Elem(kern, inspace_points, prefactors = np.mean(G_inv @ G_inv @ G, 0)).normalized()
            if unsigned:
                return rval.unsigned_projection().normalized(), rval2.unsigned_projection().normalized()
            else:
                return rval, rval2
            
    
    def unsigned_projection(self):
        assert(len(self) == 1)
        return self.updated(unsigned_projection(self.inspace_points, self.prefactors, self.k, False))

    
    def __call__(self, argument):
        return inner(self, FiniteVec(self.k, argument, np.ones(len(argument))))


def unsigned_projection(support_points, factors, kernel, use_jax = False):
    import scipy as sp
    from numpy.random import rand
    if use_jax:
        from jax.numpy import dot
        from jax import grad
    else:
        from autograd.numpy import dot
        from autograd import grad

    G = kernel(support_points).astype(float)
    c = 2*np.dot(factors, G)
    cost = lambda f: dot(dot(f, G), f) - dot(c, f)
    res = sp.optimize.minimize(cost, rand(len(factors))+ 0.0001, jac = grad(cost), bounds = [(0., None)] * len(factors))

    return res["x"]

def distr_estimate_optimization(kernel, support_points, est="support", unsigned = True, use_jax = False):
    import scipy as sp
    from numpy.random import rand
    if use_jax:
        from jax.numpy import dot, log
        from jax.scipy.special import logsumexp
        from jax import grad
    else:
        from autograd.numpy import dot, log
        from autograd.scipy.special import logsumexp
        from autograd import grad

    G = kernel(support_points).astype(float)
    if est == "support":
        #minimum variance of solution evaluated in support points
        if not unsigned:
            cost = lambda f: (dot(f, G)**2).mean() - dot(f, G).mean()**2 
        else:
            def cost(f):
                tmp = logsumexp(log(f).reshape((f.size, 1)) + log(G), 0)
                log_second_moment = logsumexp(tmp * 2) - log(len(tmp))
                log_expect = logsumexp(tmp) - log(len(tmp))
                assert()
    elif est == "density":
        #minimum negative log likelihood of support_points under solution
        cost = lambda f: -log(dot(f, G)).sum()

    if unsigned:
        bounds = [(0., None)] * len(support_points)
    else:
        assert(est == "support")
        bounds = None

    g_cost = grad(cost)
    res = sp.optimize.minimize(lambda x: cost(x), rand(len(support_points))+ 0.0001, jac = lambda x: g_cost(x), bounds = bounds)

    return res["x"]/res["x"].sum()

V1T = TypeVar("V1T")
V2T = TypeVar("V2T")

class CombVec(Vec, Generic[V1T, V2T]):
    def __init__(self, v1:V1T, v2:V2T, operation):
        assert(len(v1) == len(v2))
        self.__len = len(v1)
        (self.v1, self.v2) = (v1, v2)
        self.operation = operation

    def inner(self, Y:"CombVec[V1T, V2T]"=None, full=True):
        if Y is None:
            Y = self
        else:
            assert(Y.operation == self.operation)
        return self.operation(self.v1.inner(Y.v1), self.v2.inner(Y.v2))

    def __len__(self):
        return self.__len

    def updated(self, prefactors):
        raise NotImplementedError()


def inner(X, Y=None, full=True):
    return X.inner(Y, full)

