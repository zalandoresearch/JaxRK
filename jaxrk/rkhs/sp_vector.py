import jax.numpy as np
from jax import jit, vmap, lax

from .base import Vec, Op, RkhsObject

class SpVec(Vec):
    # FIXME: Reference index enters Kernel regression as input; with shift-invariant kernel we get dependence on reference index
    # other idea: different kernel on the embedded "joint"
    """
        RKHS feature vector for stochastic processes (sp).

        Parameters
        ==========
        kern - kernel for observations and indizes
        idx_obs_points - all observations of all process realizations, each realization is ordered by ascending index, followed by the next realization 
        realization_num_obs - number of observations for each realization in a list

        kern_idx - kernel for comparing observation index to reference point (0)
        prefactors - prefactor/weight for realization. If use_subtrajectories is True, prefactors for subtrajectories are expected.
        use_subtrajectories - whether to use subtrajectories
    """
    def __init__(self, kern, idx_obs_points, realization_num_obs, prefactors = None, use_subtrajectories = True, use_inner = "gen_gauss"):
        self.k = kern
        self.inspace_points = idx_obs_points

        self.use_inner = use_inner
        
        self.use_subtrajectories = use_subtrajectories

        assert(len(realization_num_obs.shape) <= 1)
        assert realization_num_obs[-1] == len(idx_obs_points), "realization_num_obs[-1] expected to be exactly the number of observed points" 

        if realization_num_obs[0] == 0:
            self.process_boundaries = realization_num_obs
        else:
            self.process_boundaries = np.hstack([[0], realization_num_obs])


        self.use_subtrajectories = use_subtrajectories

        if not use_subtrajectories:
            self.__reduce_sum_func = lambda x: np.mean(x, axis = 0, keepdims = True) 
            self.__len = len(self.process_boundaries[:-1])
        else:
            def reduce_func(x):
                    return np.cumsum(x, axis = 0) / np.arange(1, x.shape[0] + 1).reshape((-1, 1))
            self.__reduce_sum_func = reduce_func
            self.__len = self.process_boundaries[-1]

        if prefactors is None:
            prefactors = np.ones(self.__len)/self.__len

        assert(prefactors.shape[0] == self.__len)
        assert(len(prefactors.shape) == 1)
        
        self.prefactors = prefactors

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                np.all(other.prefactors == self.prefactors) and
                np.all(other.inspace_points == self.inspace_points) and
                other.k == self.k )

    def __len__(self):
        return self.__len
    
    def inner(self, Y=None, full=True):
        if not full and Y is not None:
            raise ValueError(
                "Ambiguous inputs: `diagonal` and `y` are not compatible.")
        assert(full)

        if Y is not None and Y != self:
            assert(self.k == Y.k)
            gram_self = self.k(self.inspace_points).astype(float)
            gram_mix = self.k(self.inspace_points, Y.inspace_points).astype(float)
            gram_other = self.k(Y.inspace_points).astype(float)
        else:
            Y = self
            gram_self = gram_mix = gram_other = self.k(self.inspace_points).astype(float)
        
        r1 = self.reduce_gram(gram_mix, axis = 0)
        gram_mix_red = Y.reduce_gram(r1, axis = 1)
        gram_self_red = np.diagonal(self.reduce_gram(self.reduce_gram(gram_self, axis = 0), axis = 1)).reshape((-1, 1))
        gram_other_red = np.diagonal(Y.reduce_gram(Y.reduce_gram(gram_other, axis = 0), axis = 1)).reshape((1, -1))
        rval = {"gen_gauss":  np.exp(-(gram_self_red + gram_other_red - 2 * gram_mix_red)**0.9), # generalized gaussian
                "linear": gram_mix_red,
                "poly": (gram_mix_red + 1 )**10}
        return rval[self.use_inner]
    
    def updated(self, prefactors):
        assert(len(self.prefactors) == len(prefactors))
        return SpVec(self.k, self.inspace_points, self.process_boundaries, self.prefactors, self.use_subtrajectories)
 
    def reduce_gram(self, gram, axis = 0):
        if axis != 0:
            gram = np.swapaxes(gram, axis, 0)
        rval = []         
        for beg_proc, end_proc in np.array([self.process_boundaries[:-1], self.process_boundaries[1:]]).T:
            rval.append(self.__reduce_sum_func(gram[beg_proc:end_proc]))
        rval = np.vstack(rval)
        if axis != 0:
            rval = np.swapaxes(rval, axis, 0)
        return rval

