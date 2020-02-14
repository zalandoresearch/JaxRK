import jax.numpy as np
from jax import jit


from .base import Vec, Op, RkhsObject

class SiEdSpVec(Vec):
    """
        RKHS feature vector for shift-invariant (si) stochastic processes (sp) with observations at equidistant (ed) indices.
        If kern_idx is given, use it on index of observation with reference point at index 0

        Parameters
        ==========
        kern_obs - kernel for observations
        obs_points - all observations of all process realizations, each realization is ordered by ascending index, followed by the next realization 
        realization_num_obs - number of observations for each realization in a list

        kern_idx - kernel for comparing observation index to reference point (0)
        prefactors - prefactor/weight for realization. If use_subtrajectories is True, prefactors for subtrajectories are expected.
        use_subtrajectories - whether to use subtrajectories
    """
    def __init__(self, kern_obs, obs_points, realization_num_obs, kern_idx = None, prefactors = None, use_subtrajectories = True):
        self.k_obs = kern_obs
        self.inspace_points = obs_points

        
        self.use_subtrajectories = use_subtrajectories

        assert(len(realization_num_obs.shape) <= 1)

        if realization_num_obs[0] == 0:
            self.process_boundaries = realization_num_obs
        else:
            self.process_boundaries = np.hstack([[0], realization_num_obs])

        max_obs = np.max(self.process_boundaries[1:] - self.process_boundaries[:-1])
        if kern_idx is None:
            self.gram_idx = np.ones((max_obs, 1))
        else:
            self.gram_idx = kern_idx(np.arange(-max_obs, 0).reshape((-1, 1)), np.zeros((1, 1)))

        self.k_idx = kern_idx
        self.use_subtrajectories = use_subtrajectories

        if not use_subtrajectories:
            self.__reduce_sum_func = lambda x: np.sum(x * self.gram_idx[-len(x):], axis = 0, keepdims = True) 
            self.__len = len(self.process_boundaries[:-1])
        else:
            if kern_idx is None:
                self.__reduce_sum_func = lambda x: np.cumsum(x, axis = 0)
            else:
                self.__reduce_sum_func = lambda x: cumsum_after_idxkern(x, self.gram_idx)
            self.__len = self.process_boundaries[-1]

        if prefactors is None:
            prefactors = np.ones(self.__len)

        assert(prefactors.shape[0] == self.__len)
        assert(len(prefactors.shape) == 1)
        
        self.prefactors = prefactors

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                np.all(other.prefactors == self.prefactors) and
                np.all(other.inspace_points == self.inspace_points) and
                other.k_obs == self.k_obs and
                np.all(other.gram_idx == self.gram_idx))

    def __len__(self):
        return self.__len
    
    def inner(self, Y=None, full=True):
        if not full and Y is not None:
            raise ValueError(
                "Ambiguous inputs: `diagonal` and `y` are not compatible.")
        assert(full)

        if Y is not None:
            assert(self.k_obs == Y.k_obs)
        else:
            Y = self
        gram = self.k_obs(self.inspace_points, Y.inspace_points).astype(float)
        r1 = self.reduce_gram(gram, axis = 0)
        r2 = Y.reduce_gram(r1, axis = 1)
        return r2
    
    def updated(self, prefactors):
        assert(len(self.prefactors) == len(prefactors))
        return SiEdSpVec(self.k_obs, self.inspace_points, self.process_boundaries, self.k_idx, self.prefactors, self.use_subtrajectories)
    
    
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


def cumsum_after_idxkern(gram, idx_gram, return_before_sum = False):
    outres = np.outer(idx_gram, gram).reshape((idx_gram.shape[0], gram.shape[0], gram.shape[1]))
    if return_before_sum:
        sum_func = lambda x: x
        arr_func = sum_func
    else:
        sum_func = lambda x: np.sum(x, 0)
        arr_func = np.array
    rval = arr_func([sum_func(outres[np.arange(idx_gram.size)[-i:], np.arange(idx_gram.size)[:i]]) for i in range(1, gram.shape[0]+1)])
    if return_before_sum is False:
        assert rval.shape == gram.shape
    return rval

class SiSpVec(Vec):
    """
        RKHS feature vector for shift-invariant (si) stochastic processes (sp)
    """
    def __init__(self, kern_idx, kern_obs, idx_points, obs_points, realization_num_obs, prefactors = None, shift_invariant_refpoints = None, use_subtrajectories = True):
        assert()
        self.k_idx = kern_idx
        self.k_obs = kern_obs
        if shift_invariant_refpoint is not None:
            self.idx_points = idx_points - shift_invariant_refpoint
        else:
            self.idx_points = idx_points
        self.obs_points = obs_points

        
        self.use_subtrajectories = use_subtrajectories

        assert(len(realization_num_obs.shape) <= 1)

        if realization_num_obs[0] == 0:
            self.process_boundaries = realization_num_obs
        else:
            self.process_boundaries = np.hstack([[0], np.realization_num_obs])

        if not use_subtrajectories:
            self.__reduce_sum_func = lambda x: np.sum(x, axis = 1, keepdims = True)
            self.__len = len(self.process_boundaries[:-1])
        else:
            self.__reduce_sum_func = lambda x: np.cumsum(x, axis = 1)
            self.__len = self.process_boundaries[-1]

        if prefactors is None:
            prefactors = np.ones(self.__len)
        elif prefactors == "kernel":
            assert()
            assert(shift_invariant_refpoint is not None)
            prefactors = self.k_idx(idx_points, shift_invariant_refpoint).squeeze()

        assert(prefactors.shape[0] == self.__len)
        assert(len(prefactors.shape) == 1)
        
        self.prefactors = prefactors

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                np.all(other.prefactors == self.prefactors) and
                np.all(other.inspace_points == self.inspace_points) and
                other.k == self.k)

    def __len__(self):
        return self.__len
    
    def inner(self, Y=None, full=True):
        if not full and Y is not None:
            raise ValueError(
                "Ambiguous inputs: `diagonal` and `y` are not compatible.")
        assert(full)

        if Y is not None:
            assert(self.k_idx == Y.k_idx and self.k_obs == Y.k_obs)
        else:
            Y = self
        gram = self.k_obs(self.obs_points, Y.obs_points).astype(float) * self.k_idx(self.idx_points, Y.idx_points).astype(float)
        #print("G",gram.shape)
        r1 = self.reduce_gram(gram, axis = 0)
        #print("r1",r1.shape)
        r2 = Y.reduce_gram(r1, axis = 1)
        #print("r2",r2.shape)
        return r2
    
    def updated(self, prefactors):
        assert(len(self.prefactors) == len(prefactors))
        return SpVec(self.k_idx, self.k_obs, self.idx_points, self.obs_points, self.prefactors)
    
    
    def reduce_gram_looped(self, gram, axis = 0):
        if axis != 0:
            gram = np.swapaxes(gram, axis, 0)
        rval = []         
        for i, beg_proc in enumerate(self.process_boundaries[:-1]):
            rval[i] = self.__reduce_sum_func(gram[beg_proc:self.process_boundaries[i + 1]])
        rval = np.array(rval)
        if axis != 0:
            rval = np.swapaxes(rval, axis, 0)
        return rval
