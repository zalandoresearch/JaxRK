import jax.numpy as np
from jax import jit, vmap, lax

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
        assert realization_num_obs[-1] == len(obs_points), "realization_num_obs[-1] expected to be exactly the number of observed points" 

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
            self.__reduce_sum_func = lambda x: np.mean(x * self.gram_idx[-len(x):], axis = 0, keepdims = True) 
            self.__len = len(self.process_boundaries[:-1])
        else:
            if kern_idx is None:
                def reduce_func(x):
                    return np.cumsum(x, axis = 0) / np.arange(1, x.shape[0] + 1).reshape((-1, 1))
            else:
                def reduce_func(x):
                    return cumsum_after_idxkern(x, self.gram_idx) / np.arange(1, x.shape[0] + 1).reshape((-1, 1))
            self.__reduce_sum_func = reduce_func
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
    outres = np.outer(idx_gram, gram).reshape((idx_gram.shape[0], gram.shape[0], gram.shape[1])).sum(2)
    outres_cs = np.cumsum(np.flip(np.cumsum(outres, 1), 0), 0)
    if return_before_sum:
        sum_func = lambda x: x
        arr_func = sum_func
    else:
        sum_func = lambda x: np.sum(x, 0)
        arr_func = np.array
    
    sumup_1 = lambda i: sum_func(outres[np.arange(idx_gram.size)[-i:], np.arange(idx_gram.size)[:i]])
    def append_trace(i, val):
        val.append(np.trace(outres_cs, -i))
        return val
    


    for i in range(1, gram.shape[0]+1):
        print(i, np.arange(idx_gram.size)[-i:], np.arange(idx_gram.size)[:i])
    for i in range(1, gram.shape[0]+1):
        print(i, sumup_1(i))
    print(lax.fori_loop(0, gram.shape[0], append_trace, []))
    print(outres_cs, outres)
    return np.array(lax.fori_loop(0, gram.shape[0], append_trace, []))
    #assert()
    #print([sumup_2(i) for i in range(1, gram.shape[0]+1)])
    #print([sumup_1(i) for i in range(1, gram.shape[0]+1)])
    rval = arr_func([sumup_1(i) for i in range(1, gram.shape[0]+1)])
    #tmp = np.zeros(gram.shape[0], *gram.shape)


    #rval = sumup_vmap(np.arange(1, gram.shape[0]+1))
    #if return_before_sum is False:
    #    assert rval.shape == gram.shape
    return rval

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
    def __init__(self, kern, idx_obs_points, realization_num_obs, prefactors = None, use_subtrajectories = True):
        self.k = kern
        self.inspace_points = idx_obs_points

        
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

        if Y is not None:
            assert(self.k == Y.k)
        else:
            Y = self
        gram = self.k(self.inspace_points, Y.inspace_points).astype(float)
        r1 = self.reduce_gram(gram, axis = 0)
        r2 = Y.reduce_gram(r1, axis = 1)
        return r2
    
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

