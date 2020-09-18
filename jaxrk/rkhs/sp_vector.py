import jax.numpy as np
from jax import jit, vmap, lax

from .base import Vec, Map, RkhsObject

from jaxrk.rkhs.vector import FiniteVec, CombVec
from jaxrk.rkhs.operator import Cmo
from jaxrk.reduce import GramReduce, NoReduce

class SpVec(Vec):
    # FIXME: Reference index enters Kernel regression as input; with shift-invariant kernel we get dependence on reference index
    # other idea: different kernel on the embedded "joint"
    """
        RKHS feature vector for stochastic processes (sp).

        Parameters
        ==========
        kern                - kernel for observations and indizes
        idx_obs_points      - all observations of all process realizations, each realization is ordered by ascending index, followed by the next realization 
        realization_num_obs - number of observations for each realization in a list
        use_subtrajectories - whether to use subtrajectories
        use_inner           - which inner product to use for comparing trajectories
        gram_reduce         - which reduction to apply to gram matrix
    """
    def __init__(self, kern, idx_obs_points, realization_num_obs,
                       use_subtrajectories = True, use_inner = "gen_gauss", gram_reduce:GramReduce = NoReduce()):
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
        self._gram_reduce = gram_reduce



    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                other.inspace_points.shape == self.inspace_points.shape and
                np.all(other.inspace_points == self.inspace_points) and
                other.k == self.k and
                self._gram_reduce == other._gram_reduce )

    def __len__(self):
        if self._gram_reduce is None:
            return self.__len
        else:
            return self._gram_reduce.new_len(self.__len)

    def _inner_raw(self, Y=None, full=True):
        if not full and Y is not None:
            raise ValueError(
                "Ambiguous inputs: `diagonal` and `y` are not compatible.")
        assert(full)

        if Y is not None and Y != self:
            assert(self.k == Y.k and self.use_inner == Y.use_inner)
            gram_self = self.k(self.inspace_points).astype(float)
            gram_mix = self.k(self.inspace_points, Y.inspace_points).astype(float)
            gram_other = self.k(Y.inspace_points).astype(float)
        else:
            Y = self
            gram_self = gram_mix = gram_other = self.k(self.inspace_points).astype(float)
        
        r1 = self._raw_reduce_gram(gram_mix, axis = 0)
        gram_mix_red = Y._raw_reduce_gram(r1, axis = 1)
        if self.use_inner == "linear" or self.use_inner == "poly":
            return {"gram_mix_red": gram_mix_red}
        elif self.use_inner == "gen_gauss":
            gram_self_red = np.diagonal(self._raw_reduce_gram(self._raw_reduce_gram(gram_self, axis = 0), axis = 1)).reshape((-1, 1))
            gram_other_red = np.diagonal(Y._raw_reduce_gram(Y._raw_reduce_gram(gram_other, axis = 0), axis = 1)).reshape((1, -1))
            return {"gram_mix_red": gram_mix_red,
                    "gram_self_red": gram_self_red,
                    "gram_other_red": gram_other_red}
    
    def _inner_postprocess(self, raw):
        if self.use_inner =="gen_gauss": 
            dist = np.clip(raw["gram_self_red"] + raw["gram_other_red"] - 2 * raw["gram_mix_red"], 0)
            return np.exp(-dist**0.6) # generalized gaussian
        elif self.use_inner =="linear": 
            return 500*raw["gram_mix_red"]
        elif self.use_inner == "poly":
            return (raw["gram_mix_red"] + 1 )**3

    def inner(self, Y=None, full=True, raw_cache = None):
        if raw_cache is None:
            raw_cache = self._inner_raw(Y, full)
            return self.inner(Y, full, raw_cache)
        gram = self._inner_postprocess(raw_cache)
        r1 = self.reduce_gram(gram, axis = 0)
        if Y is None:
            Y = self
        r2 = Y.reduce_gram(r1, axis = 1)
        return r2
    
    def updated(self, prefactors):
       raise(NotImplementedError)
 
    def _raw_reduce_gram(self, gram, axis = 0):
        if axis != 0:
            gram = np.swapaxes(gram, axis, 0)
        rval = []         
        for beg_proc, end_proc in np.array([self.process_boundaries[:-1], self.process_boundaries[1:]]).T:
            rval.append(self.__reduce_sum_func(gram[beg_proc:end_proc]))
        rval = np.vstack(rval)
        if axis != 0:
            rval = np.swapaxes(rval, axis, 0)
        return rval
    
    def reduce_gram(self, gram, axis = 0):
        return self._gram_reduce(gram, axis)

class UpdatableSpVecInner(object):
    def __init__(self, unchanging:SpVec, updatable:SpVec):
        self.unchanging = unchanging
        self.updatable = updatable
        self._num_obs = updatable.inspace_points.shape[0]
        self.current_raw = self.unchanging._inner_raw(updatable)
        self.current_gram = self.unchanging.inner(self.updatable, raw_cache = self.current_raw).squeeze()

    def update(self, new_idx, new_obs):
        new_idx_obs = np.array((new_idx, new_obs)).reshape(1,-1)

        gram_mixed = self.current_raw["gram_mix_red"]
        upd_mixed = self.unchanging._raw_reduce_gram(self.unchanging.k(self.unchanging.inspace_points, new_idx_obs))
        new_gram_mixed = (gram_mixed * self._num_obs + upd_mixed)/(self._num_obs + 1)
        self.current_raw["gram_mix_red"] = new_gram_mixed
        
        if self.unchanging.use_inner == "gen_gauss":

            history_newpoint = self.updatable._raw_reduce_gram(self.updatable.k(self.updatable.inspace_points,
                                                                                     new_idx_obs))
            upd_self = self.updatable.k(new_idx_obs) + 2 * self._num_obs * history_newpoint

            self.current_raw["gram_other_red"] = (self.current_raw["gram_other_red"] * self._num_obs**2 + upd_self) / (self._num_obs + 1)**2

        self.updatable = SpVec(self.unchanging.k,
                                    np.vstack([self.updatable.inspace_points, new_idx_obs]),
                                    np.array([self._num_obs + 1]),
                                    use_subtrajectories = False,
                                    use_inner=self.unchanging.use_inner)
        self._num_obs += 1
        self.current_gram = self.unchanging.inner(self.updatable, raw_cache = self.current_raw).squeeze()

class RolloutSpVec(object):
    def __init__(self, cm_op:Cmo[SpVec, FiniteVec], initial_spvec:SpVec, dim_index):
        assert(len(initial_spvec) == 1)
        self._inc = (initial_spvec.inspace_points[1:, :dim_index] -  initial_spvec.inspace_points[:-1, :dim_index]).mean(0)
        self._cmo = cm_op
        self._next_idx = initial_spvec.inspace_points[-1,  :dim_index] + self._inc

        self.uinner = UpdatableSpVecInner(self._cmo.inp_feat, initial_spvec)
        gram = self._cmo.inp_feat.inner(initial_spvec, raw_cache = self.uinner.current_raw)
        
        self.current_outp_emb =  FiniteVec.construct_RKHS_Elem(self._cmo.outp_feat.k,
                                                                self._cmo.outp_feat.inspace_points,
                                                                np.squeeze(self._cmo.matr @ gram))

    def update(self, new_obs = None, new_idx = "auto"):
        assert new_obs is not None
        if new_idx is None or new_idx == "auto":
            new_idx = self._next_idx        
        self._next_idx = new_idx + self._inc

        self.uinner.update(new_idx, new_obs)

        self.current_outp_emb = self.current_outp_emb.updated(self._cmo.matr @ self.uinner.current_gram)

class RolloutCombVec(object):
    def __init__(self, cm_op:Cmo[CombVec[SpVec, FiniteVec], FiniteVec], initial_spvec:SpVec, dim_index):
        assert(len(initial_spvec) == 1)
        self._inc = (initial_spvec.inspace_points[1:, :dim_index] -  initial_spvec.inspace_points[:-1, :dim_index]).mean(0)
        self._cmo = cm_op
        self._next_idx = initial_spvec.inspace_points[-1,  :dim_index] + self._inc

        self.uinner = UpdatableSpVecInner(self._cmo.inp_feat.v1, initial_spvec)
        gram = self._cmo.inp_feat.v1.inner(initial_spvec, raw_cache = self.uinner.current_raw)
        


    def update(self, new_obs = None, new_idx = "auto"):
        assert new_obs is not None
        if new_idx is None or new_idx == "auto":
            new_idx = self._next_idx        
        self._next_idx = new_idx + self._inc

        self.uinner.update(new_idx, new_obs)

    def get_embedding(self, idx):
        inp_gram = self._cmo.inp_feat.reduce_gram(self._cmo.inp_feat.combine(self.uinner.current_gram, self._cmo.inp_feat.v2(idx)))
        return self.current_outp_emb.updated(self._cmo.matr @ inp_gram)