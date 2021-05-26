import copy
from ..reduce.base import Center
from typing import List, Union, Iterable, Iterator

import jax.numpy as np
from jax import jit, lax, vmap
from jax.ops import index, index_add, index_update
from ..reduce import Reduce, NoReduce
from ..rkhs.operator import Cmo
from ..rkhs.vector import CombVec, FiniteVec

from .base import Map, RkhsObject, Vec


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
                       use_subtrajectories = True, use_inner = "gen_gauss", reduce:Reduce = []):
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
            self.__len = Reduce.final_len(len(self.process_boundaries[:-1]), reduce)
        else:
            def reduce_func(x):
                    return np.cumsum(x, axis = 0) / np.arange(1, x.shape[0] + 1).reshape((-1, 1))
            self.__reduce_sum_func = reduce_func
            self.__len = Reduce.final_len(self.process_boundaries[-1], reduce)
        self._reduce = reduce



    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                other.inspace_points.shape == self.inspace_points.shape and
                np.all(other.inspace_points == self.inspace_points) and
                other.k == self.k and
                self._reduce == other._reduce )

    def __len__(self):
        return self.__len

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
    
    def extend_reduce(self, r:List[Reduce]) -> "SpVec":
        if r is None or len(r) == 0:
            return self
        else:
            _r = copy(self._reduce)
            _r.extend(r)
            return SpVec(self.k, self.inspace_points, self.process_boundaries, self.use_subtrajectories, self.use_inner, _r)

    def centered(self) -> "SpVec":
        return self.extend_reduce([Center()])
 
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
        return Reduce.apply(gram, self._reduce, axis)

class UpdatableSpVecInner(object):
    def __init__(self, unchanging:SpVec, updatable:SpVec):
        assert len(updatable) == 1 and updatable.use_subtrajectories is False
        self.unchanging = unchanging
        self.updatable = updatable
        self._num_obs = updatable.inspace_points.shape[0]
        self.current_raw = self.unchanging._inner_raw(updatable)
        self.current_gram = self.unchanging.inner(self.updatable, raw_cache = self.current_raw).squeeze()

    def update(self, new_idx, new_obs):
        new_idx_obs = np.concatenate((new_idx, np.atleast_1d(new_obs)), 0).reshape(1,-1)

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

class RolloutIdx(Iterator[np.array], Iterable[np.array]):
    def __init__(self, start:np.array, periodicities:np.array = None, stepsizes:np.array = None, example:np.array = None, example_set_periodicity = False):
        """An IndexRollout object zig-zags through indices with periodicities given in initialization.
        At most one periodicity can be NaN, in which case it is taken to be ever-increasing

        Args:
            periodicities (np.array): The periodicities, the first one can be np.inf. When a digit would become larger than its periodicity, the previous digit is increased
            stepsizes (np.array, optional): [description]. Defaults to None. Step sizes for increasing the index.
            example (np.array, optional): [description]. Defaults to None. Examples of consecutive indices (in rows) for inferring step sizes.
        """
        self.current = start
        
        if example is not None:
            self.stepsizes = np.array([np.median(x[x>0]) for x in (example[1:] - example[:-1]).T])
            if example_set_periodicity:
                self.periodicities = example.max(0) + self.stepsizes
            else:
                self.periodicities = np.array([np.inf] * self.stepsizes.size)
        else:
            assert stepsizes is not None and periodicities is not None
            self.stepsizes = stepsizes
            self.periodicities = periodicities

    def __next__(self) -> np.array:
        assert self.current.shape == self.periodicities.shape
        rval = self.current
        for i in range(self.current.size - 1, -1, -1):
            upd = self.current[i] + self.stepsizes[i]
            if upd >= self.periodicities[i]:
                rval = rval.at[i].set(0.)
            else:
                rval = rval.at[i].set(upd)
                break
        self.current = rval
        return rval
    
    def next(self) -> np.array:
        return self.__next__()
    
    def __iter__(self) -> Iterator[np.array]:
        return self


class RolloutSp(object):
    def __init__(self,
                 cm_op:Union[Cmo[SpVec, FiniteVec], Cmo[CombVec[SpVec, FiniteVec], FiniteVec]],
                 initial_spvec:SpVec,
                 dim_index:int,
                 idx_ro:RolloutIdx = None):
        assert len(initial_spvec) == 1
        self._cmo = cm_op
        self._idx_ro = idx_ro

        if cm_op.inp_feat.__class__ == CombVec:
            self.uinner = UpdatableSpVecInner(self._cmo.inp_feat.v1, initial_spvec)
            self.current_outp_emb = self._cmo.outp_feat.sum(True)
            self.get_embedding = self.__get_embedding_CombVec
            self.__update_current_outp_emb = lambda : None
        else:
            self.uinner = UpdatableSpVecInner(self._cmo.inp_feat, initial_spvec)
            self.current_outp_emb =  FiniteVec.construct_RKHS_Elem(self._cmo.outp_feat.k,
                                                                    self._cmo.outp_feat.inspace_points,
                                                                    np.squeeze(self._cmo.matr @ self.uinner.current_gram))
            self.get_embedding = self.__get_embedding_SpVec
            self.__update_current_outp_emb = self.__update_current_outp_emb_SpVec



    def update(self, new_obs, new_idx = None):
        if new_idx is None:
            assert self._idx_ro is not None
            new_idx = self._idx_ro.next()

        self.uinner.update(new_idx, new_obs)
        
        self.__update_current_outp_emb()
    
    def rollout(self, num_timesteps:int):
        rval = []
        rval_idx = []
        for i in range(num_timesteps):
            idx = self._idx_ro.next()
            new_point = self.current_outp_emb.point_representant()
            assert new_point.size == 1
            rval_idx.append(idx)
            rval.append(new_point)
            self.update(new_point, idx)
        return (np.array(rval_idx), np.array(rval))
    
    def __update_current_outp_emb_SpVec(self):
        assert len(self.uinner.current_gram.shape) == 1 or self.uinner.current_gram.shape[1] == 1
        new_map = (self._cmo.matr @ self.uinner.current_gram).squeeze()
        assert len(new_map.shape) == 1
        self.current_outp_emb = self.current_outp_emb.updated(new_map[np.newaxis, :])
    
    def __get_embedding_SpVec(self, idx = None):
        return self.current_outp_emb


    def __get_embedding_CombVec(self, idx = None):
        if idx is None:
            idx = self._idx_ro.current
        
        idx_gram = self._cmo.inp_feat.v2(np.atleast_2d(idx)).squeeze()
        assert len(self.uinner.current_gram.shape) == 1 and len(idx_gram.shape) == 1
        inp_gram = self._cmo.inp_feat.combine(self.uinner.current_gram, idx_gram)
        inp_gram = self._cmo.inp_feat.reduce_gram(inp_gram[:, np.newaxis], 0)#.squeeze()
        assert len(self.uinner.current_gram.shape) == 1 or self.uinner.current_gram.shape[1] == 1
        return self.current_outp_emb.updated((self._cmo.matr @ inp_gram).T)

