
"""
Created on Thu Jan 10 10:01:56 2019

@author: Ingmar Schuster
"""


from typing import Callable
from abc import ABC, abstractmethod

import jax.numpy as np
from jaxrk.core.typing import Array
from .base import Reduce

class CenterInpFeat(Reduce):
    """Center input for input features of a centered operator.
        To be applied to uncentered feature vector Φ = [Φ_1, …, Φ_n].

        Args:
            inp_feat_uncentered_gram (np.array): The output of inp_feat_uncentered.inner(), where inp_feat_uncentered == Φ.
    """
    inp_feat_uncentered_gram:Array

    @classmethod
    def __const_term_init(cls, g):
        assert len(g.shape) == 2
        assert g.shape[0] == g.shape[1]
        mean = g.mean(axis = 1, keepdims=True)
        return mean.mean() - mean


    def setup(self,):
        self.const_term = self.variable('constants', "const_term", CenterInpFeat.__const_term_init, self.inp_feat_uncentered_gram)

    def reduce_first_ax(self, inp:np.array) -> np.array:
        return inp - inp.mean(0, keepdims=True) + self.const_term

    def new_len(self, original_len:int) -> int:
        assert original_len == self.const_term.size
        return original_len

class DecenterOutFeat(Reduce):
    """Decenter output for output features of a centered operator.
       Based on prefactors α and to be applied to uncentered feature vector Ψ = [Ψ_1, …, Ψ_n] with mean μ, correctly calculate 
       μ(y) + Σ_i α_i(Ψ_i(y) - μ(y)) 
       when given
       [Ψ_1(y), …, Ψ_n(y)]
       as input.

        Args:
            lin_map (np.array): Linear map to apply to features. If there are n input features, expected to be of shape (m, n).
    """
    
    lin_map:Array

    def setup(self, ):       
        assert len(lin_map.shape) == 2
        self.corr_fact = (1. - np.sum(self.lin_map, 1, keepdims=True))

    def reduce_first_ax(self, inp:np.array) -> np.array:
        return self.corr_fact * np.mean(inp, axis = 0, keepdims = True) + self.lin_map @ inp

    def new_len(self, original_len:int) -> int:
        original_len == len(self.lin_map)
        return original_len
