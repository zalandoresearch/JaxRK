
"""
Created on Thu Jan 10 10:01:56 2019

@author: Ingmar Schuster
"""


from typing import Callable
from abc import ABC, abstractmethod

import jax.numpy as np

from .base import Reduce

class CenterInpFeat(Reduce):
    def __init__(self, inp_feat_uncentered_gram:np.array) -> None:
        """Center input for input features of a centered operator.
        To be applied to uncentered feature vector Φ = [Φ_1, …, Φ_n].

        Args:
            inp_feat_uncentered_gram (np.array): The output of inp_feat_uncentered.inner(), where inp_feat_uncentered == Φ.
        """
        assert len(inp_feat_uncentered_gram.shape) == 2
        assert inp_feat_uncentered_gram.shape[0] == inp_feat_uncentered_gram.shape[1]
        mean = inp_feat_uncentered_gram.mean(axis = 1, keepdims=True)
        self.const_term = mean.mean() - mean 

    def reduce_first_ax(self, inp:np.array) -> np.array:
        return inp - inp.mean(0, keepdims=True) + self.const_term

    def new_len(self, original_len:int) -> int:
        assert original_len == self.const_term.size
        return original_len

class DecenterOutFeat(Reduce):

    def __init__(self, lin_map:np.array) -> None:
        """Decenter output for output features of a centered operator.
       Based on prefactors α and to be applied to uncentered feature vector Ψ = [Ψ_1, …, Ψ_n] with mean μ, correctly calculate 
       μ(y) + Σ_i α_i(Ψ_i(y) - μ(y)) 
       when given
       [Ψ_1(y), …, Ψ_n(y)]
       as input.

        Args:
            lin_map (np.array): Linear map to apply to features. If there are n input features, expected to be of shape (m, n).
        """
        assert len(lin_map.shape) == 2

        self.lin_map = lin_map
        self.corr_fact = (1. - np.sum(self.lin_map, 1, keepdims=True))

    #def __call__(self, inp:np.array, axis:int = 0) -> np.array:
    #    return (
    #            + np.sum(inp.astype(self.prefactors.dtype) * np.expand_dims(self.prefactors, axis=(axis+1)%2), axis = axis))

    def reduce_first_ax(self, inp:np.array) -> np.array:
        return self.corr_fact * np.mean(inp, axis = 0, keepdims = True) + self.lin_map @ inp

    def new_len(self, original_len:int) -> int:
        original_len == len(self.lin_map)
        return original_len
