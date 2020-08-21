"""
Created on Thu Jan 10 10:01:56 2019

@author: Ingmar Schuster
"""


from typing import Callable
from abc import ABC, abstractmethod

import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats
from jax.numpy import exp, log, sqrt
from jax.scipy.special import logsumexp




class GramReduce(Callable, ABC):
    """The basic reduction type."""
    def __call__(self, gram:np.array, axis:int = 0) -> np.array:
        if axis != 0:
            gram = np.swapaxes(gram, axis, 0)
        rval = self.reduce_first_ax(gram)
        if axis != 0:
            rval = np.swapaxes(rval, axis, 0)
        return rval

    @abstractmethod
    def reduce_first_ax(self, gram:np.array) -> np.array:
        pass

    @abstractmethod
    def new_len(self, original_len:int) -> int:
        pass

class NoReduce(Callable, ABC):
    def __call__(self, gram:np.array, axis:int = 0) -> np.array:
        return gram

    def reduce_first_ax(self, gram:np.array) -> np.array:
        return gram
    
    def new_len(self, original_len:int) -> int:
        return original_len