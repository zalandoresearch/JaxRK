from dataclasses import dataclass
import jax.numpy as np, jax.scipy as sp
from jax.scipy.special import expit, logit
from typing import Callable, Union
from jaxrk.core.typing import Bijection

from jaxrk.core.module import Module

@dataclass
class Sigmoid(Bijection):
    lower_bound:float
    upper_bound:float
    def setup(self,):
        assert self.lower_bound < self.upper_bound

    def __call__(self, x):
        ex = np.exp(-x)
        return (self.upper_bound + self.lower_bound * ex) / (1 + ex)

    def inv(self, y):
        return -np.log((self.upper_bound - self.lower_bound)/(y - self.lower_bound) - 1)

class SoftPlus(Bijection):
    def __call__(self, x):
        return np.where(x >= 1, x, np.log(1 + np.exp(x - 1)))

    def inv(self, y):
        return np.where(y >= 1, y, np.log(np.exp(y) - 1) + 1)

@dataclass
class CholeskyBijection(Bijection):
    diag_bij:Bijection = SoftPlus()
    lower:bool = True

    def _standardize(self, inp):
        assert len(inp.shape) == 2
        assert inp.shape[0] == inp.shape[1]
        if self.lower:
            return inp
        else:
            return inp.T

    def param_to_chol(self, param):
        param = self._standardize(param)

        return np.tril(param, -1) + self.diag_bij(np.diagonal(param))

    def chol_to_param(self, chol):
        chol = self._standardize(chol)
        return np.tril(chol, -1) + self.diag_bij.inv(np.diagonal(chol))

    def __call__(self, x):
        c = self.param_to_chol(x)
        return c @ c.T
    
    def inv(self, y):
        return self.chol_to_param(sp.linalg.cholesky(y, lower=True))

    
    