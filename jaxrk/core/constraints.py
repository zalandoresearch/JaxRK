from dataclasses import dataclass
import jax.numpy as np, jax.scipy as sp
from jax.scipy.special import expit, logit
from typing import Callable, Union
from ..core.typing import Bijection
import jax.lax as lax


class Sigmoid(Bijection):
    def __init__(self, lower_bound:float, upper_bound:float):
        assert lower_bound < upper_bound
        assert lower_bound is not None and upper_bound is not None
        self.lower_bound, self.upper_bound = lower_bound, upper_bound

    def __call__(self, x):
        ex = np.exp(-x)
        return (self.upper_bound + self.lower_bound * ex) / (1 + ex)

    def inv(self, y):
        return -np.log((self.upper_bound - self.lower_bound)/(y - self.lower_bound) - 1)
class SoftPlus(Bijection):
    def __init__(self, lower_bound:float = 0.):
        assert lower_bound is not None
        self.lower_bound = lower_bound

    def __call__(self, x):
        return np.where(x >= 1, x, np.log(1 + np.exp(x - 1))) + self.lower_bound

    def inv(self, y):
        y = y - self.lower_bound
        return np.where(y >= 1, y, np.log(np.exp(y) - 1) + 1)


def squareplus(x):
  return lax.mul(0.5, lax.add(x, lax.sqrt(lax.add(lax.square(x), 4.))))
class SoftMinus(Bijection):
    def __init__(self, upper_bound:float = 0.):
        assert upper_bound is not None
        self.sp = SoftPlus(-upper_bound)
    
    def __call__(self, x):
        return self.sp.__call__(-x)
    
    def inv(self, y):
        return -self.sp.inv(y)
    
    
def SoftBd(lower_bound:float = None, upper_bound:float = None) -> Union[Sigmoid, SoftPlus, SoftMinus]:
    if lower_bound is None:
        assert upper_bound is not None, "Require one bound."
        return SoftMinus(upper_bound)
    elif upper_bound is None:
        assert lower_bound is not None, "Require one bound."
        return SoftPlus(lower_bound)
    else:
        return Sigmoid(lower_bound, upper_bound)

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

    
    
