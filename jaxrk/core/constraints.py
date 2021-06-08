from dataclasses import dataclass
import jax.numpy as np, jax.scipy as sp
from jax.scipy.special import expit, logit
from typing import Callable, Union

from numpy.core.defchararray import upper
from ..core.typing import Bijection
import jax.lax as lax



def squareplus(x):
    #(x + np.sqrt(x**2 + 4))/2
    return lax.mul(0.5, lax.add(x, lax.sqrt(lax.add(lax.square(x), 4.))))

def inv_squareplus(y):
    #(y**2-1)/y
    return lax.div(lax.sub(lax.square(y), 1.), y)

def squareplus_deriv(x):
    #(1 + x/np.sqrt(4 + x**2))/2
    return lax.mul(0.5, lax.add(lax.div(x, lax.sqrt(lax.add(lax.square(x), 4.))), 1.))



class SquashingToBounded(Bijection):
    def __init__(self, lower_bound:float, upper_bound:float, f = squareplus_deriv):
        assert lower_bound < upper_bound
        assert lower_bound is not None and upper_bound is not None
        self.lower_bound = lower_bound
        self.scale = upper_bound - lower_bound
        self.f = f

    def __call__(self, x):
        return self.scale * np.clip(self.f(x), 0., 1.) + self.lower_bound

    def inv(self, y):
        raise NotImplementedError
        return -np.log((self.upper_bound - self.lower_bound)/(y - self.lower_bound) - 1)

class NonnegToLowerBd(Bijection):
    def __init__(self, lower_bound:float = 0., f = squareplus ):
        assert lower_bound is not None
        self.lower_bound = lower_bound
        self.f = f

    def __call__(self, x):
        return np.clip(self.f(x), 0.) + self.lower_bound
    
    def inv(self, y):
        raise NotImplementedError
        y = y - self.lower_bound
        return self.f_inv(y)

class FlipLowerToUpperBound(Bijection):
    def __init__(self, upper_bound:float, lb_bij:Callable[..., Bijection]):
        assert upper_bound is not None
        self.lb = lb_bij(-upper_bound)
    
    def __call__(self, x):
        return -self.lb.__call__(-x)
    
    def inv(self, y):
        return -self.lb.inv(y)

def NonnegToUpperBd(upper_bound:float = 0.):
    return FlipLowerToUpperBound(upper_bound, NonnegToLowerBd)
    
    
def SoftBound(l:float = None, u:float = None) -> Bijection:
    if l is None:
        assert u is not None, "Require one bound."
        return NonnegToUpperBd(u)
    elif u is None:
        assert l is not None, "Require one bound."
        return NonnegToLowerBd(l)
    else:
        return SquashingToBounded(l, u)

@dataclass
class CholeskyBijection(Bijection):
    diag_bij:Bijection = NonnegToLowerBd()
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

    
    
