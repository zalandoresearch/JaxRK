from abc import ABC, abstractmethod
import jax.numpy as np

from typing import Union

from jaxrk.core.constraints import SoftPlus
from jaxrk.core.init_fn import ConstFn
from jaxrk.core.typing import *
from jaxrk.utilities.distances import dist

class Scaler(ABC):
    @abstractmethod
    def __call__(self, inp):
        raise NotImplementedError()
    
    @abstractmethod
    def inv(self):
        raise NotImplementedError()
    
    @abstractmethod
    def scale(self):
        raise NotImplementedError()
class NoScaler(Scaler):
    def __call__(self, inp):
        return inp
    
    def inv(self):
        return np.ones(1)
    
    def scale(self):
        return np.ones(1)

class SimpleScaler(Scaler):
    def __init__(self, scale:Union[Array, float]):
        """Scale input either by global scale parameter or by per-dimension scaling parameters

        Args:
            scale (Union[Array, float]): Scaling parameter(s).
        """
        super().__init__()
        if isinstance(scale, float):
            scale = np.array([[scale]])        
        assert np.all(scale > 0.)
        self.s = scale
    
    @classmethod
    def make_unconstr(cls, scale:Union[Array, float], bij: Bijection = SoftPlus()) -> "SimpleScaler":
        return SimpleScaler(bij(scale))

    def inv(self):
        return 1./self.s
    
    def scale(self):
        return self.s

    def __call__(self, inp):
        if inp is None:
            return None
        assert len(inp.shape) == len(self.scale().shape)
        
        return self.s * inp

class ScaledPairwiseDistance:
    """A class for computing scaled pairwise distance for stationary/RBF kernels, depending only on

        dist = ‖x - x'‖^p

    For some power p.
    """


    def __init__(self,
                 scaler:Scaler = NoScaler(),
                 power:float = 2.):
        """Compute scaled pairwise distance, given by
            |X_i-Y_j|^p for all i, j

        Args:
            scaler (Scaler, optional): Scaling module. Defaults to NoScaler().
            power (float, optional): Power p that the pairwise distance is taken to. Defaults to 2..
        """
        super().__init__()
        self.power = power
        if scaler.scale().size == 1:
            self.gs = scaler
            self.ds = NoScaler()
            self.is_global = True
        else:
            self.gs = NoScaler()
            self.ds = scaler
            self.is_global = False
    
    def _get_scale_param(self):
        if self.is_global:
            return self.gs.inv()
        else:
            return self.ds.inv()**(1./self.power)

    def __call__(self, X, Y=None, diag = False,):
        if diag:
            if Y is None:
                rval = np.zeros(X.shape[0])
            else:
                assert(X.shape == Y.shape)
                rval = self.gs(np.sum(np.abs(self.ds(X) - self.ds(Y))**self.power, 1))
        else:
            rval = self.gs(dist(self.ds(X), self.ds(Y), power = self.power))
        return rval