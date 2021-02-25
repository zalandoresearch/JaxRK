import jax.numpy as np

from jaxrk.core import Module
from jaxrk.core.constraints import SoftPlus
from jaxrk.core.init_fn import ConstFn
from jaxrk.core.typing import *
from jaxrk.utilities.eucldist import eucldist

class NoScaler(Module):
    def __call__(self, inp):
        return inp
    
    def inv(self):
        return 1.
    
    def scale(self):
        return 1.

class Scaler(Module):
    bij: Bijection = SoftPlus()
    init: ConstOrInitFn = ConstFn(np.ones(1))
    scale_shape: Shape = (1, 1)


    def setup(self):
        self.s = self.bij(self.const_or_param("s", self.init, self.scale_shape, np.float32, self.bij))

    def inv(self):
        return 1./self.s
    
    def scale(self):
        return self.s

    def __call__(self, inp):
        assert len(inp.shape) == len(self.scale_shape)
        
        return self.s * inp

class ScaledPairwiseDistance(Module):
    """A class for computing scaled pairwise distance for stationary/RBF kernels, depending only on

        dist = ‖x - x'‖^p

    For some power p.
    """

    scale:bool = True
    power:float = 2.
    scale_dim:int = 1
    scale_init:ConstOrInitFn = ConstFn(np.ones(1))

    def setup(self,):
        assert self.scale_dim >= 1
        if not self.scale:
            self.ds = self.gs = NoScaler()
        else:
            if self.scale_dim > 0:
                self.gs = Scaler(init = self.scale_init)
                self.ds = NoScaler()
            else:
                self.gs = NoScaler()
                self.ds = Scaler(scale_shape = (1, self.scale_dim), init = self.scale_init)  


    
    def _get_scale_param(self):
        if self.scale_dim > 1:
            return self.gs.inv()**(1./self.power)
        else:
            return self.ds.inv()

    def __call__(self, X, Y=None, diag = False,):
        assert len(X.shape) == 2
      

        if diag:
            if Y is None:
                dists = np.zeros(X.shape[0])
            else:
                assert(X.shape == Y.shape)
                dists = self.gs(np.sum(np.abs(self.ds(X) - self.ds(Y))**self.power, 1))
        else:
            sY = None
            if Y is not None:
                sY = self.ds(Y)
            dists = self.gs(eucldist(self.ds(X), sY, power = self.power))
        return dists